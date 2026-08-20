"""
AnviksaAI — the reasoning engine.

Everything here is counted statistics and arithmetic. No model call, no
network, no randomness. It runs offline in single-digit milliseconds, which
is why it is the part you build first and the part you can still demo when
the venue wifi dies.

    from engine import Engine
    eng = Engine.load("counts.npz", "severity.yaml")
    out = eng.assess(evidence={"E_55": True, "E_91": False}, actions_taken=["ecg"])

`out` is shaped like contract.json. Hand it straight to the frontend.

THE ONE RULE: evidence is three-valued.
    True    observed present
    False   observed ABSENT (asked, answered no)
    missing UNKNOWN — never asked
Unknowns are OMITTED from the likelihood, not scored as absent. Treat them as
absent and you multiply P(not-e | pathology) for every symptom nobody asked
about, which systematically penalises exactly the conditions that would have
caused those symptoms. Your Can't-Miss list would then suppress the diagnoses
it exists to surface, while your eval numbers still look fine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

EPS = 1e-12


def _entropy(p, axis=0):
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


@dataclass
class Engine:
    cond: np.ndarray          # (P, F)  P(feature=1 | pathology)
    prior: np.ndarray         # (P,)
    pathologies: list
    features: list
    questions: list           # human-readable text per feature
    severity: dict            # pathology -> harm_if_missed
    excluded_by: dict         # pathology -> [actions that take it off the panel]
    default_severity: int = 3

    # DDXPlus flattens 223 evidence codes into 972 binary features, because
    # four location variables carry 165 values each. `parents` maps each
    # feature back to its evidence code so the question engine asks "Does the
    # pain radiate to another location?" instead of "Pain in the tonsil(L)?".
    parents: list = field(default_factory=list)
    parent_questions: list = field(default_factory=list)

    # Likelihood temperature. THE most important knob in the engine.
    #
    # Naive Bayes over 1M rows produces very sharp conditionals, and
    # multiplying ~10 of them drives the posterior to 100.0% / 0.0%. That
    # confidence is unearned: the independence assumption is false (the
    # flattened location values are mutually exclusive by construction, and
    # real symptoms correlate), so the model counts the same evidence
    # several times over.
    #
    # For THIS product that collapse is fatal, not cosmetic. Can't-Miss
    # exists to surface conditions sitting at 3-8%. If every posterior is
    # 100/0 there is nothing left to rank and the panel goes empty.
    #
    # tau < 1 damps each factor: logP += tau * log P(e|path). Tune it on
    # VALIDATE against calibration, never on test. ~0.25 is a sane start.
    tau: float = 0.15

    # Nothing below this severity may ever appear on the Can't-Miss panel.
    panel_min_severity: int = 10

    _fidx: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, counts_path="counts.npz", severity_path="severity.yaml"):
        z = np.load(counts_path, allow_pickle=True)
        if yaml is None:
            raise ImportError("pip install pyyaml")
        with open(severity_path, encoding="utf-8") as f:
            sev = yaml.safe_load(f)
        eng = cls(
            cond=z["cond"], prior=z["prior"],
            pathologies=list(z["pathologies"]),
            features=list(z["features"]),
            questions=list(z["questions"]),
            severity=sev.get("pathologies", {}),
            excluded_by=sev.get("excluded_by", {}),
            default_severity=sev.get("default", 3),
            parents=list(z["parents"]) if "parents" in z.files else [],
            parent_questions=(list(z["parent_questions"])
                              if "parent_questions" in z.files else []),
        )
        eng._fidx = {f: i for i, f in enumerate(eng.features)}
        missing = [p for p in eng.pathologies if p not in eng.severity]
        if missing:
            print(f"[engine] {len(missing)} pathologies have no severity, "
                  f"defaulting to {eng.default_severity}: {missing[:5]}")
        return eng

    def sev(self, pathology) -> int:
        return int(self.severity.get(pathology, self.default_severity))

    # ----------------------------------------------------------- posterior
    def posterior(self, evidence: dict) -> np.ndarray:
        """
        evidence: {feature_name: True|False}. Anything absent from the dict is
        UNKNOWN and contributes nothing. Log space — the products underflow
        to zero in float64 after ~40 features otherwise.
        """
        logp = np.log(np.clip(self.prior, EPS, None)).copy()
        for feat, val in evidence.items():
            j = self._fidx.get(feat)
            if j is None:
                continue                      # unknown code: ignore, don't guess
            col = np.clip(self.cond[:, j], EPS, 1 - EPS)
            logp += self.tau * np.log(col if val else 1.0 - col)
        logp -= logp.max()                    # stabilise before exp
        p = np.exp(logp)
        return p / p.sum()

    # ------------------------------------------------------- expected harm
    def ranked_risk(self, post, actions_taken=()):
        """
        risk = P(pathology | evidence) x harm_if_missed, EXCLUDING anything the
        clinician has already ruled out.

        The exclusion filter is not a nicety. Without it, severity=100 means
        PE and ACS clear any threshold on every chest complaint forever, the
        panel never changes, and the doctor stops reading it inside a week.
        Alert fatigue is the single most common way tools like this die in
        the field. The panel must answer "what have you not ruled out",
        which is a question that can be *finished*.
        """
        taken = {a.lower() for a in actions_taken}
        rows = []
        for i, path in enumerate(self.pathologies):
            excl = [a.lower() for a in self.excluded_by.get(path, [])]
            if excl and taken.intersection(excl):
                continue
            s = self.sev(path)
            # A severity floor, not just a risk threshold. The panel means
            # "catastrophic if missed", so a self-limiting condition is
            # categorically ineligible no matter how probable — otherwise a
            # 99%-certain panic attack (1.0 x 1) clears any risk threshold and
            # the Can't-Miss panel leads with "Panic attack", which is absurd.
            if s < self.panel_min_severity:
                continue
            rows.append({
                "key": path, "name": path,
                "p": float(post[i]), "severity": s,
                "risk": float(post[i] * s),
                "level": self._level(post[i] * s),
                "excluded_by": self.excluded_by.get(path, []),
            })
        rows.sort(key=lambda r: -r["risk"])
        return rows

    @staticmethod
    def _level(risk):
        if risk >= 5.0:
            return "critical"
        if risk >= 0.75:
            return "serious"
        return "low"

    # --------------------------------------------------- the next question
    def best_question(self, post, evidence, top_k=4, severity_weighted=True):
        """
        IG(q) = H(D) - SUM_a P(a) H(D | a), vectorised over every unasked
        feature at once.

        severity_weighted=True computes the entropy over the RISK-weighted
        distribution instead of the raw posterior. That asks "which question
        best resolves what could hurt this patient" rather than "which best
        resolves my uncertainty in general" — value of information, not raw
        information gain. It is the right objective for this product and it
        is the answer to give when a judge pushes on the maths.
        """
        asked = {self._fidx[f] for f in evidence if f in self._fidx}
        sv = np.array([self.sev(p) for p in self.pathologies], dtype=float)
        c = np.clip(self.cond, EPS, 1 - EPS)          # (P, F)

        # --- the OBJECTIVE: which question to rank first -------------------
        w = post
        if severity_weighted:
            w = post * sv
            w = w / w.sum()
        pw = np.clip(w @ c, EPS, 1 - EPS)
        ig = _entropy(w) - (
            pw * _entropy((w[:, None] * c) / pw[None, :], axis=0)
            + (1 - pw) * _entropy((w[:, None] * (1 - c)) / (1 - pw)[None, :], axis=0)
        )
        ig[list(asked)] = -np.inf                      # never re-ask

        # --- the DISPLAY: what the doctor is shown -------------------------
        # Always the true posterior, never the severity-weighted one. The
        # weighting decides which question is worth asking; it must never
        # leak into the numbers on screen, or every branch reads ~100%
        # because the lethal condition dominates the weights.
        pr = np.clip(post @ c, EPS, 1 - EPS)
        post_yes = (post[:, None] * c) / pr[None, :]
        post_no = (post[:, None] * (1 - c)) / (1 - pr)[None, :]
        top_i = int(np.argmax(post * sv))              # the leading can't-miss

        # Dedupe by PARENT evidence code. Without this the top five candidates
        # are five values of the same 165-way location variable and the panel
        # reads as five near-identical questions.
        out, seen_parents = [], set()
        for j in np.argsort(-ig):
            if len(out) >= top_k:
                break
            if not np.isfinite(ig[j]) or ig[j] <= 0:
                continue
            feat = self.features[j]
            parent = self.parents[j] if self.parents else feat
            if parent in seen_parents:
                continue
            seen_parents.add(parent)

            # Render the parent's real question; carry the winning value as
            # the answer hint. "Does the pain radiate to another location?"
            # + hint "left arm" — not "Pain radiates to left arm? yes/no".
            text = (self.parent_questions[j] if self.parent_questions
                    else self.questions[j])
            hint = None
            if "=" in feat:
                label = self.questions[j]
                hint = label[label.rfind("[") + 1:-1] if label.endswith("]") \
                    else feat.split("=", 1)[1]

            out.append({
                "id": feat,
                "parent": parent,
                "ig": round(float(ig[j]), 3),
                "en": text,
                "answer_hint": hint,
                "about": self.pathologies[top_i],
                "branches": {
                    "yes": round(float(post_yes[top_i, j]), 4),
                    "no": round(float(post_no[top_i, j]), 4),
                },
            })
        return out

    # ------------------------------------------------------------ misfits
    def misfits(self, post, evidence, limit=3):
        """
        Findings the leading diagnosis fails to explain.

        Scored as a likelihood ratio: how much better some other pathology
        explains this present finding than the leading one does. Computed,
        not an LLM's opinion — it falls out of the same conditional table.
        """
        lead = int(np.argmax(post))
        lead_name = self.pathologies[lead]
        rows = []
        for feat, val in evidence.items():
            if val is not True:
                continue
            j = self._fidx.get(feat)
            if j is None:
                continue
            p_lead = max(float(self.cond[lead, j]), EPS)
            best_other = float(np.max(np.delete(self.cond[:, j], lead)))
            lr = best_other / p_lead
            if lr > 3.0:                       # explained >3x better elsewhere
                alt = int(np.argmax(np.where(
                    np.arange(len(self.pathologies)) == lead, -1, self.cond[:, j])))
                rows.append({
                    "feature": feat,
                    "text": f"{self.questions[j]} — {lead_name} does not "
                            f"explain this ({p_lead:.1%} of cases); "
                            f"{self.pathologies[alt]} does ({best_other:.0%}).",
                    "against": lead_name,
                    "ratio": round(lr, 1),
                })
        rows.sort(key=lambda r: -r["ratio"])
        return rows[:limit]

    # ------------------------------------------------------------- assess
    def assess(self, evidence: dict, actions_taken=(), top_dx=7,
               abstain_entropy=0.72):
        """
        One call -> the whole contract payload.

        Abstention: if the posterior is still close to uniform we refuse to
        rank and say what to ask instead. A system that declines to guess
        reads as far more serious than one that always has an answer, and it
        costs about twenty minutes to implement.
        """
        post = self.posterior(evidence)
        h_norm = float(_entropy(post) / np.log2(len(post)))
        risks = self.ranked_risk(post, actions_taken)
        qs = self.best_question(post, evidence)

        order = np.argsort(-post)[:top_dx]
        differential = [{
            "key": self.pathologies[i], "name": self.pathologies[i],
            "p": round(float(post[i]), 4),
        } for i in order]

        abstaining = h_norm > abstain_entropy
        return {
            "abstaining": abstaining,
            "entropy": round(h_norm, 3),
            "differential": [] if abstaining else differential,
            "cant_miss": [] if abstaining else
                         [r for r in risks if r["level"] != "low"][:4],
            "best_question": qs[0] if qs else None,
            "runners_up": qs[1:],
            "misfits": [] if abstaining else self.misfits(post, evidence),
            "message": ("Insufficient information to rank — ask the questions "
                        "below.") if abstaining else None,
        }


if __name__ == "__main__":
    import sys
    eng = Engine.load(*(sys.argv[1:3] or ["counts.npz", "severity.yaml"]))
    print(json.dumps(eng.assess({}), indent=2)[:1200])
