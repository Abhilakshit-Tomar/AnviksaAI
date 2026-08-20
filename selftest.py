"""
Verify the engine's maths before DDXPlus finishes downloading.

    python selftest.py

Builds a small synthetic world with the same shape as the real one and
asserts the five properties the product actually depends on. If these pass,
the reasoning is sound and only the data is missing.

    1. unknown != absent
    2. expected harm re-ranks against the differential
    3. exclusion gating lets the panel finish
    4. information gain picks the discriminating question
    5. it abstains when it has nothing to go on
"""
import numpy as np
from engine import Engine

# ---------------------------------------------------------------- fixture
PATHS = ["Panic attack", "Pulmonary embolism", "Unstable angina",
         "GERD", "Spontaneous pneumothorax"]
FEATS = ["chest_tightness", "dyspnoea", "calf_swelling", "recent_immobility",
         "oral_contraceptive", "exertional", "worse_lying_flat", "sudden_onset"]
QS = ["Chest tightness?", "Breathless?", "Swelling or pain in one calf?",
      "Long journey or immobility recently?", "On the contraceptive pill?",
      "Brought on by exertion?", "Worse lying flat?", "Did it start suddenly?"]

# P(feature=1 | pathology)
COND = np.array([
    # tight  dysp  calf  immob   ocp  exert  flat  sudden
    [0.85, 0.80, 0.02, 0.05, 0.30, 0.10, 0.05, 0.70],  # panic attack
    [0.60, 0.90, 0.45, 0.55, 0.40, 0.35, 0.10, 0.80],  # PE
    [0.90, 0.55, 0.02, 0.05, 0.25, 0.75, 0.10, 0.45],  # unstable angina
    [0.35, 0.10, 0.01, 0.05, 0.25, 0.05, 0.70, 0.15],  # GERD
    [0.70, 0.75, 0.02, 0.05, 0.25, 0.15, 0.05, 0.90],  # pneumothorax
])
PRIOR = np.array([0.44, 0.10, 0.07, 0.34, 0.05])
SEV = {"Panic attack": 1, "Pulmonary embolism": 100, "Unstable angina": 100,
       "GERD": 1, "Spontaneous pneumothorax": 30}
EXCL = {"Pulmonary embolism": ["d_dimer", "ctpa"],
        "Unstable angina": ["ecg", "troponin"],
        "Spontaneous pneumothorax": ["cxr"]}

# what the doctor hears in the room
PRESENTING = {"chest_tightness": True, "dyspnoea": True, "sudden_onset": True}
# ...plus what the pill strip and the transcript add
WITH_RECORDS = {**PRESENTING, "oral_contraceptive": True,
                "recent_immobility": True}


def build():
    e = Engine(cond=COND, prior=PRIOR, pathologies=PATHS, features=FEATS,
               questions=QS, severity=SEV, excluded_by=EXCL)
    e._fidx = {f: i for i, f in enumerate(FEATS)}
    # tau=1.0 here on purpose. Damping exists to fix naive-Bayes
    # overconfidence on 972 real features counted over a million rows; this
    # fixture is 5 pathologies and 8 hand-written conditionals, so there is
    # nothing to damp. These checks test the LOGIC, not the calibration —
    # tune tau against real counts with tune.py.
    e.tau = 1.0
    return e


def main():
    eng = build()
    ok = lambda label: print(f"  PASS  {label}")
    p_of = lambda post, n: float(post[eng.pathologies.index(n)])

    # -- 1. unknown != absent ------------------------------------------------
    print("\n1. unknown is not absent")
    post = eng.posterior(PRESENTING)
    post_absent = eng.posterior({**PRESENTING, "calf_swelling": False,
                                 "recent_immobility": False,
                                 "oral_contraceptive": False})
    print(f"     PE, risk factors UNKNOWN : {p_of(post, 'Pulmonary embolism'):.1%}")
    print(f"     PE, risk factors ABSENT  : "
          f"{p_of(post_absent, 'Pulmonary embolism'):.1%}")
    assert p_of(post_absent, "Pulmonary embolism") < p_of(post, "Pulmonary embolism")
    assert p_of(post, "Pulmonary embolism") > 0.05, \
        "PE must survive while nobody has asked"
    ok("unknowns are omitted, not silently scored as negatives")

    # -- 2. expected harm re-ranks ------------------------------------------
    print("\n2. expected harm inverts the differential")
    top_dx = eng.pathologies[int(np.argmax(post))]
    risks = eng.ranked_risk(post)
    print(f"     differential leads with : {top_dx} ({p_of(post, top_dx):.1%})")
    print(f"     can't-miss  leads with : {risks[0]['name']} "
          f"(p={risks[0]['p']:.1%} x {risks[0]['severity']} = {risks[0]['risk']:.1f})")
    assert risks[0]["name"] != top_dx, "risk ranking must differ from probability"
    assert risks[0]["p"] < p_of(post, top_dx), \
        "the top risk should be LESS probable than the top diagnosis"
    ok("a low-probability lethal condition outranks a likely benign one")

    # -- 3. exclusion gating -------------------------------------------------
    print("\n3. exclusion gating lets the panel finish")
    before = [r["name"] for r in eng.ranked_risk(post) if r["level"] != "low"]
    after = [r["name"] for r in eng.ranked_risk(post, ["d_dimer", "ecg"])
             if r["level"] != "low"]
    print(f"     before any workup : {before}")
    print(f"     after d-dimer+ECG : {after}")
    assert "Pulmonary embolism" in before and "Pulmonary embolism" not in after
    assert len(after) < len(before), "the panel must be able to empty"
    ok("ruled-out conditions leave — no permanent wall of dread")

    # -- 4. information gain -------------------------------------------------
    print("\n4. information gain picks the discriminator")
    lead_risk = eng.ranked_risk(post)[0]["name"]
    print(f"     from symptoms alone (branches shown for {lead_risk}):")
    qs = eng.best_question(post, PRESENTING, top_k=3)
    for q in qs:
        print(f"       {q['ig']:.3f} bit  {q['en']:<38} "
              f"yes -> {q['branches']['yes']:.0%} / no -> {q['branches']['no']:.0%}")
    assert qs[0]["id"] in {"calf_swelling", "recent_immobility"}, \
        f"expected a PE discriminator to win, got {qs[0]['id']}"

    print("     after the pill strip and the travel history:")
    post2 = eng.posterior(WITH_RECORDS)
    qs2 = eng.best_question(post2, WITH_RECORDS, top_k=3)
    for q in qs2:
        print(f"       {q['ig']:.3f} bit  {q['en']:<38} "
              f"yes -> {q['branches']['yes']:.0%} / no -> {q['branches']['no']:.0%}")
    assert qs2[0]["id"] == "calf_swelling", \
        f"expected calf swelling to win, got {qs2[0]['id']}"

    for q in qs + qs2:
        swing = abs(q["branches"]["yes"] - q["branches"]["no"])
        assert 0.0 <= q["branches"]["yes"] <= 1.0, "branches must be probabilities"
    top_swing = abs(qs2[0]["branches"]["yes"] - qs2[0]["branches"]["no"])
    assert top_swing > 0.15, \
        f"the winning question must move the top risk meaningfully ({top_swing:.2f})"
    for q in qs2:
        assert q["id"] not in WITH_RECORDS, "must never re-ask a known feature"
    ok("the highest-value question is the one that discriminates")

    # -- 5. end to end -------------------------------------------------------
    print("\n5. full assess() payload")
    out = eng.assess(WITH_RECORDS)
    assert not out["abstaining"] and out["cant_miss"] and out["best_question"]
    dx = ", ".join(f"{d['name']} {d['p']:.0%}" for d in out["differential"][:3])
    cm = ", ".join(f"{c['name']} {c['risk']:.1f}" for c in out["cant_miss"])
    print(f"     differential : {dx}")
    print(f"     can't-miss   : {cm}")
    print(f"     ask next     : {out['best_question']['en']}")
    ok("assess() returns a contract-shaped payload")

    # -- 6. abstention -------------------------------------------------------
    print("\n6. abstention")
    blank = eng.assess({})
    print(f"     no evidence  : abstaining={blank['abstaining']}, "
          f"entropy={blank['entropy']}")
    print(f"     w/ evidence  : abstaining={out['abstaining']}, "
          f"entropy={out['entropy']}")
    assert blank["abstaining"], "tune abstain_entropy — should abstain on nothing"
    assert not blank["differential"], "must not show a ranking while abstaining"
    ok("declines to rank when it has nothing to go on")

    # -- 7. severity floor ---------------------------------------------------
    print("\n7. the severity floor")
    certain = eng.posterior({"chest_tightness": True, "dyspnoea": True,
                             "worse_lying_flat": False, "exertional": False,
                             "calf_swelling": False, "recent_immobility": False})
    names = [r["name"] for r in eng.ranked_risk(certain)]
    print(f"     panic attack p={p_of(certain, 'Panic attack'):.1%}, "
          f"panel = {names[:3]}")
    assert "Panic attack" not in names and "GERD" not in names, \
        "severity-1 conditions must never reach the panel, however probable"
    ok("a near-certain benign condition still cannot enter Can't-Miss")

    print("\nAll checks passed. The reasoning is sound — plug in real counts.\n")


if __name__ == "__main__":
    main()
