"""propose.py — findings in, candidate conditions out.

This module receives a set of findings and NOTHING ELSE. It never sees the
transcript, never sees the documents, never sees the patient's words. That is
not an oversight to be tidied up later; it is the reason the extraction step
can be trusted. A model holding both the raw transcript and a list of
conditions starts extracting findings *because* they fit a condition it is
already entertaining, and this project has already been burned by exactly
that — an earlier provider fabricated an immobility finding on a transcript
that never mentioned immobility.

WHAT IT RETURNS, AND WHAT IT REFUSES TO RETURN.

It returns conditions worth not forgetting, each with the findings that
support it, the findings that argue against it, and a readable reason. It
does not return a probability, a score, a confidence, or a rank. Not because
the model cannot produce one — it will happily produce a number for anything
— but because that number would be unfalsifiable. There is no dataset behind
it, no calibration, and nothing downstream that could ever check it. Three
significant figures of fabricated precision, presented to a clinician under
time pressure, is worse than no number at all.

THE CANDIDATE LIST IS UNBOUNDED. The model is given no list of conditions to
choose from. The previous build scored a fixed set of 49 synthetic
pathologies that contained no aortic dissection, no subarachnoid haemorrhage,
no meningitis, no sepsis, no ectopic pregnancy, no torsion and no cauda
equina — while including Ebola and Chagas. A tool whose job is "what
catastrophic thing have you not excluded" cannot have a closed list of things
it is able to say.

severity.yaml is NOT that list. It rates conditions for how bad missing them
would be, and a condition proposed here that has no entry there still reaches
the panel, marked unrated. It is never silently dropped.

WHAT THIS MODULE HONESTLY IS. The candidates come from a general-purpose
language model. If a chat assistant would name aortic dissection on this
presentation, so will this. With no eval, this cannot claim to be more
accurate than a doctor typing the case into a chat window, and nobody should
say otherwise. What it can claim: it happens without anyone being asked to
type a prompt, it keeps state across a consult, and the dangerousness ranking
applied to its output is a reviewed file rather than a generation. The
reasoning below is the safety net, which is why every candidate must carry it.
"""

import json

import llm
import vocabulary

SYSTEM_PROMPT = """You are helping a clinician in a two-minute outpatient \
consultation remember what is worth not forgetting.

You are given a set of findings about one patient. You are NOT given the \
transcript, and you should not ask for it. You are not the person who \
decides what happens to this patient.

WHAT YOU RETURN: conditions consistent with these findings, each with the \
findings that support it, the findings that argue against it, and a short \
reason a clinician can read and check in three seconds.

WHAT YOU MUST NOT RETURN:
- Any probability, percentage, score, confidence or likelihood. Not in the \
reasoning text either. There is no data behind such a number and it would be \
fabricated precision.
- A ranking. Order does not matter; another system orders these.
- A single answer. You are not diagnosing. Naming the diagnosis is exactly \
what this tool refuses to do.

WHAT TO INCLUDE
- The obvious, common, benign explanation, when one fits. A panel that only \
ever shows catastrophes gets ignored within a week, and the clinician's own \
working diagnosis appearing beside the dangerous one is what makes the list \
readable rather than contrarian.
- Any dangerous condition that fits, even partially, even if it is unlikely. \
This is the whole point. Unlikely-but-lethal belongs on the list; another \
system decides how it is displayed.
- Conditions that fit the findings TOGETHER rather than one at a time. Pain \
worse lying flat, better leaning forward, after a recent viral illness is a \
picture; each of those findings alone is nearly meaningless. Reason about \
combinations, not a checklist.

HOW MANY: between three and ten. Fewer if the findings genuinely support \
few. Do not pad the list to reach a number.

IF THE FINDINGS ARE TOO THIN, SAY SO BY RETURNING AN EMPTY LIST. A handful \
of nonspecific findings does not justify naming conditions. Returning \
nothing is a correct answer and is much better than a plausible-looking list \
generated from almost no information — the clinician cannot tell the \
difference by looking, and that is what makes it dangerous.

FOR EACH CANDIDATE
- supported_by: ids of findings this patient HAS that support it. Only ids \
listed as present below.
- opposed_by: ids of findings this patient was asked about and DENIED that \
argue against it. Only ids listed as absent below.
- unresolved: ids of findings NOT yet known for this patient whose answer \
would most change whether this condition belongs on the list. Only ids from \
the catalog that appear in neither the present nor the absent list. This is \
the most useful field you produce: it is what lets the clinician be asked \
one good question instead of twenty.
- reasoning: one or two sentences naming the actual findings. "Sudden \
breathlessness with chest tightness in a patient on a combined oral \
contraceptive after two days of bus travel" — not "clinical picture is \
suggestive". A clinician who cannot check your reasoning cannot catch your \
mistake, and there is nothing else here that would catch it."""

_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_candidates",
        "description": (
            "Record the conditions worth keeping in mind for this patient. "
            "Return an empty list if the findings are too thin to support "
            "naming anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "condition": {
                                "type": "string",
                                "description": (
                                    "The condition, named as a clinician would "
                                    "write it. No abbreviations a colleague "
                                    "would have to guess at."
                                ),
                            },
                            "reasoning": {
                                "type": "string",
                                "description": (
                                    "One or two sentences naming the actual "
                                    "findings that put this on the list. No "
                                    "numbers, no percentages, no hedging "
                                    "phrases that say nothing."
                                ),
                            },
                            "supported_by": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Finding ids from the PRESENT list only.",
                            },
                            "opposed_by": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Finding ids from the ABSENT list only.",
                            },
                            "unresolved": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Finding ids that are currently UNKNOWN and "
                                    "whose answer would most change whether "
                                    "this condition belongs on the list."
                                ),
                            },
                        },
                        "required": ["condition", "reasoning", "supported_by", "unresolved"],
                    },
                }
            },
            "required": ["candidates"],
        },
    },
}


def propose(findings):
    """findings: {finding_id: True|False} from extract.extract().

    -> [{"condition", "reasoning", "supported_by", "opposed_by", "unresolved"}]

    No probability, no score, no rank. engine.py applies severity and support
    ordering to this; nothing here is ordered meaningfully.
    """
    present = sorted(f for f, state in findings.items() if state is True)
    absent = sorted(f for f, state in findings.items() if state is False)

    # Nothing present means nothing to reason from. Denials alone cannot
    # support a candidate, and asking the model to produce a list anyway is
    # how a panel ends up showing generic prior-driven suggestions styled as
    # though they came from this patient. Abstain instead, visibly.
    if not present:
        return []

    cache_key = llm.content_hash(llm.MODEL, json.dumps(present), json.dumps(absent))
    cpath = llm.CACHE_DIR / f"propose_{cache_key}.json"

    def produce():
        def render(ids):
            return "\n".join(f"  {f}  ({vocabulary.label(f)})" for f in ids) or "  (none)"

        unknown = [f for f in vocabulary.FINDINGS if f not in findings]
        user_content = (
            f"PRESENT — this patient has these:\n{render(present)}\n\n"
            f"ABSENT — asked about and denied:\n{render(absent)}\n\n"
            f"UNKNOWN — not discussed, valid ids for `unresolved` "
            f"({len(unknown)} of them):\n{render(unknown)}\n\n"
            "Call propose_candidates."
        )
        return llm.call_tool(SYSTEM_PROMPT, user_content, _TOOL)

    parsed = llm.cached(cpath, produce)

    present_set, absent_set = set(present), set(absent)
    candidates = []
    for raw in parsed.get("candidates") or []:
        condition = (raw.get("condition") or "").strip()
        reasoning = (raw.get("reasoning") or "").strip()
        # Every candidate carries its reasoning. One that cannot justify
        # itself cannot be checked by the clinician, and the clinician is the
        # entire safety net — so it does not reach the panel at all.
        if not condition or not reasoning:
            continue

        def keep(key, allowed):
            """Filter to real ids in the right state. A model claiming a
            finding supports a condition when the patient does not have that
            finding is asserting evidence that does not exist — the one thing
            a clinician skim-reading a panel would not catch."""
            return [f for f in (raw.get(key) or []) if f in allowed]

        candidates.append({
            "condition": condition,
            "reasoning": reasoning,
            "supported_by": keep("supported_by", present_set),
            "opposed_by": keep("opposed_by", absent_set),
            # Unresolved must be genuinely unknown: not present, not absent,
            # and a real finding. Anything else would produce a question the
            # patient has already answered, which is the bug this project
            # spent a whole build cycle on.
            "unresolved": [
                f for f in (raw.get("unresolved") or [])
                if f in vocabulary.FINDINGS and f not in findings
            ],
        })

    return candidates


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print('usage: python propose.py \'{"chest_tightness": true, "fever": false}\'')
        sys.exit(1)
    _f = json.loads(sys.argv[1])
    for _c in propose(_f):
        print(f"\n{_c['condition']}")
        print(f"  {_c['reasoning']}")
        print(f"  supported by: {', '.join(vocabulary.label(x) for x in _c['supported_by']) or '-'}")
        print(f"  opposed by:   {', '.join(vocabulary.label(x) for x in _c['opposed_by']) or '-'}")
        print(f"  unresolved:   {', '.join(vocabulary.label(x) for x in _c['unresolved']) or '-'}")
