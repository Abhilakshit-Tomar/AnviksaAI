"""
The hero case, in real DDXPlus evidence codes, at every stage of the demo.

    python demo_case.py            # print all three stages
    python demo_case.py --dump     # write contract.json for the frontend

Frontend lane: run `--dump` and build against the file. It is REAL engine
output over real counts, not a hand-written mock, so if it renders correctly
the integration is already most of the way done.

34F, Marathi. Chest tightness and breathlessness since last night, sudden
onset. The doctor asks whether she is stressed and writes "?anxiety / panic
attack". A photographed blister strip turns out to be a combined oral
contraceptive; the transcript mentions a flight home four days ago.

Every code below is real. Notably:
    E_100  "Do you currently take hormones?"       <- the pill strip
    E_204  "Traveled out of the country recently?" <- the flight
    E_151  "Swelling in one or more areas?"        <- the question it asks
    E_152  "Where is the swelling located?" V_120 = calf(L)

NOTE: E_204 is *international* travel, so the transcript says a flight home,
not the 12-hour bus journey in the earlier draft. A domestic bus ride maps to
no DDXPlus code. Given how many patients in an Indian OPD are returning Gulf
workers, the flight is arguably the more authentic detail anyway.
"""
import argparse, json

from engine import Engine

# --- stage 1: what the doctor hears in two minutes -------------------------
PRESENTING = {
    "E_53": True,          # pain, related to the reason for consulting
    "E_55=V_101": True,    # ...located in the upper chest
    "E_59=9": True,        # appeared fast
    "E_66": True,          # shortness of breath
    "E_155": True,         # heart racing / palpitations
    "E_16": True,          # feels anxious          <- the anchor
}

# --- stage 2: what the blister strip and the transcript add ----------------
WITH_RECORDS = {**PRESENTING,
                "E_100": True,        # currently taking hormones
                "E_204=V_6": True}    # travelled (Asia) in the last 4 weeks

# --- stage 3: the answer to the computed question --------------------------
ANSWERED = {**WITH_RECORDS,
            "E_151": True,            # swelling somewhere
            "E_152=V_120": True}      # ...in the left calf

# --- stage 4: the clinician acts. the panel must be able to empty. ---------
WORKED_UP = ["ecg", "troponin", "d_dimer"]

STAGES = [("1 presenting", PRESENTING, ()),
          ("2 + pill strip + travel", WITH_RECORDS, ()),
          ("3 + unilateral calf swelling", ANSWERED, ()),
          ("4 + ECG, troponin, d-dimer done", ANSWERED, WORKED_UP)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="write contract.json")
    ap.add_argument("--stage", type=int, default=2,
                    help="which stage to dump (default 2, the reveal)")
    ap.add_argument("--dump-all", action="store_true",
                    help="write contract.json as {stages:[...]} covering "
                         "stages 2-4 (reveal, answered, worked-up) so the "
                         "frontend's step-through has real data for all "
                         "three panel states, not just one snapshot")
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()

    eng = Engine.load()
    if args.tau is not None:
        eng.tau = args.tau
    print(f"tau={eng.tau}  panel_min_severity={eng.panel_min_severity}\n")

    for label, evidence, actions in STAGES:
        out = eng.assess(evidence, actions_taken=actions)
        print("=" * 74)
        print(f"{label}   (entropy {out['entropy']}, "
              f"abstaining {out['abstaining']})")
        print("  differential : " + ", ".join(
            f"{d['name']} {d['p']:.1%}" for d in out["differential"][:4]))
        print("  can't-miss   : " + (", ".join(
            f"{c['name']} {c['p']:.3f}x{c['severity']}={c['risk']:.1f} "
            f"[{c['level']}]" for c in out["cant_miss"]) or "— empty —"))
        q = out["best_question"]
        if q:
            hint = f"  -> {q['answer_hint']}" if q.get("answer_hint") else ""
            print(f"  ask next     : [{q['ig']:.3f} bit] {q['en']}{hint}")
            print(f"                 yes -> {q['about']} {q['branches']['yes']:.0%}"
                  f"  /  no -> {q['branches']['no']:.0%}")
        for m in out["misfits"][:2]:
            print(f"  doesn't fit  : {m['text']}")

    if args.dump_all:
        # stages 2-4 (0-indexed 1-3): the reveal, the answered question, and
        # the worked-up/excluded state. Stage 1 ("presenting") is never shown
        # in the UI — Can't-Miss stays locked until the commit, and by the
        # time it unlocks the pill strip and travel history are already in.
        stages_out = []
        for label, evidence, actions in STAGES[1:4]:
            payload = eng.assess(evidence, actions_taken=actions)
            payload["_stage"] = label
            payload["_evidence"] = evidence
            stages_out.append(payload)
        doc = {
            "_note": ("Real engine output over real DDXPlus counts, one "
                      "entry per demo stage. Frontend builds against this "
                      "shape — see web/index.html's STAGE_LABELS."),
            "stages": stages_out,
        }
        with open("contract.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"\nwrote contract.json with {len(stages_out)} stages "
              f"({', '.join(s['_stage'] for s in stages_out)})")
    elif args.dump:
        label, evidence, actions = STAGES[args.stage - 1]
        payload = eng.assess(evidence, actions_taken=actions)
        payload["_stage"] = label
        payload["_evidence"] = evidence
        payload["_note"] = ("Real engine output over real DDXPlus counts. "
                            "Frontend builds against this shape.")
        with open("contract.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote contract.json from stage {args.stage} ({label})")


if __name__ == "__main__":
    main()
