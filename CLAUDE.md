# AnviksaAI

A diagnostic co-pilot for a 2-minute Indian OPD consultation. It does **not**
output a diagnosis. It outputs the conditions that are *not yet excluded and
would be catastrophic if missed*, plus the single question that best
discriminates between them, spoken aloud in the patient's language.

Hackathon build, ~30 hours, 4 people. Optimise for a working demo over
completeness. Prefer deleting scope to half-finishing it.

## Architecture

```
Sarvam captures and speaks  ->  our code decides  ->  Sarvam speaks
```

| Stage | Owner | What |
|---|---|---|
| capture | `api/capture/` | `saaras-v3` batch STT (diarized), `sarvam-vision` on prescriptions |
| normalize | `api/capture/evidence.py` | text -> DDXPlus evidence codes, three-valued |
| infer | `engine.py` | naive bayes over DDXPlus counts |
| rank | `engine.py` | `risk = P x severity`, minus anything already excluded |
| ask | `engine.py` | information gain over the risk-weighted posterior |
| speak | `api/capture/tts.py` | `bulbul-v3` in the patient's language |

## Non-negotiable rules

**No language model ever assigns a clinical number.** The LLM reads
(transcript -> evidence codes) and writes (structured state -> English note).
Every probability comes from counted co-occurrences. If you find yourself
parsing a percentage out of a completion, stop — that breaks the entire pitch.

**Evidence is three-valued: present / absent / UNKNOWN.** Unknowns are
*omitted* from the likelihood, never scored as absent. Scoring unknowns as
absent multiplies `P(¬e | pathology)` for every symptom nobody asked about,
which systematically penalises exactly the conditions that would have caused
those symptoms — the Can't-Miss list would then suppress the diagnoses it
exists to surface, while the eval still looks fine.

**Severity is log-scaled (1/3/10/30/100), never linear.** On a linear scale a
71% panic attack outranks a 15% PE and the panel quietly starts agreeing with
the doctor.

**Never tune `severity.yaml` to improve a metric.** It's a clinical prior. Tune
it and expected-harm ranking collapses back into probability ranking.

**Exclusion gating is required, not optional.** A condition leaves the panel
once the clinician has done the thing that rules it out (`excluded_by` in
`severity.yaml`). Without it, PE and ACS appear on every chest complaint
forever and the doctor stops reading the panel within a week.

**Severity weighting picks the question; it must never reach the screen.**
`best_question` ranks by information gain over the *risk-weighted*
distribution. The branch previews shown to the doctor come from the *true*
posterior. Mixing them makes every branch read ~100%.

**The frontend imports nothing from the backend.** It renders one JSON shape
from one endpoint, or reads `contract.json` from disk. This is the rule that
prevents the hour-25 integration disaster.

**Cache every Sarvam call on generation.** Vision is capped at 10–30 req/min on
every tier — that's the ceiling you'll hit, not credits. Never call it in a
render loop. Handle `429` with exponential backoff.

**Tune on validate, report on test.** Knobs: Laplace `alpha`, `_level()`
thresholds, `abstain_entropy`.

## UI rules

- **No chat interface.** Ever. It's the tell that this is a GPT wrapper, and
  nobody prompts anything during a 2-minute consult.
- **The reveal gate (restored 2026-08-21, in a new shape).** Differential
  and Can't-Miss stay hidden until BOTH: the clinician has committed a
  note, AND at least one REAL live input has happened — a live recording
  sent, or a document uploaded and scanned (either one satisfies it, not
  both required). The scripted STAGES walkthrough alone, no matter how far
  it's progressed, never reveals anything by itself — only genuine
  captured evidence plus a real commit does. This briefly went through a
  no-gate phase the same day; both changes were deliberate, explicit
  product decisions made in conversation, not oversights.
- Print the arithmetic on screen (`0.096 × 100`) under each risk score.
- Abstain visibly when the posterior is flat rather than guessing.

## Layout

```
engine.py          posterior, expected harm, exclusion gating, info gain, misfits
severity.yaml      harm table + exclusion rules   <- clinical judgment, not data
build_counts.py    DDXPlus -> counts.npz. run once, COMMIT THE OUTPUT
selftest.py        six assertions, no data needed
web/index.html     single file, no build step
api/               FastAPI: POST /assess -> the contract payload
demo/              hero case, recorded Marathi, pill strip photo
```

## Demo case (everything below exists in DDXPlus)

34F, Marathi. Chest tightness + dyspnoea, sudden onset. Doctor asks about
stress, writes "?anxiety / panic attack". A photographed pill strip reveals a
combined oral contraceptive; the transcript mentions a 12-hour bus journey.
Panic attack leads the differential; **pulmonary embolism leads Can't-Miss**.
The computed question is about unilateral calf swelling.

Gastritis, pancreatitis, peptic ulcer disease and biliary colic are **NOT** in
DDXPlus — do not use them.

## Don't build

Auth, accounts, a real database, medical imaging, any model trained from
scratch, a mobile app, more than one demo language, or any accuracy claim that
isn't measured.
