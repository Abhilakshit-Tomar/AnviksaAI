# AnviksaAI

A recall aid for a 2-minute Indian OPD consultation. It does **not** diagnose,
and it does **not** tell you how likely anything is. It reads what the patient
actually said and brought, proposes what is worth not forgetting, flags which
of those are catastrophic if missed, and asks the one question most likely to
tell the conditions apart.

The problem it solves is not "the doctor doesn't know medicine." It is: in the
ninety seconds available, with a queue outside, a clinician cannot hold
everything the patient offered in working memory at once. Something gets
dropped. Occasionally the thing that gets dropped is the thing that kills them.

Medicine reaches certainty through tests and confirmation. We are upstream of
that: we only make sure nothing dangerous silently fell off the list first.

## Why this isn't a chat wrapper

The question every reviewer will ask: how is this different from a doctor
pasting the case into ChatGPT? The honest answer has two halves, and the
second one matters.

**Nobody types a prompt.** To ask an AI "here's what my patient has", you
must already know what to include — and the failure being addressed is
*forgetting*. If you remembered the calf swelling well enough to type it, you
did not need the tool. Input here is ambient: a recording of a conversation
that was happening anyway, and a photo of a document the patient brought.
Chat answers the question you thought to ask; this surfaces what you didn't.

**Three things a chat structurally cannot do.** Track exclusion state across a
consult, so the panel can be *finished* rather than re-listing PE on every
chest complaint forever. Avoid re-asking what was already answered. And
refuse to name the diagnosis — ask a model what the patient has and it tells
you; the refusal is the product.

**Severity is not the model's opinion.** The axis deciding what gets flagged
as lethal is a reviewed file, not a generation that drifts between runs.

**And the uncomfortable half.** The candidate list comes from the same kind of
model — if ChatGPT would name aortic dissection, so will this. The reasoning
shown is model-generated. With no eval, **this cannot claim to be more
accurate than a doctor using ChatGPT**, and nobody should say otherwise.

The honest claim is narrow: it happens without being asked, it keeps state a
chat cannot, and the dangerousness ranking is human-authored. Every one of
those is a rule below. Erode any of them and the honest answer to this
question becomes "we aren't different" — which is why they are written as
non-negotiable rather than preferences.

## Architecture

```
Sarvam captures  ->  LLM reads and proposes  ->  our rulebook triages  ->  clinician decides
```

| Stage | Owner | What |
|---|---|---|
| capture | `capture.py` | `saaras-v3` batch STT (diarized), `sarvam-vision` on documents |
| extract | `extract.py` | this patient's transcript/documents -> findings, three-valued |
| propose | `propose.py` | findings -> candidate conditions + what supports/opposes each |
| triage | `engine.py` | severity band, exclusion state, support ordering |
| ask | `engine.py` | the finding that most divides the current candidate list |
| speak | `capture.speak()` | `bulbul-v3` — **currently unwired**, see below |

There is no `api/` package; every module above sits at the repo root.
`capture.speak()` exists and works but nothing calls it — the "ask this
aloud" button went when the scripted walkthrough did, and `/analyze` has no
TTS step. Either wire it back to the next question or delete it; leaving a
documented stage that no code path reaches is how the last round of stale
docs happened.

Two LLM calls, never one. Extraction must not see the candidate list, or the
model starts extracting findings *because* they fit a condition it is already
considering. That is not hypothetical — an earlier provider fabricated
`E_110` (immobility) on a transcript that never mentioned it, and this
project switched providers over exactly that failure.

## Non-negotiable rules

**No number ever reaches the screen.** Not a percentage, not a score, not a
confidence. Nobody is ever 100% sure in medicine, and three significant
figures imply a precision no input here can support. Show a severity band and
a support strength, in words.

**No language model ever assigns severity or exclusion.** The LLM proposes
candidates and says what supports them. `severity.yaml` — hand-written,
clinician-reviewed — decides what is dangerous and what rules it out. Ask a
model "how bad is it to miss this" and the answer drifts between runs, and
there is no artifact anyone can audit.

**Exclusion proposes; it never auto-removes.** When the clinician records a
CTPA, the panel *asks* "mark PE as ruled out?" — it does not silently drop it.
Every other error in this system adds noise to a list a human is already
reading. An exclusion error *removes a warning*, and nobody ever sees what
they were not shown. That is the only failure mode here shaped like patient
harm, so a human makes every removal.

**Severity and support are two axes. Never multiply them.** `P x severity` was
meaningful when `P` was a real probability. Without one, a single blended
number is a fabrication that discards information the clinician wants
separately: how bad if missed, and how well does this actually fit.

**Findings need stable names.** Extraction maps to a fixed vocabulary we
author. Free-form findings cannot be matched across turns, so "don't re-ask
what was already answered" silently fails — which produced a real
stuck-on-the-same-question bug (a categorical code's whole-question denial
never landed in the asked set, so the identical question returned forever).

**Every candidate carries its reasoning.** "Aortic dissection — because
tearing pain radiating to the back, sudden onset." With no dataset and no
eval, the visible reasoning and the clinician *are* the entire safety net.
A suggestion whose justification cannot be read cannot be checked.

**Three-valued findings: present / absent / UNKNOWN.** Never guess absent for
something nobody asked about. An unasked symptom is not a denied symptom, and
conflating them penalises exactly the conditions that would have caused it.

**`temperature=0` and a fixed seed, always.** Confirmed live: identical input
returned different findings run to run until these were pinned, which changed
what was on screen for reasons that had nothing to do with the patient. Vary
the seed only on retry, never on the first attempt.

**Cache every model call on generation.** Sarvam vision is capped at 10-30
req/min on every tier. Handle `429` with exponential backoff. Never call in a
render loop.

**Evidence arrives in pieces, and re-extraction is over the whole thing.**
A consult is not one input — it is a recording, then a document, then another
recording, each landing minutes apart. Extraction re-runs across the FULL
accumulated transcript and document text every time, never incrementally over
just the new chunk. A later statement can correct an earlier one ("no, it's
the right calf, not the left"), and only a model seeing both can resolve
that; merging per-chunk findings would keep both as true. This costs a full
call per update. Pay it.

**Two things survive across updates; everything else recomputes.**
*Confirmed exclusions* — once the clinician ticks "PE ruled out by CTPA",
that holds for the rest of the consult and must be sent with every subsequent
call. *Answered findings* — which is only possible because the vocabulary
gives findings stable ids; this is precisely what failed before, when a
whole-question denial could not be matched and the same question returned
forever. Candidates, ordering, and the next question are recomputed fresh
each time.

**The panel must not churn.** A list that reshuffles on every sentence is
unreadable, and worse, the clinician loses track of what they already
considered. Show what changed rather than silently reordering.

## What was deleted, and why it must not come back

**DDXPlus, entirely.** 49 synthetic pathologies, 223 evidence codes. Deleted
because a tool whose job is "what catastrophic thing have you not excluded"
was running on a dataset containing no aortic dissection, no subarachnoid
haemorrhage, no meningitis, no sepsis, no ectopic pregnancy, no torsion, no
cauda equina — while including Ebola and Chagas. Its base rates were foreign
to an Indian OPD, and its statistics were generated by a simulator, not
observed in patients.

**Naive Bayes, the posterior, `tau`, `abstain_entropy`.** Naive Bayes scores
each condition by multiplying findings independently — nothing anywhere could
represent findings that only mean something *together*. Pericarditis
demonstrated this concretely: "pain worse lying flat, better leaning forward"
plus "recent viral infection" is a classic pair, but scored separately the
viral-infection term dragged toward the dozen more common respiratory
illnesses in the set, and pericarditis never surfaced. No rephrasing fixed
it, because the architecture had nowhere to put the joint meaning.

**Information gain and "best question."** Optimal only with respect to a
proxy objective over a synthetic distribution — the label overclaimed. The
replacement is honest: the finding whose support most *divides* the live
candidate set. Nobody can have the best question.

**Calibration and accuracy claims.** With no ground truth there is no eval,
no `tune.py`, no measured accuracy. This is the deliberate price of unbounded
coverage. Do not claim a number that cannot be measured.

## UI rules

- **No chat interface.** Ever. Nobody prompts anything during a 2-minute
  consult, and it is the tell that this is a wrapper.
- **The reveal gate.** Differential and Can't-Miss stay hidden until BOTH: the
  clinician has committed a note, AND at least one real live input has
  happened (a recording sent, or a document scanned — either satisfies it).
- **Abstain visibly.** When nothing was extracted, say so. Never show a
  generic prior-driven suggestion styled as though it came from this patient.
- **Severity-unrated is a visible tier.** A proposed condition absent from
  `severity.yaml` still appears, marked unrated. It is never silently dropped.

## Layout

```
capture.py       Sarvam STT / vision / TTS, all disk-cached by content hash
extract.py       transcript/documents -> findings (fixed vocabulary, three-valued)
propose.py       findings -> candidates + supporting/opposing findings + reasoning
engine.py        severity band, exclusion state, support ordering, next question
severity.yaml    harm-if-missed + exclusion rules   <- clinical judgment, hand-written
findings.yaml    the finding vocabulary             <- hand-written
selftest.py      invariants: never re-ask, exclusion, severity floor, determinism
main.py          FastAPI: POST /capture (STT/OCR), POST /analyze (propose+triage)
web/index.html   single file, no build step
```

**Keep `/capture` and `/analyze` separate.** They look mergeable and are not.
`/capture` does STT or OCR only; `/analyze` does extract + propose + triage
over everything accumulated so far. When the frontend used one combined
endpoint, every recording and every upload ran a full extraction whose result
was thrown away, then ran a second one — double the cost and double the
latency on every single action. Merging them back reintroduces that.

## The canonical case

There is no committed golden case yet — `demo_case.py` and `contract.json`
were written in DDXPlus codes and go away at step 8. Author a replacement
alongside `findings.yaml`, because without one nothing defines what correct
output looks like and step 6's regression check has nothing to compare to.

The case that has always exercised this product properly: 34F, sudden chest
tightness and dyspnoea, a photographed strip revealing a combined oral
contraceptive, and a long journey mentioned in passing. The doctor is
thinking panic attack. **Pulmonary embolism must appear on Can't-Miss**, and
the next question should be about unilateral calf swelling. Anything that
fails to surface PE here is broken, whatever else it gets right.

## Migration status (2026-08-22)

This document describes the target design. The code still implements the old
DDXPlus one.

**The app does not currently run.** The rewritten `severity.yaml` names
conditions the old DDXPlus engine has never heard of, and sets `default: null`
so an unlisted condition reads as *unrated* rather than silently benign —
which `engine.py`'s `int(default_severity)` cannot handle. This breakage is
intentional and expected; it clears when step 5 lands. The superseded file is
kept as `severity.yaml.ddxplus-legacy` (same convention as
`extract.py.groq-batched-wip`) if the old pipeline needs to be run meanwhile.

Each step is ordered by dependency, and each has an acceptance test — the
migration is done when all of them pass.

1. ~~Rewrite this file~~ **done**

2. ~~Rewrite `severity.yaml`~~ **done** — 91 conditions, 75 panel-eligible,
   every one with an exclusion rule (verified: 0 missing, 0 orphans).
   *Blocking, not code:* **a clinician must review it.** It is drafted from
   published can't-miss lists, not verified, and with no eval downstream it is
   the only safeguard in the system.

3. **Author `findings.yaml`** — the finding vocabulary. Stable id + the
   question to ask the patient, organised by system. Must cover what
   discriminates the conditions in `severity.yaml`, especially the joint
   findings DDXPlus could not express (thunderclap onset, tearing pain
   radiating to back, worse lying flat / better leaning forward).
   *Accepts when:* every severity-100 condition has findings that can tell it
   apart from its nearest neighbours, and the canonical case above is
   expressible in the vocabulary.

4. **Write `propose.py`, rewrite `extract.py`** — two separate LLM calls.
   `extract` maps this patient's transcript/documents to `findings.yaml`
   ids, three-valued. `propose` takes findings only (never the transcript)
   and returns candidates + supporting/opposing findings + reasoning. Both
   cached, `temperature=0`, fixed seed. Update `/analyze` in `main.py` to
   chain them.
   *Accepts when:* irrelevant input yields zero candidates; a real case
   yields candidates each carrying readable justification.

5. **Strip `engine.py`** — delete `posterior()`, `tau`, `abstain_entropy`,
   `_entropy`, the information-gain block. Keep and rework `ranked_risk()`
   into severity-band triage, `misfits()`, `_level()`, exclusion state. Add
   `next_question()`: the finding whose support most divides the live
   candidate set.
   *Accepts when:* no probability is computed anywhere, and the app runs
   again (this is what clears the breakage noted above).

6. **Rewrite `selftest.py`** as an invariant suite — never re-ask an answered
   finding; exclusion proposes and never auto-removes; severity floor holds;
   identical input gives identical output; an unrated condition renders
   instead of crashing; irrelevant input fabricates nothing. Plus the two
   `severity.yaml` invariants (every eligible condition has an exclusion
   rule; no orphan rules).
   *Accepts when:* it runs offline in seconds with no data files.

7. **Rewrite the panels in `web/index.html`** — no percentages, no printed
   arithmetic. Severity band and support strength as separate axes. Exclusion
   becomes a confirm prompt. Severity-unrated shown as its own tier. Rename
   "Highest-value question" to "Worth asking next".
   *Accepts when:* no number appears anywhere in the rendered output.

8. **Delete the DDXPlus remnants** — `counts.npz`, `build_counts.py`,
   `tune.py`, `demo_case.py`, `contract.json`, `severity.yaml.ddxplus-legacy`.
   Update `README.md`. The ~980 MB of local `ddxplus/` and `release_*.zip`
   can be deleted from disk too; nothing reads them any more.
   *Accepts when:* a fresh clone runs with no external download — which has
   never been true, since `extract.py` currently reads its catalog from the
   gitignored `ddxplus/release_evidences.json`.

Unchanged throughout: `capture.py`, the `/capture` + `/analyze` split, the
live accumulator and its debounce, the reveal gate, the "Doesn't Fit" panel.

## Later, not now

Recorded so they aren't re-derived, explicitly deferred until the migration
finishes. Nothing here is a todo.

**Read exclusions off the report instead of asking for a tick.** The
clinician photographs the d-dimer result; the document already flows through
`/capture`. Proposing "mark PE ruled out?" from the report itself closes the
loop that currently needs manual input, and it is the single highest-value
thing after the migration.

**The note-versus-transcript check.** The committed note and the full
transcript both already exist in the same session, and nothing compares them.
*"You wrote ?anxiety — the patient also mentioned a long journey and is on a
combined contraceptive."* That is the anchoring bias this product exists to
interrupt, stated directly, and it needs no new inputs.

**Local prevalence as an explicit prior.** An Indian OPD is not a Swiss ED.
Dengue, TB, enteric fever and rheumatic heart disease sit at wholly different
base rates. This is the one place local data genuinely helps — as a stated,
reviewable prior, not resurrected population statistics.

**Speak the question in the patient's language — properly this time.** This
was always billed as a must and was never built. The old build shipped
`HINDI_QUESTIONS`, a dict containing exactly one hardcoded Hindi sentence for
one question id; every other language, and Hindi for any other question, fell
through to speaking the raw English in an English voice. It went unnoticed
because the scripted walkthrough pinned the displayed question to that one id,
so the fallback never fired in a demo.

`capture.translate()` already does this and always did — it takes source and
target language codes and was already being called in the other direction
(hi-IN -> en-IN) for the transcript gloss. The work is small.

**But do not ship it unverified.** Sarvam's translation quality on clinical
phrasing across ten Indian languages is unknown, and a mistranslated question
asked aloud produces a wrong answer that enters the findings pipeline as
genuine evidence. Needs a native speaker to check the medical phrasing —
same standard as `severity.yaml`, for the same reason.

**Measurement.** The deepest gap. Everything above is unfalsifiable until a
clinician scores real consults on whether the panel was useful or noise. This
is what separates a working demo from a tool anyone should rely on, and no
amount of engineering substitutes for it.

## Don't build

Auth, accounts, a real database, medical imaging, any model trained from
scratch, a mobile app, more than one demo language, or any accuracy claim
that isn't measured — which, now, is all of them.

Specifically tempting and specifically wrong: generating `severity.yaml` with
an LLM (it deletes the only audit surface), and putting percentages back
because they look authoritative.
