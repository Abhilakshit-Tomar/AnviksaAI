# AnviksaAI — engine

The reasoning core. No model call, no network, no randomness. Runs offline in
milliseconds, which is why it's the part you build first and the part that
still demos when the venue wifi dies.

```
pip install numpy pandas pyyaml
python selftest.py                      # verify the maths right now, no data needed
python build_counts.py --data ./ddxplus # once DDXPlus lands. commit counts.npz
python -c "from engine import Engine; print(Engine.load().assess({'E_55':True}))"
```

## Where to get DDXPlus

**Use Figshare, not the HuggingFace mirror.**

<https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585>

You need five files:

```
release_train_patients.zip      ~1M rows — this builds the counts
release_validate_patients.zip
release_test_patients.zip       hold-out for your recall number
release_evidences.json          <- the one that matters
release_conditions.json
```

`release_evidences.json` is why Figshare and not HF. It maps evidence codes to
**English question text** and to the meaning of each categorical value. Without
it your highest-value-question panel displays `E_54=V_11` instead of *"Any
swelling or pain in one calf?"* — the demo beat dies and you'd have to
hand-write 223 question strings at 4am. The HF mirror ships the patient rows
but not reliably the schema files.

Unzip everything into `./ddxplus/`. CC-BY-4.0, so attribution in your deck.

Start the download before you sleep — it's large and venue wifi will be worse
than yours.

## Files

| File | What it is |
|---|---|
| `build_counts.py` | DDXPlus → `P(feature\|pathology)` + priors. Run once, **commit the output**. Never make the demo machine rebuild from 1.3M rows. |
| `severity.yaml` | The hand-curated harm table and the exclusion rules. **The most attackable artifact in the project — get a clinician to read it.** |
| `engine.py` | Posterior, expected harm, exclusion gating, information gain, misfits, abstention. |
| `selftest.py` | Six assertions on a synthetic fixture. Run before the data lands. |

## The four things that are easy to get wrong

**1 · Unknown is not absent.** Evidence is three-valued. Anything not in the
dict is unknown and is *omitted* from the likelihood. Score unknowns as
absent and you multiply `P(¬e | pathology)` for every symptom nobody asked
about — systematically penalising exactly the conditions that would have
caused those symptoms. Your Can't-Miss list would suppress the diagnoses it
exists to surface while your eval numbers still looked fine.

**2 · Severity is log-scaled.** Tiers are 1 / 3 / 10 / 30 / 100, not 1–5. On a
linear scale a 71% panic attack outranks a 15% PE and the panel quietly starts
agreeing with the doctor.

**3 · Exclusion gating is not optional.** With severity 100, PE and ACS clear
any threshold on every chest complaint forever. Without `excluded_by` the
panel never changes and the doctor stops reading it inside a week — alert
fatigue is the most common way tools like this die in the field. The panel
answers *"what have you not ruled out"*, a question that can be **finished**.

**4 · 223 evidence codes flatten to 972 features, and the question engine
must not see them raw.** Four DDXPlus location variables carry 165 values
each (`E_55` pain site, `E_57` radiation, `E_133` affected region, `E_152`
swelling site). Flattening every `(code, value)` pair is right for the
likelihood, but left alone the question engine proposes *"Do you have pain in
the tonsil (L)?"* and fills the runners-up with five variants of the same
variable. `build_counts.py` records each feature's parent code;
`best_question` dedupes by parent and renders the parent's real question text
with the winning value as `answer_hint`. Note `E_57` is the demo's radiation
question — this is not a corner case.

**5 · Severity weighting picks the question; it must never reach the screen.**
`best_question` ranks candidates by information gain over the *risk-weighted*
distribution — "what best resolves what could hurt this patient" rather than
"what best resolves my uncertainty." But the branch previews shown to the
doctor come from the true posterior. Mix them up and every branch reads ~100%
because the lethal condition dominates the weights. (This was a real bug here;
the separation is deliberate.)

## What selftest proves

```
1. unknown != absent        PE 14.7% unknown  vs  3.7% if scored absent
2. harm re-ranks            differential: Panic attack 71%
                            can't-miss:   Pulmonary embolism (14.7% x 100)
3. exclusion gating         [PE, UA, pneumothorax] -> [pneumothorax] after d-dimer+ECG
4. information gain         0.214 bit "recent immobility?"  yes->65% / no->8%
5. contract-shaped payload
6. abstains on no evidence  entropy 0.804 > 0.72
```

## Output shape

`assess()` returns the contract payload directly — hand it to the frontend:

```python
{
  "abstaining": False, "entropy": 0.457,
  "differential":  [{"key","name","p"}, ...],
  "cant_miss":     [{"key","name","p","severity","risk","level","excluded_by"}, ...],
  "best_question": {"id","ig","en","about","branches":{"yes","no"}},
  "runners_up":    [...],
  "misfits":       [{"feature","text","against","ratio"}, ...],
}
```

## Wiring notes

- `actions_taken` is a list of strings the doctor has done (`["ecg","d_dimer"]`).
  Drive it from checkboxes or from parsing the note. This is what makes the
  panel clear.
- Tune `_level()` thresholds (5.0 / 0.75) once you see real risk magnitudes —
  they depend on your priors.
- Tune `abstain_entropy` (0.72) the same way. With 49 pathologies the
  normalised entropy behaves differently than with 5.
- `severity_weighted=False` on `best_question` gives you plain information
  gain, if you want to show both in the deck.

## Expect the numbers to move

The figures in the HTML prototype are illustrative. When you run this against
real DDXPlus counts you will get different probabilities — that is the point,
and you should re-derive the prototype's constants from real output before you
demo. What must not change is the *shape*: a benign condition leading the
differential while a lethal one leads Can't-Miss.
