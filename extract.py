"""
extract.py — the translator. Turns a consultation transcript (plus, if
available, prescription/document text) into DDXPlus evidence codes:
{code: True} for findings confirmed present, {code: False} for findings
EXPLICITLY denied, code omitted entirely if not discussed at all.

day-plan.md step 5's one hard rule: never guess False for anything not
discussed. Scoring an undiscussed symptom as absent multiplies
P(not-e | pathology) for every symptom nobody asked about, which
systematically penalises exactly the conditions Can't-Miss exists to
surface — the eval would still look fine while the panel quietly suppresses
the diagnoses it's supposed to catch. This file's whole job is to preserve
that three-way distinction; get it wrong and engine.py's math is still
correct but is being fed a lie.

Uses Sarvam's sarvam-105b (OpenAI-compatible-shaped chat completions, via
the sarvamai SDK already used by capture.py) with FORCED tool-calling for
structured output. Needs SARVAM_API_KEY in .env — same key capture.py
already uses, no new credential.

PROVIDER HISTORY (each switch driven by a real, verified problem with the
previous one — see git log for the full trail, and extract.py.groq-
batched-wip for the untouched Groq attempt, kept as reference not deleted):
  Anthropic (planned, never had a key)
  -> Gemini (google-genai): gemini-3.7-flash worked but 503'd
     intermittently on the full 223-code catalog, and this key's free
     tier caps at a plain 20 requests/DAY for it — too low to be usable.
     gemini-3.1-flash-lite and gemini-3.5-flash-lite were both tried as
     cheaper alternatives and REJECTED: verified via real side-by-side
     runs that both fabricate PRESENT findings for topics never discussed
     at all (pregnancy status, family psychiatric history, inability to
     get up for 3+ days — none mentioned anywhere in the test transcript).
  -> Groq (openai/gpt-oss-20b, batched): switched for better rate limits,
     but this key's Groq org caps EVERY text model at a shared 8000
     tokens/minute ceiling — confirmed identically across 3 unrelated
     model families, so no model choice fixes it. Had to split the
     223-code catalog into 4 size-bounded batches (greedy bin-packed by
     serialized JSON size, since 4 of the 223 codes — the body-location
     multi-choice codes — are ~20x the size of a typical code and blow
     out a fixed-count split) to fit under the cap at all. Left
     unfinished (test run hit intermittent 400 "tool_use_failed" on one
     batch, not yet fully debugged) when the decision was made to try
     Sarvam instead — that code is preserved untouched in
     extract.py.groq-batched-wip in case Groq becomes worth returning to
     (e.g. with a Dev Tier upgrade).
  -> Sarvam (sarvam-105b): sarvam-105b's context window is large enough
     (128K, per the model card) that the full 223-code catalog + system
     prompt + transcript fits in ONE request — no batching/merging logic
     needed, unlike Groq. Also already the provider capture.py uses for
     STT/vision/TTS, so no new API key.

PROMOTED to active extract.py 2026-08-21 after re-running the exact
fabrication check that rejected the Gemini variants (same
demo/mock_hi-IN/mock_consult_lines.json transcript): came back clean, 5/5
codes traced to something actually said, no E_167/E_29/E_110-style
invention. Side by side, the still-active-at-the-time Gemini flash-lite
fabricated E_110 on this identical input in the same run.

The real installed sarvamai client was introspected before any code was
written — chat.completions is a plain bound METHOD (not a
client.chat.completions.create() sub-resource the way OpenAI/Groq/Gemini
are shaped), confirmed via inspect.signature(), and its real request/
response TYPES were read directly from the installed package source
(sarvamai/requests/chat_completion_tool.py, .../function_definition.py,
sarvamai/types/chat_completion_message_tool_call.py,
.../function_call.py — not docs, not guessed):
    client.chat.completions(model="sarvam-105b", messages=[...],
        tools=[{"type": "function", "function": {"name": ..., "description":
        ..., "parameters": {...}}}],
        tool_choice={"type": "function", "function": {"name": "record_evidence"}})
    -> resp.choices[0].message.tool_calls[0].function.name / .arguments
       (.arguments is a JSON STRING, confirmed both from the FunctionCall
       type's `arguments: str` field AND a real live call — same as
       Groq's shape, unlike Gemini's already-parsed dict).
sarvam-105b (not sarvam-105b-conversations — that variant is tuned for
real-time voice, not one-shot structured extraction, per the model docs).
The default SarvamAI() client timeout is too short for this model's
real latency (a real call hit httpx.ReadTimeout at the default) — the
client below passes timeout=90.0 explicitly, confirmed against the
constructor's real signature, not assumed.

Response includes a `reasoning_content` field (the model narrates its
reasoning before the tool call, observed on a real response) — irrelevant
to extract()'s parsing since only tool_calls[0].function.arguments is
read, but worth knowing if debugging: max_tokens needs enough headroom
for that reasoning trace plus the final tool call on the full catalog, not
just the JSON output size.

KNOWN GAP, confirmed not a prompt-wording problem: 6 of DDXPlus's 15
categorical/multi-choice codes (E_59, E_56, E_58, E_134, E_132, E_136 —
onset-speed and intensity-style 0-10 scales) ship with an EMPTY
value_meaning in release_evidences.json — no V_ labels at all, just bare
integers. _load_catalog() faithfully passes that through as an empty
"values" mapping, so the model has no listed option to match against and
correctly leaves these codes out per rule 3, even when the transcript
clearly discusses onset speed (e.g. "started suddenly"). Checked directly
against the raw JSON, not assumed — this is a real dataset gap, not
something extract.py's prompt can word its way around; a fix would mean
hand-authoring a value_meaning mapping for these 6 codes, out of scope for
today.

Usage:
    from extract import extract
    evidence = extract(transcript_text, doc_text)
    # -> {"E_66": True, "E_100": True, "E_55=V_101": True, "E_91": False, ...}
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

MODEL = os.environ.get("SARVAM_CHAT_MODEL", "sarvam-105b")
# ^ Not sarvam-105b-conversations — that variant is tuned for real-time
# voice turns, not one-shot structured extraction over a 223-code catalog.

_client = None


def _client_singleton():
    global _client
    if _client is None:
        _client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"], timeout=90.0)
    return _client


class _ToolCallMissing(Exception):
    """Raised when a forced-tool-call response comes back with no
    tool_calls anyway — observed live 2026-08-21 on sarvam-105b (a real
    502 in main.py's /assess, not a hypothetical): resp.choices[0].message
    .tool_calls was empty despite tool_choice forcing record_evidence.
    Same transient shape extract.py.groq-batched-wip documented on Groq's
    gpt-oss-20b ("tool_use_failed", confirmed flaky by retrying the exact
    same input and having it succeed) — treated the same way here, as
    retryable, not a hard failure on the first occurrence."""


def _with_retry(fn, *args, max_retries=5, **kwargs):
    """Retries on 429/5xx and on a forced-tool-call response coming back
    without a tool_calls entry (see _ToolCallMissing) — the same
    transient-failure shape seen on every provider this project has used
    so far (Sarvam's own STT/vision calls in capture.py, Gemini, Groq).

    If the caller passed a fixed `seed` (extract() does, for reproducible
    evidence extraction), every RETRY bumps it by the attempt number.
    Observed live 2026-08-21: with temperature=0 AND a fixed seed, a
    _ToolCallMissing response is fully deterministic — every retry sent
    the byte-identical request and got the byte-identical empty-tool-calls
    response back, 5 times, guaranteed failure. The seed only needs to
    move on retry; attempt 0 still uses the caller's original seed, so a
    normal successful call is unaffected and stays reproducible."""
    delay = 2.0
    base_seed = kwargs.get("seed")
    for attempt in range(max_retries):
        if base_seed is not None and attempt > 0:
            kwargs["seed"] = base_seed + attempt
        try:
            resp = fn(*args, **kwargs)
            if not getattr(resp.choices[0].message, "tool_calls", None):
                raise _ToolCallMissing("response had no tool_calls despite forced tool_choice")
            return resp
        except (ApiError, _ToolCallMissing) as e:
            status = getattr(e, "status_code", None) if isinstance(e, ApiError) else None
            retryable = isinstance(e, _ToolCallMissing) or status in (429, 500, 502, 503, 504)
            if retryable and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


_EVIDENCE_PATH = Path(__file__).parent / "ddxplus" / "release_evidences.json"
_catalog_cache = None


def _load_catalog():
    """Compact, English-only view of release_evidences.json for the
    prompt — drops French fields and unused metadata. ~10k tokens for all
    223 codes as of this dataset, checked directly, not estimated."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    raw = json.loads(_EVIDENCE_PATH.read_text(encoding="utf-8"))
    catalog = []
    for code, ev in raw.items():
        entry = {"code": code, "question": ev["question_en"], "type": ev["data_type"]}
        # B = binary, C = categorical (one value from a small fixed set,
        # e.g. speed of onset), M = multi-choice (e.g. location, travel region)
        if ev["data_type"] in ("C", "M"):
            entry["values"] = {v: m.get("en", v)
                                for v, m in ev.get("value_meaning", {}).items()}
            entry["default_value"] = ev.get("default_value")
        catalog.append(entry)
    _catalog_cache = catalog
    return catalog


SYSTEM_PROMPT = """You extract structured clinical evidence codes from a \
doctor-patient consultation transcript and, if provided, prescription or \
document text — strictly for a differential-diagnosis engine that expects \
DDXPlus evidence codes as input.

Rules, in order of importance:

1. THREE-VALUED LOGIC. For each code in the catalog, decide: PRESENT (the \
text affirmatively describes this finding), ABSENT (the text explicitly \
denies or rules this out — "no fever", "never smoked"), or UNDISCUSSED \
(anything not clearly one of those two). Only ever record PRESENT or \
ABSENT. Never guess, infer, or default to ABSENT just because something \
wasn't mentioned — leave it out of both lists entirely. This is the single \
most important rule: a wrongly-guessed ABSENT silently penalises exactly \
the diagnoses that would have caused that symptom.

2. Binary ("B" type) codes: present/absent, no value needed.

3. Categorical/multi-choice ("C"/"M" type) codes: only record as present \
if a specific listed value clearly matches what was said (match against \
the code's "values" mapping). If the topic is discussed but no listed \
value clearly fits, leave it out rather than guessing the closest one. \
NEVER record a code's own default_value as present — the default means \
"no finding", not a finding; recording it would be indistinguishable from \
a real finding and would corrupt the evidence the same way a wrong ABSENT \
guess would.

4. MEDICATION NAMES ARE FACTS, NOT INFERENCE. If document text (a \
prescription, blister strip, medication list) names a specific drug, \
match it to the well-established pharmacological class that drug belongs \
to — this is reading what's written, the same way you'd match a listed \
value for a categorical code (rule 3), not clinical inference about the \
patient's condition. Example: a blister strip listing "Ethinylestradiol" \
and/or "Levonorgestrel" states the patient is taking a combined hormonal \
contraceptive; record the "currently take hormones" code PRESENT on that \
basis alone. This does NOT extend to inferring a diagnosis, symptom, or \
risk factor the text never names — only to identifying what a named, \
already-stated substance actually is.

5. Beyond matching a named substance to its class (rule 4), only extract \
what the given text actually states. Do not use outside medical knowledge \
to infer findings the text didn't mention, even if they seem clinically \
likely. Do not invent a finding just because it seems plausible or common \
for the presenting complaint — if it isn't in the text, it does not go in \
present OR absent.

6. When genuinely uncertain whether to record something, leave it out. \
Omission is always safe here (it becomes UNKNOWN, correctly excluded from \
the likelihood); a wrong guess is not."""


_TOOL = {
    "type": "function",
    "function": {
        "name": "record_evidence",
        "description": "Record which DDXPlus evidence codes are present or explicitly absent in the given text.",
        "parameters": {
            "type": "object",
            "properties": {
                "present": {
                    "type": "array",
                    "description": 'Findings confirmed present. Include "value" (a V_ code) only for type C/M codes.',
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "e.g. E_66"},
                            "value": {"type": "string", "description": "V_ code, only for C/M type codes"},
                        },
                        "required": ["code"],
                    },
                },
                "absent": {
                    "type": "array",
                    "description": "Findings EXPLICITLY denied in the text (not just unmentioned). Type B codes only.",
                    "items": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                },
            },
            "required": ["present", "absent"],
        },
    },
}


def extract(transcript, doc_text=""):
    """transcript: a string, OR a list of utterance dicts (each needs some
    combination of speaker/text keys — see the normalization below) such
    as web/index.html's UTT constants or capture.transcribe()'s
    diarized_transcript.entries.
    doc_text: str — e.g. the "text" field from capture.read_image()'s
    return value. Optional; pass "" if there's no document.

    Returns {code: True} / {"CODE=VALUE": True} for present findings,
    {code: False} for explicitly denied findings — matching the format
    demo_case.py's PRESENTING/WITH_RECORDS/ANSWERED dicts use. Codes not
    discussed are omitted entirely.

    ONE call, full catalog — sarvam-105b's context window fits it, unlike
    Groq's org-wide 8000 TPM cap which forced batching (see module
    docstring's PROVIDER HISTORY).
    """
    if isinstance(transcript, (list, tuple)):
        lines = []
        for u in transcript:
            speaker = u.get("speaker") or u.get("w") or "?"
            text = u.get("text") or u.get("transcript") or u.get("e") or u.get("orig") or ""
            lines.append(f"{speaker}: {text}")
        transcript = "\n".join(lines)

    catalog = _load_catalog()
    client = _client_singleton()

    user_content = (
        f"EVIDENCE CATALOG (JSON array, {len(catalog)} codes):\n"
        f"{json.dumps(catalog, ensure_ascii=False)}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"DOCUMENT TEXT (prescription/blister strip OCR — may be empty):\n{doc_text or '(none)'}\n\n"
        "Call record_evidence with every code you can confidently classify "
        "as present or absent per the rules. Leave everything else out."
    )

    resp = _with_retry(
        client.chat.completions,
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_evidence"}},
        # Observed live 2026-08-21: with no sampling params set, three back-
        # to-back calls on the IDENTICAL transcript+doc_text returned
        # different evidence sets (one run dropped E_16 "anxious" and the
        # specific calf-swelling location value entirely), which flips
        # engine.py's abstain decision run to run independent of anything
        # actually in the input. temperature=0 + a fixed seed makes the
        # same input produce the same extraction, not a fresh dice roll
        # per call — confirmed real params via inspect.signature(), not
        # guessed (see module docstring).
        temperature=0.0,
        seed=0,
        max_tokens=4096,
    )

    # .arguments is a JSON STRING here (unlike Gemini's already-parsed
    # dict — confirmed via a real call and the FunctionCall type source,
    # see module docstring).
    tool_call = resp.choices[0].message.tool_calls[0]
    parsed = json.loads(tool_call.function.arguments)

    evidence = {}
    for item in parsed.get("present", []):
        code = item["code"]
        value = item.get("value")
        key = f"{code}={value}" if value else code
        evidence[key] = True
    for item in parsed.get("absent", []):
        evidence[item["code"]] = False
    return evidence


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("usage: python extract.py <transcript-text-file> [doc-text-file]")
        sys.exit(1)
    transcript_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    doc_text = Path(sys.argv[2]).read_text(encoding="utf-8") if len(sys.argv) > 2 else ""
    result = extract(transcript_text, doc_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
