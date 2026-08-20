"""
capture.py — the ears (and mouth). Turns a recorded audio file into text,
a photographed prescription into text, and text into spoken audio.

Three functions, matching day-plan.md step 4:
    transcribe(path)        -> str            (saaras-v3)
    read_image(path)        -> dict            (sarvam-vision / doc_ai)
    speak(text, out_path)   -> out_path         (bulbul-v3)

Every call is cached to cache/ keyed by a hash of its input, so the same
audio/image/text is never sent to Sarvam twice — vision is capped at
10-30 req/min on every tier per CLAUDE.md, that's the ceiling this project
will hit, not credits. 429s are retried with exponential backoff.

Built and verified against the ACTUAL installed sarvamai==0.1.30 client,
not its docs — those turned out to be wrong about several method names
(target_language_code vs language_code, wait_until_complete() not
existing on doc_ai jobs). See test_sarvam.py's commit history for how
that was found. If you extend this file, verify new calls the same way:
introspect the real client/response objects before trusting a docs page.
"""
import base64
import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sarvamai import SarvamAI

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

_client = None


def client():
    global _client
    if _client is None:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY not set — check .env")
        _client = SarvamAI(api_subscription_key=key)
    return _client


def _hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:24]


def _with_retry(fn, *args, max_retries=5, **kwargs):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------
def transcribe(path, language_code="mr-IN"):
    """Audio file -> transcript text, via saaras-v3.

    NOTE: this uses the simple (single-call, non-batch, non-diarized)
    speech_to_text.transcribe — the exact call test_sarvam.py proved
    works. CLAUDE.md's architecture table calls for *batch* STT with
    diarization for the real multi-speaker consult recording. Before
    wiring that in: introspect dir(client().speech_to_text) and whatever
    a batch call returns, the same way doc_ai's real polling surface was
    found — don't trust a docs page's method names for it, one already
    turned out to be invented (job.wait_until_complete() doesn't exist on
    doc_ai; the equivalent speech_to_text_job flow may have the same
    problem). If the demo audio is short/simple enough that extract.py
    can tell patient from doctor by grammatical person alone, diarization
    may not be needed for the hackathon demo at all.
    """
    path = Path(path)
    audio_bytes = path.read_bytes()
    cpath = CACHE_DIR / f"stt_{_hash(audio_bytes, language_code)}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())["transcript"]

    with open(path, "rb") as f:
        resp = _with_retry(
            client().speech_to_text.transcribe,
            file=f, model="saaras:v3", language_code=language_code,
        )
    transcript = getattr(resp, "transcript", None) or resp["transcript"]
    cpath.write_text(json.dumps({"transcript": transcript}, ensure_ascii=False),
                      encoding="utf-8")
    return transcript


# --------------------------------------------------------------------------
def read_image(path, language="en-IN"):
    """Photo of a prescription/blister strip -> extracted text, via
    sarvam-vision (doc_ai.digitise). Polls manually — get_status()/
    get_results() are the real methods on this SDK version, not the
    wait_until_complete()/get_file_results() the docs describe."""
    path = Path(path)
    img_bytes = path.read_bytes()
    cpath = CACHE_DIR / f"vision_{_hash(img_bytes, language)}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())

    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "pdf": "application/pdf"}.get(path.suffix.lstrip(".").lower(),
                                          "application/octet-stream")
    with open(path, "rb") as f:
        job = _with_retry(
            client().doc_ai.digitise,
            file=[(path.name, f, mime)], language=language, output_format="md",
        )

    deadline = time.time() + 60
    status = job.status
    while status not in ("completed", "failed"):
        if time.time() > deadline:
            raise TimeoutError(f"doc_ai job {job.job_id} still '{status}' after 60s")
        time.sleep(2)
        status = client().doc_ai.get_status(job.job_id).status
    if status == "failed":
        raise RuntimeError(f"doc_ai job {job.job_id} failed")

    results = client().doc_ai.get_results(job.job_id)
    blocks = []
    for doc in results.documents:
        for page in doc.pages:
            for b in page.blocks:
                text = b["text"] if isinstance(b, dict) else b.text
                blocks.append(text)
    extracted = {"blocks": blocks, "text": "\n".join(blocks)}
    cpath.write_text(json.dumps(extracted, ensure_ascii=False), encoding="utf-8")
    return extracted


# --------------------------------------------------------------------------
def speak(text, out_path, language_code="mr-IN", speaker="shubh"):
    """Text -> spoken audio, via bulbul-v3. Writes a WAV file at out_path
    and returns out_path. Cached by (text, language_code, speaker), so the
    same line is never re-synthesised — matters for both rate limits and
    for the fallback video sounding consistent take to take."""
    key_hash = _hash(text, language_code, speaker)
    cpath = CACHE_DIR / f"tts_{key_hash}.wav"
    if not cpath.exists():
        resp = _with_retry(
            client().text_to_speech.convert,
            text=text, language_code=language_code, speaker=speaker,
            model="bulbul:v3",
        )
        audios = getattr(resp, "audios", None) or resp["audios"]
        cpath.write_bytes(base64.b64decode(audios[0]))
    out_path = Path(out_path)
    out_path.write_bytes(cpath.read_bytes())
    return str(out_path)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("usage: python capture.py <audio-or-image-path>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if p.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
        print(transcribe(p))
    else:
        print(json.dumps(read_image(p), ensure_ascii=False, indent=2))
