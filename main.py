"""
main.py — the wire. FastAPI server tying capture -> extract -> engine
together, per day-plan.md step 6.

    POST /assess
        Accepts EITHER of two request bodies, distinguished by Content-Type
        (this is one route, not two — see the content-type branch in
        assess() below):

        1. application/json (the scripted demo flow, unchanged):
           {"audio_paths": [str, ...], "image_path": str|null,
            "actions_taken": [str], "language_code": str}
           audio_paths are server-side file paths (resolved relative to the
           repo root main.py runs from) — this is what web/index.html's
           loadContract() sends, replaying demo/mock_hi-IN/*.wav.

        2. multipart/form-data (live recording and/or patient-supplied
           documents, additional paths):
           an optional file field "audio" (any audio format the browser's
           MediaRecorder produced — typically audio/webm), optional
           repeated file fields "documents" (multiple prescriptions/pill
           strip photos/reports — any image or PDF capture.read_image()
           accepts), plus optional form fields "language_code",
           "image_path" (single legacy path, kept for parity with the JSON
           branch), "actions_taken" (JSON-encoded list, e.g. '["ecg"]').
           At least one of "audio" or "documents" is required. Every
           upload is written to a temp file — audio is handed to
           capture.transcribe() and each document to capture.read_image()
           the exact same way a demo/mock_hi-IN path would be — capture.py
           itself needed no changes for either. Temp files are deleted
           after the request finishes, success or failure.

        Both branches converge on the same response shape: the contract.json
        shape (differential, cant_miss, best_question, runners_up, misfits,
        ...) PLUS "transcript" and "_evidence", so the caller can see what
        capture.py heard and what extract.py derived from it, not just the
        final numbers. Also "scanned_documents" — a list of
        {"name": filename, "chars": N, "text": full OCR text} for every
        document OCR'd this call (empty list if none), so the frontend's
        "Patient-supplied records" panel can show a real count/list — and,
        per document, the actual extracted text on click — instead of the
        scripted demo's hardcoded chips. Also, best-effort, "transcript_en" — an English
        translation (Sarvam text.translate) of "transcript", present
        whenever language_code isn't already "en-IN" and translation
        succeeds; a translation failure just omits the field. language_code
        also selects the STT language passed to capture.transcribe() for
        every audio_path; defaults to "hi-IN".

    POST /capture  -> lean STT-or-OCR-only sibling of /assess, no
        extract()/engine.assess() call. multipart/form-data with either one
        "audio" file (+ optional "language_code") -> {"transcript": str,
        "transcript_en": str|absent}, or one-or-more "documents" files ->
        {"scanned_documents": [{"name", "chars", "text"}, ...]}. This is
        what web/index.html's live-recording and document-upload flows
        actually call now: previously they hit /assess for the STT/OCR AND
        got a full extract()+engine.assess()+TTS run whose result was
        immediately thrown away (the panels were painted from a SEPARATE
        /analyze call over the accumulated transcript, per that route's own
        docstring below) — every recording or upload was silently burning
        two full Sarvam extraction calls instead of one. /capture does only
        the STT/OCR half; /analyze does the extract+engine half once, on the
        accumulated text. /assess itself is untouched, still available for
        the JSON audio_paths branch and any caller that wants the combined
        one-shot pipeline.

        audio_paths (JSON branch) is a LIST because the demo's consultation
        is recorded as several short single-utterance files
        (demo/mock_hi-IN/mock_line_*.wav) rather than one long take — each is
        transcribed and the results are joined into one transcript for
        extract.py. A single-element list works fine for a one-file
        consultation too. The multipart branch always sends exactly one
        recording, so it's wrapped in a single-element list the same way.

        Any failure in transcription, document scanning, or evidence
        extraction is caught and returned as a clean 502 JSON error
        ({"detail": "..."}) rather than an unhandled 500 — there's no
        offline fallback anymore, so a live failure mid-pitch needs to fail
        gracefully, not crash.

    GET /contract.json  -> serves contract.json from the repo root, so the
        EXISTING static frontend (web/index.html's fetch('../contract.json'))
        keeps working unchanged when served through this app instead of
        `python -m http.server`.

    GET /web/...  -> serves web/index.html and anything else under web/.

    --offline  -> every /assess call replays demo/offline_response.json
        instead of touching Sarvam/Anthropic at all. day-plan.md's explicit
        safety net for bad venue wifi. YOU MUST CREATE THAT FILE before this
        flag is usable — it isn't generated by this script. Easiest way:
        `python demo_case.py --dump --stage 2` to get a real engine payload
        into contract.json, copy it to demo/offline_response.json, and add
        a "transcript" key with whatever the real backup recording actually
        says (or the mock transcript, if that's what's being demoed).

NOTE: only the FastAPI/Starlette plumbing here is "obviously correct" —
capture.py and extract.py's own internals were each wrong on first attempt
and needed verification against the real SDKs. This file hasn't been run
yet (no network from where it was written). Before trusting it: run it,
POST a real request at /assess, and confirm the response shape actually
matches what web/index.html would need — don't assume the glue is right
just because each piece was individually verified.

Usage:
    pip install fastapi "uvicorn[standard]" pydantic
    python main.py                  # live, http://127.0.0.1:8000/web/
    python main.py --offline        # replay demo/offline_response.json
    python main.py --port 8080
"""
import argparse
import asyncio
import json
import traceback
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import capture
import extract
from engine import Engine

REPO_ROOT = Path(__file__).parent
CONTRACT_PATH = REPO_ROOT / "contract.json"
OFFLINE_RESPONSE_PATH = REPO_ROOT / "demo" / "offline_response.json"

app = FastAPI(title="AnviksaAI")

_engine = None
_offline = False  # set from --offline in __main__; read by the /assess route


def get_engine():
    global _engine
    if _engine is None:
        _engine = Engine.load()
    return _engine


class AssessRequest(BaseModel):
    audio_paths: list[str]
    image_path: str | None = None
    actions_taken: list[str] = []
    # BCP-47 code, restricted client-side to bulbul-v3's 11 TTS-supported
    # languages (see web/index.html's LANGUAGES) even though saaras-v3 STT
    # itself understands more — a language with no TTS voice couldn't ever
    # speak best_question aloud, so there's no point offering it here.
    language_code: str = "hi-IN"


class AnalyzeRequest(BaseModel):
    transcript: str = ""
    doc_text: str = ""
    actions_taken: list[str] = []


def _run_assessment(audio_paths, documents, actions_taken, language_code):
    """The actual assess pipeline: transcribe -> scan -> extract -> engine.
    Shared by both branches of assess() below — JSON
    audio_paths/image_path and multipart-uploaded audio/documents converge
    here once each has a server-side file path in hand, so this is the ONE
    place that logic lives.

    documents: list of {"path": str, "name": str} — "name" is the
    human-readable filename (the original upload's filename for the
    multipart branch, or Path(image_path).name for the JSON branch) used
    only for "scanned_documents" below and error messages; "path" is what's
    actually opened."""
    # capture.transcribe() returns the full batch-STT result dict (diarized
    # entries, request_id, etc.) — extract.py only needs the flat transcript
    # text, not per-speaker attribution, so that's all that's passed on.
    # See capture.py's transcribe() docstring for the full return shape if
    # diarization ever needs to reach extract.py too.
    transcript_parts = []
    for p in audio_paths:
        audio_path = Path(p)
        if not audio_path.exists():
            raise HTTPException(400, f"audio_path not found: {audio_path}")
        try:
            stt_result = capture.transcribe(str(audio_path), language_code=language_code)
        except Exception as e:
            raise HTTPException(502, f"transcription failed for {audio_path.name}: {e}")
        transcript_parts.append(stt_result["transcript"])
    transcript = "\n".join(transcript_parts)

    # Each uploaded document (medication list, supplement info, prescription,
    # pill-strip photo, ...) is OCR'd independently via the same
    # capture.read_image() path already verified for the single blister-strip
    # photo, then every document's extracted text is joined into ONE doc_text
    # blob — extract.extract() has always taken a single doc_text string, and
    # concatenating here (rather than teaching it about multiple documents)
    # keeps that function unchanged. scanned_documents mirrors this loop so
    # the frontend can show what was actually scanned, not just a page count.
    doc_chunks = []
    scanned_documents = []
    for doc in documents:
        img_path = Path(doc["path"])
        if not img_path.exists():
            raise HTTPException(400, f"image_path not found: {img_path}")
        try:
            text = capture.read_image(str(img_path))["text"]
        except Exception as e:
            raise HTTPException(502, f"document scan failed for {doc['name']}: {e}")
        doc_chunks.append(text)
        scanned_documents.append({"name": doc["name"], "chars": len(text), "text": text})
    doc_text = "\n\n".join(doc_chunks)

    try:
        evidence = extract.extract(transcript, doc_text)
    except Exception as e:
        raise HTTPException(502, f"evidence extraction failed: {e}")

    payload = get_engine().assess(evidence, actions_taken=actions_taken)
    payload["transcript"] = transcript
    payload["_evidence"] = evidence
    payload["scanned_documents"] = scanned_documents

    # Best-effort English gloss of the transcript, for the frontend to show
    # under the in-language transcript (the "what did I just record" check
    # for a professional who doesn't read the selected language). Skipped
    # entirely when the language IS English — there's nothing to gloss —
    # and, like the TTS block below, a translation failure never fails the
    # whole /assess call: the field is just absent.
    if transcript.strip() and language_code != "en-IN":
        try:
            payload["transcript_en"] = capture.translate(
                transcript, source_language_code=language_code,
                target_language_code="en-IN")
        except Exception:
            pass

    return payload


@app.post("/assess")
async def assess(request: Request):
    if _offline:
        if not OFFLINE_RESPONSE_PATH.exists():
            raise HTTPException(
                500,
                f"--offline is set but {OFFLINE_RESPONSE_PATH} doesn't exist "
                "yet — see this file's module docstring for how to create it.",
            )
        return json.loads(OFFLINE_RESPONSE_PATH.read_text(encoding="utf-8"))

    content_type = request.headers.get("content-type", "")
    tmp_paths = []  # every temp file this request creates, cleaned up in `finally` below
    try:
        if content_type.startswith("multipart/form-data"):
            # Live-recording and/or patient-document path: the browser's
            # MediaRecorder output and/or scanned files, uploaded as real
            # files instead of server-side paths. Each is saved to a temp
            # file so capture.transcribe()/capture.read_image() can be
            # called exactly as they are for the scripted flow — they take
            # a path either way, they never learn the difference.
            form = await request.form()
            audio_paths = []
            upload = form.get("audio")
            if upload is not None:
                suffix = Path(getattr(upload, "filename", "") or "").suffix or ".webm"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await upload.read())
                    audio_tmp_path = Path(tmp.name)
                tmp_paths.append(audio_tmp_path)
                audio_paths = [str(audio_tmp_path)]

            documents = []
            for doc_upload in form.getlist("documents"):
                name = getattr(doc_upload, "filename", "") or "document"
                suffix = Path(name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await doc_upload.read())
                    doc_tmp_path = Path(tmp.name)
                tmp_paths.append(doc_tmp_path)
                documents.append({"path": str(doc_tmp_path), "name": name})

            legacy_image_path = form.get("image_path")
            if legacy_image_path:
                documents.append({"path": legacy_image_path, "name": Path(legacy_image_path).name})

            if not audio_paths and not documents:
                raise HTTPException(400, "multipart /assess requires an 'audio' file and/or 'documents' files")

            actions_raw = form.get("actions_taken")
            # run_in_threadpool, not a direct call: _run_assessment is plain
            # synchronous blocking code (Sarvam/extract calls, 10-90s each)
            # -- called directly from this async handler, it froze uvicorn's
            # entire event loop for that whole duration, refusing even NEW
            # incoming connections. Confirmed live 2026-08-21: the third of
            # three sequential /assess calls got net::ERR_CONNECTION_REFUSED
            # while the second was still blocking the loop.
            payload = await run_in_threadpool(
                _run_assessment,
                audio_paths=audio_paths,
                documents=documents,
                actions_taken=(json.loads(actions_raw) if actions_raw else []),
                language_code=(form.get("language_code") or "hi-IN"),
            )
        else:
            try:
                body = await request.json()
                req = AssessRequest(**body)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(422, f"invalid /assess JSON body: {e}")
            documents = [{"path": req.image_path, "name": Path(req.image_path).name}] if req.image_path else []
            payload = await run_in_threadpool(
                _run_assessment,
                audio_paths=req.audio_paths,
                documents=documents,
                actions_taken=req.actions_taken,
                language_code=req.language_code,
            )
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)

    return payload


@app.post("/capture")
async def capture_only(request: Request):
    """See module docstring: STT-or-OCR only, no extract()/engine.assess().
    Exactly one of "audio" or one-or-more "documents" per call — mirrors
    how liveSend()/docSend() in web/index.html already call this one at a
    time, so there's no need to support a mixed audio+documents request."""
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(400, "/capture requires multipart/form-data")

    form = await request.form()
    tmp_paths = []
    try:
        upload = form.get("audio")
        if upload is not None:
            suffix = Path(getattr(upload, "filename", "") or "").suffix or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await upload.read())
                audio_tmp_path = Path(tmp.name)
            tmp_paths.append(audio_tmp_path)
            language_code = form.get("language_code") or "hi-IN"
            try:
                stt_result = await run_in_threadpool(
                    capture.transcribe, str(audio_tmp_path), language_code=language_code)
            except Exception as e:
                # Temporary diagnostic: a real /capture 502 was reported live
                # 2026-08-22 with no way to see WHY from the access log alone
                # (a caught exception -> HTTPException never prints a
                # traceback). Printed here, not raised further, so it stays
                # in server stdout/stderr for whoever's watching the process.
                traceback.print_exc()
                raise HTTPException(502, f"transcription failed: {e}")
            transcript = stt_result["transcript"]
            result = {"transcript": transcript}
            if transcript.strip() and language_code != "en-IN":
                try:
                    result["transcript_en"] = await run_in_threadpool(
                        capture.translate, transcript,
                        source_language_code=language_code, target_language_code="en-IN")
                except Exception:
                    pass
            return result

        doc_uploads = form.getlist("documents")
        if doc_uploads:
            docs_meta = []
            for doc_upload in doc_uploads:
                name = getattr(doc_upload, "filename", "") or "document"
                suffix = Path(name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await doc_upload.read())
                    doc_tmp_path = Path(tmp.name)
                tmp_paths.append(doc_tmp_path)
                docs_meta.append((name, doc_tmp_path))

            # OCR every document CONCURRENTLY, not one at a time -- these
            # are independent Sarvam vision calls, so a doctor uploading
            # 3 records at once was paying 3x the wall time for no reason.
            # CLAUDE.md's 10-30 req/min vision cap is a per-MINUTE ceiling
            # on a demo-sized handful of documents in one request; it's
            # the "render loop" (re-scanning on every repaint) that
            # warning is actually about, not a single batched upload.
            async def _scan(name, path):
                try:
                    result = await run_in_threadpool(capture.read_image, str(path))
                except Exception as e:
                    raise RuntimeError(f"{name}: {e}") from e
                return name, result["text"]

            try:
                results = await asyncio.gather(*[_scan(n, p) for n, p in docs_meta])
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(502, f"document scan failed for {e}")

            scanned_documents = [{"name": name, "chars": len(text), "text": text}
                                  for name, text in results]
            return {"scanned_documents": scanned_documents}

        raise HTTPException(400, "/capture requires an 'audio' file or 'documents' files")
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Lean sibling of /assess: extract.extract() + engine.assess() only,
    no capture.py calls at all (no STT/OCR/TTS, no Sarvam usage beyond
    extract() itself). For recomputing Can't-Miss every time live evidence
    changes (a new live recording lands, a new document gets scanned)
    without re-burning Sarvam STT/vision quota re-transcribing/re-OCRing
    text the caller already has from an earlier /assess response — see
    web/index.html's runAnalyze()."""
    try:
        evidence = extract.extract(req.transcript, req.doc_text)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(502, f"evidence extraction failed: {e}")
    payload = get_engine().assess(evidence, actions_taken=req.actions_taken)
    payload["_evidence"] = evidence
    return payload


@app.get("/contract.json")
def serve_contract():
    if not CONTRACT_PATH.exists():
        raise HTTPException(404, "contract.json doesn't exist yet — run demo_case.py --dump-all")
    return FileResponse(CONTRACT_PATH, media_type="application/json")


@app.get("/")
def root():
    return RedirectResponse("/web/")


# Mounted at /web (NOT at "/") deliberately — mounting the whole repo root
# would also serve .env, the ddxplus/ CSVs, and .git/ over HTTP. Only
# web/index.html and the one explicit /contract.json route above are
# actually exposed.
app.mount("/web", StaticFiles(directory=str(REPO_ROOT / "web"), html=True), name="web")

# Also safe to expose directly: demo/ only ever holds the mock consult
# audio, the synthetic blister-strip photo, and their JSON manifest — no
# secrets, unlike the repo root. Lets web/index.html's per-line play
# buttons fetch the real recorded wav files straight from the browser
# instead of round-tripping them through /assess as base64.
app.mount("/demo", StaticFiles(directory=str(REPO_ROOT / "demo")), name="demo")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                     help="replay demo/offline_response.json, no API calls at all")
    # PORT env var is the convention hosting platforms (Render, Railway, ...)
    # inject and expect the app to bind to — --port still overrides it for
    # local runs. --host defaults to 0.0.0.0 (not 127.0.0.1) because a
    # loopback-only bind is unreachable from outside the container on any
    # of those platforms; harmless for local dev too, since localhost still
    # resolves to it.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    _offline = args.offline

    if _offline and not OFFLINE_RESPONSE_PATH.exists():
        print(f"WARNING: --offline set but {OFFLINE_RESPONSE_PATH} doesn't "
              f"exist yet. /assess will 500 until it's created — see this "
              f"file's module docstring.")

    import uvicorn
    print(f"Serving on http://{args.host}:{args.port}/web/  "
          f"({'OFFLINE — no API calls' if _offline else 'LIVE'})")
    uvicorn.run(app, host=args.host, port=args.port)
