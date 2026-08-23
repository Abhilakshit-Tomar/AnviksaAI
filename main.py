"""main.py — the wire. FastAPI, two routes, kept deliberately apart.

    POST /capture   multipart/form-data. STT or OCR, nothing else.
        Either one "audio" file (+ optional "language_code")
            -> {"transcript": str, "transcript_en": str|absent}
        or one-or-more "documents" files
            -> {"scanned_documents": [{"name", "chars", "text"}, ...]}

    POST /analyze   application/json. extract -> propose -> triage, over
        everything accumulated so far.
            {"transcript": str, "doc_text": str,
             "confirmed_exclusions": [str], "asked": [str]}
            -> the panel payload (see engine.assess)

    GET  /web/...   the single-file frontend.

KEEP THESE TWO SEPARATE. They look mergeable and are not. When the frontend
used one combined endpoint, every recording and every upload ran a full
extraction whose result was immediately thrown away, then ran a second one
from the panel refresh — double the Sarvam cost and double the latency on
every single action. That merged route (/assess) has been deleted rather
than left available, because leaving it available is how it comes back.

WHY /analyze RE-SENDS THE WHOLE TRANSCRIPT EVERY TIME. Evidence arrives in
pieces: a recording, then a document, then another recording, minutes apart.
Extraction re-runs across the FULL accumulation every time, never
incrementally over the new chunk. A later statement can correct an earlier
one ("no, it's the right calf, not the left") and only a model seeing both
can resolve that; merging per-chunk findings would keep both as true. This
costs a full call per update, and the disk cache in llm.py absorbs the
repeats.

TWO THINGS SURVIVE ACROSS UPDATES, and the client sends them back on every
call: confirmed_exclusions (once a clinician ticks "PE ruled out by CTPA",
that holds for the rest of the consult) and asked (findings already put to
the patient, so the panel never re-asks). Candidates, ordering and the next
question all recompute fresh.

THERE IS NO OFFLINE REPLAY MODE AND NO PRE-COMPUTED PAYLOAD. --offline
replayed demo/offline_response.json and GET /contract.json served a
pre-computed engine result; both are gone. They are how a "live" demo kept
showing canned output that had nothing to do with the person in the room.
Everything on screen comes from this patient, this session, or the panel
says plainly that it has nothing.

Usage:
    pip install -r requirements.txt
    python main.py            # http://127.0.0.1:8000/web/
    python main.py --port 8080
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import capture
import extract
import propose
from engine import Engine

REPO_ROOT = Path(__file__).parent

app = FastAPI(title="AnviksaAI")

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = Engine.load()
    return _engine


class AnalyzeRequest(BaseModel):
    transcript: str = ""
    doc_text: str = ""
    # Conditions a CLINICIAN has confirmed as ruled out. Never inferred here.
    confirmed_exclusions: list[str] = []
    # Findings already put to the patient but not yet answered.
    asked: list[str] = []


@app.post("/capture")
async def capture_only(request: Request):
    """STT or OCR only — no extract(), no propose(), no engine.

    Exactly one "audio" file, or one-or-more "documents", per call. That
    mirrors how liveSend()/docSend() in web/index.html call it, so there is
    no need to support a mixed audio+documents request.
    """
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
                # A caught exception turned into an HTTPException prints no
                # traceback, which left a real live 502 with nothing in the
                # access log to diagnose it from. Printed to server stderr.
                traceback.print_exc()
                raise HTTPException(502, f"transcription failed: {e}")
            transcript = stt_result["transcript"]
            result = {"transcript": transcript}
            # English gloss, so a clinician who does not read the selected
            # language can still check what was just recorded. Best-effort:
            # a translation failure omits the field rather than failing the
            # capture that succeeded.
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

            # OCR every document CONCURRENTLY. These are independent vision
            # calls, so a doctor uploading three records at once was paying
            # three times the wall clock for no reason. The 10-30 req/min
            # vision cap is a per-minute ceiling; what that warning is
            # actually about is re-scanning inside a render loop, not one
            # batched upload.
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

            return {"scanned_documents": [
                {"name": name, "chars": len(text), "text": text}
                for name, text in results
            ]}

        raise HTTPException(400, "/capture requires an 'audio' file or 'documents' files")
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


def _pipeline(req: AnalyzeRequest):
    """extract -> propose -> triage. Two model calls, never one.

    Synchronous and blocking (two chat completions, tens of seconds), which
    is why the route below runs it in a threadpool rather than calling it
    directly: called inline from an async handler it froze uvicorn's event
    loop for the whole duration and refused even new incoming connections.
    Confirmed live — the third of three sequential requests got
    ERR_CONNECTION_REFUSED while the second was still blocking.
    """
    findings, quotes = extract.extract(req.transcript, req.doc_text)
    candidates = propose.propose(findings)
    payload = get_engine().assess(
        candidates, findings,
        confirmed_exclusions=req.confirmed_exclusions,
        asked=req.asked,
    )
    # The quote behind each finding. With no dataset and no eval, a clinician
    # being able to trace a finding back to the words that produced it is a
    # real part of the safety net — and it is how a fabricated finding gets
    # caught.
    payload["_findings"] = findings
    payload["_quotes"] = quotes
    return payload


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        return await run_in_threadpool(_pipeline, req)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(502, f"analysis failed: {e}")


@app.get("/")
def root():
    return RedirectResponse("/web/")


# Mounted at /web, NOT at "/". Mounting the repo root would also serve .env
# and .git/ over HTTP.
app.mount("/web", StaticFiles(directory=str(REPO_ROOT / "web"), html=True), name="web")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    # PORT is the env var hosting platforms inject and expect the app to bind
    # to. --host defaults to 0.0.0.0 because a loopback-only bind is
    # unreachable from outside a container; harmless locally, since localhost
    # still resolves to it.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    import uvicorn
    print(f"Serving on http://{args.host}:{args.port}/web/")
    uvicorn.run(app, host=args.host, port=args.port)
