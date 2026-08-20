"""
Sarvam smoke test — day-plan.md step 1. Do this first: if the key is dead,
better to find out now than at 16:00.

Proves the credentials in .env work against all three services the product
needs (saaras-v3 STT, sarvam-vision, bulbul-v3 TTS) without needing any
pre-recorded audio or scanned document — it round-trips its own test data:

  1. bulbul-v3 speaks a line of Hindi text -> WAV bytes.
  2. saaras-v3 transcribes that WAV back to text (proves STT + the audio
     bulbul-v3 produces are compatible with each other).
  3. A tiny synthetic image with printed English text is generated on the
     fly and sarvam-vision reads it back.

Usage:
    pip install -U sarvamai python-dotenv pillow
    python test_sarvam.py

This only proves the credentials and the three model names are live. It is
NOT the batch/diarized STT flow or the doc_ai flow that capture.py (step 4)
will actually use for the real demo audio and the blister-strip photo —
those need real files and are worth a second, separate smoke test once
demo/ has the recorded audio and the pill-strip photo in it.
"""
import base64
import io
import os
import sys
import time

def fail(step, e):
    print(f"\n FAILED at: {step}")
    print(f"   {type(e).__name__}: {e}")
    print("\nIf this is an auth error, the key in .env is wrong or expired —")
    print("check the Sarvam dashboard. If it's anything else, paste this")
    print("whole output back so we can see which of the three services is down.")
    sys.exit(1)

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("(python-dotenv not installed — reading SARVAM_API_KEY from the "
              "environment directly. `pip install python-dotenv` to load .env.)")

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        print("SARVAM_API_KEY is not set. Create a .env file in the repo root:")
        print("  SARVAM_API_KEY=your_key_here")
        sys.exit(1)
    print(f"SARVAM_API_KEY loaded ({api_key[:4]}...{api_key[-4:]}, "
          f"{len(api_key)} chars)\n")

    try:
        from sarvamai import SarvamAI
    except ImportError:
        print("sarvamai package not installed. Run: pip install -U sarvamai")
        sys.exit(1)

    client = SarvamAI(api_subscription_key=api_key)

    # ---- 1. bulbul-v3 : text -> speech -----------------------------------
    print("=" * 70)
    print("1/3  bulbul-v3 (text-to-speech)")
    # UNVERIFIED translation of the demo's swelling question — not checked
    # by a native speaker, same caveat generate_mock_demo.py's Hindi lines
    # carry. Good enough to smoke-test the round trip, not to ship.
    text_hi = "क्या आपके शरीर में कहीं सूजन है?"
    try:
        tts_resp = client.text_to_speech.convert(
            text=text_hi,
            language_code="hi-IN",
            speaker="shubh",
            model="bulbul:v3",
        )
        audios = getattr(tts_resp, "audios", None) or tts_resp["audios"]
        if not audios:
            raise RuntimeError("response had no audio in `audios`")
        wav_bytes = base64.b64decode(audios[0])
        print(f" bulbul-v3 responded: {len(wav_bytes)} bytes of WAV audio")
        with open("cache_test_bulbul.wav", "wb") as f:
            f.write(wav_bytes)
        print("   saved to cache_test_bulbul.wav for you to listen to")
    except Exception as e:
        fail("bulbul-v3 text-to-speech", e)

    # ---- 2. saaras-v3 : speech -> text ------------------------------------
    print("\n" + "=" * 70)
    print("2/3  saaras-v3 (speech-to-text) — transcribing the clip just made")
    try:
        with open("cache_test_bulbul.wav", "rb") as f:
            stt_resp = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                language_code="hi-IN",
            )
        transcript = getattr(stt_resp, "transcript", None) or stt_resp["transcript"]
        print(f" saaras-v3 responded: \"{transcript}\"")
        print(f"   (sent: \"{text_hi}\")")
    except Exception as e:
        fail("saaras-v3 speech-to-text", e)

    # ---- 3. sarvam-vision : image -> text ---------------------------------
    print("\n" + "=" * 70)
    print("3/3  sarvam-vision (document intelligence) — reading a generated test image")
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print(" Pillow not installed — skipping the vision test.")
        print("   Run: pip install pillow    (or supply your own test image)")
    else:
        try:
            img = Image.new("RGB", (500, 160), "white")
            d = ImageDraw.Draw(img)
            d.text((20, 60), "ETHINYLESTRADIOL 0.03mg", fill="black")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            job = client.doc_ai.digitise(
                file=[("test_strip.png", buf, "image/png")],
                language="en-IN",
                output_format="md",
            )
            # No wait-for-completion helper on the job object in this SDK
            # version — poll get_status(job_id) by hand. A tiny one-page
            # synthetic image finishes in ~1s; 60s is a generous timeout.
            deadline = time.time() + 60
            status = job.status
            while status not in ("completed", "failed"):
                if time.time() > deadline:
                    raise TimeoutError(f"doc_ai job {job.job_id} still "
                                        f"'{status}' after 60s")
                time.sleep(2)
                status = client.doc_ai.get_status(job.job_id).status
            if status == "failed":
                raise RuntimeError(f"doc_ai job {job.job_id} failed")
            results = client.doc_ai.get_results(job.job_id)
            text = results.documents[0].pages[0].blocks[0]["text"]
            print(f" sarvam-vision read back: \"{text}\"")
        except Exception as e:
            fail("sarvam-vision document intelligence", e)

    print("\n" + "=" * 70)
    print("ALL THREE SERVICES RESPONDED. Credentials are good.")
    print("Next: day-plan.md step 2 (wire web/index.html) and step 3")
    print("(record the real Hindi audio + photograph a real blister strip).")

if __name__ == "__main__":
    main()
