"""
Generates a synthetic demo/ folder using bulbul-v3, so capture.py (and
later extract.py) can be built and tested end-to-end today without
waiting on a human with a microphone.

NOT a substitute for the real recorded audio before the actual pitch —
deck-guide.md's script has a teammate role-playing the patient live, in
their own voice, and the backup video should sound like that same person.
This is for wiring the pipeline now; do the real recording per
day-plan.md step 3 before rehearsing.

Language: the demo's target language is Marathi (mr-IN) — that's what
CLAUDE.md, deck-guide.md and the frontend's transcript are built around.
Defaulting to Hindi (hi-IN) here is a deliberate, temporary choice for
today only, because nobody on the team can currently verify Marathi
output by ear. The Hindi lines below are MY translation, not checked by a
native speaker — same caveat day-plan.md already puts on the Marathi
lines ("Get the Marathi checked by a native speaker before you rehearse").
Treat both as unverified until a native speaker of the relevant language
confirms them; don't let this become the pitch language without that
check just because it was the default during testing.

Usage:
    python generate_mock_demo.py                # hi-IN, today's default
    python generate_mock_demo.py --lang mr-IN    # the real target language

Writes into demo/mock_<lang>/:
    mock_line_00_patient.wav ... mock_line_04_patient.wav   one file per
                                                              utterance
    mock_consult_lines.json      ground truth: which file is which
                                  speaker/text, to check capture.py's
                                  transcript against
    mock_blister_strip.png       SYNTHETIC pill-strip label — not a real
                                  photo, and not language-dependent (it's
                                  an English drug label either way).
                                  sarvam-vision on an actual strip will
                                  look different (real fonts, real
                                  packaging, likely worse print quality).
                                  Swap for a real photo before rehearsing.
"""
import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from capture import speak

# Same sentences as web/index.html's UTT/Q_ASK/Q_ANS constants, so the mock
# audio matches what the frontend already narrates — just in whichever
# language is being tested today.
LINES_BY_LANG = {
    "mr-IN": [
        {"speaker": "patient", "text": "छातीत जड वाटतंय आणि श्वास घ्यायला त्रास होतोय."},
        {"speaker": "patient", "text": "काल रात्रीपासून. अचानक सुरू झालं. छाती धडधडतेय."},
        {"speaker": "doctor",  "text": "काही टेन्शन आहे का? घरी सगळं ठीक आहे ना?"},
        {"speaker": "patient", "text": "नाही, तसं काही नाही. चार दिवसांपूर्वी विमानाने परत आले."},
        {"speaker": "patient", "text": "हो, डाव्या पायाच्या पोटरीला सूज आहे."},
    ],
    # UNVERIFIED translation — see the module docstring. Good enough to
    # exercise capture.py/extract.py today, not good enough to ship.
    "hi-IN": [
        {"speaker": "patient", "text": "छाती में भारीपन महसूस हो रहा है और सांस लेने में तकलीफ हो रही है।"},
        {"speaker": "patient", "text": "कल रात से। अचानक शुरू हुआ। दिल तेज़ धड़क रहा है।"},
        {"speaker": "doctor",  "text": "कोई टेंशन है क्या? घर में सब ठीक है ना?"},
        {"speaker": "patient", "text": "हाँ, घबराहट सी हो रही है, पर घर में सब ठीक है। चार दिन पहले फ्लाइट से वापस आई।"},
        {"speaker": "patient", "text": "हाँ, बाएं पैर की पिंडली में सूजन है।"},
    ],
}
SPEAKER_VOICE = {"patient": "priya", "doctor": "shubh"}  # any two distinct
# bulbul:v3 speakers — picked only to make the mock files easy to tell
# apart while testing, not a decision about the product's actual voice.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi-IN", choices=sorted(LINES_BY_LANG),
                    help="BCP-47 code — hi-IN for today's testing (default), "
                         "mr-IN for the real target language")
    args = ap.parse_args()

    out_dir = Path("demo") / f"mock_{args.lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, line in enumerate(LINES_BY_LANG[args.lang]):
        out = out_dir / f"mock_line_{i:02d}_{line['speaker']}.wav"
        speak(line["text"], out, language_code=args.lang,
              speaker=SPEAKER_VOICE[line["speaker"]])
        manifest.append({**line, "file": out.name})
        print(f"wrote {out}")

    (out_dir / "mock_consult_lines.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'mock_consult_lines.json'} ({len(manifest)} lines)")

    from PIL import Image, ImageDraw
    img = Image.new("RGB", (500, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 60), "ETHINYLESTRADIOL 0.03mg", fill="black")
    d.text((20, 100), "LEVONORGESTREL 0.15mg", fill="black")
    img_path = out_dir / "mock_blister_strip.png"
    img.save(img_path)
    print(f"wrote {img_path}  (SYNTHETIC — replace with a real photo before rehearsing)")


if __name__ == "__main__":
    main()
