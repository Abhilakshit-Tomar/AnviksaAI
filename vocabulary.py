"""vocabulary.py — loads findings.yaml, once.

extract.py maps a transcript onto these ids. propose.py reasons over them.
engine.py orders them. web/index.html prints their labels. Four readers, one
loader, so a typo in the YAML fails in one place at import time rather than
four places at request time.

The file itself carries the reasoning for why the vocabulary is fixed at all
and why every finding is binary — read findings.yaml, not this docstring.
"""

from pathlib import Path

import yaml

_PATH = Path(__file__).parent / "findings.yaml"

_raw = yaml.safe_load(_PATH.read_text(encoding="utf-8"))

# id -> {"label": str, "ask": str, "system": str}
FINDINGS = {}
for _system, _group in _raw["findings"].items():
    for _fid, _meta in _group.items():
        if _fid in FINDINGS:
            raise ValueError(
                f"findings.yaml: duplicate id {_fid!r} "
                f"(in {FINDINGS[_fid]['system']} and {_system}). "
                "Ids must be unique across systems — they are what lets a "
                "finding be matched across turns of the same consult."
            )
        FINDINGS[_fid] = {"label": _meta["label"], "ask": _meta["ask"], "system": _system}

# exclusion token -> human label, e.g. "ctpa" -> "CT pulmonary angiogram".
# Not findings: nobody asks a patient whether they have a CTPA. These are
# actions a clinician records, and they need labels because they appear in
# the one prompt in this system that removes a warning from the screen.
INVESTIGATIONS = dict(_raw["investigations"])


def label(fid):
    """Panel label for a finding id, falling back to the raw id so an
    unrecognised finding renders visibly rather than vanishing."""
    entry = FINDINGS.get(fid)
    return entry["label"] if entry else fid


def question(fid):
    """The question to put to the patient. Returns None for an unknown id."""
    entry = FINDINGS.get(fid)
    return entry["ask"] if entry else None


def is_clinician_only(fid):
    """True for findings phrased for the clinician — examination and
    observation items. These must never be spoken aloud to the patient.

    Marked by convention in findings.yaml's `ask` rather than by a separate
    field: the parenthetical prefix is already there, is visible to whoever
    edits the file, and cannot drift out of sync with the text it describes.
    """
    ask = question(fid) or ""
    return ask.startswith("(Examination)") or ask.startswith("(Observation)")


def catalog_for_prompt():
    """Compact view for an LLM prompt: id + label + the question it answers.

    `system` is deliberately included — it groups related findings so the
    model reads "these are the respiratory ones" rather than a flat list of
    two hundred strings.
    """
    return [
        {"id": fid, "label": m["label"], "asks": m["ask"], "system": m["system"]}
        for fid, m in FINDINGS.items()
    ]
