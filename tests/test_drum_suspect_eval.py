"""scripts/drum_suspect_eval.py: loading the reviewer notes from the human files."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("drum_suspect_eval", ROOT / "scripts" / "drum_suspect_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def test_load_notes(tmp_path):
    h = tmp_path / "drums_human"
    (h / ".drafts").mkdir(parents=True)
    n1 = [{"text": "snares under the sweep all dropped", "range": [80.0, 105.0]}]
    (h / "a.drums.json").write_text(json.dumps({"track": "a", "onsets": [], "notes": n1}))
    (h / "b.drums.json").write_text(json.dumps({"track": "b", "onsets": []}))  # no notes key
    (h / "c.drums.json").write_text(json.dumps({"track": "c", "onsets": [], "notes": [{"text": "old", "range": None}]}))
    (h / ".drafts" / "c.drums.json").write_text(json.dumps({"track": "c", "onsets": [], "notes": [{"text": "new", "range": None}]}))
    assert ev.load_notes(h) == {"a": n1, "c": [{"text": "new", "range": None}]}
