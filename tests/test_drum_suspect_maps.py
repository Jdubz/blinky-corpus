"""The committed suspect maps (edm/drums_suspect/, scripts/drum_suspect.py). Standard library only,
so this runs wherever the corpus does: the labels service (blinky_time labels-service/server.py
`suspect_public`) reads these files, and this is the contract it reads."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EDM = ROOT / "edm"


def _strict(text: str, name: str):
    """Strict JSON (RFC 8259): NaN and Infinity, which json.dumps writes by default, are refused."""
    def bad(c):
        raise ValueError(f"{name}: {c} is not JSON")
    return json.loads(text, parse_constant=bad)


def _maps():
    return [f for f in sorted((EDM / "drums_suspect").glob("*.json")) if f.name != "PROVENANCE.json"]


def test_every_track_has_a_map():
    assert [f.stem for f in _maps()] == sorted(p.stem for p in EDM.glob("*.mp3"))


@pytest.mark.parametrize("f", _maps(), ids=lambda f: f.stem)
def test_corpus_map_is_well_formed(f):
    manifest = json.loads((EDM / "track_manifest.json").read_text())
    d = _strict(f.read_text(), f.name)
    assert d["schema"] == 2 and d["track"] == f.stem
    assert d["duration"] == pytest.approx(manifest[f.stem]["duration"])
    thr = d["grid"]["attack_threshold"]
    assert thr is None or (isinstance(thr, (int, float)) and math.isfinite(thr))
    for r in d["ranges"]:
        assert 0 <= r["start"] < r["end"] <= d["duration"] + 1e-6 and r["tier"] in ("high", "medium", "low")
    starts = [b["start"] for b in d["bars"]]
    assert starts == sorted(starts)
    for b in d["bars"]:
        assert 0 <= b["start"] < b["end"] <= d["duration"] + 1e-6
        assert b["tier"] in ("high", "medium", "low", None)
        assert (b["rank"] is None) == (b["tier"] is None)
        assert b["rank"] is None or (isinstance(b["rank"], int) and b["rank"] >= 1)
        for k in ("missing_kick", "missing_snare", "disagree", "unexpected_kick", "unexpected_snare"):
            assert isinstance(b[k], int) and not isinstance(b[k], bool) and b[k] >= 0
        assert isinstance(b["filter"], bool)
        ex = b["expected"]
        assert len(ex) == b["missing_kick"] + b["missing_snare"]
        assert [e["time"] for e in ex] == sorted(e["time"] for e in ex)
        for e in ex:
            assert e["type"] in ("kick", "snare") and e["at"] in ("attack", "grid")
            assert 0 <= e["time"] <= d["duration"] and 0.5 <= e["p"] <= 1.0
            # within the bar, give or take the attack window at its edges
            assert b["start"] - 0.025 <= e["time"] <= b["end"] + 0.025
