"""repS (blinky_scoring/repetition.py) and the repetition maps (edm/repetition/, scripts/repetition.py).

The golden files (tests/golden/repetition/) are blinky-firmware consistency.py's own output
(scripts/repetition_golden.py). The scorer and map-contract tests are standard library; the
find_repetition one needs numpy and ffmpeg (CI has both) and skips without them. CI's
"Repetition maps are current" step checks every map against scripts/repetition.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EDM = ROOT / "edm"
GOLDEN = sorted((ROOT / "tests" / "golden" / "repetition").glob("*.json"))
sys.path.insert(0, str(ROOT))
from blinky_scoring.repetition import (  # noqa: E402
    MIN_SCORED,
    Repetition,
    consistency,
    load_repetition,
    regions_of,
    repetition_path,
)


def _strict(text: str, name: str):
    def bad(c):
        raise ValueError(f"{name}: {c} is not JSON")
    return json.loads(text, parse_constant=bad)


def _rep(d: dict) -> Repetition:
    return Repetition(d["lag_s"], d["frame_s"], d["n_frames"], tuple(tuple(r) for r in d["regions"]))


# -- the scorer, against consistency.py's own results -----------------------------------------

def test_there_are_golden_files():
    assert len(GOLDEN) >= 2


@pytest.mark.parametrize("f", GOLDEN, ids=[f.stem for f in GOLDEN])
def test_the_scorer_matches_consistency_py(f):
    g = json.loads(f.read_text())
    rep = _rep(g)
    for case in g["cases"]:
        got = consistency(case["times"], case["strengths"], rep, case["t_start"], case["t_end"])
        want = case["result"]
        assert got["n"] == want["n"], case["name"]
        for k in ("rep", "rep_s", "coverage", "lag_s"):
            if want[k] is None:
                assert got[k] is None, (case["name"], k)
            else:
                assert got[k] == pytest.approx(want[k], abs=1e-9), (case["name"], k)
        assert want["n"] >= MIN_SCORED, f"{case['name']}: a golden case should score"


def test_too_few_detections_score_none():
    rep = Repetition(2.0, 0.008, 3000, ((0, 3000),))
    out = consistency([3.0, 5.0, 7.0], [1.0, 1.0, 1.0], rep, 0.0, 24.0)
    assert out["rep"] is None and out["rep_s"] is None and out["n"] < MIN_SCORED


def test_the_same_hits_every_repeat_score_one_and_strength_changes_cost():
    # A 2 s loop played 12 times: every frame repeats one lag later, but the last lag's.
    rep = Repetition(2.0, 0.008, 3000, ((0, 2750),))
    bar = [0.0, 0.5, 0.75, 1.0, 1.5]
    times = [n * 2.0 + h for n in range(12) for h in bar]
    out = consistency(times, [1.0] * len(times), rep, 2.0, 24.0)
    assert out["rep"] == pytest.approx(1.0) and out["rep_s"] == pytest.approx(1.0)
    alternate = [0.5 if (k // 5) % 2 else 1.0 for k in range(len(times))]
    soft = consistency(times, alternate, rep, 2.0, 24.0)
    assert soft["rep"] == pytest.approx(1.0) and soft["rep_s"] < 0.7


def test_regions_round_trip():
    mask = [False, True, True, False, True]
    assert regions_of(mask) == [[1, 3], [4, 5]]
    assert Repetition(1.0, 0.1, 5, ((1, 3), (4, 5))).mask() == mask


def test_a_malformed_map_raises(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema": 1, "lag_s": 2.0, "frame_s": 0.008, "n_frames": 10,
                             "regions": [[5, 3]]}))
    with pytest.raises(ValueError):
        load_repetition(p)


# -- the maps -------------------------------------------------------------------------------

def _maps():
    return sorted((EDM / "repetition").glob("*.json"))


def test_every_track_has_a_map():
    assert [f.stem for f in _maps()] == sorted(p.stem for p in EDM.glob("*.mp3"))


@pytest.mark.parametrize("f", _maps(), ids=[f.stem for f in _maps()])
def test_corpus_map_is_well_formed_and_of_this_audio(f):
    d = _strict(f.read_text(), f.name)  # no NaN
    assert d["track"] == f.stem
    rep = load_repetition(repetition_path(EDM, f.stem))  # raises on a malformed map
    assert 1.5 <= rep.lag_s <= 8.0 and rep.frame_s == 0.008
    manifest = json.loads((EDM / "track_manifest.json").read_text())
    assert rep.n_frames * rep.frame_s == pytest.approx(manifest[f.stem]["duration"], abs=0.2)
    sha = hashlib.sha256((EDM / f"{f.stem}.mp3").read_bytes()).hexdigest()
    assert d["audio_sha256"] == sha, "the audio changed: regenerate the map (scripts/repetition.py)"


# -- find_repetition (numpy + ffmpeg) ---------------------------------------------------------

@pytest.mark.parametrize("f", GOLDEN, ids=[f.stem for f in GOLDEN])
def test_find_repetition_matches_consistency_py(f):
    pytest.importorskip("numpy")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    sys.path.insert(0, str(ROOT / "scripts"))
    import repetition  # scripts/repetition.py

    g = json.loads(f.read_text())
    mp3 = EDM / f"{g['track']}.mp3"
    if hashlib.sha256(repetition.decode(mp3)).hexdigest() != g["pcm_sha256"]:
        # Another ffmpeg decodes slightly differently; the comparison still holds within its
        # tolerance unless the decode moved a lot, which then fails below.
        print(f"note: {g['track']} decodes to other PCM than the golden's")
    assert repetition.differences(g, repetition.build_map(mp3)) == []
