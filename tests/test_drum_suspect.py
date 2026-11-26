"""scripts/drum_suspect.py on synthetic grid-locked tracks (needs numpy; skipped without it)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("drum_suspect", ROOT / "scripts" / "drum_suspect.py")
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

BPM = 128.0
BEAT = 60.0 / BPM
BARS = 32
DUR = BARS * 4 * BEAT + 1.0


def _track(snare_gap=range(0, 0)):
    """Four-on-the-floor kicks, snares on 2 and 4; the mix loses the snares in `snare_gap` bars
    (a filtered section) while the stem keeps them."""
    kicks = [0.5 + i * BEAT for i in range(BARS * 4)]
    snares = [0.5 + (4 * b + q) * BEAT for b in range(BARS) for q in (1, 3)]
    mix_sn = [t for t in snares if int((t - 0.5) / (4 * BEAT)) not in snare_gap]
    mix = {"kick": kicks, "snare": mix_sn}
    stem = {"kick": kicks, "snare": snares}
    hop = 44 / 44100  # the features script's real hop: 0.9977 ms, so frame i is NOT i ms
    n = int(DUR / hop)
    feats = {f"os_{b}_{s}": np.zeros(n, np.float32) for b in ("low", "mid", "high") for s in ("mix", "drums")}
    for t in kicks + snares:
        for k in feats:
            feats[k][int(round(t / hop))] = 1.0
    feats["os_hop_s"] = np.array([hop])
    m = int(DUR * 100)
    for k in ("centroid_drums", "rolloff_drums", "centroid_mix", "rolloff_mix"):
        feats[k] = np.full(m, 2000.0, np.float32)  # no filter anywhere
    feats["tempo_librosa"] = np.array([BPM / 2], np.float32)
    return mix, stem, feats


def test_grid_recovers_tempo_and_phase():
    mix, _, _ = _track()
    g = ds.fit_grid(mix["kick"] + mix["snare"], tempo_hint=BPM / 2)
    assert g["T"] == pytest.approx(BEAT, rel=1e-4)
    assert g["on16"] == 1.0


def test_filtered_section_is_high_tier_and_clean_bars_are_not():
    mix, stem, feats = _track(snare_gap=range(10, 14))
    grid, bars = ds.analyse(mix, stem, feats, DUR, float(feats["tempo_librosa"][0]))
    ds.tier_bars(bars, DUR)
    assert grid["bpm"] == pytest.approx(BPM, rel=1e-3)
    bar_s = 4 * BEAT
    in_gap = lambda b: 0.5 + 10 * bar_s - 0.01 <= b["start"] < 0.5 + 14 * bar_s - 0.01
    gap = [b for b in bars if in_gap(b)]
    assert len(gap) == 4
    assert all(b["tier"] == "high" and b["disagree"] == 2 and b["missing_snare"] == 2 for b in gap)
    # late in the track too: a ms-indexed envelope would have drifted ~70 ms off its slots by here
    late = [b for b in bars if b["start"] > 50]
    assert late and grid["attack_threshold"] == 1.0
    assert all(b["score"] == 0 and b["tier"] is None for b in bars if not in_gap(b))
    rs = ds.ranges_from(bars)
    assert rs[0]["tier"] == "high" and rs[0]["bars"] == 4 and rs[0]["disagree"] == 8


def test_corpus_files_are_well_formed():
    edm = ROOT / "edm"
    manifest = json.loads((edm / "track_manifest.json").read_text())
    files = sorted((edm / "drums_suspect").glob("*.json"))
    stems = sorted(p.stem for p in edm.glob("*.mp3"))
    assert [f.stem for f in files if f.name != "PROVENANCE.json"] == stems
    for f in files:
        if f.name == "PROVENANCE.json":
            continue
        d = json.loads(f.read_text())
        assert d["schema"] == 1 and d["track"] == f.stem
        assert d["duration"] == pytest.approx(manifest[f.stem]["duration"])
        for r in d["ranges"]:
            assert 0 <= r["start"] < r["end"] <= d["duration"] + 1e-6 and r["tier"] in ("high", "medium", "low")
        starts = [b["start"] for b in d["bars"]]
        assert starts == sorted(starts)
