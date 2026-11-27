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
    # The two missing snares of each gap bar are offered at their attacks; no other bar offers any.
    snares = [0.5 + (4 * b + q) * BEAT for b in range(10, 14) for q in (1, 3)]
    ex = [e for b in gap for e in b["expected"]]
    assert [e["type"] for e in ex] == ["snare"] * 8 and all(e["at"] == "attack" for e in ex)
    assert all(abs(e["time"] - t) < 0.0015 for e, t in zip(ex, snares))
    assert all(e["p"] == 1.0 and e["attack"] >= grid["attack_threshold"] for e in ex)
    assert not any(b["expected"] for b in bars if not in_gap(b))


def test_expected_hit_without_a_transient_peak_sits_on_the_grid():
    """A missing snare whose only attack is in the kick band has no 150 Hz-5 kHz peak to sit on."""
    mix, stem, feats = _track(snare_gap=range(10, 11))
    hop = float(feats["os_hop_s"][0])
    gap = [0.5 + (4 * 10 + q) * BEAT for q in (1, 3)]
    for t in gap:
        for src in ("mix", "drums"):
            feats[f"os_mid_{src}"][int(round(t / hop))] = 0.0
    grid, bars = ds.analyse(mix, stem, feats, DUR, float(feats["tempo_librosa"][0]))
    ex = [e for b in bars for e in b["expected"]]
    assert [e["at"] for e in ex] == ["grid", "grid"]
    assert all(abs(e["time"] - t) < 0.002 for e, t in zip(ex, gap))  # the fitted grid's slot time


def test_attack_peak_needs_a_local_maximum_above_the_threshold():
    hop = 0.001
    x = np.zeros(200, np.float32)
    x[100] = 1.0
    assert ds._attack_peak(x, 0.1, 0.02, hop, 0.5) == pytest.approx(0.1005)
    assert ds._attack_peak(x, 0.1, 0.02, hop, 1.5) is None  # too weak
    x2 = np.linspace(0, 1, 200).astype(np.float32)  # rising into the window's edge: no peak
    assert ds._attack_peak(x2, 0.1, 0.02, hop, 0.1) is None


def test_too_few_confident_attacks_writes_no_threshold_not_infinity():
    """Under 20 labelled attacks in confident bars there's no attack threshold: the map says so
    with null (and no missing-hit evidence), never `Infinity`, which isn't JSON."""
    mix, stem, feats = _track()
    # The stem disagrees with the mix by two kicks in all but two bars: two confident bars.
    stem = dict(stem, kick=sorted(stem["kick"] + [0.5 + (4 * b + q + 0.5) * BEAT
                                                 for b in range(2, BARS) for q in (0, 2)]))
    grid, bars = ds.analyse(mix, stem, feats, DUR, float(feats["tempo_librosa"][0]))
    assert grid["confident_bars"] == 2 and grid["attack_threshold"] is None
    assert not any(b["expected"] or b["missing_kick"] or b["missing_snare"] for b in bars)
    ds.tier_bars(bars, DUR)
    json.dumps({"grid": grid, "bars": bars}, allow_nan=False)
