#!/usr/bin/env python3
"""Repetition maps: each track's repeat lag and repeated regions, for repS (blinky_scoring/repetition.py).

Writes edm/repetition/<stem>.json. The regions are a property of the audio alone, so they are
computed once per track here (numpy + ffmpeg) and checked in; a scorer only matches detections
against them, with the standard library.

``find_repetition`` is blinky-firmware's ``tests/tracker/consistency.py`` (at 7ffed1b), copied
line for line: log-mel frames of the clean decoded file, the repeat lag that stands highest above
the lag curve's local median, and the frames whose +-0.25 s patch correlates with the patch one lag
later above the track's own off-grid null. The audio is decoded as that repo's
``tests/tracker/run_tracker_eval.py`` decodes it (ffmpeg to 16 kHz mono s16le, / 32768).
tests/test_repetition.py holds this copy to consistency.py's own output on two tracks
(tests/golden/repetition/), and the checked-in maps to this script.

    python scripts/repetition.py [--edm edm] [--out edm/repetition]   # write every map
    python scripts/repetition.py --check [--out DIR]                    # compare, don't write

--check regenerates every map and compares it with the checked-in one (the lag to 1 us, the
regions to MAX_MASK_DIFF of the frames: BLAS sums differ in the last bits between CPUs); with
--out it also writes the regenerated maps there. Needs numpy and ffmpeg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blinky_scoring.repetition import SCHEMA, regions_of  # noqa: E402

SOURCE = "blinky-firmware tests/tracker/consistency.py find_repetition at 7ffed1b (scripts/repetition.py)"
MAX_MASK_DIFF = 0.001  # share of frames two runs of find_repetition may disagree on
LAG_TOL_S = 1e-6

# --- consistency.py, verbatim ---------------------------------------------------------------

SR = 16000
HOP = 128  # 8 ms, half the firmware's frame
WIN = 1024
N_MELS = 40
FMIN, FMAX = 30.0, 8000.0
LAG_MIN_S, LAG_MAX_S = 1.5, 8.0  # a bar at 160 BPM up to two bars at 60
PATCH_S = 0.25
NULL_QUANTILE = 0.99
NULL_LAGS = 64
MIN_BAR_S = 1.3  # a 4-beat bar at 185 BPM, faster than anything in edm/
NULL_OFF_GRID = 0.35  # of a 16th-note step: null lags sit between grid lines
BASELINE_S = 0.5  # local-median window of the lag curve


def _mel_filterbank() -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    edges = 700.0 * (10 ** (np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2) / 2595.0) - 1.0)
    bins = np.fft.rfftfreq(WIN, 1.0 / SR)
    fb = np.zeros((N_MELS, len(bins)))
    for m in range(N_MELS):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0.0, None)
    assert (fb.sum(axis=1) > 0).all(), "empty mel band"
    return fb


def log_mel(x: np.ndarray) -> np.ndarray:
    """Log-mel frames (n_frames, N_MELS), from float samples."""
    if len(x) < WIN:
        raise ValueError(f"audio too short for one frame ({len(x)} samples)")
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(WIN), axis=1))
    return np.log1p(1000.0 * (mag @ _mel_filterbank().T))


def _window_sums(v: np.ndarray, half: int) -> np.ndarray:
    """Sum over frames [i-half, i+half] (clipped at the ends), for each frame i."""
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(len(v))
    return c[np.minimum(i + half + 1, len(v))] - c[np.maximum(i - half, 0)]


def _patch_corr(f: np.ndarray, lag: int, half: int) -> np.ndarray:
    """Pearson r between the patch at frame i and the patch at i + lag, for i < n - lag."""
    a, b = f[:-lag], f[lag:]
    k = _window_sums(np.ones(len(a)), half) * f.shape[1]
    sa, sb = _window_sums(a.sum(1), half), _window_sums(b.sum(1), half)
    saa, sbb = _window_sums((a * a).sum(1), half), _window_sums((b * b).sum(1), half)
    sab = _window_sums((a * b).sum(1), half)
    cov = sab - sa * sb / k
    var = (saa - sa * sa / k) * (sbb - sb * sb / k)
    # A flat patch (digital silence) has no variance: it repeats nothing, so r = 0.
    return np.where(var > 1e-12, cov / np.sqrt(np.maximum(var, 1e-12)), 0.0)


def find_repetition(x: np.ndarray, seed: int = 0) -> tuple[float, float, np.ndarray]:
    """(lag_s, frame_s, mask) of clean float audio at 16 kHz (consistency.py returns them as a
    Repetition)."""
    f = log_mel(x)
    f = f - f.mean(axis=0)
    frame_s = HOP / SR
    lo, hi = int(LAG_MIN_S / frame_s), int(LAG_MAX_S / frame_s)
    if len(f) < 2 * hi:
        raise ValueError(f"audio too short to find a repeat ({len(f) * frame_s:.1f} s)")
    norm = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    sim = np.array([(norm[:-lag] * norm[lag:]).sum(1).mean() for lag in range(lo, hi + 1)])
    peaks = np.flatnonzero((sim[1:-1] > sim[:-2]) & (sim[1:-1] >= sim[2:])) + 1
    if not len(peaks):
        raise ValueError("no repeat: frame similarity has no peak inside the lag range")
    w = int(round(BASELINE_S / frame_s))
    base = np.array([np.median(sim[max(0, i - w):i + w + 1]) for i in range(len(sim))])
    k = int(peaks[np.argmax((sim - base)[peaks])])
    y0, y1, y2 = sim[k - 1], sim[k], sim[k + 1]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    lag = lo + k
    half = int(round(PATCH_S / frame_s))
    r = _patch_corr(f, lag, half)
    rng = np.random.default_rng(seed)
    bars = 1
    while (lag + frac) * frame_s / (2 * bars) >= MIN_BAR_S:
        bars *= 2
    grid = (lag + frac) / (16 * bars)  # a 16th note, in frames
    cands = [L for L in range(lo, hi + 1) if abs(L / grid - round(L / grid)) > NULL_OFF_GRID]
    if not cands:
        raise ValueError("no off-grid lag for the null")
    null = np.concatenate([_patch_corr(f, int(L), half)[::half]
                           for L in rng.choice(cands, min(NULL_LAGS, len(cands)), replace=False)])
    thr = float(np.quantile(null, NULL_QUANTILE))
    mask = np.zeros(len(f), dtype=bool)
    mask[:len(r)] = r > thr
    return (lag + frac) * frame_s, frame_s, mask

# --- end of consistency.py ------------------------------------------------------------------


def decode(mp3: Path) -> bytes:
    """16 kHz mono s16le, as run_tracker_eval.py's decode() writes it."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")
    return subprocess.run([ffmpeg, "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", "16000",
                           "-f", "s16le", "-"], check=True, capture_output=True).stdout


def samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0


def build_map(mp3: Path) -> dict:
    lag_s, frame_s, mask = find_repetition(samples(decode(mp3)))
    return {
        "schema": SCHEMA,
        "track": mp3.stem,
        "audio_sha256": hashlib.sha256(mp3.read_bytes()).hexdigest(),
        "source": SOURCE,
        "lag_s": float(lag_s),
        "frame_s": float(frame_s),
        "n_frames": int(len(mask)),
        "regions": regions_of(mask.tolist()),
    }


def mask_of(m: dict) -> np.ndarray:
    out = np.zeros(m["n_frames"], dtype=bool)
    for a, b in m["regions"]:
        out[a:b] = True
    return out


def differences(want: dict, got: dict) -> list[str]:
    """Why two maps of one track aren't the same (empty: they are)."""
    out = []
    if want.get("audio_sha256") != got.get("audio_sha256"):
        out.append("audio_sha256 differs")
    if abs(want["lag_s"] - got["lag_s"]) > LAG_TOL_S:
        out.append(f"lag {got['lag_s']:.6f} s, expected {want['lag_s']:.6f} s")
    if want["n_frames"] != got["n_frames"]:
        out.append(f"{got['n_frames']} frames, expected {want['n_frames']}")
    else:
        diff = float((mask_of(want) != mask_of(got)).mean())
        if diff > MAX_MASK_DIFF:
            out.append(f"regions differ on {diff:.2%} of the frames (allowed {MAX_MASK_DIFF:.1%})")
    return out


def dump(m: dict) -> str:
    """The map as JSON, one region per line."""
    head = json.dumps({k: v for k, v in m.items() if k != "regions"}, indent=1, allow_nan=False)
    regions = ",\n  ".join(json.dumps(r) for r in m["regions"])
    return head[:-2] + ',\n "regions": [' + (f"\n  {regions}\n " if regions else "") + "]\n}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edm", type=Path, default=ROOT / "edm")
    ap.add_argument("--out", type=Path, help="where to write (default edm/repetition; with --check, "
                                             "where to put the regenerated maps)")
    ap.add_argument("--check", action="store_true", help="compare with the checked-in maps, write nothing there")
    args = ap.parse_args()
    have = args.edm / "repetition"
    out = args.out if args.out or args.check else have
    if out:
        out.mkdir(parents=True, exist_ok=True)
    bad = 0
    mp3s = sorted(args.edm.glob("*.mp3"))
    if not mp3s:
        raise SystemExit(f"no tracks in {args.edm}")
    for mp3 in mp3s:
        m = build_map(mp3)
        cov = float(mask_of(m).mean())
        print(f"{mp3.stem:32s} lag {m['lag_s']:.3f} s  repeated {cov:.0%}", flush=True)
        if out:
            (out / f"{mp3.stem}.json").write_text(dump(m))
        if args.check:
            path = have / f"{mp3.stem}.json"
            why = differences(json.loads(path.read_text()), m) if path.exists() else ["no map"]
            for w in why:
                print(f"::error file={path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}::"
                      f"{mp3.stem}: {w}")
            bad += bool(why)
    if args.check:
        extra = sorted({p.stem for p in have.glob("*.json")} - {p.stem for p in mp3s})
        for s in extra:
            print(f"::error::{s}: a map without a track")
        bad += len(extra)
        if bad:
            print(f"{bad} map(s) out of date: run `python scripts/repetition.py` (numpy + ffmpeg) and "
                  "commit edm/repetition/", file=sys.stderr)
            return 1
        print(f"{len(mp3s)} maps current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
