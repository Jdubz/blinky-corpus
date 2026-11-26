#!/usr/bin/env python3
"""Suspect bars: where the automatic drum labels of a track are most likely wrong (#4).

Writes edm/drums_suspect/<stem>.json, a review-priority map for the drum reviewer. It is not
ground truth and the scorer never reads it.

Mix-level ADTOF fails in sections: wherever a filter is on the drums it drops snares (it hears
them as hi-hats) or adds false hits. EDM is grid-locked and repetitive, so a bar can be checked
against the bars around it:

1. Grid. One constant tempo per track, fitted to the mix-ADTOF kick and snare times. Candidate
   beat periods are the local maxima of the events' vector strength at the beat rate
   (60-200 BPM) plus the --tempo-hint multiples. Each is refined by least squares of the event
   times on their nearest 16th-note index (a metronome fit). The winner has the largest share of
   events within 12 ms of the 16th grid, less that share's chance rate. It is then moved by
   octaves into 100-200 BPM, so a bar's 16 slots are real 16ths. Bars start at the phase with
   the most kicks on slot 0 and snares on 4 and 12.
2. Per bar and slot:
   - which classes the mix-ADTOF and stem-ADTOF labels put there (nearest slot, within 0.4 slot);
   - the strongest attack in the kick and snare bands (30-150 Hz, 150 Hz-5 kHz) of the mix or
     the drum stem. The high band is left out because hats put an attack on every 16th.
3. Confident bars: mix and stem labels agree (at most 1 slot differs), no filter flag, at least
   one label. A slot's expected rate P_c is the share of confident bars within 16 bars that have
   class c there. With fewer than 4 such bars, all confident bars in the track are used. This is
   the "periodic N-gram" idea: events one pattern length apart predict each other (Paulus;
   Ishizuka et al. 2020's repetition-aware bigram).
4. Per bar, the evidence:
   - disagree: slots where the mix and stem labels differ;
   - missing_c: slots with P_c >= 0.5, no mix label, and an attack at least as strong as the
     10th percentile of labelled attacks in confident bars;
   - unexpected_c: slots with P_c <= 0.1 that carry a mix label;
   - filter: the drum stem's per-bar log spectral centroid or 85% rolloff, level or slope, more
     than 3 robust SDs from the track's median.

5. Per bar, `expected`: each missing_c slot as a hit the reviewer can offer, {"time", "type", "p"
   (P_c), "attack" (the slot's attack strength), "at"}. The time is the strongest 150 Hz-5 kHz
   onset-strength peak of the mix or the drum stem within +-min(20 ms, 0.4 slot) of the slot, when
   that peak is a local maximum at least as strong as the attack threshold ("at": "attack"),
   otherwise the grid time ("at": "grid"). That band is the transient both drums share and the one
   the reviewer snaps to. Hints only: never ground truth, never published as labels (schema 2).

score = disagree + sum(P over missing slots) + 0.05 * unexpected + 0.05 * filter.
On the one reviewed track, disagreement and the pattern's missing hits each localised the
owner's corrections about equally well, and they come from independent evidence. The
unexpected-hit and filter evidence was weak, so those two only break ties (see PROVENANCE.json).

Bars are ranked by score. `tier` is "high" for the top-scoring bars covering 20% of the track,
"medium" for the next 20%, and "low" for any other bar with a non-zero score. Ranges merge
consecutive bars of the same tier.

    python scripts/drum_suspect.py --features DIR [--edm edm] [--out edm/drums_suspect]

DIR holds <stem>.npz from scripts/drum_suspect_features.py (run on the Spark as a gpu_task), which
records the envelopes' hop as `os_hop_s`.
Needs numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SLOTS = 16
W_BARS = 16
GRID_TOL_S = 0.012
FILTER_KEYS = ("centroid_drums", "rolloff_drums")
TIERS = (("high", 0.20), ("medium", 0.40))
CLASSES = ("kick", "snare")
SCHEMA = 2
# The onset-strength bands an expected hit's attack is looked for in: the 150 Hz-5 kHz transient of
# the mix and the drum stem, for both classes (a kick's 30-150 Hz energy peaks a few ms after its
# attack; the reviewer snaps to the same band).
PEAK_BANDS = (("mid", "mix"), ("mid", "drums"))


# --- grid ---------------------------------------------------------------------------------


def _refine(t: np.ndarray, T: float, iters: int = 4):
    t0 = 0.0
    for _ in range(iters):
        s = T / 4
        ph = np.angle(np.mean(np.exp(2j * np.pi * t / s))) / (2 * np.pi) * s
        k = np.round((t - ph) / s)
        ok = np.abs(t - (ph + k * s)) < 0.25 * s
        if ok.sum() < 8:
            return None
        A = np.vstack([np.ones(ok.sum()), k[ok]]).T
        (t0, s_new), *_ = np.linalg.lstsq(A, t[ok], rcond=None)
        T = 4 * s_new
    return T, t0


def _grid_score(t: np.ndarray, T: float, t0: float):
    s = T / 4
    res = t - (t0 + np.round((t - t0) / s) * s)
    on = float(np.mean(np.abs(res) <= GRID_TOL_S))
    return on - min(1.0, 2 * GRID_TOL_S / s), on, res


def fit_grid(times: list[float], tempo_hint: float | None = None) -> dict:
    t = np.asarray(sorted(times), dtype=float)
    if len(t) < 16:
        raise ValueError(f"only {len(t)} kick/snare labels: too few to fit a grid")
    Ts = 60.0 / np.arange(60, 200, 0.05)
    vs = np.array([abs(np.mean(np.exp(2j * np.pi * t / T))) for T in Ts])
    peaks = [i for i in range(1, len(vs) - 1) if vs[i] >= vs[i - 1] and vs[i] > vs[i + 1]]
    cands = [Ts[i] for i in sorted(peaks, key=lambda i: -vs[i])[:25]]
    if tempo_hint:
        cands += [60.0 / (tempo_hint * f) for f in (1, 2, 0.5, 1.5, 2 / 3, 4 / 3, 0.75)]
    best = None
    for T in cands:
        r = _refine(t, T)
        if r is None or not (60 / 200 <= r[0] <= 60 / 50):
            continue
        sc = _grid_score(t, *r)[0]
        if best is None or sc > best[0] + 1e-9:
            best = (sc, r[0], r[1])
    if best is None:
        raise ValueError("no tempo candidate fits")
    _, T, t0 = best
    while 60 / T < 100:
        T /= 2
    while 60 / T >= 200:
        T *= 2
    _, on, res = _grid_score(t, T, t0)
    return {"T": float(T), "t0": float(t0), "on16": on, "res_ms_median": float(np.median(np.abs(res)) * 1000)}


# --- analysis -----------------------------------------------------------------------------


def _occupancy(times, origin, s, nbars, first_bar):
    occ = np.zeros((nbars, SLOTS), bool)
    for t in times:
        k = (t - origin) / s
        kr = int(round(k))
        if abs(k - kr) > 0.4:
            continue
        b, sl = divmod(kr - first_bar * SLOTS, SLOTS)
        if 0 <= b < nbars:
            occ[b, sl] = True
    return occ


def _attack_peak(x: np.ndarray, tc: float, w: float, hop: float, thr: float) -> float | None:
    """The time of the strongest onset-strength peak of `x` within +-w of tc, or None when there's
    no local maximum there at least `thr` strong (the window's edge rising into a neighbour's attack
    doesn't count). Frame i covers [i, i+1) * hop, so its peak is at (i + 0.5) * hop, as the
    reviewer's snapping puts it, refined by a parabola through the peak and its two neighbours."""
    i0, i1 = max(1, int(np.ceil((tc - w) / hop - 0.5))), min(len(x) - 2, int(np.floor((tc + w) / hop - 0.5)))
    if i1 < i0:
        return None
    seg = x[i0:i1 + 1].astype(np.float64)
    i = int(np.argmax(seg)) + i0
    a, b, c = float(x[i - 1]), float(x[i]), float(x[i + 1])
    if b < thr or not (b >= a and b > c):
        return None
    den = a - 2 * b + c
    off = min(0.5, max(-0.5, 0.5 * (a - c) / den)) if den < 0 else 0.0
    return (i + 0.5 + off) * hop


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 1e-9 else 1e-9)


def analyse(mix: dict, stem: dict, feats, duration: float, tempo_hint: float | None = None):
    g = fit_grid(mix["kick"] + mix["snare"], tempo_hint)
    s, t0 = g["T"] / 4, g["t0"]
    k_first = int(np.floor(-t0 / s))
    k_last = int(np.ceil((duration - t0) / s))
    best = None
    for ph in range(4):  # downbeat: the bar phase with the most kicks on 1 and snares on 2 and 4
        fb = int(np.floor((k_first - 4 * ph) / SLOTS))
        nb = (k_last - (fb * SLOTS + 4 * ph)) // SLOTS + 1
        kk = _occupancy(mix["kick"], t0 + 4 * ph * s, s, nb, fb)
        sn = _occupancy(mix["snare"], t0 + 4 * ph * s, s, nb, fb)
        sc = int(kk[:, 0].sum() + sn[:, 4].sum() + sn[:, 12].sum())
        if best is None or sc > best[0]:
            best = (sc, ph, fb, nb)
    _, ph, fb, nb = best
    origin = t0 + 4 * ph * s

    def bar_start(b):
        return origin + (fb + b) * SLOTS * s

    def slot_time(b, sl):
        return origin + ((fb + b) * SLOTS + sl) * s

    lab = {src: {c: _occupancy(d[c], origin, s, nb, fb) for c in CLASSES} for src, d in (("mix", mix), ("stem", stem))}

    # Attacks: the strongest kick/snare-band onset strength within +-min(20 ms, 0.4 slot).
    bands = {}
    for band in ("low", "mid"):
        for src in ("mix", "drums"):
            x = np.asarray(feats[f"os_{band}_{src}"], dtype=np.float32)
            bands[(band, src)] = x / (np.percentile(x, 99.5) or 1.0)
    n = min(len(x) for x in bands.values())
    att_all = np.max(np.vstack([x[:n] for x in bands.values()]), axis=0)
    # Where an expected hit's attack is: the transient band of the mix and the stem (PEAK_BANDS).
    att_peak = np.max(np.vstack([bands[k][:n] for k in PEAK_BANDS]), axis=0)
    hop = float(feats["os_hop_s"][0])  # 44 samples at 44.1 kHz: 0.9977 ms, not 1 ms
    w = min(0.020, 0.4 * s)
    att = np.zeros((nb, SLOTS))
    for b in range(nb):
        for sl in range(SLOTS):
            tc = slot_time(b, sl)
            i0, i1 = max(0, int((tc - w) / hop)), min(n, int((tc + w) / hop) + 1)
            if i1 > i0:
                att[b, sl] = att_all[i0:i1].max()

    # Filter activity: per-bar median log2 centroid / rolloff of the drum stem (10 ms frames).
    def per_bar(x, hop=0.01):
        out = np.full(nb, np.nan)
        for b in range(nb):
            seg = x[max(0, int(bar_start(b) / hop)):max(0, int(bar_start(b + 1) / hop))]
            seg = seg[seg > 0]
            if len(seg):
                out[b] = np.median(np.log2(seg))
        return out

    fz = np.zeros(nb)
    for key in FILTER_KEYS:
        lv = per_bar(np.asarray(feats[key]))
        if np.all(np.isnan(lv)):
            continue
        slope = np.abs(np.gradient(np.nan_to_num(lv, nan=np.nanmedian(lv))))
        fz = np.maximum(fz, np.nan_to_num(np.maximum(np.abs(_robust_z(lv)), _robust_z(slope))))
    filt = fz > 3

    union = {c: lab["mix"][c] | lab["stem"][c] for c in CLASSES}
    dis = sum((lab["mix"][c] ^ lab["stem"][c]).sum(1) for c in CLASSES)
    has = sum(lab["mix"][c].sum(1) for c in CLASSES) > 0
    conf = has & (dis <= 1) & ~filt
    labelled = [att[conf][union[c][conf]] for c in CLASSES] if conf.any() else []
    labelled = np.concatenate(labelled) if len(labelled) else np.array([])
    att_thr = float(np.percentile(labelled, 10)) if len(labelled) >= 20 else float("inf")
    gpat = {c: union[c][conf].mean(0) if conf.sum() >= 4 else np.zeros(SLOTS) for c in CLASSES}
    bars = []
    for b in range(nb):
        lo, hi = max(0, b - W_BARS), min(nb, b + W_BARS + 1)
        cw = np.where(conf[lo:hi])[0] + lo
        row = {"bar": b, "start": round(max(0.0, bar_start(b)), 4), "end": min(duration, round(bar_start(b + 1), 4))}
        miss_w = 0.0
        ext_n = 0
        expected: list[dict] = []
        for c in CLASSES:
            P = union[c][cw].mean(0) if len(cw) >= 4 else gpat[c]
            miss = (P >= 0.5) & ~lab["mix"][c][b] & (att[b] >= att_thr)
            extra = (P <= 0.1) & lab["mix"][c][b]
            for sl in np.where(miss)[0]:
                tg = slot_time(b, sl)
                tp = _attack_peak(att_peak, tg, w, hop, att_thr)
                expected.append({"time": round(tp if tp is not None else tg, 4), "type": c,
                                 "p": round(float(P[sl]), 2), "attack": round(float(att[b, sl]), 3),
                                 "at": "attack" if tp is not None else "grid"})
            row[f"missing_{c}"] = int(miss.sum())
            row[f"unexpected_{c}"] = int(extra.sum())
            miss_w += float(P[miss].sum())
            ext_n += int(extra.sum())
        row["disagree"] = int(dis[b])
        row["filter"] = bool(filt[b])
        row["filter_z"] = round(float(fz[b]), 2)
        row["score"] = round(row["disagree"] + miss_w + 0.05 * ext_n + (0.05 if row["filter"] else 0.0), 3)
        row["expected"] = sorted((e for e in expected if 0 <= e["time"] <= duration), key=lambda e: e["time"])
        if row["end"] > row["start"]:
            bars.append(row)
    grid = {"bpm": round(60.0 / g["T"], 4), "beat_s": round(g["T"], 6), "origin_s": round(origin, 4),
            "slots_per_bar": SLOTS, "fit_within_12ms": round(g["on16"], 3),
            "fit_residual_ms_median": round(g["res_ms_median"], 2), "bars": len(bars),
            "confident_bars": int(conf.sum()), "attack_threshold": round(att_thr, 4)}
    return grid, bars


def tier_bars(bars: list[dict], duration: float) -> None:
    """Rank by score (ties: earlier first); top bars covering 20% of the track are "high",
    the next 20% "medium", any other bar with a score "low", the rest None."""
    order = sorted((b for b in bars if b["score"] > 0), key=lambda b: (-b["score"], b["start"]))
    acc = 0.0
    for rank, b in enumerate(order, 1):
        b["rank"] = rank
        b["tier"] = next((name for name, share in TIERS if acc < share * duration), "low")
        acc += b["end"] - b["start"]
    for b in bars:
        b.setdefault("rank", None)
        b.setdefault("tier", None)


def ranges_from(bars: list[dict]) -> list[dict]:
    out: list[dict] = []
    for b in bars:
        if b["tier"] is None:
            continue
        if out and out[-1]["tier"] == b["tier"] and abs(out[-1]["end"] - b["start"]) < 1e-3:
            o = out[-1]
            o["end"] = b["end"]
            o["bars"] += 1
            o["score"] = round(o["score"] + b["score"], 3)
            for k in ("disagree", "missing_kick", "missing_snare", "unexpected_kick", "unexpected_snare"):
                o[k] += b[k]
            o["filter"] = o["filter"] or b["filter"]
        else:
            out.append({k: b[k] for k in ("start", "end", "tier", "score", "disagree", "missing_kick",
                                          "missing_snare", "unexpected_kick", "unexpected_snare", "filter")}
                       | {"bars": 1})
    return out


# --- files --------------------------------------------------------------------------------


def load_labels(path: Path) -> dict[str, list[float]]:
    doc = json.loads(path.read_text())
    if doc.get("skipped"):
        raise ValueError(f"{path}: marked skipped")
    out = {c: sorted(float(o["time"]) for o in doc["onsets"] if o.get("type") == c) for c in CLASSES}
    if not out["kick"] and not out["snare"]:
        raise ValueError(f"{path}: no kicks or snares")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--edm", type=Path, default=Path(__file__).resolve().parents[1] / "edm")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.edm / "drums_suspect"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.edm / "track_manifest.json").read_text())
    stems = sorted(p.stem for p in a.edm.glob("*.mp3"))
    for stem in stems:
        mix = load_labels(a.edm / "drums_adtof" / f"{stem}.kick_weighted.json")
        st = load_labels(a.edm / "drums_adtof_stem" / f"{stem}.kick_weighted.json")
        feats = np.load(a.features / f"{stem}.npz")
        duration = float(manifest[stem]["duration"])
        grid, bars = analyse(mix, st, feats, duration, float(feats["tempo_librosa"][0]))
        tier_bars(bars, duration)
        doc = {"schema": SCHEMA, "track": stem, "duration": duration,
               "purpose": "review priority only: where mix-level ADTOF labels are most likely wrong. Not ground truth.",
               "inputs": {"mix": "edm/drums_adtof", "stem": "edm/drums_adtof_stem"},
               "grid": grid, "ranges": ranges_from(bars), "bars": bars}
        (out / f"{stem}.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        hi = sum(r["end"] - r["start"] for r in doc["ranges"] if r["tier"] == "high")
        print(f"{stem}: {grid['bpm']:.2f} BPM, {grid['bars']} bars, high-tier {100 * hi / duration:.0f}% of the track")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
