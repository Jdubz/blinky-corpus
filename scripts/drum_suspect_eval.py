#!/usr/bin/env python3
"""How well do the suspect tiers and the candidate label sets match the human corrections?

For every track with a human file (edm/drums_human/<stem>.drums.json, or its autosaved draft in
drums_human/.drafts/, which is newer), compares mix-level ADTOF with the human labels:

- With reviewed ranges: the corrections are ADTOF's misses AND its false hits (+-50 ms, per
  class) inside them. Each candidate set (ADTOF, ADTOF on the drum stem, their union) gets
  per-class P/R/F at +-50 and +-20 ms, one-to-one matching (mir_eval convention), over the
  reviewed ranges only.
- Without reviewed ranges: the only known corrections are the hits the human added, so it
  reports how many of those each set finds (a recall-only view).

It also reports what share of the corrections each suspect tier (edm/drums_suspect/) covers,
against that tier's share of the track. Read-only; standard library only.

    python scripts/drum_suspect_eval.py [--edm edm] [--human DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CLASSES = ("kick", "snare")


def match(est, ref, tol):
    """Maximum one-to-one matching within +-tol (greedy in time order is optimal on a line)."""
    est, pairs, j = sorted(est), [], 0
    for r in sorted(ref):
        while j < len(est) and est[j] < r - tol:
            j += 1
        if j < len(est) and est[j] <= r + tol:
            pairs.append((r, est[j]))
            j += 1
    return pairs


def labels(path: Path) -> dict[str, list[float]]:
    doc = json.loads(path.read_text())
    return {c: sorted(float(o["time"]) for o in doc["onsets"] if o.get("type") == c) for c in CLASSES}


def union(a, b, w=0.030):
    return {c: sorted(a[c] + [t for t in b[c] if not any(abs(t - u) <= w for u in a[c])]) for c in CLASSES}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edm", type=Path, default=Path(__file__).resolve().parents[1] / "edm")
    ap.add_argument("--human", type=Path, default=None, help="default: EDM/drums_human")
    a = ap.parse_args()
    human = a.human or a.edm / "drums_human"
    docs: dict[str, tuple[Path, dict]] = {}
    for f in sorted(human.glob("*.drums.json")) + sorted((human / ".drafts").glob("*.drums.json")):
        d = json.loads(f.read_text())
        docs[d["track"]] = (f, d)  # the draft, read second, is the newer state
    if not docs:
        raise SystemExit(f"no human files under {human}")
    for stem, (f, d) in sorted(docs.items()):
        hum = {c: sorted(float(o["time"]) for o in d["onsets"] if o["type"] == c) for c in CLASSES}
        rev = d.get("reviewed_ranges") or []

        def inr(t):
            return any(x <= t <= y for x, y in rev)

        mix = labels(a.edm / "drums_adtof" / f"{stem}.kick_weighted.json")
        sets = {"ADTOF (mix)": mix}
        sp = a.edm / "drums_adtof_stem" / f"{stem}.kick_weighted.json"
        if sp.is_file():
            stem_l = labels(sp)
            sets["ADTOF on drum stem"] = stem_l
            sets["ADTOF mix + stem (union)"] = union(mix, stem_l)
        reviewed = sum(y - x for x, y in rev)
        print(f"{stem} ({f.relative_to(human.parent) if human.parent in f.parents else f}): "
              + (f"{reviewed:.1f} s reviewed" if rev else "NO reviewed ranges: only the added hits are known"))
        if rev:
            ref = {c: [t for t in hum[c] if inr(t)] for c in CLASSES}
            print(f"  ground truth in reviewed ranges: {len(ref['kick'])} kicks, {len(ref['snare'])} snares")
            for name, s in sets.items():
                cells = []
                for c in CLASSES:
                    est = [t for t in s[c] if inr(t)]
                    for tol in (0.050, 0.020):
                        tp = len(match(est, ref[c], tol))
                        p = tp / len(est) if est else float("nan")
                        r = tp / len(ref[c]) if ref[c] else float("nan")
                        fm = 2 * p * r / (p + r) if est and ref[c] and (p + r) > 0 else 0.0
                        cells.append(f"{c} F{int(tol * 1000)} {fm:.3f} (P {p:.2f} R {r:.2f})")
                    err = [abs(e - r_) * 1000 for r_, e in match(est, ref[c], 0.050)]
                    cells.append(f"{c} med|err| {statistics.median(err):.1f} ms" if err else f"{c} med|err| -")
                print(f"  {name:26s} " + " | ".join(cells))
        else:
            added = {c: [t for t in hum[c] if not any(abs(t - u) < 5e-5 for u in mix[c])] for c in CLASSES}
            print(f"  added by hand: {len(added['kick'])} kicks, {len(added['snare'])} snares")
            for name, s in sets.items():
                print(f"  {name:26s} " + " | ".join(
                    f"{c} {len(match(s[c], added[c], 0.050))}/{len(added[c])} (+-20 ms: {len(match(s[c], added[c], 0.020))})"
                    for c in CLASSES))
        # corrections the suspect tiers should point at
        errs = []
        for c in CLASSES:
            ref = [t for t in hum[c] if not rev or inr(t)]
            est = [t for t in mix[c] if not rev or inr(t)]
            m = match(est, ref, 0.050)
            mr, me = {r for r, _ in m}, {e for _, e in m}
            errs += [t for t in ref if t not in mr]
            if rev:
                errs += [t for t in est if t not in me]
        sus_p = a.edm / "drums_suspect" / f"{stem}.json"
        if not sus_p.is_file() or not errs:
            continue
        sus = json.loads(sus_p.read_text())

        def tier(t):
            return next((b["tier"] for b in sus["bars"] if b["start"] <= t < b["end"]), None)

        for name, tiers in (("high", {"high"}), ("high+medium", {"high", "medium"}), ("any tier", {"high", "medium", "low"})):
            n = sum(tier(t) in tiers for t in errs)
            share = sum(r["end"] - r["start"] for r in sus["ranges"] if r["tier"] in tiers) / sus["duration"]
            print(f"  suspect {name:12s} covers {n}/{len(errs)} corrections in {100 * share:.0f}% of the track")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
