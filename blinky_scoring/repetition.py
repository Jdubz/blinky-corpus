"""Repetition consistency (repS): does repeated audio get repeated light? Label-free.

Where a track's audio at time t matches its audio one repeat later (t + L), a detector that
follows the music fires the same hits at both, with the same strengths. The reference is the
track's own audio, so no labels are involved.

The reference implementation is blinky-firmware's ``tests/tracker/consistency.py``. It has two
halves:

1. ``find_repetition``: the repeat lag L and the repeated regions, from log-mel frames of the
   clean audio (numpy). They depend only on the audio, so they are computed once per track by
   ``scripts/repetition.py`` and checked in as ``edm/repetition/<stem>.json`` (a "map").
2. ``consistency``: matching a run's detections against a map. That half is here, ported to the
   standard library, so blinky-server can score repS wherever it runs.

``tests/test_repetition.py`` holds both halves to consistency.py's own output on two tracks
(``tests/golden/repetition/``, written by consistency.py itself: scripts/repetition_golden.py).

  rep       2 * matched / (detections in the regions + detections one lag later).
  repS      Strength agreement: 1 - sum|s_a - s_b| / sum max(s_a, s_b), where an unmatched
            detection pairs with 0. 1 = the same hits at the same strengths.
  coverage  Share of the scored time that is a repeated region.
  n         Detections scored. Below MIN_SCORED, rep and repS are None (consistency.py: NaN):
            too few to say anything.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1
MATCH_TOL_S = 0.032  # two firmware frames: a hit that moves by one frame still counts
MIN_SCORED = 20  # fewer detections in repeated audio than this is no evidence either way


@dataclass(frozen=True)
class Repetition:
    """A track's repeat lag and repeated regions: frame i (at i * frame_s) repeats one lag
    later when some region [a, b) holds a <= i < b."""

    lag_s: float
    frame_s: float
    n_frames: int
    regions: tuple[tuple[int, int], ...]

    def mask(self) -> list[bool]:
        m = [False] * self.n_frames
        for a, b in self.regions:
            m[a:b] = [True] * (b - a)
        return m


def repetition_path(edm_dir: str | Path, stem: str) -> Path:
    return Path(edm_dir) / "repetition" / f"{stem}.json"


def regions_of(mask: Sequence[bool]) -> list[list[int]]:
    """The [a, b) frame ranges where ``mask`` is True."""
    out: list[list[int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i])
            start = None
    if start is not None:
        out.append([start, len(mask)])
    return out


def load_repetition(path: str | Path) -> Repetition:
    """A map (edm/repetition/<stem>.json). Raises on anything malformed: never a silent default."""
    d = json.loads(Path(path).read_text())
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA}")
    lag, frame, n = d["lag_s"], d["frame_s"], d["n_frames"]
    if not (isinstance(lag, (int, float)) and lag > 0 and isinstance(frame, (int, float))
            and frame > 0 and isinstance(n, int) and n > 0):
        raise ValueError(f"{path}: bad lag_s/frame_s/n_frames {lag!r}/{frame!r}/{n!r}")
    regions = tuple((int(a), int(b)) for a, b in d["regions"])
    prev = 0
    for a, b in regions:
        if not prev <= a < b <= n:
            raise ValueError(f"{path}: region [{a}, {b}) out of order or outside 0..{n}")
        prev = b
    return Repetition(lag_s=float(lag), frame_s=float(frame), n_frames=n, regions=regions)


def _repeated(mask: Sequence[bool], frame_s: float, t: float) -> bool:
    # round() is half-to-even, as numpy's np.round in consistency.py.
    i = round(t / frame_s)
    return 0 <= i < len(mask) and mask[i]


def _match(a: Sequence[float], b: Sequence[float], tol: float) -> list[tuple[int, int]]:
    """One-to-one pairs (i, j) with |a_i - b_j| <= tol, closest first (b sorted)."""
    cand = []
    for i, ai in enumerate(a):
        for j in range(bisect.bisect_left(b, ai - tol), bisect.bisect_right(b, ai + tol)):
            cand.append((abs(ai - b[j]), i, j))
    cand.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for d, i, j in cand:
        if d <= tol and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return pairs


def consistency(
    times: Sequence[float],
    strengths: Sequence[float],
    rep: Repetition,
    t_start: float,
    t_end: float,
) -> dict[str, float | int | None]:
    """rep / rep_s / coverage for detections (seconds of track time, 0-1 strengths) scored over
    [t_start, t_end]: consistency.py's ``consistency``, step for step."""
    order = sorted(range(len(times)), key=lambda k: times[k])
    ts = [float(times[k]) for k in order]
    ss = [float(strengths[k]) for k in order]
    mask = rep.mask()
    fs = rep.frame_s
    L = rep.lag_s
    lo, hi = t_start, t_end - L
    frames = range(math.ceil(lo / fs), int(hi / fs) + 1)
    coverage = (
        sum(_repeated(mask, fs, k * fs) for k in frames) / len(frames) if len(frames) else None
    )
    # A pair counts when either side lies in the region; a lone detection only when it lies
    # strictly inside. Otherwise a hit on a region's edge loses its partner to a fraction of a
    # frame of lag error (or the tolerance).
    tol = MATCH_TOL_S
    grow = math.ceil(tol / fs)
    near = [False] * len(mask)
    for a, b in rep.regions:
        near[max(0, a - grow):min(len(mask), b + grow)] = [True] * (
            min(len(mask), b + grow) - max(0, a - grow)
        )

    def side(t: float) -> tuple[bool, bool]:
        inside = lo <= t <= hi and _repeated(mask, fs, t)
        cand = lo - tol <= t <= hi + tol and _repeated(near, fs, t)
        return cand, inside

    def pick(seq: list[float]) -> tuple[list[float], list[float], list[bool]]:
        t_, s_, k_ = [], [], []
        for t, s in zip(seq, ss, strict=True):
            cand, inside = side(t)
            if cand:
                t_.append(t)
                s_.append(s)
                k_.append(inside)
        return t_, s_, k_

    ta, sa, ka = pick(ts)
    tb, sb, kb = pick([t - L for t in ts])
    pairs = [(i, j) for i, j in _match(ta, tb, tol) if ka[i] or kb[j]]
    ia = {i for i, _ in pairs}
    ib = {j for _, j in pairs}
    lone_a = [i for i in range(len(ta)) if ka[i] and i not in ia]
    lone_b = [j for j in range(len(tb)) if kb[j] and j not in ib]
    n = 2 * len(pairs) + len(lone_a) + len(lone_b)
    if n < MIN_SCORED:
        return {"rep": None, "rep_s": None, "coverage": coverage, "lag_s": L, "n": n}
    diff = sum(abs(sa[i] - sb[j]) for i, j in pairs)
    top = sum(max(sa[i], sb[j]) for i, j in pairs)
    lone = sum(sa[i] for i in lone_a) + sum(sb[j] for j in lone_b)
    den = top + lone
    return {
        "rep": 2 * len(pairs) / n,
        # den is 0 only when every scored detection has strength 0: nothing to agree on.
        "rep_s": 1.0 - (diff + lone) / den if den > 0 else None,
        "coverage": coverage,
        "lag_s": L,
        "n": n,
    }
