"""The constant lag between a run's detections and the labels, and where it came from.

Onset and drum evaluation (mir_eval.onset, madmom.evaluation.onsets, MIREX, the drum
transcription literature) matches detections to labels inside a fixed window (+-50 ms here)
and assumes both are on the same clock. A known constant offset is removed beforehand as a
fixed, calibrated value (madmom's `delay`; Boeck et al., ISMIR 2012, shift every online
detection by one frame), never fitted per file. So, best first:

1. **measured**: the caller measured the lag (e.g. `tools/device_eval.py`'s sync clicks). Used
   as is. The music is still searched in a narrow window around it (+-`CHECK_HALF_WIDTH_MS`)
   and a disagreement over the match tolerance is reported in `note`.
2. **estimated**: no measurement, so the lag is searched for in the music. The box needs this:
   its audio start is when ffplay was spawned, and its detections come 0.24-0.53 s later
   (median 0.27 s), varying run to run (blinky-corpus PR evidence, 2026-11-25).
3. **fallback**: the music can't give a lag (too few detections near a label). The caller's
   fallback lag is used, and the run is scored and flagged, never dropped (review of #307).

The search maximises the one-to-one match count over a window of candidate lags: the
cross-correlation of the detection train with the label train under a +-tolerance box kernel
(Knapp & Carter 1976 for cross-correlation delay estimation), counted one-to-one as the
metrics are. It replaces matching each detection to its *nearest* label, which can't see a
lag over half the label spacing: on the box (true lag ~300 ms, kicks and snares ~200-400 ms
apart) it locked onto an alias near 0 or left the bounds and fell back.

On periodic drums a lag one beat (or half-beat) away matches almost as well: the cross-
correlation of a periodic signal peaks every period. The window is the first defence: it
spans the lags the rig produces and no more. The second is reported, not hidden: the peaks of
the match count at least 2 x tolerance apart (so no detection can match one label at both)
are compared with the best one by McNemar's test (McNemar 1947; exact binomial on the labels
one lag matches and the other doesn't). Peaks it can't tell from the best tie with it: the
run is `ambiguous`, and the tied peak nearest the rig's calibrated typical lag
(`LagSearch.prior_ms`) is taken, so every run of a periodic track resolves the same way. Such
runs are still scored, and flagged; a consumer ranking builds leaves them out or measures the
lag.

The chosen peak is refined with the nearest-label histogram estimator (`scoring.
estimate_audio_latency`) on the detections shifted by the coarse lag, where nearest-label
matching is valid (the residual is well under half the label spacing). The search grid is
10 ms, the estimator's bucket width, so a refinement from a coarse lag near 0 reproduces the
old estimate whenever that one was right: simulated runs (the firmware tracker harness, a
few frames of lag) score as before.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Where the lag comes from (`LagEstimate.source`).
MEASURED = "measured"
ESTIMATED = "estimated"
FALLBACK = "fallback"

GRID_MS = 10  # the search step; a multiple of the refinement's 10 ms histogram buckets
# Fewest one-to-one matches that make a lag: the nearest-label estimator's old minimum (3).
MIN_MATCHES = 3
# The measured-lag cross-check searches this far either side of the measured value.
CHECK_HALF_WIDTH_MS = 100.0
# McNemar's test level: below it, the best lag is told apart from the runner-up.
AMBIGUITY_ALPHA = 0.05

@dataclass(frozen=True)
class LagEstimate:
    """The lag a run is scored at, and the evidence for it."""

    lag_ms: float
    source: str  # MEASURED, ESTIMATED or FALLBACK
    search: LagSearch  # the lags searched (for MEASURED: the cross-check's, around it)
    matches: int  # one-to-one matches at the coarse lag (0 for FALLBACK with nothing)
    runner_up_ms: float | None = None  # best lag >= 2 x tolerance from the chosen one
    runner_up_matches: int | None = None
    ambiguous: bool = False  # the music can't tell lag_ms from runner_up_ms
    check_ms: float | None = None  # MEASURED only: the music's lag near the measured one
    note: str | None = None  # why it fell back, or what disagreed

    def as_dict(self) -> dict[str, Any]:
        return {
            "lagMs": round(self.lag_ms, 1),
            "lagSource": self.source,
            "lagWindowMs": [self.search.lo_ms, self.search.hi_ms],
            "lagPriorMs": self.search.prior_ms,
            "lagMatches": self.matches,
            "lagRunnerUpMs": None if self.runner_up_ms is None else round(self.runner_up_ms, 1),
            "lagRunnerUpMatches": self.runner_up_matches,
            "lagAmbiguous": self.ambiguous,
            "lagCheckMs": None if self.check_ms is None else round(self.check_ms, 1),
            "lagNote": self.note,
        }


def _match_count(est: list[float], ref: list[float], tol: float) -> int:
    """Size of the maximum one-to-one matching of two sorted lists within +-tol (greedy in
    time order is optimal for equal windows on a line). Same as drum_scoring.match_count."""
    j = n = 0
    for r in ref:
        while j < len(est) and est[j] < r - tol:
            j += 1
        if j < len(est) and est[j] <= r + tol:
            n += 1
            j += 1
    return n


def _matched_refs(est: list[float], ref: list[float], tol: float) -> set[int]:
    """Indices of the refs the same matching pairs up."""
    j = 0
    out: set[int] = set()
    for i, r in enumerate(ref):
        while j < len(est) and est[j] < r - tol:
            j += 1
        if j < len(est) and est[j] <= r + tol:
            out.add(i)
            j += 1
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs (binomial, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def _grid(lo: float, hi: float) -> list[int]:
    start = math.ceil(lo / GRID_MS) * GRID_MS
    return list(range(int(start), int(math.floor(hi / GRID_MS) * GRID_MS) + 1, GRID_MS))


@dataclass(frozen=True)
class LagSearch:
    """Where to look for a run's lag: the window of lags the rig can produce, and its
    calibrated typical lag (the prior a tie between aliases goes to)."""

    lo_ms: float
    hi_ms: float
    prior_ms: float


# Simulated runs (the firmware tracker harness feeds the audio straight in): the lag is the
# detector's own few frames. The window is the old plausible-latency bounds
# (scoring.PLAUSIBLE_LATENCY_*); a tie goes to the lag nearest 0, as the nearest-label
# estimator's did.
SIMULATED = LagSearch(-50.0, 300.0, 0.0)
# Acoustic runs on the test box: ffplay spawn -> speaker -> mic -> detection. Box runs on
# 2026-11-24..25 (27 runs whose lag the music fixes, 3-5 per firmware, 1-5 per track; corpus
# PR evidence): median 272 ms, p10 245, p90 337, one slow start at 530. The window keeps
# ~135 ms below the fastest and ~170 above the slowest and no wider: every extra beat period
# of width is another alias.
ACOUSTIC = LagSearch(100.0, 700.0, 270.0)


@dataclass(frozen=True)
class _Peak:
    lag: int  # centre of the plateau (ms, on the grid)
    first: int  # its grid lags, first and last
    last: int
    n: int  # one-to-one matches


def _peaks(lags: list[int], counts: list[int], sep_ms: float) -> list[_Peak]:
    """Local maxima of the match count (plateaus: runs of equal counts with lower or no
    neighbours), best first, each at least `sep_ms` from every better one (non-maximum
    suppression), so no two can share a detection-label pair."""
    plateaus: list[_Peak] = []
    i = 0
    while i < len(lags):
        j = i
        while j + 1 < len(lags) and counts[j + 1] == counts[i]:
            j += 1
        left = counts[i - 1] if i > 0 else -1
        right = counts[j + 1] if j + 1 < len(lags) else -1
        if counts[i] > left and counts[i] > right:
            plateaus.append(_Peak(lags[(i + j) // 2], lags[i], lags[j], counts[i]))
        i = j + 1
    plateaus.sort(key=lambda p: (-p.n, -(p.last - p.first), p.first))
    kept: list[_Peak] = []
    for p in plateaus:
        if all(p.last < k.first - sep_ms or p.first > k.last + sep_ms for k in kept):
            kept.append(p)
    return kept


@dataclass(frozen=True)
class SearchResult:
    lag: int  # the chosen coarse lag (ms, on the grid)
    matches: int
    runner_up: int | None  # the best other peak (among the tied ones when ambiguous)
    runner_up_matches: int | None
    ambiguous: bool  # other peaks the music can't tell from the best one


def search_lag(
    det_s: list[float], ref_s: list[float], search: LagSearch, tol_s: float
) -> SearchResult | None:
    """The coarse lag for `det_s - lag` against `ref_s`: the peak of the one-to-one match
    count over the grid in `search`'s window. Peaks that McNemar's test can't tell from the
    best one (p >= AMBIGUITY_ALPHA) tie with it: the run is ambiguous, and the tied peak
    nearest `search.prior_ms` is chosen. None when no lag reaches MIN_MATCHES."""
    det = sorted(det_s)
    ref = sorted(ref_s)
    lags = _grid(search.lo_ms, search.hi_ms)
    if not lags or not det or not ref:
        return None
    counts = [_match_count([t - lag / 1000.0 for t in det], ref, tol_s) for lag in lags]
    if max(counts) < MIN_MATCHES:
        return None
    peaks = _peaks(lags, counts, 2.0 * tol_s * 1000.0)
    best = peaks[0]

    def matched(lag: int) -> set[int]:
        return _matched_refs([t - lag / 1000.0 for t in det], ref, tol_s)

    at_best = matched(best.lag)
    tied = [best] + [
        p for p in peaks[1:]
        if mcnemar_p(len(at_best - (m := matched(p.lag))), len(m - at_best)) >= AMBIGUITY_ALPHA
    ]
    chosen = min(tied, key=lambda p: (abs(p.lag - search.prior_ms), -p.n))
    others = [p for p in (tied if len(tied) > 1 else peaks) if p is not chosen]
    ru = max(others, key=lambda p: (p.n, -abs(p.lag - search.prior_ms))) if others else None
    return SearchResult(
        lag=chosen.lag, matches=chosen.n,
        runner_up=ru.lag if ru else None, runner_up_matches=ru.n if ru else None,
        ambiguous=len(tied) > 1,
    )


def _refine(
    det: list[tuple[float, float]], ref_s: list[float], coarse_ms: int, duration_s: float,
    tol_s: float,
) -> float:
    """The coarse lag plus the nearest-label histogram estimate of what's left (valid once
    the residual is well under half the label spacing). Kept only if it stays within the
    match tolerance of the coarse lag, which it must if the coarse lag is right."""
    from .scoring import estimate_audio_latency

    fine = estimate_audio_latency(
        [{"timestamp_ms": t * 1000.0 - coarse_ms, "strength": s} for t, s in det],
        [{"time": t, "strength": 1.0, "expect_trigger": True} for t in ref_s],
        duration_s * 1000.0,
    )
    if fine is None or abs(fine) > tol_s * 1000.0:
        return float(coarse_ms)
    return coarse_ms + fine


def estimate_lag(
    detections: list[tuple[float, float]],
    ref_s: list[float],
    duration_s: float,
    *,
    search: LagSearch,
    tol_s: float,
    measured_ms: float | None = None,
    fallback_ms: float = 0.0,
) -> LagEstimate:
    """The lag to score `detections` (time_s, strength) against `ref_s` at; see the module
    docstring for the order (measured, estimated, fallback). Fallbacks, ambiguous lags and
    measured lags the music disagrees with are logged as well as reported."""
    det_s = [t for t, _ in detections]
    if measured_ms is not None:
        check_search = LagSearch(
            measured_ms - CHECK_HALF_WIDTH_MS, measured_ms + CHECK_HALF_WIDTH_MS, measured_ms
        )
        found = search_lag(det_s, ref_s, check_search, tol_s)
        check = _refine(detections, ref_s, found.lag, duration_s, tol_s) if found else None
        note = None
        if check is None:
            note = "the music gave no lag near the measured one to check it against"
        elif abs(check - measured_ms) > tol_s * 1000.0:
            note = f"the music's lag ({check:.0f} ms) disagrees with the measured one"
        if note:
            log.warning("measured lag %.1f ms: %s", measured_ms, note)
        return LagEstimate(
            lag_ms=measured_ms, source=MEASURED, search=check_search,
            matches=found.matches if found else 0, check_ms=check, note=note,
        )
    found = search_lag(det_s, ref_s, search, tol_s)
    if found is None:
        note = (
            f"fewer than {MIN_MATCHES} detections match a label at any lag in "
            f"{search.lo_ms:.0f}..{search.hi_ms:.0f} ms; scored at the fallback {fallback_ms:.0f} ms"
        )
        log.warning("lag fallback: %s", note)
        return LagEstimate(lag_ms=fallback_ms, source=FALLBACK, search=search, matches=0, note=note)
    lag = _refine(detections, ref_s, found.lag, duration_s, tol_s)
    note = None
    if found.ambiguous:
        note = (
            f"lag {lag:.0f} ms ({found.matches} matches) and {found.runner_up} ms "
            f"({found.runner_up_matches}) aren't told apart (McNemar p >= {AMBIGUITY_ALPHA}): "
            f"periodic drums; took the one nearer the rig's typical {search.prior_ms:.0f} ms. "
            "Measure the lag to rank this run"
        )
        log.warning("ambiguous lag: %s", note)
    return LagEstimate(
        lag_ms=lag, source=ESTIMATED, search=search, matches=found.matches,
        runner_up_ms=None if found.runner_up is None else float(found.runner_up),
        runner_up_matches=found.runner_up_matches, ambiguous=found.ambiguous, note=note,
    )
