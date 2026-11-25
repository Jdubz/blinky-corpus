"""Drum-hit scoring: the primary metric for the onset detector (#294, CLAUDE.md).

The firmware detects kicks and snares; hats and other changes are false positives. All-onset
F1 scored 0.68 while the hat looked "almost entirely random" (#280), so a run is scored on:

  kick_r50, kick_r30  share of kicks with a detection within +-50 / +-30 ms
  snare_r50           share of snares with a detection within +-50 ms (PROVISIONAL until #284)
  on_drum             share of detections within +-50 ms of a kick or snare
  drum_f              F1 of detections against kicks + snares at +-50 ms
  lone_hat            share of lone hats (no kick or snare within 50 ms) that fired a detection
  det_per_s           detections per second

Matching is one-to-one. A constant per-run lag is removed first, as the #280 study did, so a
fixed pipeline delay isn't scored as misses: the caller's measured lag, else one searched for
against the kick + snare labels within a window of lags the rig can produce, else the caller's
fallback (`lag.py` has the method and its sources). Where it came from is in the score
(`lagSource`, with the evidence: `lagMatches`, `lagRunnerUpMs`, `lagAmbiguous`, `lagNote`).

Labels, best first (`drum_label_path`):
  1. `edm/drums_human/<stem>.drums.json` marked complete: hand-curated ground truth (#4),
     used as is.
  2. `edm/drums_adtof/<stem>.kick_weighted.json`: ADTOF (synthetic kick F1 0.93, snare 1.00),
     used as is. The consensus cleaning below was built for the band generator's false
     positives; applied to ADTOF it would drop real hits the onset consensus misses.
  3. `edm/kick_weighted/` (the band-onset generator, legacy) cleaned by
     `ml-training/scripts/clean_kick_weighted.py`'s rule: an onset is kept only if a full-mix
     consensus onset backed by >= 2 systems lies within 70 ms. `ml-training/tests/
     test_drum_label_sync.py` checks this copy of the rule against that script.
A human file's hats (not curated) come from the ADTOF file, for `lone_hat`.

Pure Python: blinky-server doesn't depend on numpy.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lag import FALLBACK, SIMULATED, LagEstimate, LagSearch, estimate_lag
from .types import DrumLabels

CLEAN_TOLERANCE_SEC = 0.070
CLEAN_MIN_SYSTEMS = 2
DRUM_TOL_SEC = 0.050
TIGHT_TOL_SEC = 0.030
LONE_HAT_SEC = 0.050


@dataclass
class DrumScore:
    kick_r50: float
    kick_r30: float
    snare_r50: float
    on_drum: float
    drum_f: float
    lone_hat: float
    det_per_s: float
    lag_ms: float | None
    n_kicks: int
    n_snares: int
    n_detections: int
    # False when the lag couldn't be measured or estimated (too few detections near a label)
    # and the caller's fallback was used (`lag.source == "fallback"`). The run is still
    # scored: a silent detector scores 0.
    lag_estimated: bool = True
    provisional: list[str] = field(default_factory=lambda: ["snare_r50"])
    lag: LagEstimate | None = None  # where lag_ms came from, and the evidence for it

    def as_dict(self) -> dict[str, Any]:
        def r(x: float) -> float | None:
            return None if math.isnan(x) else round(x, 4)

        out: dict[str, Any] = {
            "kickR50": r(self.kick_r50),
            "kickR30": r(self.kick_r30),
            "snareR50": r(self.snare_r50),
            "onDrum": r(self.on_drum),
            "drumF": r(self.drum_f),
            "loneHat": r(self.lone_hat),
            "detPerSec": r(self.det_per_s),
            "lagMs": None
            if self.lag_ms is None or math.isnan(self.lag_ms)
            else round(self.lag_ms, 1),
            "kicks": self.n_kicks,
            "snares": self.n_snares,
            "detections": self.n_detections,
            "lagEstimated": self.lag_estimated,
            "provisional": self.provisional,
        }
        if self.lag is not None:
            out.update({k: v for k, v in self.lag.as_dict().items() if k != "lagMs"})
        return out


def clean_kick_weighted(kw: dict[str, Any], consensus: dict[str, Any]) -> list[dict[str, Any]]:
    """The kept `kick_weighted` onsets (see the module docstring for the rule)."""
    supported = sorted(
        o["time"] for o in consensus.get("onsets", []) if o.get("systems", 1) >= CLEAN_MIN_SYSTEMS
    )
    if not supported:
        raise ValueError(f"no consensus onsets backed by >= {CLEAN_MIN_SYSTEMS} systems")
    kept = []
    for o in kw.get("onsets", []):
        i = bisect.bisect_left(supported, o["time"])
        near = [abs(supported[j] - o["time"]) for j in (i - 1, i) if 0 <= j < len(supported)]
        if near and min(near) <= CLEAN_TOLERANCE_SEC:
            kept.append(o)
    return kept


HUMAN_DIR = "drums_human"
ADTOF_DIR = "drums_adtof"
LEGACY_DIR = "kick_weighted"


def _human_path(corpus_dir: Path, stem: str) -> Path:
    return corpus_dir / HUMAN_DIR / f"{stem}.drums.json"


def _adtof_path(corpus_dir: Path, stem: str) -> Path:
    return corpus_dir / ADTOF_DIR / f"{stem}.kick_weighted.json"


def _human_complete(path: Path) -> bool:
    return path.is_file() and json.loads(path.read_text()).get("complete") is True


def drum_label_path(corpus_dir: Path, stem: str) -> Path:
    """The best drum labels for a track: complete human ground truth, else ADTOF, else the
    legacy band-generator file (which may not exist; callers check)."""
    human = _human_path(corpus_dir, stem)
    if _human_complete(human):
        return human
    adtof = _adtof_path(corpus_dir, stem)
    if adtof.is_file():
        return adtof
    return corpus_dir / LEGACY_DIR / f"{stem}.kick_weighted.json"


def load_drum_labels(kw_path: Path, consensus_path: Path) -> DrumLabels:
    """Load the labels at ``kw_path`` (any of the three kinds ``drum_label_path`` returns)."""
    data = json.loads(kw_path.read_text())
    if kw_path.parent.name == HUMAN_DIR:
        if data.get("complete") is not True:
            raise ValueError(f"{kw_path}: human drum labels are not marked complete")
        onsets = data.get("onsets", [])
        adtof = _adtof_path(kw_path.parent.parent, data.get("track") or kw_path.name.split(".")[0])
        hats = (
            sorted(o["time"] for o in json.loads(adtof.read_text()).get("onsets", [])
                   if o["type"] == "hihat")
            if adtof.is_file()
            else []
        )
        return DrumLabels(
            sorted(o["time"] for o in onsets if o["type"] == "kick"),
            sorted(o["time"] for o in onsets if o["type"] == "snare"),
            hats,
            source=str(kw_path),
        )
    if data.get("skipped"):
        raise ValueError(f"{kw_path}: drum labels marked skipped (separation failed upstream)")
    if kw_path.parent.name == ADTOF_DIR:
        kept = data.get("onsets", [])
    else:
        kept = clean_kick_weighted(data, json.loads(consensus_path.read_text()))

    def times(kind: str) -> list[float]:
        return sorted(o["time"] for o in kept if o["type"] == kind)

    return DrumLabels(
        times("kick"),
        times("snare"),
        times("hihat"),
        source=str(kw_path),
        dropped=len(data.get("onsets", [])) - len(kept),
    )


def match_count(est: list[float], ref: list[float], tol: float) -> int:
    """Size of the maximum one-to-one matching between two sorted lists within +-tol. For
    events on a line with equal windows, greedy in time order is optimal."""
    j = n = 0
    for r in ref:
        while j < len(est) and est[j] < r - tol:
            j += 1
        if j < len(est) and est[j] <= r + tol:
            n += 1
            j += 1
    return n


def _near(sorted_xs: list[float], t: float, tol: float) -> bool:
    i = bisect.bisect_left(sorted_xs, t - tol)
    return i < len(sorted_xs) and sorted_xs[i] <= t + tol


def score_drums(
    detections: list[tuple[float, float]],
    labels: DrumLabels,
    duration_s: float,
    lag_ms: float | None = None,
    fallback_lag_ms: float = 0.0,
    *,
    lag_search: LagSearch = SIMULATED,
) -> DrumScore:
    """Score detections (time_s, strength) against drum labels on the same clock.

    `lag_ms` is the caller's measured lag, used as is (and cross-checked against the music,
    reported in `lag`). Without it the lag is searched for against kicks + snares within
    `lag_search`'s window: the default fits simulated runs (a few frames of detector delay);
    acoustic runs pass `lag.ACOUSTIC`. When the music gives no lag (too few
    detections near a label, e.g. a detector a tuning value has silenced), the run is scored
    at `fallback_lag_ms` and flagged (`lag_estimated` False, `lag.source` "fallback"), never
    dropped: dropping it would let a tuner prefer values that silence the detector (review of
    #307). Metrics are NaN only where the labels themselves are empty (e.g. no kicks)."""
    drums = sorted(labels.kicks + labels.snares)
    lag = estimate_lag(
        detections,
        drums,
        duration_s,
        search=lag_search,
        tol_s=DRUM_TOL_SEC,
        measured_ms=lag_ms,
        fallback_ms=fallback_lag_ms,
    )
    lag_ms = lag.lag_ms
    det_per_s = len(detections) / duration_s if duration_s > 0 else float("nan")
    est = sorted(t - lag_ms / 1000.0 for t, _ in detections)

    def recall(ref: list[float], tol: float) -> float:
        return match_count(est, ref, tol) / len(ref) if ref else float("nan")

    tp = match_count(est, drums, DRUM_TOL_SEC)
    p = tp / len(est) if est else 0.0
    r = tp / len(drums) if drums else float("nan")
    lone = [h for h in labels.hats if not _near(drums, h, LONE_HAT_SEC)]
    return DrumScore(
        kick_r50=recall(labels.kicks, DRUM_TOL_SEC),
        kick_r30=recall(labels.kicks, TIGHT_TOL_SEC),
        snare_r50=recall(labels.snares, DRUM_TOL_SEC),
        # No detections: on-drum is 0 (as precision is), so silencing can't look neutral.
        on_drum=(sum(_near(drums, t, DRUM_TOL_SEC) for t in est) / len(est)) if est else 0.0,
        drum_f=(2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
        lone_hat=(sum(_near(est, h, DRUM_TOL_SEC) for h in lone) / len(lone))
        if lone
        else float("nan"),
        det_per_s=det_per_s,
        lag_ms=lag_ms,
        n_kicks=len(labels.kicks),
        n_snares=len(labels.snares),
        n_detections=len(est),
        lag_estimated=lag.source != FALLBACK,
        lag=lag,
    )
