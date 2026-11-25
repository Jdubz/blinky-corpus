"""The run lag (lag.py): search, refinement, measured lags, fallbacks and periodic music."""

from __future__ import annotations

import logging
import random

import pytest

from blinky_scoring.drum_scoring import score_drums
from blinky_scoring.lag import (
    ACOUSTIC,
    ESTIMATED,
    FALLBACK,
    MEASURED,
    SIMULATED,
    LagSearch,
    estimate_lag,
    mcnemar_p,
)
from blinky_scoring.scoring import estimate_audio_latency
from blinky_scoring.types import DrumLabels

TOL = 0.05


def groove(bars: int = 16, bpm: float = 128.0) -> tuple[list[float], list[float]]:
    """A non-repeating groove: four-on-the-floor kicks, snares on 2 and 4, and every fourth
    bar a fill (extra kicks on the off-beats of beats 3 and 4, no snare on 4)."""
    beat = 60.0 / bpm
    kicks: list[float] = []
    snares: list[float] = []
    for b in range(bars):
        t0 = 1.0 + b * 4 * beat
        kicks += [t0 + i * beat for i in range(4)]
        if b % 4 == 3:
            kicks += [t0 + 2.5 * beat, t0 + 3.5 * beat]
            snares.append(t0 + beat)
        else:
            snares += [t0 + beat, t0 + 3 * beat]
    return sorted(kicks), sorted(snares)


def detect(times: list[float], lag_s: float, jitter_s: float = 0.0, seed: int = 0,
           keep: float = 1.0, extra: int = 0, span: float = 0.0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    out = [(t + lag_s + rng.uniform(-jitter_s, jitter_s), 1.0) for t in times if rng.random() < keep]
    out += [(rng.uniform(0.0, span), 0.6) for _ in range(extra)]
    return sorted(out)


def nearest_label_estimate(det: list[tuple[float, float]], ref: list[float], dur: float) -> float | None:
    return estimate_audio_latency(
        [{"timestamp_ms": t * 1000, "strength": s} for t, s in det],
        [{"time": r, "strength": 1.0} for r in ref],
        dur * 1000,
    )


def test_a_box_lag_over_half_the_label_spacing_is_found() -> None:
    # The box: ~300 ms from ffplay's spawn to detection, kicks and snares ~230 ms apart. The
    # old nearest-label estimate can't see it (it reports an alias near 0); the search does.
    kicks, snares = groove()
    ref = sorted(kicks + snares)
    d = detect(ref, 0.300, jitter_s=0.008, keep=0.8, extra=10, span=32.0)
    old = nearest_label_estimate(d, ref, 32.0)
    assert old is not None and abs(old - 300) > 100
    est = estimate_lag(d, ref, 32.0, search=ACOUSTIC, tol_s=TOL)
    assert est.source == ESTIMATED and not est.ambiguous and est.note is None
    assert abs(est.lag_ms - 300) < 10
    s = score_drums(d, DrumLabels(kicks, snares, []), 32.0, lag_search=ACOUSTIC)
    assert s.kick_r50 > 0.7 and s.on_drum > 0.85 and s.as_dict()["lagSource"] == ESTIMATED


@pytest.mark.parametrize("lag_ms", [0, 12, 25, 40, 80])
def test_simulated_lags_score_exactly_as_the_nearest_label_estimate(lag_ms: int) -> None:
    # The harness (no acoustic path): the refinement on the 10 ms grid reproduces the old
    # estimate exactly, so harness scores don't move.
    kicks, snares = groove()
    ref = sorted(kicks + snares)
    for seed in range(5):
        d = detect(ref, lag_ms / 1000, jitter_s=0.012, seed=seed, keep=0.85, extra=25, span=32.0)
        est = estimate_lag(d, ref, 32.0, search=SIMULATED, tol_s=TOL)
        assert est.source == ESTIMATED
        assert est.lag_ms == pytest.approx(nearest_label_estimate(d, ref, 32.0), abs=1e-9)


def test_strictly_periodic_drums_are_flagged_ambiguous_not_hidden() -> None:
    # Adversarial: one kick every 0.5 s and nothing else. Lags 250 and 750 ms... and 250 and
    # every 500 ms step match equally: no music can tell them apart within the window.
    kicks = [1.0 + 0.5 * i for i in range(60)]
    d = detect(kicks, 0.250, jitter_s=0.005)
    est = estimate_lag(d, kicks, 32.0, search=LagSearch(100.0, 900.0, 270.0), tol_s=TOL)
    assert est.source == ESTIMATED and est.ambiguous
    assert est.runner_up_ms is not None and abs(abs(est.runner_up_ms - est.lag_ms) - 500) < 30
    assert est.runner_up_matches is not None and est.runner_up_matches >= est.matches - 1
    assert est.note and "McNemar" in est.note
    s = score_drums(d, DrumLabels(kicks, [], []), 32.0, lag_search=LagSearch(100.0, 900.0, 270.0))
    assert s.as_dict()["lagAmbiguous"] is True  # still scored, and says so


@pytest.mark.parametrize("prior_ms", [150.0, 400.0, 650.0])
def test_a_periodic_tie_goes_to_the_calibrated_prior(prior_ms: float) -> None:
    # Kicks every 0.25 s (sixteenths at 60 BPM): true lag 400 ms, aliases at 150 and 650.
    # Whichever alias is nearest the rig's typical lag wins, every run alike, and it's flagged.
    kicks = [1.0 + 0.25 * i for i in range(120)]
    d = detect(kicks, 0.400, jitter_s=0.004, seed=3)
    est = estimate_lag(d, kicks, 32.0, search=LagSearch(100.0, 700.0, prior_ms), tol_s=TOL)
    assert est.ambiguous and abs(est.lag_ms - prior_ms) < 20
    assert est.runner_up_ms is not None and abs(est.runner_up_ms - est.lag_ms) >= 200


def test_a_fill_breaks_the_periodic_tie() -> None:
    # The same idea with real structure (fills every fourth bar): the true lag wins clearly.
    kicks, snares = groove()
    ref = sorted(kicks + snares)
    d = detect(ref, 0.420, jitter_s=0.006, keep=0.9)
    est = estimate_lag(d, ref, 32.0, search=ACOUSTIC, tol_s=TOL)
    assert not est.ambiguous and abs(est.lag_ms - 420) < 10
    assert est.runner_up_matches is not None and est.runner_up_matches < est.matches


def test_the_window_bounds_the_search() -> None:
    kicks, snares = groove()
    ref = sorted(kicks + snares)
    d = detect(ref, 0.300)
    est = estimate_lag(d, ref, 32.0, search=LagSearch(-50.0, 200.0, 0.0), tol_s=TOL)
    assert est.lag_ms <= 250  # 300 is outside; whatever it finds is in (or near) the window


def test_too_few_matches_falls_back_explicitly(caplog: pytest.LogCaptureFixture) -> None:
    kicks, snares = groove()
    ref = sorted(kicks + snares)
    with caplog.at_level(logging.WARNING, logger="blinky_scoring.lag"):
        est = estimate_lag([(5.0, 1.0)], ref, 32.0, search=ACOUSTIC, tol_s=TOL,
                           fallback_ms=123.0)
    assert est.source == FALLBACK and est.lag_ms == 123.0 and est.matches == 0
    assert est.note and "fallback" in est.note
    assert any("lag fallback" in r.message for r in caplog.records)
    s = score_drums([], DrumLabels(kicks, snares, []), 32.0, fallback_lag_ms=50.0)
    d = s.as_dict()
    assert d["lagSource"] == FALLBACK and d["lagEstimated"] is False and d["lagMs"] == 50.0
    assert d["lagNote"] and d["drumF"] == 0.0


def test_a_measured_lag_is_used_and_cross_checked(caplog: pytest.LogCaptureFixture) -> None:
    kicks, snares = groove()
    ref = sorted(kicks + snares)
    d = detect(ref, 0.300, jitter_s=0.005)
    est = estimate_lag(d, ref, 32.0, search=ACOUSTIC, tol_s=TOL, measured_ms=300.0)
    assert est.source == MEASURED and est.lag_ms == 300.0 and est.note is None
    assert est.check_ms is not None and abs(est.check_ms - 300) < 10
    # A wrong measurement is still used (it's the caller's), but the disagreement is reported.
    with caplog.at_level(logging.WARNING, logger="blinky_scoring.lag"):
        bad = estimate_lag(d, ref, 32.0, search=ACOUSTIC, tol_s=TOL, measured_ms=230.0)
    assert bad.source == MEASURED and bad.lag_ms == 230.0
    assert bad.note and "disagrees" in bad.note and caplog.records


def test_the_score_reports_where_its_lag_came_from() -> None:
    kicks, snares = groove()
    d = score_drums(detect(sorted(kicks + snares), 0.03), DrumLabels(kicks, snares, []), 32.0)
    out = d.as_dict()
    for k in ("lagMs", "lagSource", "lagWindowMs", "lagMatches", "lagRunnerUpMs",
              "lagRunnerUpMatches", "lagAmbiguous", "lagCheckMs", "lagNote", "lagEstimated"):
        assert k in out
    assert out["lagSource"] == ESTIMATED and out["lagWindowMs"] == [SIMULATED.lo_ms, SIMULATED.hi_ms]


def test_mcnemar_exact() -> None:
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_p(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_p(10, 10) == 1.0
    # Long tracks: thousands of discordant labels must not overflow.
    assert mcnemar_p(1000, 1000) == 1.0
    assert mcnemar_p(1300, 1000) < 1e-9
