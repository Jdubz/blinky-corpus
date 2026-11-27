"""Drum-hit scoring (#294): matching, the metrics, and their plumbing into device runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from blinky_scoring.drum_scoring import clean_kick_weighted, match_count, score_drums
from blinky_scoring.scoring import format_score_summary, score_device_run
from blinky_scoring.track_discovery import (
    discover_tracks,
    load_ground_truth,
    segment_ground_truth,
)
from blinky_scoring.types import (
    DrumLabels,
    GroundTruth,
    GroundTruthHit,
    TestData,
    TransientEvent,
)

CORPUS = Path(__file__).resolve().parents[1] / "edm"

# A two-bar groove at 120 BPM: kick on 1 and 3, snare on 2 and 4, hats on the off-beats.
KICKS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SNARES = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
HATS = [1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
LABELS = DrumLabels(KICKS, SNARES, HATS)


def det(times: list[float]) -> list[tuple[float, float]]:
    return [(t, 1.0) for t in times]


def test_match_count_is_one_to_one_and_maximal() -> None:
    assert match_count([1.00, 1.01], [1.0], 0.05) == 1  # two detections, one label
    assert match_count([1.0], [0.98, 1.02], 0.05) == 1  # one detection, two labels
    # Greedy nearest would pair 1.04 with 1.06 and strand 1.0; the maximum matching is 2.
    assert match_count([1.04, 1.10], [1.0, 1.06], 0.05) == 2
    assert match_count([], [1.0], 0.05) == 0


def test_a_drum_only_detector_scores_perfectly_through_a_constant_lag() -> None:
    # 40 ms late throughout: a pipeline delay, removed before scoring.
    s = score_drums(det([t + 0.04 for t in sorted(KICKS + SNARES)]), LABELS, 7.0)
    assert s.lag_ms is not None and abs(s.lag_ms - 40) < 5
    assert s.kick_r50 == 1.0 and s.kick_r30 == 1.0 and s.snare_r50 == 1.0
    assert s.on_drum == 1.0 and s.drum_f == 1.0
    assert s.lone_hat == 0.0


def test_a_hat_detector_is_off_drum() -> None:
    # Fires on every lone hat and nothing else; its lag is given (no drums to estimate it on).
    s = score_drums(det(HATS), LABELS, 7.0, lag_ms=0.0)
    assert s.kick_r50 == 0.0 and s.on_drum == 0.0 and s.drum_f == 0.0
    assert s.lone_hat == 1.0


def test_kicks_only_and_spurious_detections() -> None:
    s = score_drums(det([*KICKS, 0.3, 0.6]), LABELS, 7.0, lag_ms=0.0)
    assert s.kick_r50 == 1.0 and s.snare_r50 == 0.0
    assert math.isclose(s.on_drum, 6 / 8)
    p, r = 6 / 8, 6 / 12
    assert math.isclose(s.drum_f, 2 * p * r / (p + r))
    assert s.provisional == ["snare_r50"]  # snare labels are unreliable until #284


def test_a_silent_detector_scores_zero_not_missing() -> None:
    # Too few detections to estimate a lag: scored at the fallback, never left out, so a
    # tuner can't prefer a value that silences the detector (review of #307).
    for d in ([], det([1.0])):
        s = score_drums(d, LABELS, 7.0)
        assert s.lag_estimated is False
        assert s.kick_r50 <= 1 / 6 and s.drum_f <= 0.2 and s.as_dict()["kickR50"] is not None
    s = score_drums([], LABELS, 7.0)
    assert s.kick_r50 == 0.0 and s.drum_f == 0.0 and s.on_drum == 0.0


def test_empty_labels_are_nan() -> None:
    s = score_drums(det(KICKS), DrumLabels([], [], []), 7.0, lag_ms=0.0)
    assert math.isnan(s.kick_r50) and s.as_dict()["kickR50"] is None


def test_a_zero_length_run_serialises() -> None:
    import json

    json.dumps(score_drums([], LABELS, 0.0).as_dict(), allow_nan=False)


def test_cleaning_needs_two_systems_within_70ms() -> None:
    kw = {
        "onsets": [
            {"time": 1.0, "type": "kick"},
            {"time": 2.0, "type": "kick"},
            {"time": 3.0, "type": "snare"},
        ]
    }
    cons = {
        "onsets": [
            {"time": 1.05, "systems": 3},
            {"time": 2.0, "systems": 1},
            {"time": 3.2, "systems": 5},
        ]
    }
    assert [o["time"] for o in clean_kick_weighted(kw, cons)] == [1.0]


def test_segment_shifts_drum_labels_with_the_seek() -> None:
    gt = GroundTruth(pattern="t", duration_ms=10_000, hits=[], onsets=[], drums=LABELS)
    seg = segment_ground_truth(gt, seek_sec=2.0, duration_ms=2000)
    assert seg.drums is not None
    assert seg.drums.kicks == [0.0, 1.0] and seg.drums.snares == [0.5, 1.5]


def test_the_corpus_has_drum_labels_and_they_load() -> None:
    tracks = discover_tracks(CORPUS)
    assert len(tracks) == 18 and all("drum_labels" in t for t in tracks)
    t = tracks[0]
    gt = load_ground_truth(t["ground_truth"], t.get("human_edits"), t["drum_labels"])
    # ADTOF labels, used as is (no consensus cleaning), until human ground truth is complete.
    assert gt.drums is not None and len(gt.drums.kicks) > 50 and len(gt.drums.snares) > 20
    assert "/drums_adtof/" in gt.drums.source and gt.drums.dropped == 0


def _write(path: Path, obj: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def test_drum_label_precedence_human_then_adtof_then_legacy(tmp_path: Path) -> None:
    from blinky_scoring.drum_scoring import drum_label_path

    legacy = tmp_path / "kick_weighted" / "t.kick_weighted.json"
    assert drum_label_path(tmp_path, "t") == legacy  # nothing exists: the legacy path
    adtof = _write(tmp_path / "drums_adtof" / "t.kick_weighted.json", {"onsets": []})
    assert drum_label_path(tmp_path, "t") == adtof
    human = _write(tmp_path / "drums_human" / "t.drums.json", {"complete": False, "onsets": []})
    assert drum_label_path(tmp_path, "t") == adtof  # an unfinished review isn't ground truth
    _write(human, {"complete": True, "onsets": []})
    assert drum_label_path(tmp_path, "t") == human


def _human_track(tmp_path: Path, **over: object) -> Path:
    """Track t: its mp3, ADTOF labels with a hat, and a complete human file for that mp3."""
    import hashlib

    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"ID3 not really audio")
    _write(tmp_path / "drums_adtof" / "t.kick_weighted.json", {"onsets": [
        {"time": 1.0, "type": "kick", "weight": 1.0}, {"time": 1.25, "type": "hihat", "weight": 0.0},
    ]})
    doc = {
        "schema": 1,
        "track": "t",
        "audio_sha256": hashlib.sha256(mp3.read_bytes()).hexdigest(),
        "complete": True,
        "onsets": [{"time": 2.0, "type": "snare"}, {"time": 1.0, "type": "kick"}],
    }
    return _write(tmp_path / "drums_human" / "t.drums.json", doc | over)


def test_human_labels_load_as_is_with_hats_from_adtof(tmp_path: Path) -> None:
    from blinky_scoring.drum_scoring import load_drum_labels

    human = _human_track(tmp_path)
    labels = load_drum_labels(human, tmp_path / "unused.json")
    assert labels.kicks == [1.0] and labels.snares == [2.0] and labels.hats == [1.25]
    _write(human, {"complete": False, "onsets": []})
    with pytest.raises(ValueError, match="not marked complete"):
        load_drum_labels(human, tmp_path / "unused.json")


def test_human_labels_for_other_audio_are_refused(tmp_path: Path) -> None:
    """Human ground truth is for one mp3 (audio_sha256). If the track's audio changed under it
    (a PR that replaces an mp3 changes no label file, so the label checks don't look), its times
    belong to other audio: refused loudly, never scored against."""
    from blinky_scoring.drum_scoring import load_drum_labels

    human = _human_track(tmp_path)
    (tmp_path / "t.mp3").write_bytes(b"ID3 a re-encoded mp3")
    with pytest.raises(ValueError, match="audio_sha256"):
        load_drum_labels(human, tmp_path / "unused.json")
    (tmp_path / "t.mp3").unlink()
    with pytest.raises(ValueError, match="no audio"):
        load_drum_labels(human, tmp_path / "unused.json")


def test_human_labels_are_for_their_own_track(tmp_path: Path) -> None:
    from blinky_scoring.drum_scoring import load_drum_labels

    human = _human_track(tmp_path, track="other")
    with pytest.raises(ValueError, match="track"):
        load_drum_labels(human, tmp_path / "unused.json")


def test_human_labels_without_their_adtof_hats_raise(tmp_path: Path) -> None:
    """lone_hat needs the ADTOF hats; a missing ADTOF file is a missing label file, not 0 hats."""
    from blinky_scoring.drum_scoring import load_drum_labels

    human = _human_track(tmp_path)
    (tmp_path / "drums_adtof" / "t.kick_weighted.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_drum_labels(human, tmp_path / "unused.json")


def test_adtof_labels_are_not_consensus_cleaned(tmp_path: Path) -> None:
    from blinky_scoring.drum_scoring import load_drum_labels

    adtof = _write(tmp_path / "drums_adtof" / "t.kick_weighted.json", {"onsets": [
        {"time": 1.0, "type": "kick", "weight": 1.0}, {"time": 5.0, "type": "snare", "weight": 1.0},
    ]})
    labels = load_drum_labels(adtof, _write(tmp_path / "c.json", {"onsets": []}))
    assert labels.kicks == [1.0] and labels.snares == [5.0] and labels.dropped == 0


def test_device_runs_report_drums() -> None:
    start = 1_000_000.0
    gt = GroundTruth(
        pattern="t",
        duration_ms=7000,
        hits=[GroundTruthHit(time=t, type="onset", strength=1.0) for t in KICKS + SNARES],
        drums=LABELS,
    )
    data = TestData(
        duration=7000.0,
        start_time=start,
        transients=[
            # 300 ms after the labels: a box run's playback start + speaker + mic + detection.
            TransientEvent(timestamp_ms=start + t * 1000 + 300, type="onset", strength=1.0)
            for t in KICKS
        ],
    )
    summary = format_score_summary(score_device_run(data, start, gt))
    assert summary["drums"]["kickR50"] == 1.0 and summary["drums"]["snareR50"] == 0.0
    assert summary["drums"]["lagSource"] == "estimated"
    assert abs(summary["drums"]["lagMs"] - 300) < 5 and not summary["drums"]["lagAmbiguous"]
    no_drums = GroundTruth(pattern="t", duration_ms=7000, hits=gt.hits)
    assert format_score_summary(score_device_run(data, start, no_drums))["drums"] is None
