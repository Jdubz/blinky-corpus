# blinky-corpus

The validation corpus for blinky, and the code that scores a run against it.

- `edm/`: 18 representative EDM tracks (`*.mp3`) with their labels: `*.onsets_consensus.json` (onsets), `*.pattern_gt.json`, `kick_weighted/` (drum labels).
- `blinky_scoring/`: scoring a detector run against those labels. It covers onset tracking, kick/snare drum metrics (`drum_scoring.py`, against cleaned `kick_weighted` labels), rhythm, and label loading (`track_discovery.py`, `onset_label_merge.py`). Pure standard library.

It's the only validation corpus. Firmware tuning, blinky-firmware's corpus regression gate and blinky-server's validation suite all measure against it. Lock in performance here before measuring anything else, and never add a "held-out" or adversarial set as a headline number. Detectors are scored on kicks and snares, not on all onsets or beats.

Used as a git submodule at `corpus/` in [blinky-firmware](https://github.com/Jdubz/blinky-firmware) and [blinky_time](https://github.com/Jdubz/blinky_time). Install it with `pip install -e corpus` and import `blinky_scoring`. A change to scoring or labels is a PR here, then a submodule bump in each consumer.

History comes from blinky_time's `blinky-test-player/music/edm/` and `blinky-server/blinky_server/testing/` (git filter-repo, 2026-11-04, Jdubz/blinky_time#357).
