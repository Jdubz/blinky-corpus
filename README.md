# blinky-corpus

The validation corpus for blinky, and the code that scores a run against it.

- `edm/`: 18 representative EDM tracks (`*.mp3`) with their labels: `*.onsets_consensus.json` (onsets), `*.pattern_gt.json`, `kick_weighted/` (drum labels from the old band-onset generator, known bad), `drums_adtof/` (ADTOF drum labels, the starting point for hand-curated ground truth: #4).
- `edm/drums_human/<stem>.drums.json`: hand-curated kick/snare ground truth (#4), written by `tools/drum-reviewer/`. The complete list, not a diff: `{"schema": 1, "track", "audio_sha256" (of the mp3), "duration", "started_from" (seed provenance), "reviewed_ranges": [[t0, t1], ...], "complete", "onsets": [{"time": s, "type": "kick"|"snare"}], "updated"}`. The scorer uses it once `complete` is true. Full schema and workflow: [`tools/drum-reviewer/README.md`](tools/drum-reviewer/README.md).
- `tools/drum-reviewer/`: the reviewer for those files (`python tools/drum-reviewer/server.py`, then open http://localhost:8766).
- `archive/beats_gt/`: the old beat ground truth (`*.beats.json`) for the same tracks, archived 2026-04-29 when the task became drum hits. Never an eval target or a training label.
- `blinky_scoring/`: scoring a detector run against those labels. It covers onset tracking, kick/snare drum metrics (`drum_scoring.py`, against complete `drums_human/` ground truth, else `drums_adtof/`), rhythm, and label loading (`track_discovery.py`, `onset_label_merge.py`). Pure standard library.

It's the only validation corpus. Firmware tuning, blinky-firmware's corpus regression gate and blinky-server's validation suite all measure against it. Lock in performance here before measuring anything else, and never add a "held-out" or adversarial set as a headline number. Detectors are scored on kicks and snares, not on all onsets or beats.

Used as a git submodule at `corpus/` in [blinky-firmware](https://github.com/Jdubz/blinky-firmware) and [blinky_time](https://github.com/Jdubz/blinky_time). Install it with `pip install -e corpus` and import `blinky_scoring`. A change to scoring or labels is a PR here, then a submodule bump in each consumer.

History comes from blinky_time's `blinky-test-player/music/edm/` and `blinky-server/blinky_server/testing/` (git filter-repo, 2026-11-04, Jdubz/blinky_time#357).
