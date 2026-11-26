# Claude Code Instructions for blinky-corpus

Critical rules only. What the corpus is and how it's consumed: `README.md`. The firmware lives in [blinky-firmware](https://github.com/Jdubz/blinky-firmware) and the tooling in [blinky_time](https://github.com/Jdubz/blinky_time); each has its own `CLAUDE.md`.

## Changes reach consumers only through a pin bump

blinky-firmware (its corpus regression gate) and blinky_time (blinky-server's validation suite, ml-training) pin this repo as their `corpus/` submodule. A merge to `main` changes nothing for them until a PR bumps their `corpus` pin (an ordinary PR, not Dependabot) and its CI passes. A change that breaks a consumer shows up on the bump PR, so changes that a consumer must adopt in step (a renamed function, a new label file it requires) need a paired PR there.

PRs go to `main` with CI green and the review addressed.

## The only validation corpus is `edm/` (18 tracks)

Representative EDM with clear percussion. Lock in performance here before measuring anything else. **Never add a "held-out" / "hard" / "adversarial" set as a default or a headline metric**: it normalises low scores and has burned weeks before (`edm_holdout/`, deleted 2026-04-27). Harder material, if ever, is a separate tier-2 number, never a default in scripts.

## The task is drum hits, not beats

Detectors are scored on **kicks and snares**: recall at ±30/±50 ms and the share of detections that land on a kick or snare, against the best drum labels a track has (`blinky_scoring.drum_scoring.drum_label_path`): complete hand-curated ground truth in `edm/drums_human/` (#4), else ADTOF in `edm/drums_adtof/`; the band-generator `kick_weighted/` is legacy. All-onset F1 is secondary at most. Ground truth is onset labels (`*.onsets_consensus.json`, `drums_human/`, `drums_adtof/`), never beat-tracker output (`consensus_v*`, allin1, beat_this, beatnet, demucs_beats): those teach a detector to suppress off-beat hits.

## Human edits are ground truth

Human drum labels (`edm/drums_human/`) arrive from the labels service (blinky_time `labels-service/`), which holds no credential: the scheduled `labels-sync.yml` pulls its export, checks it with main's `tools/labels/labelcheck.py`, commits only label files as a `labels/auto-*` batch, starts `tests.yml` on it and auto-merges it. The gate is those label checks (the schema, no mass deletion, no empty class, and a `labels/` branch changes only label files), also in the required **Scoring tests** job. Those batches merge themselves, so never weaken the checks, give anything else write access to `labels/auto-*` branches (the dispatched run trusts their workflows because only labels-sync writes them), or add a write credential to the labels service. `tools/labels/labelcheck.py` and blinky_time's `labels-service/labelcheck.py` are the same file: change both.

`*.onsets_human.json` overlays (from blinky_time's label reviewer) are the final truth and must flow through scoring. `onset_label_merge.apply_human_edits()` refuses an overlay whose `source_count` no longer matches the auto labels (`HumanEditDriftError`). Never regenerate auto labels under an overlay, and never weaken that check.

## No silent fallbacks

Errors fail loudly. A missing label file, an empty track, a malformed overlay or a zero denominator that means a bug raises; it's never scored as 0 or skipped quietly. Where 0 is the correct answer (no detections on a silent track), say so in a comment.

## No network topology

The repo is public: no host names, private or tailnet addresses, or mDNS names, in code, docs or tests. Name machines by role (the fleet server, the test box, a workstation); examples use `192.0.2.x` and `example.invalid`. Host config lives outside the repo in `~/.config/blinky/` (keys in blinky_time's CLAUDE.md). `scripts/check_topology.py` runs in CI; a justified exception goes in `.topology-allowlist` with its reason.
