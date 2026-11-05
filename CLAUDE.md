# Claude Code Instructions for blinky-corpus

Critical rules only. What the corpus is and how it's consumed: `README.md`. The firmware lives in [blinky-firmware](https://github.com/Jdubz/blinky-firmware) and the tooling in [blinky_time](https://github.com/Jdubz/blinky_time); each has its own `CLAUDE.md`.

## Changes reach consumers only through a pin bump

blinky-firmware (its corpus regression gate) and blinky_time (blinky-server's validation suite, ml-training) pin this repo as their `corpus/` submodule. A merge to `main` changes nothing for them until Dependabot opens a bump PR in each, and that PR's CI passes. A change that breaks a consumer shows up on the bump PR, so changes that a consumer must adopt in step (a renamed function, a new label file it requires) need a paired PR there.

PRs go to `main` with CI green and the review addressed.

## The only validation corpus is `edm/` (18 tracks)

Representative EDM with clear percussion. Lock in performance here before measuring anything else. **Never add a "held-out" / "hard" / "adversarial" set as a default or a headline metric**: it normalises low scores and has burned weeks before (`edm_holdout/`, deleted 2026-04-27). Harder material, if ever, is a separate tier-2 number, never a default in scripts.

## The task is drum hits, not beats

Detectors are scored on **kicks and snares**: recall at ±30/±50 ms and the share of detections that land on a kick or snare, against cleaned `kick_weighted` labels. All-onset F1 is secondary at most. Ground truth is onset labels (`*.onsets_consensus.json`, `kick_weighted/`), never beat-tracker output (`consensus_v*`, allin1, beat_this, beatnet, demucs_beats): those teach a detector to suppress off-beat hits.

## Human edits are ground truth

`*.onsets_human.json` overlays (from blinky_time's label reviewer) are the final truth and must flow through scoring. `onset_label_merge.apply_human_edits()` refuses an overlay whose `source_count` no longer matches the auto labels (`HumanEditDriftError`). Never regenerate auto labels under an overlay, and never weaken that check.

## No silent fallbacks

Errors fail loudly. A missing label file, an empty track, a malformed overlay or a zero denominator that means a bug raises; it's never scored as 0 or skipped quietly. Where 0 is the correct answer (no detections on a silent track), say so in a comment.
