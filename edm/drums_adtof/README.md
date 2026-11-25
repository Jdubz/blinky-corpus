# ADTOF drum labels for edm/

Kick, snare and hi-hat hits for the 18 edm/ mixes from ADTOF (Zehren et al., github.com/MZehren/ADTOF; model and weights CC BY-NC-SA 4.0), run through the pinned ADTOF-pytorch port that blinky_time's `ml-training/scripts/adtof_labels.py` uses. The files use the `kick_weighted` schema. Hi-hats have weight 0 and are not scored. `PROVENANCE.json` records the generator, the corpus commit that was labelled, and the workflow run (blinky_time `edm-drum-labels.yml`, CPU, GitHub runner).

These are **automatic** labels. They are the starting point for hand-curated ground truth (Jdubz/blinky-corpus#4), not the ground truth itself.

On blinky_time's drum label suite (`ml-training/scripts/drum_label_adtof.json` and `drum_label_baseline.json`), ADTOF scores:
- kick F1 0.93 and snare F1 1.00 on synthetic drums, against 0.66 and 0.16 for the band-onset generator behind `../kick_weighted/`;
- 0 snares labelled as kicks, against 106 of 120.

The scorer still reads `../kick_weighted/` until it is switched in its own PR.
