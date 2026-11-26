# tools/labels

The label checks for `edm/drums_human/`, and the seed sources' precedence. The reviewer that writes these files is the labels service in blinky_time, [`labels-service/`](https://github.com/Jdubz/blinky_time/tree/master/labels-service); its page is https://blinky-time.joshwentworth.com/labels/.

## labelcheck.py: anti-vandalism, not process

The labels service holds no credential. `.github/workflows/labels-sync.yml` runs on a schedule (main's own code) and runs `sync.py`. It pulls the service's export, treats it as untrusted, writes only label files that pass these checks onto a `labels/auto-*` branch made from `main`, starts `tests.yml` on it and auto-merges the PR. The gate is this script, run by `sync.py` and again in the required **Scoring tests** job. Every changed human file must:

1. be added or modified, never renamed, and deleted only on an ordinary PR with the owner's `labels-override` label (reverting a batch that added a track's first human file);
2. match the schema: keys, types, sorted 4-decimal onsets, merged reviewed ranges, the mp3's sha256 and the manifest duration;
3. not be vandalism against what it replaces (`main`'s human file, else the seed it started from):
   - it may not delete more than half of that file's hits (counted regardless of class, so re-classing isn't deleting);
   - it may not leave a class empty while a reference has hits in it.

A `labels/` branch may change nothing but human label files (`--only-labels`). A file that fails is left out of the batch, and a batch that fails CI stays open for the owner.

Partial progress is fine: the scorer uses a human file only once it's `complete`. The owner's `labels-override` PR label skips step 3 on ordinary PRs. Git history is the undo: revert on `main`, and the labels service takes `main`'s version.

```sh
python tools/labels/labelcheck.py --base origin/main [--only-labels]   # the changes against main
python tools/labels/labelcheck.py --all                                # every file's schema
```

The same file is blinky_time's `labels-service/labelcheck.py`, which the service runs before it publishes. Keep the two identical.

## seeds.json

This is the seed precedence. A track's labels start from its human file, else the first source here that has a file for it. The `references` sources are ghost ticks only. To add a better seed, add its directory under `edm/` and put its entry first; the labels service reads this file from `main`.
