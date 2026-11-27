#!/usr/bin/env python3
"""The repS golden files (tests/golden/repetition/), written by blinky-firmware's consistency.py itself.

tests/test_repetition.py holds this repo's two copies of it to these files: scripts/repetition.py's
find_repetition (the maps) and blinky_scoring/repetition.py's consistency (the scorer). The repos
share no code, so consistency.py is read from a blinky-firmware checkout given on the command line,
only here, and only to write these files. Regenerate them only when consistency.py changes (and
change both copies to match it).

    python scripts/repetition_golden.py --consistency ../blinky-firmware/tests/tracker/consistency.py \
        --firmware-sha <sha> [--tracks breakbeat-drive techno-minimal-01]

Each file: the track's lag and regions from consistency.find_repetition, and consistency.consistency's
result on three detection lists: the track's consensus onsets (just a realistic list of event times
and strengths; nothing here is judged against labels) over the window the test box plays, the same
with every third one dropped and the strengths jittered, and over the whole track after the
firmware eval's 10 s warm-up. Needs numpy and ffmpeg.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from blinky_scoring.repetition import regions_of  # noqa: E402
from repetition import decode, samples  # noqa: E402


def dump(doc: dict) -> str:
    """One line per region and per case."""
    head = {k: v for k, v in doc.items() if k not in ("regions", "cases")}
    out = json.dumps(head, indent=1, allow_nan=False)[:-2]
    for k in ("regions", "cases"):
        out += f',\n "{k}": [\n  ' + ",\n  ".join(json.dumps(v, allow_nan=False) for v in doc[k]) + "\n ]"
    return out + "\n}\n"


BOX_WINDOW_S = 35.0  # blinky-server's staging suite plays validation_ms = 35 s of each track
WARMUP_S = 10.0  # blinky-firmware run_tracker_eval.py


def nan_none(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consistency", type=Path, required=True)
    ap.add_argument("--firmware-sha", required=True, help="the blinky-firmware commit it came from")
    ap.add_argument("--tracks", nargs="+", default=["breakbeat-drive", "techno-minimal-01"])
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "golden" / "repetition")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("consistency", args.consistency)
    c = importlib.util.module_from_spec(spec)
    sys.modules["consistency"] = c  # its dataclass looks its module up
    spec.loader.exec_module(c)
    manifest = json.loads((ROOT / "edm" / "track_manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for stem in args.tracks:
        mp3 = ROOT / "edm" / f"{stem}.mp3"
        pcm = decode(mp3)
        x = samples(pcm)
        rep = c.find_repetition(x)
        onsets = sorted(json.loads((ROOT / "edm" / f"{stem}.onsets_consensus.json").read_text())["onsets"],
                        key=lambda o: o["time"])
        seek = float(manifest[stem]["seekOffset"])
        end = len(x) / c.SR
        rng = random.Random(stem)
        cases = []
        for name, t0, t1, thin in (("box window", seek, seek + BOX_WINDOW_S, False),
                                   ("box window, flaky", seek, seek + BOX_WINDOW_S, True),
                                   ("whole track", WARMUP_S, end, False)):
            sel = [o for o in onsets if t0 <= o["time"] <= t1]
            if thin:
                sel = [o for k, o in enumerate(sel) if k % 3 != 2]
            times = [round(o["time"], 4) for o in sel]
            strengths = [round(min(1.0, o["strength"] * (rng.uniform(0.6, 1.0) if thin else 1.0)), 4)
                         for o in sel]
            res = c.consistency(times, strengths, rep, t0, t1)
            res = {k: int(v) if k == "n" else float(v) for k, v in res.items()}
            cases.append({"name": name, "t_start": t0, "t_end": t1, "times": times,
                          "strengths": strengths, "result": nan_none(res)})
            print(f"{stem}: {name}: {cases[-1]['result']}")
        doc = {
            "source": f"blinky-firmware tests/tracker/consistency.py at {args.firmware_sha}",
            "track": stem,
            "audio_sha256": hashlib.sha256(mp3.read_bytes()).hexdigest(),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "lag_s": float(rep.lag_s),
            "frame_s": float(rep.frame_s),
            "n_frames": int(len(rep.mask)),
            "regions": regions_of(rep.mask.tolist()),
            "cases": cases,
        }
        (args.out / f"{stem}.json").write_text(dump(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
