#!/usr/bin/env python3
"""Validate human drum label files (edm/drums_human/<stem>.drums.json).

Two callers use the same checks:

* the hosted reviewer, before it turns a save into a pull request;
* corpus CI, on every pull request that touches edm/drums_human/:

      python tools/drum-reviewer/labelcheck.py --base origin/main      # a PR
      python tools/drum-reviewer/labelcheck.py --all                   # main: every file's schema

A pull request that changes a human label file passes only if:

1. it changes exactly that one file, and adds or modifies it (no deletes, no renames);
2. the file matches the schema the reviewer writes (keys, types, sorted 4-decimal onsets, merged
   ranges, the audio's sha256, the manifest duration);
3. the track is 100% reviewed (`complete` is true and the ranges cover the whole track);
4. it's plausible against the file it replaces (the base branch's human file, else the seed it
   started from): it doesn't delete more than half of that file's hits, no class count lands far
   outside the range of every reference, and no class is empty while any reference has hits in it.

`--override` (CI passes it when the owner labels the PR `labels-override`) skips step 4 only.

Seed sources and their precedence come from seeds.json (see load_sources). Python standard library only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = 1
LABEL_TYPES = ("kick", "snare")
HUMAN_DIR = "drums_human"
HUMAN_SUFFIX = ".drums.json"
MATCH_TOL_S = 0.050  # two hits within +-50 ms are the same hit
GAP_TOL_S = 0.010  # reviewed ranges closer than this are merged
END_TOL_S = 0.100  # a track is complete when reviewed to within this of each end
DURATION_TOL_S = 1.0  # file duration vs track_manifest.json duration
MAX_ONSETS = 50_000
MAX_RANGES = 10_000
MAX_STARTED_FROM_BYTES = 4096
STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
SEED_NAME_RE = re.compile(r"^[a-z0-9_]{1,40}$")
HUMAN_KEYS = ("schema", "track", "audio_sha256", "duration", "started_from", "reviewed_ranges",
              "complete", "onsets", "updated")

# Plausibility (step 4). A correction moves, adds, deletes and re-classes hits; it doesn't throw most
# of them away. Deletions are counted class-agnostic, so a kick relabelled as a snare isn't a delete.
MAX_DELETED_SHARE = 0.5  # refuse deleting more than half of the replaced file's hits...
MIN_HITS_FOR_DELETE_CHECK = 8  # ...when it has at least this many
COUNT_LOW_FACTOR = 0.5  # a class count below half of every reference's...
COUNT_HIGH_FACTOR = 2.0  # ...or above twice every reference's is refused...
COUNT_SLACK = 8  # ...give or take this many hits (small counts are noisy)

DEFAULT_SEEDS_FILE = Path(__file__).resolve().parent / "seeds.json"


# --- shared label helpers (the reviewer imports these) --------------------------------------


def merge_ranges(ranges: list[list[float]], gap: float = GAP_TOL_S) -> list[list[float]]:
    rs = sorted([float(a), float(b)] for a, b in ranges if b > a)
    out: list[list[float]] = []
    for a, b in rs:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def is_complete(ranges: list[list[float]], duration: float) -> bool:
    m = merge_ranges(ranges)
    return len(m) == 1 and m[0][0] <= END_TOL_S and m[0][1] >= duration - END_TOL_S


def match_count(a: list[float], b: list[float], tol: float = MATCH_TOL_S) -> int:
    """Greedy one-to-one matches between two sorted time lists within +-tol."""
    i = j = n = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol:
            n += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return n


def f_measure(a: list[float], b: list[float]) -> float | None:
    if not a and not b:
        return None  # nothing to disagree about in this class
    return 2 * match_count(a, b) / (len(a) + len(b))


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dumps_doc(doc: dict[str, Any]) -> str:
    """JSON with one onset per line, so a git diff of a correction stays readable."""
    head = {k: v for k, v in doc.items() if k != "onsets"}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)},")
    rows = [f"    {json.dumps(o)}" for o in doc["onsets"]]
    lines.append('  "onsets": [' + ("\n" + ",\n".join(rows) + "\n  ]" if rows else "]"))
    lines.append("}")
    return "\n".join(lines) + "\n"


# --- seed sources ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedSource:
    """An automatic label set a track can be seeded from (kick_weighted schema: onsets of
    {time, type}). Precedence is the order in seeds.json; a human file always comes first."""

    name: str
    dir: str  # relative to edm/
    suffix: str
    label: str
    warning: str | None = None

    def path(self, edm: Path, stem: str) -> Path:
        return edm / self.dir / f"{stem}{self.suffix}"


def _parse_sources(path: Path, entries: Any, key: str) -> list[SeedSource]:
    if not isinstance(entries, list):
        raise ValueError(f"{path}: '{key}' must be a list")
    out: list[SeedSource] = []
    for s in entries:
        if not isinstance(s, dict):
            raise ValueError(f"{path}: {key} entries must be objects, got {s!r}")
        name, d, suffix = s.get("name"), s.get("dir"), s.get("suffix")
        if not isinstance(name, str) or not SEED_NAME_RE.match(name) or name in ("human", "empty"):
            raise ValueError(f"{path}: bad {key} name {name!r}")
        if not isinstance(d, str) or not STEM_RE.match(d) or ".." in d or d == HUMAN_DIR:
            raise ValueError(f"{path}: {key} {name}: bad dir {d!r} (one directory under edm/)")
        if not isinstance(suffix, str) or not re.fullmatch(r"\.[A-Za-z0-9._-]{1,60}", suffix):
            raise ValueError(f"{path}: {key} {name}: bad suffix {suffix!r}")
        label = s.get("label", name)
        warning = s.get("warning")
        if not isinstance(label, str) or (warning is not None and not isinstance(warning, str)):
            raise ValueError(f"{path}: {key} {name}: label/warning must be strings")
        out.append(SeedSource(name, d, suffix, label, warning))
    return out


def load_sources(path: Path | None = None) -> tuple[list[SeedSource], list[SeedSource]]:
    """(seeds, references) from seeds.json. Seeds are in precedence order. References
    ("references", optional) are reference-only label sets: the page offers them as ghost ticks,
    but they never seed a track and the label checks don't compare against them."""
    path = Path(path or DEFAULT_SEEDS_FILE)
    doc = json.loads(path.read_text())
    seeds = doc.get("seeds") if isinstance(doc, dict) else None
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path}: expected a non-empty 'seeds' list")
    out = _parse_sources(path, seeds, "seed")
    refs = _parse_sources(path, doc.get("references", []), "reference")
    if len({s.name for s in out + refs}) != len(out) + len(refs):
        raise ValueError(f"{path}: duplicate seed names")
    return out, refs


def load_seeds(path: Path | None = None) -> list[SeedSource]:
    return load_sources(path)[0]


def split_auto_labels(doc: dict[str, Any], name: str) -> dict[str, Any]:
    """kick_weighted-schema onsets -> {"kick": [...], "snare": [...], "hihat": [...], "other": n}."""
    out: dict[str, Any] = {"kick": [], "snare": [], "hihat": [], "other": 0}
    onsets = doc.get("onsets")
    if not isinstance(onsets, list):
        raise ValueError(f"{name}: no 'onsets' list")
    for o in onsets:
        if not isinstance(o, dict) or not finite(o.get("time")):
            raise ValueError(f"{name}: malformed onset {o!r}")
        t = o.get("type")
        if t in ("kick", "snare", "hihat"):
            out[t].append(float(o["time"]))
        else:
            out["other"] += 1
    for k in ("kick", "snare", "hihat"):
        out[k].sort()
    return out


def seed_labels(edm: Path, stem: str, seeds: list[SeedSource]) -> dict[str, dict[str, Any]]:
    """{seed name: split labels} for every seed file of this track that exists and isn't skipped,
    in precedence order."""
    out: dict[str, dict[str, Any]] = {}
    for s in seeds:
        p = s.path(edm, stem)
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        if isinstance(doc, dict) and doc.get("skipped"):
            continue
        out[s.name] = split_auto_labels(doc, p.name)
    return out


def human_labels(doc: dict[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {t: [] for t in LABEL_TYPES}
    for o in doc.get("onsets") or []:
        if isinstance(o, dict) and o.get("type") in LABEL_TYPES and finite(o.get("time")):
            out[o["type"]].append(float(o["time"]))
    for t in LABEL_TYPES:
        out[t].sort()
    return out


# --- the checks --------------------------------------------------------------------------------


def manifest_duration(edm: Path, stem: str) -> float | None:
    path = edm / "track_manifest.json"
    if not path.is_file():
        return None
    entry = json.loads(path.read_text()).get(stem)
    if isinstance(entry, dict) and finite(entry.get("duration")):
        return float(entry["duration"])
    return None


def check_schema(doc: Any, stem: str, audio_sha256: str, mdur: float | None) -> list[str]:
    """Problems with a human label file's structure; [] when it is exactly what the reviewer writes."""
    if not isinstance(doc, dict):
        return ["the file is not a JSON object"]
    p: list[str] = []
    missing = [k for k in HUMAN_KEYS if k not in doc]
    extra = sorted(set(doc) - set(HUMAN_KEYS))
    if missing:
        p.append(f"missing keys: {', '.join(missing)}")
    if extra:
        p.append(f"unknown keys: {', '.join(extra)}")
    if missing:
        return p
    if doc["schema"] != SCHEMA:
        p.append(f"schema must be {SCHEMA}, got {doc['schema']!r}")
    if doc["track"] != stem:
        p.append(f"track is {doc['track']!r} but the file is for {stem!r}")
    if doc["audio_sha256"] != audio_sha256:
        p.append(f"audio_sha256 {str(doc['audio_sha256'])[:12]}... doesn't match edm/{stem}.mp3 "
                 f"({audio_sha256[:12]}...)")
    dur = doc["duration"]
    if not finite(dur) or dur <= 0:
        p.append(f"duration must be a positive number, got {dur!r}")
        return p
    if mdur is not None and abs(mdur - dur) > DURATION_TOL_S:
        p.append(f"duration {dur:.3f}s disagrees with track_manifest.json ({mdur:.3f}s)")
    sf = doc["started_from"]
    if sf is not None and not isinstance(sf, (dict, str)):
        p.append("started_from must be an object")
    elif len(json.dumps(sf)) > MAX_STARTED_FROM_BYTES:
        p.append("started_from is too large")
    if not isinstance(doc["updated"], str):
        p.append("updated must be an ISO 8601 timestamp")
    else:
        try:
            _dt.datetime.fromisoformat(doc["updated"])
        except ValueError:
            p.append(f"updated is not an ISO 8601 timestamp: {doc['updated']!r}")

    ranges = doc["reviewed_ranges"]
    if not isinstance(ranges, list) or len(ranges) > MAX_RANGES:
        p.append("reviewed_ranges must be a list")
        ranges = []
    for r in ranges:
        if (not isinstance(r, list) or len(r) != 2 or not all(finite(x) for x in r)
                or not (0 <= r[0] < r[1] <= dur + END_TOL_S)):
            p.append(f"bad reviewed range {r!r}")
            ranges = []
            break
    if ranges and merge_ranges(ranges) != [[float(a), float(b)] for a, b in ranges]:
        p.append("reviewed_ranges must be sorted and merged")
    if not isinstance(doc["complete"], bool):
        p.append("complete must be true or false")
    elif ranges is not None and doc["complete"] != is_complete(ranges, float(dur)):
        p.append(f"complete is {doc['complete']} but the reviewed ranges say "
                 f"{is_complete(ranges, float(dur))}")

    onsets = doc["onsets"]
    if not isinstance(onsets, list) or len(onsets) > MAX_ONSETS:
        return p + ["onsets must be a list"]
    prev: tuple[float, str] | None = None
    for o in onsets:
        if not isinstance(o, dict) or set(o) != {"time", "type"}:
            p.append(f"malformed onset {o!r} (exactly time and type)")
            break
        t, ty = o["time"], o["type"]
        if ty not in LABEL_TYPES:
            p.append(f"onset type must be kick or snare, got {ty!r}")
            break
        if not finite(t) or t < 0 or t > dur + END_TOL_S:
            p.append(f"onset time out of range: {t!r}")
            break
        if round(float(t), 4) != t:
            p.append(f"onset time {t!r} has more than 4 decimals")
            break
        key = (float(t), ty)
        if prev is not None and key <= prev:
            p.append(f"onsets must be sorted by time with no duplicates (at {t})")
            break
        prev = key
    return p


def check_complete(doc: dict[str, Any]) -> list[str]:
    ranges = doc.get("reviewed_ranges") or []
    dur = doc.get("duration")
    if doc.get("complete") is not True or not finite(dur) or not is_complete(ranges, float(dur)):
        share = 0.0
        if finite(dur) and dur > 0:
            share = sum(b - a for a, b in merge_ranges(ranges)) / dur
        return [f"the track is not 100% reviewed ({100 * share:.1f}%): only complete tracks are "
                "accepted"]
    return []


def plausibility(new: dict[str, list[float]], replaced: tuple[str, dict[str, list[float]]] | None,
                 references: dict[str, dict[str, list[float]]]) -> list[str]:
    """Problems with how far `new` moved from what it replaces. `replaced` is (name, labels) of the
    file it replaces; `references` is every label set the counts are compared against (the replaced
    file is added to it)."""
    p: list[str] = []
    refs = dict(references)
    if replaced is not None:
        refs.setdefault(replaced[0], replaced[1])
        name, old = replaced
        old_all = sorted(t for ty in LABEL_TYPES for t in old.get(ty, []))
        new_all = sorted(t for ty in LABEL_TYPES for t in new.get(ty, []))
        if len(old_all) >= MIN_HITS_FOR_DELETE_CHECK:
            deleted = len(old_all) - match_count(old_all, new_all)
            if deleted > MAX_DELETED_SHARE * len(old_all):
                p.append(f"deletes {deleted} of the {len(old_all)} hits in {name} "
                         f"({100 * deleted / len(old_all):.0f}%; the limit is "
                         f"{100 * MAX_DELETED_SHARE:.0f}%)")
    if not refs:
        return p
    for ty in LABEL_TYPES:
        counts = [len(r.get(ty, [])) for r in refs.values()]
        n, lo, hi = len(new.get(ty, [])), min(counts), max(counts)
        if n == 0 and hi > 0:
            some = [name for name, r in refs.items() if r.get(ty)]
            p.append(f"no {ty} hits, but {', '.join(some)} {'has' if len(some) == 1 else 'have'} some")
        elif n < COUNT_LOW_FACTOR * lo - COUNT_SLACK or n > COUNT_HIGH_FACTOR * hi + COUNT_SLACK:
            p.append(f"{n} {ty} hits is far outside the references' range {lo}-{hi} "
                     f"({', '.join(refs)})")
    return p


def replaced_labels(doc: dict[str, Any], base_doc: dict[str, Any] | None,
                    seeds: dict[str, dict[str, Any]]) -> tuple[str, dict[str, list[float]]] | None:
    """What a new human file replaces: the existing human file, else the seed it says it started
    from, else the best seed available."""
    if base_doc is not None:
        return "the existing human file", human_labels(base_doc)
    sf = doc.get("started_from")
    src = sf.get("source") if isinstance(sf, dict) else sf if isinstance(sf, str) else None
    if src in seeds:
        return src, {t: seeds[src][t] for t in LABEL_TYPES}
    for name, lab in seeds.items():
        return name, {t: lab[t] for t in LABEL_TYPES}
    return None


def check_human_doc(edm: Path, stem: str, doc: Any, base_doc: dict[str, Any] | None,
                    seeds: list[SeedSource], *, require_complete: bool = True,
                    check_plausible: bool = True, audio_sha256: str | None = None) -> list[str]:
    """Every check for one human label file. `base_doc` is the human file it replaces (None if new)."""
    sha = audio_sha256 or file_sha256(edm / f"{stem}.mp3")
    p = check_schema(doc, stem, sha, manifest_duration(edm, stem))
    if p:
        return p
    if require_complete:
        p += check_complete(doc)
    if check_plausible:
        avail = seed_labels(edm, stem, seeds)
        refs = {n: {t: lab[t] for t in LABEL_TYPES} for n, lab in avail.items()}
        p += plausibility(human_labels(doc), replaced_labels(doc, base_doc, avail), refs)
    return p


# --- CI: the pull request's changed files ---------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def changed_files(repo: Path, base: str) -> tuple[str, list[tuple[str, str]]]:
    mb = _git(repo, "merge-base", base, "HEAD").strip()
    out = _git(repo, "-c", "core.quotePath=false", "diff", "--name-status", "--no-renames", mb, "HEAD")
    rows = [tuple(line.split("\t", 1)) for line in out.splitlines() if line.strip()]
    return mb, [(s, path) for s, path in rows]  # type: ignore[misc]


def check_pr(repo: Path, base: str, seeds_file: Path | None, override: bool) -> list[str]:
    edm = repo / "edm"
    mb, changes = changed_files(repo, base)
    label_changes = [(s, f) for s, f in changes if f.startswith(f"edm/{HUMAN_DIR}/")]
    if not label_changes:
        print("no human label files changed")
        return []
    if len(changes) != 1:
        listing = ", ".join(f"{s} {f}" for s, f in changes)
        return [f"a pull request that changes a human label file must change only that one file; "
                f"this one changes {len(changes)}: {listing}"]
    status, path = changes[0]
    m = re.fullmatch(rf"edm/{HUMAN_DIR}/([^/]+){re.escape(HUMAN_SUFFIX)}", path)
    if not m or not STEM_RE.match(m.group(1)):
        return [f"{path} is not a human label file (edm/{HUMAN_DIR}/<track>{HUMAN_SUFFIX})"]
    if status not in ("A", "M"):
        return [f"{path}: status {status}; a human label file may only be added or modified"]
    stem = m.group(1)
    if not (edm / f"{stem}.mp3").is_file():
        return [f"{path}: no such track edm/{stem}.mp3"]
    try:
        doc = json.loads((repo / path).read_text())
    except json.JSONDecodeError as e:
        return [f"{path} is not valid JSON: {e}"]
    base_doc = None
    try:
        base_doc = json.loads(_git(repo, "show", f"{mb}:{path}"))
    except subprocess.CalledProcessError:
        pass  # a new file: it replaces a seed
    seeds = load_seeds(seeds_file or repo / "tools" / "drum-reviewer" / "seeds.json")
    p = check_human_doc(edm, stem, doc, base_doc, seeds, check_plausible=not override)
    if override and not p:
        print("labels-override: plausibility checks skipped by the owner's label")
    if not p:
        print(f"{path}: valid, 100% reviewed" + ("" if override else ", plausible"))
    return [f"{path}: {x}" for x in p]


def check_all(repo: Path) -> list[str]:
    """Every human label file on a branch: schema only (partial files are allowed on main)."""
    edm = repo / "edm"
    p: list[str] = []
    files = sorted((edm / HUMAN_DIR).glob(f"*{HUMAN_SUFFIX}"))
    for f in files:
        stem = f.name[: -len(HUMAN_SUFFIX)]
        if not (edm / f"{stem}.mp3").is_file():
            p.append(f"{f.name}: no such track")
            continue
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            p.append(f"{f.name}: not valid JSON: {e}")
            continue
        p += [f"{f.name}: {x}" for x in check_schema(doc, stem, file_sha256(edm / f"{stem}.mp3"),
                                                     manifest_duration(edm, stem))]
    print(f"{len(files)} human label files checked")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="the PR's base ref (e.g. origin/main): check the PR's changes")
    g.add_argument("--all", action="store_true", help="schema-check every human label file")
    ap.add_argument("--repo", type=Path, default=Path("."), help="the corpus checkout (default .)")
    ap.add_argument("--seeds", type=Path, help="seeds.json (default: the repo's)")
    ap.add_argument("--override", action="store_true",
                    help="skip the plausibility checks (the owner's labels-override PR label)")
    args = ap.parse_args(argv)
    problems = check_all(args.repo) if args.all else check_pr(args.repo, args.base, args.seeds,
                                                              args.override)
    for x in problems:
        print(f"::error::{x}" if "GITHUB_ACTIONS" in __import__("os").environ else x,
              file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
