#!/usr/bin/env python3
"""Drum-hit ground-truth reviewer: hand-curate kick/snare labels for the edm/ corpus.

Serves one page (index.html, no external fetches) and a small JSON API. All audio analysis
(band envelopes, onset strength, spectrogram) runs in the browser; this server only lists tracks,
seeds labels and takes saves.

Local mode (one person, a corpus clone):

    python tools/drum-reviewer/server.py [--port 8766] [--host 127.0.0.1] [--edm edm/]

    Save writes edm/drums_human/<stem>.drums.json into the clone; drafts go to
    edm/drums_human/.drafts/ (gitignored).

Hosted mode (many people, a public URL; see README.md "Hosted"):

    REVIEWER_GITHUB_TOKEN=... python tools/drum-reviewer/server.py --host 0.0.0.0 \\
        --state /state --checkout /corpus --git-url https://github.com/<owner>/blinky-corpus.git \\
        --github-repo <owner>/blinky-corpus --allow-host <public name> --trust-cf-connecting-ip

    The corpus is the service's own clone, refreshed from GitHub. Edit locks, drafts (per track
    and contributor), pending saves and logs live in --state. A save of a 100%-reviewed track
    becomes a pull request with auto-merge on; nothing is written into the clone.

Files (paths relative to edm/):
    <stem>.mp3                          the audio
    drums_human/<stem>.drums.json       human ground truth
    <seed dir>/<stem><suffix>           seed labels, in the precedence of seeds.json (read-only)

Seed order when a track opens: human file > the seeds.json sources, in order.
Python standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs, unquote, urlparse

STATIC_DIR = Path(__file__).resolve().parent
if str(STATIC_DIR) not in sys.path:  # labelcheck.py and hosted.py sit next to this file
    sys.path.insert(0, str(STATIC_DIR))

import hosted  # noqa: E402
import labelcheck  # noqa: E402
from labelcheck import (  # noqa: E402
    DURATION_TOL_S, END_TOL_S, LABEL_TYPES, MAX_ONSETS, MAX_RANGES, SCHEMA, STEM_RE, dumps_doc,
    f_measure, file_sha256, is_complete, merge_ranges,
)

DEFAULT_EDM = STATIC_DIR.parents[1] / "edm"
MAX_POST_BYTES = 2 * 1024 * 1024  # a 1000-hit track is ~50 KB

# Hosted mode: token buckets per client address (capacity, per seconds), and one global bucket for
# saves, since each one opens a pull request.
DEFAULT_LIMITS: dict[str, tuple[int, float]] = {
    "request": (600, 300.0),  # any request: 600 per 5 min
    "lock": (30, 3600.0),  # taking an edit lock (renewals are free): 30 per hour
    "draft": (240, 600.0),  # autosaves: 240 per 10 min
    "save": (6, 3600.0),  # completed saves (pull requests): 6 per hour
    "save_global": (30, 3600.0),  # all clients together: 30 pull requests per hour
}


class ApiError(Exception):
    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


# --- Corpus layout ----------------------------------------------------------


class Corpus:
    def __init__(self, edm: Path, seeds_file: Path | None = None):
        self.edm = Path(edm).resolve()
        self.human_dir = self.edm / labelcheck.HUMAN_DIR
        self.draft_dir = self.human_dir / ".drafts"
        self._seeds_file = seeds_file
        self._seeds: tuple[float, list[labelcheck.SeedSource]] | None = None
        self._sha_cache: dict[str, tuple[float, int, str]] = {}
        self._lock = threading.Lock()

    def seeds_file(self) -> Path:
        """The corpus's own seeds.json when it has one (a new seed source takes effect when it's
        merged), else the one next to this file."""
        if self._seeds_file is not None:
            return self._seeds_file
        own = self.edm.parent / "tools" / "drum-reviewer" / "seeds.json"
        return own if own.is_file() else labelcheck.DEFAULT_SEEDS_FILE

    def seeds(self) -> list[labelcheck.SeedSource]:
        f = self.seeds_file()
        mtime = f.stat().st_mtime
        with self._lock:
            if self._seeds is None or self._seeds[0] != mtime:
                self._seeds = (mtime, labelcheck.load_seeds(f))
            return self._seeds[1]

    # Every path built from a request goes through validate_stem() first, and the
    # result is checked to resolve inside its directory (belt and braces).
    def validate_stem(self, stem: Any) -> str:
        if not isinstance(stem, str) or not STEM_RE.match(stem) or ".." in stem:
            raise ApiError(400, f"invalid track name: {stem!r}")
        if not (self.edm / f"{stem}.mp3").is_file():
            raise ApiError(404, f"no such track: {stem}")
        return stem

    def _inside(self, path: Path, parent: Path) -> Path:
        if path.parent.resolve() != parent.resolve() or Path(path.name).name != path.name:
            raise ApiError(400, "path escapes its directory")
        return path

    def audio_path(self, stem: str) -> Path:
        return self._inside(self.edm / f"{stem}.mp3", self.edm)

    def human_path(self, stem: str) -> Path:
        return self._inside(self.human_dir / f"{stem}{labelcheck.HUMAN_SUFFIX}", self.human_dir)

    def draft_path(self, stem: str) -> Path:
        return self._inside(self.draft_dir / f"{stem}{labelcheck.HUMAN_SUFFIX}", self.draft_dir)

    def seed_path(self, seed: labelcheck.SeedSource, stem: str) -> Path:
        return self._inside(seed.path(self.edm, stem), self.edm / seed.dir)

    def stems(self) -> list[str]:
        return sorted(p.stem for p in self.edm.glob("*.mp3") if STEM_RE.match(p.stem))

    def audio_sha256(self, stem: str) -> str:
        path = self.audio_path(stem)
        st = path.stat()
        with self._lock:
            hit = self._sha_cache.get(stem)
            if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
                return hit[2]
        sha = file_sha256(path)
        with self._lock:
            self._sha_cache[stem] = (st.st_mtime, st.st_size, sha)
        return sha

    def manifest_duration(self, stem: str) -> float | None:
        return labelcheck.manifest_duration(self.edm, stem)


def now_iso() -> str:
    return hosted.now_iso()


def read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ApiError(500, f"{path.name} is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ApiError(500, f"{path.name}: expected a JSON object")
    return doc


# --- Labels -----------------------------------------------------------------


def split_auto_labels(doc: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return labelcheck.split_auto_labels(doc, name)
    except ValueError as e:
        raise ApiError(500, str(e)) from e


def provenance(path: Path, source: str, doc: dict[str, Any], rel_root: Path) -> dict[str, Any]:
    gen = doc.get("generator")
    if gen is None and "source" in doc:  # kick_weighted files name their detector here
        gen = {"source": doc["source"], "bands_hz": doc.get("bands_hz")}
    try:
        rel = str(path.relative_to(rel_root))
    except ValueError:
        rel = path.name
    return {"source": source, "file": rel, "sha256": file_sha256(path), "generator": gen}


def covered(ranges: list[list[float]], duration: float | None) -> float:
    total = sum(b - a for a, b in merge_ranges(ranges))
    if not duration:
        return total
    return min(total, duration)


def human_onsets(doc: dict[str, Any], name: str) -> list[dict[str, Any]]:
    onsets = doc.get("onsets")
    if not isinstance(onsets, list):
        raise ApiError(500, f"{name}: no 'onsets' list")
    for o in onsets:
        if not isinstance(o, dict) or o.get("type") not in LABEL_TYPES or not isinstance(
            o.get("time"), (int, float)
        ):
            raise ApiError(500, f"{name}: malformed onset {o!r}")
    return onsets


class SeedDoc(NamedTuple):
    seed: labelcheck.SeedSource
    path: Path
    doc: dict[str, Any]
    labels: dict[str, Any]


def seed_docs(c: Corpus, stem: str) -> tuple[list[SeedDoc], list[str]]:
    """The seed files of a track that exist and aren't skipped, best first; plus the names of
    those marked skipped."""
    out, skipped = [], []
    for s in c.seeds():
        p = c.seed_path(s, stem)
        if not p.is_file():
            continue
        doc = read_json(p)
        if doc.get("skipped"):
            skipped.append(s.name)
            continue
        out.append(SeedDoc(s, p, doc, split_auto_labels(doc, p.name)))
    return out, skipped


# --- Track list and seeding ---------------------------------------------------


def track_summary(c: Corpus, stem: str, human_override: dict[str, Any] | None = None) -> dict[str, Any]:
    hp, dp = c.human_path(stem), c.draft_path(stem)
    human = human_override if human_override is not None else (
        read_json(hp) if hp.is_file() else None)
    seeds, _ = seed_docs(c, stem)

    duration = None
    if human and isinstance(human.get("duration"), (int, float)):
        duration = float(human["duration"])
    if duration is None:
        duration = c.manifest_duration(stem)

    if human is not None:
        ons = human_onsets(human, hp.name)
        counts = {t: sum(1 for o in ons if o["type"] == t) for t in LABEL_TYPES}
        label_source = "human"
        ranges = human.get("reviewed_ranges") or []
    elif seeds:
        counts = {t: len(seeds[0].labels[t]) for t in LABEL_TYPES}
        label_source, ranges = seeds[0].seed.name, []
    else:
        counts = {t: 0 for t in LABEL_TYPES}
        label_source, ranges = "none", []

    # Ranking hint: how much the two best seeds disagree, per class.
    disagreement = None
    if len(seeds) >= 2:
        a, b = seeds[0].labels, seeds[1].labels
        fs = [f for f in (f_measure(a[t], b[t]) for t in LABEL_TYPES) if f is not None]
        disagreement = round(1 - sum(fs) / len(fs), 3) if fs else None

    reviewed_s = covered(ranges, duration)
    return {
        "stem": stem,
        "duration": duration,
        "label_source": label_source,
        "counts": counts,
        "has_human": human is not None,
        "has_adtof": any(s.seed.name == "drums_adtof" for s in seeds),
        "has_draft": dp.is_file(),
        "complete": bool(human and human.get("complete")),
        "reviewed_s": round(reviewed_s, 3),
        "reviewed_share": round(reviewed_s / duration, 4) if duration else None,
        "disagreement": disagreement,
        "disagreement_between": [s.seed.name for s in seeds[:2]] if len(seeds) >= 2 else None,
        "updated": human.get("updated") if human else None,
    }


def list_tracks(c: Corpus) -> list[dict[str, Any]]:
    return [track_summary(c, s) for s in c.stems()]


def seeds_public(c: Corpus) -> list[dict[str, Any]]:
    return [{"name": s.name, "label": s.label, "warning": s.warning} for s in c.seeds()]


def load_track(c: Corpus, stem: str, human_override: dict[str, Any] | None = None,
               with_draft: bool = True) -> dict[str, Any]:
    """Everything the page needs to open a track, seeded human > the seeds.json sources."""
    stem = c.validate_stem(stem)
    hp, dp = c.human_path(stem), c.draft_path(stem)
    sha = c.audio_sha256(stem)
    seeds, skipped = seed_docs(c, stem)
    refs: dict[str, Any] = {s.seed.name: s.labels for s in seeds}

    out: dict[str, Any] = {
        "stem": stem,
        "audio_url": f"/audio/{stem}.mp3",
        "audio_sha256": sha,
        "manifest_duration": c.manifest_duration(stem),
        "references": refs,
        "warning": None,
        "error": None,
        "reviewed_ranges": [],
        "complete": False,
        "updated": None,
    }
    human = human_override if human_override is not None else (
        read_json(hp) if hp.is_file() else None)
    if human is not None:
        doc = human
        out["seed"] = "human"
        out["started_from"] = doc.get("started_from")
        out["onsets"] = [
            {"time": float(o["time"]), "type": o["type"]} for o in human_onsets(doc, hp.name)
        ]
        out["reviewed_ranges"] = merge_ranges(doc.get("reviewed_ranges") or [])
        out["complete"] = bool(doc.get("complete"))
        out["updated"] = doc.get("updated")
        if doc.get("audio_sha256") != sha:
            out["error"] = (
                f"{hp.name} was made against different audio (audio_sha256 "
                f"{str(doc.get('audio_sha256'))[:12]}... vs the mp3's {sha[:12]}...). "
                "Saving is refused until this is resolved by hand."
            )
    elif seeds:
        best = seeds[0]
        out["seed"] = best.seed.name
        out["started_from"] = provenance(best.path, best.seed.name, best.doc, c.edm.parent)
        out["onsets"] = [{"time": t, "type": k} for k in LABEL_TYPES for t in best.labels[k]]
        skipped_better = [n for n in skipped
                          if [s.name for s in c.seeds()].index(n)
                          < [s.name for s in c.seeds()].index(best.seed.name)]
        if best.seed.warning or skipped_better:
            out["warning"] = " ".join(filter(None, [best.seed.warning] + [
                f"(The {n} file is marked skipped.)" for n in skipped_better]))
    else:
        out["seed"] = "empty"
        out["started_from"] = {"source": "empty"}
        out["onsets"] = []
        out["warning"] = "No labels found for this track: starting from nothing."
    out["onsets"].sort(key=lambda o: o["time"])

    out["draft"] = None
    if with_draft and dp.is_file():
        draft = read_json(dp)
        if draft.get("audio_sha256") == sha:
            out["draft"] = draft  # the page offers to restore it
        else:  # made against other audio: never offered for restore
            out["warning"] = " ".join(filter(None, [out["warning"], (
                f"Ignoring {dp.name} in .drafts: it was made against different audio.")]))
    return out


# --- Saving -------------------------------------------------------------------

_UNSET: Any = object()


def validate_doc(c: Corpus, stem: str, body: dict[str, Any],
                 existing: dict[str, Any] | None = _UNSET) -> dict[str, Any]:
    """Validate a save/draft body from the page and build the file to write. `existing` is the
    human file it replaces (default: the one on disk)."""
    if not isinstance(body, dict):
        raise ApiError(400, "body must be a JSON object")
    sha = c.audio_sha256(stem)
    if body.get("audio_sha256") != sha:
        raise ApiError(
            409, "the mp3 changed since this page loaded it (audio_sha256 mismatch): reload"
        )
    hp = c.human_path(stem)
    if existing is _UNSET:
        existing = read_json(hp) if hp.is_file() else None
    if existing is not None and existing.get("audio_sha256") != sha:
        raise ApiError(
            409,
            f"{hp.name} was made against different audio (audio_sha256 mismatch); "
            "refusing to overwrite it",
        )

    duration = body.get("duration")
    if not labelcheck.finite(duration) or duration <= 0:
        raise ApiError(400, "duration must be a positive number")
    mdur = c.manifest_duration(stem)
    if mdur is not None and abs(mdur - duration) > DURATION_TOL_S:
        raise ApiError(
            400, f"duration {duration:.3f}s disagrees with track_manifest.json ({mdur:.3f}s)"
        )

    onsets = body.get("onsets")
    if not isinstance(onsets, list) or len(onsets) > MAX_ONSETS:
        raise ApiError(400, "onsets must be a list")
    clean = set()
    for o in onsets:
        if not isinstance(o, dict):
            raise ApiError(400, f"malformed onset {o!r}")
        t, ty = o.get("time"), o.get("type")
        if ty not in LABEL_TYPES:
            raise ApiError(400, f"onset type must be kick or snare, got {ty!r}")
        if not labelcheck.finite(t) or t < 0 or t > duration + END_TOL_S:
            raise ApiError(400, f"onset time out of range: {t!r}")
        clean.add((round(float(t), 4), ty))  # exact duplicates collapse
    out_onsets = [{"time": t, "type": ty} for t, ty in sorted(clean)]

    ranges = body.get("reviewed_ranges", [])
    if not isinstance(ranges, list) or len(ranges) > MAX_RANGES:
        raise ApiError(400, "reviewed_ranges must be a list")
    for r in ranges:
        if (
            not isinstance(r, list)
            or len(r) != 2
            or not all(labelcheck.finite(x) for x in r)
            or not (0 <= r[0] < r[1] <= duration + END_TOL_S)
        ):
            raise ApiError(400, f"bad reviewed range {r!r}")
    dur = round(float(duration), 4)
    # Round, then merge again: rounding can close a gap, and the file must hold merged ranges.
    merged = merge_ranges([[round(a, 3), round(min(b, duration), 3)] for a, b in merge_ranges(ranges)])
    merged = [[round(a, 3), round(b, 3)] for a, b in merged]

    # Provenance is the seed the labels started from; an existing human file keeps its own.
    started_from = existing.get("started_from") if existing else body.get("started_from")
    if started_from is not None and not isinstance(started_from, (dict, str)):
        raise ApiError(400, "started_from must be an object")
    if len(json.dumps(started_from)) > labelcheck.MAX_STARTED_FROM_BYTES:
        raise ApiError(400, "started_from too large")

    return {
        "schema": SCHEMA,
        "track": stem,
        "audio_sha256": sha,
        "duration": dur,
        "started_from": started_from,
        "reviewed_ranges": merged,
        "complete": is_complete(merged, dur),
        "onsets": out_onsets,
        "updated": now_iso(),
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)  # mkstemp makes it 0600; a label file is ordinary repo content
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


_write_lock = threading.Lock()


def save_track(c: Corpus, stem: str, body: dict[str, Any]) -> dict[str, Any]:
    stem = c.validate_stem(stem)
    with _write_lock:
        doc = validate_doc(c, stem, body)
        atomic_write(c.human_path(stem), dumps_doc(doc))
        dp = c.draft_path(stem)
        if dp.is_file():
            dp.unlink()  # the saved file supersedes it
    return {"ok": True, "complete": doc["complete"], "updated": doc["updated"],
            "path": str(c.human_path(stem).relative_to(c.edm.parent)), "count": len(doc["onsets"])}


def draft_doc(c: Corpus, stem: str, body: dict[str, Any], existing: Any = _UNSET) -> dict[str, Any]:
    doc = validate_doc(c, stem, body, existing)
    doc["draft"] = True
    if isinstance(body.get("seed"), str):
        doc["seed"] = body["seed"][:40]
    return doc


def save_draft(c: Corpus, stem: str, body: dict[str, Any]) -> dict[str, Any]:
    stem = c.validate_stem(stem)
    with _write_lock:
        doc = draft_doc(c, stem, body)
        atomic_write(c.draft_path(stem), dumps_doc(doc))
    return {"ok": True, "updated": doc["updated"]}


# --- Hosted mode --------------------------------------------------------------------


class Who(NamedTuple):
    """Who is asking. Nothing here is authenticated: the name is self-entered and the client id is a
    random per-tab id. They keep honest people from colliding; abuse is handled by rate limits, the
    logs and corpus CI."""

    client: str | None
    name: str
    ip_hash: str
    draft_key: str | None


class HostedApp:
    def __init__(self, corpus: Corpus, state: Path, publisher: hosted.Publisher | None,
                 checkout: hosted.Checkout | None = None, *, lock_ttl: float = 900.0,
                 trust_cf: bool = False, refresh_s: float = 60.0, log: hosted.JsonLog | None = None,
                 limits: dict[str, tuple[int, float]] | None = None):
        self.c = corpus
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        self.publisher = publisher
        self.checkout = checkout
        self.rw = checkout.rw if checkout else hosted.RWLock()
        self.trust_cf = trust_cf
        self.refresh_s = refresh_s
        self.log = log or hosted.JsonLog()
        self.salt = hosted.Salt(self.state / "ip-salt")
        self.locks = hosted.LockTable(self.state / "locks.json", ttl=lock_ttl)
        self.drafts = hosted.DraftStore(self.state / "drafts")
        self.pending = hosted.PendingStore(self.state / "pending.json")
        self.limiter = hosted.RateLimiter(limits or DEFAULT_LIMITS)
        self._save_lock = threading.Lock()
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._thread: threading.Thread | None = None

    # -- who
    def who(self, headers: Any, peer: str) -> tuple[Who, str]:
        ip = peer
        if self.trust_cf:
            cf = (headers.get("CF-Connecting-IP") or "").strip()
            try:
                ip = str(ipaddress.ip_address(cf))
            except ValueError:
                pass
        client = headers.get("X-Reviewer-Client")
        if client is not None and not hosted.CLIENT_RE.match(client):
            raise ApiError(400, "bad X-Reviewer-Client")
        try:
            name = hosted.normalize_name(unquote(headers.get("X-Reviewer-Name") or ""))
        except ValueError as e:
            raise ApiError(400, str(e)) from None
        ih = self.salt.ip_hash(ip)
        return Who(client, name, ih, hosted.DraftStore.key_for(name, client)), ih

    def limit(self, kind: str, key: str) -> None:
        ok, wait = self.limiter.allow(kind, key)
        if not ok:
            self.log.emit("rate_limited", kind=kind, ip=key, retry_after=wait)
            raise ApiError(429, f"too many requests ({kind}): try again in {wait} s",
                           retry_after=wait)

    # -- state of a track
    def effective_human(self, stem: str) -> tuple[dict[str, Any] | None, str | None]:
        rec = self.pending.get(stem)
        if rec and rec.get("state") in hosted.PendingStore.ACTIVE:
            return json.loads(rec["text"]), "pending"
        hp = self.c.human_path(stem)
        if hp.is_file():
            return read_json(hp), "main"
        return None, None

    def version(self, stem: str) -> str:
        """Opaque: changes whenever the track's current human labels change."""
        rec = self.pending.get(stem)
        if rec and rec.get("state") in hosted.PendingStore.ACTIVE:
            return "p-" + hashlib.sha256(rec["text"].encode()).hexdigest()[:16]
        hp = self.c.human_path(stem)
        if hp.is_file():
            return "m-" + hosted.git_blob_sha(hp.read_bytes())[:16]
        return "none"

    def pending_public(self, stem: str) -> dict[str, Any] | None:
        rec = self.pending.get(stem)
        if not rec or rec.get("state") not in hosted.PendingStore.ACTIVE:
            return None
        return {"pr": rec["pr"], "url": rec["url"], "state": rec["state"],
                "contributor": rec.get("contributor") or "someone", "created": rec.get("created")}

    def config(self) -> dict[str, Any]:
        return {"hosted": True, "lock_ttl_s": self.locks.ttl, "seeds": seeds_public(self.c),
                "saving": self.publisher is not None}

    def tracks(self, who: Who) -> list[dict[str, Any]]:
        locks = self.locks.all()
        out = []
        with self.rw.read():
            for stem in self.c.stems():
                eff, src = self.effective_human(stem)
                t = track_summary(self.c, stem, human_override=eff if src == "pending" else None)
                t["has_draft"] = self.drafts.has(stem, who.draft_key)
                lk = locks.get(stem)
                t["lock"] = lk.public(who.client) if lk else None
                t["pending"] = self.pending_public(stem)
                out.append(t)
        return out

    def track(self, stem: str, who: Who) -> dict[str, Any]:
        with self.rw.read():
            stem = self.c.validate_stem(stem)
            eff, src = self.effective_human(stem)
            view = load_track(self.c, stem, human_override=eff if src == "pending" else None,
                              with_draft=False)
            view["version"] = self.version(stem)
        d = self.drafts.load(stem, who.draft_key)
        if d is not None and d.get("audio_sha256") != view["audio_sha256"]:
            view["warning"] = " ".join(filter(None, [
                view["warning"], "Ignoring your draft: it was made against different audio."]))
            d = None
        view["draft"] = d
        lk = self.locks.get(stem)
        view["lock"] = lk.public(who.client) if lk else None
        view["pending"] = self.pending_public(stem)
        view["hosted"] = True
        return view

    # -- writes
    def _need_client(self, who: Who) -> str:
        if not who.client:
            raise ApiError(400, "X-Reviewer-Client required")
        return who.client

    def _need_lock(self, stem: str, who: Who) -> None:
        if not self.locks.holds(stem, self._need_client(who)):
            lk = self.locks.get(stem)
            msg = (f"{lk.name or 'Someone'} is editing this track" if lk else
                   "your edit lock has lapsed: reopen the track to edit again")
            raise ApiError(423, msg, lock=lk.public(who.client) if lk else None)

    def lock(self, stem: str, who: Who, body: dict[str, Any]) -> dict[str, Any]:
        client = self._need_client(who)
        action = body.get("action") if isinstance(body, dict) else None
        if action == "release":
            if self.locks.release(stem, client):
                self.log.emit("lock_released", stem=stem, ip=who.ip_hash, contributor=who.name)
            return {"ok": True}
        if action == "renew":
            try:
                lk = self.locks.renew(stem, client)
            except hosted.Locked as e:
                cur = self.locks.get(stem)
                raise ApiError(423, e.reason, lock=cur.public(client) if cur else None) from None
            return {"ok": True, "lock": lk.public(client), "ttl_s": self.locks.ttl}
        if action != "acquire":
            raise ApiError(400, "action must be acquire, renew or release")
        if not self.locks.holds(stem, client):
            self.limit("lock", who.ip_hash)
        try:
            lk, new = self.locks.acquire(stem, client, who.name, who.ip_hash,
                                         takeover=bool(body.get("takeover")))
        except hosted.Locked as e:
            cur = self.locks.get(stem)
            raise ApiError(423, e.reason, lock=cur.public(client) if cur else None) from None
        if new:
            self.log.emit("lock_acquired", stem=stem, ip=who.ip_hash, contributor=who.name)
        return {"ok": True, "lock": lk.public(client), "ttl_s": self.locks.ttl}

    def draft(self, stem: str, who: Who, body: dict[str, Any]) -> dict[str, Any]:
        self._need_lock(stem, who)
        self.limit("draft", who.ip_hash)
        if who.draft_key is None:
            raise ApiError(400, "no draft key")
        with self.rw.read():
            eff, _ = self.effective_human(stem)
            doc = draft_doc(self.c, stem, body, existing=eff)
        doc["contributor"] = who.name
        try:
            self.drafts.save(stem, who.draft_key, dumps_doc(doc))
        except OverflowError as e:
            raise ApiError(507, str(e)) from None
        return {"ok": True, "updated": doc["updated"]}

    def save(self, stem: str, who: Who, body: dict[str, Any]) -> dict[str, Any]:
        self._need_lock(stem, who)
        if self.publisher is None:
            raise ApiError(503, "Saving isn't switched on yet on this server. Your work is kept as a "
                                "draft (it autosaves); submit it once saving is on.")
        self.limit("save", who.ip_hash)
        self.limit("save_global", "all")
        with self._save_lock:
            with self.rw.read():
                eff, src = self.effective_human(stem)
                if src == "pending":
                    p = self.pending_public(stem)
                    raise ApiError(409, f"A save of this track is waiting for CI (PR #{p['pr']}). "
                                        "Try again once it has merged; your work is kept as a draft.")
                doc = validate_doc(self.c, stem, body, existing=eff)
                text = dumps_doc(doc)
                if who.draft_key:  # the work survives whatever happens next
                    d = dict(doc, draft=True, contributor=who.name)
                    if isinstance(body.get("seed"), str):
                        d["seed"] = body["seed"][:40]
                    self.drafts.save(stem, who.draft_key, dumps_doc(d))
                if body.get("base_version") != self.version(stem):
                    raise ApiError(409, "Someone saved this track since you opened it. Reload the "
                                        "track to see their labels; your work is kept as a draft.")
                if not doc["complete"]:
                    raise ApiError(422, "Only a 100%-reviewed track can be submitted. Your work is "
                                        "kept as a draft: finish the review, then save.")
                problems = labelcheck.check_human_doc(self.c.edm, stem, json.loads(text), eff,
                                                      self.c.seeds(), audio_sha256=doc["audio_sha256"])
                hp = self.c.human_path(stem)
                expected = hosted.git_blob_sha(hp.read_bytes()) if hp.is_file() else None
                rel = str(hp.relative_to(self.c.edm.parent))
            if problems:
                self.log.emit("save_refused", stem=stem, ip=who.ip_hash, contributor=who.name,
                              problems=problems)
                raise ApiError(422, "Not accepted: " + "; ".join(problems) +
                               ". Your work is kept as a draft.", problems=problems)
            nk = sum(1 for o in doc["onsets"] if o["type"] == "kick")
            ns = len(doc["onsets"]) - nk
            save_id = str(uuid.uuid4())
            name = who.name or "anonymous"
            headline = f"labels({stem}): human drum labels, {nk} kicks, {ns} snares"
            pr_body = (f"Saved from the hosted drum reviewer by `{name}`: `{rel}`, 100% reviewed, "
                       f"{nk} kicks and {ns} snares. It replaces "
                       f"{'the existing human file' if eff else 'the seed labels'}.\n\n"
                       "Corpus CI validates the file (`tools/drum-reviewer/labelcheck.py`), and "
                       "auto-merge (squash) merges it once CI passes. Closing this PR stops it: the "
                       "service puts the labels back into the contributor's drafts.")
            trailers = (f"Contributed-by: {name}\nClient-IP-Hash: {who.ip_hash}\n"
                        f"Reviewer-Save-Id: {save_id}")
            try:
                res = self.publisher.publish(stem, rel, text, expected, headline, pr_body, trailers)
            except hosted.Conflict:
                self._kick.set()  # refresh the checkout now
                self.log.emit("save_conflict", stem=stem, ip=who.ip_hash, contributor=who.name)
                raise ApiError(409, "The corpus changed since you opened this track. Reload it in a "
                                    "minute; your work is kept as a draft.") from None
            except hosted.GitHubError as e:
                self.log.emit("save_failed", stem=stem, ip=who.ip_hash, error=e.message,
                              status=e.status)
                raise ApiError(502, "Couldn't open the pull request on GitHub. Your work is kept "
                                    "as a draft; try again later.") from None
            rec = {"stem": stem, "pr": res["pr"], "url": res["url"], "branch": res["branch"],
                   "state": "open", "created": now_iso(), "contributor": who.name,
                   "ip": who.ip_hash, "draft_key": who.draft_key, "save_id": save_id,
                   "automerge": res["automerge"], "text": text}
            self.pending.put(stem, rec)
        self.drafts.delete(stem, who.draft_key)
        self.locks.release(stem, who.client or "")
        entry = {"ts": rec["created"], "stem": stem, "pr": res["pr"], "contributor": who.name,
                 "ip": who.ip_hash, "save_id": save_id, "kicks": nk, "snares": ns,
                 "automerge": res["automerge"]}
        with open(self.state / "saves.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.log.emit("save", **{k: v for k, v in entry.items() if k != "ts"},
                      automerge_error=res.get("automerge_error"))
        return {"ok": True, "complete": True, "count": len(doc["onsets"]), "pr": res["pr"],
                "url": res["url"], "automerge": res["automerge"], "updated": doc["updated"]}

    # -- background: refresh the checkout, follow open pull requests
    def tick(self) -> None:
        if self.checkout is not None:
            try:
                self.checkout.refresh()
            except Exception as e:  # GitHub down: keep serving what we have, say so
                self.log.emit("refresh_failed", error=str(e)[:400])
        for stem, rec in self.pending.all().items():
            if rec["state"] == "open" and self.publisher is not None:
                try:
                    st = self.publisher.poll(rec)
                except hosted.GitHubError as e:
                    self.log.emit("pr_poll_failed", stem=stem, pr=rec["pr"], error=e.message)
                    continue
                if st == "merged":
                    self.pending.update(stem, state="merged", merged_at=time.time())
                    self.log.emit("pr_merged", stem=stem, pr=rec["pr"], contributor=rec["contributor"])
                    self.publisher.cleanup(rec)
                elif st == "closed":
                    self._restore_draft(stem, rec)
                    self.pending.clear(stem)
                    self.log.emit("pr_closed", stem=stem, pr=rec["pr"], contributor=rec["contributor"])
                    self.publisher.cleanup(rec)
                    continue
            rec = self.pending.get(stem)
            if rec and rec["state"] == "merged":
                with self.rw.read():
                    hp = self.c.human_path(stem)
                    landed = hp.is_file() and hp.read_text() == rec["text"]
                # Landed in the checkout; or long merged and the file has moved on since.
                if landed or time.time() - rec.get("merged_at", 0) > 1800:
                    self.pending.clear(stem)
                    self.log.emit("pr_landed", stem=stem, pr=rec["pr"], in_checkout=landed)

    def _restore_draft(self, stem: str, rec: dict[str, Any]) -> None:
        key = rec.get("draft_key")
        if not key or self.drafts.has(stem, key):
            return  # a newer draft wins
        doc = json.loads(rec["text"])
        doc.update(draft=True, contributor=rec.get("contributor") or "")
        try:
            self.drafts.save(stem, key, dumps_doc(doc))
        except OverflowError:
            self.log.emit("draft_restore_failed", stem=stem, pr=rec["pr"])

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as e:  # never let the loop die quietly
                    self.log.emit("tick_failed", error=repr(e)[:400])
                self._kick.wait(self.refresh_s)
                self._kick.clear()

        self._thread = threading.Thread(target=loop, name="hosted-tick", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()


def import_drafts(app: HostedApp, files: list[Path], contributor: str) -> list[str]:
    """Copy drafts made by the local reviewer (edm/drums_human/.drafts/<stem>.drums.json) into the
    hosted draft store under a contributor's name. The source files are only read."""
    name = hosted.normalize_name(contributor)
    if not name:
        raise SystemExit("--contributor is required: the name you will type into the page")
    key = hosted.DraftStore.key_for(name, None)
    done = []
    for f in files:
        doc = json.loads(Path(f).read_text())
        stem = app.c.validate_stem(doc.get("track"))
        if doc.get("audio_sha256") != app.c.audio_sha256(stem):
            raise SystemExit(f"{f}: made against different audio than the corpus's {stem}.mp3")
        doc.update(draft=True, contributor=name)
        app.drafts.save(stem, key, dumps_doc(doc))
        app.log.emit("draft_imported", stem=stem, contributor=name, source=Path(f).name)
        done.append(stem)
    return done


# --- HTTP ---------------------------------------------------------------------
#
# No auth, but other web pages open in the reviewer's browser must not be able to write labels:
#   * every request's Host must be an IP literal, localhost, this machine's name or an
#     --allow-host name, which defeats DNS rebinding (a hostile name resolving to this server);
#   * a POST must be Content-Type: application/json (a cross-origin page can't send that
#     without a CORS preflight) and, when it carries an Origin, the Origin must be this server
#     or one of the --cors-origin pages (hosted mode: the product site serves the page and this
#     server is its API). Preflights are answered for those origins only.


def _split_host(value: str) -> str:
    v = value.strip().lower()
    if v.startswith("["):  # [ipv6]:port
        return v[1 : v.find("]")] if "]" in v else v
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def local_names() -> set[str]:
    names = {"localhost"}
    for n in (socket.gethostname(), socket.getfqdn()):
        if n:
            names |= {n.lower(), n.lower().split(".")[0], n.lower().split(".")[0] + ".local"}
    return names


def host_allowed(host: str | None, allowed: set[str]) -> bool:
    if not host:
        return False
    h = _split_host(host)
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return h in allowed


# The page is self-contained: inline script and styles, its analysis worker from a Blob URL.
HOSTED_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; worker-src blob:; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


CORS_HEADERS = "Content-Type, X-Reviewer-Client, X-Reviewer-Name"


class Handler(BaseHTTPRequestHandler):
    corpus: Corpus  # set by make_server()
    allowed_hosts: set[str]  # set by make_server()
    app: HostedApp | None = None  # hosted mode
    cors_origins: frozenset[str] = frozenset()  # pages on other origins allowed to call the API
    page_url: str | None = None  # hosted: where GET / sends people (the product page)
    server_version = "drum-reviewer/1"
    timeout = 60  # a stalled client can't hold a thread forever

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet audio/api chatter
        if self.app is not None:
            return  # hosted mode logs one JSON line per request instead
        if args and isinstance(args[0], str) and ("/audio/" in args[0] or "/api/" in args[0]):
            if len(args) > 1 and str(args[1]).startswith(("2", "3")):
                return
        super().log_message(fmt, *args)

    def send_response(self, code: int, message: str | None = None) -> None:
        self._status = code
        super().send_response(code, message)

    def _cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if origin and origin in self.cors_origins else None

    def end_headers(self) -> None:
        if self.app is not None:
            for k, v in HOSTED_HEADERS.items():
                self.send_header(k, v)
        if self.cors_origins:
            self.send_header("Vary", "Origin")
            origin = self._cors_origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Expose-Headers", "Retry-After")
        super().end_headers()

    def _send_json(self, data: Any, status: int = 200, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _stem_param(self, q: dict[str, list[str]]) -> str:
        vals = q.get("stem")
        if not vals:
            raise ApiError(400, "missing ?stem=")
        return self.corpus.validate_stem(vals[0])

    def _dispatch(self, fn) -> None:
        t0 = time.monotonic()
        self._status = 0
        self._ip = None
        try:
            if self.app is not None:
                self._who, self._ip = self.app.who(self.headers, self.client_address[0])
                if urlparse(self.path).path != "/healthz":
                    self.app.limit("request", self._ip)
            fn()
        except ApiError as e:
            extra = {"Retry-After": str(e.extra["retry_after"])} if "retry_after" in e.extra else None
            self._send_json({"error": e.message, **e.extra}, e.status, extra)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # a malformed label file, a bug: say so, loudly
            if self.app is not None:
                self.app.log.emit("error", method=self.command, path=urlparse(self.path).path,
                                  error=repr(e)[:400])
            else:
                self.log_error("%s %s failed: %r", self.command, self.path, e)
            self._send_json({"error": f"server error: {e!r}"}, 500)
        finally:
            if self.app is not None:
                url = urlparse(self.path)
                stem = (parse_qs(url.query).get("stem") or [None])[0]
                self.app.log.emit("http", method=self.command, path=url.path[:200],
                                  stem=stem[:80] if isinstance(stem, str) else None,
                                  status=self._status, ms=round(1000 * (time.monotonic() - t0), 1),
                                  ip=self._ip)

    def _check_host(self) -> None:
        if not host_allowed(self.headers.get("Host"), self.allowed_hosts):
            raise ApiError(403, f"Host {self.headers.get('Host')!r} not allowed (--allow-host)")

    def _check_post(self) -> None:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ApiError(415, "POST must be Content-Type: application/json")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.cors_origins:
            o = urlparse(origin)
            if o.netloc.lower() != (self.headers.get("Host") or "").lower():
                raise ApiError(403, f"cross-origin POST from {origin!r} refused")

    def do_GET(self) -> None:
        self._dispatch(lambda: (self._check_host(), self._get()))

    def do_POST(self) -> None:
        self._dispatch(lambda: (self._check_host(), self._check_post(), self._post()))

    def do_OPTIONS(self) -> None:
        self._dispatch(lambda: (self._check_host(), self._preflight()))

    def _preflight(self) -> None:
        """CORS preflight: answered only for a --cors-origin page asking for GET or POST."""
        if not self._cors_origin():
            raise ApiError(403, "origin not allowed")
        if (self.headers.get("Access-Control-Request-Method") or "") not in ("GET", "POST"):
            raise ApiError(405, "method not allowed")
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", CORS_HEADERS)
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _get(self) -> None:
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        app = self.app
        if path in ("/", "/index.html") and self.page_url:
            self.send_response(302)
            self.send_header("Location", self.page_url)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path in ("/", "/index.html"):
            body = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/healthz":
            self._send_json({"ok": True})
        elif path == "/api/config":
            self._send_json(app.config() if app else {"hosted": False, "seeds": seeds_public(self.corpus)})
        elif path == "/api/tracks":
            if app:
                self._send_json({"tracks": app.tracks(self._who), "seeds": seeds_public(self.corpus)})
            else:
                self._send_json({"tracks": list_tracks(self.corpus), "edm": str(self.corpus.edm),
                                 "seeds": seeds_public(self.corpus)})
        elif path == "/api/track":
            stem = self._stem_param(q)
            self._send_json(app.track(stem, self._who) if app else load_track(self.corpus, stem))
        elif path.startswith("/audio/"):
            name = unquote(path[len("/audio/"):])
            if not name.endswith(".mp3"):
                raise ApiError(404, "not found")
            stem = self.corpus.validate_stem(name[: -len(".mp3")])
            self._serve_audio(self.corpus.audio_path(stem))
        else:
            raise ApiError(404, "not found")

    def _read_body(self) -> Any:
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ApiError(411, "Content-Length required") from None
        if n < 0 or n > MAX_POST_BYTES:
            raise ApiError(413, f"body over {MAX_POST_BYTES} bytes")
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ApiError(400, f"invalid JSON: {e}") from e

    def _post(self) -> None:
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        if self.app is not None:
            app, who = self.app, self._who
            routes = {"/api/save": app.save, "/api/draft": app.draft, "/api/lock": app.lock}
            if path not in routes:
                raise ApiError(404, "not found")
            stem = self._stem_param(q)
            body = self._read_body()
            self._send_json(routes[path](stem, who, body))
            return
        routes = {"/api/save": save_track, "/api/draft": save_draft}
        if path in routes:
            stem = self._stem_param(q)
            body = self._read_body()
            self._send_json(routes[path](self.corpus, stem, body))
        else:
            raise ApiError(404, "not found")

    def _serve_audio(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = 200
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or (not m.group(1) and not m.group(2)):
                raise ApiError(416, "bad Range")
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:  # suffix range: the last N bytes
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                raise ApiError(416, "range not satisfiable")
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "public, max-age=300")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def make_server(
    edm: Path, host: str = "127.0.0.1", port: int = 8766, allow_hosts: tuple[str, ...] = (),
    app: HostedApp | None = None, cors_origins: tuple[str, ...] = (), page_url: str | None = None,
) -> ThreadingHTTPServer:
    corpus = app.c if app is not None else Corpus(edm)
    if not corpus.edm.is_dir():
        raise SystemExit(f"edm dir not found: {corpus.edm}")
    for o in cors_origins:
        u = urlparse(o)
        if u.scheme not in ("http", "https") or not u.netloc or u.path not in ("", "/") or u.query:
            raise SystemExit(f"--cors-origin must be scheme://host[:port], got {o!r}")
    allowed = local_names() | {h.lower() for h in allow_hosts}
    handler = type("BoundHandler", (Handler,), {
        "corpus": corpus, "allowed_hosts": allowed, "app": app, "page_url": page_url,
        "cors_origins": frozenset(o.rstrip("/") for o in cors_origins)})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) for this machine only; 0.0.0.0 to serve the LAN")
    ap.add_argument("--edm", type=Path, default=None, help=f"corpus dir (default {DEFAULT_EDM}, or "
                    "<--checkout>/edm in hosted mode)")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="extra hostname the page may be opened under (IPs, localhost and this "
                         "machine's name are always allowed)")
    h = ap.add_argument_group("hosted mode (README.md \"Hosted\")")
    h.add_argument("--state", type=Path, help="state dir (locks, drafts, pending saves, logs): "
                   "turns on hosted mode")
    h.add_argument("--checkout", type=Path, help="the service's own corpus clone (cloned if missing)")
    h.add_argument("--git-url", help="where --checkout is cloned and refreshed from")
    h.add_argument("--git-branch", default="main")
    h.add_argument("--refresh", type=float, default=60.0, metavar="S",
                   help="refresh the clone and follow open pull requests every S seconds")
    h.add_argument("--github-repo", metavar="OWNER/NAME", help="where saves become pull requests; "
                   "the token comes from $REVIEWER_GITHUB_TOKEN")
    h.add_argument("--github-api", default="https://api.github.com")
    h.add_argument("--lock-ttl", type=float, default=900.0, metavar="S")
    h.add_argument("--cors-origin", action="append", default=[], metavar="URL",
                   help="a page origin (scheme://host) allowed to call this API, e.g. the product "
                        "site that serves the reviewer page")
    h.add_argument("--page-url", help="where GET / redirects (the product page), instead of serving "
                                      "the page here")
    h.add_argument("--trust-cf-connecting-ip", action="store_true",
                   help="take the client address from CF-Connecting-IP (only behind a Cloudflare "
                        "tunnel that is the sole way in)")
    h.add_argument("--import-draft", type=Path, action="append", default=[], metavar="FILE",
                   help="copy a local draft into the hosted draft store and exit")
    h.add_argument("--contributor", help="with --import-draft: the name the drafts belong to")
    args = ap.parse_args(argv)

    app = None
    if args.state is not None:
        log = hosted.JsonLog()
        checkout = None
        if args.checkout is not None:
            if not args.git_url:
                ap.error("--checkout needs --git-url")
            checkout = hosted.Checkout(args.checkout.resolve(), args.git_url, args.git_branch, log=log)
            checkout.ensure()
        edm = args.edm or (args.checkout / "edm" if args.checkout else None)
        if edm is None:
            ap.error("hosted mode needs --checkout or --edm")
        publisher = None
        if not args.import_draft:
            if not args.github_repo:
                ap.error("hosted mode needs --github-repo")
            token = os.environ.get("REVIEWER_GITHUB_TOKEN", "")
            if token:
                gh = hosted.GitHub(args.github_repo, token, args.github_api)
                publisher = hosted.Publisher(gh, args.git_branch, log)
            else:  # locks and drafts work; a save answers 503 until a token is configured
                log.emit("publishing_disabled", reason="REVIEWER_GITHUB_TOKEN is empty")
        app = HostedApp(Corpus(edm), args.state, publisher, checkout, lock_ttl=args.lock_ttl,
                        trust_cf=args.trust_cf_connecting_ip, refresh_s=args.refresh, log=log)
        if args.import_draft:
            done = import_drafts(app, args.import_draft, args.contributor or "")
            print(f"imported {len(done)} draft(s): {', '.join(done)}", file=sys.stderr)
            return
    elif args.import_draft:
        ap.error("--import-draft needs --state")

    srv = make_server(args.edm or DEFAULT_EDM, args.host, args.port, tuple(args.allow_host), app,
                      tuple(args.cors_origin), args.page_url)
    c: Corpus = srv.RequestHandlerClass.corpus  # type: ignore[attr-defined]
    stems = c.stems()
    if app is not None:
        app.log.emit("start", tracks=len(stems), port=args.port, repo=args.github_repo)
        app.start()
    else:
        print(f"drum-reviewer: {len(stems)} tracks in {c.edm}")
        print(f"  human GT -> {c.human_dir}   drafts -> {c.draft_dir}")
        print(f"  open http://{'localhost' if args.host in ('127.0.0.1', 'localhost') else args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app is not None:
            app.stop()


if __name__ == "__main__":
    main()
