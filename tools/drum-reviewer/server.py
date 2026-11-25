#!/usr/bin/env python3
"""Drum-hit ground-truth reviewer: hand-curate kick/snare labels for the edm/ corpus.

Serves one page (index.html, no external fetches, works offline on a LAN) and a
small JSON API. All audio analysis (band envelopes, onset strength, spectrogram)
runs in the browser; this server only lists tracks, seeds labels and writes files.

    python tools/drum-reviewer/server.py [--port 8766] [--host 127.0.0.1] [--edm edm/]

Files (paths relative to --edm):
    <stem>.mp3                          the audio
    drums_human/<stem>.drums.json       human ground truth (written by Save)
    drums_human/.drafts/<stem>.drums.json   autosaved drafts (gitignored)
    drums_adtof/<stem>.kick_weighted.json   ADTOF labels (seed, read-only)
    kick_weighted/<stem>.kick_weighted.json band-detector labels (seed of last resort, read-only)

Seed order when a track opens: human file > drums_adtof > kick_weighted.
Python standard library only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

STATIC_DIR = Path(__file__).resolve().parent
DEFAULT_EDM = STATIC_DIR.parents[1] / "edm"

SCHEMA = 1
MAX_POST_BYTES = 2 * 1024 * 1024  # a 1000-hit track is ~50 KB
MAX_ONSETS = 50_000
MAX_RANGES = 10_000
MATCH_TOL_S = 0.050  # ranking hint: ADTOF vs band labels matched at +-50 ms
GAP_TOL_S = 0.010  # reviewed ranges closer than this are merged
END_TOL_S = 0.100  # a track is complete when reviewed to within this of each end
DURATION_TOL_S = 1.0  # client-decoded duration vs track_manifest.json duration
LABEL_TYPES = ("kick", "snare")
STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --- Corpus layout ----------------------------------------------------------


class Corpus:
    def __init__(self, edm: Path):
        self.edm = Path(edm).resolve()
        self.human_dir = self.edm / "drums_human"
        self.draft_dir = self.human_dir / ".drafts"
        self.adtof_dir = self.edm / "drums_adtof"
        self.kw_dir = self.edm / "kick_weighted"
        self._sha_cache: dict[str, tuple[float, int, str]] = {}
        self._lock = threading.Lock()

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
        return self._inside(self.human_dir / f"{stem}.drums.json", self.human_dir)

    def draft_path(self, stem: str) -> Path:
        return self._inside(self.draft_dir / f"{stem}.drums.json", self.draft_dir)

    def adtof_path(self, stem: str) -> Path:
        return self._inside(self.adtof_dir / f"{stem}.kick_weighted.json", self.adtof_dir)

    def kw_path(self, stem: str) -> Path:
        return self._inside(self.kw_dir / f"{stem}.kick_weighted.json", self.kw_dir)

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
        path = self.edm / "track_manifest.json"
        if not path.is_file():
            return None
        entry = json.loads(path.read_text()).get(stem)
        if isinstance(entry, dict) and isinstance(entry.get("duration"), (int, float)):
            return float(entry["duration"])
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ApiError(500, f"{path.name} is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ApiError(500, f"{path.name}: expected a JSON object")
    return doc


# --- Labels -----------------------------------------------------------------


def split_auto_labels(doc: dict[str, Any], name: str) -> dict[str, list[float]]:
    """kick_weighted-schema onsets -> {"kick": [...], "snare": [...], "hihat": [...], "other": n}."""
    out: dict[str, Any] = {"kick": [], "snare": [], "hihat": [], "other": 0}
    onsets = doc.get("onsets")
    if not isinstance(onsets, list):
        raise ApiError(500, f"{name}: no 'onsets' list")
    for o in onsets:
        if not isinstance(o, dict) or not isinstance(o.get("time"), (int, float)):
            raise ApiError(500, f"{name}: malformed onset {o!r}")
        t = o.get("type")
        if t in ("kick", "snare", "hihat"):
            out[t].append(float(o["time"]))
        else:
            out["other"] += 1
    for k in ("kick", "snare", "hihat"):
        out[k].sort()
    return out


def provenance(path: Path, source: str, doc: dict[str, Any], rel_root: Path) -> dict[str, Any]:
    gen = doc.get("generator")
    if gen is None and "source" in doc:  # kick_weighted files name their detector here
        gen = {"source": doc["source"], "bands_hz": doc.get("bands_hz")}
    try:
        rel = str(path.relative_to(rel_root))
    except ValueError:
        rel = path.name
    return {"source": source, "file": rel, "sha256": file_sha256(path), "generator": gen}


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


def merge_ranges(ranges: list[list[float]], gap: float = GAP_TOL_S) -> list[list[float]]:
    rs = sorted([float(a), float(b)] for a, b in ranges if b > a)
    out: list[list[float]] = []
    for a, b in rs:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(ranges: list[list[float]], duration: float | None) -> float:
    total = sum(b - a for a, b in merge_ranges(ranges))
    if not duration:
        return total
    return min(total, duration)


def is_complete(ranges: list[list[float]], duration: float) -> bool:
    m = merge_ranges(ranges)
    return len(m) == 1 and m[0][0] <= END_TOL_S and m[0][1] >= duration - END_TOL_S


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


# --- Track list and seeding ---------------------------------------------------


def track_summary(c: Corpus, stem: str) -> dict[str, Any]:
    hp, ap, kp, dp = c.human_path(stem), c.adtof_path(stem), c.kw_path(stem), c.draft_path(stem)
    human = read_json(hp) if hp.is_file() else None
    adtof_doc = read_json(ap) if ap.is_file() else None
    adtof = (
        split_auto_labels(adtof_doc, ap.name)
        if adtof_doc is not None and not adtof_doc.get("skipped")
        else None
    )
    kw = split_auto_labels(read_json(kp), kp.name) if kp.is_file() else None

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
    elif adtof is not None:
        counts = {t: len(adtof[t]) for t in LABEL_TYPES}
        label_source, ranges = "drums_adtof", []
    elif kw is not None:
        counts = {t: len(kw[t]) for t in LABEL_TYPES}
        label_source, ranges = "kick_weighted", []
    else:
        counts = {t: 0 for t in LABEL_TYPES}
        label_source, ranges = "none", []

    # Ranking hint: how much ADTOF and the old band detector disagree, per class.
    disagreement = None
    if adtof is not None and kw is not None:
        fs = [f for f in (f_measure(adtof[t], kw[t]) for t in LABEL_TYPES) if f is not None]
        disagreement = round(1 - sum(fs) / len(fs), 3) if fs else None

    reviewed_s = covered(ranges, duration)
    return {
        "stem": stem,
        "duration": duration,
        "label_source": label_source,
        "counts": counts,
        "has_human": human is not None,
        "has_adtof": adtof is not None,
        "has_draft": dp.is_file(),
        "complete": bool(human and human.get("complete")),
        "reviewed_s": round(reviewed_s, 3),
        "reviewed_share": round(reviewed_s / duration, 4) if duration else None,
        "disagreement": disagreement,
        "updated": human.get("updated") if human else None,
    }


def list_tracks(c: Corpus) -> list[dict[str, Any]]:
    return [track_summary(c, s) for s in c.stems()]


def load_track(c: Corpus, stem: str) -> dict[str, Any]:
    """Everything the page needs to open a track, seeded human > ADTOF > kick_weighted."""
    stem = c.validate_stem(stem)
    hp, ap, kp, dp = c.human_path(stem), c.adtof_path(stem), c.kw_path(stem), c.draft_path(stem)
    sha = c.audio_sha256(stem)
    refs: dict[str, Any] = {}
    adtof_doc = read_json(ap) if ap.is_file() else None
    kw_doc = read_json(kp) if kp.is_file() else None
    adtof_skipped = bool(adtof_doc and adtof_doc.get("skipped"))
    if adtof_skipped:  # ADTOF gave up on this track upstream: never seed from it
        adtof_doc = None
    if adtof_doc is not None:
        refs["drums_adtof"] = split_auto_labels(adtof_doc, ap.name)
    if kw_doc is not None:
        refs["kick_weighted"] = split_auto_labels(kw_doc, kp.name)

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
    if hp.is_file():
        doc = read_json(hp)
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
    elif adtof_doc is not None:
        out["seed"] = "drums_adtof"
        out["started_from"] = provenance(ap, "drums_adtof", adtof_doc, c.edm.parent)
        lab = refs["drums_adtof"]
        out["onsets"] = [{"time": t, "type": k} for k in LABEL_TYPES for t in lab[k]]
    elif kw_doc is not None:
        out["seed"] = "kick_weighted"
        out["started_from"] = provenance(kp, "kick_weighted", kw_doc, c.edm.parent)
        lab = refs["kick_weighted"]
        out["onsets"] = [{"time": t, "type": k} for k in LABEL_TYPES for t in lab[k]]
        out["warning"] = (
            "Seeded from kick_weighted: band-detector labels, known bad "
            "(e.g. most snares labelled kicks). Check every hit."
            + (" (The ADTOF file is marked skipped.)" if adtof_skipped else "")
        )
    else:
        out["seed"] = "empty"
        out["started_from"] = {"source": "empty"}
        out["onsets"] = []
        out["warning"] = "No labels found for this track: starting from nothing."
    out["onsets"].sort(key=lambda o: o["time"])

    out["draft"] = None
    if dp.is_file():
        draft = read_json(dp)
        if draft.get("audio_sha256") == sha:
            out["draft"] = draft  # the page offers to restore it
        else:  # made against other audio: never offered for restore
            out["warning"] = " ".join(filter(None, [out["warning"], (
                f"Ignoring {dp.name} in .drafts: it was made against different audio.")]))
    return out


# --- Saving -------------------------------------------------------------------


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_doc(c: Corpus, stem: str, body: dict[str, Any]) -> dict[str, Any]:
    """Validate a save/draft body from the page and build the file to write."""
    if not isinstance(body, dict):
        raise ApiError(400, "body must be a JSON object")
    sha = c.audio_sha256(stem)
    if body.get("audio_sha256") != sha:
        raise ApiError(
            409, "the mp3 changed since this page loaded it (audio_sha256 mismatch): reload"
        )
    hp = c.human_path(stem)
    existing = read_json(hp) if hp.is_file() else None
    if existing is not None and existing.get("audio_sha256") != sha:
        raise ApiError(
            409,
            f"{hp.name} was made against different audio (audio_sha256 mismatch); "
            "refusing to overwrite it",
        )

    duration = body.get("duration")
    if not _finite(duration) or duration <= 0:
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
        if not _finite(t) or t < 0 or t > duration + END_TOL_S:
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
            or not all(_finite(x) for x in r)
            or not (0 <= r[0] < r[1] <= duration + END_TOL_S)
        ):
            raise ApiError(400, f"bad reviewed range {r!r}")
    merged = [[round(a, 3), round(min(b, duration), 3)] for a, b in merge_ranges(ranges)]

    # Provenance is the seed the labels started from; an existing human file keeps its own.
    started_from = existing.get("started_from") if existing else body.get("started_from")
    if started_from is not None and not isinstance(started_from, (dict, str)):
        raise ApiError(400, "started_from must be an object")
    if len(json.dumps(started_from)) > 4096:
        raise ApiError(400, "started_from too large")

    return {
        "schema": SCHEMA,
        "track": stem,
        "audio_sha256": sha,
        "duration": round(float(duration), 4),
        "started_from": started_from,
        "reviewed_ranges": merged,
        "complete": is_complete(merged, float(duration)),
        "onsets": out_onsets,
        "updated": now_iso(),
    }


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


def save_draft(c: Corpus, stem: str, body: dict[str, Any]) -> dict[str, Any]:
    stem = c.validate_stem(stem)
    with _write_lock:
        doc = validate_doc(c, stem, body)
        doc["draft"] = True
        if isinstance(body.get("seed"), str):
            doc["seed"] = body["seed"][:40]
        atomic_write(c.draft_path(stem), dumps_doc(doc))
    return {"ok": True, "updated": doc["updated"]}


def discard_draft(c: Corpus, stem: str) -> dict[str, Any]:
    stem = c.validate_stem(stem)
    dp = c.draft_path(stem)
    with _write_lock:
        if dp.is_file():
            dp.unlink()
    return {"ok": True}


# --- HTTP ---------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    corpus: Corpus  # set by make_server()
    server_version = "drum-reviewer/1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet audio/api chatter
        if args and isinstance(args[0], str) and ("/audio/" in args[0] or "/api/" in args[0]):
            if len(args) > 1 and str(args[1]).startswith(("2", "3")):
                return
        super().log_message(fmt, *args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stem_param(self, q: dict[str, list[str]]) -> str:
        vals = q.get("stem")
        if not vals:
            raise ApiError(400, "missing ?stem=")
        return self.corpus.validate_stem(vals[0])

    def _dispatch(self, fn) -> None:
        try:
            fn()
        except ApiError as e:
            self._send_json({"error": e.message}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # a malformed label file, a bug: say so, loudly
            self.log_error("%s %s failed: %r", self.command, self.path, e)
            self._send_json({"error": f"server error: {e!r}"}, 500)

    def do_GET(self) -> None:
        self._dispatch(self._get)

    def do_POST(self) -> None:
        self._dispatch(self._post)

    def _get(self) -> None:
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        if path in ("/", "/index.html"):
            body = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/tracks":
            self._send_json({"tracks": list_tracks(self.corpus), "edm": str(self.corpus.edm)})
        elif path == "/api/track":
            self._send_json(load_track(self.corpus, self._stem_param(q)))
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
        routes = {"/api/save": save_track, "/api/draft": save_draft}
        if path in routes:
            stem = self._stem_param(q)
            body = self._read_body()
            self._send_json(routes[path](self.corpus, stem, body))
        elif path == "/api/discard-draft":
            self._send_json(discard_draft(self.corpus, self._stem_param(q)))
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


def make_server(edm: Path, host: str = "127.0.0.1", port: int = 8766) -> ThreadingHTTPServer:
    corpus = Corpus(edm)
    if not corpus.edm.is_dir():
        raise SystemExit(f"edm dir not found: {corpus.edm}")
    handler = type("BoundHandler", (Handler,), {"corpus": corpus})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) for this machine only; 0.0.0.0 to serve the LAN")
    ap.add_argument("--edm", type=Path, default=DEFAULT_EDM, help=f"corpus dir (default {DEFAULT_EDM})")
    args = ap.parse_args()
    srv = make_server(args.edm, args.host, args.port)
    c: Corpus = srv.RequestHandlerClass.corpus  # type: ignore[attr-defined]
    stems = c.stems()
    print(f"drum-reviewer: {len(stems)} tracks in {c.edm}")
    print(f"  human GT -> {c.human_dir}   drafts -> {c.draft_dir}")
    print(f"  open http://{'localhost' if args.host in ('127.0.0.1', 'localhost') else args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
