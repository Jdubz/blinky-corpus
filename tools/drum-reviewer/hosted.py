"""Building blocks for the hosted (multi-user, public) drum reviewer. server.py wires them together.

* JsonLog: one JSON object per line on stdout, for the log pipeline.
* ip_hash: a keyed hash of the client address, so logs and commit trailers can tell clients apart
  without recording an address.
* RateLimiter: token buckets per (kind, client).
* LockTable: short-lived per-track edit locks, persisted across restarts.
* DraftStore: drafts per track and contributor, server-side.
* PendingStore: saves that are open pull requests, until the corpus checkout shows them merged.
* GitHub / Publisher: a completed save becomes a branch commit and a pull request with auto-merge.
* Checkout: the service's own clone of the corpus, refreshed from GitHub.

Python standard library only.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
NAME_MAX = 40
NAME_PUNCT = " .'-_"  # besides letters, marks and digits in any script: no '@', markup or control chars


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def git_blob_sha(data: bytes) -> str:
    """The sha git (and GitHub's contents API) gives a file with these bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def name_ok(name: str) -> bool:
    """The page checks the same rule (index.html NAME_RE: [\\p{L}\\p{M}\\p{N}_ .'-]{0,40} after NFC)."""
    return len(name) <= NAME_MAX and all(
        unicodedata.category(ch)[0] in "LMN" or ch in NAME_PUNCT for ch in name)


def normalize_name(raw: str | None) -> str:
    """A contributor's self-entered name: NFC, trimmed, single-spaced, at most 40 letters, marks,
    digits (any script), spaces and . ' - _. Raises ValueError otherwise ('' means anonymous)."""
    name = " ".join(unicodedata.normalize("NFC", raw or "").split())
    if not name_ok(name):
        raise ValueError("a name may use up to 40 letters, digits, spaces and . ' - _")
    return name


# --- logging ------------------------------------------------------------------------------------


class JsonLog:
    def __init__(self, stream: TextIO | None = None):
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        rec = {"ts": now_iso(), "event": event, **{k: v for k, v in fields.items() if v is not None}}
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()


class Salt:
    """The key for ip_hash, created once in the state directory (0600). Losing it only means hashes
    from before and after stop matching."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            atomic_write_text(path, secrets.token_hex(32) + "\n", mode=0o600)
        self.key = bytes.fromhex(path.read_text().strip())

    def ip_hash(self, ip: str) -> str:
        return hmac.new(self.key, ip.encode(), hashlib.sha256).hexdigest()[:16]


# --- rate limits --------------------------------------------------------------------------------


class RateLimiter:
    """Token buckets. rules: {kind: (capacity, seconds to refill one token... as capacity per
    `period` seconds)}. allow() takes one token and returns (True, 0) or (False, retry_after_s)."""

    def __init__(self, rules: dict[str, tuple[int, float]], clock: Callable[[], float] = time.monotonic):
        self.rules = rules
        self.clock = clock
        self._b: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, kind: str, key: str) -> tuple[bool, int]:
        cap, period = self.rules[kind]
        rate = cap / period
        now = self.clock()
        with self._lock:
            tokens, last = self._b.get((kind, key), (float(cap), now))
            tokens = min(float(cap), tokens + (now - last) * rate)
            if tokens >= 1:
                self._b[(kind, key)] = (tokens - 1, now)
                ok, wait = True, 0
            else:
                self._b[(kind, key)] = (tokens, now)
                ok, wait = False, max(1, int((1 - tokens) / rate + 0.999))
            if len(self._b) > 20_000:  # bound memory: forget buckets that have refilled
                for k, (t, l) in list(self._b.items()):
                    c, p = self.rules[k[0]]
                    if t + (now - l) * c / p >= c:
                        del self._b[k]
        return ok, wait


# --- locks --------------------------------------------------------------------------------------


@dataclass
class Lock:
    stem: str
    client: str
    name: str
    ip_hash: str
    acquired: float
    expires: float

    def public(self, me: str | None = None, ttl: float | None = None) -> dict[str, Any]:
        return {
            "name": self.name or "someone",
            "until": _dt.datetime.fromtimestamp(self.expires, _dt.timezone.utc)
            .replace(microsecond=0).isoformat(),
            "mine": me is not None and me == self.client,
        }


class Locked(Exception):
    def __init__(self, lock: Lock, reason: str):
        super().__init__(reason)
        self.lock = lock
        self.reason = reason


class LockTable:
    """One edit lock per track. A lock lasts `ttl` seconds from its last renewal. A client holds at
    most one lock (taking a new one drops the old), and one address at most `max_per_ip`."""

    def __init__(self, path: Path, ttl: float = 900.0, max_per_ip: int = 3,
                 clock: Callable[[], float] = time.time):
        self.path, self.ttl, self.max_per_ip, self.clock = path, ttl, max_per_ip, clock
        self._lock = threading.Lock()
        self._locks: dict[str, Lock] = {}
        if path.is_file():
            try:
                for d in json.loads(path.read_text()).get("locks", []):
                    lk = Lock(**d)
                    self._locks[lk.stem] = lk
            except (ValueError, TypeError):
                self._locks = {}  # a corrupt lock file only costs the current locks

    def _save(self) -> None:
        atomic_write_text(self.path, json.dumps({"locks": [asdict(l) for l in self._locks.values()]}))

    def _expire(self) -> bool:
        now = self.clock()
        dead = [s for s, l in self._locks.items() if l.expires <= now]
        for s in dead:
            del self._locks[s]
        return bool(dead)

    def get(self, stem: str) -> Lock | None:
        with self._lock:
            if self._expire():
                self._save()
            return self._locks.get(stem)

    def all(self) -> dict[str, Lock]:
        with self._lock:
            if self._expire():
                self._save()
            return dict(self._locks)

    def acquire(self, stem: str, client: str, name: str, ip_hash: str,
                takeover: bool = False) -> tuple[Lock, bool]:
        """Take or renew the lock. Returns (lock, newly_acquired). Raises Locked when someone else
        holds it. `takeover` hands over a lock held under the same non-empty name (the same person
        on another machine)."""
        with self._lock:
            self._expire()
            now = self.clock()
            cur = self._locks.get(stem)
            if cur is not None and cur.client != client:
                same_person = takeover and name and cur.name == name
                if not same_person:
                    raise Locked(cur, f"being edited by {cur.name or 'someone'}")
            new = cur is None or cur.client != client
            if new:
                held = [l for l in self._locks.values() if l.ip_hash == ip_hash and l.client != client]
                if len(held) >= self.max_per_ip:
                    raise Locked(held[0], f"your network already holds {len(held)} edit locks")
            for s, l in list(self._locks.items()):  # one lock per client
                if l.client == client and s != stem:
                    del self._locks[s]
            lk = Lock(stem, client, name, ip_hash, now if new else cur.acquired, now + self.ttl)
            self._locks[stem] = lk
            self._save()
            return lk, new

    def renew(self, stem: str, client: str) -> Lock:
        with self._lock:
            self._expire()
            cur = self._locks.get(stem)
            if cur is None or cur.client != client:
                raise Locked(cur or Lock(stem, "", "", "", 0, 0), "your edit lock has lapsed")
            cur.expires = self.clock() + self.ttl
            self._save()
            return cur

    def holds(self, stem: str, client: str) -> bool:
        lk = self.get(stem)
        return lk is not None and lk.client == client

    def release(self, stem: str, client: str) -> bool:
        with self._lock:
            cur = self._locks.get(stem)
            if cur is None or cur.client != client:
                return False
            del self._locks[stem]
            self._save()
            return True

    def drop(self, stem: str) -> None:
        with self._lock:
            if self._locks.pop(stem, None) is not None:
                self._save()


# --- drafts -------------------------------------------------------------------------------------


class DraftStore:
    """drafts/<stem>/<key>.json. A named contributor's key is derived from the name (case-folded),
    so the same name resumes from any machine; an anonymous contributor's key is the browser's id."""

    def __init__(self, root: Path, max_per_track: int = 100, max_total: int = 2000):
        self.root, self.max_per_track, self.max_total = root, max_per_track, max_total
        self._lock = threading.Lock()

    @staticmethod
    def key_for(name: str, client: str | None) -> str | None:
        if name:
            return "n-" + hashlib.sha256(name.casefold().encode()).hexdigest()[:20]
        if client and CLIENT_RE.match(client):
            return "c-" + client
        return None

    def path(self, stem: str, key: str) -> Path:
        if not re.fullmatch(r"[nc]-[A-Za-z0-9_-]{8,64}", key) or "/" in stem or stem.startswith("."):
            raise ValueError("bad draft key")
        return self.root / stem / f"{key}.json"

    def load(self, stem: str, key: str | None) -> dict[str, Any] | None:
        if key is None:
            return None
        p = self.path(stem, key)
        if not p.is_file():
            return None
        try:
            doc = json.loads(p.read_text())
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    def has(self, stem: str, key: str | None) -> bool:
        return key is not None and self.path(stem, key).is_file()

    def save(self, stem: str, key: str, text: str) -> None:
        p = self.path(stem, key)
        with self._lock:
            if not p.is_file():
                if len(list(p.parent.glob("*.json"))) >= self.max_per_track:
                    raise OverflowError(f"too many drafts for {stem}")
                if sum(1 for _ in self.root.glob("*/*.json")) >= self.max_total:
                    raise OverflowError("the draft store is full")
            atomic_write_text(p, text)

    def delete(self, stem: str, key: str | None) -> None:
        if key is None:
            return
        try:
            self.path(stem, key).unlink()
        except FileNotFoundError:
            pass


# --- pending saves (open pull requests) -----------------------------------------------------------


class PendingStore:
    """stem -> the save that is an open pull request. A track with one takes no further save until
    the corpus checkout shows it merged, or GitHub shows the PR closed."""

    ACTIVE = ("open", "merged")

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._d: dict[str, dict[str, Any]] = {}
        if path.is_file():
            self._d = json.loads(path.read_text()).get("pending", {})

    def _save(self) -> None:
        atomic_write_text(self.path, json.dumps({"pending": self._d}, indent=1))

    def get(self, stem: str) -> dict[str, Any] | None:
        with self._lock:
            p = self._d.get(stem)
            return dict(p) if p else None

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._d.items()}

    def put(self, stem: str, rec: dict[str, Any]) -> None:
        with self._lock:
            self._d[stem] = rec
            self._save()

    def update(self, stem: str, **fields: Any) -> None:
        with self._lock:
            if stem in self._d:
                self._d[stem].update(fields)
                self._save()

    def clear(self, stem: str) -> None:
        with self._lock:
            if self._d.pop(stem, None) is not None:
                self._save()


# --- GitHub -------------------------------------------------------------------------------------


class GitHubError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub {status}: {message}")
        self.status = status
        self.message = message


class Conflict(Exception):
    """The corpus on GitHub no longer matches what the contributor edited."""


class GitHub:
    """The few REST/GraphQL calls the publisher needs. The token is a fine-grained token on the
    corpus repo only: Contents read/write, Pull requests read/write."""

    def __init__(self, repo: str, token: str, api: str = "https://api.github.com", timeout: float = 20):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError(f"bad repo {repo!r}")
        if not token:
            raise ValueError("no GitHub token")
        self.repo, self._token, self.api, self.timeout = repo, token, api.rstrip("/"), timeout

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.api + path, data=data, method=method, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "blinky-drum-reviewer",
            **({"Content-Type": "application/json"} if data is not None else {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("message", "")
            except ValueError:
                msg = raw[:200].decode(errors="replace")
            raise GitHubError(e.code, msg) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise GitHubError(0, f"unreachable: {e}") from None

    def r(self, method: str, sub: str, body: Any = None) -> Any:
        return self.request(method, f"/repos/{self.repo}{sub}", body)[1]

    def graphql(self, query: str, variables: dict[str, Any]) -> Any:
        _, out = self.request("POST", "/graphql", {"query": query, "variables": variables})
        if out and out.get("errors"):
            raise GitHubError(200, "; ".join(e.get("message", "?") for e in out["errors"]))
        return (out or {}).get("data")

    # high level
    def branch_head(self, branch: str) -> str:
        return self.r("GET", f"/git/ref/heads/{branch}")["object"]["sha"]

    def file_sha(self, path: str, ref: str) -> str | None:
        try:
            return self.r("GET", f"/contents/{path}?ref={ref}")["sha"]
        except GitHubError as e:
            if e.status == 404:
                return None
            raise

    def create_branch(self, name: str, sha: str) -> None:
        self.r("POST", "/git/refs", {"ref": f"refs/heads/{name}", "sha": sha})

    def delete_branch(self, name: str) -> None:
        self.r("DELETE", f"/git/refs/heads/{name}")

    def put_file(self, branch: str, path: str, text: str, message: str, sha: str | None) -> str:
        body = {"message": message, "branch": branch,
                "content": base64.b64encode(text.encode()).decode()}
        if sha:
            body["sha"] = sha
        return self.r("PUT", f"/contents/{path}", body)["commit"]["sha"]

    def create_pr(self, head: str, base: str, title: str, body: str) -> dict[str, Any]:
        return self.r("POST", "/pulls", {"head": head, "base": base, "title": title, "body": body,
                                         "maintainer_can_modify": True})

    def enable_automerge(self, node_id: str, headline: str, body: str) -> None:
        self.graphql(
            "mutation($id: ID!, $h: String!, $b: String!) { enablePullRequestAutoMerge(input: "
            "{pullRequestId: $id, mergeMethod: SQUASH, commitHeadline: $h, commitBody: $b}) "
            "{ pullRequest { number } } }", {"id": node_id, "h": headline, "b": body})

    def get_pr(self, number: int) -> dict[str, Any]:
        return self.r("GET", f"/pulls/{int(number)}")

    def update_branch(self, number: int, head_sha: str) -> None:
        self.r("PUT", f"/pulls/{int(number)}/update-branch", {"expected_head_sha": head_sha})


class Publisher:
    """A completed save -> a branch `labels/<stem>-<UTC time>` with one commit, a pull request into
    the base branch, and auto-merge (squash). The branch is made from the base branch's current head,
    and only if the label file there is still the one the contributor started from."""

    def __init__(self, gh: GitHub, base: str = "main", log: JsonLog | None = None):
        self.gh, self.base, self.log = gh, base, log or JsonLog()
        self._lock = threading.Lock()  # one publish at a time

    def publish(self, stem: str, path: str, text: str, expected_sha: str | None,
                headline: str, body: str, trailers: str) -> dict[str, Any]:
        with self._lock:
            head = self.gh.branch_head(self.base)
            current = self.gh.file_sha(path, head)
            if current != expected_sha:
                raise Conflict(f"{path} on {self.base} changed since this track was opened")
            branch = f"labels/{stem}-{_dt.datetime.now(_dt.timezone.utc):%Y%m%d-%H%M%S}"
            self.gh.create_branch(branch, head)
            try:
                commit = self.gh.put_file(branch, path, text, f"{headline}\n\n{body}\n\n{trailers}",
                                          current)
                pr = self.gh.create_pr(branch, self.base, headline, f"{body}\n\n```\n{trailers}\n```")
            except GitHubError:
                try:
                    self.gh.delete_branch(branch)
                except GitHubError:
                    pass
                raise
            automerge, why = True, None
            try:
                self.gh.enable_automerge(pr["node_id"], headline, trailers)
            except GitHubError as e:  # the PR stands; the owner can merge it by hand
                automerge, why = False, e.message
            return {"pr": pr["number"], "url": pr["html_url"], "branch": branch, "commit": commit,
                    "automerge": automerge, "automerge_error": why}

    def poll(self, rec: dict[str, Any]) -> str:
        """The pull request's state: open, merged or closed. Brings a branch that has fallen behind
        the base up to date (the base requires up-to-date branches), so auto-merge can go on."""
        pr = self.gh.get_pr(rec["pr"])
        if pr.get("merged"):
            return "merged"
        if pr.get("state") == "closed":
            return "closed"
        if pr.get("mergeable_state") == "behind":
            try:
                self.gh.update_branch(rec["pr"], pr["head"]["sha"])
                self.log.emit("pr_update_branch", pr=rec["pr"], stem=rec.get("stem"))
            except GitHubError as e:
                self.log.emit("pr_update_branch_failed", pr=rec["pr"], error=e.message)
        return "open"

    def cleanup(self, rec: dict[str, Any]) -> None:
        try:
            self.gh.delete_branch(rec["branch"])
        except GitHubError:
            pass  # already gone (or the repo deletes merged branches itself)


# --- the corpus checkout ---------------------------------------------------------------------------


class RWLock:
    """Many readers or one writer: requests read the checkout, a refresh rewrites it."""

    def __init__(self) -> None:
        self._c = threading.Condition()
        self._readers = 0
        self._writer = False

    def read(self) -> "_Guard":
        return _Guard(self._acquire_r, self._release_r)

    def write(self) -> "_Guard":
        return _Guard(self._acquire_w, self._release_w)

    def _acquire_r(self) -> None:
        with self._c:
            while self._writer:
                self._c.wait()
            self._readers += 1

    def _release_r(self) -> None:
        with self._c:
            self._readers -= 1
            self._c.notify_all()

    def _acquire_w(self) -> None:
        with self._c:
            while self._writer or self._readers:
                self._c.wait()
            self._writer = True

    def _release_w(self) -> None:
        with self._c:
            self._writer = False
            self._c.notify_all()


class _Guard:
    def __init__(self, a: Callable[[], None], r: Callable[[], None]):
        self.a, self.r = a, r

    def __enter__(self) -> None:
        self.a()

    def __exit__(self, *exc: Any) -> None:
        self.r()


class Checkout:
    """The service's own clone of the corpus (never anyone's working copy): cloned on first start,
    then fetched and hard-reset to origin/<branch> when it moves. Only data is read from it; the
    code that runs is the image's."""

    def __init__(self, root: Path, url: str, branch: str = "main", rw: RWLock | None = None,
                 log: JsonLog | None = None):
        self.root, self.url, self.branch = root, url, branch
        self.rw = rw or RWLock()
        self.log = log or JsonLog()

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        r = subprocess.run(["git", *args], cwd=cwd or self.root, capture_output=True, text=True,
                           timeout=600, env=env)
        if r.returncode:
            raise RuntimeError(f"git {' '.join(args[:2])} failed: {r.stderr.strip()[-400:]}")
        return r.stdout

    def ensure(self) -> None:
        if (self.root / ".git").is_dir():
            return
        self.root.parent.mkdir(parents=True, exist_ok=True)
        with self.rw.write():
            self._git("clone", "--branch", self.branch, "--single-branch", "--no-tags", self.url,
                      str(self.root), cwd=self.root.parent)
        self.log.emit("checkout_cloned", head=self.head())

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def refresh(self) -> bool:
        """Fetch; when origin moved, reset the working tree to it. True if it changed."""
        self._git("fetch", "--no-tags", "--prune", "origin", f"+refs/heads/{self.branch}:"
                  f"refs/remotes/origin/{self.branch}")
        new = self._git("rev-parse", f"refs/remotes/origin/{self.branch}").strip()
        if new == self.head():
            return False
        with self.rw.write():
            self._git("reset", "--hard", "--quiet", new)
            self._git("clean", "-fdq")
        self.log.emit("checkout_refreshed", head=new)
        return True
