"""Hosted (multi-user) drum reviewer: locks, versions, drafts, rate limits, CORS, and the save ->
branch -> pull request path. Saves are pushed to a local bare git repository standing in for GitHub,
and tests/fake_github.py serves the one (unauthenticated) pull-request lookup. No network."""

from __future__ import annotations

import base64
import hashlib
import http.client
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "drum-reviewer"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location("drum_reviewer_server_hosted", TOOLS / "server.py")
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)
import hosted  # noqa: E402
import labelcheck  # noqa: E402
from fake_github import FakeGitHub  # noqa: E402

DUR = 10.0
REPO = "owner/corpus"
SITE = "https://site.example.invalid"
KICKS = [0.5 + i for i in range(10)]
SNARES = [1.0 + i for i in range(9)]
HUMAN_A = "edm/drums_human/a.drums.json"


def _w(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _kw(kicks, snares, **extra):
    ons = [{"time": t, "type": "kick", "weight": 1.0} for t in kicks]
    ons += [{"time": t, "type": "snare", "weight": 1.0} for t in snares]
    return {"onsets": sorted(ons, key=lambda o: o["time"]), **extra}


@pytest.fixture
def edm(tmp_path):
    d = tmp_path / "corpus" / "edm"
    d.mkdir(parents=True)
    for stem in ("a", "b", "c"):
        (d / f"{stem}.mp3").write_bytes(f"fake mp3 {stem}".encode() * 50)
        _w(d / "drums_adtof" / f"{stem}.kick_weighted.json",
           _kw(KICKS, SNARES, generator={"algorithm": "adtof_frame_rnn"}))
        _w(d / "kick_weighted" / f"{stem}.kick_weighted.json", _kw(KICKS + SNARES, []))
    _w(d / "track_manifest.json", {s: {"duration": DUR} for s in ("a", "b", "c")})
    return d


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(cwd, *args, input=None):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          cwd=cwd, check=True, capture_output=True, text=True, input=input).stdout


@pytest.fixture
def gh():
    fake = FakeGitHub(REPO)
    url = fake.start()
    yield fake, url
    fake.stop()


class Env:
    def __init__(self, app, port, log, fake, edm, origin=None, work=None):
        self.app, self.port, self.log, self.fake, self.edm = app, port, log, fake, edm
        self.origin, self.work = origin, work

    def req(self, method, path, body=None, client="client-aaaa", name="Ana", headers=None):
        h = {"Host": "api.example.invalid"}
        if client:
            h["X-Reviewer-Client"] = client
        if name is not None:
            h["X-Reviewer-Name"] = name
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h.update({"Content-Type": "application/json", "Content-Length": str(len(data))})
        h.update(headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        raw = r.read()
        out = (r.status, dict(r.getheaders()), json.loads(raw) if raw and r.getheader(
            "Content-Type", "").startswith("application/json") else raw)
        conn.close()
        return out

    def events(self, name=None):
        evs = [json.loads(line) for line in self.log.getvalue().splitlines()]
        return [e for e in evs if name is None or e["event"] == name]

    # -- the stand-in for GitHub's repo
    def branches(self):
        out = git(self.origin, "for-each-ref", "--format=%(refname:short)", "refs/heads/labels/")
        return out.split()

    def show(self, rev, path=None):
        return git(self.origin, "show", f"{rev}:{path}" if path else rev)

    def push_to_main(self, files: dict[str, str], msg="change on main"):
        """Someone else's merge lands on main."""
        git(self.work, "pull", "-q", "origin", "main")
        for rel, text in files.items():
            p = self.work / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        git(self.work, "add", "-A")
        git(self.work, "commit", "-qm", msg)
        git(self.work, "push", "-q", "origin", "HEAD:main")


def make_env(edm, tmp_path, fake_url, fake, *, limits=None, publisher=True, trust_cf=False):
    log = io.StringIO()
    jl = hosted.JsonLog(log)
    tmp_path.mkdir(parents=True, exist_ok=True)
    origin = work = None
    corpus_edm = edm
    pub = None
    if publisher:
        src = edm.parent
        if not (src / ".git").exists():
            git(src, "init", "-q", "-b", "main")
            git(src, "add", "-A")
            git(src, "commit", "-qm", "corpus")
        origin = tmp_path / "origin.git"
        git(tmp_path, "clone", "-q", "--bare", str(src), str(origin))
        work = tmp_path / "work"
        git(tmp_path, "clone", "-q", str(origin), str(work))
        co = hosted.Checkout(tmp_path / "co", origin.as_uri(), log=jl)
        co.ensure()
        corpus_edm = co.root / "edm"
        pub = hosted.GitPusher(co, origin.as_uri(), hosted.PublicGitHub(REPO, fake_url),
                               poll_every=0, log=jl)
    app = srv.HostedApp(srv.Corpus(corpus_edm), tmp_path / "state", pub,
                        pub.co if pub else None, log=jl, limits=limits, trust_cf=trust_cf)
    s = srv.make_server(corpus_edm, "127.0.0.1", 0, ("api.example.invalid",), app, (SITE,))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, Env(app, s.server_address[1], log, fake, corpus_edm, origin, work)


@pytest.fixture
def env(edm, tmp_path, gh):
    fake, url = gh
    s, e = make_env(edm, tmp_path / "env", url, fake)
    yield e
    s.shutdown()
    s.server_close()


def seed_body(e, stem, *, ranges=((0, DUR),), onsets=None, version=None):
    t = e.req("GET", f"/api/track?stem={stem}")[2]
    ons = onsets if onsets is not None else t["onsets"]
    return {"audio_sha256": t["audio_sha256"], "duration": DUR, "started_from": t["started_from"],
            "seed": t["seed"], "onsets": ons, "reviewed_ranges": [list(r) for r in ranges],
            "base_version": t["version"] if version is None else version}


def lock(e, stem, client="client-aaaa", name="Ana", headers=None, **kw):
    return e.req("POST", f"/api/lock?stem={stem}", {"action": "acquire", **kw}, client=client, name=name,
                 headers=headers)


# --- building blocks ----------------------------------------------------------------------------


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_lock_table(tmp_path):
    clk = Clock()
    lt = hosted.LockTable(tmp_path / "locks.json", ttl=900, max_per_ip=2, clock=clk)
    lk, new = lt.acquire("a", "c1", "Ana", "ip1")
    assert new and lt.holds("a", "c1")
    with pytest.raises(hosted.Locked) as e:
        lt.acquire("a", "c2", "Bo", "ip2")
    assert e.value.lock.name == "Ana"
    clk.t += 600
    lt.renew("a", "c1")  # renewal pushes the expiry out again
    clk.t += 600
    with pytest.raises(hosted.Locked):
        lt.acquire("a", "c2", "Bo", "ip2")
    with pytest.raises(hosted.Locked):
        lt.renew("a", "c2")  # only the holder renews
    clk.t += 901  # idle past the TTL: the lock lapses and someone else can take it
    lk, new = lt.acquire("a", "c2", "Bo", "ip2")
    assert new and lk.client == "c2"
    with pytest.raises(hosted.Locked):
        lt.renew("a", "c1")
    # the same name on another machine may take over; another name may not
    with pytest.raises(hosted.Locked):
        lt.acquire("a", "c3", "Cy", "ip3", takeover=True)
    lt.acquire("a", "c3", "Bo", "ip3", takeover=True)
    assert lt.get("a").client == "c3"
    # one lock per client: taking b drops a
    lt.acquire("b", "c3", "Bo", "ip3")
    assert lt.get("a") is None and lt.get("b").client == "c3"
    # at most max_per_ip locks per address
    lt.acquire("x", "c4", "", "ip9")
    lt.acquire("y", "c5", "", "ip9")
    with pytest.raises(hosted.Locked) as e:
        lt.acquire("z", "c6", "", "ip9")
    assert "network" in e.value.reason
    assert not lt.release("b", "c1") and lt.release("b", "c3")
    # persisted: a restart keeps the locks
    lt2 = hosted.LockTable(tmp_path / "locks.json", ttl=900, clock=clk)
    assert set(lt2.all()) == {"x", "y"}


def test_rate_limiter():
    clk = Clock(0)
    rl = hosted.RateLimiter({"save": (2, 3600.0)}, clock=clk)
    assert rl.allow("save", "ip1") == (True, 0)
    assert rl.allow("save", "ip1") == (True, 0)
    ok, wait = rl.allow("save", "ip1")
    assert not ok and 1700 <= wait <= 1800  # one token per 30 min
    assert rl.allow("save", "ip2")[0]  # per client
    clk.t += 1800
    assert rl.allow("save", "ip1")[0]


@pytest.mark.parametrize("raw,ok", [("Ana", "Ana"), ("  José   M. ", "José M."), ("", ""),
                                    ("O'Neil-Smith_2", "O'Neil-Smith_2"), ("李雷", "李雷"),
                                    ("अमित", "अमित"), ("Jose\u0301", "José")])  # marks; NFD -> NFC
def test_names(raw, ok):
    assert hosted.normalize_name(raw) == ok


@pytest.mark.parametrize("raw", ["@owner", "a" * 41, "<b>x</b>", "x`y", "nul\x00byte", "zero\u200bwidth",
                                 "Ana\U0001F600"])
def test_bad_names(raw):
    with pytest.raises(ValueError):
        hosted.normalize_name(raw)


def test_git_blob_sha_matches_git(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"hello\n")
    out = subprocess.run(["git", "hash-object", str(f)], capture_output=True, text=True, check=True)
    assert hosted.git_blob_sha(b"hello\n") == out.stdout.strip()


# --- locks, drafts and versions over HTTP --------------------------------------------------------


def test_lock_is_needed_to_write(env):
    st, _, b = lock(env, "a")
    assert st == 200 and b["lock"]["mine"]
    st, _, b = lock(env, "a", client="client-bbbb", name="Bo")
    assert st == 423 and b["lock"]["name"] == "Ana" and not b["lock"]["mine"]
    body = seed_body(env, "a", ranges=[(0, 1)])
    st, _, b = env.req("POST", "/api/draft?stem=a", body, client="client-bbbb", name="Bo")
    assert st == 423 and "Ana" in b["error"]
    st, _, b = env.req("POST", "/api/save?stem=a", body, client="client-bbbb", name="Bo")
    assert st == 423
    assert env.req("POST", "/api/draft?stem=a", body)[0] == 200
    tracks = {t["stem"]: t for t in env.req("GET", "/api/tracks", client="client-bbbb", name="Bo")[2]["tracks"]}
    assert tracks["a"]["lock"] == {"name": "Ana", "until": tracks["a"]["lock"]["until"], "mine": False}
    # release: now Bo can take it
    assert env.req("POST", "/api/lock?stem=a", {"action": "release"})[0] == 200
    assert lock(env, "a", client="client-bbbb", name="Bo")[0] == 200


def test_drafts_follow_the_name(env):
    lock(env, "a")
    assert env.req("POST", "/api/draft?stem=a", seed_body(env, "a", ranges=[(0, 2)]))[0] == 200
    # the same name (any case) on another machine resumes it
    t = env.req("GET", "/api/track?stem=a", client="client-other", name="ana")[2]
    assert t["draft"]["reviewed_ranges"] == [[0, 2]] and t["draft"]["contributor"] == "Ana"
    assert {x["stem"]: x for x in env.req("GET", "/api/tracks", client="client-other", name="ANA")[2]
            ["tracks"]}["a"]["has_draft"]
    # someone else doesn't see it
    assert env.req("GET", "/api/track?stem=a", client="client-bbbb", name="Bo")[2]["draft"] is None
    # anonymous drafts belong to the browser (tab) id
    env.req("POST", "/api/lock?stem=a", {"action": "release"})
    lock(env, "a", client="client-anon1", name="")
    assert env.req("POST", "/api/draft?stem=a", seed_body(env, "a", ranges=[(0, 3)]),
                   client="client-anon1", name="")[0] == 200
    assert env.req("GET", "/api/track?stem=a", client="client-anon1", name="")[2]["draft"]
    assert env.req("GET", "/api/track?stem=a", client="client-anon2", name="")[2]["draft"] is None


def test_bad_identity_headers(env):
    assert env.req("GET", "/api/tracks", client="x")[0] == 400  # too short
    assert env.req("GET", "/api/tracks", name="%40owner")[0] == 400
    assert env.req("POST", "/api/lock?stem=a", {"action": "acquire"}, client=None)[0] == 400


# --- saving -> a pushed branch -> the workflow's pull request --------------------------------


def test_incomplete_save_is_kept_as_draft(env):
    lock(env, "a")
    st, _, b = env.req("POST", "/api/save?stem=a", seed_body(env, "a", ranges=[(0, 5)]))
    assert st == 422 and "100%" in b["error"]
    assert env.req("GET", "/api/track?stem=a")[2]["draft"]["reviewed_ranges"] == [[0, 5]]
    assert env.branches() == []


def human_text(env, stem="a", kicks=KICKS):
    """A valid human label file someone else saved (100% reviewed)."""
    body = {"audio_sha256": sha(env.edm / f"{stem}.mp3"), "duration": DUR,
            "started_from": {"source": "drums_adtof"}, "reviewed_ranges": [[0, DUR]],
            "onsets": [{"time": t, "type": "kick"} for t in kicks] + [{"time": t, "type": "snare"} for t in SNARES]}
    return labelcheck.dumps_doc(srv.validate_doc(env.app.c, stem, body, existing=None))


def saved(env, stem="a"):
    lock(env, stem)
    st, _, b = env.req("POST", f"/api/save?stem={stem}", seed_body(env, stem))
    assert st == 200, b
    return b


def test_save_pushes_one_commit_on_a_labels_branch(env):
    lock(env, "a")
    env.req("POST", "/api/draft?stem=a", seed_body(env, "a", ranges=[(0, 1)]))
    st, _, b = env.req("POST", "/api/save?stem=a", seed_body(env, "a"))
    assert st == 200, b
    (branch,) = env.branches()
    assert b["branch"] == branch and branch.startswith("labels/a-") and b["pr"] is None
    assert b["url"] == f"https://github.com/{REPO}/tree/{branch}"
    # one commit on top of main, changing exactly the one label file
    main = git(env.origin, "rev-parse", "main").strip()
    assert git(env.origin, "rev-parse", f"{branch}^").strip() == main
    assert git(env.origin, "diff", "--name-status", main, branch).split() == ["A", HUMAN_A]
    msg = git(env.origin, "log", "-1", "--format=%B", branch)
    assert msg.startswith("labels(a): human drum labels, 10 kicks, 9 snares")
    assert "Contributed-by: Ana" in msg
    ip_hash = next(line.split(": ")[1] for line in msg.splitlines() if line.startswith("Client-IP-Hash"))
    assert len(ip_hash) == 16 and "127.0.0.1" not in msg
    assert git(env.origin, "log", "-1", "--format=%an", branch).strip() == "Ana (drum reviewer)"
    # the committed file is exactly what corpus CI accepts
    doc = json.loads(env.show(branch, HUMAN_A))
    assert labelcheck.check_human_doc(env.edm, "a", doc, None, labelcheck.load_seeds()) == []
    # the service's clone: working tree untouched
    assert not (env.edm / "drums_human" / "a.drums.json").exists()
    assert git(env.app.checkout.root, "status", "--porcelain") == ""
    # afterwards: pending, the lock released, the draft gone, the save logged
    t = env.req("GET", "/api/track?stem=a")[2]
    assert t["pending"]["branch"] == branch and t["pending"]["pr"] is None and t["seed"] == "human"
    assert t["lock"] is None and t["draft"] is None
    (save,) = env.events("save")
    assert save["branch"] == branch and save["contributor"] == "Ana" and save["ip"] == ip_hash
    assert json.loads((env.app.state / "saves.jsonl").read_text())["branch"] == branch
    # the workflow opens the PR; the next tick picks it up (one unauthenticated lookup)
    env.fake.open_pr(branch)
    env.app.tick()
    t = env.req("GET", "/api/track?stem=a")[2]
    assert t["pending"]["pr"] == 1 and t["pending"]["url"].endswith("/pull/1")
    assert env.events("pr_opened")[0]["pr"] == 1 and env.fake.auth_seen == []  # no token, ever
    # a pending track takes no further save
    lock(env, "a")
    st, _, b = env.req("POST", "/api/save?stem=a", seed_body(env, "a"))
    assert st == 409 and "PR #1" in b["error"]


def test_stale_version_is_refused(env):
    lock(env, "a")
    body = seed_body(env, "a")
    assert body["base_version"] == "none"
    # someone's save lands in the checkout meanwhile
    human = env.edm / "drums_human" / "a.drums.json"
    doc = srv.validate_doc(env.app.c, "a", {**body, "reviewed_ranges": [[0, DUR]]}, existing=None)
    human.parent.mkdir(exist_ok=True)
    human.write_text(labelcheck.dumps_doc(doc))
    st, _, b = env.req("POST", "/api/save?stem=a", body)
    assert st == 409 and "since you opened" in b["error"]
    human.unlink()
    assert env.branches() == []


def test_conflict_when_main_moved_under_the_file(env):
    lock(env, "a")
    body = seed_body(env, "a")
    env.push_to_main({HUMAN_A: human_text(env)})  # the clone hasn't refreshed: publish fetches, sees it
    st, _, b = env.req("POST", "/api/save?stem=a", body)
    assert st == 409 and "changed" in b["error"]
    assert env.branches() == [] and env.req("GET", "/api/track?stem=a")[2]["draft"]


def test_push_failure_keeps_the_work(env):
    lock(env, "a")
    env.app.publisher.push_url = (env.origin.parent / "nowhere.git").as_uri()
    st, _, b = env.req("POST", "/api/save?stem=a", seed_body(env, "a"))
    assert st == 502 and "draft" in b["error"]
    assert env.req("GET", "/api/track?stem=a")[2]["pending"] is None
    assert env.req("GET", "/api/track?stem=a")[2]["draft"] and env.events("save_failed")


def test_implausible_save_is_refused(env):
    lock(env, "a")
    few = [{"time": 0.5, "type": "kick"}, {"time": 1.0, "type": "snare"}]
    st, _, b = env.req("POST", "/api/save?stem=a", seed_body(env, "a", onsets=few))
    assert st == 422 and any("deletes" in p for p in b["problems"])
    assert env.branches() == [] and env.events("save_refused")


def test_save_rate_limit(edm, tmp_path, gh):
    fake, url = gh
    s, e = make_env(edm, tmp_path / "e", url, fake, limits={**srv.DEFAULT_LIMITS, "save": (1, 3600.0)})
    try:
        saved(e, "a")
        lock(e, "b")
        st, h, b = e.req("POST", "/api/save?stem=b", seed_body(e, "b"))
        assert st == 429 and int(h["Retry-After"]) > 0
        assert e.events("rate_limited")[0]["kind"] == "save"
    finally:
        s.shutdown()
        s.server_close()


def test_lock_grab_rate_limit(edm, tmp_path, gh):
    fake, url = gh
    s, e = make_env(edm, tmp_path / "e", url, fake, limits={**srv.DEFAULT_LIMITS, "lock": (2, 3600.0)})
    try:
        assert lock(e, "a")[0] == 200
        assert lock(e, "a")[0] == 200  # renewing your own lock is free
        assert lock(e, "b")[0] == 200
        assert lock(e, "c")[0] == 429
    finally:
        s.shutdown()
        s.server_close()


def test_publishing_disabled(edm, tmp_path, gh):
    fake, url = gh
    s, e = make_env(edm, tmp_path / "e", url, fake, publisher=False)
    try:
        assert e.req("GET", "/api/config")[2]["saving"] is False
        lock(e, "a")
        assert e.req("POST", "/api/draft?stem=a", seed_body(e, "a"))[0] == 200
        st, _, b = e.req("POST", "/api/save?stem=a", seed_body(e, "a"))
        assert st == 503 and "draft" in b["error"]
    finally:
        s.shutdown()
        s.server_close()


# --- following the branch and its pull request --------------------------------------------------


def test_merged_then_landed(env):
    b = saved(env)
    text = env.show(b["branch"], HUMAN_A)
    env.app.tick()
    assert env.app.pending.get("a")["state"] == "open"  # no PR yet: waiting for the workflow
    env.fake.open_pr(b["branch"])
    env.fake.prs[1].update(state="closed", merged_at="2026-11-26T00:00:00Z")
    env.app.tick()
    assert env.app.pending.get("a")["state"] == "merged" and env.branches() == []  # branch deleted
    assert env.req("GET", "/api/track?stem=a")[2]["pending"]["state"] == "merged"
    env.push_to_main({HUMAN_A: text}, "the squash merge")
    env.app.tick()  # refreshes the clone, sees the file landed
    assert env.app.pending.get("a") is None
    t = env.req("GET", "/api/track?stem=a")[2]
    assert t["pending"] is None and t["seed"] == "human" and t["version"].startswith("m-")


def test_own_merge_seen_on_main_first_is_not_a_conflict(env):
    """The squash merge reaches main (and the clone) before the throttled PR lookup reports it:
    main then holds this save's own file, which is a landing, not a conflict."""
    b = saved(env)
    text = env.show(b["branch"], HUMAN_A)
    env.fake.open_pr(b["branch"])
    env.app.tick()  # PR picked up, still open
    env.push_to_main({HUMAN_A: text}, "the squash merge")  # the lookup hasn't said "merged" yet
    env.app.publisher.poll_every = 3600  # and won't, this tick
    env.app.tick()
    assert not env.events("pr_conflict") and not env.events("branch_rebased")
    assert env.app.pending.get("a") is None and env.events("pr_landed")[0]["in_checkout"] is True
    t = env.req("GET", "/api/track?stem=a")[2]
    assert t["draft"] is None and t["seed"] == "human" and t["pending"] is None
    assert env.branches() == []


def test_closed_pr_comes_back_as_a_draft(env):
    b = saved(env)
    onsets = json.loads(env.show(b["branch"], HUMAN_A))["onsets"]
    env.fake.open_pr(b["branch"])
    env.fake.prs[1]["state"] = "closed"
    env.app.tick()
    assert env.app.pending.get("a") is None and env.events("pr_closed")
    t = env.req("GET", "/api/track?stem=a")[2]
    assert t["pending"] is None and t["draft"]["onsets"] == onsets
    assert env.branches() == []


def test_branch_is_rebuilt_when_main_moves(env):
    b = saved(env)
    env.fake.open_pr(b["branch"])
    env.push_to_main({"README.md": "someone else's change\n"})
    env.app.tick()
    main = git(env.origin, "rev-parse", "main").strip()
    assert git(env.origin, "rev-parse", f"{b['branch']}^").strip() == main  # up to date again
    assert git(env.origin, "diff", "--name-only", main, b["branch"]).split() == [HUMAN_A]
    assert env.app.pending.get("a")["base"] == main and env.events("branch_rebased")
    env.app.tick()  # nothing more to do
    assert len(env.events("branch_rebased")) == 1


def test_main_changing_the_same_file_ends_the_save(env):
    b = saved(env)
    env.fake.open_pr(b["branch"])
    env.push_to_main({HUMAN_A: human_text(env, kicks=KICKS[:-1])}, "a different save of the same track")
    env.app.tick()
    assert env.app.pending.get("a") is None and env.events("pr_conflict")
    assert env.req("GET", "/api/track?stem=a")[2]["draft"]  # the labels are back as a draft
    assert env.branches() == []


def test_lookup_failures_and_throttling(env):
    b = saved(env)
    env.fake.fail_next = 403  # e.g. the unauthenticated rate limit
    env.app.tick()
    assert env.events("pr_poll_failed") and env.app.pending.get("a")["state"] == "open"
    clock = [100.0]
    p = hosted.GitPusher(env.app.checkout, env.origin.as_uri(), env.app.publisher.gh,
                         poll_every=65, clock=lambda: clock[0])
    rec = env.app.pending.get("a")
    assert p.poll(rec)[0] == "waiting"
    assert p.poll(rec)[0] == "skip"  # at most one lookup per 65 s across all saves
    clock[0] += 66
    env.fake.open_pr(b["branch"])
    assert p.poll(rec)[0] == "open"


def test_ssh_pushes_pin_github_host_keys(tmp_path):
    co = hosted.Checkout(tmp_path / "co", "file:///x")
    gh = hosted.PublicGitHub(REPO)
    with pytest.raises(ValueError):
        hosted.GitPusher(co, "ssh://git@ssh.github.com:443/owner/corpus.git", gh)
    p = hosted.GitPusher(co, "ssh://git@ssh.github.com:443/owner/corpus.git", gh,
                         key_file=tmp_path / "k", known_hosts=TOOLS / "github_known_hosts")
    cmd = p._env["GIT_SSH_COMMAND"]
    assert "StrictHostKeyChecking=yes" in cmd and f"UserKnownHostsFile={TOOLS / 'github_known_hosts'}" in cmd
    assert "GlobalKnownHostsFile=/dev/null" in cmd and "IdentitiesOnly=yes" in cmd


def test_known_hosts_are_githubs_published_keys():
    # https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
    published = {"SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU", "SHA256:p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM",
                 "SHA256:uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s"}
    out = subprocess.run(["ssh-keygen", "-lf", str(TOOLS / "github_known_hosts")], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    by_host: dict[str, set[str]] = {}
    for line in out:
        bits, fp, host = line.split()[:3]
        by_host.setdefault(host, set()).add(fp)
    assert by_host == {"github.com": published, "[ssh.github.com]:443": published}


def test_deploy_key_file(tmp_path):
    assert srv.deploy_key_file("") is None
    with pytest.raises(SystemExit):
        srv.deploy_key_file(base64.b64encode(b"not a key").decode())
    key = b"-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\n"
    p = srv.deploy_key_file(base64.b64encode(key).decode())
    try:
        assert p.read_bytes() == key and (p.stat().st_mode & 0o777) == 0o600 and str(p).startswith("/tmp/")
    finally:
        os.unlink(p)


# --- HTTP surface: CORS, client address, headers, logs -------------------------------------------


def test_cors(env):
    pre = {"Origin": SITE, "Access-Control-Request-Method": "POST",
           "Access-Control-Request-Headers": "content-type,x-reviewer-client,x-reviewer-name"}
    st, h, _ = env.req("OPTIONS", "/api/lock?stem=a", headers=pre)
    assert st == 204 and h["Access-Control-Allow-Origin"] == SITE
    assert "X-Reviewer-Client" in h["Access-Control-Allow-Headers"]
    assert env.req("OPTIONS", "/api/lock?stem=a", headers={**pre, "Origin": "https://evil.example"})[0] == 403
    st, h, _ = env.req("GET", "/api/tracks", headers={"Origin": SITE})
    assert st == 200 and h["Access-Control-Allow-Origin"] == SITE and "Origin" in h["Vary"]
    st, h, _ = env.req("GET", "/api/tracks", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in h
    assert lock(env, "a", headers={"Origin": "https://evil.example"})[0] == 403
    assert lock(env, "a", headers={"Origin": SITE})[0] == 200
    st, h, b = env.req("GET", "/audio/a.mp3", headers={"Origin": SITE, "Range": "bytes=0-9"})
    assert st == 206 and h["Access-Control-Allow-Origin"] == SITE and len(b) == 10


def test_hosted_headers_and_logs(env):
    st, h, _ = env.req("GET", "/api/tracks")
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"] and h["X-Content-Type-Options"] == "nosniff"
    https = env.events("http")
    assert https[-1]["path"] == "/api/tracks" and https[-1]["status"] == 200
    assert len(https[-1]["ip"]) == 16 and "127.0.0.1" not in env.log.getvalue()
    assert env.req("GET", "/healthz", client=None)[2] == {"ok": True}


def test_client_address_from_cloudflare_only_when_trusted(edm, tmp_path, gh):
    fake, url = gh
    hashes = {}
    for trust in (False, True):
        s, e = make_env(edm, tmp_path / str(trust), url, fake, trust_cf=trust)
        try:
            for ip in ("198.51.100.7", "203.0.113.9"):
                e.req("GET", "/api/tracks", headers={"CF-Connecting-IP": ip})
            hashes[trust] = {x["ip"] for x in e.events("http")}
        finally:
            s.shutdown()
            s.server_close()
    assert len(hashes[False]) == 1  # the header is ignored: every request is the socket peer
    assert len(hashes[True]) == 2


def test_page_url_redirect(edm, tmp_path, gh):
    fake, url = gh
    log = io.StringIO()
    app = srv.HostedApp(srv.Corpus(edm), tmp_path / "st", None, log=hosted.JsonLog(log))
    s = srv.make_server(edm, "127.0.0.1", 0, (), app, (SITE,), page_url=SITE + "/labels/")
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", s.server_address[1], timeout=5)
        conn.request("GET", "/", headers={"Host": "127.0.0.1"})
        r = conn.getresponse()
        assert r.status == 302 and r.getheader("Location") == SITE + "/labels/"
    finally:
        s.shutdown()
        s.server_close()


# --- importing the owner's drafts, seeds, the checkout -------------------------------------------


def test_import_drafts_leaves_the_source_alone(edm, tmp_path):
    app = srv.HostedApp(srv.Corpus(edm), tmp_path / "st", None, log=hosted.JsonLog(io.StringIO()))
    body = {"audio_sha256": sha(edm / "a.mp3"), "duration": DUR, "onsets": [{"time": 0.5, "type": "kick"}],
            "reviewed_ranges": [[0, 2]], "started_from": {"source": "drums_adtof"}, "seed": "drums_adtof"}
    src = tmp_path / "box" / ".drafts" / "a.drums.json"
    src.parent.mkdir(parents=True)
    src.write_text(labelcheck.dumps_doc(srv.draft_doc(app.c, "a", body, existing=None)))
    before, mtime = src.read_bytes(), src.stat().st_mtime_ns
    assert srv.import_drafts(app, [src], "Owner Name") == ["a"]
    assert src.read_bytes() == before and src.stat().st_mtime_ns == mtime
    d = app.drafts.load("a", hosted.DraftStore.key_for("owner name", None))
    assert d["contributor"] == "Owner Name" and d["reviewed_ranges"] == [[0, 2]]
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps({**json.loads(before), "audio_sha256": "0" * 64}))
    with pytest.raises(SystemExit):
        srv.import_drafts(app, [bad], "Owner Name")


def test_a_better_seed_listed_first_wins(edm, tmp_path):
    _w(edm / "drums_sep" / "a.kick_weighted.json", _kw([1.25], [2.25], generator={"algorithm": "sep"}))
    seeds = tmp_path / "seeds.json"
    base = json.loads(labelcheck.DEFAULT_SEEDS_FILE.read_text())
    base["seeds"].insert(0, {"name": "drums_sep", "dir": "drums_sep", "suffix": ".kick_weighted.json",
                             "label": "separated drums"})
    seeds.write_text(json.dumps(base))
    c = srv.Corpus(edm, seeds_file=seeds)
    a = srv.load_track(c, "a")
    assert a["seed"] == "drums_sep" and a["onsets"] == [{"time": 1.25, "type": "kick"},
                                                        {"time": 2.25, "type": "snare"}]
    assert list(a["references"]) == ["drums_sep", "drums_adtof", "kick_weighted"]
    assert srv.load_track(c, "b")["seed"] == "drums_adtof"  # b has no separated-drums file
    t = srv.track_summary(c, "a")
    assert t["label_source"] == "drums_sep" and t["disagreement_between"] == ["drums_sep", "drums_adtof"]


def test_bad_seeds_file(tmp_path):
    for doc in ({"seeds": []}, {"seeds": [{"name": "Bad Name", "dir": "x", "suffix": ".json"}]},
                {"seeds": [{"name": "x", "dir": "../up", "suffix": ".json"}]},
                {"seeds": [{"name": "x", "dir": "drums_human", "suffix": ".json"}]}):
        p = tmp_path / "s.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(ValueError):
            labelcheck.load_seeds(p)


def test_checkout_clone_and_refresh(tmp_path):
    up = tmp_path / "upstream"
    up.mkdir()

    def git(*a, cwd=up):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a],
                       cwd=cwd, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (up / "f.txt").write_text("one")
    git("add", "f.txt")
    git("commit", "-qm", "one")
    co = hosted.Checkout(tmp_path / "co", up.as_uri(), log=hosted.JsonLog(io.StringIO()))
    co.ensure()
    assert (tmp_path / "co" / "f.txt").read_text() == "one"
    assert co.refresh() is False
    (up / "f.txt").write_text("two")
    git("commit", "-qam", "two")
    (tmp_path / "co" / "stray").write_text("x")  # nothing local survives a refresh
    assert co.refresh() is True
    assert (tmp_path / "co" / "f.txt").read_text() == "two" and not (tmp_path / "co" / "stray").exists()


@pytest.mark.parametrize("ranges", [
    [[0, DUR]],
    [[0, 5.00049], [5.0104, DUR]],  # a gap that rounding closes
    [[0.0004, 3.3333333], [3.3333333, 6.66666], [6.6721, DUR + 0.05]],
    [[0, 2], [4, 6]],
])
def test_what_the_server_writes_passes_corpus_ci(edm, ranges):
    c = srv.Corpus(edm)
    body = {"audio_sha256": sha(edm / "a.mp3"), "duration": DUR - 0.00004, "started_from": {"source": "drums_adtof"},
            "onsets": [{"time": t + 0.000049, "type": "kick"} for t in KICKS] + [{"time": 1.0, "type": "snare"}] * 2,
            "reviewed_ranges": ranges}
    doc = json.loads(labelcheck.dumps_doc(srv.validate_doc(c, "a", body, existing=None)))
    assert labelcheck.check_schema(doc, "a", sha(edm / "a.mp3"), DUR) == []


NAME_SAMPLES = ["Ana", "José M.", "Jose\u0301", "अमित", "李雷", "\U0001D504na", "٣٤", "Ana\u00a0B", "  x  y ",
                "O'Neil-Smith_2", "a@b", "<b>", "x`y", "zero\u200bwidth", "Ana\U0001F600", "a" * 40, "a" * 41,
                "e\u0301" * 21, "tab\there", ""]


def test_page_and_server_agree_on_names():
    """A name the page accepts but the server refuses would make every API call fail (400)."""
    import re
    import shutil

    node = shutil.which("node")
    assert node, "node is needed to run the page's name check (CI has it)"
    page = (TOOLS / "index.html").read_text()
    regex = re.search(r"const NAME_RE = (/.+?/u);", page).group(1)
    js = (f"const NAME_RE = {regex};\n"
          "const names = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
          "console.log(JSON.stringify(names.map(n => NAME_RE.test(n.normalize('NFC').trim().replace(/\\s+/g, ' ')))));")
    out = subprocess.run([node, "-e", js], input=json.dumps(NAME_SAMPLES), capture_output=True, text=True,
                         check=True).stdout
    page_ok = json.loads(out)

    def server_ok(n):
        try:
            hosted.normalize_name(n)
            return True
        except ValueError:
            return False

    assert {n: p for n, p in zip(NAME_SAMPLES, page_ok)} == {n: server_ok(n) for n in NAME_SAMPLES}
