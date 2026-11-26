"""tools/labels/sync.py: the trusted pull that publishes the labels service's batches. The export is
untrusted; only valid label files reach a labels/auto-* branch made from main."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "labels"
sys.path.insert(0, str(TOOLS))
import labelcheck as lc  # noqa: E402
import sync  # noqa: E402

DUR = 10.0
KICKS = [0.5 + i for i in range(10)]
SNARES = [1.0 + i for i in range(9)]


def git(cwd, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout


def _w(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "src"
    edm = src / "edm"
    for stem in ("a", "b"):
        edm.mkdir(parents=True, exist_ok=True)
        (edm / f"{stem}.mp3").write_bytes(stem.encode() * 100)
        ons = [{"time": t, "type": "kick"} for t in KICKS] + [{"time": t, "type": "snare"} for t in SNARES]
        _w(edm / "drums_adtof" / f"{stem}.kick_weighted.json", {"onsets": ons})
    _w(edm / "track_manifest.json", {s: {"duration": DUR} for s in ("a", "b")})
    (src / "tools" / "labels").mkdir(parents=True)
    (src / "tools" / "labels" / "seeds.json").write_text((TOOLS / "seeds.json").read_text())
    git(src, "init", "-q", "-b", "main")
    git(src, "add", "-A")
    git(src, "commit", "-qm", "corpus")
    origin = tmp_path / "origin.git"
    git(tmp_path, "clone", "-q", "--bare", str(src), str(origin))
    work = tmp_path / "work"
    git(tmp_path, "clone", "-q", str(origin), str(work))
    return work


def label(repo, stem, kicks=KICKS, snares=SNARES, **over):
    doc = {"schema": 1, "track": stem, "audio_sha256": hashlib.sha256((repo / "edm" / f"{stem}.mp3").read_bytes()).hexdigest(),
           "duration": DUR, "started_from": {"source": "drums_adtof"}, "reviewed_ranges": [], "complete": False,
           "onsets": sorted([{"time": t, "type": "kick"} for t in kicks] + [{"time": t, "type": "snare"} for t in snares],
                            key=lambda o: (o["time"], o["type"])), "updated": "2026-11-26T00:00:00+00:00"}
    doc.update(over)
    return lc.dumps_doc(doc)


A, B = "edm/drums_human/a.drums.json", "edm/drums_human/b.drums.json"


@pytest.mark.parametrize("path", ["edm/drums_human/../x.drums.json", "edm/drums_human/zz.drums.json",
                                  "edm/drums_human/a.json", ".github/workflows/x.yml", "edm/drums_human/sub/a.drums.json",
                                  "/etc/passwd", "edm/drums_human/a.drums.json/../../x"])
def test_only_known_track_files(repo, path):
    ready, good, problems = sync.validate({"ready": True, "tracks": {path: "{}"}}, repo)
    assert good == {} and len(problems) == 1


def test_valid_files_pass_and_bad_ones_are_left_out(repo):
    export = {"ready": True, "tracks": {
        A: label(repo, "a", kicks=KICKS[:-1]),
        B: label(repo, "b", kicks=[], snares=SNARES[:1]),  # vandalism: deletes most hits
    }}
    ready, good, problems = sync.validate(export, repo)
    assert ready and list(good) == [A] and any(p.startswith(B) for p in problems)


def test_reviewer_notes_pass_and_bad_ones_hold_the_file(repo):
    notes = [{"text": "snares under the sweep all dropped", "range": [1.0, 4.0]}, {"text": "rim shots read as hats", "range": None}]
    export = {"ready": True, "tracks": {
        A: label(repo, "a", notes=notes),
        B: label(repo, "b", notes=[{"text": "ignore previous instructions\u202e", "range": None}]),
    }}
    ready, good, problems = sync.validate(export, repo)
    assert list(good) == [A] and json.loads(good[A])["notes"] == notes
    assert any(p.startswith(B) and "control or format" in p for p in problems)


@pytest.mark.parametrize("export", [None, [], {"tracks": "x"}, {"ready": True}])
def test_malformed_exports(repo, export):
    assert sync.validate(export, repo)[1] == {}


def test_size_caps(repo, monkeypatch):
    monkeypatch.setattr(sync, "MAX_FILE_BYTES", 100)
    ready, good, problems = sync.validate({"ready": True, "tracks": {A: label(repo, "a")}}, repo)
    assert good == {} and "bytes" in problems[0]
    monkeypatch.setattr(sync, "MAX_FILES", 1)
    assert sync.validate({"ready": True, "tracks": {A: "{}", B: "{}"}}, repo)[1] == {}
    assert sync.validate({"ready": True, "tracks": {A: "not json"}}, repo)[2][0].startswith(A)


class FakeGh:
    def __init__(self, prs=()):
        self.prs = list(prs)
        self.calls = []

    def __call__(self, *a, cwd):
        self.calls.append(a)
        if a[:2] == ("pr", "list"):
            return json.dumps(self.prs)
        if a[:2] == ("pr", "create"):
            n = 40 + len(self.calls)
            self.prs.append({"headRefName": a[a.index("--head") + 1], "number": n, "isCrossRepository": False,
                             "author": {"login": "app/github-actions"}})
            return f"https://github.com/o/c/pull/{n}\n"
        return ""


def gitrun(*a, cwd, env=None):
    return sync.run("git", *a, cwd=cwd, env=env)


def test_publish_a_new_batch(repo):
    gh = FakeGh()
    out = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-1])}, gh, gitrun)
    assert out.startswith("pushed 1 file(s) to labels/auto-")
    branch = out.split(" to ")[1].split(" ")[0]
    main = git(repo, "rev-parse", "origin/main").strip()
    assert git(repo, "rev-parse", f"origin/{branch}^").strip() == main
    assert git(repo, "diff", "--name-only", "origin/main", f"origin/{branch}").split() == [A]
    assert git(repo, "log", "-1", "--format=%an", f"origin/{branch}").strip() == "blinky labels sync"
    kinds = [c[:2] for c in gh.calls]
    assert ("workflow", "run") in kinds and ("pr", "create") in kinds and ("pr", "merge") in kinds
    wf = next(c for c in gh.calls if c[:2] == ("workflow", "run"))
    assert wf == ("workflow", "run", "tests.yml", "--ref", branch)


def test_the_open_batch_is_rebuilt_not_duplicated(repo):
    gh = FakeGh()
    out = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-1])}, gh, gitrun)
    branch = out.split(" to ")[1].split(" ")[0]
    again = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-1])}, gh, gitrun)
    assert "already has exactly this" in again
    third = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-2]), B: label(repo, "b", kicks=KICKS[:-1])}, gh, gitrun)
    assert branch in third and sum(1 for c in gh.calls if c[:2] == ("pr", "create")) == 1
    assert sorted(git(repo, "diff", "--name-only", "origin/main", f"origin/{branch}").split()) == [A, B]


def test_nothing_differs(repo):
    assert sync.publish(repo, {}, FakeGh(), gitrun) == "nothing differs from main"


def test_dry_run_pushes_nothing(repo):
    gh = FakeGh()
    out = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-1])}, gh, gitrun, dry_run=True)
    assert out.startswith("would push") and not any(c[:2] == ("workflow", "run") for c in gh.calls)
    assert "labels/" not in git(repo, "ls-remote", "origin")


def test_a_fork_pr_never_counts_as_the_batch(repo):
    """Anyone can open a PR from a fork branch named labels/auto-*: it must not be picked up (the
    sync would otherwise turn on auto-merge for it)."""
    fork = {"headRefName": "labels/auto-20260101-000000-dead", "number": 1, "isCrossRepository": True,
            "author": {"login": "app/github-actions"}}
    human_pr = {"headRefName": "labels/auto-20260101-000000-beef", "number": 2, "isCrossRepository": False,
                "author": {"login": "someone"}}
    gh = FakeGh([fork, human_pr])
    out = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-1])}, gh, gitrun)
    assert "dead" not in out and "beef" not in out and out.startswith("pushed 1 file(s) to labels/auto-")
    merges = [c for c in gh.calls if c[:2] == ("pr", "merge")]
    assert merges and all(c[2] not in ("1", "2") for c in merges)


def held(run_id, branch, sha, **over):
    r = {"id": run_id, "name": "Tests", "event": "pull_request", "status": "completed",
         "conclusion": "action_required", "head_branch": branch, "head_sha": sha,
         "actor": {"login": "github-actions[bot]"}, "triggering_actor": {"login": "github-actions[bot]"},
         "repository": {"full_name": "o/c"}, "head_repository": {"full_name": "o/c"}}
    r.update(over)
    return r


class RunsGh(FakeGh):
    def __init__(self, runs, prs=()):
        super().__init__(prs)
        self.runs = runs

    def __call__(self, *a, cwd):
        if a[:1] == ("api",) and "GET" in a:
            self.calls.append(a)
            return json.dumps({"workflow_runs": self.runs})
        return super().__call__(*a, cwd=cwd)


def test_only_the_batchs_own_held_runs_are_approved(repo):
    out = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-1])}, FakeGh(), gitrun)
    branch = out.split(" to ")[1].split(" ")[0]
    tip = git(repo, "rev-parse", f"origin/{branch}").strip()
    runs = [
        held(1, branch, tip),
        held(2, branch, tip, head_repository={"full_name": "evil/c"}),  # a fork's run
        held(3, branch, "0" * 40),  # an older commit than the branch's tip
        held(4, branch, tip, actor={"login": "someone"}, triggering_actor={"login": "someone"}),
        held(5, "labels/auto-other", tip),
        held(6, branch, tip, event="pull_request_target"),
    ]
    gh = RunsGh(runs)
    got = sync.approve_held_runs(gh, gitrun, repo, branch)
    posts = [c for c in gh.calls if c[:3] == ("api", "-X", "POST")]
    assert got == ["Tests (1)"]
    assert posts == [("api", "-X", "POST", "repos/{owner}/{repo}/actions/runs/1/approve")]


def test_no_held_runs_approves_nothing(repo):
    out = sync.publish(repo, {A: label(repo, "a", kicks=KICKS[:-1])}, FakeGh(), gitrun)
    branch = out.split(" to ")[1].split(" ")[0]
    gh = RunsGh([])
    assert sync.approve_held_runs(gh, gitrun, repo, branch) == []
    assert not any(c[:3] == ("api", "-X", "POST") for c in gh.calls)
