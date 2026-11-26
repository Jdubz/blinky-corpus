#!/usr/bin/env python3
"""Publish the labels service's shared labels into the corpus: the trusted half of the pull model.

The labels service (blinky_time labels-service/) holds no credential. It offers its edited tracks at
GET <api>/api/export. This script runs from .github/workflows/labels-sync.yml on a schedule, which
is the corpus's own code on main, with that workflow's GITHUB_TOKEN:

1. Fetch the export. Nothing ready (people are still editing) and no batch open: done.
2. Treat it as untrusted input. Every entry must be:
   - a path edm/drums_human/<stem>.drums.json, where <stem> is one of the corpus's edm/ tracks (no
     separators, no '..', nothing else);
   - within the size caps (per file and per batch);
   - JSON that parses and passes main's label checks (labelcheck.py: schema, no mass deletion, no
     empty class).
   A file that fails is left out and reported.
3. Write ONLY those files into a checkout of main, commit them as one batch, and check the commit
   against main again (labelcheck --only-labels: the diff touches nothing outside edm/drums_human/).
4. Push it to the open batch's labels/auto-* branch (rebuilt on today's main), or to a new one.
   Start tests.yml on the branch (its "Scoring tests" job is main's required check), open the PR if
   there isn't one, and turn on auto-merge (squash).
5. On the next run, approve the batch PR's own held pull_request runs (approve_held_runs): GitHub holds
   them for the Actions bot, and a held run blocks the merge.

    python tools/labels/sync.py --api https://blinky-labels-api.joshwentworth.com [--dry-run]

Python standard library, plus git and the gh CLI (both on GitHub's runners).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import secrets
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelcheck  # noqa: E402

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_BATCH_BYTES = 20 * 1024 * 1024
MAX_FILES = 100
PATH_RE = re.compile(r"^edm/drums_human/([A-Za-z0-9][A-Za-z0-9._-]{0,200})\.drums\.json$")
BRANCH_PREFIX = "labels/auto-"
BOT = {"GIT_AUTHOR_NAME": "blinky labels sync", "GIT_AUTHOR_EMAIL": "labels-sync@blinky-time.joshwentworth.com",
       "GIT_COMMITTER_NAME": "blinky labels sync", "GIT_COMMITTER_EMAIL": "labels-sync@blinky-time.joshwentworth.com"}

Run = Callable[..., str]


def run(*cmd: str, cwd: Path, env: dict[str, str] | None = None, check: bool = True) -> str:
    import os
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])}: {(r.stderr or r.stdout).strip()[-500:]}")
    return r.stdout


def fetch_export(api: str, timeout: float = 30) -> Any:
    req = urllib.request.Request(api.rstrip("/") + "/api/export",
                                 headers={"User-Agent": "blinky-corpus-labels-sync", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(MAX_BATCH_BYTES + 1024 * 1024 + 1)
    if len(raw) > MAX_BATCH_BYTES + 1024 * 1024:
        raise ValueError("the export is over the batch size cap")
    return json.loads(raw)


def validate(export: Any, repo: Path, seeds_file: Path | None = None) -> tuple[bool, dict[str, str], list[str]]:
    """(ready, {path: text} that pass, problems). Nothing from the export is trusted."""
    if not isinstance(export, dict) or not isinstance(export.get("tracks"), dict):
        return False, {}, ["the export is not {\"ready\", \"tracks\": {...}}"]
    ready = export.get("ready") is True
    edm = repo / "edm"
    known = {p.stem for p in edm.glob("*.mp3")}
    seeds = labelcheck.load_seeds(seeds_file or repo / "tools" / "labels" / "seeds.json")
    good: dict[str, str] = {}
    problems: list[str] = []
    total = 0
    items = list(export["tracks"].items())
    if len(items) > MAX_FILES:
        return False, {}, [f"the export lists {len(items)} files (the cap is {MAX_FILES})"]
    for path, text in items:
        m = PATH_RE.match(path) if isinstance(path, str) else None
        if not m or ".." in path or m.group(1) not in known:
            problems.append(f"{str(path)[:120]!r}: not edm/drums_human/<one of the corpus's tracks>.drums.json")
            continue
        if not isinstance(text, str) or len(text.encode()) > MAX_FILE_BYTES:
            problems.append(f"{path}: not a string, or over {MAX_FILE_BYTES} bytes")
            continue
        total += len(text.encode())
        if total > MAX_BATCH_BYTES:
            problems.append(f"{path}: the batch is over {MAX_BATCH_BYTES} bytes")
            continue
        try:
            doc = json.loads(text)
        except ValueError as e:
            problems.append(f"{path}: not JSON: {e}")
            continue
        main_file = repo / path
        base = json.loads(main_file.read_text()) if main_file.is_file() else None
        p = labelcheck.check_human_doc(edm, m.group(1), doc, base, seeds)
        if p:
            problems += [f"{path}: {x}" for x in p]
            continue
        good[path] = text
    return ready, good, problems


BATCH_AUTHOR = "app/github-actions"  # this workflow's GITHUB_TOKEN opens the batches


def open_batch(gh: Run, repo: Path) -> tuple[str, int] | None:
    """The open batch (branch, number), if any: a PR this workflow opened, from a labels/auto-* branch
    of this repo. A fork's PR with the same branch name (anyone can open one) never counts."""
    prs = json.loads(gh("pr", "list", "--state", "open", "--base", "main", "--json",
                        "headRefName,number,isCrossRepository,author", "--limit", "50", cwd=repo) or "[]")
    mine = [(p["headRefName"], p["number"]) for p in prs
            if p["headRefName"].startswith(BRANCH_PREFIX) and p.get("isCrossRepository") is False
            and (p.get("author") or {}).get("login") == BATCH_AUTHOR]
    return sorted(mine, key=lambda x: x[1])[0] if mine else None


BOT_ACTOR = "github-actions[bot]"  # who the batch PR's pull_request runs are started by


def approve_held_runs(gh: Run, git: Run, repo: Path, branch: str) -> list[str]:
    """Approve the held pull_request runs of this workflow's own open batch. Returns what it approved.

    GitHub holds the pull_request runs of a PR opened by GITHUB_TOKEN as "action_required" (the repo's
    approval policy for first-time contributors counts the Actions bot as one), and a held run keeps the
    PR BLOCKED even though tests.yml's dispatched run already passed Scoring tests on the same commit.

    Only runs that are exactly ours are approved: event pull_request, started by the Actions bot, from
    this repository (never a fork), on the open batch's branch, at the commit that branch points to now.
    That commit is main plus label files this job wrote and checked, so the workflows the run executes
    ARE main's (see labels-sync.yml). Anything else stays held for a human."""
    git("fetch", "--quiet", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}", cwd=repo)
    tip = git("rev-parse", f"origin/{branch}", cwd=repo).strip()
    raw = gh("api", "-X", "GET", "repos/{owner}/{repo}/actions/runs", "-f", f"branch={branch}",
             "-f", "event=pull_request", "-f", "status=action_required", "-f", "per_page=50", cwd=repo)
    approved = []
    for r in (json.loads(raw or "{}").get("workflow_runs") or []):
        ours = (r.get("event") == "pull_request" and r.get("status") == "completed"
                and r.get("conclusion") == "action_required"
                and r.get("head_branch") == branch and r.get("head_sha") == tip
                and (r.get("actor") or {}).get("login") == BOT_ACTOR
                and (r.get("triggering_actor") or {}).get("login") == BOT_ACTOR
                and (r.get("head_repository") or {}).get("full_name") == (r.get("repository") or {}).get("full_name"))
        if not ours:
            print(f"::warning::left held: run {r.get('id')} ({r.get('name')}) is not this batch's own run")
            continue
        gh("api", "-X", "POST", f"repos/{{owner}}/{{repo}}/actions/runs/{int(r['id'])}/approve", cwd=repo)
        approved.append(f"{r.get('name')} ({r['id']})")
    return approved


def publish(repo: Path, files: dict[str, str], gh: Run, git: Run, dry_run: bool = False) -> str:
    """Commit `files` on top of origin/main and publish them as the batch. Returns what it did."""
    if not files:
        return "nothing differs from main"
    git("checkout", "--quiet", "--detach", "origin/main", cwd=repo)
    for path, text in files.items():
        dest = repo / path
        if dest.parent != repo / "edm" / "drums_human":
            raise RuntimeError(f"refusing to write {path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    git("add", "--", "edm/drums_human", cwd=repo)
    changed = git("diff", "--cached", "--name-only", cwd=repo).split()
    if not changed:
        return "nothing differs from main"
    batch = open_batch(gh, repo)
    names = [Path(p).name.split(".")[0] for p in changed]
    subject = f"labels: drum reviewer autosave ({len(changed)} track{'s' if len(changed) != 1 else ''})"
    body = ("\n".join(f"- {n}" for n in names) + "\n\nThe shared labels of the tracks above, as the labels "
            "service (blinky_time labels-service/) offers them, checked by main's tools/labels/labelcheck.py "
            "and committed by .github/workflows/labels-sync.yml. Auto-merge merges them once Scoring tests "
            "passes.\n")
    git("commit", "--quiet", "-m", subject, "-m", body, cwd=repo, env=BOT)
    # main's own check, once more, on exactly what will be pushed
    problems = labelcheck.check_pr(repo, "origin/main", None, override=False, only_labels=True)
    if problems:
        raise RuntimeError("the batch fails main's label checks: " + "; ".join(problems))
    if batch:
        branch, number = batch
        git("fetch", "--quiet", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}", cwd=repo)
        same_tree = git("rev-parse", "HEAD^{tree}", cwd=repo) == git("rev-parse", f"origin/{branch}^{{tree}}", cwd=repo)
        on_main = git("rev-parse", f"origin/{branch}^", cwd=repo) == git("rev-parse", "origin/main", cwd=repo)
        if same_tree and on_main:
            if not dry_run:
                gh("pr", "merge", str(number), "--auto", "--squash", cwd=repo)
            return f"#{number} already has exactly this"
    else:
        branch = f"{BRANCH_PREFIX}{_dt.datetime.now(_dt.timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        number = None
    if dry_run:
        return f"would push {len(changed)} file(s) to {branch}"
    git("push", "--quiet", "--force", "origin", f"HEAD:refs/heads/{branch}", cwd=repo)
    # A push made with GITHUB_TOKEN starts no workflow, so start tests.yml on the branch: its Scoring
    # tests job is main's required check (see labels-sync.yml for why that's safe).
    gh("workflow", "run", "tests.yml", "--ref", branch, cwd=repo)
    if number is None:
        url = gh("pr", "create", "--base", "main", "--head", branch, "--title", subject, "--body", body, cwd=repo).strip()
        number = int(url.rstrip("/").split("/")[-1])
    gh("pr", "merge", str(number), "--auto", "--squash", cwd=repo)
    return f"pushed {len(changed)} file(s) to {branch} (#{number})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", required=True, help="the labels service's public URL")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    repo = args.repo.resolve()

    def gh(*a: str, cwd: Path) -> str:
        return run("gh", *a, cwd=cwd)

    def git(*a: str, cwd: Path, env: dict[str, str] | None = None) -> str:
        return run("git", *a, cwd=cwd, env=env)

    git("fetch", "--quiet", "origin", "+refs/heads/main:refs/remotes/origin/main", cwd=repo)
    git("checkout", "--quiet", "--detach", "origin/main", cwd=repo)
    # The open batch's pull_request runs appear a few seconds after the run that pushed it, so each run
    # approves the ones the previous run's push started.
    batch = open_batch(gh, repo)
    if batch and not args.dry_run:
        for a in approve_held_runs(gh, git, repo, batch[0]):
            print(f"approved held run {a} of #{batch[1]}")
    export = fetch_export(args.api)
    ready, files, problems = validate(export, repo)
    for p in problems:
        print(f"::warning::left out: {p}")
    if not files:
        print("nothing to publish")
        return 0
    if not ready and open_batch(gh, repo) is None:
        print(f"{len(files)} track(s) changed, but people are still editing: next run")
        return 0
    print(publish(repo, files, gh, git, args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
