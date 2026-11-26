#!/usr/bin/env python3
"""End-to-end check of a running hosted reviewer (CI runs it against the container image, with
tests/fake_github.py standing in for GitHub):

    python tests/e2e_hosted.py --api http://127.0.0.1:8766 --fake http://127.0.0.1:9999 \\
        --origin https://site.example.invalid --host api.example.invalid

Walks the multi-user flow over HTTP the way two browsers would: CORS, locks, drafts that follow a
name, an incomplete save kept as a draft, a complete save that becomes a pull request with
auto-merge, a pending track refusing a second save, the merge being followed, and rate limiting.
Exits non-zero on the first failure. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


class E2E:
    def __init__(self, api: str, fake: str, origin: str, host: str | None):
        self.api, self.fake, self.origin, self.host = api.rstrip("/"), fake.rstrip("/"), origin, host

    def call(self, method: str, path: str, body: Any = None, client: str | None = "e2e-client-a",
             name: str = "E2E Ana", origin: str | None = "default",
             extra: dict[str, str] | None = None) -> tuple[int, dict[str, str], Any]:
        h = {"X-Reviewer-Name": name}
        if client:
            h["X-Reviewer-Client"] = client
        if self.host:
            h["Host"] = self.host
        if origin:
            h["Origin"] = self.origin if origin == "default" else origin
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        h.update(extra or {})
        req = urllib.request.Request(self.api + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw, st, hd = r.read(), r.status, dict(r.headers)
        except urllib.error.HTTPError as e:
            raw, st, hd = e.read(), e.code, dict(e.headers)
        ctype = hd.get("Content-Type", "")
        return st, hd, (json.loads(raw) if raw and ctype.startswith("application/json") else raw)

    def fake_state(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.fake + "/_fake/state", timeout=10) as r:
            return json.loads(r.read())

    def fake_post(self, path: str, body: Any) -> None:
        req = urllib.request.Request(self.fake + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--origin", required=True, help="the --cors-origin the server was started with")
    ap.add_argument("--host", help="Host header (an --allow-host name)")
    ap.add_argument("--wait", type=float, default=120, help="seconds to wait for the server")
    args = ap.parse_args()
    e = E2E(args.api, args.fake, args.origin, args.host)

    deadline = time.monotonic() + args.wait
    while True:
        try:
            if e.call("GET", "/healthz", client=None, origin=None)[0] == 200:
                break
        except OSError:
            pass
        if time.monotonic() > deadline:
            check(False, f"server answered /healthz within {args.wait:.0f} s")
        time.sleep(1)
    check(True, "server is up")

    st, _, cfg = e.call("GET", "/api/config")
    check(st == 200 and cfg["hosted"] and cfg["saving"], "hosted, saving on")
    st, h, _ = e.call("OPTIONS", "/api/save?stem=x", extra={
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-reviewer-client,x-reviewer-name"})
    check(st == 204 and h.get("Access-Control-Allow-Origin") == args.origin, "CORS preflight for the site")
    st, _, _ = e.call("OPTIONS", "/api/save?stem=x", origin="https://evil.example",
                      extra={"Access-Control-Request-Method": "POST"})
    check(st == 403, "CORS preflight refused for another origin")
    check("Content-Security-Policy" in e.call("GET", "/api/config")[1], "security headers")

    st, h, tracks = e.call("GET", "/api/tracks")
    check(st == 200 and h.get("Access-Control-Allow-Origin") == args.origin, "track list over CORS")
    free = [t for t in tracks["tracks"] if not t["has_human"] and not t["pending"] and t["lock"] is None]
    check(len(free) >= 3, f"{len(free)} tracks without human labels to work on")
    stem, other, third = free[0]["stem"], free[1]["stem"], free[2]["stem"]

    st, h, audio = e.call("GET", f"/audio/{stem}.mp3", extra={"Range": "bytes=0-1023"})
    check(st == 206 and len(audio) == 1024 and h.get("Access-Control-Allow-Origin") == args.origin,
          "audio range request over CORS")

    lock = lambda s, **kw: e.call("POST", f"/api/lock?stem={s}", {"action": "acquire"}, **kw)  # noqa: E731
    check(lock(stem)[0] == 200, "A takes the edit lock")
    st, _, b = lock(stem, client="e2e-client-b", name="E2E Bo")
    check(st == 423 and b["lock"]["name"] == "E2E Ana", "B is told A is editing")
    st, _, view = e.call("GET", f"/api/track?stem={stem}")
    body = {"audio_sha256": view["audio_sha256"], "duration": view["manifest_duration"],
            "started_from": view["started_from"], "seed": view["seed"], "onsets": view["onsets"],
            "reviewed_ranges": [[0, 10.0]], "base_version": view["version"]}
    check(e.call("POST", f"/api/draft?stem={stem}", body, client="e2e-client-b", name="E2E Bo")[0] == 423,
          "B can't write a draft")
    check(e.call("POST", f"/api/draft?stem={stem}", body)[0] == 200, "A's draft autosaves")
    st, _, v2 = e.call("GET", f"/api/track?stem={stem}", client="e2e-client-elsewhere", name="e2e ana")
    check(v2["draft"] and v2["draft"]["reviewed_ranges"] == [[0, 10.0]], "A's draft follows the name to another machine")
    st, _, b = e.call("POST", f"/api/save?stem={stem}", body)
    check(st == 422 and "100%" in b["error"], "an incomplete save is refused and kept as a draft")

    body["reviewed_ranges"] = [[0, view["manifest_duration"]]]
    st, _, b = e.call("POST", f"/api/save?stem={stem}", body)
    check(st == 200 and b.get("pr") == 1, f"a complete save opens a pull request ({b})")
    fs = e.fake_state()
    (commit,) = fs["commits"]
    check(commit["path"] == f"edm/drums_human/{stem}.drums.json" and commit["branch"].startswith(f"labels/{stem}-"),
          "one commit, on a labels/ branch, of the one label file")
    check("Contributed-by: E2E Ana" in commit["message"] and "Client-IP-Hash: " in commit["message"],
          "commit trailers name the contributor and a hashed client address")
    check("127.0.0.1" not in commit["message"] and "127.0.0.1" not in json.dumps(fs["prs"]),
          "no client address in the commit or the PR")
    check(fs["automerge"] and fs["automerge"][0]["merge_method"] == "SQUASH", "auto-merge (squash) is on")

    st, _, t = e.call("GET", f"/api/track?stem={stem}")
    check(t["pending"]["pr"] == 1 and t["seed"] == "human" and t["lock"] is None,
          "the track shows the pending save, its lock released")
    lock(stem)
    st, _, b = e.call("POST", f"/api/save?stem={stem}", dict(body, base_version=t["version"]))
    check(st == 409 and "PR #1" in b["error"], "a pending track takes no second save")
    e.call("POST", f"/api/lock?stem={stem}", {"action": "release"})

    e.fake_post("/_fake/pr/1", {"merged": True, "state": "closed"})
    deadline = time.monotonic() + 60
    while e.call("GET", f"/api/track?stem={stem}")[2]["pending"]["state"] != "merged":
        if time.monotonic() > deadline:
            check(False, "the merge was noticed within 60 s")
        time.sleep(1)
    check(commit["branch"] in e.fake_state()["deleted"], "the merge is followed and the branch deleted")

    # Lock grabs are rate limited per client address (30 per hour by default): different tabs,
    # same address, alternating between two tracks.
    codes = [lock(other if i % 2 else third, client=f"e2e-flood-{i:04d}", name="")[0] for i in range(40)]
    check(429 in codes, f"lock grabs are rate limited ({codes.count(429)} of 40 refused)")
    print("e2e: all checks passed")


if __name__ == "__main__":
    main()
