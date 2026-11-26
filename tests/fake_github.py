#!/usr/bin/env python3
"""A fake of the few GitHub REST/GraphQL calls the hosted drum reviewer makes (tools/drum-reviewer/
hosted.py), for the unit tests (in-process) and CI's end-to-end run against the container:

    python tests/fake_github.py --port 9999 --repo OWNER/NAME --token TOKEN

Test hooks: GET /_fake/state (everything it was asked to do), POST /_fake/pr/<n> {fields} (change a
pull request: merged, state, mergeable_state), POST /_fake/main {"path": ..., "text": ...} (change a
file on main), POST /_fake/fail {"call": "put_file", "status": 500} (make a call fail once).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeGitHub:
    def __init__(self, repo: str = "owner/corpus", token: str = "test-token",
                 main_files: dict[str, bytes] | None = None):
        self.repo, self.token = repo, token
        self.lock = threading.Lock()
        head = secrets.token_hex(20)
        self.branches: dict[str, dict[str, Any]] = {"main": {"sha": head, "files": dict(main_files or {})}}
        self.prs: dict[int, dict[str, Any]] = {}
        self.commits: list[dict[str, Any]] = []
        self.automerge: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.deleted: list[str] = []
        self.updated_branches: list[int] = []
        self.fail: dict[str, int] = {}
        self.server: ThreadingHTTPServer | None = None

    # -- state for tests
    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "branches": {k: {"sha": v["sha"], "files": sorted(v["files"])} for k, v in self.branches.items()},
                "prs": self.prs, "commits": self.commits, "automerge": self.automerge,
                "calls": self.calls, "deleted": self.deleted, "updated_branches": self.updated_branches,
            }

    def set_main_file(self, path: str, data: bytes) -> None:
        with self.lock:
            self.branches["main"]["files"][path] = data
            self.branches["main"]["sha"] = secrets.token_hex(20)

    def start(self, port: int = 0) -> str:
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, status: int, body: Any = None) -> None:
                raw = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _body(self) -> Any:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n)) if n else None

            def _go(self, method: str) -> None:
                url = urlparse(self.path)
                if url.path.startswith("/_fake/"):
                    return self._control(method, url.path, self._body())
                if self.headers.get("Authorization") != f"Bearer {fake.token}":
                    return self._send(401, {"message": "Bad credentials"})
                body = self._body()
                with fake.lock:
                    status, out = fake.handle(method, url.path, parse_qs(url.query), body)
                self._send(status, out)

            def _control(self, method: str, path: str, body: Any) -> None:
                if method == "GET" and path == "/_fake/state":
                    return self._send(200, fake.state())
                m = re.fullmatch(r"/_fake/pr/(\d+)", path)
                if method == "POST" and m:
                    with fake.lock:
                        fake.prs[int(m.group(1))].update(body or {})
                    return self._send(200, {"ok": True})
                if method == "POST" and path == "/_fake/main":
                    fake.set_main_file(body["path"], body["text"].encode())
                    return self._send(200, {"ok": True})
                if method == "POST" and path == "/_fake/fail":
                    with fake.lock:
                        fake.fail[body["call"]] = int(body.get("status", 500))
                    return self._send(200, {"ok": True})
                self._send(404, {"message": "Not Found"})

            def do_GET(self) -> None:
                self._go("GET")

            def do_POST(self) -> None:
                self._go("POST")

            def do_PUT(self) -> None:
                self._go("PUT")

            def do_DELETE(self) -> None:
                self._go("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def _failing(self, call: str) -> tuple[int, Any] | None:
        st = self.fail.pop(call, None)
        return (st, {"message": f"injected failure in {call}"}) if st else None

    # -- the API
    def handle(self, method: str, path: str, q: dict[str, list[str]], body: Any) -> tuple[int, Any]:
        self.calls.append(f"{method} {path}")
        if method == "POST" and path == "/graphql":
            f = self._failing("automerge")
            if f:
                return 200, {"errors": [{"message": f[1]["message"]}]}
            v = body["variables"]
            n = next((n for n, p in self.prs.items() if p["node_id"] == v["id"]), None)
            if n is None or "enablePullRequestAutoMerge" not in body["query"]:
                return 200, {"errors": [{"message": "Could not resolve to a node"}]}
            method_m = re.search(r"mergeMethod:\s*(\w+)", body["query"])
            self.automerge.append({"pr": n, "merge_method": method_m.group(1) if method_m else None,
                                   "headline": v["h"], "body": v["b"]})
            return 200, {"data": {"enablePullRequestAutoMerge": {"pullRequest": {"number": n}}}}
        pre = f"/repos/{self.repo}"
        if not path.startswith(pre):
            return 404, {"message": "Not Found"}
        sub = path[len(pre):]
        m = re.fullmatch(r"/git/ref/heads/(.+)", sub)
        if method == "GET" and m:
            b = self.branches.get(m.group(1))
            return (200, {"object": {"sha": b["sha"]}}) if b else (404, {"message": "Not Found"})
        m = re.fullmatch(r"/contents/(.+)", sub)
        if method == "GET" and m:
            p, ref = unquote(m.group(1)), (q.get("ref") or ["main"])[0]
            files = next((b["files"] for b in self.branches.values() if b["sha"] == ref),
                         self.branches.get(ref, {}).get("files"))
            if files is None or p not in files:
                return 404, {"message": "Not Found"}
            return 200, {"sha": blob_sha(files[p]), "path": p}
        if method == "PUT" and m:
            f = self._failing("put_file")
            if f:
                return f
            p = unquote(m.group(1))
            b = self.branches.get(body["branch"])
            if b is None:
                return 404, {"message": "Branch not found"}
            cur = b["files"].get(p)
            if (cur is None) != (body.get("sha") is None) or (cur is not None and blob_sha(cur) != body["sha"]):
                return 409, {"message": "sha does not match"}
            data = base64.b64decode(body["content"])
            b["files"][p] = data
            b["sha"] = secrets.token_hex(20)
            self.commits.append({"branch": body["branch"], "path": p, "message": body["message"],
                                 "text": data.decode()})
            return 201, {"commit": {"sha": b["sha"]}, "content": {"sha": blob_sha(data)}}
        if method == "POST" and sub == "/git/refs":
            name = body["ref"].removeprefix("refs/heads/")
            if name in self.branches:
                return 422, {"message": "Reference already exists"}
            src = next(b for b in self.branches.values() if b["sha"] == body["sha"])
            self.branches[name] = {"sha": body["sha"], "files": dict(src["files"])}
            return 201, {"ref": body["ref"]}
        m = re.fullmatch(r"/git/refs/heads/(.+)", sub)
        if method == "DELETE" and m:
            if self.branches.pop(m.group(1), None) is None:
                return 422, {"message": "Reference does not exist"}
            self.deleted.append(m.group(1))
            return 204, None
        if method == "POST" and sub == "/pulls":
            f = self._failing("create_pr")
            if f:
                return f
            n = len(self.prs) + 1
            self.prs[n] = {"number": n, "node_id": f"PR_{n}", "html_url": f"https://github.com/{self.repo}/pull/{n}",
                           "head": {"ref": body["head"], "sha": self.branches[body["head"]]["sha"]},
                           "base": body["base"], "title": body["title"], "body": body["body"],
                           "state": "open", "merged": False, "mergeable_state": "blocked"}
            return 201, self.prs[n]
        m = re.fullmatch(r"/pulls/(\d+)", sub)
        if method == "GET" and m:
            pr = self.prs.get(int(m.group(1)))
            return (200, pr) if pr else (404, {"message": "Not Found"})
        m = re.fullmatch(r"/pulls/(\d+)/update-branch", sub)
        if method == "PUT" and m:
            n = int(m.group(1))
            self.updated_branches.append(n)
            self.prs[n]["mergeable_state"] = "blocked"
            return 202, {"message": "Updating pull request branch."}
        return 404, {"message": f"fake: no route for {method} {sub}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--repo", default="owner/corpus")
    ap.add_argument("--token", default="test-token")
    args = ap.parse_args()
    fake = FakeGitHub(args.repo, args.token)
    print(f"fake GitHub for {args.repo} on {fake.start(args.port)}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
