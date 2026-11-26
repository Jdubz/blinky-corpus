#!/usr/bin/env python3
"""Browser check of the hosted reviewer page, in the production shape: the page served from one
origin (the product site), the API on another (CORS). CI runs it with Playwright's Chromium:

    python tests/browser_hosted.py --page http://127.0.0.1:8000/ --api http://127.0.0.1:8767 \\
        --fake http://127.0.0.1:9999

Two people open the same track: the first gets the edit lock, the second gets a view-only page that
names the first and refuses edits. The first reviews the whole track and submits it, which opens a
pull request (on the fake GitHub). Fails on any page error.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from playwright.sync_api import sync_playwright

HELP_SEEN = "drumReviewer.helpSeen.v1"
NAME = "drumReviewer.contributor.v1"


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--page", required=True)
    ap.add_argument("--api", required=True)
    ap.add_argument("--fake", required=True)
    args = ap.parse_args()
    with urllib.request.urlopen(args.api + "/api/tracks", timeout=10) as r:
        stem = next(t["stem"] for t in json.loads(r.read())["tracks"] if not t["has_human"] and not t["pending"])

    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])

        def person(name: str):
            ctx = browser.new_context()
            ctx.add_init_script(f"localStorage.setItem({HELP_SEEN!r}, '1'); localStorage.setItem({NAME!r}, {name!r});")
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(f"{name}: {e}"))
            # A refused lock or save (423/409/422) is an expected answer, but Chromium logs every 4xx.
            page.on("console", lambda m: m.type == "error" and not m.text.startswith(
                "Failed to load resource: the server responded with a status of 4") and errors.append(f"{name}: {m.text}"))
            page.on("dialog", lambda d: d.accept())
            page.goto(f"{args.page}#{stem}")
            page.wait_for_function("typeof S !== 'undefined' && S.stem !== null && S.meta !== null", timeout=30000)
            return page

        ana = person("Ana")
        ana.wait_for_function("S.ana !== null", timeout=120000)
        check(ana.evaluate("CFG.hosted && S.lockHeld && !S.readOnly"), "Ana opens the track and holds the edit lock")
        check("Ana" in ana.inner_text("#who"), "Ana's name is shown")
        check(ana.inner_text("#btn-save") == "Submit", "the save button reads Submit")

        bo = person("Bo")
        bo.wait_for_function("S.duration > 0", timeout=120000)
        check(bo.evaluate("S.readOnly && !S.lockHeld"), "Bo gets the track view-only")
        check("Ana" in bo.inner_text("#banner-warn"), "Bo is told Ana is editing")
        n = bo.evaluate("S.onsets.length")
        bo.locator("#overlay").hover()
        bo.keyboard.press("k")
        check(bo.evaluate("S.onsets.length") == n, "Bo's edits are refused")
        bo.evaluate("openTracks()")
        bo.wait_for_selector("#tracks-body tr")
        check("being edited by Ana" in bo.inner_text("#tracks-body"), "the track list says Ana is editing")

        ana.evaluate("edit(() => addRange(0, S.duration))")
        check(ana.evaluate("isComplete()"), "Ana marks the whole track reviewed")
        ana.wait_for_function("S.lastDraft !== null", timeout=15000)
        check(True, "the draft autosaves")
        ana.locator("#overlay").click()
        ana.keyboard.press("Control+s")
        ana.wait_for_function("S.readOnly && S.meta.pending && S.meta.pending.pr === 1", timeout=30000)
        check("pull request #1" in ana.inner_text("#banner-warn"), "Ana's submit opens pull request #1")
        with urllib.request.urlopen(args.fake + "/_fake/state", timeout=10) as r:
            st = json.loads(r.read())
        check(len(st["commits"]) == 1 and "Contributed-by: Ana" in st["commits"][0]["message"],
              "the pull request carries Ana's name")
        browser.close()
    check(not errors, "no page errors" + ("" if not errors else ": " + " | ".join(errors[:5])))
    print("browser: all checks passed")


if __name__ == "__main__":
    main()
