"""Tests for tools/drum-reviewer/server.py: listing, seeding, saving and refusal paths.

Fixture audio is a few fake bytes: the server never decodes audio (the page does), it
only hashes and serves it.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import os
import threading
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).resolve().parents[1] / "tools" / "drum-reviewer" / "server.py"
_spec = importlib.util.spec_from_file_location("drum_reviewer_server", SERVER_PY)
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)

DUR = 10.0
JSON = {"Content-Type": "application/json"}
ADTOF_GEN = {"algorithm": "adtof_frame_rnn", "version": 1}


def _w(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _kw(onsets, **extra):
    return {"onsets": [{"time": t, "weight": 1.0, "type": ty} for t, ty in onsets], **extra}


@pytest.fixture
def edm(tmp_path):
    d = tmp_path / "edm"
    d.mkdir()
    for stem in ("a", "b", "c"):
        (d / f"{stem}.mp3").write_bytes(f"fake mp3 {stem}".encode() * 50)
    _w(d / "track_manifest.json", {s: {"duration": DUR} for s in ("a", "b", "c")})
    # a: ADTOF + band labels. ADTOF has 2 kicks + 1 snare + a hat; band agrees on one kick only.
    _w(d / "drums_adtof" / "a.kick_weighted.json",
       _kw([(1.0, "kick"), (2.0, "kick"), (1.5, "snare"), (1.25, "hihat")], generator=ADTOF_GEN))
    _w(d / "kick_weighted" / "a.kick_weighted.json",
       _kw([(1.01, "kick"), (1.5, "kick"), (3.0, "kick")], source="bandpass_onset_detect"))
    # b: band labels only.
    _w(d / "kick_weighted" / "b.kick_weighted.json", _kw([(0.5, "kick"), (0.75, "snare")],
                                                         source="bandpass_onset_detect"))
    # c: a human file (half reviewed) over ADTOF and band labels.
    _w(d / "drums_adtof" / "c.kick_weighted.json", _kw([(4.0, "kick")], generator=ADTOF_GEN))
    _w(d / "kick_weighted" / "c.kick_weighted.json", _kw([(4.0, "kick")]))
    _w(d / "drums_human" / "c.drums.json", {
        "schema": 1, "track": "c", "audio_sha256": sha(d / "c.mp3"), "duration": DUR,
        "started_from": {"source": "drums_adtof", "generator": ADTOF_GEN},
        "reviewed_ranges": [[0.0, 5.0]], "complete": False,
        "onsets": [{"time": 4.0012, "type": "kick"}, {"time": 4.5, "type": "snare"}],
        "updated": "2026-11-24T00:00:00+00:00",
    })
    return d


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def corpus(edm):
    return srv.Corpus(edm)


def body(c, stem, onsets, ranges=(), duration=DUR, **kw):
    return {"audio_sha256": c.audio_sha256(stem), "duration": duration,
            "onsets": [{"time": t, "type": ty} for t, ty in onsets],
            "reviewed_ranges": [list(r) for r in ranges], **kw}


# --- listing and seeding ----------------------------------------------------------


def test_track_list_status(corpus):
    tracks = {t["stem"]: t for t in srv.list_tracks(corpus)}
    assert sorted(tracks) == ["a", "b", "c"]
    a, b, c = tracks["a"], tracks["b"], tracks["c"]
    assert a["label_source"] == "drums_adtof" and a["counts"] == {"kick": 2, "snare": 1}
    # kick F = 2*1/(2+3) = 0.4; snare F = 0 (ADTOF 1, band 0) -> disagreement 1 - 0.2
    assert a["disagreement"] == pytest.approx(0.8)
    assert b["label_source"] == "kick_weighted" and b["disagreement"] is None
    assert c["label_source"] == "human" and c["has_human"] and not c["complete"]
    assert c["reviewed_share"] == pytest.approx(0.5)
    assert c["counts"] == {"kick": 1, "snare": 1}


def test_seed_order(corpus):
    a = srv.load_track(corpus, "a")
    assert a["seed"] == "drums_adtof" and a["warning"] is None
    assert a["started_from"]["source"] == "drums_adtof"
    assert a["started_from"]["generator"] == ADTOF_GEN
    assert a["started_from"]["file"] == "edm/drums_adtof/a.kick_weighted.json"
    assert [o["type"] for o in a["onsets"]] == ["kick", "snare", "kick"]  # sorted by time
    assert a["references"]["drums_adtof"]["hihat"] == [1.25]  # hats are reference only

    b = srv.load_track(corpus, "b")
    assert b["seed"] == "kick_weighted" and "known bad" in b["warning"]
    assert b["started_from"]["generator"]["source"] == "bandpass_onset_detect"

    c = srv.load_track(corpus, "c")
    assert c["seed"] == "human" and c["error"] is None
    assert c["onsets"] == [{"time": 4.0012, "type": "kick"}, {"time": 4.5, "type": "snare"}]
    assert c["reviewed_ranges"] == [[0.0, 5.0]]
    assert c["started_from"]["source"] == "drums_adtof"


def test_reference_set_is_a_ghost_layer_never_a_seed(corpus, edm):
    """seeds.json "references" (edm/drums_adtof_stem/, ADTOF on the drum stem) show up as ghost
    ticks only: never a seed, even when it's the track's only label file."""
    _w(edm / "drums_adtof_stem" / "a.kick_weighted.json",
       _kw([(1.0, "kick"), (1.5, "snare"), (1.75, "snare"), (1.25, "hihat")]))
    a = srv.load_track(corpus, "a")
    assert a["seed"] == "drums_adtof"
    assert [o["time"] for o in a["onsets"]] == [1.0, 1.5, 2.0]  # still the ADTOF seed
    assert a["references"]["drums_adtof_stem"]["snare"] == [1.5, 1.75]
    assert "drums_adtof_stem" not in srv.load_track(corpus, "b")["references"]
    (edm / "d.mp3").write_bytes(b"x" * 10)
    _w(edm / "drums_adtof_stem" / "d.kick_weighted.json", _kw([(1.0, "kick")]))
    d = srv.load_track(corpus, "d")
    assert d["seed"] == "empty" and d["onsets"] == []
    assert d["references"]["drums_adtof_stem"]["kick"] == [1.0]
    assert srv.track_summary(corpus, "d")["label_source"] == "none"
    pub = {s["name"]: s for s in srv.seeds_public(corpus)}
    assert pub["drums_adtof_stem"]["reference"] is True and "reference" not in pub["drums_adtof"]


def test_references_in_seeds_json_are_validated(tmp_path):
    lc = srv.labelcheck
    good = {"seeds": [{"name": "drums_adtof", "dir": "drums_adtof", "suffix": ".kick_weighted.json"}]}
    p = tmp_path / "seeds.json"
    p.write_text(json.dumps(good))
    assert lc.load_sources(p)[1] == []  # "references" is optional
    for refs in ([{"name": "drums_adtof", "dir": "x", "suffix": ".json"}],  # clashes with a seed
                 [{"name": "r", "dir": "../x", "suffix": ".json"}],
                 {"name": "r"}):
        p.write_text(json.dumps({**good, "references": refs}))
        with pytest.raises(ValueError):
            lc.load_sources(p)


def test_committed_reference_sets_parse():
    """Every committed reference file belongs to a corpus track and parses as the page expects."""
    edm = SERVER_PY.parents[2] / "edm"
    c = srv.Corpus(edm)
    assert c.references(), "seeds.json lists no reference sets"
    for r in c.references():
        files = sorted((edm / r.dir).glob(f"*{r.suffix}"))
        assert files, r.dir
        assert {f.name.removesuffix(r.suffix) for f in files} <= set(c.stems())
        for f in files:
            lab = srv.split_auto_labels(json.loads(f.read_text()), f.name)
            assert lab["kick"] or lab["snare"], f.name


def test_skipped_adtof_is_not_a_seed(corpus, edm):
    _w(edm / "drums_adtof" / "b.kick_weighted.json", {"skipped": True, "onsets": []})
    b = srv.load_track(corpus, "b")
    assert b["seed"] == "kick_weighted" and "skipped" in b["warning"]


def test_no_labels_starts_empty(corpus, edm):
    (edm / "d.mp3").write_bytes(b"x" * 10)
    d = srv.load_track(corpus, "d")
    assert d["seed"] == "empty" and d["onsets"] == [] and d["warning"]


# --- saving --------------------------------------------------------------------


def test_save_writes_complete_list(corpus, edm):
    r = srv.save_track(corpus, "a", body(corpus, "a", [(2.000049, "snare"), (1.00004, "kick"),
                                                       (1.00004, "kick")], ranges=[(0, 4)],
                                         started_from={"source": "drums_adtof"}))
    assert r["ok"] and r["complete"] is False and r["count"] == 2
    doc = json.loads((edm / "drums_human" / "a.drums.json").read_text())
    assert doc["schema"] == 1 and doc["track"] == "a"
    assert doc["audio_sha256"] == sha(edm / "a.mp3")
    assert doc["started_from"] == {"source": "drums_adtof"}
    # sorted, 4 decimals, exact duplicates collapsed
    assert doc["onsets"] == [{"time": 1.0, "type": "kick"}, {"time": 2.0, "type": "snare"}]
    assert doc["reviewed_ranges"] == [[0, 4]] and doc["complete"] is False
    assert doc["updated"].endswith("+00:00")
    assert (edm / "drums_human" / "a.drums.json").stat().st_mode & 0o777 == 0o644


def test_complete_when_fully_reviewed(corpus, edm):
    srv.save_track(corpus, "a", body(corpus, "a", [(1.0, "kick")],
                                     ranges=[(0.05, 3.0), (3.005, 6.0), (5.0, DUR)]))
    doc = json.loads((edm / "drums_human" / "a.drums.json").read_text())
    assert doc["complete"] is True and doc["reviewed_ranges"] == [[0.05, DUR]]


def test_existing_human_keeps_its_provenance(corpus, edm):
    srv.save_track(corpus, "c", body(corpus, "c", [(4.0, "kick")],
                                     started_from={"source": "something else"}))
    doc = json.loads((edm / "drums_human" / "c.drums.json").read_text())
    assert doc["started_from"]["source"] == "drums_adtof"


def test_draft_roundtrip_and_save_clears_it(corpus, edm):
    srv.save_draft(corpus, "a", body(corpus, "a", [(1.5, "snare")], ranges=[(0, 1)], seed="drums_adtof"))
    draft = edm / "drums_human" / ".drafts" / "a.drums.json"
    assert draft.is_file() and not (edm / "drums_human" / "a.drums.json").exists()
    t = srv.load_track(corpus, "a")
    assert t["seed"] == "drums_adtof"  # a draft never becomes the seed silently
    assert t["draft"]["onsets"] == [{"time": 1.5, "type": "snare"}]
    assert {x["stem"]: x for x in srv.list_tracks(corpus)}["a"]["has_draft"]
    srv.save_track(corpus, "a", body(corpus, "a", [(1.5, "snare")]))
    assert not draft.exists()


@pytest.mark.parametrize("bad", [
    {"onsets": [{"time": 1.0, "type": "hihat"}]},
    {"onsets": [{"time": -0.1, "type": "kick"}]},
    {"onsets": [{"time": DUR + 5, "type": "kick"}]},
    {"onsets": [{"time": float("nan"), "type": "kick"}]},
    {"onsets": [{"time": "1.0", "type": "kick"}]},
    {"onsets": "nope"},
    {"reviewed_ranges": [[3, 2]]},
    {"reviewed_ranges": [[0, 1, 2]]},
    {"duration": 0},
    {"duration": DUR + 30},  # disagrees with track_manifest.json
    {"started_from": 7},
])
def test_save_validation(corpus, edm, bad):
    b = {**body(corpus, "a", [(1.0, "kick")]), **bad}
    with pytest.raises(srv.ApiError) as e:
        srv.save_track(corpus, "a", b)
    assert e.value.status == 400
    assert not (edm / "drums_human" / "a.drums.json").exists()


def test_sha_mismatch_from_page_refused(corpus, edm):
    b = body(corpus, "a", [(1.0, "kick")])
    b["audio_sha256"] = "0" * 64
    with pytest.raises(srv.ApiError) as e:
        srv.save_track(corpus, "a", b)
    assert e.value.status == 409
    assert not (edm / "drums_human" / "a.drums.json").exists()


def test_human_file_for_other_audio_refused(corpus, edm):
    hp = edm / "drums_human" / "c.drums.json"
    doc = json.loads(hp.read_text())
    doc["audio_sha256"] = "f" * 64
    hp.write_text(json.dumps(doc))
    before = hp.read_text()
    t = srv.load_track(corpus, "c")
    assert t["error"] and "different audio" in t["error"]
    for fn in (srv.save_track, srv.save_draft):
        with pytest.raises(srv.ApiError) as e:
            fn(corpus, "c", body(corpus, "c", [(4.0, "kick")]))
        assert e.value.status == 409
    assert hp.read_text() == before


def test_draft_for_other_audio_not_offered(corpus, edm):
    _w(edm / "drums_human" / ".drafts" / "a.drums.json",
       {"audio_sha256": "e" * 64, "onsets": [], "reviewed_ranges": []})
    t = srv.load_track(corpus, "a")
    assert t["draft"] is None and "different audio" in t["warning"]


def test_write_is_atomic(corpus, edm, monkeypatch):
    hp = edm / "drums_human" / "c.drums.json"
    before = hp.read_text()

    def boom(src, dst):
        raise OSError("disk on fire")

    monkeypatch.setattr(srv.os, "replace", boom)
    with pytest.raises(OSError):
        srv.save_track(corpus, "c", body(corpus, "c", [(4.0, "kick")]))
    assert hp.read_text() == before
    assert sorted(os.listdir(hp.parent)) == ["c.drums.json"]  # no temp file left behind


def test_saved_file_format_is_line_per_onset(corpus, edm):
    srv.save_track(corpus, "a", body(corpus, "a", [(1.0, "kick"), (1.5, "snare")]))
    text = (edm / "drums_human" / "a.drums.json").read_text()
    assert '    {"time": 1.0, "type": "kick"},\n    {"time": 1.5, "type": "snare"}\n' in text
    assert json.loads(text)["onsets"][1]["type"] == "snare"


def test_scorer_reads_a_saved_file(corpus, edm):
    from blinky_scoring.drum_scoring import drum_label_path, load_drum_labels

    srv.save_track(corpus, "a", body(corpus, "a", [(1.0, "kick"), (1.5, "snare")], ranges=[(0, DUR)]))
    path = drum_label_path(edm, "a")
    assert path == edm / "drums_human" / "a.drums.json"
    labels = load_drum_labels(path, edm / "a.onsets_consensus.json")
    assert labels.kicks == [1.0] and labels.snares == [1.5] and labels.hats == [1.25]


# --- HTTP: traversal, caps, audio ranges ---------------------------------------------


@pytest.fixture
def web(edm):
    s = srv.make_server(edm, "127.0.0.1", 0)
    th = threading.Thread(target=s.serve_forever, daemon=True)
    th.start()

    def req(method, path, data=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", s.server_address[1], timeout=10)
        conn.request(method, path, body=data, headers=headers or {})
        r = conn.getresponse()
        out = (r.status, dict(r.getheaders()), r.read())
        conn.close()
        return out

    yield req
    s.shutdown()
    s.server_close()


def test_http_lists_and_serves_page(web):
    st, _, b = web("GET", "/api/tracks")
    assert st == 200 and len(json.loads(b)["tracks"]) == 3
    st, h, b = web("GET", "/")
    assert st == 200 and b"<title>Drum Reviewer</title>" in b


@pytest.mark.parametrize("path", [
    "/api/track?stem=../a",
    "/api/track?stem=..%2Fedm%2Fa",
    "/api/track?stem=/etc/passwd",
    "/api/track?stem=.drafts",
    "/api/track?stem=a%00",
    "/audio/..%2Ftrack_manifest.mp3",
    "/audio/../a.mp3",
    "/api/track?stem=nope",
    "/api/track",
])
def test_http_refuses_bad_stems(web, path):
    st, _, b = web("GET", path)
    assert st in (400, 404), (path, st)
    assert "error" in json.loads(b)


def test_http_save_traversal_refused(web, edm):
    data = json.dumps({"onsets": []}).encode()
    for stem in ("../a", "..%2F..%2Fx", "drums_human/c", "a.mp3/../b"):
        st, _, _ = web("POST", f"/api/save?stem={stem}", data, {"Content-Length": str(len(data)), **JSON})
        assert st in (400, 404)
    assert not list(edm.parent.glob("*.drums.json"))


def test_http_post_size_cap(web):
    st, _, b = web("POST", "/api/save?stem=a", b"x",
                    {"Content-Length": str(srv.MAX_POST_BYTES + 1), **JSON})
    assert st == 413


def test_http_save_and_sha_refusal(web, edm, corpus):
    good = json.dumps(body(corpus, "a", [(1.0, "kick")])).encode()
    st, _, b = web("POST", "/api/save?stem=a", good, {"Content-Length": str(len(good)), **JSON})
    assert st == 200 and json.loads(b)["ok"]
    bad = json.dumps({**body(corpus, "a", [(1.0, "kick")]), "audio_sha256": "1" * 64}).encode()
    st, _, b = web("POST", "/api/save?stem=a", bad, {"Content-Length": str(len(bad)), **JSON})
    assert st == 409 and "mismatch" in json.loads(b)["error"]


def test_http_refuses_cross_site_writes(web, edm, corpus):
    data = json.dumps(body(corpus, "a", [(1.0, "kick")], ranges=[(0, DUR)])).encode()
    n = {"Content-Length": str(len(data))}
    # a "simple" cross-origin request (no preflight) can't carry application/json
    assert web("POST", "/api/save?stem=a", data, {**n, "Content-Type": "text/plain"})[0] == 415
    assert web("POST", "/api/save?stem=a", data, n)[0] == 415
    st, _, _ = web("POST", "/api/save?stem=a", data, {**n, **JSON, "Origin": "https://evil.example"})
    assert st == 403
    # DNS rebinding: a hostile name resolving to this server
    st, _, _ = web("POST", "/api/draft?stem=a", data, {**n, **JSON, "Host": "evil.example:8766",
                                                        "Origin": "http://evil.example:8766"})
    assert st == 403
    assert web("GET", "/api/tracks", headers={"Host": "evil.example"})[0] == 403
    assert not (edm / "drums_human" / "a.drums.json").exists()
    assert not (edm / "drums_human" / ".drafts").exists()
    # the page itself: same origin, JSON
    st, _, _ = web("POST", "/api/save?stem=a", data, {**n, **JSON, "Origin": "http://localhost:8766",
                                                       "Host": "localhost:8766"})
    assert st == 200


def test_host_allowed():
    names = srv.local_names() | {"studio.example.invalid"}
    for ok in ("127.0.0.1:8766", "[::1]:8766", "192.0.2.20", "localhost:8766", "studio.example.invalid:1",
               f"{srv.socket.gethostname()}:8766"):
        assert srv.host_allowed(ok, names), ok
    for bad in ("evil.example", "evil.example:8766", "", None, "localhost.evil.example"):
        assert not srv.host_allowed(bad, names), bad


def test_http_audio_range(web, edm):
    raw = (edm / "a.mp3").read_bytes()
    st, h, b = web("GET", "/audio/a.mp3", headers={"Range": "bytes=10-19"})
    assert st == 206 and b == raw[10:20] and h["Content-Range"] == f"bytes 10-19/{len(raw)}"
    st, _, b = web("GET", "/audio/a.mp3")
    assert st == 200 and b == raw
