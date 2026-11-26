"""tools/labels/labelcheck.py: the anti-vandalism checks corpus CI runs on human label files (the labels
service in blinky_time runs the same file before it publishes)."""

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

DUR = 10.0
KICKS = [0.5 + i for i in range(10)]
SNARES = [1.0 + i for i in range(9)]


def _w(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _kw(kicks, snares):
    ons = [{"time": t, "type": "kick"} for t in kicks] + [{"time": t, "type": "snare"} for t in snares]
    return {"onsets": sorted(ons, key=lambda o: o["time"])}


def human(edm: Path, stem: str, kicks=KICKS, snares=SNARES, ranges=((0.0, DUR),), **over):
    ons = sorted([(round(t, 4), "kick") for t in kicks] + [(round(t, 4), "snare") for t in snares])
    ranges = [list(r) for r in ranges]
    doc = {"schema": 1, "track": stem,
           "audio_sha256": hashlib.sha256((edm / f"{stem}.mp3").read_bytes()).hexdigest(),
           "duration": DUR, "started_from": {"source": "drums_adtof"},
           "reviewed_ranges": ranges, "complete": lc.is_complete(ranges, DUR),
           "onsets": [{"time": t, "type": ty} for t, ty in ons],
           "updated": "2026-11-25T00:00:00+00:00"}
    doc.update(over)
    return doc


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "corpus"
    edm = r / "edm"
    edm.mkdir(parents=True)
    for stem in ("a", "b"):
        (edm / f"{stem}.mp3").write_bytes(stem.encode() * 100)
        _w(edm / "drums_adtof" / f"{stem}.kick_weighted.json", _kw(KICKS, SNARES))
        _w(edm / "kick_weighted" / f"{stem}.kick_weighted.json", _kw(KICKS + SNARES, []))
    _w(edm / "track_manifest.json", {s: {"duration": DUR} for s in ("a", "b")})
    tools = r / "tools" / "labels"
    tools.mkdir(parents=True)
    (tools / "seeds.json").write_text(lc.DEFAULT_SEEDS_FILE.read_text())
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    git(r, "checkout", "-qb", "pr")
    return r


def git(r, *a):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=r,
                   check=True, capture_output=True)


def check(repo, stem, doc, base=None, **kw):
    return lc.check_human_doc(repo / "edm", stem, doc, base, lc.load_seeds(), **kw)


def test_valid_file(repo):
    assert check(repo, "a", human(repo / "edm", "a")) == []


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.update(extra=1), "unknown keys"),
    (lambda d: d.pop("updated"), "missing keys"),
    (lambda d: d.update(schema=2), "schema"),
    (lambda d: d.update(track="b"), "track is"),
    (lambda d: d.update(audio_sha256="0" * 64), "audio_sha256"),
    (lambda d: d.update(duration=DUR + 5), "track_manifest"),
    (lambda d: d["onsets"].reverse(), "sorted"),
    (lambda d: d["onsets"].append(dict(d["onsets"][-1])), "sorted"),
    (lambda d: d["onsets"][0].update(time=0.12345), "4 decimals"),
    (lambda d: d["onsets"][0].update(type="hihat"), "kick or snare"),
    (lambda d: d["onsets"][0].update(weight=1), "malformed onset"),
    (lambda d: d.update(reviewed_ranges=[[0, 5], [5.001, DUR]]), "merged"),
    (lambda d: d.update(complete=False), "complete is False"),
    (lambda d: d.update(updated="yesterday"), "ISO 8601"),
])
def test_schema_problems(repo, mutate, expect):
    doc = human(repo / "edm", "a")
    mutate(doc)
    p = check(repo, "a", doc)
    assert any(expect in x for x in p), p


GOOD_NOTES = [{"text": "snares under the filter sweep 1:20-1:45 all dropped", "range": [1.0, 4.5]},
              {"text": "ADTOF calls the rim shots hats", "range": None},
              {"text": "kick < snare here, <b>not</b> HTML: plain text \u00e9", "range": [0.0, DUR]}]


def test_reviewer_notes_are_accepted(repo):
    doc = human(repo / "edm", "a", notes=GOOD_NOTES)
    assert check(repo, "a", doc) == []
    assert check(repo, "a", human(repo / "edm", "a", notes=[])) == []
    # they round-trip through the file format, one per line
    text = lc.dumps_doc(doc)
    assert json.loads(text)["notes"] == GOOD_NOTES and text.count('{"text": ') == 3


@pytest.mark.parametrize("notes,expect", [
    ("snares dropped", "notes must be a list"),
    ([{"text": "x", "range": None}] * (lc.MAX_NOTES + 1), "notes must be a list"),
    ([{"text": "x"}], "malformed note"),
    ([{"text": "x", "range": None, "author": "me"}], "malformed note"),
    (["x"], "malformed note"),
    ([{"text": "", "range": None}], "1-1000 characters"),
    ([{"text": "x" * (lc.MAX_NOTE_CHARS + 1), "range": None}], "1-1000 characters"),
    ([{"text": 5, "range": None}], "1-1000 characters"),
    ([{"text": "bell\x07", "range": None}], "control or format"),
    ([{"text": "two\nlines", "range": None}], "control or format"),
    ([{"text": "rtl \u202e override", "range": None}], "control or format"),
    ([{"text": "zero\u200bwidth", "range": None}], "control or format"),
    ([{"text": " padded", "range": None}], "control or format"),
    ([{"text": "x", "range": [2.0, 1.0]}], "bad note range"),
    ([{"text": "x", "range": [0.0, DUR + 5]}], "bad note range"),
    ([{"text": "x", "range": [0.12345, 1.0]}], "bad note range"),
    ([{"text": "x", "range": [0.0]}], "bad note range"),
    ([{"text": "x", "range": "0-1"}], "bad note range"),
])
def test_bad_reviewer_notes_are_refused(repo, notes, expect):
    p = check(repo, "a", human(repo / "edm", "a", notes=notes))
    assert any(expect in x for x in p), p


def test_clean_note_text():
    assert lc.clean_note_text("  snares\u202e under\tthe\x00 sweep\n1:20  ") == "snares under the sweep 1:20"
    assert lc.clean_note_text("\u200b\u2028") == ""
    assert lc.clean_note_text("<script>x</script> \u00e9") == "<script>x</script> \u00e9"  # data, kept as text


def test_partial_progress_is_fine(repo):
    # the scorer only uses a complete file, so partial ones are accepted
    assert check(repo, "a", human(repo / "edm", "a", ranges=[(0, 6)])) == []


def test_deleting_most_hits_is_refused(repo):
    p = check(repo, "a", human(repo / "edm", "a", kicks=KICKS[:3], snares=SNARES[:2]))
    assert any("deletes 14 of the 19 hits in drums_adtof" in x for x in p), p


def test_reclassing_is_not_deleting(repo):
    # most snares relabelled as kicks: no hit is deleted, the counts stay in the references' range
    assert check(repo, "a", human(repo / "edm", "a", kicks=KICKS + SNARES[2:], snares=SNARES[:2])) == []
    # all of them: still no deletes, but an empty snare class while ADTOF has snares is refused
    p = check(repo, "a", human(repo / "edm", "a", kicks=KICKS + SNARES, snares=[]))
    assert p == ["no snare hits, but drums_adtof has some"]


def test_empty_class(repo):
    # no kicks, while every reference has some
    p = lc.plausibility({"kick": [], "snare": SNARES}, None, {"x": {"kick": KICKS, "snare": SNARES}})
    assert p == ["no kick hits, but x has some"]
    # an empty class is fine only when every reference has none in it too
    assert lc.plausibility({"kick": KICKS, "snare": []}, None,
                           {"x": {"kick": KICKS, "snare": []}}) == []


def test_replaces_the_existing_human_file(repo):
    edm = repo / "edm"
    base = human(edm, "a", kicks=[0.25 + i * 0.5 for i in range(20)], snares=[])
    new = human(edm, "a", kicks=[0.25], snares=[])
    p = check(repo, "a", new, base=base)
    assert any("in the existing human file" in x for x in p), p


def write_and_commit(repo, files: dict[str, str | None]):
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "change")


def test_a_batch_of_label_files(repo, capsys):
    edm = repo / "edm"
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(edm, "a")),
                            "edm/drums_human/b.drums.json": lc.dumps_doc(human(edm, "b", ranges=[(0, 2)]))})
    assert lc.check_pr(repo, "main", None, override=False) == []
    assert capsys.readouterr().out.count("valid, not vandalism") == 2


def test_one_bad_file_in_a_batch_is_named(repo):
    edm = repo / "edm"
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(edm, "a")),
                            "edm/drums_human/b.drums.json": lc.dumps_doc(human(edm, "b", kicks=KICKS[:2], snares=SNARES[:1]))})
    (p,) = lc.check_pr(repo, "main", None, override=False)
    assert p.startswith("edm/drums_human/b.drums.json: deletes")


def test_a_labels_branch_changes_only_label_files(repo):
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(repo / "edm", "a")),
                            "README.md": "sneaky"})
    assert lc.check_pr(repo, "main", None, override=False) == []  # an ordinary PR may mix
    (p,) = lc.check_pr(repo, "main", None, override=False, only_labels=True)
    assert "may only change human label files" in p and "README.md" in p


def test_pr_cannot_delete_a_label_file(repo):
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(repo / "edm", "a"))})
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "pr")
    git(repo, "checkout", "-qb", "pr2")
    write_and_commit(repo, {"edm/drums_human/a.drums.json": None})
    (p,) = lc.check_pr(repo, "main", None, override=False)
    assert "status D" in p and "labels-override" in p
    # the owner's labels-override label allows it (reverting a batch that added the file) ...
    assert lc.check_pr(repo, "main", None, override=True) == []
    # ... but never on a labels/ batch branch
    (p,) = lc.check_pr(repo, "main", None, override=True, only_labels=True)
    assert "status D" in p


def test_pr_without_label_files_is_ignored(repo):
    write_and_commit(repo, {"README.md": "docs", "tools/x.py": "pass"})
    assert lc.check_pr(repo, "main", None, override=False) == []


def test_pr_plausibility_against_the_base_branch_file(repo):
    edm = repo / "edm"
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(edm, "a"))})
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "pr")
    git(repo, "checkout", "-qb", "pr2")
    write_and_commit(repo, {"edm/drums_human/a.drums.json":
                            lc.dumps_doc(human(edm, "a", kicks=KICKS[:2], snares=SNARES[:2]))})
    p = lc.check_pr(repo, "main", None, override=False)
    assert any("existing human file" in x for x in p), p
    # the owner's labels-override skips the vandalism checks, never the schema
    assert lc.check_pr(repo, "main", None, override=True) == []
    write_and_commit(repo, {"edm/drums_human/a.drums.json":
                            lc.dumps_doc(human(edm, "a", schema=2))})
    assert any("schema must be 1" in x for x in lc.check_pr(repo, "main", None, override=True))


def test_check_all_schema_only(repo):
    edm = repo / "edm"
    (edm / "drums_human").mkdir()
    (edm / "drums_human" / "a.drums.json").write_text(lc.dumps_doc(human(edm, "a", ranges=[(0, 3)])))
    assert lc.check_all(repo) == []  # partial files are fine on main
    (edm / "drums_human" / "b.drums.json").write_text(json.dumps({"schema": 1}))
    assert any("b.drums.json: missing keys" in x for x in lc.check_all(repo))


def test_cli_exit_codes(repo):
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(repo / "edm", "a"))})
    assert lc.main(["--repo", str(repo), "--base", "main"]) == 0
    write_and_commit(repo, {"edm/drums_human/a.drums.json":
                            lc.dumps_doc(human(repo / "edm", "a", kicks=[], snares=SNARES[:1]))})
    assert lc.main(["--repo", str(repo), "--base", "main"]) == 1
