"""tools/drum-reviewer/labelcheck.py: the human-label checks corpus CI and the hosted reviewer run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "drum-reviewer"
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
    tools = r / "tools" / "drum-reviewer"
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


def test_must_be_complete(repo):
    p = check(repo, "a", human(repo / "edm", "a", ranges=[(0, 6)]))
    assert p == ["the track is not 100% reviewed (60.0%): only complete tracks are accepted"]
    assert check(repo, "a", human(repo / "edm", "a", ranges=[(0, 6)]), require_complete=False) == []


def test_deleting_most_hits_is_refused(repo):
    p = check(repo, "a", human(repo / "edm", "a", kicks=KICKS[:3], snares=SNARES[:2]))
    assert any("deletes 14 of the 19 hits in drums_adtof" in x for x in p), p


def test_reclassing_is_not_deleting(repo):
    # most snares relabelled as kicks: no hit is deleted, the counts stay in the references' range
    assert check(repo, "a", human(repo / "edm", "a", kicks=KICKS + SNARES[2:], snares=SNARES[:2])) == []
    # all of them: still no deletes, but an empty snare class while ADTOF has snares is refused
    p = check(repo, "a", human(repo / "edm", "a", kicks=KICKS + SNARES, snares=[]))
    assert p == ["no snare hits, but drums_adtof has some"]


def test_counts_far_outside_the_references(repo):
    many = [i * 0.05 for i in range(1, 199)]
    p = check(repo, "a", human(repo / "edm", "a", kicks=many))
    assert any("kick hits is far outside" in x for x in p), p


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


def test_pr_with_one_valid_file(repo, capsys):
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(repo / "edm", "a"))})
    assert lc.check_pr(repo, "main", None, override=False) == []
    assert "valid, 100% reviewed, plausible" in capsys.readouterr().out


def test_pr_must_change_one_file_only(repo):
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(repo / "edm", "a")),
                            "README.md": "sneaky"})
    (p,) = lc.check_pr(repo, "main", None, override=False)
    assert "only that one file" in p and "README.md" in p


def test_pr_cannot_delete_a_label_file(repo):
    write_and_commit(repo, {"edm/drums_human/a.drums.json": lc.dumps_doc(human(repo / "edm", "a"))})
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "pr")
    git(repo, "checkout", "-qb", "pr2")
    write_and_commit(repo, {"edm/drums_human/a.drums.json": None})
    (p,) = lc.check_pr(repo, "main", None, override=False)
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
    # the owner's labels-override skips plausibility, never the schema or completeness
    assert lc.check_pr(repo, "main", None, override=True) == []
    write_and_commit(repo, {"edm/drums_human/a.drums.json":
                            lc.dumps_doc(human(edm, "a", ranges=[(0, 3)]))})
    assert any("not 100% reviewed" in x for x in lc.check_pr(repo, "main", None, override=True))


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
                            lc.dumps_doc(human(repo / "edm", "a", ranges=[(0, 1)]))})
    assert lc.main(["--repo", str(repo), "--base", "main"]) == 1
