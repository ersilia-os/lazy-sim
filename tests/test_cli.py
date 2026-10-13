"""Smoke tests for the lazysim CLI, plus parity with the original CLAMP.

The tests use a temporary lazy-sim home. The CLAMP weights are downloaded into it
once, unless ``LAZYSIM_CLAMP_WEIGHTS`` points to a local copy. Everything else runs
offline: ``fetch`` reads a catalog of file:// URLs.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from lazy_sim.cli import cli
from lazy_sim.featurizers import clamp
from lazy_sim.io.remote import sha256
from lazy_sim.io.tables import HITS_COLUMNS

DATA = Path(__file__).parent / "data"
K = 10


def _run(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result.output


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        path = tmp_path_factory.mktemp("home")
        mp.setenv("LAZYSIM_HOME", str(path))
        yield path


@pytest.fixture(scope="module")
def library(home, tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("built") / "artifacts"
    _run("index", "-i", DATA / "library.csv", "-a", artifacts, "-t", 2, "--embeddings")
    return artifacts


def test_index(library):
    built = json.loads((library / "manifest.json").read_text())["library"]
    assert (built["indexed"], built["skipped"]) == (200, 2)
    skipped = {row["id"] for row in _read(library / "skipped.csv")}
    assert skipped == {"INVALID-1", "INVALID-2"}
    assert not (library / clamp.WEIGHTS_FILE).exists()
    assert sha256(clamp.ensure_weights()) == clamp.WEIGHTS_SHA256
    embeddings = np.load(library / "clamp_embeddings.npy")
    assert embeddings.shape == (200, clamp.DIM)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)


def test_search_and_info(library, tmp_path):
    hits_csv = tmp_path / "hits.csv"
    _run("search", "-i", DATA / "queries.csv", "-a", library, "-o", hits_csv, "-k", K)
    hits = _read(hits_csv)
    assert tuple(hits[0]) == HITS_COLUMNS
    queries = [q for q in _read(DATA / "queries.csv") if q["id"] != "INVALID-Q"]
    assert len(hits) == K * len(queries)
    for q in queries:
        top = next(h for h in hits if h["query_id"] == q["id"] and h["rank"] == "1")
        assert top["hit_smiles"] == q["smiles"]
        assert float(top["similarity"]) > 0.99
    assert "200" in _run("info", "-a", library)
    assert "torch" not in sys.modules


def test_fetch_and_catalog(library, home, tmp_path, monkeypatch):
    root = library.parent

    def files(*names):
        paths = [library / n for n in names]
        return [
            {
                "path": str(p.relative_to(root)),
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in paths
        ]

    searchable = ("library.db", "skipped.csv", "clamp.tv")
    entry = {
        "title": "Tiny test library",
        "version": "v1",
        "molecules": 200,
        "built_with": {"featurizer": "clamp", "weights_sha256": clamp.WEIGHTS_SHA256},
        "files": {
            "library": files(*searchable, "manifest.json"),
            "embeddings": files("clamp_embeddings.npy"),
        },
    }
    (root / "catalog.json").write_text(json.dumps({"libraries": {"tiny": entry}}))
    monkeypatch.setenv("LAZYSIM_CATALOG_URL", (root / "catalog.json").as_uri())

    assert "tiny" in _run("catalog", "--remote")
    assert "No libraries" in _run("catalog")
    _run("fetch", "tiny", "--embeddings")
    assert "tiny" in _run("catalog")
    assert (home / "libraries" / "tiny" / "clamp_embeddings.npy").exists()

    queries = DATA / "queries.csv"
    _run("search", "-i", queries, "-a", "tiny", "-o", tmp_path / "a.csv")
    _run("search", "-i", queries, "-a", library, "-o", tmp_path / "b.csv")
    assert _read(tmp_path / "a.csv") == _read(tmp_path / "b.csv")
    _run("info", "-a", "tiny")
    again = CliRunner().invoke(cli, ["fetch", "tiny"])
    assert again.exit_code != 0 and "not empty" in again.output


def test_clamp_matches_original(home):
    rows = _read(DATA / "clamp_reference.csv")
    reference = np.load(DATA / "clamp_reference.npy")
    fps = clamp.fingerprints([row["smiles"] for row in rows])
    embeddings = clamp.encode(clamp.load_weights(), fps)
    assert np.abs(embeddings - reference).max() < 1e-5
