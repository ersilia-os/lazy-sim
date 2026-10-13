"""Smoke tests for the lazysim CLI, plus parity with the original CLAMP.

The tests use a temporary lazy-sim home. The CLAMP weights are downloaded into it
once, unless ``LAZYSIM_CLAMP_WEIGHTS`` points to a local copy. Everything else runs
offline: ``fetch`` reads a catalog of file:// URLs. The ECFP parts need the
``[ecfp]`` extra and are skipped without it.
"""

import csv
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from rdkit import Chem

from lazy_sim.cli import cli
from lazy_sim.featurizers import clamp, ecfp
from lazy_sim.io.remote import sha256
from lazy_sim.io.tables import HITS_COLUMNS

DATA = Path(__file__).parent / "data"
K = 10
HAS_FPSIM2 = importlib.util.find_spec("FPSim2") is not None
FEATURIZERS = ["clamp", "ecfp"] if HAS_FPSIM2 else ["clamp"]
needs_fpsim2 = pytest.mark.skipif(not HAS_FPSIM2, reason="needs lazy-sim[ecfp]")


def _run(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result.output


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _members():
    return [q for q in _read(DATA / "queries.csv") if q["id"] != "INVALID-Q"]


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        path = tmp_path_factory.mktemp("home")
        mp.setenv("LAZYSIM_HOME", str(path))
        yield path


@pytest.fixture(scope="module")
def library(home, tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("built") / "artifacts"
    featurizers = [arg for f in FEATURIZERS for arg in ("-f", f)]
    _run(
        "index",
        "-i",
        DATA / "library.csv",
        "-a",
        artifacts,
        "-t",
        2,
        *featurizers,
        "--embeddings",
    )
    return artifacts


def test_index(library):
    built = json.loads((library / "manifest.json").read_text())["library"]
    assert (built["indexed"], built["skipped"]) == (200, 2)
    skipped = {row["id"] for row in _read(library / "skipped.csv")}
    assert skipped == {"INVALID-1", "INVALID-2"}
    assert (
        sorted(p.stem for p in library.glob("*.json") if p.stem != "manifest")
        == FEATURIZERS
    )
    assert not (library / clamp.WEIGHTS_FILE).exists()
    assert sha256(clamp.ensure_weights()) == clamp.WEIGHTS_SHA256
    embeddings = np.load(library / "clamp_embeddings.npy")
    assert embeddings.shape == (200, clamp.DIM)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
    if HAS_FPSIM2:
        bits = np.load(library / "ecfp_embeddings.npy")
        assert bits.shape == (200, ecfp.DIM // 8) and bits.dtype == np.uint8


def test_search_and_info(library, tmp_path):
    hits_csv = tmp_path / "hits.csv"
    _run("search", "-i", DATA / "queries.csv", "-a", library, "-o", hits_csv, "-k", K)
    hits = _read(hits_csv)
    assert tuple(hits[0]) == HITS_COLUMNS
    assert len(hits) == K * len(_members())
    for q in _members():
        top = next(h for h in hits if h["query_id"] == q["id"] and h["rank"] == "1")
        assert top["hit_smiles"] == q["smiles"]
        assert float(top["similarity"]) > 0.99
    assert "200" in _run("info", "-a", library)
    assert "torch" not in sys.modules


def test_search_without_ids(library, tmp_path):
    smiles_only = tmp_path / "smiles.csv"
    with open(smiles_only, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles"])
        writer.writerows([q["smiles"]] for q in _members())
        writer.writerow(["C1CC"])  # invalid: skipped, reported as row-6
    hits_csv = tmp_path / "hits.csv"
    result = CliRunner().invoke(
        cli, ["search", "-i", str(smiles_only), "-a", str(library), "-o", str(hits_csv)]
    )
    assert result.exit_code == 0, result.output
    assert "row-6" in result.output
    ids = {h["query_id"] for h in _read(hits_csv)}
    assert ids == {
        Chem.MolToInchiKey(Chem.MolFromSmiles(q["smiles"])) for q in _members()
    }


@needs_fpsim2
def test_search_ecfp(library, tmp_path):
    hits_csv = tmp_path / "hits.csv"
    _run(
        "search",
        "-i",
        DATA / "queries.csv",
        "-a",
        library,
        "-o",
        hits_csv,
        "-k",
        K,
        "-f",
        "ecfp",
    )
    hits = _read(hits_csv)
    assert len(hits) == K * len(_members())
    con = sqlite3.connect(library / "library.db")
    row_of = dict(con.execute("SELECT id, row FROM library"))
    bits = np.unpackbits(np.load(library / "ecfp_embeddings.npy"), axis=1).astype(bool)
    for q in _members():
        mine = [h for h in hits if h["query_id"] == q["id"]]
        assert mine[0]["hit_smiles"] == q["smiles"]
        assert mine[0]["similarity"] == "1.0000"
        query = bits[row_of[q["id"]]]
        for h in mine:  # the stored bits give the same Tanimoto as FPSim2
            hit = bits[row_of[h["hit_id"]]]
            tanimoto = (query & hit).sum() / (query | hit).sum()
            assert abs(tanimoto - float(h["similarity"])) < 1e-4


@needs_fpsim2
def test_add_featurizer(home, tmp_path):
    lib = tmp_path / "lib"
    _run("index", "-i", DATA / "library.csv", "-a", lib, "-t", 2)
    _run("index", "-a", lib, "-f", "ecfp")
    assert (lib / "clamp.json").exists() and (lib / "ecfp.json").exists()
    again = CliRunner().invoke(cli, ["index", "-a", str(lib), "-f", "ecfp"])
    assert again.exit_code != 0 and "already has ecfp" in again.output


def test_ecfp_needs_extra(home, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "FPSim2", None)
    args = ["index", "-i", str(DATA / "library.csv"), "-a", str(tmp_path / "x")]
    result = CliRunner().invoke(cli, args + ["-f", "ecfp"])
    assert result.exit_code != 0 and "lazy-sim[ecfp]" in result.output


def test_fetch_and_catalog(library, home, tmp_path, monkeypatch):
    root = library.parent

    def files(*names):
        return [
            {
                "path": str((library / n).relative_to(root)),
                "sha256": sha256(library / n),
                "bytes": (library / n).stat().st_size,
            }
            for n in names
        ]

    featurizers = {}
    for name in FEATURIZERS:
        fm = json.loads((library / f"{name}.json").read_text())
        featurizers[name] = {
            "similarity": fm["similarity"],
            "built_with": {"weights_sha256": fm.get("weights_sha256")},
            "files": files(fm["index_file"], f"{name}.json"),
            "embeddings": files(f"{name}_embeddings.npy"),
        }
    entry = {
        "title": "Tiny test library",
        "version": "v1",
        "molecules": 200,
        "files": files("library.db", "skipped.csv", "manifest.json"),
        "featurizers": featurizers,
    }
    (root / "catalog.json").write_text(json.dumps({"libraries": {"tiny": entry}}))
    monkeypatch.setenv("LAZYSIM_CATALOG_URL", (root / "catalog.json").as_uri())

    assert "tiny" in _run("catalog", "--remote")
    assert "No libraries" in _run("catalog")
    _run("fetch", "tiny")
    fetched = home / "libraries" / "tiny"
    assert not (fetched / "clamp_embeddings.npy").exists()
    _run("fetch", "tiny", "--embeddings")  # adds them to the library already here
    assert (fetched / "clamp_embeddings.npy").exists()
    assert not (fetched / "ecfp.json").exists()
    if HAS_FPSIM2:
        _run("fetch", "tiny", "-f", "ecfp")
        assert (fetched / "ecfp.json").exists()
        assert "ecfp" in _run("catalog")
    _run("fetch", "tiny")  # nothing left to download

    queries = DATA / "queries.csv"
    _run("search", "-i", queries, "-a", "tiny", "-o", tmp_path / "a.csv")
    _run("search", "-i", queries, "-a", library, "-o", tmp_path / "b.csv")
    assert _read(tmp_path / "a.csv") == _read(tmp_path / "b.csv")
    _run("info", "-a", "tiny")


def test_clamp_matches_original(home):
    rows = _read(DATA / "clamp_reference.csv")
    reference = np.load(DATA / "clamp_reference.npy")
    fps = clamp.fingerprints([row["smiles"] for row in rows])
    embeddings = clamp.encode(clamp.load_weights(), fps)
    assert np.abs(embeddings - reference).max() < 1e-5
