"""Maintainer script: build the ``ersilia-reference`` library that ``lazysim fetch`` serves.

1. Download the Ersilia Reference Library (pinned commit) and check its sha256.
2. Use each molecule's InChIKey as its id; SMILES are kept exactly as given.
3. Index it with lazy-sim (CLAMP, ECFP and SAND), keeping the full embeddings.
4. Lay the files out as on S3 under ``data/libraries/``, and write ``catalog.json``
   as the published one plus this library (so nobody else's entries are lost).

Published versions are never edited: a rebuild goes in a new version folder. The
one exception is ``--add FEATURIZER``, which indexes the published library with
one more featurizer. It only adds files (the featurizer's index, manifest and
embeddings) and a catalog entry; the library's own files stay as published, so
rows stay aligned for everyone who already fetched it.

Usage
-----
LAZYSIM_CLAMP_WEIGHTS=data/clamp/clamp_compound_encoder_v1.npz \\
    LAZYSIM_SAND_WEIGHTS=data/sand/sand_encoder_v1.npz \\
    python scripts/build_reference_library.py
python scripts/build_reference_library.py --add sand

Then upload with ``eosvc upload --path data/libraries``.
"""

import csv
import json
import multiprocessing as mp
import shutil
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

import lazy_sim
from lazy_sim.io import artifacts, remote
from lazy_sim.io.remote import catalog_url, sha256

NAME, VERSION, TITLE = "ersilia-reference", "v1", "Ersilia Reference Library"
FEATURIZERS = ["clamp", "ecfp", "sand"]
COMMIT = "7b4f378351cb480db61018198a19356eb288a3f7"
SOURCE = (
    "https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs/blob/"
    f"{COMMIT}/inputs/reference_library_smiles.csv"
)
RAW = SOURCE.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
SOURCE_SHA256 = "9437df08beca7af51e2b519c4710b04385e12e4d29abb66acb267c28f9934658"
BUILD = Path("tmp/build") / NAME
LIBRARIES = Path("data/libraries")
OUT = LIBRARIES / NAME / VERSION


def inchikeys(smiles_list):
    _block = rdBase.BlockLogs()  # noqa: F841
    keys = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        keys.append(Chem.MolToInchiKey(mol) if mol is not None else "")
    return keys


def prepare_input():
    BUILD.mkdir(parents=True, exist_ok=True)
    raw = BUILD / "reference_library_smiles.csv"
    urllib.request.urlretrieve(RAW, raw)
    assert sha256(raw) == SOURCE_SHA256, "the source file changed"
    with open(raw, newline="") as fh:
        smiles = [row["standardized_smiles"] for row in csv.DictReader(fh)]
    batches = [smiles[i : i + 10_000] for i in range(0, len(smiles), 10_000)]
    with mp.get_context("spawn").Pool() as pool:
        keys = [k for batch in pool.map(inchikeys, batches) for k in batch]
    ids = [k or f"row-{i + 1}" for i, k in enumerate(keys)]
    library_csv = BUILD / f"ersilia_reference_library_{VERSION}.csv"
    with open(library_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "smiles"])
        writer.writerows(zip(ids, smiles))
    repeated = sum(c - 1 for c in Counter(k for k in keys if k).values() if c > 1)
    print(
        f"{len(smiles)} molecules, {keys.count('')} without InChIKey, "
        f"{repeated} repeated InChIKeys"
    )
    return library_csv


def describe(*paths):
    return [
        {
            "path": p.relative_to(LIBRARIES).as_posix(),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in paths
    ]


def published_catalog():
    """Return the catalog.json on S3, so libraries added from elsewhere are kept."""
    try:
        with urllib.request.urlopen(catalog_url(), timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # nothing published yet; S3 answers 403 here
            return {"libraries": {}}
        raise


def featurizer_entry(name):
    """Describe a featurizer of the library in ``OUT`` for ``catalog.json``."""
    fm = json.loads((OUT / f"{name}.json").read_text())
    built_with = {"lazysim": fm["lazysim_version"], fm["engine"]: fm["engine_version"]}
    if "weights_sha256" in fm:
        built_with["weights_sha256"] = fm["weights_sha256"]
    return {
        "similarity": fm["similarity"],
        "built_with": built_with,
        "files": describe(OUT / fm["index_file"], OUT / f"{name}.json"),
        "embeddings": describe(OUT / f"{name}_embeddings.npy"),
    }


def publish(built):
    """Move a built library into the S3 layout and record it in catalog.json."""
    manifest = json.loads((built / "manifest.json").read_text())
    manifest["source"] = {
        "name": NAME,
        "version": VERSION,
        "title": TITLE,
        "url": SOURCE,
    }
    (built / "manifest.json").write_text(json.dumps(manifest, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    for path in built.iterdir():
        shutil.move(path, OUT / path.name)
    featurizers = {name: featurizer_entry(name) for name in FEATURIZERS}
    catalog = published_catalog()
    catalog["libraries"][NAME] = {
        "title": TITLE,
        "version": VERSION,
        "molecules": manifest["library"]["indexed"],
        "source": SOURCE,
        "license": "GPL-3.0 (ersilia-model-hub-maintained-inputs)",
        "files": describe(
            *(OUT / n for n in ("library.db", "skipped.csv", "manifest.json"))
        ),
        "featurizers": featurizers,
    }
    catalog_path = LIBRARIES / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote {OUT} and {catalog_path}")


def add(names):
    """Index the published library with more featurizers and list them in the catalog."""
    catalog = published_catalog()
    published = remote.read_catalog()[NAME]
    OUT.mkdir(parents=True, exist_ok=True)
    for f in published["files"]:
        path = OUT / Path(f["path"]).name
        if not path.exists():
            remote.download_files([(f["url"], path, f["sha256"], f["bytes"])], NAME)
        assert sha256(path) == f["sha256"], f"{path} differs from the published file"
    missing = [n for n in names if n not in artifacts.featurizers_in(OUT)]
    if missing:
        lazy_sim.index(None, OUT, featurizers=missing, embeddings=True)
    for name in names:
        catalog["libraries"][NAME]["featurizers"][name] = featurizer_entry(name)
    catalog_path = LIBRARIES / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Added {', '.join(names)} to {OUT}; wrote {catalog_path}")


def main():
    library_csv = prepare_input()
    built = BUILD / "index"
    lazy_sim.index(library_csv, built, featurizers=FEATURIZERS, embeddings=True)
    publish(built)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--add"]:
        add(sys.argv[2:])
    else:
        main()
