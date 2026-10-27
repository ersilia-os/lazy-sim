"""Maintainer script: build an Ersilia-maintained library that ``lazysim fetch`` serves.

1. Download the library's source file and check it: the sha256 where the source is
   pinned, and always the number of molecules.
2. Write the ``id`` and ``smiles`` columns lazy-sim reads. The Ersilia Reference
   Library takes each molecule's InChIKey as its id; the other libraries keep the
   id of their supplier. SMILES are kept exactly as given.
3. Index it with lazy-sim, one featurizer at a time (ECFP, CLAMP, Mol-JEPA and SAND),
   keeping the full embeddings so users can fetch them (``--no-embeddings`` skips
   them, which cannot be undone later: a featurizer is indexed once).
4. Lay the files out as on S3 under ``data/libraries/``, and write ``catalog.json``
   as the published one plus this library (so nobody else's entries are lost).

Every finished featurizer stays in ``tmp/build/<library>/index``, so a run that stops
(a crash, a killed job) resumes with the featurizers still missing instead of from
scratch. A featurizer that was interrupted starts again from its beginning. The seconds
each one took are kept in ``tmp/build/<library>/timings.json``.

Published libraries are never edited: a rebuild is published under a new name
(``..._v1``). The one exception is ``--add FEATURIZER``, which indexes the published
library with one more featurizer. It only adds files (the featurizer's index, manifest
and embeddings) and a catalog entry; the library's own files stay as published, so
rows stay aligned for everyone who already fetched it.

Usage
-----
python scripts/build_library.py --list
python scripts/build_library.py enamine_hll_460K_v0 -t 14
python scripts/build_library.py enamine_hll_460K_v0 -f ecfp
python scripts/build_library.py ersilia_reference_library_1M_v0 --add moljepa -t 10

The weights are read from ``LAZYSIM_<NAME>_WEIGHTS`` (CLAMP, SAND and MOLJEPA) when
set, and downloaded to ``~/lazysim/models`` otherwise. ``-t`` caps worker processes
and threads (all CPUs by default); numpy is already loaded when the script reads it, so
set ``OPENBLAS_NUM_THREADS`` too.

Then upload with ``eosvc upload --path data/libraries``.
"""

import argparse
import csv
import json
import multiprocessing as mp
import shutil
import time
import urllib.error
import urllib.request
from array import array
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

import lazy_sim
from lazy_sim.io import artifacts, remote
from lazy_sim.io.remote import catalog_url, sha256

# Fastest first, so a failure late in a build costs the least. Only the SAND and
# Mol-JEPA times are measured; ECFP runs no neural network.
FEATURIZERS = ["ecfp", "clamp", "moljepa", "sand"]
# Featurizers whose licence restricts how their index and embeddings may be used
LICENSES = {"moljepa": "CC BY-NC 4.0 (Mol-JEPA weights), non-commercial use only"}
LICENSE = "GPL-3.0 (ersilia-model-hub-maintained-inputs)"

_REFERENCE_COMMIT = "7b4f378351cb480db61018198a19356eb288a3f7"
_MAINTAINED = (
    "https://eosvc-public.s3.eu-central-1.amazonaws.com/"
    "ersilia-model-hub-maintained-inputs/data/processed"
)


def _maintained(folder, title, molecules, sha256=None):
    """Describe a library of ``data/processed`` in the Ersilia Maintained Inputs."""
    return {
        "title": title,
        "version": "v0",
        "molecules": molecules,
        "source": f"{_MAINTAINED}/{folder}/{folder}_smiles_ids.csv",
        "sha256": sha256,  # None until the first build prints it
        "smiles": "smiles",
        "id": "collection_id",
    }


# The name is the library's published name: <library>_<size>_v<N>
LIBRARIES = {
    "ersilia_reference_library_1M_v0": {
        "title": "Ersilia Reference Library",
        "version": "v0",
        "molecules": 1_355_109,
        "source": (
            "https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs/blob/"
            f"{_REFERENCE_COMMIT}/inputs/reference_library_smiles.csv"
        ),
        "sha256": "9437df08beca7af51e2b519c4710b04385e12e4d29abb66acb267c28f9934658",
        "smiles": "standardized_smiles",
        "id": None,  # the InChIKey
    },
    "enamine_hll_460K_v0": _maintained(
        "enamine_hit_locator_460k_v0",
        "Enamine Hit Locator Library",
        460_160,
        "63849870ef48c7316070c0b38dc6af0a4fff4106a5c5a703f3ac07148067e8c2",
    ),
    "coconut_715K_v0": _maintained("coconut_715k_v0", "COCONUT", 715_822),
    "enamine_lsc_2M_v0": _maintained(
        "enamine_liquid_stock_2m_v0", "Enamine Liquid Stock Collection", 2_492_445
    ),
    "molport_5M_v0": _maintained(
        "molport_screening_compounds_5m_v0", "Molport Screening Compounds", 5_322_863
    ),
    "enamine_real_sample_10M_v0": _maintained(
        "enamine_real_sample_10m_v0", "Enamine REAL Sample", 10_353_526
    ),
}
BUILD = Path("tmp/build")
OUTPUT = Path("data/libraries")


def inchikeys(smiles_list):
    _block = rdBase.BlockLogs()  # noqa: F841
    keys = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        keys.append(Chem.MolToInchiKey(mol) if mol is not None else "")
    return keys


def fetch_source(name):
    """Download the source file of a library once, and check it."""
    lib = LIBRARIES[name]
    folder = BUILD / name
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / Path(lib["source"]).name
    if not raw.exists():
        url = lib["source"].replace("github.com", "raw.githubusercontent.com")
        part = raw.with_name(raw.name + ".part")  # a stopped download is never kept
        urllib.request.urlretrieve(url.replace("/blob/", "/"), part)
        part.replace(raw)
    digest = sha256(raw)
    if lib["sha256"] is None:
        print(f"sha256 of the source: {digest} (pin it in LIBRARIES)")
    else:
        assert digest == lib["sha256"], "the source file changed"
    return raw


def _write_with_inchikeys(raw, smiles_column, library_csv):
    with open(raw, newline="") as fh:
        smiles = [row[smiles_column] for row in csv.DictReader(fh)]
    batches = [smiles[i : i + 10_000] for i in range(0, len(smiles), 10_000)]
    with mp.get_context("spawn").Pool() as pool:
        keys = [k for batch in pool.map(inchikeys, batches) for k in batch]
    ids = [k or f"row-{i + 1}" for i, k in enumerate(keys)]
    with open(library_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "smiles"])
        writer.writerows(zip(ids, smiles))
    repeated = sum(c - 1 for c in Counter(k for k in keys if k).values() if c > 1)
    print(
        f"{len(smiles)} molecules, {keys.count('')} without InChIKey, "
        f"{repeated} repeated InChIKeys"
    )
    if repeated:
        raise ValueError(f"{raw} has {repeated} repeated InChIKeys")
    return len(smiles)


def _repeated(hashes):
    """Return the values that occur more than once among `hashes`."""
    ordered = sorted(hashes)
    return {a for a, b in zip(ordered, ordered[1:]) if a == b}


def _examples(raw, column, hashes, n=3):
    """Return up to `n` values of `column` in `raw` whose hash is in `hashes`."""
    found = []
    with open(raw, newline="") as fh:
        for row in csv.DictReader(fh):
            if hash(row[column]) in hashes:
                found.append(row[column])
                if len(found) == n:
                    break
    return found


def _write_with_ids(raw, smiles_column, id_column, library_csv):
    # Streamed, because the largest library has more than ten million rows. Repeats
    # are looked for by 64-bit hash, which keeps the memory use low.
    ids, smiles = array("q"), array("q")
    n = blank = 0
    with open(raw, newline="") as src, open(library_csv, "w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(["id", "smiles"])
        for row in csv.DictReader(src):
            n += 1
            blank += not row[id_column]
            mol_id = row[id_column] or f"row-{n}"
            ids.append(hash(mol_id))
            smiles.append(hash(row[smiles_column]))
            writer.writerow([mol_id, row[smiles_column]])
    print(f"{n} molecules, {blank} without an id")
    for column, hashes in ((id_column, ids), (smiles_column, smiles)):
        repeated = _repeated(hashes)
        if repeated:
            raise ValueError(
                f"{raw} has {len(repeated)} repeated values in {column!r}, "
                f"such as {_examples(raw, column, repeated)}"
            )
    return n


def prepare_input(name):
    """Write ``<name>.csv`` with the ``id`` and ``smiles`` columns lazy-sim reads."""
    lib = LIBRARIES[name]
    raw = fetch_source(name)
    library_csv = BUILD / name / f"{name}.csv"
    if lib["id"] is None:
        n = _write_with_inchikeys(raw, lib["smiles"], library_csv)
    else:
        n = _write_with_ids(raw, lib["smiles"], lib["id"], library_csv)
    assert n == lib["molecules"], f"{name} has {n} molecules, not {lib['molecules']}"
    return library_csv


def describe(*paths):
    return [
        {
            "path": p.relative_to(OUTPUT).as_posix(),
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


def featurizer_entry(out, name):
    """Describe a featurizer of the library in `out` for ``catalog.json``."""
    fm = json.loads((out / f"{name}.json").read_text())
    built_with = {"lazysim": fm["lazysim_version"], fm["engine"]: fm["engine_version"]}
    if "weights_sha256" in fm:
        built_with["weights_sha256"] = fm["weights_sha256"]
    entry = {
        "similarity": fm["similarity"],
        "built_with": built_with,
        "files": describe(out / fm["index_file"], out / f"{name}.json"),
    }
    embeddings = out / artifacts.embeddings_file(name)
    if embeddings.exists():  # only libraries built with --embeddings have them
        entry["embeddings"] = describe(embeddings)
    if name in LICENSES:
        entry["license"] = LICENSES[name]
    return entry


def check_unpublished(name):
    """Raise unless the library has no files yet: published libraries are not edited."""
    out = OUTPUT / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: published libraries are not edited")


def publish(name, built):
    """Move a built library into the S3 layout and record it in catalog.json."""
    lib = LIBRARIES[name]
    out = OUTPUT / name
    check_unpublished(name)
    manifest = json.loads((built / "manifest.json").read_text())
    manifest["source"] = {
        "name": name,
        "version": lib["version"],
        "title": lib["title"],
        "url": lib["source"],
    }
    (built / "manifest.json").write_text(json.dumps(manifest, indent=2))
    out.mkdir(parents=True, exist_ok=True)
    for path in built.iterdir():
        shutil.move(path, out / path.name)
    featurizers = {f: featurizer_entry(out, f) for f in FEATURIZERS}
    catalog = published_catalog()
    catalog["libraries"][name] = {
        "title": lib["title"],
        "version": lib["version"],
        "molecules": manifest["library"]["indexed"],
        "source": lib["source"],
        "license": LICENSE,
        "files": describe(
            *(out / n for n in ("library.db", "skipped.csv", "manifest.json"))
        ),
        "featurizers": featurizers,
    }
    catalog_path = OUTPUT / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote {out} and {catalog_path}")


def record_time(name, featurizer, seconds):
    path = BUILD / name / "timings.json"
    timings = json.loads(path.read_text()) if path.exists() else {}
    timings[featurizer] = round(seconds)
    path.write_text(json.dumps(timings, indent=2))
    print(f"{featurizer} took {seconds / 3600:.2f} h")


def build(name, featurizers, embeddings=True, threads=None):
    """Index a library one featurizer at a time, then publish it once all are built."""
    check_unpublished(name)
    built = BUILD / name / "index"
    if built.exists() and not (built / artifacts.MANIFEST).exists():
        shutil.rmtree(built)  # it stopped while reading molecules; nothing to resume
    for featurizer in (f for f in FEATURIZERS if f in featurizers):
        if featurizer in artifacts.featurizers_in(built):
            print(f"{featurizer} is already built")
            continue
        has_library = (built / artifacts.MANIFEST).exists()
        library_csv = None if has_library else prepare_input(name)
        start = time.perf_counter()
        lazy_sim.index(
            library_csv,
            built,
            featurizers=[featurizer],
            embeddings=embeddings,
            threads=threads,
        )
        record_time(name, featurizer, time.perf_counter() - start)
    missing = [f for f in FEATURIZERS if f not in artifacts.featurizers_in(built)]
    if missing:
        print(f"Still to build: {', '.join(missing)}. Run again to continue.")
    else:
        publish(name, built)


def add(name, names, embeddings=True, threads=None):
    """Index the published library with more featurizers and list them in the catalog."""
    out = OUTPUT / name
    catalog = published_catalog()
    published = remote.read_catalog()[name]
    out.mkdir(parents=True, exist_ok=True)
    for f in published["files"]:
        path = out / Path(f["path"]).name
        if not path.exists():
            remote.download_files([(f["url"], path, f["sha256"], f["bytes"])], name)
        assert sha256(path) == f["sha256"], f"{path} differs from the published file"
    missing = [n for n in names if n not in artifacts.featurizers_in(out)]
    if missing:
        lazy_sim.index(
            None, out, featurizers=missing, embeddings=embeddings, threads=threads
        )
    for n in names:
        catalog["libraries"][name]["featurizers"][n] = featurizer_entry(out, n)
    catalog_path = OUTPUT / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Added {', '.join(names)} to {out}; wrote {catalog_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library", nargs="?", choices=sorted(LIBRARIES))
    parser.add_argument("--list", action="store_true", help="List the libraries.")
    parser.add_argument(
        "-f",
        "--featurizer",
        action="append",
        choices=FEATURIZERS,
        help="Build only this featurizer (repeat for several); the library is "
        "published once all four are built.",
    )
    parser.add_argument(
        "--add",
        nargs="+",
        choices=FEATURIZERS,
        metavar="FEATURIZER",
        help="Add featurizers to the published library instead of building it.",
    )
    parser.add_argument(
        "--embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the full embeddings, for users who fetch them (the default).",
    )
    parser.add_argument("-t", "--threads", type=int)
    args = parser.parse_args()
    if args.list:
        for key, lib in LIBRARIES.items():
            print(f"{key:34s}{lib['molecules']:>12,d}  {lib['title']}")
    elif args.library is None:
        parser.error("name a library, or use --list")
    elif args.add:
        add(args.library, args.add, args.embeddings, args.threads)
    else:
        build(
            args.library, args.featurizer or FEATURIZERS, args.embeddings, args.threads
        )
