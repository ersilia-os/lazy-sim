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
   as the published one plus the libraries built here and not uploaded yet (so nobody
   else's entries are lost, and building one library never drops another).

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

To build the featurizers at the same time, for example as separate cluster jobs,
build the first one alone, since it creates the library; then run the others with
``--no-publish``, which never publishes; then run the plain command to publish:
python scripts/build_library.py enamine_lsc_2M_v0 -f ecfp
python scripts/build_library.py enamine_lsc_2M_v0 -f clamp --no-publish
python scripts/build_library.py enamine_lsc_2M_v0 -f moljepa --no-publish
python scripts/build_library.py enamine_lsc_2M_v0 -f sand --no-publish
python scripts/build_library.py enamine_lsc_2M_v0

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

import numpy as np
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


def _maintained(folder, title, molecules, sha256=None, duplicates=0):
    """Describe a library of ``data/processed`` in the Ersilia Maintained Inputs.

    `molecules` is the number of rows of the source. `duplicates` is the number of rows
    to drop because they repeat the SMILES of an earlier row (0 stops the build at any).
    """
    return {
        "title": title,
        "version": "v0",
        "molecules": molecules,
        "source": f"{_MAINTAINED}/{folder}/{folder}_smiles_ids.csv",
        "sha256": sha256,  # None until the first build prints it
        "smiles": "smiles",
        "id": "collection_id",
        "duplicates": duplicates,
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
    # 4 of the 715,822 rows are not valid SMILES; the catalog counts the 715,818 indexed
    "coconut_715K_v0": _maintained(
        "coconut_715k_v0",
        "COCONUT",
        715_822,
        "15f7d7e153d4d1802eac6ee3e2694e78e5b53dac360cb14e19eb51a2c3a19644",
    ),
    "enamine_lsc_2M_v0": _maintained(
        "enamine_liquid_stock_2m_v0",
        "Enamine Liquid Stock Collection",
        2_492_445,
        "cf0e23935e38f2ebe6c7869a6e5664b825f1ceb6fbd8e3cda3b602f371da0777",
    ),
    # 9,087 rows repeat an earlier row (same id, same SMILES) and 210 SMILES carry two
    # ids; the first row of each SMILES is kept, so 5,313,566 molecules are indexed
    "molport_5M_v0": _maintained(
        "molport_screening_compounds_5m_v0",
        "Molport Screening Compounds",
        5_322_863,
        "a6ec06fbfa4d73e32aaf448a4f89d97bc07382fb32ab7b634e03b232675a710c",
        duplicates=9_297,
    ),
    "enamine_real_sample_10M_v0": _maintained(
        "enamine_real_sample_10m_v0",
        "Enamine REAL Sample",
        10_353_526,
        "8f6031f8e4171c45b41d73aca2d6c63fb35dff2a83ffd9e86ae81fdba28dc895",
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


def _repeats(ids, smiles):
    """Return the rows that repeat the SMILES of an earlier row, and that earlier row.

    `ids` and `smiles` are the 64-bit hashes of each row's values. Keeping the first
    row of each SMILES leaves every id once, unless an id carries two different
    SMILES; that is a conflict in the source, not a repeat, and raises.
    """
    ids, smiles = np.frombuffer(ids, np.int64), np.frombuffer(smiles, np.int64)
    pairs = ids * np.int64(1_000_003) + smiles  # wraps around on purpose
    if len(np.unique(pairs)) != len(np.unique(ids)):
        raise ValueError("an id carries two different SMILES")
    _, first, inverse = np.unique(smiles, return_index=True, return_inverse=True)
    earlier = first[inverse.reshape(-1)]
    rows = np.flatnonzero(earlier != np.arange(len(smiles)))
    return rows, earlier[rows]


def _rows_at(raw, wanted, smiles_column, id_column):
    """Return ``{row: (id, smiles)}`` for the (0-based) rows in `wanted`."""
    found = {}
    with open(raw, newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh)):
            if n in wanted:
                found[n] = (row[id_column] or f"row-{n + 1}", row[smiles_column])
    return found


def _write_with_ids(raw, smiles_column, id_column, library_csv, duplicates=0):
    """Write the ``id`` and ``smiles`` columns, without the repeats of an earlier SMILES.

    The rows dropped are listed in ``duplicates.csv`` next to the library. There must
    be exactly `duplicates` of them, so a library that should have none stops the
    build at its first. Returns the number of rows of `raw`.
    """
    # Streamed, because the largest library has more than ten million rows. Repeats
    # are looked for by 64-bit hash, which keeps the memory use low.
    ids, smiles = array("q"), array("q")
    blank = 0
    with open(raw, newline="") as src:
        for n, row in enumerate(csv.DictReader(src), start=1):
            blank += not row[id_column]
            ids.append(hash(row[id_column] or f"row-{n}"))
            smiles.append(hash(row[smiles_column]))
    print(f"{len(ids)} molecules, {blank} without an id")
    dropped, kept = _repeats(ids, smiles)
    dropped, kept = dropped.tolist(), kept.tolist()
    skip = set(dropped)
    found = _rows_at(raw, skip | set(kept), smiles_column, id_column) if skip else {}
    if len(skip) != duplicates:
        some = [(r + 1, *found[r]) for r in dropped[:3]]
        raise ValueError(
            f"{raw} has {len(skip)} rows that repeat an earlier SMILES, not "
            f"{duplicates}" + (f", such as {some}" if some else "")
        )
    if skip:
        log = library_csv.with_name("duplicates.csv")
        with open(log, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["row", "id", "smiles", "kept_row", "kept_id"])
            for r, k in zip(dropped, kept):
                writer.writerow([r + 1, *found[r], k + 1, found[k][0]])
        same_id = sum(found[r][0] == found[k][0] for r, k in zip(dropped, kept))
        print(
            f"{len(skip)} rows dropped for repeating an earlier SMILES "
            f"({same_id} with the same id), listed in {log}"
        )
    with open(raw, newline="") as src, open(library_csv, "w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(["id", "smiles"])
        for n, row in enumerate(csv.DictReader(src)):
            if n not in skip:
                writer.writerow([row[id_column] or f"row-{n + 1}", row[smiles_column]])
    return len(ids)


def prepare_input(name):
    """Write ``<name>.csv`` with the ``id`` and ``smiles`` columns lazy-sim reads."""
    lib = LIBRARIES[name]
    raw = fetch_source(name)
    library_csv = BUILD / name / f"{name}.csv"
    if lib["id"] is None:
        n = _write_with_inchikeys(raw, lib["smiles"], library_csv)
    else:
        n = _write_with_ids(
            raw, lib["smiles"], lib["id"], library_csv, lib["duplicates"]
        )
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


def current_catalog():
    """Return the catalog.json on S3 plus the libraries built here but not uploaded.

    Starting from S3 alone would drop a library that is built but not uploaded yet as
    soon as the next one is published. S3 wins for what it lists; from the local
    catalog come the libraries it lacks and the featurizers it lacks.
    """
    catalog = published_catalog()
    local = OUTPUT / "catalog.json"
    if local.exists():
        for name, entry in json.loads(local.read_text())["libraries"].items():
            if not (OUTPUT / name).is_dir():
                continue  # the library was deleted here, so its entry is stale
            kept = catalog["libraries"].setdefault(name, entry)
            for f, featurizer in entry["featurizers"].items():
                kept["featurizers"].setdefault(f, featurizer)
    return catalog


def write_catalog(catalog):
    path = OUTPUT / "catalog.json"
    path.write_text(json.dumps(catalog, indent=2))
    return path


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


def library_entry(name, out):
    """Describe the library in `out`, with the featurizers in it, for ``catalog.json``."""
    lib = LIBRARIES[name]
    manifest = json.loads((out / artifacts.MANIFEST).read_text())
    built = artifacts.featurizers_in(out)
    return {
        "title": lib["title"],
        "version": lib["version"],
        "molecules": manifest["library"]["indexed"],
        "source": lib["source"],
        "license": LICENSE,
        "files": describe(
            *(out / n for n in ("library.db", "skipped.csv", "manifest.json"))
        ),
        "featurizers": {f: featurizer_entry(out, f) for f in FEATURIZERS if f in built},
    }


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
    if lib.get("duplicates"):  # the build checked that this many rows were dropped
        manifest["source"]["duplicates_dropped"] = lib["duplicates"]
    (built / "manifest.json").write_text(json.dumps(manifest, indent=2))
    out.mkdir(parents=True, exist_ok=True)
    for path in built.iterdir():
        shutil.move(path, out / path.name)
    catalog = current_catalog()
    catalog["libraries"][name] = library_entry(name, out)
    print(f"Wrote {out} and {write_catalog(catalog)}")


def record_time(name, featurizer, seconds):
    path = BUILD / name / "timings.json"
    timings = json.loads(path.read_text()) if path.exists() else {}
    timings[featurizer] = round(seconds)
    path.write_text(json.dumps(timings, indent=2))
    print(f"{featurizer} took {seconds / 3600:.2f} h")


def build(name, featurizers, embeddings=True, threads=None, publish_when_built=True):
    """Index a library one featurizer at a time, then publish it once all are built.

    With `publish_when_built` False, the library is never published here, so several
    runs can build different featurizers of it at the same time; a final run with the
    default publishes it.
    """
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
    elif publish_when_built:
        publish(name, built)
    else:
        print("All featurizers are built. Run again without --no-publish to publish.")


def add(name, names, embeddings=True, threads=None):
    """Index the published library with more featurizers and list them in the catalog."""
    out = OUTPUT / name
    catalog = current_catalog()
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
    print(f"Added {', '.join(names)} to {out}; wrote {write_catalog(catalog)}")


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
    parser.add_argument(
        "--publish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish the library once all four featurizers are built (the default). "
        "--no-publish only builds, so featurizers can be built in parallel.",
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
            args.library,
            args.featurizer or FEATURIZERS,
            args.embeddings,
            args.threads,
            args.publish,
        )
