"""Build a library (`index`), download a pre-built one (`fetch`), and `search` it."""

import csv
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from rich.progress import MofNCompleteColumn, Progress

from . import __version__
from .engines import get_engine
from .featurizers import get_featurizer
from .io import artifacts, home, remote, tables
from .utils.chem import inchikey, validate
from .utils.logging import console, logger
from .utils.parallel import ChunkPool, set_threads

LIBRARY_CHUNK = 50_000
QUERY_CHUNK = 10_000


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _names(featurizers):
    names = [featurizers] if isinstance(featurizers, str) else list(featurizers)
    for name in names:
        get_featurizer(name)
    return list(dict.fromkeys(names))


def _reason(smiles):
    return "empty SMILES" if not smiles else "invalid SMILES"


def _build_library(input_csv, out, threads):
    """Validate the input, write ``library.db``, ``skipped.csv`` and the manifest."""
    n_rows = sum(1 for _ in tables.iter_rows(input_csv))
    con = artifacts.create_library_db(out)
    n_indexed = seen = 0
    progress = Progress(
        *Progress.get_default_columns(), MofNCompleteColumn(), console=console
    )
    with (
        ChunkPool(validate, threads) as pool,
        open(out / artifacts.SKIPPED, "w", newline="") as fh,
        progress,
    ):
        skipped = csv.writer(fh)
        skipped.writerow(artifacts.SKIPPED_COLUMNS)
        task = progress.add_task("Reading molecules", total=n_rows)
        chunks = tables.iter_chunks(input_csv, LIBRARY_CHUNK)
        for chunk, valid in pool.iter_chunks(chunks):
            rows = []
            for i, ((mol_id, smiles), ok) in enumerate(zip(chunk, valid)):
                if ok:
                    rows.append((n_indexed + len(rows), mol_id, smiles))
                else:
                    skipped.writerow([seen + i + 1, mol_id, smiles, _reason(smiles)])
            con.executemany("INSERT INTO library VALUES (?, ?, ?)", rows)
            n_indexed += len(rows)
            seen += len(chunk)
            progress.update(task, completed=seen)
    con.commit()
    con.close()
    if n_indexed == 0:
        raise ValueError(f"No valid SMILES in {input_csv}")
    manifest = {
        "lazysim_version": __version__,
        "created": _now(),
        "library": {
            "file": Path(input_csv).name,
            "sha256": remote.sha256(input_csv),
            "rows": n_rows,
            "indexed": n_indexed,
            "skipped": n_rows - n_indexed,
        },
    }
    artifacts.write_manifest(out, manifest)
    if n_rows > n_indexed:
        logger.warning(
            "Skipped %d rows; see %s", n_rows - n_indexed, out / artifacts.SKIPPED
        )


def index(input_csv, artifacts_dir, featurizers, embeddings=False, threads=None):
    """Build a searchable library, or add featurizers to an existing one.

    Parameters
    ----------
    input_csv : str or Path or None
        Library CSV with ``smiles`` and ``id`` columns. None adds `featurizers`
        to the existing library at `artifacts_dir`, using its molecules.
    artifacts_dir : str or Path
        Library name (a folder in ``~/lazysim/libraries``) or folder path.
    featurizers : str or list of str
        Featurizers to index with, among ``"clamp"``, ``"ecfp"``, ``"sand"``
        and ``"moljepa"``.
    embeddings : bool
        Also keep each featurizer's full embeddings, rows aligned with the library.
    threads : int, optional
        Worker processes and threads; all logical CPUs by default.

    Returns
    -------
    Path
        The artifacts folder.
    """
    set_threads(threads)
    names = _names(featurizers)
    folder = home.resolve(artifacts_dir)
    start = time.perf_counter()
    workers = threads or os.cpu_count()
    if input_csv is None:
        artifacts.read_manifest(folder)
        present = set(artifacts.featurizers_in(folder)) & set(names)
        if present:
            raise ValueError(f"{folder} already has {', '.join(sorted(present))}")
    else:
        tables.check_columns(input_csv)
        folder = artifacts.prepare_dir(folder)
        logger.info("Reading %s", input_csv)
        _build_library(input_csv, folder, workers)
    for name in names:
        feat = get_featurizer(name)
        details = get_engine(feat.ENGINE).build(feat, folder, workers, embeddings)
        manifest = {
            "featurizer": feat.NAME,
            "similarity": feat.SIMILARITY,
            **details,
            "lazysim_version": __version__,
            "created": _now(),
        }
        artifacts.write_featurizer(folder, feat.NAME, manifest)
    logger.success(
        "Indexed with %s in %.0f s → %s",
        ", ".join(names),
        time.perf_counter() - start,
        folder,
    )
    return folder


def fetch(name, featurizers, artifacts_dir=None, embeddings=False):
    """Download a library that Ersilia has already indexed.

    Parameters
    ----------
    name : str
        Library name, as listed by ``lazysim catalog --remote``.
    featurizers : str or list of str
        Featurizers to download, among those the library was indexed with.
    artifacts_dir : str or Path, optional
        Destination name or path; ``~/lazysim/libraries/<name>`` by default.
        Fetching into a copy of the same library adds the missing featurizers.
    embeddings : bool
        Also download each featurizer's full embeddings.

    Returns
    -------
    Path
        The artifacts folder.
    """
    names = _names(featurizers)
    catalog = remote.read_catalog()
    if name not in catalog:
        raise ValueError(
            f"No library {name!r} to fetch; available: {', '.join(sorted(catalog))}"
        )
    entry = catalog[name]
    missing = [n for n in names if n not in entry["featurizers"]]
    if missing:
        raise ValueError(f"{name} has no {', '.join(missing)} index to fetch")
    for n in names:
        built = entry["featurizers"][n].get("built_with", {}).get("weights_sha256")
        if built and built != get_featurizer(n).WEIGHTS_SHA256:
            raise ValueError(f"{name} ({n}) needs a different lazy-sim version")
    folder = home.resolve(artifacts_dir or name)
    items, present = [], []
    if (folder / artifacts.MANIFEST).exists():
        local = remote.sha256(folder / artifacts.MANIFEST)
        published = next(
            f for f in entry["files"] if f["path"].endswith("/manifest.json")
        )
        if local != published["sha256"]:
            raise FileExistsError(f"{folder} holds a different library")
        present = artifacts.featurizers_in(folder)
    else:
        folder = artifacts.prepare_dir(folder)
        # The library manifest goes last: a folder without one is unfinished.
        items += sorted(
            entry["files"], key=lambda f: f["path"].endswith("manifest.json")
        )
    for n in names:
        published = entry["featurizers"][n]
        # An index already here is kept; only its missing embeddings are added.
        files = [] if n in present else list(published["files"])
        if embeddings:
            files += [
                f
                for f in published.get("embeddings", [])
                if not (folder / Path(f["path"]).name).exists()
            ]
        # Each featurizer's manifest goes last, marking its index as complete.
        items += sorted(files, key=lambda f: f["path"].endswith(f"/{n}.json"))
    downloads = [
        (f["url"], folder / Path(f["path"]).name, f["sha256"], f["bytes"])
        for f in items
    ]
    if downloads:
        try:
            remote.download_files(downloads, f"Fetching {name}")
        except BaseException:
            if present:  # partial downloads never land; the library is untouched
                logger.error(
                    "Fetch did not finish; %s is unchanged, run it again", name
                )
            else:
                logger.error("Fetch did not finish; delete %s and try again", folder)
            raise
    for n in names:
        feat = get_featurizer(n)
        if hasattr(feat, "ensure_weights"):
            feat.ensure_weights()
    if downloads:
        logger.success("Fetched %s → %s", name, folder)
    else:
        logger.success("%s is already fetched → %s", name, folder)
    return folder


def _check_index(folder, featurizer):
    available = artifacts.featurizers_in(folder)
    if not available:
        raise ValueError(f"{folder} has no finished index")
    if featurizer not in available:
        raise ValueError(
            f"{folder} has no {featurizer} index (it has {', '.join(available)}); "
            f"add it with 'lazysim fetch' or 'lazysim index -a {folder} -f {featurizer}'"
        )


def _check_versions(manifest):
    built = manifest["lazysim_version"]
    if built != __version__:
        logger.warning(
            "Index built with lazy-sim %s; installed is %s", built, __version__
        )
    engine = manifest["engine"]
    package = {"turbovec": "turbovec", "fpsim2": "FPSim2"}[engine]
    try:
        installed = version(package)
    except Exception:
        return
    if manifest["engine_version"] != installed:
        logger.warning(
            "Index built with %s %s; installed is %s",
            package,
            manifest["engine_version"],
            installed,
        )


def search(input_csv, artifacts_dir, output_csv, featurizer, k=100, threads=None):
    """Find the `k` most similar library molecules for each query.

    Parameters
    ----------
    input_csv : str or Path
        Query CSV with a ``smiles`` column and, optionally, ``id``. Without
        ``id``, each query is named by its InChIKey (or ``row-<n>``).
    artifacts_dir : str or Path
        Library name (in ``~/lazysim/libraries``) or artifacts folder path.
    output_csv : str or Path
        Output with one row per hit: ``query_id, query_smiles, rank, hit_id,
        hit_smiles, similarity``.
    featurizer : str
        Index to search: ``"ecfp"`` (exact Tanimoto of ECFP fingerprints), or
        ``"clamp"``, ``"sand"`` or ``"moljepa"`` (cosine of their embeddings).
    k : int
        Hits per query.
    threads : int, optional
        Worker processes and threads; all logical CPUs by default.

    Returns
    -------
    int
        Number of queries searched; invalid queries are skipped with a warning.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    set_threads(threads)
    folder = home.resolve(artifacts_dir, must_exist=True)
    feat = get_featurizer(featurizer)
    _check_index(folder, featurizer)
    manifest = artifacts.read_featurizer(folder, featurizer)
    _check_versions(manifest)
    tables.check_columns(input_csv, require_id=False)
    start = time.perf_counter()
    n_library = artifacts.read_manifest(folder)["library"]["indexed"]
    k = min(k, n_library)
    n_queries = sum(1 for _ in tables.iter_rows(input_csv, require_id=False))
    workers = threads or os.cpu_count()
    con = artifacts.open_library_db(folder)
    searched, seen, invalid = 0, 0, []
    engine = get_engine(feat.ENGINE)
    with (
        engine.Searcher(feat, folder, manifest, workers, n_queries) as searcher,
        open(output_csv, "w", newline="") as fh,
    ):
        writer = tables.hits_writer(fh)
        chunks = tables.iter_chunks(input_csv, QUERY_CHUNK, require_id=False)
        for chunk in chunks:
            ok = validate([smiles for _, smiles in chunk])
            queries = []
            for n, ((query_id, smiles), good) in enumerate(zip(chunk, ok), seen + 1):
                if query_id is None:  # no id column: name queries by InChIKey
                    query_id = (good and inchikey(smiles)) or f"row-{n}"
                if good:
                    queries.append((query_id, smiles))
                else:
                    invalid.append(query_id)
            seen += len(chunk)
            if not queries:
                continue
            results = searcher.search([smiles for _, smiles in queries], k)
            hits = artifacts.lookup(con, {int(r) for _, rows in results for r in rows})
            for (query_id, query_smiles), (scores, rows) in zip(queries, results):
                for rank, (row, score) in enumerate(zip(rows, scores), 1):
                    hit_id, hit_smiles = hits[int(row)]
                    writer.writerow(
                        [
                            query_id,
                            query_smiles,
                            rank,
                            hit_id,
                            hit_smiles,
                            f"{score:.4f}",
                        ]
                    )
            searched += len(queries)
    con.close()
    if invalid:
        shown = ", ".join(invalid[:5]) + (", ..." if len(invalid) > 5 else "")
        logger.warning("Skipped %d invalid queries: %s", len(invalid), shown)
    logger.success(
        "Searched %d queries against %d molecules (%s, %s) in %.1f s → %s",
        searched,
        n_library,
        featurizer,
        feat.SIMILARITY,
        time.perf_counter() - start,
        output_csv,
    )
    return searched
