"""Build a library (`index`), download a pre-built one (`fetch`), and `search` it."""

import csv
import os
import random
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import turbovec
from rich.progress import MofNCompleteColumn, Progress

from . import __version__
from .featurizers import get_featurizer
from .io import artifacts, home, remote, tables
from .utils.logging import console, logger
from .utils.parallel import FingerprintPool, set_threads

INDEX_CHUNK = 50_000
QUERY_CHUNK = 10_000
CALIBRATION_SIZE = 2_048
BITS = 4
POOL_MIN_QUERIES = 5_000


def _scan(path):
    """Count rows and draw a seeded uniform sample of SMILES (reservoir sampling)."""
    rng = random.Random(0)
    sample, n = [], 0
    for n, (_, smiles) in enumerate(tables.iter_rows(path), 1):
        if n <= CALIBRATION_SIZE:
            sample.append(smiles)
        else:
            j = rng.randrange(n)
            if j < CALIBRATION_SIZE:
                sample[j] = smiles
    return n, sample


def _reason(smiles):
    return "empty SMILES" if not smiles else "invalid SMILES"


def index(input_csv, artifacts_dir, featurizer="clamp", embeddings=False, threads=None):
    """Embed every molecule of a library and build a searchable index.

    The index stores 4 bits per dimension (recall@100 about 0.96 on ChEMBL).

    Parameters
    ----------
    input_csv : str or Path
        Library CSV with ``smiles`` and ``id`` columns.
    artifacts_dir : str or Path
        Library name (a folder in ``~/lazysim/libraries``) or a path to a new or
        empty folder.
    featurizer : str
        Featurizer name; only ``"clamp"`` for now.
    embeddings : bool
        Also keep the full float32 embeddings, rows aligned with the index.
    threads : int, optional
        Worker processes and threads; all logical CPUs by default.

    Returns
    -------
    Path
        The artifacts folder.
    """
    set_threads(threads)
    feat = get_featurizer(featurizer)
    tables.check_columns(input_csv)
    out = artifacts.prepare_dir(home.resolve(artifacts_dir))
    start = time.perf_counter()
    logger.info("Scanning %s", input_csv)
    n_rows, sample = _scan(input_csv)
    weights = feat.load_weights()
    idx = turbovec.TurboQuantIndex(feat.DIM, BITS)
    emb_path = out / artifacts.embeddings_file(feat.NAME)
    stored = (
        artifacts.open_embeddings(emb_path, n_rows, feat.DIM) if embeddings else None
    )
    n_indexed = n_skipped = 0
    with FingerprintPool(feat.fingerprints, threads or os.cpu_count()) as pool:
        calibration = [fp for fp in pool.map(sample) if fp is not None]
        if not calibration:
            raise ValueError(f"No valid SMILES in {input_csv}")
        idx.calibrate(feat.encode(weights, calibration))
        con = artifacts.create_library_db(out)
        progress = Progress(
            *Progress.get_default_columns(), MofNCompleteColumn(), console=console
        )
        with open(out / artifacts.SKIPPED, "w", newline="") as fh, progress:
            skipped = csv.writer(fh)
            skipped.writerow(artifacts.SKIPPED_COLUMNS)
            task = progress.add_task(f"Embedding with {feat.NAME}", total=n_rows)
            seen = 0
            chunks = tables.iter_chunks(input_csv, INDEX_CHUNK)
            for chunk, fps in pool.iter_chunks(chunks):
                keep = []
                for i, ((mol_id, smiles), fp) in enumerate(zip(chunk, fps)):
                    if fp is None:
                        skipped.writerow(
                            [seen + i + 1, mol_id, smiles, _reason(smiles)]
                        )
                    else:
                        keep.append(i)
                if keep:
                    vectors = feat.encode(weights, [fps[i] for i in keep])
                    idx.add(vectors)
                    if stored is not None:
                        stored[n_indexed : n_indexed + len(keep)] = vectors
                    con.executemany(
                        "INSERT INTO library VALUES (?, ?, ?)",
                        ((n_indexed + j, *chunk[i]) for j, i in enumerate(keep)),
                    )
                n_indexed += len(keep)
                n_skipped += len(chunk) - len(keep)
                seen += len(chunk)
                progress.update(task, completed=seen)
    if n_indexed == 0:
        raise ValueError(f"No valid SMILES in {input_csv}")
    if stored is not None:
        stored.flush()
        del stored
        if n_indexed < n_rows:
            artifacts.shrink_embeddings(emb_path, n_indexed, feat.DIM)
    index_name = artifacts.index_file(feat.NAME)
    idx.write(str(out / index_name))
    con.commit()
    con.close()
    manifest = {
        "lazysim_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "featurizer": {
            "name": feat.NAME,
            "dim": feat.DIM,
            "weights_file": feat.WEIGHTS_FILE,
            "weights_sha256": feat.WEIGHTS_SHA256,
        },
        "backend": {
            "name": "turbovec",
            "version": version("turbovec"),
            "bits": BITS,
            "file": index_name,
        },
        "library": {
            "file": Path(input_csv).name,
            "sha256": remote.sha256(input_csv),
            "rows": n_rows,
            "indexed": n_indexed,
            "skipped": n_skipped,
        },
    }
    artifacts.write_manifest(out, manifest)
    if n_skipped:
        logger.warning("Skipped %d rows; see %s", n_skipped, out / artifacts.SKIPPED)
    logger.success(
        "Indexed %d molecules in %.0f s → %s",
        n_indexed,
        time.perf_counter() - start,
        out,
    )
    return out


def fetch(name, artifacts_dir=None, embeddings=False):
    """Download a library that Ersilia has already indexed.

    Parameters
    ----------
    name : str
        Library name, as listed by ``lazysim catalog --remote``.
    artifacts_dir : str or Path, optional
        Destination name or path; ``~/lazysim/libraries/<name>`` by default.
    embeddings : bool
        Also download the full float32 embeddings.

    Returns
    -------
    Path
        The artifacts folder.
    """
    catalog = remote.read_catalog()
    if name not in catalog:
        raise ValueError(
            f"No library {name!r} to fetch; available: {', '.join(sorted(catalog))}"
        )
    entry = catalog[name]
    feat = get_featurizer(entry["built_with"]["featurizer"])
    if entry["built_with"]["weights_sha256"] != feat.WEIGHTS_SHA256:
        raise ValueError(f"{name} needs a different lazy-sim version")
    out = artifacts.prepare_dir(home.resolve(artifacts_dir or name))
    groups = ["library"] + (["embeddings"] if embeddings else [])
    files = [f for group in groups for f in entry["files"][group]]
    # The manifest goes last: a folder without one is an unfinished fetch.
    files.sort(key=lambda f: Path(f["path"]).name == artifacts.MANIFEST)
    items = [
        (f["url"], out / Path(f["path"]).name, f["sha256"], f["bytes"]) for f in files
    ]
    try:
        remote.download_files(items, f"Fetching {name}")
    except BaseException:
        logger.error("Fetch did not finish; delete %s and try again", out)
        raise
    feat.ensure_weights()
    logger.success("Fetched %s → %s", name, out)
    return out


def _check_versions(manifest):
    built = manifest["lazysim_version"], manifest["backend"]["version"]
    installed = __version__, version("turbovec")
    for name, was, now in zip(("lazy-sim", "turbovec"), built, installed):
        if was != now:
            logger.warning("Index built with %s %s; installed is %s", name, was, now)


def search(input_csv, artifacts_dir, output_csv, k=100, threads=None):
    """Find the `k` most similar library molecules for each query.

    Parameters
    ----------
    input_csv : str or Path
        Query CSV with ``smiles`` and ``id`` columns.
    artifacts_dir : str or Path
        Library name (in ``~/lazysim/libraries``) or artifacts folder path.
    output_csv : str or Path
        Output with one row per hit: ``query_id, query_smiles, rank, hit_id,
        hit_smiles, similarity`` (cosine similarity of the embeddings).
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
    manifest = artifacts.read_manifest(folder)
    feat = get_featurizer(manifest["featurizer"]["name"])
    if manifest["featurizer"]["weights_sha256"] != feat.WEIGHTS_SHA256:
        raise ValueError(
            f"{folder} was built with different {feat.NAME} weights; re-index it"
        )
    _check_versions(manifest)
    tables.check_columns(input_csv)
    start = time.perf_counter()
    weights = feat.load_weights()
    idx = turbovec.TurboQuantIndex.load(str(folder / manifest["backend"]["file"]))
    k = min(k, len(idx))
    con = artifacts.open_library_db(folder)
    n_queries = sum(1 for _ in tables.iter_rows(input_csv))
    processes = (threads or os.cpu_count()) if n_queries >= POOL_MIN_QUERIES else 1
    searched, invalid = 0, []
    with (
        FingerprintPool(feat.fingerprints, processes) as pool,
        open(output_csv, "w", newline="") as fh,
    ):
        writer = tables.hits_writer(fh)
        for chunk, fps in pool.iter_chunks(tables.iter_chunks(input_csv, QUERY_CHUNK)):
            keep = [i for i, fp in enumerate(fps) if fp is not None]
            invalid += [chunk[i][0] for i, fp in enumerate(fps) if fp is None]
            if not keep:
                continue
            scores, rows = idx.search(feat.encode(weights, [fps[i] for i in keep]), k)
            hits = artifacts.lookup(con, np.unique(rows))
            for q, i in enumerate(keep):
                query_id, query_smiles = chunk[i]
                for rank, (row, score) in enumerate(zip(rows[q], scores[q]), 1):
                    hit_id, hit_smiles = hits[int(row)]
                    similarity = f"{min(float(score), 1.0):.4f}"
                    writer.writerow(
                        [query_id, query_smiles, rank, hit_id, hit_smiles, similarity]
                    )
            searched += len(keep)
    con.close()
    if invalid:
        shown = ", ".join(invalid[:5]) + (", ..." if len(invalid) > 5 else "")
        logger.warning("Skipped %d invalid queries: %s", len(invalid), shown)
    logger.success(
        "Searched %d queries against %d molecules in %.1f s → %s",
        searched,
        len(idx),
        time.perf_counter() - start,
        output_csv,
    )
    return searched
