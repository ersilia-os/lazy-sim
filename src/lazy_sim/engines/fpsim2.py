"""Bit fingerprints compared by exact Tanimoto similarity, with FPSim2.

FPSim2 is an optional dependency: ``pip install lazy-sim[ecfp]``.
"""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

from rich.progress import MofNCompleteColumn, Progress

from ..io import artifacts
from ..utils.logging import console, logger
from ..utils.parallel import ChunkPool

CHUNK = 50_000


def _require():
    try:
        from FPSim2 import FPSim2Engine
        from FPSim2.io import create_db_file
    except ImportError:
        raise ValueError(
            "This featurizer needs FPSim2: pip install 'lazy-sim[ecfp]'"
        ) from None
    return FPSim2Engine, create_db_file


def _progress():
    return Progress(
        *Progress.get_default_columns(), MofNCompleteColumn(), console=console
    )


def _write_embeddings(feat, con, path, n, threads):
    stored = artifacts.open_embeddings(path, n, feat.DIM // 8, "uint8")
    done = 0
    with ChunkPool(feat.packed_fingerprints, threads) as pool, _progress() as bar:
        task = bar.add_task(f"Storing {feat.NAME} bits", total=n)
        for _, packed in pool.iter_chunks(artifacts.iter_library(con, CHUNK)):
            stored[done : done + len(packed)] = packed
            done += len(packed)
            bar.update(task, completed=done)
    stored.flush()


def build(feat, folder, threads, embeddings):
    """Fingerprint every molecule of ``library.db`` into an FPSim2 file.

    Parameters
    ----------
    feat : module
        A bit-fingerprint featurizer, with ``FP_TYPE`` and ``FP_PARAMS``.
    folder : Path
        Artifacts folder.
    threads : int
        Worker processes, used to store the embeddings.
    embeddings : bool
        Also keep the fingerprints as packed bits, rows aligned with the library.

    Returns
    -------
    dict
        Engine details for the featurizer manifest.
    """
    import tables

    _, create_db_file = _require()
    con = artifacts.open_library_db(folder)
    n = artifacts.count_rows(con)
    index_file = f"{feat.NAME}.h5"
    with _progress() as bar:
        task = bar.add_task(f"Fingerprinting with {feat.NAME}", total=n)

        def pairs():
            for chunk in artifacts.iter_library(con, CHUNK):
                for row, smiles in chunk:
                    yield smiles, row
                bar.advance(task, len(chunk))

        create_db_file(
            pairs(),
            str(folder / index_file),
            "smiles",
            feat.FP_TYPE,
            dict(feat.FP_PARAMS),
        )
    with tables.open_file(folder / index_file) as h5:
        stored = h5.root.fps.nrows
    if stored != n:
        logger.warning("FPSim2 stored %d of %d molecules", stored, n)
    if embeddings:
        path = folder / artifacts.embeddings_file(feat.NAME)
        _write_embeddings(feat, con, path, n, threads)
    con.close()
    return {
        "engine": "fpsim2",
        "engine_version": version("FPSim2"),
        "fp_type": feat.FP_TYPE,
        "fp_params": feat.FP_PARAMS,
        "index_file": index_file,
    }


class Searcher:
    """Find the library rows with the highest Tanimoto similarity to each query.

    Parameters
    ----------
    feat : module
        The featurizer the index was built with.
    folder : Path
        Artifacts folder.
    manifest : dict
        The featurizer manifest.
    threads : int
        Queries searched at the same time, one thread each (about 3x faster
        than splitting each query across threads).
    n_queries : int
        Number of queries to come (unused).
    """

    def __init__(self, feat, folder, manifest, threads, n_queries):
        engine, _ = _require()
        self.engine = engine(str(folder / manifest["index_file"]))
        self.threads = threads

    def __enter__(self):
        self.executor = ThreadPoolExecutor(self.threads)
        return self

    def __exit__(self, *exc):
        self.executor.shutdown()

    def _top_k(self, query, k):
        hits = self.engine.top_k(query, k=k, threshold=0.0, n_workers=1)
        return hits["coeff"], hits["mol_id"]

    def search(self, smiles, k):
        """Return one ``(scores, rows)`` pair per query, best first."""
        return list(self.executor.map(lambda query: self._top_k(query, k), smiles))
