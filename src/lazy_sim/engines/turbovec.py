"""Dense embeddings compared by cosine similarity, in a 4-bit turbovec index."""

from importlib.metadata import version

import numpy as np
import turbovec
from rich.progress import MofNCompleteColumn, Progress

from ..io import artifacts
from ..utils.logging import console
from ..utils.parallel import ChunkPool

BITS = 4
CHUNK = 50_000
CALIBRATION_SIZE = 2_048
POOL_MIN_QUERIES = 5_000


def _embed(feat, weights, fps):
    if any(fp is None for fp in fps):
        raise RuntimeError(f"A valid molecule has no {feat.NAME} fingerprint")
    return feat.encode(weights, fps)


def build(feat, folder, threads, embeddings):
    """Embed every molecule of ``library.db`` and write the turbovec index.

    Parameters
    ----------
    feat : module
        A dense featurizer, with ``fingerprints``, ``load_weights`` and ``encode``.
    folder : Path
        Artifacts folder.
    threads : int
        Worker processes.
    embeddings : bool
        Also keep the full float32 embeddings.

    Returns
    -------
    dict
        Engine details for the featurizer manifest.
    """
    weights = feat.load_weights()
    con = artifacts.open_library_db(folder)
    n = artifacts.count_rows(con)
    idx = turbovec.TurboQuantIndex(feat.DIM, BITS)
    emb_path = folder / artifacts.embeddings_file(feat.NAME)
    stored = (
        artifacts.open_embeddings(emb_path, n, feat.DIM, "float32")
        if embeddings
        else None
    )
    progress = Progress(
        *Progress.get_default_columns(), MofNCompleteColumn(), console=console
    )
    with ChunkPool(feat.fingerprints, threads) as pool, progress:
        sample = artifacts.sample_smiles(con, CALIBRATION_SIZE)
        idx.calibrate(_embed(feat, weights, pool.map(sample)))
        task = progress.add_task(f"Embedding with {feat.NAME}", total=n)
        done = 0
        for _, fps in pool.iter_chunks(artifacts.iter_library(con, CHUNK)):
            vectors = _embed(feat, weights, fps)
            idx.add(vectors)
            if stored is not None:
                stored[done : done + len(vectors)] = vectors
            done += len(vectors)
            progress.update(task, completed=done)
    con.close()
    if stored is not None:
        stored.flush()
        del stored
    index_file = f"{feat.NAME}.tv"
    idx.write(str(folder / index_file))
    return {
        "engine": "turbovec",
        "engine_version": version("turbovec"),
        "bits": BITS,
        "index_file": index_file,
        "weights_sha256": feat.WEIGHTS_SHA256,
    }


class Searcher:
    """Embed queries and find their nearest library rows in a turbovec index.

    Parameters
    ----------
    feat : module
        The featurizer the index was built with.
    folder : Path
        Artifacts folder.
    manifest : dict
        The featurizer manifest.
    threads : int
        Worker processes, used only for large query sets.
    n_queries : int
        Number of queries to come.
    """

    def __init__(self, feat, folder, manifest, threads, n_queries):
        if manifest["weights_sha256"] != feat.WEIGHTS_SHA256:
            raise ValueError(
                f"{folder} was built with different {feat.NAME} weights; re-index it"
            )
        self.feat = feat
        self.weights = feat.load_weights()
        self.idx = turbovec.TurboQuantIndex.load(str(folder / manifest["index_file"]))
        processes = threads if n_queries >= POOL_MIN_QUERIES else 1
        self.pool = ChunkPool(feat.fingerprints, processes)

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, *exc):
        self.pool.__exit__(*exc)

    def search(self, smiles, k):
        """Return one ``(scores, rows)`` pair per query, best first."""
        vectors = _embed(self.feat, self.weights, self.pool.map(smiles))
        scores, rows = self.idx.search(vectors, k)
        return list(zip(np.minimum(scores, 1.0), rows))
