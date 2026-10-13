"""Search engines: how a featurizer's index is built and queried.

Each engine module exposes ``build(feat, folder, threads, embeddings)``, which
writes the index from ``library.db`` and returns the details to record in the
featurizer manifest, and ``Searcher(feat, folder, manifest, threads, n_queries)``,
a context manager whose ``search(smiles, k)`` returns one ``(scores, rows)`` pair
per query, best first.
"""

import importlib


def get_engine(name):
    """Return the engine module registered under `name`."""
    return importlib.import_module(f"{__name__}.{name}")
