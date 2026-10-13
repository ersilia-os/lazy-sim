"""Featurizers that turn SMILES into L2-normalised embeddings."""

import importlib

AVAILABLE = ("clamp",)


def get_featurizer(name):
    """Return the featurizer module registered under `name`.

    Parameters
    ----------
    name : str
        One of `AVAILABLE`.

    Returns
    -------
    module
        Exposes ``NAME``, ``DIM``, ``WEIGHTS_FILE``, ``WEIGHTS_SHA256``,
        ``fingerprints``, ``ensure_weights``, ``load_weights`` and ``encode``.
    """
    if name not in AVAILABLE:
        raise ValueError(
            f"Unknown featurizer {name!r}; available: {', '.join(AVAILABLE)}"
        )
    return importlib.import_module(f"{__name__}.{name}")
