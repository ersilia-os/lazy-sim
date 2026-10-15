"""Featurizers: how molecules are represented, and which engine searches them.

Each featurizer module exposes ``NAME``, ``ENGINE`` (a module in
`lazy_sim.engines`) and ``SIMILARITY`` (what the scores mean), plus whatever its
engine needs to build an index and embed queries.
"""

import importlib

AVAILABLE = ("clamp", "ecfp", "sand", "moljepa", "monroe")


def get_featurizer(name):
    """Return the featurizer module registered under `name`.

    Parameters
    ----------
    name : str
        One of `AVAILABLE`.

    Returns
    -------
    module
    """
    if name not in AVAILABLE:
        raise ValueError(
            f"Unknown featurizer {name!r}; available: {', '.join(AVAILABLE)}"
        )
    return importlib.import_module(f"{__name__}.{name}")
