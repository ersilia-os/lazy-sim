"""Comparisons of featurizers on Ersilia's libraries, stored with lazy-sim.

Ersilia computes them with ``scripts/compare_featurizers.py``; each library has one
JSON file in ``lazy_sim/results/`` holding every pair of featurizers compared.
"""

import json
from importlib import resources


def _folder():
    return resources.files("lazy_sim") / "results"


def libraries():
    """Return the names of the libraries with stored comparisons."""
    folder = _folder()
    if not folder.is_dir():
        return []
    return sorted(
        p.name[: -len(".json")] for p in folder.iterdir() if p.name.endswith(".json")
    )


def pair_key(a, b):
    """Name a pair of featurizers the same way, whatever their order."""
    return "-".join(sorted((a, b)))


def pair(library, a, b):
    """Return a library's stored results and its comparison of featurizers a and b.

    Parameters
    ----------
    library : str
        Ersilia library name, such as ``ersilia_reference_library_1M_v0``.
    a, b : str
        Featurizers, in any order.

    Returns
    -------
    tuple of dict
        The library's results (``library``, ``molecules``, ``pairs``) and the pair.

    Raises
    ------
    ValueError
        If nothing is stored for that library or that pair; the message lists what is.
    """
    path = _folder() / f"{library}.json"
    if not path.is_file():
        stored = ", ".join(libraries()) or "none yet"
        raise ValueError(f"No comparisons stored for {library!r}. Libraries: {stored}")
    stored = json.loads(path.read_text())
    entry = stored["pairs"].get(pair_key(a, b))
    if entry is None:
        pairs = ", ".join(" vs ".join(k.split("-")) for k in sorted(stored["pairs"]))
        raise ValueError(f"No comparison of {a} and {b} for {library}. Stored: {pairs}")
    return stored, entry
