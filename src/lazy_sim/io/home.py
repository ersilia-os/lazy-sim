"""The lazy-sim home folder (``~/lazysim``; ``$LAZYSIM_HOME`` moves it).

It holds named libraries in ``libraries/`` and the featurizer weights, downloaded
once and shared by every library, in ``models/``.

Every ``-a`` option takes either a library name, meaning a folder in the home, or
a path. A name is letters, digits and ``_.-``, starting with a letter or digit;
anything else (``./mylib``, ``/data/chembl``) is a path.
"""

import json
import os
import re
from pathlib import Path

from .artifacts import MANIFEST

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def home_dir():
    """Return the lazy-sim home folder."""
    return Path(os.environ.get("LAZYSIM_HOME") or Path.home() / "lazysim")


def libraries_dir():
    """Return the folder of named libraries."""
    return home_dir() / "libraries"


def models_dir():
    """Return the folder of featurizer weights."""
    return home_dir() / "models"


def is_name(value):
    """Tell whether `value` is a library name rather than a path."""
    return _NAME.fullmatch(str(value)) is not None


def resolve(value, must_exist=False):
    """Turn a library name or a path into the artifacts folder it refers to.

    Parameters
    ----------
    value : str or Path
        Library name or folder path.
    must_exist : bool
        Raise FileNotFoundError unless the folder holds a finished library.

    Returns
    -------
    Path
    """
    path = libraries_dir() / str(value) if is_name(value) else Path(value).expanduser()
    if must_exist and not (path / MANIFEST).exists():
        if is_name(value):
            raise FileNotFoundError(
                f"No library named {value!r} in {libraries_dir()}; "
                "see 'lazysim catalog' and 'lazysim catalog --remote'"
            )
        raise FileNotFoundError(
            f"No {MANIFEST} in {path}: not a lazy-sim library, or it did not finish"
        )
    return path


def list_local():
    """Return ``(name, path, manifest)`` for every finished library in the home."""
    root = libraries_dir()
    if not root.exists():
        return []
    found = []
    for path in sorted(root.iterdir()):
        if (path / MANIFEST).exists():
            found.append((path.name, path, json.loads((path / MANIFEST).read_text())))
    return found
