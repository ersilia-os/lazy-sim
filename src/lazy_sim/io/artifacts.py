"""The artifacts folder that holds an indexed library.

The featurizer weights are not in it: they are shared, in ``~/lazysim/models``.

Layout
------
manifest.json               what was indexed, how, and with which versions (last)
<featurizer>.tv             turbovec index; slot i is row i of library.db
library.db                  SQLite table ``library(row, id, smiles)``
skipped.csv                 input rows that could not be embedded
<featurizer>_embeddings.npy optional full float32 embeddings, rows as in library.db
"""

import json
import os
import sqlite3
from pathlib import Path

MANIFEST = "manifest.json"
LIBRARY_DB = "library.db"
SKIPPED = "skipped.csv"
SKIPPED_COLUMNS = ("input_row", "id", "smiles", "reason")
_SQL_BATCH = 900


def index_file(featurizer):
    """Return the index file name for `featurizer`."""
    return f"{featurizer}.tv"


def embeddings_file(featurizer):
    """Return the embeddings file name for `featurizer`."""
    return f"{featurizer}_embeddings.npy"


def open_embeddings(path, rows, dim):
    """Create a float32 ``.npy`` file for up to `rows` embeddings, as a memmap."""
    import numpy as np  # lazy: the CLI imports this module before thread setup

    return np.lib.format.open_memmap(path, "w+", np.float32, (rows, dim))


def shrink_embeddings(path, rows, dim):
    """Keep only the first `rows` rows of a file made by `open_embeddings`."""
    import numpy as np

    with open(path, "r+b") as fh:
        np.lib.format.read_magic(fh)
        np.lib.format.read_array_header_1_0(fh)
        offset = fh.tell()
        header = repr({"descr": "<f4", "fortran_order": False, "shape": (rows, dim)})
        fh.seek(10)  # magic string, version and header length come first
        fh.write(header.ljust(offset - 11).encode("latin1") + b"\n")
        fh.truncate(offset + rows * dim * 4)


def disk_size(path):
    """Return the total size of the files in `path`, formatted in MB or GB."""
    total = sum(f.stat().st_size for f in Path(path).iterdir() if f.is_file())
    return f"{total / 1e9:.2f} GB" if total >= 1e9 else f"{total / 1e6:.1f} MB"


def prepare_dir(path):
    """Create the artifacts folder, refusing one that already has files in it."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; choose a new artifacts folder")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(path, manifest):
    """Write ``manifest.json`` atomically; its presence marks a finished index."""
    tmp = Path(path) / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, Path(path) / MANIFEST)


def read_manifest(path):
    """Read ``manifest.json`` from an artifacts folder.

    Parameters
    ----------
    path : str or Path
        Artifacts folder.

    Returns
    -------
    dict
        The manifest.
    """
    file = Path(path) / MANIFEST
    if not file.exists():
        raise FileNotFoundError(
            f"No {MANIFEST} in {path}: not a lazy-sim index, or indexing did not finish"
        )
    return json.loads(file.read_text())


def create_library_db(path):
    """Create ``library.db`` and return an open connection."""
    con = sqlite3.connect(Path(path) / LIBRARY_DB)
    con.execute("CREATE TABLE library (row INTEGER PRIMARY KEY, id TEXT, smiles TEXT)")
    return con


def open_library_db(path):
    """Open ``library.db`` read-only."""
    uri = (Path(path) / LIBRARY_DB).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def lookup(con, rows):
    """Map index rows to ``(id, smiles)``.

    Parameters
    ----------
    con : sqlite3.Connection
        Open ``library.db``.
    rows : iterable of int
        Index rows to look up.

    Returns
    -------
    dict of int to tuple of (str, str)
    """
    rows = [int(r) for r in rows]
    found = {}
    for start in range(0, len(rows), _SQL_BATCH):
        batch = rows[start : start + _SQL_BATCH]
        marks = ",".join("?" * len(batch))
        query = f"SELECT row, id, smiles FROM library WHERE row IN ({marks})"
        found.update({r: (i, s) for r, i, s in con.execute(query, batch)})
    return found
