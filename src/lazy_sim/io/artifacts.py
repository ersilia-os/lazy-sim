"""The artifacts folder that holds a library and one index per featurizer.

The featurizer weights are not in it: they are shared, in ``~/lazysim/models``.

Layout
------
manifest.json               the library: input file, counts, versions (written last)
library.db                  SQLite table ``library(row, id, smiles)``
skipped.csv                 input rows that could not be parsed
<featurizer>.json           how that featurizer's index was built (written last)
<featurizer>.<ext>          its index; slot or id i is row i of library.db
<featurizer>_embeddings.npy optional full embeddings, row i is row i of library.db

A featurizer is available when its ``.json`` exists.
"""

import json
import os
import random
import sqlite3
from pathlib import Path

MANIFEST = "manifest.json"
LIBRARY_DB = "library.db"
SKIPPED = "skipped.csv"
SKIPPED_COLUMNS = ("input_row", "id", "smiles", "reason")
_SQL_BATCH = 900


def featurizer_file(featurizer):
    """Return the name of `featurizer`'s manifest file."""
    return f"{featurizer}.json"


def embeddings_file(featurizer):
    """Return the name of `featurizer`'s embeddings file."""
    return f"{featurizer}_embeddings.npy"


def open_embeddings(path, rows, width, dtype):
    """Create a ``.npy`` file of shape ``(rows, width)``, opened as a memmap."""
    import numpy as np  # lazy: the CLI imports this module before thread setup

    return np.lib.format.open_memmap(path, "w+", np.dtype(dtype), (rows, width))


def files_size(paths):
    """Return the total size of `paths`, formatted in MB or GB."""
    total = sum(Path(p).stat().st_size for p in paths if Path(p).is_file())
    return f"{total / 1e9:.2f} GB" if total >= 1e9 else f"{total / 1e6:.1f} MB"


def prepare_dir(path):
    """Create the artifacts folder, refusing one that already has files in it."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; choose a new artifacts folder")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def write_manifest(path, manifest):
    """Write ``manifest.json`` atomically; its presence marks a finished library."""
    _write_json(Path(path) / MANIFEST, manifest)


def read_manifest(path):
    """Read ``manifest.json`` from an artifacts folder.

    Parameters
    ----------
    path : str or Path
        Artifacts folder.

    Returns
    -------
    dict
        The library manifest.
    """
    file = Path(path) / MANIFEST
    if not file.exists():
        raise FileNotFoundError(
            f"No {MANIFEST} in {path}: not a lazy-sim library, or it did not finish"
        )
    return json.loads(file.read_text())


def write_featurizer(path, featurizer, manifest):
    """Write ``<featurizer>.json`` atomically; it marks a finished index."""
    _write_json(Path(path) / featurizer_file(featurizer), manifest)


def read_featurizer(path, featurizer):
    """Read ``<featurizer>.json`` from an artifacts folder."""
    return json.loads((Path(path) / featurizer_file(featurizer)).read_text())


def featurizers_in(path):
    """Return the names of the featurizers whose index in `path` is finished."""
    return sorted(f.stem for f in Path(path).glob("*.json") if f.name != MANIFEST)


def create_library_db(path):
    """Create ``library.db`` and return an open connection."""
    con = sqlite3.connect(Path(path) / LIBRARY_DB)
    con.execute("CREATE TABLE library (row INTEGER PRIMARY KEY, id TEXT, smiles TEXT)")
    return con


def open_library_db(path):
    """Open ``library.db`` read-only."""
    uri = (Path(path) / LIBRARY_DB).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def count_rows(con):
    """Return the number of molecules in ``library.db``."""
    return con.execute("SELECT COUNT(*) FROM library").fetchone()[0]


def iter_library(con, size):
    """Yield lists of at most `size` ``(row, smiles)``, in row order."""
    start, n = 0, count_rows(con)
    while start < n:
        query = (
            "SELECT row, smiles FROM library WHERE row >= ? AND row < ? ORDER BY row"
        )
        yield con.execute(query, (start, start + size)).fetchall()
        start += size


def sample_smiles(con, n, seed=0):
    """Return the SMILES of `n` rows of ``library.db`` drawn at random (all if fewer)."""
    total = count_rows(con)
    rows = sorted(random.Random(seed).sample(range(total), min(n, total)))
    return [smiles for _, (_, smiles) in sorted(lookup(con, rows).items())]


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
