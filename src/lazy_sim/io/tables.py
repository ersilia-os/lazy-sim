"""Input CSVs (``smiles`` and, for libraries, ``id`` columns) and the hits table.

Query CSVs may leave out ``id``; their rows then come back with id None.
"""

import csv

HITS_COLUMNS = (
    "query_id",
    "query_smiles",
    "rank",
    "hit_id",
    "hit_smiles",
    "similarity",
)


def _open(path):
    return open(path, newline="", encoding="utf-8-sig")


def _positions(path, header, require_id):
    needed = ("smiles", "id") if require_id else ("smiles",)
    missing = [c for c in needed if c not in header]
    if missing:
        raise ValueError(f"{path} must have {' and '.join(needed)} columns")
    i_id = header.index("id") if "id" in header else None
    return i_id, header.index("smiles")


def check_columns(path, require_id=True):
    """Raise ValueError unless the CSV at `path` has the columns it needs.

    Parameters
    ----------
    path : str or Path
        CSV file.
    require_id : bool
        Whether an ``id`` column is required besides ``smiles``.
    """
    with _open(path) as fh:
        _positions(path, next(csv.reader(fh), []), require_id)


def iter_rows(path, require_id=True):
    """Yield ``(id, smiles)`` for every data row; id is None without an id column."""
    with _open(path) as fh:
        reader = csv.reader(fh)
        i_id, i_smi = _positions(path, next(reader, []), require_id)
        width = max(i for i in (i_id, i_smi) if i is not None) + 1
        for row in reader:
            if row:
                row += [""] * (width - len(row))
                yield (None if i_id is None else row[i_id]), row[i_smi]


def iter_chunks(path, size, require_id=True):
    """Yield lists of at most `size` ``(id, smiles)`` rows from the CSV at `path`."""
    chunk = []
    for row in iter_rows(path, require_id):
        chunk.append(row)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def hits_writer(fh):
    """Return a csv writer on `fh` with the hits header already written."""
    writer = csv.writer(fh)
    writer.writerow(HITS_COLUMNS)
    return writer
