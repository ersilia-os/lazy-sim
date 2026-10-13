"""Input CSVs (``smiles`` and ``id`` columns) and the output hits table."""

import csv

REQUIRED_COLUMNS = ("smiles", "id")
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


def _positions(path, header):
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(
            f"{path} must have 'smiles' and 'id' columns; missing {missing}"
        )
    return header.index("id"), header.index("smiles")


def check_columns(path):
    """Raise ValueError unless the CSV at `path` has ``smiles`` and ``id`` columns."""
    with _open(path) as fh:
        _positions(path, next(csv.reader(fh), []))


def iter_rows(path):
    """Yield ``(id, smiles)`` for every data row of the CSV at `path`."""
    with _open(path) as fh:
        reader = csv.reader(fh)
        i_id, i_smi = _positions(path, next(reader, []))
        width = max(i_id, i_smi) + 1
        for row in reader:
            if row:
                row += [""] * (width - len(row))
                yield row[i_id], row[i_smi]


def iter_chunks(path, size):
    """Yield lists of at most `size` ``(id, smiles)`` rows from the CSV at `path`."""
    chunk = []
    for row in iter_rows(path):
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
