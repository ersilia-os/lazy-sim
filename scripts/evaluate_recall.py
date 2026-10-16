"""Maintainer script: how many true nearest neighbours the indexed search returns.

The library needs the featurizer's embeddings (``lazysim fetch NAME -f FEATURIZER
--embeddings``). Comparing each query with every embedding gives the exact answer
(cosine for CLAMP, SAND, Mol-JEPA and Monroe; Tanimoto for ECFP).

1. Draw random library molecules as queries (seeded).
2. Search the index with ``lazy_sim.search``, as ``lazysim search`` does, for k + 1
   hits (k being the largest asked for).
3. Search the embeddings exactly, for k + 1 hits too.
4. Drop each query's own row from both lists. Recall@k is the share of the indexed
   top k that are true top-k neighbours, averaged over the queries. A hit counts
   when its exact similarity is at least that of the true k-th neighbour, so
   molecules tied at the cut-off are interchangeable (common with Tanimoto). The
   score error compares the indexed similarity with the exact one for the same pair.

``evaluate_speed.py`` measures how long both searches take.

Usage
-----
python scripts/evaluate_recall.py ersilia_reference_library_1M_v0 -f clamp -k 1,10,100,1000
"""

import argparse
import csv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

import lazy_sim
from lazy_sim.featurizers import AVAILABLE
from lazy_sim.io import artifacts, home

CHUNK = 20_000
TIE = 1e-6  # similarities closer than this are a tie


def cosine(queries, block):
    """Cosine of unit-length float32 embeddings."""
    return queries @ block.T


def tanimoto(queries, block):
    """Tanimoto of packed bit fingerprints (rows of ``np.packbits``)."""
    q = np.unpackbits(queries, axis=1).astype(np.float32)
    b = np.unpackbits(block, axis=1).astype(np.float32)
    common = q @ b.T
    union = q.sum(axis=1)[:, None] + b.sum(axis=1) - common
    return np.divide(common, union, out=np.zeros_like(common), where=union > 0)


SIMILARITY = {
    "clamp": cosine,
    "ecfp": tanimoto,
    "sand": cosine,
    "moljepa": cosine,
    "monroe": cosine,
}


def neighbour_counts(value):
    """Parse ``-k``: one count, or several separated by commas, such as 1,10,100."""
    try:
        counts = sorted({int(k) for k in value.split(",")})
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integers such as 1,10,100: {value}")
    if counts[0] < 1:
        raise argparse.ArgumentTypeError("k must be at least 1")
    return counts


def draw_queries(con, n, seed):
    """Return `n` random library rows, sorted, with their ``(id, smiles)``."""
    total = artifacts.count_rows(con)
    rows = np.sort(np.random.default_rng(seed).choice(total, min(n, total), False))
    return rows, artifacts.lookup(con, rows)


def _keep_best(scores, rows, k):
    if scores.shape[1] > k:
        cols = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        scores = np.take_along_axis(scores, cols, axis=1)
        rows = np.take_along_axis(rows, cols, axis=1)
    return scores, rows


def exact_search(embeddings, queries, k, threads, similarity):
    """Return the similarities and rows of the `k` nearest molecules, best first.

    The embeddings are read chunk by chunk; each thread keeps the best hits of its
    share of the queries.
    """
    parts = np.array_split(np.arange(len(queries)), min(threads, len(queries)))
    best = [
        (np.empty((len(p), 0), np.float32), np.empty((len(p), 0), np.int64))
        for p in parts
    ]
    with ThreadPoolExecutor(threads) as pool:
        for start in range(0, len(embeddings), CHUNK):
            block = similarity(queries, np.asarray(embeddings[start : start + CHUNK]))
            ids = np.arange(start, start + block.shape[1])

            def merge(i):
                scores, rows = best[i]
                part = block[parts[i]]
                return _keep_best(
                    np.concatenate([scores, part], axis=1),
                    np.concatenate([rows, np.broadcast_to(ids, part.shape)], axis=1),
                    k,
                )

            best = list(pool.map(merge, range(len(parts))))
    scores = np.concatenate([s for s, _ in best])
    rows = np.concatenate([r for _, r in best])
    order = np.argsort(-scores, axis=1)
    return (
        np.take_along_axis(scores, order, axis=1),
        np.take_along_axis(rows, order, axis=1),
    )


def indexed_hits(hits_csv, row_of_id):
    """Read the hits CSV into ``{query_id: (rows, similarities)}``, best first."""
    found = {}
    with open(hits_csv, newline="") as fh:
        for hit in csv.DictReader(fh):
            rows, scores = found.setdefault(hit["query_id"], ([], []))
            rows.append(row_of_id[hit["hit_id"]])
            scores.append(float(hit["similarity"]))
    return found


def rows_of(con, ids):
    """Map library ids to rows, reading ``library.db`` once."""
    ids = set(ids)
    return {i: r for r, i in con.execute("SELECT row, id FROM library") if i in ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library", help="Library name or folder.")
    parser.add_argument("-f", "--featurizer", choices=AVAILABLE, required=True)
    parser.add_argument(
        "-k",
        type=neighbour_counts,
        default=[100],
        help="Neighbours evaluated; several separated by commas, such as 1,10,100.",
    )
    parser.add_argument("-n", "--queries", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-t", "--threads", type=int)
    args = parser.parse_args()
    k, featurizer = args.k[-1], args.featurizer
    similarity = SIMILARITY.get(featurizer)
    if similarity is None:
        parser.error(f"no exact search for {featurizer} in this script yet")

    folder = home.resolve(args.library, must_exist=True)
    embeddings_npy = folder / artifacts.embeddings_file(featurizer)
    if not embeddings_npy.exists():
        parser.error(
            f"{folder} has no {embeddings_npy.name}: lazysim fetch NAME "
            f"-f {featurizer} --embeddings"
        )
    embeddings = np.load(embeddings_npy, mmap_mode="r")
    con = artifacts.open_library_db(folder)
    rows, molecules = draw_queries(con, args.queries, args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        queries_csv, hits_csv = Path(tmp) / "queries.csv", Path(tmp) / "hits.csv"
        with open(queries_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "smiles"])
            writer.writerows(molecules[int(r)] for r in rows)
        lazy_sim.search(
            queries_csv,
            folder,
            hits_csv,
            k=k + 1,
            featurizer=featurizer,
            threads=args.threads,
        )
        with open(hits_csv, newline="") as fh:
            hit_ids = {hit["hit_id"] for hit in csv.DictReader(fh)}
        found = indexed_hits(hits_csv, rows_of(con, hit_ids))
    con.close()

    q_vectors = np.asarray(embeddings[rows])
    threads = args.threads or os.cpu_count()
    true_scores, true_rows = exact_search(
        embeddings, q_vectors, k + 1, threads, similarity
    )

    recall = {n: [] for n in args.k}
    errors = []
    for i, row in enumerate(rows):
        got_rows, got_scores = found[molecules[int(row)][0]]
        got_rows = np.array(got_rows)
        exact = similarity(q_vectors[i : i + 1], embeddings[got_rows])[0]
        errors.extend(np.abs(np.array(got_scores) - np.minimum(exact, 1.0)))
        got = exact[got_rows != row][:k]  # exact similarity of the indexed hits
        true = true_scores[i][true_rows[i] != row][:k]
        for n in args.k:
            recall[n].append(np.sum(got[:n] >= true[n - 1] - TIE) / n)

    console = Console()
    queries = "1 query" if len(rows) == 1 else f"{len(rows):,} queries"
    console.print(f"{folder.name}, {featurizer}: {queries}, {k + 1} hits each")
    table = Table()
    table.add_column("k", justify="right")
    table.add_column("Recall@k", justify="right")
    for n in args.k:
        table.add_row(f"{n:,}", f"{np.mean(recall[n]):.4f}")
    console.print(table)
    console.print(f"Score error: mean {np.mean(errors):.4f}, max {np.max(errors):.4f}")


if __name__ == "__main__":
    main()
