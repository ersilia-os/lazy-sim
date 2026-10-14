"""Maintainer script: measure how many true nearest neighbours CLAMP search returns.

The library needs its CLAMP embeddings (``lazysim fetch NAME --embeddings``): they
give the exact answer, the cosine of each query against every library molecule.

1. Draw random library molecules as queries (seeded).
2. Search them with ``lazy_sim.search``, as ``lazysim search`` does, for k + 1 hits
   (the largest k asked for).
3. Compute the exact k + 1 nearest neighbours from the stored embeddings.
4. Drop each query's own row from both lists. For each k asked for, recall@k is the
   share of the true k nearest neighbours that lazy-sim also returned in its top k,
   averaged over the queries. The score error compares lazy-sim's similarity with
   the exact cosine of the same pair.

Usage
-----
python scripts/evaluate_recall.py ersilia-reference -k 1,10,100,1000
"""

import argparse
import csv
import tempfile
import time
from pathlib import Path

import numpy as np

import lazy_sim
from lazy_sim.io import artifacts, home

CHUNK = 20_000


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


def exact_top(embeddings, queries, k):
    """Return the rows and cosines of the `k` nearest library molecules, best first."""
    rows = np.empty((len(queries), 0), np.int64)
    scores = np.empty((len(queries), 0), np.float32)
    for start in range(0, len(embeddings), CHUNK):
        block = queries @ np.asarray(embeddings[start : start + CHUNK]).T
        scores = np.concatenate([scores, block], axis=1)
        block_rows = np.arange(start, start + block.shape[1])
        rows = np.concatenate([rows, np.broadcast_to(block_rows, block.shape)], axis=1)
        if scores.shape[1] > k:
            keep = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            rows = np.take_along_axis(rows, keep, axis=1)
            scores = np.take_along_axis(scores, keep, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(rows, order, axis=1), np.take_along_axis(
        scores, order, axis=1
    )


def lazysim_top(hits_csv, row_of_id):
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


def without(rows, row, k):
    """Return the first `k` of `rows`, leaving `row` (the query itself) out."""
    return [r for r in rows if r != row][:k]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library", help="Library name or folder.")
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
    k = args.k[-1]

    folder = home.resolve(args.library, must_exist=True)
    embeddings = np.load(folder / artifacts.embeddings_file("clamp"), mmap_mode="r")
    con = artifacts.open_library_db(folder)
    rows, molecules = draw_queries(con, args.queries, args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        queries_csv, hits_csv = Path(tmp) / "queries.csv", Path(tmp) / "hits.csv"
        with open(queries_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "smiles"])
            writer.writerows(molecules[int(r)] for r in rows)
        start = time.perf_counter()
        lazy_sim.search(
            queries_csv,
            folder,
            hits_csv,
            k=k + 1,
            featurizer="clamp",
            threads=args.threads,
        )
        search_s = time.perf_counter() - start
        with open(hits_csv, newline="") as fh:
            hit_ids = {hit["hit_id"] for hit in csv.DictReader(fh)}
        found = lazysim_top(hits_csv, rows_of(con, hit_ids))
    con.close()

    q_vectors = np.asarray(embeddings[rows])
    true_rows, _ = exact_top(embeddings, q_vectors, k + 1)

    recall = {n: [] for n in args.k}
    errors = []
    for i, row in enumerate(rows):
        query_id = molecules[int(row)][0]
        got_rows, got_scores = found[query_id]
        got = without(got_rows, row, k)
        true = without(true_rows[i].tolist(), row, k)
        for n in args.k:
            recall[n].append(len(set(got[:n]) & set(true[:n])) / n)
        exact = embeddings[np.array(got_rows)] @ q_vectors[i]
        errors.extend(np.abs(np.array(got_scores) - np.minimum(exact, 1.0)))

    print(
        f"{folder.name}: {len(rows):,} queries, {k + 1} hits each in {search_s:.1f} s"
    )
    for n in args.k:
        print(f"recall@{n} = {np.mean(recall[n]):.4f}")
    print(f"score error: mean {np.mean(errors):.4f}, max {np.max(errors):.4f}")


if __name__ == "__main__":
    main()
