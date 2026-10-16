"""Maintainer script: how far two featurizers agree on the nearest neighbours of a query.

The library needs both featurizers' indexes and embeddings (``lazysim fetch NAME -f A
-f B --embeddings``). Queries are random library molecules (seeded, the same ones as
``evaluate_recall.py``); each query's own row is left out of its neighbours.

Each query gets two top-k lists, one per featurizer, in two ways:
- indexed: the hits ``lazysim search`` returns (turbovec for the neural
  featurizers, FPSim2 for ECFP);
- exact: the nearest molecules by every stored embedding (cosine for CLAMP, SAND,
  Mol-JEPA and Monroe; Tanimoto for ECFP).

For each k, three measures per query, reported as the median and the interquartile
range over the queries:
- Molecules in both: the share of the top k that are in both lists.
- Similarity: the mean exact similarity to the query of each list, under each
  featurizer's own measure, next to that of random library molecules. Scores of
  different featurizers are never compared, as their scales differ.
- Position: the median rank of one featurizer's top k in the other's exact ranking
  of the whole library. Molecules tied with a hit do not push it down; a random
  molecule sits in the middle.

``--save`` stores the result in ``src/lazy_sim/results/<library>.json``, which
``lazysim compare`` shows. Only Ersilia libraries (``lazysim fetch``) are saved.

Usage
-----
python scripts/compare_featurizers.py ersilia_reference_library_1M_v0 -f clamp -f ecfp --save
"""

import argparse
import csv
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

import lazy_sim
from evaluate_recall import (
    CHUNK,
    SIMILARITY,
    TIE,
    draw_queries,
    exact_search,
    indexed_hits,
    neighbour_counts,
    rows_of,
    tanimoto,
)
from lazy_sim.cli.commands.compare import show
from lazy_sim.io import artifacts, home, results

RESULTS = Path(__file__).resolve().parents[1] / "src" / "lazy_sim" / "results"
RANDOM = 1_000  # random library molecules giving the similarity baseline


def top_neighbours(embeddings, queries, own_rows, k, threads, similarity):
    """Return the library rows of the `k` nearest molecules of each query, best first.

    Each query's own row is left out.
    """
    _, found = exact_search(embeddings, queries, k + 1, threads, similarity)
    return np.stack([f[f != own][:k] for f, own in zip(found, own_rows)])


def indexed_neighbours(folder, featurizer, own_rows, molecules, k, threads, con):
    """Return the library rows of the `k` hits ``lazysim search`` gives each query.

    Each query's own row is left out.
    """
    with tempfile.TemporaryDirectory() as tmp:
        queries_csv, hits_csv = Path(tmp) / "queries.csv", Path(tmp) / "hits.csv"
        with open(queries_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "smiles"])
            writer.writerows(
                (f"q{i}", molecules[int(r)][1]) for i, r in enumerate(own_rows)
            )
        lazy_sim.search(
            queries_csv,
            folder,
            hits_csv,
            k=k + 1,
            featurizer=featurizer,
            threads=threads,
        )
        with open(hits_csv, newline="") as fh:
            hit_ids = {hit["hit_id"] for hit in csv.DictReader(fh)}
        found = indexed_hits(hits_csv, rows_of(con, hit_ids))
    lists = []
    for i, own in enumerate(own_rows):
        got = np.array(found[f"q{i}"][0])
        lists.append(got[got != own][:k])
    return np.stack(lists)


def similarities(embeddings, queries, hits, similarity):
    """Return the similarity of each query to its `hits` (library rows, one row each)."""
    return np.stack(
        [
            similarity(q[None], np.asarray(embeddings[h]))[0]
            for q, h in zip(queries, hits)
        ]
    )


def ranks_in(embeddings, queries, own_rows, scores, threads, similarity):
    """Rank molecules of known similarity in each query's order of the whole library.

    Parameters
    ----------
    embeddings : np.ndarray
        Library embeddings (a memmap is read chunk by chunk).
    queries : np.ndarray
        Query embeddings.
    own_rows : np.ndarray
        Library row of each query, which is left out.
    scores : np.ndarray
        Similarities to each query (one row per query) of the molecules to rank.
    threads : int
        Threads sorting the similarities.
    similarity : callable
        Similarity of query embeddings to a block of library embeddings.

    Returns
    -------
    np.ndarray
        Ranks, shaped like `scores`: 1 plus the library molecules that are more
        similar to the query by over ``TIE``. Molecules tied with one do not count.
    """
    thresholds = scores + TIE
    above = np.zeros(scores.shape, np.int64)
    parts = np.array_split(np.arange(len(queries)), min(threads, len(queries)))

    def count(block, part):
        for i, ordered in zip(part, np.sort(block[part], axis=1)):
            above[i] += len(ordered) - np.searchsorted(ordered, thresholds[i], "right")

    with ThreadPoolExecutor(threads) as pool:
        for start in range(0, len(embeddings), CHUNK):
            block = similarity(queries, np.asarray(embeddings[start : start + CHUNK]))
            inside = np.flatnonzero((own_rows >= start) & (own_rows < start + CHUNK))
            block[inside, own_rows[inside] - start] = -np.inf
            list(pool.map(lambda part: count(block, part), parts))
    return above + 1


def summary(values, digits=4):
    """Return the first quartile, median and third quartile of per-query values."""
    return [round(float(v), digits) for v in np.percentile(values, [25, 50, 75])]


def compare(lists, emb, q, own_rows, ks, threads):
    """Summarise the measures for each mode's top-k lists (``lists[mode][f]``)."""
    (a, b), modes = list(emb), list(lists)
    sim = {
        (m, f, g): similarities(emb[f], q[f], lists[m][g], SIMILARITY[f])
        for m in modes
        for f in (a, b)
        for g in (a, b)
    }
    rank = {}  # rank[m, f, g]: ranks in f's order of g's top k; one pass per f
    for f, g in ((a, b), (b, a)):
        both = np.hstack([sim[m, f, g] for m in modes])
        found = ranks_in(emb[f], q[f], own_rows, both, threads, SIMILARITY[f])
        for m, part in zip(modes, np.hsplit(found, len(modes))):
            rank[m, f, g] = part
    out = {}
    for m in modes:
        out[m] = {
            "shared": {},
            "similarity": {f: {g: {} for g in (a, b)} for f in (a, b)},
            "rank": {f: {g: {}} for f, g in ((a, b), (b, a))},
        }
        for k in ks:
            both = [
                len(np.intersect1d(x[:k], y[:k])) / k
                for x, y in zip(lists[m][a], lists[m][b])
            ]
            out[m]["shared"][str(k)] = summary(both)
            for f in (a, b):
                for g in (a, b):
                    per_query = sim[m, f, g][:, :k].mean(axis=1)
                    out[m]["similarity"][f][g][str(k)] = summary(per_query)
            for f, g in ((a, b), (b, a)):
                per_query = np.median(rank[m, f, g][:, :k], axis=1)
                out[m]["rank"][f][g][str(k)] = summary(per_query, 1)
    return out


def dumps(stored):
    """Write results as JSON, each [first quartile, median, third quartile] on one line."""
    text = json.dumps(stored, indent=2)
    flat = re.sub(
        r"\[\s+([^\[\]{}]+?)\s+\]",
        lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]",
        text,
    )
    return flat + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library", help="Library name or folder.")
    parser.add_argument(
        "-f",
        "--featurizer",
        action="append",
        choices=sorted(SIMILARITY),
        required=True,
        help="Give it twice: the two featurizers to compare.",
    )
    parser.add_argument(
        "-k",
        type=neighbour_counts,
        default=[10, 100, 1000],
        help="Neighbours compared; several separated by commas, such as 10,100.",
    )
    parser.add_argument("-n", "--queries", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-t", "--threads", type=int)
    parser.add_argument(
        "--save", action="store_true", help="Store the result for 'lazysim compare'."
    )
    args = parser.parse_args()
    if len(set(args.featurizer)) != 2 or len(args.featurizer) != 2:
        parser.error("give -f twice, with two different featurizers")
    a, b = sorted(args.featurizer)
    k, threads = args.k[-1], args.threads or os.cpu_count()

    folder = home.resolve(args.library, must_exist=True)
    source = artifacts.read_manifest(folder).get("source")
    if args.save and not source:
        parser.error(f"{folder} is not an Ersilia library: only those are saved")
    emb = {}
    for f in (a, b):
        embeddings_npy = folder / artifacts.embeddings_file(f)
        if not embeddings_npy.exists():
            parser.error(
                f"{folder} has no {embeddings_npy.name}: lazysim fetch NAME "
                f"-f {f} --embeddings"
            )
        emb[f] = np.load(embeddings_npy, mmap_mode="r")
    con = artifacts.open_library_db(folder)
    n = artifacts.count_rows(con)
    if k >= n:
        parser.error("k must be smaller than the number of library molecules")
    rows, molecules = draw_queries(con, args.queries, args.seed)
    q = {f: np.asarray(emb[f][rows]) for f in (a, b)}
    lists = {
        "indexed": {
            f: indexed_neighbours(folder, f, rows, molecules, k, args.threads, con)
            for f in (a, b)
        },
        "exact": {
            f: top_neighbours(emb[f], q[f], rows, k, threads, SIMILARITY[f])
            for f in (a, b)
        },
    }
    con.close()
    others = np.sort(
        np.random.default_rng(args.seed + 1).choice(n, min(n, RANDOM), False)
    )
    random = {
        f: summary(SIMILARITY[f](q[f], np.asarray(emb[f][others])).mean(axis=1))
        for f in (a, b)
    }

    library = source["name"] if source else folder.name
    entry = {
        "featurizers": [a, b],
        "measures": {
            f: "Tanimoto" if SIMILARITY[f] is tanimoto else "cosine" for f in (a, b)
        },
        "queries": len(rows),
        "seed": args.seed,
        "k": args.k,
        "lazysim_version": lazy_sim.__version__,
        "created": date.today().isoformat(),
        "random": random,
        **compare(lists, emb, q, rows, args.k, threads),
    }
    stored = {"library": library, "molecules": n, "pairs": {}}
    path = RESULTS / f"{library}.json"
    if args.save and path.exists():
        stored = json.loads(path.read_text())
    stored["pairs"][results.pair_key(a, b)] = entry
    stored["pairs"] = dict(sorted(stored["pairs"].items()))
    show(stored, entry)
    if args.save:
        RESULTS.mkdir(exist_ok=True)
        path.write_text(dumps(stored))
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
