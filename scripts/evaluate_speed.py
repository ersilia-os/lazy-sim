"""Maintainer script: time a lazy-sim search from launch to hits CSV, stage by stage.

Each method runs in a fresh process that loads what it needs, embeds the queries,
searches and writes a hits CSV, as ``lazysim search`` does:
- lazy-sim uses the featurizer's own search engine (turbovec for CLAMP, FPSim2 for
  ECFP);
- exact compares each query with every stored embedding (cosine for CLAMP,
  Tanimoto for ECFP). It runs only when the library has the featurizer's
  embeddings (``lazysim fetch NAME -f FEATURIZER --embeddings``).

Before each run, the model and library files are dropped from the system's file
cache, so loading them counts as a first read from disk. The queries are random
library molecules (seeded). ``evaluate_recall.py`` measures how accurate lazy-sim is.

Usage
-----
python scripts/evaluate_speed.py ersilia-reference -f clamp -n 1000 -k 100
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from evaluate_recall import SIMILARITY, draw_queries, exact_search
from lazy_sim.engines import get_engine
from lazy_sim.featurizers import AVAILABLE, DEFAULT, clamp, ecfp, get_featurizer
from lazy_sim.io import artifacts, home, tables
from lazy_sim.utils.parallel import set_threads

METHODS = {"lazy-sim": "lazy-sim", "exact": "Exact"}
STAGES = (
    "Start-up and imports",
    "Load model and library",
    "Embed queries and search",
    "Look up hits, write CSV",
)


def query_embedder(featurizer):
    """Return a function giving SMILES the layout of ``<featurizer>_embeddings.npy``."""
    if featurizer == "clamp":
        weights = clamp.load_weights()
        return lambda smiles: clamp.encode(weights, clamp.fingerprints(smiles))
    return lambda smiles: np.array(ecfp.packed_fingerprints(smiles))


def timed_run(method, featurizer, folder, queries_csv, hits_csv, k, launched, threads):
    """Search in this fresh process and print the time of each stage as JSON."""
    times = {STAGES[0]: time.time() - float(launched)}
    clock = time.perf_counter()

    def lap(stage):
        nonlocal clock
        now = time.perf_counter()
        times[stage] = now - clock
        clock = now

    folder, k, threads = Path(folder), int(k), int(threads) or os.cpu_count()
    with open(queries_csv, newline="") as fh:
        queries = [(row["id"], row["smiles"]) for row in csv.DictReader(fh)]
    smiles = [s for _, s in queries]
    if method == "lazy-sim":
        feat = get_featurizer(featurizer)
        manifest = artifacts.read_featurizer(folder, featurizer)
        engine = get_engine(feat.ENGINE)
        with engine.Searcher(feat, folder, manifest, threads, len(smiles)) as searcher:
            lap(STAGES[1])
            results = searcher.search(smiles, k)
            lap(STAGES[2])
        scores, rows = [s for s, _ in results], [r for _, r in results]
    else:
        embed = query_embedder(featurizer)
        embeddings_npy = folder / artifacts.embeddings_file(featurizer)
        embeddings = np.load(embeddings_npy, mmap_mode="r")  # read during the search
        lap(STAGES[1])
        similarity = SIMILARITY[featurizer]
        scores, rows = exact_search(embeddings, embed(smiles), k, threads, similarity)
        lap(STAGES[2])
    con = artifacts.open_library_db(folder)
    hits = artifacts.lookup(con, np.unique(np.concatenate(rows)))
    with open(hits_csv, "w", newline="") as fh:
        writer = tables.hits_writer(fh)
        for (query_id, query), q_scores, q_rows in zip(queries, scores, rows):
            for rank, (score, row) in enumerate(zip(q_scores, q_rows), 1):
                writer.writerow(
                    [query_id, query, rank, *hits[int(row)], f"{min(score, 1.0):.4f}"]
                )
    lap(STAGES[3])
    print(json.dumps(times))


def drop_from_cache(paths):
    """Make the system forget its cached copy of `paths`; False if it cannot."""
    if not hasattr(os, "posix_fadvise"):
        return False
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library", help="Library name or folder.")
    parser.add_argument("-f", "--featurizer", choices=AVAILABLE, default=DEFAULT)
    parser.add_argument("-n", "--queries", type=int, default=1_000)
    parser.add_argument("-k", type=int, default=100, help="Hits per query.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-t", "--threads", type=int)
    args = parser.parse_args()
    featurizer = args.featurizer

    folder = home.resolve(args.library, must_exist=True)
    if featurizer not in artifacts.featurizers_in(folder):
        parser.error(f"{folder} has no {featurizer} index")
    manifest = artifacts.read_featurizer(folder, featurizer)
    files = [folder / manifest["index_file"], folder / artifacts.LIBRARY_DB]
    if featurizer == "clamp":
        files.append(clamp.ensure_weights())
    embeddings_npy = folder / artifacts.embeddings_file(featurizer)
    methods = ["lazy-sim"]
    if embeddings_npy.exists() and featurizer in SIMILARITY:
        methods.append("exact")
        files.append(embeddings_npy)
    con = artifacts.open_library_db(folder)
    rows, molecules = draw_queries(con, args.queries, args.seed)
    con.close()
    set_threads(args.threads)  # inherited by the timed runs

    times = {}
    with tempfile.TemporaryDirectory() as tmp:
        queries_csv = Path(tmp) / "queries.csv"
        with open(queries_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "smiles"])
            writer.writerows(molecules[int(r)] for r in rows)
        for method in methods:
            from_disk = drop_from_cache(files)
            run = [sys.executable, __file__, "--timed-run", method, featurizer]
            run += [folder, queries_csv, Path(tmp) / "hits.csv", args.k, time.time()]
            run += [args.threads or 0]
            result = subprocess.run(
                [str(a) for a in run], stdout=subprocess.PIPE, text=True, check=True
            )
            times[method] = json.loads(result.stdout.splitlines()[-1])

    console = Console()
    queries = "1 query" if len(rows) == 1 else f"{len(rows):,} queries"
    console.print(f"{folder.name}, {featurizer}: {queries}, {args.k} hits each")
    console.print("Time from launch to hits CSV, in seconds")
    table = Table()
    table.add_column("Stage")
    for method in methods:
        table.add_column(METHODS[method], justify="right")
    for stage in STAGES:
        table.add_row(stage, *(f"{times[m][stage]:.2f}" for m in methods))
    totals = (f"{sum(times[m].values()):.2f}" for m in methods)
    table.add_row("Total", *totals, style="bold")
    console.print(table)
    if from_disk:
        console.print("Model and library files were read from disk in each run.")
    else:
        console.print("Could not drop cached files here: loading may be too fast.")
    if "exact" not in methods:
        console.print(
            f"No {featurizer} embeddings in this library, so no exact search."
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--timed-run"]:
        timed_run(*sys.argv[2:])
    else:
        main()
