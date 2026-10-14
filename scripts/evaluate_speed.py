"""Maintainer script: time ``lazysim search`` from launch to exit, stage by stage.

Two searches run, each in a fresh process:
- Indexed runs the real ``lazysim search`` command, which searches the featurizer's
  index (turbovec for CLAMP, FPSim2 for ECFP). Timers around its search engine
  record loading and searching; the remaining time goes to reading the queries,
  writing the hits CSV, and starting and stopping the process.
- Embeddings (exact) does the same work, but compares each query with every stored
  embedding (cosine for CLAMP, Tanimoto for ECFP). It runs only when the library
  has the featurizer's embeddings (``lazysim fetch NAME -f FEATURIZER --embeddings``).

The total is measured from launching the process until it exits. Before each run,
the model and library files are dropped from the system's file cache, so loading
them counts as a first read from disk. Each search runs ``--repeats`` times, and
the run with the median total is shown. The queries are random library molecules
(seeded). ``evaluate_recall.py`` measures how accurate the indexed search is.

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
from lazy_sim.cli import cli
from lazy_sim.engines import get_engine
from lazy_sim.engines.turbovec import POOL_MIN_QUERIES
from lazy_sim.featurizers import AVAILABLE, DEFAULT, clamp, ecfp, get_featurizer
from lazy_sim.io import artifacts, home, tables
from lazy_sim.utils.parallel import ChunkPool, set_threads

METHODS = {"indexed": "Indexed", "exact": "Embeddings (exact)"}
STAGES = (
    "Start-up and imports",
    "Load model and library",
    "Embed queries and search",
    "Read queries, write hits CSV",
)


def _add_timer(cls, name, spent, stage):
    """Wrap ``cls.name`` so that the time spent in it adds up in ``spent[stage]``."""
    method = getattr(cls, name)

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            spent[stage] += time.perf_counter() - start

    setattr(cls, name, timed)


def run_indexed(featurizer, folder, queries_csv, hits_csv, k, threads, launched, out):
    """Run the ``lazysim search`` command here, timing its engine; save to `out`."""
    searcher = get_engine(get_featurizer(featurizer).ENGINE).Searcher
    spent = {STAGES[1]: 0.0, STAGES[2]: 0.0}
    _add_timer(searcher, "__init__", spent, STAGES[1])
    _add_timer(searcher, "__enter__", spent, STAGES[1])
    _add_timer(searcher, "search", spent, STAGES[2])
    spent[STAGES[0]] = time.time() - float(launched)
    args = ["search", "-i", queries_csv, "-a", folder, "-o", hits_csv, "-k", k]
    args += ["-f", featurizer] + (["-t", threads] if int(threads) else [])
    cli.main(args, standalone_mode=False)
    Path(out).write_text(json.dumps(spent))


def run_exact(featurizer, folder, queries_csv, hits_csv, k, threads, launched, out):
    """Search the embeddings exactly, as ``lazysim search`` would; times to `out`.

    Queries are fingerprinted as ``lazysim search`` does: in worker processes when
    there are many of them.
    """
    spent = {STAGES[0]: time.time() - float(launched)}
    folder, k, threads = Path(folder), int(k), int(threads) or os.cpu_count()
    with open(queries_csv, newline="") as fh:
        queries = [(row["id"], row["smiles"]) for row in csv.DictReader(fh)]
    clock = time.perf_counter()
    if featurizer == "clamp":
        weights, fingerprint = clamp.load_weights(), clamp.fingerprints
    else:
        weights, fingerprint = None, ecfp.packed_fingerprints
    embeddings_npy = folder / artifacts.embeddings_file(featurizer)
    embeddings = np.load(embeddings_npy, mmap_mode="r")  # read during the search
    processes = threads if len(queries) >= POOL_MIN_QUERIES else 1
    with ChunkPool(fingerprint, processes) as pool:
        spent[STAGES[1]] = time.perf_counter() - clock
        clock = time.perf_counter()
        fps = pool.map([s for _, s in queries])
        vectors = clamp.encode(weights, fps) if weights else np.array(fps)
        similarity = SIMILARITY[featurizer]
        scores, rows = exact_search(embeddings, vectors, k, threads, similarity)
        spent[STAGES[2]] = time.perf_counter() - clock
    hits = artifacts.lookup(artifacts.open_library_db(folder), np.unique(rows))
    with open(hits_csv, "w", newline="") as fh:
        writer = tables.hits_writer(fh)
        for (query_id, query), q_scores, q_rows in zip(queries, scores, rows):
            for rank, (score, row) in enumerate(zip(q_scores, q_rows), 1):
                writer.writerow(
                    [query_id, query, rank, *hits[int(row)], f"{min(score, 1.0):.4f}"]
                )
    Path(out).write_text(json.dumps(spent))


RUNNERS = {"indexed": run_indexed, "exact": run_exact}


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


def time_run(method, args, folder, queries_csv, tmp):
    """Run one search in a fresh process; return its stage times and total."""
    out = Path(tmp) / "times.json"
    launched = time.time()
    run = [sys.executable, __file__, "--run", method, args.featurizer, folder]
    run += [queries_csv, Path(tmp) / "hits.csv", args.k, args.threads or 0]
    result = subprocess.run(
        [str(a) for a in run + [launched, out]], capture_output=True, text=True
    )
    total = time.time() - launched
    if result.returncode:
        sys.exit(f"{method} search failed:\n{result.stderr}")
    spent = json.loads(out.read_text())
    spent[STAGES[3]] = total - sum(spent.values())
    spent["Total"] = total
    return spent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library", help="Library name or folder.")
    parser.add_argument("-f", "--featurizer", choices=AVAILABLE, default=DEFAULT)
    parser.add_argument("-n", "--queries", type=int, default=1_000)
    parser.add_argument("-k", type=int, default=100, help="Hits per query.")
    parser.add_argument("-r", "--repeats", type=int, default=3)
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
    methods = ["indexed"]
    if embeddings_npy.exists() and featurizer in SIMILARITY:
        methods.append("exact")
        files.append(embeddings_npy)
    con = artifacts.open_library_db(folder)
    rows, molecules = draw_queries(con, args.queries, args.seed)
    con.close()
    set_threads(args.threads)  # inherited by the timed runs

    shown = {}
    with tempfile.TemporaryDirectory() as tmp:
        queries_csv = Path(tmp) / "queries.csv"
        with open(queries_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "smiles"])
            writer.writerows(molecules[int(r)] for r in rows)
        for method in methods:
            runs = []
            for _ in range(args.repeats):
                from_disk = drop_from_cache(files)
                runs.append(time_run(method, args, folder, queries_csv, tmp))
            shown[method] = sorted(runs, key=lambda r: r["Total"])[len(runs) // 2]

    console = Console()
    queries = "1 query" if len(rows) == 1 else f"{len(rows):,} queries"
    console.print(f"{folder.name}, {featurizer}: {queries}, {args.k} hits each")
    console.print("Time from launch to exit, in seconds")
    table = Table()
    table.add_column("Stage")
    for method in methods:
        table.add_column(METHODS[method], justify="right")
    for stage in (*STAGES, "Total"):
        style = "bold" if stage == "Total" else None
        table.add_row(stage, *(f"{shown[m][stage]:.2f}" for m in methods), style=style)
    console.print(table)
    if args.repeats > 1:
        console.print(f"Median of {args.repeats} runs by total.")
    if from_disk:
        console.print("Model and library files were read from disk in each run.")
    else:
        console.print("Could not drop cached files here: loading may be too fast.")
    if "exact" not in methods:
        console.print(
            f"No {featurizer} embeddings in this library, so only the indexed search."
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--run"]:
        RUNNERS[sys.argv[2]](*sys.argv[3:])
    else:
        main()
