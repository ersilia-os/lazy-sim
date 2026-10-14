"""Thread settings and a process pool that works one chunk ahead of its caller."""

import multiprocessing as mp
import os

_THREAD_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def set_threads(threads):
    """Cap turbovec and BLAS threads; must run before numpy and turbovec start.

    It also lets OpenBLAS threads sleep as soon as a matrix product ends, instead of
    spinning for about 0.1 s and slowing the single-core numpy steps in between
    (SAND embeds ~20% faster, CLAMP indexes ~6% faster). A value already set in the
    environment is kept.

    Parameters
    ----------
    threads : int or None
        Thread count; None leaves the defaults (all logical CPUs).
    """
    os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "4")
    if threads:
        for var in _THREAD_VARS:
            os.environ[var] = str(threads)


class ChunkPool:
    """Map a function over SMILES in worker processes, one chunk ahead of the caller.

    Parameters
    ----------
    func : callable
        Module-level function mapping a list of SMILES to a list of results.
    processes : int
        Worker processes; 1 computes in the calling process.
    """

    BATCH = 1_000

    def __init__(self, func, processes):
        self.func = func
        self.processes = processes
        self.pool = None

    def __enter__(self):
        if self.processes > 1:
            self.pool = mp.get_context("spawn").Pool(self.processes)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()

    def _submit(self, smiles):
        if self.pool is None:
            return self.func(smiles)
        batches = [
            smiles[i : i + self.BATCH] for i in range(0, len(smiles), self.BATCH)
        ]
        return self.pool.map_async(self.func, batches)

    def _collect(self, pending):
        if self.pool is None:
            return pending
        return [fp for batch in pending.get() for fp in batch]

    def map(self, smiles):
        """Return `func` applied to `smiles`, in order."""
        return self._collect(self._submit(list(smiles)))

    def iter_chunks(self, chunks):
        """Yield ``(chunk, results)`` for chunks of ``(key, smiles)`` rows.

        The next chunk is read and sent to the workers before the current one
        is handed back, so the workers overlap with the caller's work.
        """
        chunks = iter(chunks)
        current = next(chunks, None)
        if current is None:
            return
        pending = self._submit([s for _, s in current])
        for upcoming in chunks:
            upcoming_pending = self._submit([s for _, s in upcoming])
            yield current, self._collect(pending)
            current, pending = upcoming, upcoming_pending
        yield current, self._collect(pending)
