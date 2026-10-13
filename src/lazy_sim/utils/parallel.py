"""Thread settings and a process pool that fingerprints one chunk ahead."""

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

    Parameters
    ----------
    threads : int or None
        Thread count; None leaves the defaults (all logical CPUs).
    """
    if threads:
        for var in _THREAD_VARS:
            os.environ[var] = str(threads)


class FingerprintPool:
    """Compute fingerprints in worker processes, one chunk ahead of the caller.

    Parameters
    ----------
    func : callable
        Module-level function mapping a list of SMILES to a list of fingerprints.
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
        """Return the fingerprints of `smiles`, in order."""
        return self._collect(self._submit(list(smiles)))

    def iter_chunks(self, chunks):
        """Yield ``(chunk, fingerprints)`` for chunks of ``(id, smiles)`` rows.

        The next chunk is read and sent to the workers before the current one
        is handed back, so fingerprinting overlaps with the caller's work.
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
