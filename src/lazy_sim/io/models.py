"""Model weights, kept once in ``~/lazysim/models`` and shared by every library."""

import os
from pathlib import Path

from . import home, remote


def ensure(file, url, sha256, size, env_var):
    """Return the path of a weights file, downloading it the first time.

    Parameters
    ----------
    file : str
        File name in ``~/lazysim/models``.
    url : str
        Where to download it from.
    sha256 : str
        Expected checksum, checked after downloading.
    size : int
        Expected size in bytes, for the progress bar.
    env_var : str
        Environment variable that may point to a local copy instead.

    Returns
    -------
    Path
    """
    local = os.environ.get(env_var)
    path = Path(local) if local else home.models_dir() / file
    if not path.exists():
        if local:
            raise FileNotFoundError(f"{env_var} points to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        remote.download_files([(url, path, sha256, size)], file)
    return path


_SAFETENSORS_DTYPES = {
    "F16": "<f2",
    "F32": "<f4",
    "F64": "<f8",
    "I32": "<i4",
    "I64": "<i8",
}


def load_safetensors(path, sha256, prefixes):
    """Load the tensors of a ``.safetensors`` file whose names start with `prefixes`.

    The format is an 8-byte header size, a JSON header and raw little-endian data,
    so it is read with numpy alone. The sha256 is checked first.

    Parameters
    ----------
    path : Path
        The ``.safetensors`` file.
    sha256 : str
        Expected checksum.
    prefixes : tuple of str
        Tensor name prefixes to load; other tensors are skipped.

    Returns
    -------
    dict of str to numpy.ndarray
    """
    import json

    import numpy as np

    if remote.sha256(path) != sha256:
        raise ValueError(f"{path} is not the expected weights file; delete it")
    arrays = {}
    with open(path, "rb") as fh:
        size = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(size))
        for name, info in header.items():
            if name == "__metadata__" or not name.startswith(tuple(prefixes)):
                continue
            begin, end = info["data_offsets"]
            fh.seek(8 + size + begin)
            dtype = np.dtype(_SAFETENSORS_DTYPES[info["dtype"]])
            arrays[name] = np.frombuffer(fh.read(end - begin), dtype).reshape(
                info["shape"]
            )
    return arrays


def load_npz(path, sha256):
    """Load a weights ``.npz`` as a dict of arrays, checking its sha256 first."""
    import numpy as np

    if remote.sha256(path) != sha256:
        raise ValueError(f"{path} is not the expected weights file; delete it")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
