"""Downloads from Ersilia's public storage, and the catalog of fetchable libraries."""

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ..utils.logging import console

CATALOG_URL = (
    "https://eosvc-public.s3.eu-central-1.amazonaws.com/"
    "lazy-sim/data/libraries/catalog.json"
)
_BLOCK = 1 << 20


def sha256(path):
    """Return the hex sha256 of the file at `path`."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def catalog_url():
    """Return the catalog URL; ``$LAZYSIM_CATALOG_URL`` overrides the default."""
    return os.environ.get("LAZYSIM_CATALOG_URL") or CATALOG_URL


def read_catalog():
    """Read the catalog of libraries that Ersilia publishes.

    Returns
    -------
    dict
        Library name to catalog entry. Every file in an entry gets a resolved
        ``url`` next to its ``path``, ``sha256`` and ``bytes``.
    """
    url = catalog_url()
    with urllib.request.urlopen(url, timeout=30) as response:
        libraries = json.load(response)["libraries"]
    for entry in libraries.values():
        groups = [entry["files"]]
        for featurizer in entry["featurizers"].values():
            groups += [featurizer["files"], featurizer.get("embeddings", [])]
        for f in (f for files in groups for f in files):
            f["url"] = urllib.parse.urljoin(url, f["path"])
    return libraries


def download(url, dest, expected_sha256, progress=None, task=None):
    """Stream `url` into `dest`, moving it into place only if its sha256 matches."""
    dest = Path(dest)
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=".part-", delete=False
    ) as tmp:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                for block in iter(lambda: response.read(_BLOCK), b""):
                    tmp.write(block)
                    h.update(block)
                    if progress is not None:
                        progress.advance(task, len(block))
        except BaseException:
            os.unlink(tmp.name)
            raise
    if h.hexdigest() != expected_sha256:
        os.unlink(tmp.name)
        raise ValueError(f"{url} failed its checksum; please try again")
    os.replace(tmp.name, dest)


def download_files(files, description):
    """Download ``(url, dest, sha256, bytes)`` items in order, with one progress bar."""
    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task(description, total=sum(f[3] for f in files))
        for url, dest, expected_sha256, _ in files:
            download(url, dest, expected_sha256, progress, task)
