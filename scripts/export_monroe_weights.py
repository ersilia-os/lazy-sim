"""Maintainer script: export the Monroe encoder weights that lazy-sim uses.

lazy-sim runs Monroe without torch, so the encoder weights are extracted once from
the released checkpoint (``checkpoint/weights.pt`` of blazejba/Monroe at commit
57238ed, which ersilia-os/eos8gop stores with eosvc) and hosted by Ersilia.
Extracting them needs only torch.

Optionally, it also embeds a small SMILES file with Monroe's own code as served by
eos8gop (``monroe_predict.predict``: InChI, seeded ETKDGv3 conformer, GRIT
encoder), giving the reference embeddings that the test suite compares lazy-sim
against. That needs eos8gop's environment (torch 2.14.0+cpu, torch_geometric
2.8.0.post1, rdkit 2026.3.6, datamol 0.12.5) and its ``model/framework/code``.

Usage
-----
python scripts/export_monroe_weights.py <weights.pt> [<eos8gop_code> <smiles_csv> <out_npy>]

Then upload with ``eosvc upload --path data/monroe``.
"""

import csv
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

CHECKPOINT_SHA256 = "df996e8e98b12a3cda5fe5acbef3e1e9a4ea31946d3736461cd07e53a5bf205a"
MONROE_COMMIT = "57238edfffea03808abe761a00cd9a75fa41bb95"
RAW = f"https://raw.githubusercontent.com/blazejba/Monroe/{MONROE_COMMIT}"
PREFIX = "encoder."
OUT_DIR = Path("data/monroe")
WEIGHTS_FILE = "monroe_encoder_v1.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_weights(checkpoint):
    import torch

    assert sha256(checkpoint) == CHECKPOINT_SHA256, "not Monroe's released weights"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arrays = {
        key[len(PREFIX) :].replace("._orig_mod.", "."): value.numpy().astype(np.float32)
        for key, value in state.items()
        if key.startswith(PREFIX) and not key.endswith("num_batches_tracked")
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / WEIGHTS_FILE
    np.savez(path, **arrays)
    for name in ("LICENSE", "NOTICE"):  # Monroe's MIT licence and GRIT's notice
        with urllib.request.urlopen(f"{RAW}/{name}") as r:
            (OUT_DIR / f"{name}-MONROE.txt").write_bytes(r.read())
    params = sum(a.size for a in arrays.values())
    size = path.stat().st_size
    print(f"{path}  {len(arrays)} arrays, {params:,} values, {size:,} bytes")
    print(f"sha256 {sha256(path)}")


def export_reference(checkpoint, code, smiles_csv, out_npy):
    os.environ["MONROE_CHECKPOINT_DIR"] = str(Path(checkpoint).parent)
    sys.path.insert(0, str(code))
    from monroe_predict import predict

    with open(smiles_csv, newline="") as fh:
        smiles = [row["smiles"] for row in csv.DictReader(fh)]
    emb = predict(smiles)
    assert not np.isnan(emb).any(), "Monroe could not embed every reference molecule"
    np.save(out_npy, emb.astype(np.float32))
    print(f"{out_npy}  {len(smiles)} reference embeddings")


if __name__ == "__main__":
    export_weights(sys.argv[1])
    if len(sys.argv) == 5:
        export_reference(sys.argv[1], *sys.argv[2:])
