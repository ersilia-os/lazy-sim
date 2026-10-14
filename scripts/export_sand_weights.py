"""Maintainer script: export the SAND encoder weights that lazy-sim uses.

lazy-sim runs SAND without torch, so the encoder weights are extracted once from
Pfizer's released checkpoint (``sand_no_comp.ckpt``, SAND v1.0.0) and hosted by
Ersilia. Extracting them needs only torch.

Optionally, it also embeds a small SMILES file with SAND's own code
(``SANDModel.encode_smiles``), giving the reference embeddings that the test suite
compares lazy-sim against. That needs the SAND stack (torch 2.4.1, torch_geometric
2.6.1, lightning 2.4.0, jsonargparse 4.50.0, rdkit 2024.3.5) and a folder holding
the ``sand`` package, such as ``model/framework/code`` of ersilia-os/eos5mnx at
commit 711499a.

Usage
-----
python scripts/export_sand_weights.py <checkpoint> [<sand_code> <smiles_csv> <out_npy>]

Then upload with ``eosvc upload --path data/sand``.
"""

import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

CHECKPOINT_URL = (
    "https://github.com/pfizer-opensource/SAND/releases/download/v1.0.0/"
    "sand_no_comp.ckpt"
)
CHECKPOINT_SHA256 = "dd022d49edce1e9de99c3927b8455a5813e12bb2c112799a54b754dc7fbb70fb"
SAND_COMMIT = "5d753d3a42758550f1d0f5aaf24317a26e25fbc7"
LICENSE_URL = (
    f"https://raw.githubusercontent.com/pfizer-opensource/SAND/{SAND_COMMIT}/LICENSE"
)
PREFIX = "backbone.encoder.model."
OUT_DIR = Path("data/sand")
WEIGHTS_FILE = "sand_encoder_v1.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_weights(checkpoint):
    import torch

    assert sha256(checkpoint) == CHECKPOINT_SHA256, f"not {CHECKPOINT_URL}"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    arrays = {
        key[len(PREFIX) :]: value.numpy().astype(np.float32)
        for key, value in state.items()
        if key.startswith(PREFIX)
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / WEIGHTS_FILE
    np.savez(path, **arrays)
    with urllib.request.urlopen(LICENSE_URL) as r:
        (OUT_DIR / "LICENSE-SAND.txt").write_bytes(r.read())
    params = sum(a.size for a in arrays.values())
    size = path.stat().st_size
    print(f"{path}  {len(arrays)} arrays, {params:,} parameters, {size:,} bytes")
    print(f"sha256 {sha256(path)}")


def export_reference(checkpoint, sand_code, smiles_csv, out_npy):
    import torch

    sys.path.insert(0, str(sand_code))
    from sand.model import SANDModel

    config = Path(sand_code) / "sand" / "config" / "no_compression.yml"
    model = SANDModel.load_from_checkpoint(checkpoint, str(config), map_location="cpu")
    model.eval()
    with open(smiles_csv, newline="") as fh:
        smiles = [row["smiles"] for row in csv.DictReader(fh)]
    with torch.no_grad():
        emb = model.encode_smiles(smiles, batch_size=256)
    np.save(out_npy, emb.numpy().astype(np.float32))
    print(f"{out_npy}  {len(smiles)} reference embeddings")


if __name__ == "__main__":
    export_weights(sys.argv[1])
    if len(sys.argv) == 5:
        export_reference(sys.argv[1], *sys.argv[2:])
