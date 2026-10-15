"""Maintainer script: embed a SMILES file with the original Mol-JEPA, for the tests.

lazy-sim reads Mol-JEPA's weights directly from Hugging Face (``model.safetensors``,
revision 4c912b4), so unlike CLAMP, SAND and Monroe there are no weights to export.
This script writes the reference embeddings that the test suite compares lazy-sim
against (``tests/data/moljepa_reference.npy``), with Mol-JEPA's own code.

It needs eos9q2i's environment (torch 2.8.0, transformers 4.50.3, molfeat 0.10.1,
torch-geometric 2.7.0, rdkit 2026.3.4) and a snapshot of ``Flogrammer/Mol-JEPA`` at
revision 4c912b4, such as ``model/checkpoints/mol-jepa`` of ersilia-os/eos9q2i
(stored with eosvc). The model runs offline from that snapshot.

Usage
-----
python scripts/export_moljepa_reference.py <snapshot_dir> <smiles_csv> <out_npy>
"""

import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

WEIGHTS_SHA256 = "a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_reference(snapshot, smiles_csv, out_npy):
    weights = Path(snapshot) / "model.safetensors"
    assert sha256(weights) == WEIGHTS_SHA256, "not the snapshot lazy-sim pins"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    model.eval()
    with open(smiles_csv, newline="") as fh:
        smiles = [row["smiles"] for row in csv.DictReader(fh)]
    with torch.no_grad():
        emb = model(smiles).cls.numpy()
    np.save(out_npy, emb.astype(np.float32))
    print(f"{out_npy}  {len(smiles)} reference embeddings (not normalised)")


if __name__ == "__main__":
    export_reference(*sys.argv[1:4])
