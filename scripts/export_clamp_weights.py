"""Maintainer script: export the CLAMP compound-encoder weights that lazy-sim uses.

lazy-sim runs CLAMP without torch, so the encoder weights are extracted once from the
original JKU checkpoint and hosted by Ersilia. This script needs the original CLAMP
stack (torch, clamp@61575cb, mhnreact@a3e0cc6), which lazy-sim does not depend on.

Optionally, it also embeds a small SMILES file with CLAMP's own ``encode_smiles``,
giving the reference embeddings that the test suite compares lazy-sim against.

Usage
-----
python scripts/export_clamp_weights.py <checkpoint_dir> [<smiles_csv> <out_npy>]

Then upload with ``eosvc upload --path data/clamp``.
"""

import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

CHECKPOINT_SHA256 = "fc41395199dde8b7afce18a2214537c66b46602b636f27eefb22463a1e78eb2a"
HP_SHA256 = "d5430190b2848a53138865bc1335581a9730a72a4debc7953fd47dfc513d02f7"
CLAMP_COMMIT = "61575cb76f4079ccdee13ac8513906af23baf047"
LICENSE_URL = f"https://raw.githubusercontent.com/ml-jku/clamp/{CLAMP_COMMIT}/LICENSE"
OUT_DIR = Path("data/clamp")
WEIGHTS_FILE = "clamp_compound_encoder_v1.npz"

KEYS = {
    "w1": "linear_input.weight",
    "b1": "linear_input.bias",
    "w2": "linear_hidden_l.0.weight",
    "b2": "linear_hidden_l.0.bias",
    "w3": "linear_output.weight",
    "b3": "linear_output.bias",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(checkpoint_dir):
    from clamp.models.pretrained import PretrainedCLAMP

    checkpoint_dir = Path(checkpoint_dir)
    assert sha256(checkpoint_dir / "checkpoint.pt") == CHECKPOINT_SHA256
    assert sha256(checkpoint_dir / "hp.json") == HP_SHA256
    model = PretrainedCLAMP(path_dir=checkpoint_dir, device="cpu")
    model.eval()
    return model


def export_weights(model):
    state = model.compound_encoder.state_dict()
    assert set(state) == set(KEYS.values()), sorted(state)
    arrays = {}
    for name, key in KEYS.items():
        a = state[key].detach().numpy().astype(np.float32)
        arrays[name] = np.ascontiguousarray(a.T) if name.startswith("w") else a
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / WEIGHTS_FILE
    np.savez(path, **arrays)
    with urllib.request.urlopen(LICENSE_URL) as r:
        (OUT_DIR / "LICENSE-CLAMP.txt").write_bytes(r.read())
    print(f"{path}  {path.stat().st_size / 1e6:.1f} MB  sha256 {sha256(path)}")


def export_reference(model, smiles_csv, out_npy):
    import torch

    with open(smiles_csv, newline="") as fh:
        smiles = [row["smiles"] for row in csv.DictReader(fh)]
    with torch.no_grad():
        emb = torch.nn.functional.normalize(model.encode_smiles(smiles), dim=1)
    np.save(out_npy, emb.numpy().astype(np.float32))
    print(f"{out_npy}  {len(smiles)} reference embeddings")


if __name__ == "__main__":
    clamp_model = load_model(sys.argv[1])
    export_weights(clamp_model)
    if len(sys.argv) == 4:
        export_reference(clamp_model, sys.argv[2], sys.argv[3])
