"""CLAMP compound embeddings (Seidl et al., ICML 2023), without torch.

The fingerprint reproduces ``mhnreact.molutils.getFingerprint(which="morganc+rdkc",
fp_size=8192)``, quirks included, and the encoder reproduces CLAMP's compound MLP
with the original JKU weights. Both were checked against the original code:
identical fingerprints, embeddings within 1e-6.

CLAMP model terms: the model and its derivatives must not be used to exploit or
harm, or to attempt to exploit or harm, in any way.
"""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings

from ..io import models

NAME = "clamp"
ENGINE = "turbovec"
SIMILARITY = "cosine"
DIM = 768
FP_SIZE = 8192
WEIGHTS_FILE = "clamp_compound_encoder_v1.npz"
WEIGHTS_URL = (
    "https://eosvc-public.s3.eu-central-1.amazonaws.com/lazy-sim/data/clamp/"
    + WEIGHTS_FILE
)
WEIGHTS_SHA256 = "9ddd688c60af683fb1c70cc0610464e12446c05f9d04869420f338c0e80ee18b"
WEIGHTS_BYTES = 174_092_702


def _fold(counts):
    """Fold sparse counts into FP_SIZE slots; on collision the last key wins."""
    n = len(counts)
    keys = np.fromiter(counts.keys(), dtype=np.uint64, count=n)
    idx = (keys % FP_SIZE).astype(np.int64)
    vals = np.fromiter(counts.values(), dtype=np.float64, count=n)
    slots, first = np.unique(idx[::-1], return_index=True)
    out = np.zeros(FP_SIZE)
    out[slots] = vals[::-1][first]
    return out


def _morgan_counts(mol):
    fp = AllChem.GetMorganFingerprint(
        mol, 2, useChirality=True, useBondTypes=True, useFeatures=True, useCounts=True
    )
    return fp.GetNonzeroElements()


def _rdkit_counts(mol):
    return AllChem.UnfoldedRDKFingerprintCountBased(mol, maxPath=6).GetNonzeroElements()


def fingerprint(smiles):
    """Compute the CLAMP input fingerprint of one molecule.

    Parameters
    ----------
    smiles : str
        Molecule as SMILES, used exactly as given.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray) or None
        Indices (int16) and values (float32) of the non-zero entries of the
        8192-d ``log1p`` count fingerprint, or None if RDKit cannot parse and
        sanitise the SMILES, or the fingerprint is empty.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
    if mol is None:
        return None
    if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    FastFindRings(mol)
    mol.UpdatePropertyCache(strict=False)
    total = np.zeros(FP_SIZE)
    for counts_of in (_morgan_counts, _rdkit_counts):
        try:
            counts = counts_of(mol)
        except Exception:
            continue
        if counts:
            total += _fold(counts)
    fp = np.log1p(total).astype(np.float32)
    nz = np.flatnonzero(fp)
    if nz.size == 0:
        return None
    return nz.astype(np.int16), fp[nz]


def fingerprints(smiles_list):
    """Apply `fingerprint` to a list of SMILES, with RDKit messages silenced."""
    _block = rdBase.BlockLogs()  # noqa: F841 (silences RDKit while alive)
    return [fingerprint(s) for s in smiles_list]


def ensure_weights():
    """Return the path of the CLAMP weights, downloading them the first time.

    They live once in ``~/lazysim/models``, shared by every library;
    ``$LAZYSIM_CLAMP_WEIGHTS`` points to a local copy instead.

    Returns
    -------
    Path
    """
    return models.ensure(
        WEIGHTS_FILE,
        WEIGHTS_URL,
        WEIGHTS_SHA256,
        WEIGHTS_BYTES,
        "LAZYSIM_CLAMP_WEIGHTS",
    )


def load_weights():
    """Load the CLAMP compound-encoder weights, checking their sha256.

    Returns
    -------
    dict of str to numpy.ndarray
        Weights ``w1, w2, w3`` (input x output) and biases ``b1, b2, b3``.
    """
    return models.load_npz(ensure_weights(), WEIGHTS_SHA256)


def _layernorm(h, eps=1e-5):
    mu = h.mean(axis=1, keepdims=True)
    var = h.var(axis=1, keepdims=True)
    return (h - mu) / np.sqrt(var + eps)


def encode(weights, fps, batch=4096):
    """Embed fingerprints with the CLAMP compound encoder.

    Parameters
    ----------
    weights : dict of str to numpy.ndarray
        Output of `load_weights`.
    fps : list of tuple
        Non-None outputs of `fingerprint`.
    batch : int
        Rows densified at a time.

    Returns
    -------
    numpy.ndarray
        L2-normalised float32 embeddings, shape ``(len(fps), DIM)``.
    """
    out = np.empty((len(fps), DIM), np.float32)
    for start in range(0, len(fps), batch):
        part = fps[start : start + batch]
        X = np.zeros((len(part), FP_SIZE), np.float32)
        for i, (nz, vals) in enumerate(part):
            X[i, nz] = vals
        h = np.maximum(_layernorm(X @ weights["w1"] + weights["b1"]), 0)
        h = np.maximum(_layernorm(h @ weights["w2"] + weights["b2"]), 0)
        e = h @ weights["w3"] + weights["b3"]
        out[start : start + len(part)] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return out
