"""SAND shape-aware embeddings (Winter et al., Pfizer), without torch.

The graph reproduces SAND's ``RDKitGraphFeaturizer`` and the encoder reproduces its
GINE network with the released weights (``sand_no_comp.ckpt``, v1.0.0). Cosine
similarity between the embeddings approximates 3D shape overlap. How this was
checked against the original code is in ``docs/featurizers.md``.

SAND is licensed under Apache-2.0 (https://github.com/pfizer-opensource/SAND).
"""

import numpy as np
from rdkit import Chem, rdBase

from ..io import models

NAME = "sand"
ENGINE = "turbovec"
SIMILARITY = "cosine"
DIM = 512
LAYERS = 7
WEIGHTS_FILE = "sand_encoder_v1.npz"
WEIGHTS_URL = (
    "https://eosvc-public.s3.eu-central-1.amazonaws.com/lazy-sim/data/sand/"
    + WEIGHTS_FILE
)
WEIGHTS_SHA256 = "ee72a20915dc2ba44cbce0812fa34868416d38aeebb88ffb379f38a2f10d4501"
WEIGHTS_BYTES = 111_687_496

# Atom and bond categories, in SAND's column order; values outside a list fall
# into one extra bucket per column.
_ELEMENTS = [1, 6, 7, 8, 9, 14, 15, 16, 17, 26, 35, 53]
_HYBRIDIZATION = {
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3,
    Chem.HybridizationType.SP3D: 4,
    Chem.HybridizationType.SP3D2: 5,
}
_RING_SIZES = range(3, 11)
_BOND_ORDERS = [1, 1.5, 2, 3, 4]
_BOND_STEREO = {
    Chem.BondStereo.STEREOE: 1,
    Chem.BondStereo.STEREOTRANS: 1,
    Chem.BondStereo.STEREOZ: 2,
    Chem.BondStereo.STEREOCIS: 2,
    Chem.BondStereo.STEREOANY: 3,
}
# Offsets make each column's categories distinct rows of one embedding table.
_ATOM_OFFSETS = np.array(
    [0, 13, 19, 26, 32, 34, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]
)
_BOND_OFFSETS = np.array([0, 6, 8, 10, 12])


def _bucket(value, values, other):
    return values.index(value) if value in values else other


def _atom_row(atom):
    charge = atom.GetFormalCharge()
    if atom.HasProp("_CIPCode"):
        cip = {"S": 1, "R": 2}.get(atom.GetProp("_CIPCode"), 3)
    else:
        cip = 3 if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0
    return [
        _bucket(atom.GetAtomicNum(), _ELEMENTS, 12),
        _bucket(atom.GetDegree(), [0, 1, 2, 3, 4], 5),
        _bucket(charge, [-2, -1, 0, 1, 2], 5 if charge < 0 else 6),
        _bucket(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], 5),
        int(atom.GetIsAromatic()),
        _HYBRIDIZATION.get(atom.GetHybridization(), 0),
        int(atom.IsInRing()),
        *(int(atom.IsInRingSize(size)) for size in _RING_SIZES),
        cip,
    ]


def _bond_row(bond):
    if bond.GetIsAromatic():
        order = 1.5
    else:
        order = bond.GetBondTypeAsDouble()
        if order == int(order):
            order = int(order)
    rotor = (
        bond.GetBondType() == Chem.BondType.SINGLE
        and not bond.IsInRing()
        and bond.GetBeginAtom().GetDegree() > 1
        and bond.GetEndAtom().GetDegree() > 1
    )
    return [
        _bucket(order, _BOND_ORDERS, 5),
        int(bond.IsInRing()),
        int(bond.GetIsAromatic()),
        int(rotor),
        _BOND_STEREO.get(bond.GetStereo(), 0),
    ]


def fingerprint(smiles):
    """Compute the SAND input graph of one molecule.

    Parameters
    ----------
    smiles : str
        Molecule as SMILES, used exactly as given.

    Returns
    -------
    tuple of numpy.ndarray or None
        Atom categories ``(n, 16)``, directed edges ``(2, e)`` sorted by source
        then target, and bond categories ``(e, 5)``; None if RDKit cannot parse
        the SMILES or it has no atoms.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    n = mol.GetNumAtoms()
    atoms = np.array([_atom_row(a) for a in mol.GetAtoms()]) + _ATOM_OFFSETS
    pairs, rows = [], []
    for bond in mol.GetBonds():
        a, b, row = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _bond_row(bond)
        pairs += [(a, b), (b, a)]
        rows += [row, row]
    edges = np.array(pairs, dtype=np.int32).reshape(-1, 2)
    bonds = (np.array(rows).reshape(-1, 5) + _BOND_OFFSETS).astype(np.uint8)
    order = np.argsort(edges[:, 0] * n + edges[:, 1], kind="stable")
    return atoms.astype(np.uint8), edges[order].T.copy(), bonds[order]


def fingerprints(smiles_list):
    """Apply `fingerprint` to a list of SMILES, with RDKit messages silenced."""
    _block = rdBase.BlockLogs()  # noqa: F841 (silences RDKit while alive)
    return [fingerprint(s) for s in smiles_list]


def ensure_weights():
    """Return the path of the SAND weights, downloading them the first time.

    They live once in ``~/lazysim/models``; ``$LAZYSIM_SAND_WEIGHTS`` points to a
    local copy instead.

    Returns
    -------
    Path
    """
    return models.ensure(
        WEIGHTS_FILE, WEIGHTS_URL, WEIGHTS_SHA256, WEIGHTS_BYTES, "LAZYSIM_SAND_WEIGHTS"
    )


def load_weights():
    """Load the SAND encoder weights, checking their sha256.

    Returns
    -------
    dict
        The arrays `encode` needs, with linear layers as (input x output).
    """
    return prepare_weights(models.load_npz(ensure_weights(), WEIGHTS_SHA256))


def prepare_weights(raw):
    """Arrange the arrays of SAND's state dict (torch names) for `encode`."""

    def linear(name):
        return np.ascontiguousarray(raw[f"{name}.weight"].T), raw[f"{name}.bias"]

    def norm(name):
        return raw[f"{name}.weight"], raw[f"{name}.bias"]

    gnn = "gnn.conv_layers.{}.nn.{}"
    return {
        "atom": raw["atom_map.embedding.weight"],
        "bond": raw["edge_map.embedding.weight"],
        "layers": [
            {
                "eps": float(raw[f"gnn.conv_layers.{i}.eps"][0]),
                "mlp": [
                    linear(gnn.format(i, part))
                    for part in ("in_layer", "hidden_layers.0", "out_layer")
                ],
                "norm": norm(f"gnn.norm_layers.{i}"),
            }
            for i in range(LAYERS)
        ],
        "gate": [linear("aggregation.gate_net.0"), linear("aggregation.gate_net.2")],
        "node": [linear("aggregation.node_net.0"), linear("aggregation.node_net.2")],
        "head": [linear("mlp.in_layer"), linear("mlp.hidden_layers.0")],
        "head_norms": [norm("mlp.norm_layers.0"), norm("mlp.norm_layers.1")],
        "out": linear("mlp.out_layer"),
    }


def _silu(x):
    with np.errstate(over="ignore"):  # exp overflows to inf for x << 0: silu -> 0
        return x / (1 + np.exp(-x))


def _dense(x, layer):
    w, b = layer
    return x @ w + b


def _per_molecule(values, starts):
    """Sum consecutive rows of `values` per molecule (molecules start at `starts`)."""
    return np.add.reduceat(values, starts, axis=0)


def _graph_norm(z, starts, sizes, norm, eps=1e-5):
    """LayerNorm over all atoms and channels of each molecule (PyG ``mode="graph"``)."""
    count = sizes * z.shape[1]
    mean = _per_molecule(z.sum(axis=1), starts) / count
    z = z - np.repeat(mean, sizes)[:, None]
    var = _per_molecule((z * z).sum(axis=1), starts) / count
    z /= np.sqrt(np.repeat(var, sizes) + eps)[:, None]
    gamma, beta = norm
    return z * gamma + beta


def _layer_norm(y, norm, eps=1e-5):
    y = y - y.mean(axis=1, keepdims=True)
    y /= np.sqrt((y * y).mean(axis=1, keepdims=True) + eps)
    gamma, beta = norm
    return y * gamma + beta


def _incoming_slots(dst, n):
    """Group edges so that each slot holds at most one incoming edge per atom.

    Summing messages slot by slot is much faster in numpy than a scatter-add,
    since an atom has only a few bonds.
    """
    order = np.argsort(dst, kind="stable")
    counts = np.bincount(dst, minlength=n)
    rank = np.arange(len(dst)) - np.repeat(np.cumsum(counts) - counts, counts)
    return [
        (dst[order[rank == k]], order[rank == k])
        for k in range(int(counts.max(initial=0)))
    ]


def _forward(w, graphs):
    sizes = np.array([len(g[0]) for g in graphs])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n = int(sizes.sum())
    atoms = np.concatenate([g[0] for g in graphs])
    src, dst = np.concatenate([g[1] + s for g, s in zip(graphs, starts)], axis=1)
    bonds = np.concatenate([g[2] for g in graphs])
    slots = _incoming_slots(dst, n)

    h = sum(w["atom"][atoms[:, c]] for c in range(atoms.shape[1]))
    e = sum(w["bond"][bonds[:, c]] for c in range(bonds.shape[1]))
    for layer in w["layers"]:
        z = (1 + layer["eps"]) * h
        for targets, edges in slots:  # each target appears once per slot
            z[targets] += np.maximum(h[src[edges]] + e[edges], 0)
        for dense in layer["mlp"]:
            z = _silu(_dense(z, dense))
        h = _graph_norm(z, starts, sizes, layer["norm"]) + h

    # Softmax attention over each molecule's atoms, then a weighted sum.
    gate = _dense(_silu(_dense(h, w["gate"][0])), w["gate"][1])[:, 0]
    gate = np.exp(gate - np.repeat(np.maximum.reduceat(gate, starts), sizes))
    gate /= np.repeat(_per_molecule(gate, starts) + 1e-16, sizes)
    values = _dense(_silu(_dense(h, w["node"][0])), w["node"][1])
    y = _per_molecule(gate[:, None] * values, starts)

    for dense, norm in zip(w["head"], w["head_norms"]):
        y = _silu(_layer_norm(_dense(y, dense), norm))
    y = _dense(y, w["out"])
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def encode(weights, graphs, batch=256):
    """Embed molecular graphs with the SAND encoder.

    Parameters
    ----------
    weights : dict
        Output of `load_weights`.
    graphs : list of tuple
        Non-None outputs of `fingerprint`.
    batch : int
        Molecules per forward pass.

    Returns
    -------
    numpy.ndarray
        L2-normalised float32 embeddings, shape ``(len(graphs), DIM)``.
    """
    out = np.empty((len(graphs), DIM), np.float32)
    for start in range(0, len(graphs), batch):
        part = graphs[start : start + batch]
        out[start : start + len(part)] = _forward(weights, part)
    return out
