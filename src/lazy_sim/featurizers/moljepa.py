"""Mol-JEPA embeddings (Boehringer Ingelheim), without torch.

From a SMILES alone, Mol-JEPA embeds a molecular graph with a graph transformer
and passes it, with a summary token, through its multimodal transformer; the
summary token is the 512-d embedding. lazy-sim reproduces that path with the
released weights (Hugging Face ``Flogrammer/Mol-JEPA``, revision ``4c912b4``).
How this was checked against the original code is in ``docs/featurizers.md``.

Mol-JEPA's code and weights are licensed CC BY-NC 4.0: non-commercial use only.
"""

import math

import numpy as np
from rdkit import Chem, rdBase

from ..io import models
from ..utils.logging import logger

NAME = "moljepa"
ENGINE = "turbovec"
SIMILARITY = "cosine"
DIM = 512
REVISION = "4c912b450175f31b5ba913a5dc921c03b27b985a"
WEIGHTS_FILE = "moljepa_4c912b4.safetensors"
WEIGHTS_URL = (
    f"https://huggingface.co/Flogrammer/Mol-JEPA/resolve/{REVISION}/model.safetensors"
)
WEIGHTS_SHA256 = "a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14"
WEIGHTS_BYTES = 181_651_424
LICENCE = (
    "Mol-JEPA is licensed CC BY-NC 4.0: non-commercial use only "
    "(https://huggingface.co/Flogrammer/Mol-JEPA)"
)

_GRAPH = "model.encoders.graph."
_HEAD = "model.transformer_head."
_noticed = False


def _index(values):
    return {v: i for i, v in enumerate(values)}


# The graph reproduces molfeat 0.10.1's AdjGraphTransformer(AtomCalculator(),
# EdgeMatCalculator(), explicit_hydrogens=True, canonical_atom_order=True). molfeat
# concatenates its feature blocks in alphabetical order of their names; each block
# below is (first column, value -> position). Values outside a list give zeros.
_CHIRAL = _index(
    [
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_OTHER,
    ]
)
_HYBRID = _index(
    [
        Chem.HybridizationType.UNSPECIFIED,
        Chem.HybridizationType.SP3D2,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP,
        Chem.HybridizationType.S,
    ]
)
_ELEMENTS = _index(
    "C N O S F Si P Cl Br Mg Na Ca Fe As Al I B V K Tl Yb Sb Sn Ag Pd Co Se Ti Zn H "
    "Li Ge Cu Au Ni Cd In Mn Zr Cr Pt Hg Pb".split()
)
_BOND_DIRS = _index(
    [Chem.BondDir.NONE, Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT]
)
_BOND_STEREO = _index(
    [
        Chem.BondStereo.STEREONONE,
        Chem.BondStereo.STEREOANY,
        Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
    ]
)
_BOND_TYPES = _index(
    [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
)

_DEGREES, _VALENCES, _HS = _index(range(11)), _index(range(7)), _index(range(5))


def _one_hot(row, start, positions, value):
    position = positions.get(value)
    if position is not None:
        row[start + position] = 1


def _atom_row(atom):
    row = np.zeros(82, np.int8)
    _one_hot(row, 0, _CHIRAL, atom.GetChiralTag())
    _one_hot(row, 4, _DEGREES, atom.GetDegree())
    row[15] = atom.GetFormalCharge()
    _one_hot(row, 16, _HYBRID, atom.GetHybridization())
    _one_hot(row, 23, _VALENCES, atom.GetValence(Chem.ValenceType.IMPLICIT))
    row[30] = atom.GetIsAromatic()
    row[31] = atom.HasProp("_ChiralityPossible")
    row[32] = atom.IsInRing()
    row[33] = atom.GetNumRadicalElectrons()
    _one_hot(row, 34, _ELEMENTS, atom.GetSymbol())
    _one_hot(row, 77, _HS, atom.GetTotalNumHs())
    return row


def _bond_row(bond, same_ring):
    row = np.zeros(17, np.int8)
    _one_hot(row, 0, _BOND_DIRS, bond.GetBondDir())
    row[3] = bond.GetIsConjugated()
    row[4] = bond.IsInRing()
    _one_hot(row, 5, _BOND_STEREO, bond.GetStereo())
    _one_hot(row, 11, _BOND_TYPES, bond.GetBondType())
    row[15] = 1  # topological distance between bonded atoms
    row[16] = same_ring
    return row


def fingerprint(smiles):
    """Compute the Mol-JEPA input graph of one molecule.

    Hydrogens become atoms and atoms are put in RDKit's canonical order, as in
    molfeat; nothing else is done to the molecule.

    Parameters
    ----------
    smiles : str
        Molecule as SMILES, used exactly as given.

    Returns
    -------
    tuple of numpy.ndarray or None
        Atom features ``(n, 82)``, directed edges ``(2, e)`` sorted by source then
        target, and bond features ``(e, 17)``, as small integers; None if RDKit
        cannot parse the SMILES or it has no atoms.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.AddHs(mol)
    ranks = list(
        Chem.CanonicalRankAtoms(
            mol, breakTies=True, includeChirality=True, includeIsotopes=True
        )
    )
    mol = Chem.RenumberAtoms(mol, sorted(range(len(ranks)), key=ranks.__getitem__))
    atoms = np.stack([_atom_row(a) for a in mol.GetAtoms()])
    rings = [set(ring) for ring in Chem.GetSymmSSSR(mol)]
    edges, bonds = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row = _bond_row(bond, any(a in ring and b in ring for ring in rings))
        edges += [(a, b), (b, a)]
        bonds += [row, row]
    edges = np.array(edges, dtype=np.int64).reshape(-1, 2)
    bonds = np.array(bonds, dtype=np.int8).reshape(-1, 17)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return atoms, edges[order].T.copy(), bonds[order]


def fingerprints(smiles_list):
    """Apply `fingerprint` to a list of SMILES, with RDKit messages silenced."""
    _block = rdBase.BlockLogs()  # noqa: F841 (silences RDKit while alive)
    return [fingerprint(s) for s in smiles_list]


def _notice():
    global _noticed
    if not _noticed:
        logger.warning(LICENCE)
        _noticed = True


def ensure_weights():
    """Return the path of the Mol-JEPA weights, downloading them the first time.

    They are downloaded from Hugging Face, pinned to one revision, into
    ``~/lazysim/models``; ``$LAZYSIM_MOLJEPA_WEIGHTS`` points to a local copy
    instead.

    Returns
    -------
    Path
    """
    _notice()
    return models.ensure(
        WEIGHTS_FILE,
        WEIGHTS_URL,
        WEIGHTS_SHA256,
        WEIGHTS_BYTES,
        "LAZYSIM_MOLJEPA_WEIGHTS",
    )


def load_weights():
    """Load the Mol-JEPA weights used from a SMILES, checking their sha256.

    Returns
    -------
    dict
        The arrays `encode` needs, with linear layers as (input x output).
    """
    names = (_GRAPH, _HEAD, "model.cls_token")
    return prepare_weights(
        models.load_safetensors(ensure_weights(), WEIGHTS_SHA256, names)
    )


def prepare_weights(raw):
    """Arrange the arrays of Mol-JEPA's state dict (torch names) for `encode`."""

    def linear(name, bias=True):
        w = np.ascontiguousarray(raw[f"{name}.weight"].T)
        return w, (raw[f"{name}.bias"] if bias else None)

    def norm(name):
        return raw[f"{name}.weight"], raw[f"{name}.bias"]

    def heads(name):  # (8 * 512, d) -> (8, 512, d)
        return np.ascontiguousarray(raw[name].reshape(8, 512, -1))

    def stacked(name):  # (8 * 512, d) -> (8 * d, 512): head-major rows
        w = raw[name].reshape(8, 512, -1).transpose(0, 2, 1)
        return np.ascontiguousarray(w.reshape(-1, 512))

    convs = []
    for i in range(3):
        conv = f"{_GRAPH}convs.{i}."
        convs.append(
            {
                "query": linear(conv + "lin_query"),
                # Per head: key (head, query channel, input) and edge (head,
                # query channel, bond feature) weights, to fold into the query;
                # value and edge weights stacked to apply after aggregating.
                "key": heads(conv + "lin_key.weight"),
                "edge": heads(conv + "lin_edge.weight"),
                "value_after": stacked(conv + "lin_value.weight"),
                "edge_after": stacked(conv + "lin_edge.weight"),
                "value_bias": raw[conv + "lin_value.bias"].reshape(8, -1),
                "skip": linear(conv + "lin_skip"),
                "norm": norm(f"{_GRAPH}norms.{i}"),
            }
        )
    layers = []
    for i in range(2):
        layer = f"{_HEAD}transformer.layers.{i}."
        layers.append(
            {
                "norm1": norm(layer + "norm1"),
                "in_proj": (
                    np.ascontiguousarray(raw[layer + "self_attn.in_proj_weight"].T),
                    raw[layer + "self_attn.in_proj_bias"],
                ),
                "out_proj": linear(layer + "self_attn.out_proj"),
                "norm2": norm(layer + "norm2"),
                "linear1": linear(layer + "linear1"),
                "linear2": linear(layer + "linear2"),
            }
        )
    positions = raw[f"{_HEAD}modality_pos.weight"]
    return {
        "input": linear(f"{_GRAPH}input_proj"),
        "convs": convs,
        "proj": linear(f"{_GRAPH}proj"),
        "cls": raw["model.cls_token"].reshape(-1) + positions[0],
        "graph_position": positions[1],
        "layers": layers,
        "final_norm": norm(f"{_HEAD}transformer.norm"),
        "out": linear(f"{_HEAD}modality_pred.0"),
    }


def _dense(x, layer):
    w, b = layer
    return x @ w + b if b is not None else x @ w


def _erf(x):
    """Vectorised erf (Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7)."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1 - poly * np.exp(-x * x))


def _gelu(x):
    return 0.5 * x * (1 + _erf(x / math.sqrt(2)))


def _layer_norm(x, norm, eps=1e-5):
    x = x - x.mean(axis=-1, keepdims=True)
    x /= np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    gamma, beta = norm
    return x * gamma + beta


def _incoming_slots(dst, n):
    """Group edges so that each slot holds at most one incoming edge per atom."""
    order = np.argsort(dst, kind="stable")
    counts = np.bincount(dst, minlength=n)
    rank = np.arange(len(dst)) - np.repeat(np.cumsum(counts) - counts, counts)
    return [
        (dst[order[rank == k]], order[rank == k])
        for k in range(int(counts.max(initial=0)))
    ]


def _transformer_conv(x, src, bonds, slots, conv, heads=8):
    """PyG ``TransformerConv`` (heads averaged, bond features in keys and values).

    Rearranged so that per-edge work is on 512-wide inputs and 17 bond features
    instead of 4,096-wide keys and values, with the same result:

    - a head's score ``q_i . (k_j + W_e e_j)`` is ``(q_i W_k) . x_j +
      (q_i W_e) . e_j`` plus ``q_i . b_k``, which is the same for all of atom
      i's edges and so leaves the softmax unchanged;
    - its output ``sum_j a_ij (v_j + W_e e_j)`` is ``(sum_j a_ij x_j) W_v^T +
      (sum_j a_ij e_j) W_e^T + b_v sum_j a_ij``, so the value and edge
      projections are applied once per atom, after aggregating.
    """
    n, width = x.shape
    query = _dense(x, conv["query"]).reshape(n, heads, width)
    by_head = query.transpose(1, 0, 2)  # batched products per head use BLAS
    query_key = np.ascontiguousarray((by_head @ conv["key"]).transpose(1, 0, 2))
    query_edge = np.ascontiguousarray((by_head @ conv["edge"]).transpose(1, 0, 2))
    scores = [
        (
            (query_key[t] @ x[src[e]][:, :, None])[:, :, 0]
            + (query_edge[t] @ bonds[e][:, :, None])[:, :, 0]
        )
        / math.sqrt(width)
        for t, e in slots
    ]
    top = np.full((n, heads), -np.inf, np.float32)  # softmax over each atom's edges
    for (t, _), s in zip(slots, scores):
        top[t] = np.maximum(top[t], s)
    total = np.zeros((n, heads), np.float32)
    weights = []
    for (t, _), s in zip(slots, scores):
        w = np.exp(s - top[t])
        total[t] += w
        weights.append(w)
    inputs = np.zeros((n, heads, width), np.float32)  # sum_j a_ij x_j, per head
    features = np.zeros((n, heads, bonds.shape[1]), np.float32)  # sum_j a_ij e_j
    attended = np.zeros((n, heads), np.float32)  # sum_j a_ij: 1, or 0 without edges
    for (t, e), w in zip(slots, weights):
        alpha = w / (total[t] + 1e-16)
        inputs[t] += alpha[:, :, None] * x[src[e]][:, None, :]
        features[t] += alpha[:, :, None] * bonds[e][:, None, :]
        attended[t] += alpha
    out = (
        inputs.reshape(n, -1) @ conv["value_after"]
        + features.reshape(n, -1) @ conv["edge_after"]
        + attended @ conv["value_bias"]
    ) / heads
    return out + _dense(x, conv["skip"])


def _attention(x, layer, heads=4):
    """Self-attention of PyTorch's ``MultiheadAttention`` over the tokens of `x`."""
    b, tokens, width = x.shape
    size = width // heads
    qkv = _dense(x, layer["in_proj"]).reshape(b, tokens, 3, heads, size)
    q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(size)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores /= scores.sum(axis=-1, keepdims=True)
    out = (scores @ v).transpose(0, 2, 1, 3).reshape(b, tokens, width)
    return _dense(out, layer["out_proj"])


def _forward(w, graphs):
    sizes = np.array([len(g[0]) for g in graphs])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    atoms = np.concatenate([g[0] for g in graphs]).astype(np.float32)
    src, dst = np.concatenate([g[1] + s for g, s in zip(graphs, starts)], axis=1)
    bonds = np.concatenate([g[2] for g in graphs]).astype(np.float32)
    slots = _incoming_slots(dst, len(atoms))

    x = _dense(atoms, w["input"])
    for conv in w["convs"]:
        x = _layer_norm(
            _gelu(_transformer_conv(x, src, bonds, slots, conv)) + x, conv["norm"]
        )
    graph = _dense(np.maximum.reduceat(x, starts, axis=0), w["proj"])

    # The summary token sees only itself and the graph: the other modalities are
    # missing from a SMILES, so attention ignores their positions.
    tokens = np.stack(
        [np.broadcast_to(w["cls"], graph.shape), graph + w["graph_position"]], axis=1
    )
    for layer in w["layers"]:
        tokens = tokens + _attention(_layer_norm(tokens, layer["norm1"]), layer)
        hidden = np.maximum(
            _dense(_layer_norm(tokens, layer["norm2"]), layer["linear1"]), 0
        )
        tokens = tokens + _dense(hidden, layer["linear2"])
    tokens = _layer_norm(tokens, w["final_norm"])
    y = _dense(tokens[:, 0], w["out"])
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def encode(weights, graphs, batch=256):
    """Embed molecular graphs with the Mol-JEPA encoder, L2-normalised.

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
        float32 embeddings of shape ``(len(graphs), DIM)``. Mol-JEPA's own output
        is not normalised; lazy-sim normalises it so cosine is a dot product.
    """
    out = np.empty((len(graphs), DIM), np.float32)
    for start in range(0, len(graphs), batch):
        part = graphs[start : start + batch]
        out[start : start + len(part)] = _forward(weights, part)
    return out
