"""Monroe embeddings (Banaszewski & Fitzgibbon), without torch.

Monroe is a GRIT graph transformer pretrained on quantum-chemical and bioassay data.
lazy-sim reproduces the encoder that Ersilia serves as eos8gop: the molecule is
rebuilt from its InChI, given hydrogens and a seeded 3D conformer, and embedded
into 720 numbers. How this was checked against the original code is in
``docs/featurizers.md``.

Monroe (code and weights) is licensed MIT, Copyright (c) 2026 Blazej Banaszewski;
its attention layers are adapted from GRIT (MIT, Copyright (c) 2023 Liheng Ma).
The licence and notice travel with the weights.
"""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from ..io import models
from ._monroe_elements import ELEMENTS

NAME = "monroe"
ENGINE = "turbovec"
SIMILARITY = "cosine"
DIM = 720
HEADS = 10
WALK = 16  # random-walk steps
WEIGHTS_FILE = "monroe_encoder_v1.npz"
WEIGHTS_URL = (
    "https://eosvc-public.s3.eu-central-1.amazonaws.com/lazy-sim/data/monroe/"
    + WEIGHTS_FILE
)
WEIGHTS_SHA256 = "39fffb2a21f44d72be6c1ca3b37fd24ad0300e3a13c04d5b564d62a97d1a02ba"
WEIGHTS_BYTES = 234_838_658
CONFORMER_SEED = 42  # eos8gop's seed, which makes the conformer reproducible
# A conformer takes ~0.2 s, so queries use worker processes from 100 of them (the
# engine's default waits for 5,000).
POOL_MIN_QUERIES = 100

_PTABLE = Chem.GetPeriodicTable()
_H = Chem.HybridizationType
# Categorical features in Monroe's order; a value outside a list gets code len(list).
_NODE_VOCABS = (
    range(13),  # implicit valence
    range(13),  # total valence
    [_H.UNSPECIFIED, _H.SP3D2, _H.SP3D, _H.SP3, _H.SP2D, _H.SP2, _H.SP, _H.S],
    "C N O S F Si P Cl Br Mg Na Ca Fe As Al I B V K Tl Yb Sb Sn Ag Pd Co Se Ti Zn H "
    "Li Ge Cu Au Ni Cd In Mn Zr Cr Pt Hg Pb".split(),
    range(9),  # total hydrogens
    [1],  # aromatic
    [1],  # in ring
    ["R", "S"],  # CIP label
    range(1, 8),  # period
    range(13),  # degree
    range(1, 20),  # group (19: none)
)
_BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
_STEREO = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
    "chi_star",
    "chi_circle",
]
_EDGE_VOCABS = ([1], [1], _BOND_TYPES, _STEREO)
_OTHER_BOND = len(_BOND_TYPES)  # bond-type code of stereo and virtual-node edges
_MISSING = (1, 2, 4)  # float features where 0 means "unknown" in the table


def _code(value, values):
    values = list(values)
    return values.index(value) if value in values else len(values)


def _molecule(smiles):
    """Monroe's molecule: rebuilt from the InChI, or from the SMILES if that fails."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    inchi = Chem.MolToInchi(mol)
    rebuilt = Chem.MolFromInchi(inchi, sanitize=False) if inchi else None
    if rebuilt is not None:
        try:
            Chem.SanitizeMol(rebuilt)
            mol = rebuilt
        except Exception:
            pass
    mol = Chem.AddHs(mol, addCoords=False)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def _conformer(mol):
    """Seeded ETKDGv3 conformer, MMFF94s- or UFF-optimised; 2D if that fails."""
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = CONFORMER_SEED
    params.enforceChirality = True
    params.useRandomCoords = True
    params.numThreads = 1
    params.maxIterations = 500
    mol.RemoveAllConformers()
    try:
        ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=1, params=params)
        if len(ids) == 0:
            raise ValueError("no conformer")
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, "MMFF94s")
        if props is not None:
            field = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props)
            rdForceFieldHelpers.OptimizeMoleculeConfs(
                mol, field, maxIters=200, numThreads=1
            )
        elif rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                mol, maxIters=200, numThreads=1
            )
        else:
            raise ValueError("no force field")
        conformer = mol.GetConformer(int(ids[0]))
    except Exception:
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformer(0)
    return np.asarray(conformer.GetPositions(), dtype=np.float32)


def _atoms(mol):
    n = mol.GetNumAtoms()
    floats = np.zeros((n, 8), np.float32)
    codes = np.zeros((n, len(_NODE_VOCABS)), np.uint8)
    for i, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        electronegativity, ionisation, melting, period, group = ELEMENTS[z - 1]
        floats[i] = [
            atom.GetNumRadicalElectrons(),
            electronegativity,
            ionisation,
            _PTABLE.GetRcovalent(z),
            melting,
            atom.GetFormalCharge(),
            _PTABLE.GetRvdw(z),
            atom.GetMass(),
        ]
        implicit = atom.GetValence(Chem.ValenceType.IMPLICIT)
        total = implicit + atom.GetValence(Chem.ValenceType.EXPLICIT)
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        values = (
            implicit,
            total,
            atom.GetHybridization(),
            atom.GetSymbol(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            cip,
            period,
            atom.GetDegree(),
            group,
        )
        codes[i] = [_code(v, vocab) for v, vocab in zip(values, _NODE_VOCABS)]
    return floats, codes


def _stereo_edges(mol):
    """Monroe's auxiliary E/Z and R/S edges, in its order and without duplicates."""
    bonded = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded |= {(a, b), (b, a)}
    pairs, codes, seen = [], [], set()

    def add(u, v, stereo, both=True):
        key = (u, v, stereo if isinstance(stereo, str) else int(stereo))
        if (u, v) in bonded or key in seen:
            return
        seen.add(key)
        row = [1, 1, _OTHER_BOND, _code(stereo, _STEREO)]
        pairs.append((u, v))
        codes.append(row)
        if both:
            pairs.append((v, u))
            codes.append(row)

    for bond in mol.GetBonds():  # around E/Z double bonds
        stereo = bond.GetStereo()
        if bond.GetBondType() != Chem.BondType.DOUBLE or stereo not in (
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOZ,
        ):
            continue
        s3, s4 = bond.GetStereoAtoms()
        if s3 < 0 or s4 < 0:
            continue
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        n5 = [a.GetIdx() for a in bond.GetBeginAtom().GetNeighbors()]
        n5 = [i for i in n5 if i not in {i2, s3}]
        n6 = [a.GetIdx() for a in bond.GetEndAtom().GetNeighbors()]
        n6 = [i for i in n6 if i not in {i1, s4}]
        inverse = (
            Chem.BondStereo.STEREOE
            if stereo == Chem.BondStereo.STEREOZ
            else Chem.BondStereo.STEREOZ
        )
        add(s3, s4, stereo)
        if n5:
            add(n5[0], s4, inverse)
        if n6:
            add(s3, n6[0], inverse)
        if n5 and n6:
            add(n5[0], n6[0], stereo)

    for atom in mol.GetAtoms():  # around R/S centres with four neighbours
        if not atom.HasProp("_CIPCode"):
            continue
        neighbours = atom.GetNeighbors()
        if not neighbours or not all(n.HasProp("_CIPRank") for n in neighbours):
            continue
        ranked = sorted(
            neighbours, key=lambda n: int(n.GetProp("_CIPRank")), reverse=True
        )
        if len(ranked) < 4:
            continue
        a, b, c = [n.GetIdx() for n in ranked[:3]]
        if atom.GetProp("_CIPCode") != "R":
            a, b, c = c, b, a
        d = ranked[3].GetIdx()
        for u in (a, b, c):
            add(u, d, "chi_star")
        add(b, a, "chi_circle", both=False)
        add(c, b, "chi_circle", both=False)
        add(a, c, "chi_circle", both=False)
    return pairs, codes


def fingerprint(smiles):
    """Compute the Monroe input graph and conformer of one molecule.

    Parameters
    ----------
    smiles : str
        Molecule as SMILES, used exactly as given.

    Returns
    -------
    tuple of numpy.ndarray or None
        Atom properties ``(n, 8)``, atom categories ``(n, 11)``, 3D coordinates
        ``(n, 3)``, directed edges ``(2, e)`` (bonds, stereo edges, then reversed
        bonds) and edge categories ``(e, 4)``; None if RDKit cannot parse the
        SMILES or it has no atoms.
    """
    mol = _molecule(smiles)
    if mol is None:
        return None
    positions = _conformer(mol)
    floats, codes = _atoms(mol)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    bond_codes = [
        [
            _code(b.GetIsConjugated(), [1]),
            _code(b.IsInRing(), [1]),
            _code(b.GetBondType(), _BOND_TYPES),
            _code(b.GetStereo(), _STEREO),
        ]
        for b in mol.GetBonds()
    ]
    stereo_pairs, stereo_codes = _stereo_edges(mol)
    # Bonds of the four standard types are also used reversed; Monroe treats any
    # other bond type, like its stereo edges, as one-directional.
    reverse = [i for i, c in enumerate(bond_codes) if c[2] != _OTHER_BOND]
    pairs = bonds + stereo_pairs + [bonds[i][::-1] for i in reverse]
    edge_codes = bond_codes + stereo_codes + [bond_codes[i] for i in reverse]
    edges = np.array(pairs, dtype=np.int32).reshape(-1, 2).T.copy()
    edge_codes = np.array(edge_codes, dtype=np.uint8).reshape(-1, 4)
    return floats, codes, positions, edges, edge_codes


def fingerprints(smiles_list):
    """Apply `fingerprint` to a list of SMILES, with RDKit messages silenced."""
    _block = rdBase.BlockLogs()  # noqa: F841 (silences RDKit while alive)
    return [fingerprint(s) for s in smiles_list]


def ensure_weights():
    """Return the path of the Monroe weights, downloading them the first time.

    They live once in ``~/lazysim/models``; ``$LAZYSIM_MONROE_WEIGHTS`` points to a
    local copy instead.

    Returns
    -------
    Path
    """
    return models.ensure(
        WEIGHTS_FILE,
        WEIGHTS_URL,
        WEIGHTS_SHA256,
        WEIGHTS_BYTES,
        "LAZYSIM_MONROE_WEIGHTS",
    )


def load_weights():
    """Load the Monroe encoder weights, checking their sha256.

    Returns
    -------
    dict
        The arrays `encode` needs, with linear layers as (input x output).
    """
    return prepare_weights(models.load_npz(ensure_weights(), WEIGHTS_SHA256))


def prepare_weights(raw):
    """Arrange the arrays of Monroe's encoder state dict (torch names) for `encode`."""

    def linear(name, bias=True):
        w = np.ascontiguousarray(raw[f"{name}.weight"].T)
        return w, (raw[f"{name}.bias"] if bias else None)

    def batch_norm(name):
        scale = raw[f"{name}.weight"] / np.sqrt(raw[f"{name}.running_var"] + 1e-5)
        return raw[f"{name}.running_mean"], scale, raw[f"{name}.bias"]

    node_keys = [
        "valence_implicit",
        "valence_total",
        "hybridization",
        "atomic_number",
        "total_num_hs",
        "is_aromatic",
        "is_in_ring",
        "chirality",
        "period",
        "degree",
        "group",
    ]
    edge_keys = ["conjugation", "is_in_ring", "bond_type", "stereo"]
    layers = []
    for i in range(10):
        layer = f"grit_layers.{i}."
        attention = layer + "attention."
        coef = raw[layer + "deg_coef"][0]  # (720, 2)
        layers.append(
            {
                "Q": linear(attention + "Q"),
                "K": linear(attention + "K"),
                "V": linear(attention + "V"),
                "E": linear(attention + "E"),
                "Aw": raw[attention + "Aw"][:, :, 0],  # (72, heads)
                "VeRow": np.ascontiguousarray(
                    raw[attention + "VeRow"].transpose(1, 0, 2)
                ),
                "deg": (coef[:, 0], coef[:, 1]),
                "O_h": linear(layer + "O_h"),
                "O_e": linear(layer + "O_e"),
                "bn1_h": batch_norm(layer + "batch_norm1_h"),
                "bn1_e": batch_norm(layer + "batch_norm1_e"),
                "bn2_h": batch_norm(layer + "batch_norm2_h"),
                "ffn1": linear(layer + "FFN_h_layer1"),
                "ffn2": linear(layer + "FFN_h_layer2"),
            }
        )
    return {
        "node_tables": [raw[f"node_oh.{k}.weight"] for k in node_keys],
        "edge_tables": [raw[f"edge_oh.{k}.weight"] for k in edge_keys],
        "node_centers": raw["node_float_rbf.centers"],  # (8, 32)
        "node_gammas": raw["node_float_rbf.gammas"],  # (8,)
        "edge_centers": raw["edge_rbf.centers"],  # (32,)
        "edge_gamma": raw["edge_rbf.gamma"],
        "missing_fill": raw["missing_fill"],
        "node_proj": linear("node_proj"),
        "edge_proj": linear("edge_proj"),
        "walk_bias": np.ascontiguousarray(
            raw["rwpe_attn_bias.weight"].T
        ),  # (16, heads)
        "layers": layers,
        "gate": linear("pooling.gate"),
        "value_proj": linear("pooling.value_proj", bias=False)[0],
    }


def _dense(x, layer):
    w, b = layer
    return x @ w + b if b is not None else x @ w


def _batch_norm(x, norm):
    mean, scale, bias = norm
    return (x - mean) * scale + bias


def _random_walks(n, edges, edge_codes):
    """RRWP: the first 16 powers of the bond graph's random walk, in float32."""
    real = edges[:, edge_codes[:, 2] != _OTHER_BOND]
    adjacency = np.zeros((n, n), np.float32)
    adjacency[real[0], real[1]] = 1
    adjacency[real[1], real[0]] = 1
    np.fill_diagonal(adjacency, 1)
    degree = adjacency.sum(axis=1)
    step = adjacency / degree[:, None]
    powers = [step]
    for _ in range(WALK - 1):
        powers.append(powers[-1] @ step)
    powers = np.stack(powers, axis=-1)  # (n, n, 16)
    nodes = np.ascontiguousarray(np.diagonal(powers, axis1=0, axis2=1).T)
    src, dst = edges
    edge_walks = np.float32(0.5) * (powers[src, dst] + powers[dst, src])
    return nodes, edge_walks, np.log(degree + 1)


def _signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x) + np.float32(1e-9))


def _forward(w, graphs):
    """Embed a batch of graphs; each molecule also gets a virtual node."""
    n_mols = len(graphs)
    sizes = np.array([len(g[0]) for g in graphs])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n = int(sizes.sum())
    molecule = np.repeat(np.arange(n_mols), sizes)
    walks = [_random_walks(len(g[0]), g[3], g[4]) for g in graphs]
    floats = np.concatenate([g[0] for g in graphs])
    codes = np.concatenate([g[1] for g in graphs])
    positions = np.concatenate([g[2] for g in graphs])
    src, dst = np.concatenate([g[3] + s for g, s in zip(graphs, starts)], axis=1)
    edge_codes = np.concatenate([g[4] for g in graphs])
    node_walks = np.concatenate([nw for nw, _, _ in walks])
    edge_walks = np.concatenate([ew for _, ew, _ in walks])
    log_degree = np.concatenate([ld for _, _, ld in walks])[:, None]
    members = np.zeros((n_mols, n), np.float32)  # sums over each molecule's atoms
    members[molecule, np.arange(n)] = 1

    # Inputs. Edges are the molecule's own (bonds and stereo), atom -> virtual
    # node ("up") and virtual node -> atom ("down"); the last two are one per atom.
    floats = floats.copy()
    for i, col in enumerate(_MISSING):
        floats[floats[:, col] == 0, col] = w["missing_fill"][i]
    diff = floats[:, :, None] - w["node_centers"][None]
    node_rbf = np.exp(-w["node_gammas"][None, :, None] * diff * diff).reshape(n, -1)
    node_emb = sum(t[codes[:, k]] for k, t in enumerate(w["node_tables"]))
    atoms = np.concatenate([node_emb, node_rbf, node_walks], axis=1)
    virtual = np.concatenate(
        [
            sum(t[-1] for t in w["node_tables"]),
            np.zeros(atoms.shape[1] - 128, np.float32),
        ]
    )
    x = _dense(
        np.vstack([atoms, np.broadcast_to(virtual, (n_mols, len(virtual)))]),
        w["node_proj"],
    )  # atoms, then one virtual node per molecule

    def edge_input(lengths, emb, walk):
        rbf = np.exp(-w["edge_gamma"] * (lengths[:, None] - w["edge_centers"]) ** 2)
        return _dense(np.concatenate([emb, rbf, walk], axis=1), w["edge_proj"])

    edge_emb = sum(t[edge_codes[:, k]] for k, t in enumerate(w["edge_tables"]))
    lengths = np.linalg.norm(positions[src] - positions[dst], axis=1)
    to_centre = np.linalg.norm(positions, axis=1)  # the virtual node sits at 0
    vn_emb = np.broadcast_to(sum(t[-1] for t in w["edge_tables"]), (n, 128))
    no_walk = np.zeros((n, WALK), np.float32)
    edges = {
        "own": edge_input(lengths, edge_emb, edge_walks),
        "up": edge_input(to_centre, vn_emb, no_walk),
        "down": edge_input(to_centre, vn_emb, no_walk),
    }
    walk_bias = {
        "own": edge_walks @ w["walk_bias"],
        "up": np.zeros((n, HEADS), np.float32),
        "down": np.zeros((n, HEADS), np.float32),
    }
    slots = _incoming_slots(dst, n)
    vn = n + molecule  # each atom's virtual node
    degree_term = np.vstack([log_degree, np.zeros((n_mols, 1), np.float32)])

    for depth, layer in enumerate(w["layers"]):
        size = DIM // HEADS
        q = _dense(x, layer["Q"]).reshape(-1, HEADS, size)
        k = _dense(x, layer["K"]).reshape(-1, HEADS, size)
        v = _dense(x, layer["V"]).reshape(-1, HEADS, size)
        ends = {"own": (src, dst), "up": (np.arange(n), vn), "down": (vn, np.arange(n))}
        messages, logits = {}, {}
        for kind, e in edges.items():
            proj = _dense(e, layer["E"]).reshape(len(e), HEADS, 2 * size)
            s_idx, d_idx = ends[kind]
            score = _signed_sqrt(k[s_idx] * q[d_idx] * proj[:, :, :size])
            score = np.clip(score, -5, 5) + proj[:, :, size:]
            messages[kind] = score
            logit = np.einsum("ehd,dh->eh", score, layer["Aw"]) * size**-0.5
            logits[kind] = np.clip(logit + walk_bias[kind], -5, 5)

        # Softmax over each node's incoming edges: an atom gets its own edges and
        # one from its virtual node; a virtual node gets one from each atom.
        top = logits["down"].copy()
        for t, e in slots:
            top[t] = np.maximum(top[t], logits["own"][e])
        top_vn = np.maximum.reduceat(logits["up"], starts, axis=0)
        weight = {
            "own": np.exp(logits["own"] - top[dst]),
            "down": np.exp(logits["down"] - top),
            "up": np.exp(logits["up"] - top_vn[molecule]),
        }
        total = weight["down"].copy()
        for t, e in slots:
            total[t] += weight["own"][e]
        total_vn = members @ weight["up"]
        alpha = {
            "own": weight["own"] / (total[dst] + 1e-16),
            "down": weight["down"] / (total + 1e-16),
            "up": weight["up"] / (total_vn[molecule] + 1e-16),
        }
        values = np.zeros((n + n_mols, HEADS, size), np.float32)
        rows = np.zeros((n + n_mols, HEADS, size), np.float32)
        values[:n] = alpha["down"][:, :, None] * v[vn]
        rows[:n] = alpha["down"][:, :, None] * messages["down"]
        for t, e in slots:
            a = alpha["own"][e][:, :, None]
            values[t] += a * v[src[e]]
            rows[t] += a * messages["own"][e]
        up_values = (alpha["up"][:, :, None] * v[:n]).reshape(n, -1)
        up_rows = (alpha["up"][:, :, None] * messages["up"]).reshape(n, -1)
        values[n:] = (members @ up_values).reshape(n_mols, HEADS, size)
        rows[n:] = (members @ up_rows).reshape(n_mols, HEADS, size)
        rows = (rows.transpose(1, 0, 2) @ layer["VeRow"]).transpose(1, 0, 2)
        h = (values + rows).reshape(-1, DIM)
        c0, c1 = layer["deg"]
        h = h * c0 + h * degree_term * c1
        h = _batch_norm(x + _dense(h, layer["O_h"]), layer["bn1_h"])
        x = _batch_norm(
            h + _dense(np.maximum(_dense(h, layer["ffn1"]), 0), layer["ffn2"]),
            layer["bn2_h"],
        )
        if depth < len(w["layers"]) - 1:  # the last layer's edges are not used
            edges = {
                kind: _batch_norm(
                    edges[kind]
                    + _dense(messages[kind].reshape(len(e), -1), layer["O_e"]),
                    layer["bn1_e"],
                )
                for kind, e in edges.items()
            }

    # Attention pooling over each molecule's atoms and its virtual node.
    gate = _dense(x, w["gate"])
    group = np.concatenate([molecule, np.arange(n_mols)])
    order = np.argsort(group, kind="stable")  # contiguous per molecule
    first = np.searchsorted(group[order], np.arange(n_mols))
    top = np.maximum.reduceat(gate[order], first, axis=0)
    weight = np.exp(gate - top[group])
    total = np.add.reduceat(weight[order], first, axis=0)
    alpha = weight / (total[group] + 1e-16)
    values = (x @ w["value_proj"]).reshape(-1, HEADS, DIM // HEADS)
    pooled = alpha[:, :, None] * values
    out = np.add.reduceat(pooled[order].reshape(len(group), -1), first, axis=0)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _incoming_slots(dst, n):
    """Group edges so that each slot holds at most one incoming edge per atom."""
    order = np.argsort(dst, kind="stable")
    counts = np.bincount(dst, minlength=n)
    rank = np.arange(len(dst)) - np.repeat(np.cumsum(counts) - counts, counts)
    return [
        (dst[order[rank == k]], order[rank == k])
        for k in range(int(counts.max(initial=0)))
    ]


def encode(weights, graphs, batch=64):
    """Embed molecular graphs with the Monroe encoder, L2-normalised.

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
        float32 embeddings of shape ``(len(graphs), DIM)``. Monroe's own output is
        not normalised; lazy-sim normalises it so cosine is a dot product.
    """
    out = np.empty((len(graphs), DIM), np.float32)
    for start in range(0, len(graphs), batch):
        part = graphs[start : start + batch]
        out[start : start + len(part)] = _forward(weights, part)
    return out
