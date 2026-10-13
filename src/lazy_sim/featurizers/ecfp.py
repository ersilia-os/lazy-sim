"""ECFP6: 2048-bit Morgan fingerprints of radius 3, compared by exact Tanimoto.

Searching uses FPSim2 (``pip install lazy-sim[ecfp]``), which builds the same
fingerprints with RDKit's Morgan generator and these parameters.
"""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

NAME = "ecfp"
ENGINE = "fpsim2"
SIMILARITY = "tanimoto"
FP_TYPE = "Morgan"
FP_PARAMS = {"radius": 3, "fpSize": 2048}
DIM = FP_PARAMS["fpSize"]


def packed_fingerprints(smiles_list):
    """Compute packed ECFP6 bits for valid SMILES.

    Parameters
    ----------
    smiles_list : list of str
        SMILES that pass `lazy_sim.utils.chem.is_valid`.

    Returns
    -------
    list of numpy.ndarray
        One uint8 array of 256 bytes per molecule; ``np.unpackbits`` gives the
        2048 bits in RDKit's order.
    """
    _block = rdBase.BlockLogs()  # noqa: F841 (silences RDKit while alive)
    generator = rdFingerprintGenerator.GetMorganGenerator(**FP_PARAMS)
    return [
        np.packbits(generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)))
        for s in smiles_list
    ]
