"""The one validity rule every featurizer shares, so their rows line up."""

from rdkit import Chem, rdBase


def is_valid(smiles):
    """Tell whether RDKit parses and sanitises `smiles` into a molecule with atoms."""
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumAtoms() > 0


def validate(smiles_list):
    """Apply `is_valid` to a list of SMILES, with RDKit messages silenced."""
    _block = rdBase.BlockLogs()  # noqa: F841 (silences RDKit while alive)
    return [is_valid(s) for s in smiles_list]


def inchikey(smiles):
    """Return the InChIKey of a valid SMILES, or None if RDKit cannot make one."""
    _block = rdBase.BlockLogs()  # noqa: F841
    try:
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) or None
    except Exception:
        return None
