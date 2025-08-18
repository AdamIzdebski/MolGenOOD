import datamol as dm
from typing import Optional
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def standardize_smiles(smiles: str) -> Optional[str]:
    # Standardize and sanitize
    try:
        mol = dm.to_mol(smiles, sanitize=True)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol)
        mol = dm.standardize_mol(mol)
        standardized_smiles = dm.to_smiles(mol)
        if standardized_smiles is not None:
            return standardized_smiles
        else:
            warnings.warn(f"Empty SMILES string for sequence {smiles}", UserWarning)
            return None
    except Exception as e:
        warnings.warn(f"Error standardizing SMILES {smiles}: {e}", UserWarning)
        return None
    