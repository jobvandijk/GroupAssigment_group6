import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv('untested_molecules.csv')
def compute_2Ddescriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = [desc[1](mol) for desc in Descriptors.descList if not desc[0].startswith('fr_')]
    return descriptors
"""
def get_maccs_keys(smiles):
    # Convert SMILES to RDKit molecule
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return [0] * 166  # Return an array of zeros if the molecule is invalid
    # Generate MACCS keys for the molecule
    maccs_fp = MACCSkeys.GenMACCSKeys(molecule)
    # Convert the fingerprint to a list of integers (0 or 1) and skip the first bit
    return list(maccs_fp)[1:]

# Apply the function to the 'SMILES' column and create a new DataFrame for MACCS keys
maccs_keys_df = df['SMILES'].apply(get_maccs_keys)
maccs_keys_df = pd.DataFrame(maccs_keys_df.tolist(), columns=[f'MACCS_{i}' for i in range(1, 167)])"""
# Compute descriptor values for each SMILES string and add as new columns
descriptor_columns = [desc[0] for desc in Descriptors.descList if not desc[0].startswith('fr_')]
df[descriptor_columns] = df['SMILES'].apply(lambda x: pd.Series(compute_2Ddescriptors(x)))

print(df.  head())
df.to_csv('untested_2D_descriptors.csv', index=False)
