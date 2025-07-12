from rdkit import Chem
import numpy as np
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# This function receives input as either a SMILES string or Rdkit mol obeject.
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    elif isinstance(smiles_or_mol, Chem.Mol):
        return smiles_or_mol
    else:
        return None


def canonical_smiles(smiles_or_mol):
    '''
    Returns canonical SMILES of a molecule
    '''
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def max_num_atoms(smiles_list):
    max_atoms = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            if num_atoms > max_atoms:
                max_atoms = num_atoms
    return max_atoms

def get_atom_bond_chars(smiles_list):
    '''
    Returns vocabulary of atoms and bond types from a list of SMILES strings
    '''
    atom_set = set()
    bond_set = set()
    
    for smiles in smiles_list:
        mol = get_mol(smiles)
        if mol is not None:
            for atom in mol.GetAtoms():
                atom_set.add(atom.GetSymbol())

            for bond in mol.GetBonds():
                bond_set.add(str(bond.GetBondType()))
    
    return sorted(list(atom_set)), sorted(list(bond_set))




def smiles_to_graph(smiles, atom_stoi: dict, bond_stoi:dict, num_atoms, 
                    atom_dim, bond_dim):
    '''
    Converts a SMILES string to a graph representation
    '''
    mol = get_mol(smiles)
    # Initialize atom and edge tensors
    feature_matrix = np.zeros((num_atoms, atom_dim), np.float32)
    adjacency_matrix = np.zeros((bond_dim, num_atoms, num_atoms), np.float32)
    
    # Get feature and adjacency matrices
    for atom in mol.GetAtoms():
        i, atom_symbol = atom.GetIdx(), atom.GetSymbol()
        atom_idx = atom_stoi[atom_symbol]
        feature_matrix[i] = np.eye(atom_dim)[atom_idx]
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, j)
            bond_idx = bond_stoi[bond.GetBondType().name]
            adjacency_matrix[bond_idx, [i,j], [j,i]] = 1.0 # a one-hot vector representing the type of edge between i and j.
            
    return feature_matrix, adjacency_matrix
            
        
        
    
    