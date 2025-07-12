import torch
from torch.utils.data import Dataset
from model.utils import get_mol, canonical_smiles, get_atom_bond_chars, smiles_to_graph, max_num_atoms
import numpy as np

class MoleculeDataset(Dataset):
    def __init__(self, data_list, props, num_atoms, atom_dim, bond_dim,
                 atom_stoi, bond_stoi):
        self.data_list = data_list # SMILES
        self.props = props
        self.atom_chars, self.bond_chars = get_atom_bond_chars(data_list)
        self.num_atoms = max_num_atoms(self.data_list) if num_atoms is None else num_atoms
        self.atom_dim = atom_dim if atom_dim is not None else len(self.atom_chars)
        self.bond_dim = bond_dim if bond_dim is not None else len(self.bond_chars)
        self.atom_stoi = atom_stoi if atom_stoi is not None else {atom: i for i, atom in enumerate(self.atom_chars)}
        self.bond_stoi = bond_stoi if bond_stoi is not None else {bond: i for i, bond in enumerate(self.bond_chars)}
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        smiles = self.data_list[idx]
        smiles = canonical_smiles(smiles)
        x_fea, x_adj = smiles_to_graph(smiles, self.atom_stoi, self.bond_stoi,
                                      num_atoms=self.num_atoms, atom_dim=self.atom_dim, bond_dim=self.bond_dim)
        x_fea = torch.tensor(x_fea, dtype=torch.float32)
        x_adj = torch.tensor(x_adj, dtype=torch.float32)
        y = torch.tensor(self.props[idx], dtype=torch.float32)
        
        return x_fea, x_adj, y
        
        