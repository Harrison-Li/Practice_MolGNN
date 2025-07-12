import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.utils import set_seed
from model.gnn import MolGNN
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer, Config
from .dataset import MoleculeDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN model on molecular data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the CSV file containing SMILES and properties")
    parser.add_argument("--props", type=str, required=True, help="Name of the property to predict")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size for GNN layers")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    wandb.init(project="molgnn", name='molgnn1')

    # Load data
    data = pd.read_csv(f'data/{args.data_file}.csv')
    data.columns = data.columns.str.lower()  
    smiles_list = data['smiles'].tolist()
    properties = data[args.props].values

    # Create datasets
    train_smiles, val_smiles, train_props, val_props = train_test_split(smiles_list, properties, test_size=0.2, random_state=args.seed)
    train_dataset = MoleculeDataset(train_smiles, train_props, num_atoms=None, atom_dim=None, bond_dim=None, atom_stoi=None, bond_stoi=None)
    val_dataset = MoleculeDataset(val_smiles, val_props, num_atoms=None, atom_dim=None, bond_dim=None, atom_stoi=None, bond_stoi=None)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    input_dim = train_dataset.atom_dim
    output_dim = 1  # For regression tasks
    model = MolGNN(input_dim=input_dim,
                   hidden_dim=args.hidden_dim,
                   output_dim=output_dim,
                   num_layers=args.num_layers,
                   bond_dim=train_dataset.bond_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      test_dataset=val_dataset,
                      config=Config(max_epochs=args.max_epochs,
                                    batch_size=args.batch_size,
                                    learning_rate=args.learning_rate,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    ckpt_path='model_checkpoint.pth',
                                    device='cuda' if torch.cuda.is_available() else 'cpu'))
    # Train the model
    df = trainer.train(wandb)
    
                                    