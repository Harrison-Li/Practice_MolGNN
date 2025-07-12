import torch
import torch.nn as nn
import torch.nn.functional as F


class MolGNNConfig:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bond_dim):
        self.input_dim = 12
        self.hidden_dim = 128
        self.output_dim = 1
        self.num_layers = 5
        self.bond_dim = 4

class MolGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bond_dim):
        super(MolGNN, self).__init__()
        self.num_layers = num_layers
        self.bond_dim = bond_dim
        
        # Node feature layers
        self.node_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.node_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.node_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Edge feature layers
        self.edge_transform = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(bond_dim)])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
    
    def configure_optimizers(self, config):
        return torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        
    def forward(self, feature_matrix, adjacency_matrix):
              
        x = feature_matrix
        adj = adjacency_matrix #(b, bond_dim, [num_atoms, num_atoms], [num_atoms,num_atoms])
        
        # Message passing layers
        for i, layer in enumerate(self.node_layers):
            x = layer(x)
            
            # Aggregate messages from neighbors
            messages = torch.zeros_like(x)
            for bond_type_idx in range(self.bond_dim):
                bond_adj = adj[:, bond_type_idx, :, :] # (b, num_atoms, h_dim)
                neighbor_features = torch.matmul(bond_adj, x)  # (b, num_atoms, h_dim)

                # Apply bond-specific transformation
                transformed_message = self.edge_transform[bond_type_idx](neighbor_features)
                messages += transformed_message
                
            # Add messages to node features
            x = x + messages   
            x = self.layer_norms[i](x)
            x = F.relu(x)
        
        # Global pooling 
        graph_representation = torch.sum(x, dim=1)  # (b, hidden_dim)
        
        # Final prediction
        output = self.output_layer(graph_representation)
        return output