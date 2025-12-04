
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

"""
Graph Transformer encoder
Uses TransformerConv with velocity-weighted edges
"""
class GraphTransformerEncoder(nn.Module):

    def __init__(self, in_dim, hid_dim=64, n_layers=2, heads=2, dropout=0.1):
        super().__init__()
        
        self.input_lin = nn.Linear(in_dim, hid_dim)
        
        #Transformer layers
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hid_dim,
                out_channels=hid_dim // heads,
                heads=heads,
                dropout=dropout,
                beta=True,  # Gated residual for better training
                edge_dim=1,  # Velocity-weighted edge attributes
                concat=True
            )
            for n in range(n_layers)
        ])
        
        # Batch Normalization
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hid_dim) for n in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    """
    Purpose: Define forward pass
    Params:
        x: [N, in_dim] node features 
        edge_index: [2, E] edges
        edge_attr: [E, 1] velocity weighted edges
    Return:[N, hid_dim] learned cell embeddings
    """
    def forward(self, x, edge_index, edge_attr=None):
        
        x = self.input_lin(x)
        x = F.relu(x)  # ReLU 
        
        # Transformer layers with residual connections
        for conv, norm in zip(self.layers, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)  # ReLU 
            x = x + x_res  # Residual conections
            x = self.dropout(x)
        
        return x


"""
GTVelo: Final GTVelo!! 
"""
class GTVelo(nn.Module):

    def __init__(self, in_dim=50, hid_dim=64, n_layers=2, heads=2, 
                 dropout=0.1, num_classes=5):
        
        super().__init__()
        
        # Graph Transformer encoder
        self.encoder = GraphTransformerEncoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_layers=n_layers,
            heads=heads,
            dropout=dropout
        )
        
        # Classification head 
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, num_classes)
        )
        
        self.num_classes = num_classes
    
    """
    Purpose: Define forward pass
    Params:
        x: [N, in_dim] node features 
        edge_index: [2, E] edges
        edge_attr: [E, 1] velocity weighted edges
    Returns:
        logits: [N, num_classes] fate predictions
        embeddings: [N, hid_dim] learned representations
    """
    def forward(self, x, edge_index, edge_attr=None):
        
        # Transformer encoding with velocity edges
        embeddings = self.encoder(x, edge_index, edge_attr)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits, embeddings
    
    def get_embeddings(self, x, edge_index, edge_attr=None):
        return self.encoder(x, edge_index, edge_attr)

"""
Purpose: Create GTVelo
Params: num_classes, other args
Return: GTVelo transformer model
"""
def create_model(num_classes=5, **model_kwargs):
    
    # Set CPU-friendly defaults if not specified
    if 'hid_dim' not in model_kwargs:
        model_kwargs['hid_dim'] = 64
    if 'n_layers' not in model_kwargs:
        model_kwargs['n_layers'] = 2
    if 'heads' not in model_kwargs:
        model_kwargs['heads'] = 2
    
    model = GTVelo(num_classes=num_classes, **model_kwargs)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"GTVelo")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Blood cell types: {num_classes}")
    print(f"\nArchitecture:")
    print(f"\t- Num Layers: {model_kwargs['n_layers']}")
    print(f"\t- Hidden dimension: {model_kwargs['hid_dim']}")
    print(f"\t- Attention heads: {model_kwargs['heads']}")
    
    return model


if __name__ == "__main__":
    
    model = create_model(
        num_classes=5,  # Rn optimized for blood dataset
        in_dim=50,
        hid_dim=64,     
        n_layers=2,     
        heads=2     
    )
    
    


