import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GTvelo(nn.Module):
    """
    GTvelo: A Graph Transformer for RNA-velocity–aware cell fate prediction.
    Combines PCA-reduced expression features with velocity-weighted graph structure.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        use_edge_attr=True,
    ):
        super().__init__()
        self.use_edge_attr = use_edge_attr

        # Project raw PCA features into hidden space
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # TransformerConv layers + layer norms
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    edge_dim=1 if use_edge_attr else None,
                    dropout=dropout,
                    beta=True,  # allows residual gating
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Classification head for predicting the fate label
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: (N, in_dim) PCA features
        edge_index: graph edges
        edge_attr: RNA velocity alignment scores (optional)
        """

        # Velocity-aware edge attributes
        if self.use_edge_attr:
            if edge_attr is None:
                raise ValueError("GTvelo expects edge_attr for RNA velocity.")
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
        else:
            edge_attr = None

        # Initial projection
        h = self.input_proj(x)

        # TransformerConv layers with residual connections
        for conv, ln in zip(self.layers, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = ln(h + h_res)
            h = self.dropout(h)

        # Predict final cell fate
        logits = self.classifier(h)
        return logits

