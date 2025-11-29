import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torchdiffeq import odeint

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim=128, n_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hid_dim)
        # set edge_dim=1 since we'll pass scalar edge_attr shaped [E,1]
        self.layers = nn.ModuleList([
            TransformerConv(hid_dim, hid_dim // heads, heads=heads,
                            dropout=dropout, beta=True, edge_dim=1)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_lin(x)
        for conv, norm in zip(self.layers, self.norms):
            x_res = x
            # pass edge_attr explicitly (shape [E, 1])
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.gelu(x)
            x = norm(x + x_res)
            x = self.dropout(x)
        return x  # [N, hid_dim]


class ODEFunc(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Linear(hid_dim * 2, hid_dim),
        )
    def forward(self, t, z):
        return self.net(z)


class GTNode(nn.Module):
    def __init__(self, in_dim, hid_dim=128, n_layers=3, heads=4, use_ode=True, ode_steps=5, num_classes=10):
        super().__init__()
        self.encoder = GraphTransformerEncoder(in_dim, hid_dim, n_layers, heads)
        self.use_ode = use_ode
        self.ode_steps = ode_steps
        if use_ode:
            self.odefunc = ODEFunc(hid_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, hid_dim//2),
            nn.GELU(),
            nn.Linear(hid_dim//2, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None):
        h0 = self.encoder(x, edge_index, edge_attr)  # [N, hid]
        if self.use_ode:
            # integrate from t=0 to t=1 with a few eval points; shape: (T,N,H)
            t = torch.linspace(0.0, 1.0, self.ode_steps).to(h0.device)
            hT = odeint(self.odefunc, h0, t, method='rk4')[-1]
        else:
            hT = h0
        logits = self.classifier(hT)
        return logits, hT
