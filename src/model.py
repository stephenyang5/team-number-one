import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GCNConv, GINEConv, Linear


class GraphTransformer(nn.Module):
    """
    Graph Transformer model for cell fate classification using GPSConv (from GraphGPS)
    uses local graph convolutions and global attention to try to learn relationships
    at both scales
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        use_timepoint: bool = True,
        use_edge_attr: bool = True,
    ):
        """
        parameters:
            in_channels: input feature dimensions - e.g. 50 PCA dims for blood  
            hidden_channels: hidden dimension for the model
            out_channels: num cell type classes
            num_layers: num of GPSConv layers
            heads: num of attention heads in GPSConv
            dropout: dropout rate
            use_timepoint: whether to concatenate timepoint as additional node feature
            use_edge_attr: whether to use edge attributes (velocity-weighted edges)
        """
        super(GraphTransformer, self).__init__()
        self.use_timepoint = use_timepoint
        self.use_edge_attr = use_edge_attr
        # adjusts input dimension if using timepoint
        if use_timepoint:
            in_channels = in_channels + 1  #adds timepoint_norm as feature
        
        #1st layer projects inputs to hidden dimension
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # GPSConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            local_conv = GINEConv(
                nn=nn_edge,
                eps=0.0,
                train_eps=True,
                edge_dim=1,  # edge attributes are 1-dimensional
            )

            gps_conv = GPSConv(
                channels=hidden_channels,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type='multihead',  
            )
            self.convs.append(gps_conv)
        
        # batch norming layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # output projection
        self.output_proj = Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr=None, timepoint_norm=None):
        """
        Forward pass!!
        
        parameters:
            x: node features [n_nodes, in_channels]
            edge_index: edge indexes [2, n_edges]
            edge_attr: edge attributes (velocity dual weighting) [n_edges, edge_dim] (optional)
            timepoint_norm: norm timepoint [n_nodes] (optional)
        
        output:
            logits: cell type predictions [n_nodes, out_channels]
        """
        
        # concatenate timepoint if using it
        if self.use_timepoint and timepoint_norm is not None:
            # timepoint_norm is [n_nodes], need to reshape to [n_nodes, 1] for concatenation
            x = torch.cat([x, timepoint_norm.unsqueeze(1)], dim=1)

        # project input
        x = self.input_proj(x)
        
        # apply GPSConv layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # store residual
            x_residual = x
            

            if edge_attr is not None:
                # GINEConv expects edge_attr, GPSConv will pass it through
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            # batch norm
            x = bn(x)
            
            # residual connection and activation
            x = x + x_residual
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # output projection
        logits = self.output_proj(x)
        
        return logits


class GraphTransformerSimple(nn.Module):
    """
    Simplified version without timepoint or edge attributes for testing.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super(GraphTransformerSimple, self).__init__()
        
        self.input_proj = Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GCNConv(
                hidden_channels,
                hidden_channels,
                add_self_loops=True,
                normalize=True
            )
            gps_conv = GPSConv(
                channels=hidden_channels,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type='multihead',
            )
            self.convs.append(gps_conv)
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        self.output_proj = Linear(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = x + x_residual
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.output_proj(x)
        return logits

