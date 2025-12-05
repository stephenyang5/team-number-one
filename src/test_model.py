import torch
from torch_geometric.data import Data
from model import GraphTransformer, GraphTransformerSimple


def test_model():
    """sanity checking model test before setting it loose on oscar with dummy graph :( """
    
    #dummy graph matching blood data  structure
    n_nodes = 100
    n_edges = 500
    n_features = 50 
    n_classes = 5
    
    # node features PCA embeddings
    x = torch.randn(n_nodes, n_features)
    
    # edges index
    edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)

    edge_attr = torch.randn(n_edges, 1)
    
    #timepoints
    timepoint_norm = torch.randn(n_nodes)
    
    # cell type labels
    celltype = torch.randint(0, n_classes, (n_nodes,), dtype=torch.long)
    
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        timepoint_norm=timepoint_norm,
        celltype=celltype,
    )
    
    print(f"Graph created:")
    print(f"Nodes: {graph.x.shape[0]}")
    print(f"Edges: {graph.edge_index.shape[1]}")
    print(f"Node features: {graph.x.shape[1]}")
    print(f"Edge attributes: {graph.edge_attr.shape}")
    
    #test model
    print("\n" + "="*30)
    print("Testing model")
    print("="*30)
    
    model = GraphTransformer(
        in_channels=n_features,
        hidden_channels=64,
        out_channels=n_classes,
        num_layers=2,
        heads=2,
        dropout=0.1,
        use_timepoint=False,
        use_edge_attr=True,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # forward pass
    try:
        out = model(
            graph.x,
            graph.edge_index,
            edge_attr=graph.edge_attr,
            timepoint_norm=graph.timepoint_norm,
        )
        print(f"Forward pass successful")
        print(f"Output shape: {out.shape}")
        print(f"Expected: [{n_nodes}, {n_classes}]")
        assert out.shape == (n_nodes, n_classes), f"Expected shape ({n_nodes}, {n_classes}), got {out.shape}"
        print("Output shape correct!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    
    # test simple model
    print("\n" + "="*30)
    print("testing simple trnasformer")
    print("="*30)
    
    model_simple = GraphTransformerSimple(
        in_channels=n_features,
        hidden_channels=64,
        out_channels=n_classes,
        num_layers=2,
        heads=2,
        dropout=0.1,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model_simple.parameters()):,}")
    
    try:
        out_simple = model_simple(graph.x, graph.edge_index)
        print(f"Forward pass successful!")
        print(f"Output shape: {out_simple.shape}")
        assert out_simple.shape == (n_nodes, n_classes), f"Expected shape ({n_nodes}, {n_classes}), got {out_simple.shape}"
        print("Output shape correct!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    
    print("\n" + "="*30)
    print("All tests passed!")
    print("="*30)


if __name__ == '__main__':
    test_model()

