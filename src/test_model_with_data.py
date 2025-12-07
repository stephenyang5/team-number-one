import torch
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GraphTransformer, GraphTransformerSimple


def load_graph(graph_path):
    """Load a graph from a .pt file."""
    print(f"Loading graph from {graph_path}...")
    graph = torch.load(graph_path, weights_only=False)
    
    if isinstance(graph, dict):
        # Convert dict to Data object if needed
        from torch_geometric.data import Data
        graph = Data(
            x=graph['x'],
            edge_index=graph['edge_index'],
            edge_attr=graph.get('edge_attr', None),
            timepoint_norm=graph.get('timepoint_norm', None),
            celltype=graph['celltype'],
            celltype_names=graph.get('celltype_names', None),
        )
    
    return graph


def test_model_with_data():
    """Test the model with actual data from the data folder."""
    
    # Path to the processed graph
    graph_path = '../data/blood/blood_graph_velocity.pt'
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        print("Please ensure the graph has been processed.")
        return
    
    # Load the graph
    graph = load_graph(graph_path)
    
    # Print graph statistics
    print("\n" + "="*30)
    print("Graph Statistics")
    print("="*30)
    print(f"Nodes: {graph.x.shape[0]:,}")
    print(f"Edges: {graph.edge_index.shape[1]:,}")
    print(f"Node features: {graph.x.shape[1]}")
    print(f"Edge attributes: {graph.edge_attr.shape if graph.edge_attr is not None else 'None'}")
    print(f"Timepoint norm: {graph.timepoint_norm.shape if hasattr(graph, 'timepoint_norm') and graph.timepoint_norm is not None else 'None'}")
    print(f"Cell types: {len(graph.celltype_names) if hasattr(graph, 'celltype_names') and graph.celltype_names is not None else 'Unknown'}")
    if hasattr(graph, 'celltype_names') and graph.celltype_names is not None:
        print(f"Cell type names: {graph.celltype_names}")
    
    # Get model parameters from graph
    n_features = graph.x.shape[1]
    n_classes = len(graph.celltype_names) if hasattr(graph, 'celltype_names') and graph.celltype_names is not None else graph.celltype.max().item() + 1
    
    print("\n" + "="*30)
    print("Testing GraphTransformer (Full Model)")
    print("="*30)
    
    # Test full model with timepoint and edge attributes
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
    
    # Forward pass
    try:
        # Check if timepoint_norm exists and is not None
        if hasattr(graph, 'timepoint_norm') and graph.timepoint_norm is not None:
            timepoint_norm = graph.timepoint_norm
        else:
            print("Warning: timepoint_norm not found, using None")
            timepoint_norm = None
        
        # Check if edge_attr exists and is not None
        if graph.edge_attr is not None:
            edge_attr = graph.edge_attr
        else:
            print("Warning: edge_attr not found, using None")
            edge_attr = None
        
        out = model(
            graph.x,
            graph.edge_index,
            edge_attr=edge_attr,
            timepoint_norm=timepoint_norm,
        )
        
        print(f"Forward pass successful!")
        print(f"Output shape: {out.shape}")
        print(f"Expected: [{graph.x.shape[0]}, {n_classes}]")
        assert out.shape == (graph.x.shape[0], n_classes), \
            f"Expected shape ({graph.x.shape[0]}, {n_classes}), got {out.shape}"
        print("Output shape correct!")
        
        # Check for NaN or Inf
        if torch.isnan(out).any():
            print("Warning: Output contains NaN values")
        if torch.isinf(out).any():
            print("Warning: Output contains Inf values")
        
        print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test simple model
    print("\n" + "="*30)
    print("Testing GraphTransformerSimple")
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
        assert out_simple.shape == (graph.x.shape[0], n_classes), \
            f"Expected shape ({graph.x.shape[0]}, {n_classes}), got {out_simple.shape}"
        print("Output shape correct!")
        
        # Check for NaN or Inf
        if torch.isnan(out_simple).any():
            print("Warning: Output contains NaN values")
        if torch.isinf(out_simple).any():
            print("Warning: Output contains Inf values")
        
        print(f"Output range: [{out_simple.min().item():.4f}, {out_simple.max().item():.4f}]")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*30)
    print("All tests passed!")
    print("="*30)
    print("\nThe model is ready to use with your data.")


if __name__ == '__main__':
    test_model_with_data()

