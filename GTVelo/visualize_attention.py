import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GTVelo.model_transformer import create_model
from GTVelo.attention_viz import create_attention_report

if __name__ == "__main__":
    # Load trained model
    checkpoint_path = 'checkpoints/gtvelo_best.pt'
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['config']
    
    # Load data to get num_classes
    print("Loading data...")
    data = torch.load('data/blood/blood_prepared.pt')
    
    # Create model
    model = create_model(
        num_classes=data['num_classes'],
        **model_config
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Test accuracy: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
    
    # Create attention visualizations
    print("\n" + "="*60)
    print("Creating Attention Visualizations")
    print("="*60)
    
    # Optionally specify which nodes to visualize
    # You can pick interesting cells (e.g., from test set, specific cell types)
    test_mask = data.get('test_mask', torch.zeros(data['X'].shape[0], dtype=torch.bool))
    if test_mask.sum() > 0:
        test_indices = torch.where(test_mask)[0].tolist()
        node_indices = test_indices[:10]  # First 10 test nodes
        print(f"Visualizing {len(node_indices)} progenitor cells from test set")
    else:
        node_indices = None  # Will pick random nodes
    
    create_attention_report(
        model=model,
        data=data,
        output_dir='attention_viz',
        node_indices=node_indices,
        top_k_neighbors=25,
        top_k_nodes=100,
        exclude_celltypes=['Primitive erythroid cells']
    )
    
    print("\n Done! Check the 'attention_viz/' directory for visualizations.")

