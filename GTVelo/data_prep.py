import torch
import numpy as np
import scanpy as sc

"""
Purpose: Load velocity-weighted graph and prepare for temporal holdout training
Params: dataset name and temopral_cutoff
Return: Training Data Dict 
"""
def prepare_dataset(dataset, temporal_cutoff=11.0):
    # Locate the files: 
    graph_path=f"{dataset}/{dataset}_graph_velocity.pt"
    adata_path=f"{dataset}/{dataset}_data_velocity.h5ad"

    # Load graph
    print(f"\nLoading graph from {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    
    # Load adata
    print(f"Loading adata from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # Extract data
    X = graph.x  # [N, 50] PCA features
    edge_index = graph.edge_index  # [2, E]
    edge_attr = graph.edge_attr  # [E, 1] velocity-weighted
    
    # Handle timepoint 
    timepoint = torch.from_numpy(graph.timepoint).float()
    
    # Handle celltype 
    labels = torch.from_numpy(graph.celltype).long()
    
    # Add names
    celltype_names = graph.celltype_names
    
    print(f"\nDataset info:")
    print(f"Cells: {X.shape[0]:,}")
    print(f"Genes (PCA): {X.shape[1]}")
    print(f"Edges: {edge_index.shape[1]:,}")
    print(f"Cell types: {celltype_names}")
    
    # Show timepoint distribution
    print(f"\nTimepoint distribution:")
    for t in sorted(timepoint.unique().tolist()):
        n_cells = (timepoint == t).sum().item()
        print(f"  E{t}: {n_cells:,} cells")
    
    # Create temporal split
    print(f"\n Temporal cutoff = E{temporal_cutoff})")
    
    train_mask = timepoint >= temporal_cutoff
    test_mask = timepoint < temporal_cutoff
    
    # Validation: use middle timepoint from training set
    train_timepoints = timepoint[train_mask].unique()
    if len(train_timepoints) > 2:
        val_timepoint = sorted(train_timepoints.tolist())[len(train_timepoints) // 2]
        val_mask = (timepoint == val_timepoint)
        train_mask = train_mask & ~val_mask
    else:
        val_mask = torch.zeros_like(train_mask)
    
    print(f"\nSplit summary:")
    print(f"TRAIN (E{temporal_cutoff}+): {train_mask.sum():,} cells (learn from differentiated)")
    if val_mask.sum() > 0:
        print(f" VAL: {val_mask.sum():,} cells")
    print(f"TEST (E<{temporal_cutoff}): {test_mask.sum():,} cells (predict progenitor fates)")
    
    # Show label distribution in each split
    print(f"\nLabel distribution:")
    for split_name, mask in [('TRAIN', train_mask), ('TEST', test_mask)]:
        print(f"{split_name}:")
        for i, name in enumerate(celltype_names):
            count = ((labels == i) & mask).sum().item()
            pct = 100 * count / mask.sum().item() if mask.sum() > 0 else 0
            print(f"{name}: {count} ({pct:.1f}%)")
    
    # Class distribution by timepoint
    print("\nClass distribution by timepoint:")
    for t in sorted(timepoint.unique().tolist()):
        mask = timepoint == t
        print(f"\nE{t}:")
        for i, name in enumerate(celltype_names):
            count = ((labels == i) & mask).sum().item()
            print(f"{name}: {count}")
    
    # Package data
    data = {
        'X': X,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'timepoint': timepoint,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_classes': len(celltype_names),
        'celltype_names': celltype_names,
        'temporal_cutoff': temporal_cutoff,
    }
    
    return data


"""
Purpose: Save prepared dataset
Params: dataset name and data dictionary
Return: None
"""
def save_prepared_data(dataset, data):
    output_path = f"{dataset}/{dataset}_prepared.pt"
    torch.save(data, output_path)
    print(f"\n Saved prepared data to {output_path}")


if __name__ == "__main__":
    # Prepare data with appropiate cutoff
    data = prepare_dataset(
        "blood",
        temporal_cutoff=11.0  # change for dataset type
    )
    save_prepared_data('blood', data)
    
    print("DATA PREP DONE")
