import scvelo as scv
import scanpy as sc
import numpy as np
import torch

"""
Purpose: Compute RNA velocity using scVelo
Params: adata object, preprocessing parameters
Return: adata with velocity computed
"""
def compute_velocity(adata, min_shared_counts=20, n_top_genes=2000, n_pcs=50, n_neighbors=15):
    print("COMPUTING RNA VELOCITY")
    
    # scVelo preprocessing
    print("\nPreprocessing for velocity!")
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    
    # Compute velocity
    print("\nComputing velocity!")
    scv.tl.velocity(adata, mode='stochastic')
    
    # Build velocity graph
    print("\nBuilding velocity graph!")
    scv.tl.velocity_graph(adata)
    
    # Get the velocity in gene space
    velocity_genes = adata.layers['velocity']
    
    # Get the PCA loadings (genes x PCs)
    pca_loadings = adata.varm['PCs']  # [n_genes, n_pcs]
    
    # Project: velocity_pca = velocity_genes @ pca_loadings
    # Handle sparse matrix if needed
    if hasattr(velocity_genes, 'toarray'):
        velocity_genes = velocity_genes.toarray()
    
    # Replace NaNs with 0 (some genes have no velocity)
    velocity_genes = np.nan_to_num(velocity_genes, nan=0.0)
    
    velocity_pca = velocity_genes @ pca_loadings
    adata.obsm['velocity_pca'] = velocity_pca
    
    print(f"\nVelocity computed!")
    print(f"  Velocity (genes): {adata.layers['velocity'].shape}")
    print(f"  Velocity (PCA):   {velocity_pca.shape}")
    
    return adata


"""
Purpose: Reweight graph edges using velocity alignment
Params: graph object, adata with velocity, alpha hyperparameter
Return: graph with updated edge weights
"""
def reweight_edges(graph, adata, alpha=0.5):
    print("REWEIGHTING EDGES WITH VELOCITY")
    print("-" * 50)
    
    # Get velocity in PCA space
    velocity_pca = adata.obsm['velocity_pca']
    
    # Get graph data as numpy
    x = graph.x.numpy()
    edge_index = graph.edge_index.numpy()
    w_cosine = graph.edge_attr.numpy().flatten()
    
    print(f"\nAlpha: {alpha}")
    print(f"Edges: {edge_index.shape[1]:,}")
    print("Computing velocity alignment")
    
    rows = edge_index[0]
    cols = edge_index[1]
    alignments = np.zeros(edge_index.shape[1])
    
    v_i = velocity_pca[rows]             # [E, d]
    d_ij = x[cols] - x[rows]             # [E, d]

    # dot products for ALL edges
    dots = np.sum(v_i * d_ij, axis=1)

    # norms
    norm_v = np.linalg.norm(v_i, axis=1)
    norm_d = np.linalg.norm(d_ij, axis=1)

    # cosine between velocity and displacement
    alignments = dots / (norm_v * norm_d + 1e-8)
    
    # Compute new weights
    w_new = (1 - alpha) * w_cosine + alpha * alignments
    
    # Print stats
    print(f"\nAlignment stats:")
    print(f"  Range: [{alignments.min():.3f}, {alignments.max():.3f}]")
    print(f"  Mean:  {alignments.mean():.3f}")
    
    print(f"\nEdge weight stats:")
    print(f"  Before: [{w_cosine.min():.3f}, {w_cosine.max():.3f}]")
    print(f"  After:  [{w_new.min():.3f}, {w_new.max():.3f}]")
    
    # Update graph
    graph.edge_attr = torch.tensor(w_new, dtype=torch.float32).unsqueeze(1)
    graph.velocity_alignment = torch.tensor(alignments, dtype=torch.float32)
    
    print(f"\nEdges reweighted!")
    
    return graph


"""
Purpose: Save velocity adata and reweighted graph
Params: adata, graph, dataset name
Return: None
"""
def save_velocity_data(adata, graph, dataset):
    print("\nSAVING NOW")
    
    adata.write(f"{dataset}/{dataset}_velocity_data.h5ad")
    print(f"Saved: {dataset}/{dataset}_velocity_data.h5ad")
    
    torch.save(graph, f"{dataset}/{dataset}_graph_velocity.pt")
    print(f"Saved: {dataset}/{dataset}_graph_velocity.pt")


"""
Purpose: Full pipeline to compute velocity and reweight graph
Params: adata, dataset name, alpha, preprocessing params
Return: adata, graph (both updated)
"""
def velocity_pipeline(adata, dataset, alpha=0.5, min_shared_counts=20, n_top_genes=2000, n_pcs=50, n_neighbors=15):
    # Load existing graph
    graph = torch.load(f"{dataset}/{dataset}_graph.pt", weights_only=False)
    print(f"Loaded graph: {graph.x.shape[0]:,} nodes, {graph.edge_index.shape[1]:,} edges")
    
    # Compute velocity
    adata = compute_velocity(
        adata,
        min_shared_counts=min_shared_counts,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors
    )
    
    # Reweight edges
    graph = reweight_edges(graph, adata, alpha=alpha)
    
    # Save
    save_velocity_data(adata, graph, dataset)
    
    return adata, graph


# Load the adata (blood)
adata = sc.read_h5ad("blood/blood_data.h5ad")
new_adata, new_graph = velocity_pipeline(adata, 'blood', alpha=0.7)