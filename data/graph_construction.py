import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import scipy.sparse as sp
import scanpy as sc

"""
Purpose: create the graph as a pytorch graph and save 
Params: dataset name, adata object, preprocessing params
Return: None
"""

def create_graph(dataset, adata, n_neighbors=15, n_comps=50, n_pcs=50, n_top_genes=2000):

    """Data preprocessing for PCA"""
    # Standard filtering and normalization
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    print(f"Highly variable genes: {adata.var['highly_variable'].sum()}")

    # Prevent densification issue
    adata = adata[:, adata.var["highly_variable"]].copy()
    
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comps)
    expr_pca = adata.obsm['X_pca']  # [n_cells, 50]
    print(f"PCA shape: {expr_pca.shape}")
    
    """Adding Node Features"""
    x = torch.tensor(expr_pca, dtype=torch.float32)
    print(f"Node features: {x.shape[1]} (PCA components)")
    
    # Timepoint as separate attribute
    if adata.obs['timepoint'].dtype == 'object' or adata.obs['timepoint'].dtype.name == 'category':
        # Convert "E8.5" to 8.5
        timepoints = adata.obs['timepoint'].astype(str).str.replace('E', '').astype(float)
    else:
        timepoints = adata.obs['timepoint'].values
    timepoints_norm = (timepoints - timepoints.mean()) / timepoints.std()
    print(f"Timepoints: {np.unique(timepoints)}")
    print(f"Normalized Timepoints: {np.unique(timepoints_norm)}")
    
    """Adding Edges"""
    # Compute neighbors w/ PCA
    print("Calculating the neighbors with PCA")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca') 
    connectivities = adata.obsp['connectivities']
    rows, cols = connectivities.nonzero()
    
    # cosine similarity on PCA embeddings (vectorized for speed)
    print("Running cosine similarity on neighbors")
    source_embeddings = expr_pca[rows]  # [n_edges, 50]
    target_embeddings = expr_pca[cols]  # [n_edges, 50]
    
    # Compute all similarities at once
    edge_weights = np.sum(source_embeddings * target_embeddings, axis=1) / (
        np.linalg.norm(source_embeddings, axis=1) * np.linalg.norm(target_embeddings, axis=1) + 1e-8
    )
    
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    
    print(f"Edges: {edge_index.shape[1]:,}")
    print(f"Edge weights: [{edge_weights.min():.3f}, {edge_weights.max():.3f}]")
    
    """Adding Metadata (Cell Labels)"""
    celltype_cat = adata.obs['celltype'].astype('category')
    celltype_codes = torch.tensor(celltype_cat.cat.codes.values, dtype=torch.long)
    celltype_names = list(celltype_cat.cat.categories)
    print(f"Cell types: {len(celltype_names)}")
    
    """Create Graph"""
    graph = Data(
        x=x,                              # [n_cells, 50] PCA embeddings
        edge_index=edge_index,            # [2, n_edges]
        edge_attr=edge_attr,              # [n_edges, 1] cosine similarity
        timepoint=timepoints,             # [n_cells] raw timepoint
        timepoint_norm=timepoints_norm,   # [n_cells] normalized timepoint
        celltype=celltype_codes,          # [n_cells] integer cell type labels
    )
    
    # Store metadata (not tensors)
    graph.celltype_names = celltype_names
    graph.n_cells = adata.n_obs
    graph.n_pcs = n_comps
    
    print("Graph created:", graph)
    
    # Save graph as pytorch graph and rewrite adata
    torch.save(graph, f"{dataset}/{dataset}_graph.pt")
    print(f"Saved graph to {dataset}/{dataset}_graph.pt")
    
    adata.write(f"{dataset}/{dataset}_data.h5ad")
    print(f"Updated .h5ad file {dataset}/{dataset}_data.h5ad")

# Load the adata BLOOD
adata = sc.read_h5ad("blood/blood_data.h5ad")
print(f"Loaded: {adata.n_obs:,} cells and {adata.n_vars:,} genes")

create_graph('blood',adata, n_neighbors=15, n_comps=50, n_pcs=50, n_top_genes=2000)