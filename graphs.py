import scanpy as sc

adata = sc.read_h5ad("/users/hanting/csci_2952g/GSE132188_harmony.h5ad")

print(adata.obsm.keys())

sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)

# save the UMAP as a PNG
sc.pl.umap(
    adata,
    color=['day', 'clusters_fig3_final'],
    save="_GSE132188.png"   # Scanpy adds this suffix to 'figures/umap'
)
