import scanpy as sc
import numpy as np
import harmonypy as hm


adata = sc.read_h5ad("/users/hanting/csci_2952g/GSE132188_adata.h5ad.h5")


# Filter cells and genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)


# Mark mitochondrial genes (mouse uses 'mt-')
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)


# Filter out cells with high mt% or too many genes
adata = adata[adata.obs["pct_counts_mt"] < 10, :]
adata = adata[adata.obs["n_genes_by_counts"] < 5000, :]


# Normalize, log-transform, and find highly variable genes
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)


# print(adata.obs.columns)




sc.pp.pca(adata, n_comps=50)
ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'day')


adata.obsm['X_pca_harmony'] = ho.Z_corr.T


adata.write("GSE132188_harmony.h5ad")




# print(adata.obsm.keys())
adata.to_df().to_csv("GSE132188_expression.csv")
adata.obs.to_csv("GSE132188_metadata.csv")


# print(adata.layers.keys())


sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['day', 'clusters_fig3_final'])





