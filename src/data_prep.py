import scanpy as sc
import numpy as np
import os

def prepare_and_save_processed_data():
    adata = sc.read_h5ad("/users/hanting/project/data/GSE132188_adata.h5ad.h5")

    # Use existing PCA
    X_pca = adata.obsm["X_pca"]

    # Use existing adjacency
    A = adata.obsp["connectivities"].tocoo()

    # Compute diffmap then DPT pseudotime
    sc.tl.diffmap(adata)

    days = adata.obs["day"].astype(str)
    unique_days = sorted(days.unique())
    root_idx = np.where(days == unique_days[0])[0][0]
    adata.uns["iroot"] = root_idx

    sc.tl.dpt(adata)

    pseudotime = adata.obs["dpt_pseudotime"].values

    # Build directed edge weights
    rows = A.row
    cols = A.col
    base_weights = A.data

    alpha = 10.0
    dt = pseudotime[cols] - pseudotime[rows]
    dir_factor = 1.0 / (1.0 + np.exp(-alpha * dt))
    weights = base_weights * (0.5 + dir_factor)

    # Labels
    if "clusters_fig6_broad_final" in adata.obs:
        labels_raw = adata.obs["clusters_fig6_broad_final"].astype("category")
    else:
        labels_raw = adata.obs["clusters_fig3_final"].astype("category")

    labels = labels_raw.cat.codes.values
    num_classes = int(labels.max()) + 1

    # Save arrays
    os.makedirs("/users/hanting/project/processed", exist_ok=True)
    np.savez(
        "/users/hanting/project/processed/processed_data.npz",
        features=X_pca.astype(np.float32),
        adj_row=rows.astype(np.int32),
        adj_col=cols.astype(np.int32),
        adj_data=weights.astype(np.float32),
        labels=labels.astype(np.int32),
        pseudotime=pseudotime.astype(np.float32),
        num_cells=np.array([adata.n_obs], dtype=np.int32),
        num_features=np.array([X_pca.shape[1]], dtype=np.int32),
        num_classes=np.array([num_classes], dtype=np.int32),
    )

    print("Saved /users/hanting/project/processed/processed_data.npz")


if __name__ == "__main__":
    prepare_and_save_processed_data()
