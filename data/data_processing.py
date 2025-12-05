import scanpy as sc
import pandas as pd
from scipy.io import mmread
import numpy as np

"""
Purpose: Verify the matrices and counts for the trajectory chosen
Params: dataset name (blood or neurons)
Return: introns, extrons, var, obs
"""
def verify_trajectory(dataset): 

    #1. Load matrices ( cells x genes)
    print("\nLoading count matrices:")
    exon = mmread(f"{dataset}/exp_exon.mtx").tocsr()
    intron = mmread(f"{dataset}/exp_intron.mtx").tocsr()
    print(f"Spliced: {exon.shape}")
    print(f"Unspliced: {intron.shape}")

    #2. Load cell metadata
    print("\nLoading cell metadata:")
    obs = pd.read_csv(f"{dataset}/obs.csv", index_col=0)
    print(f"Cells: {len(obs):,}")
    print(f"Columns: {list(obs.columns)}")

    #3. Load gene metadata
    print("\nLoading gene metadata:")
    var = pd.read_csv("E9.5_to_E13.5_var.csv", index_col=0) #same gene file across 
    print(f"Genes: {len(var):,}")

    #4. Verify alignment
    print("\nVerify the dimensions dimensions:")
    assert exon.shape[0] == len(obs), f"Cell mismatch: {exon.shape[0]} vs {len(obs)}"
    assert exon.shape[1] == len(var), f"Gene mismatch: {exon.shape[1]} vs {len(var)}"
    print("All dimensions aligned!")

    return intron, exon, var, obs


"""
Purpose: Convert trajectory information into adata
Params: intron, extron, var, obs files
Return: adata object
"""

def convert_trajectory(intron, exon, var, obs):
    # Get the adata file for easy access
    adata = sc.AnnData(X=(exon + intron).tocsr())

    # Add cell metadata
    adata.obs = obs.copy()
    adata.obs_names = obs.index.astype(str)

    # Add gene metadata  
    adata.var = var.copy()
    adata.var_names = var['gene_short_name'].values
    adata.var_names_make_unique()

    # Add spliced/unspliced layers for velocity
    adata.layers['spliced'] = exon
    adata.layers['unspliced'] = intron


    # Get numeric timepoint from 'day' column
    adata.obs['timepoint_str'] = adata.obs['day'].astype(str)
    adata.obs['timepoint'] = (
        adata.obs['day']
        .str.replace('E', '', regex=False)
        .str.replace('b', '', regex=False)  # Handle E8.5b -> 8.5
        .astype(float)
    )

    return adata


"""
Purpose: Summarize and save data
Params: adata object, dataset name
Return: None
"""
def save_adata(adata, dataset):
    # Print final summary
    print("\nFINAL ANNDATA SUMMARY")
    print("-" * 50)
    print(f"Cells: {adata.n_obs:,}")
    print(f"Genes: {adata.n_vars:,}")
    print(f"Layers: {list(adata.layers.keys())}")
    print(f"Obs cols: {list(adata.obs.columns)}")

    print(f"\nTimepoints:")
    print(adata.obs['timepoint'].value_counts().sort_index())

    print(f"\nCell types:")
    print(adata.obs['celltype'].value_counts())

    # Save file
    adata.write(f"{dataset}/{dataset}_data.h5ad")
    print(f"Saved: {dataset}/{dataset}_data.h5ad")
"""
Purpose: create and save file adata file!
Params: dataset name
Return: None
"""
def create_adata(dataset):
    intron, extron, var, obs = verify_trajectory(dataset)
    adata = convert_trajectory(intron, extron, var, obs)
    save_adata(adata, dataset)

def main():
    # blood files
    create_adata('blood')

if __name__ == "__main__":
    main()