# betabinomo_LDA_singlecells.py>

# %%
import torch
import torch.distributions as distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import pandas as pd
import numpy as np
torch.cuda.empty_cache()

from dataclasses import dataclass
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import scipy
torch.manual_seed(42)
import scipy.sparse as sp


# %%    

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

# load data 

def load_cluster_data(input_file):

   # read in hdf file 
    summarized_data = pd.read_hdf(input_file, 'df')

    #for now just look at B and T cells
    #summarized_data = summarized_data[summarized_data["cell_type"].isin(["B"])]
    print(summarized_data.cell_type.unique())
    summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
    summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()

    coo = summarized_data[["cell_id_index", "junction_id_index", "junc_count", "Cluster_Counts", "Cluster", "junc_ratio"]]

    # just some sanity checks to make sure indices are in order 
    cell_ids_conversion = summarized_data[["cell_id_index", "cell_id", "cell_type"]].drop_duplicates()
    cell_ids_conversion = cell_ids_conversion.sort_values("cell_id_index")

    junction_ids_conversion = summarized_data[["junction_id_index", "junction_id", "Cluster"]].drop_duplicates()
    junction_ids_conversion = junction_ids_conversion.sort_values("junction_id_index")
 
    # make coo_matrix for junction counts
    coo_counts_sparse = coo_matrix((coo.junc_count, (coo.cell_id_index, coo.junction_id_index)), shape=(len(coo.cell_id_index.unique()), len(coo.junction_id_index.unique())))
    coo_counts_sparse = coo_counts_sparse.tocsr()
    juncs_nonzero = pd.DataFrame(np.transpose(coo_counts_sparse.nonzero()))
    juncs_nonzero.columns = ["cell_id_index", "junction_id_index"]
    juncs_nonzero["junc_count"] = coo_counts_sparse.data

    # do the same for cluster counts 
    cells_only = coo[["cell_id_index", "Cluster", "Cluster_Counts"]].drop_duplicates()
    merged_df = pd.merge(cells_only, junction_ids_conversion)
    coo_cluster_sparse = coo_matrix((merged_df.Cluster_Counts, (merged_df.cell_id_index, merged_df.junction_id_index)), shape=(len(merged_df.cell_id_index.unique()), len(merged_df.junction_id_index.unique())))
    coo_cluster_sparse = coo_cluster_sparse.tocsr()
    cluster_nonzero = pd.DataFrame(np.transpose(coo_cluster_sparse.nonzero()))
    cluster_nonzero.columns = ["cell_id_index", "junction_id_index"]
    cluster_nonzero["cluster_count"] = coo_cluster_sparse.data

    final_data = pd.merge(juncs_nonzero, cluster_nonzero, how='outer').fillna(0)
    final_data["clustminjunc"] = final_data["cluster_count"] - final_data["junc_count"]
    final_data["juncratio"] = final_data["junc_count"] / final_data["cluster_count"] 
    final_data = final_data.merge(cell_ids_conversion, on="cell_id_index", how="left")
    return(final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion)

# %%
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

# latest version of clusters     
input_file = '/gpfs/commons/home/kisaev/leafcutter-sc/test.h5' #this should be an argument that gets fed in
    
final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion = load_cluster_data(input_file)
    
# global variables
    
N = coo_cluster_sparse.shape[0]
J = coo_cluster_sparse.shape[1]
K = 5 # should also be an argument that gets fed in
    
# initiate instance of data class containing junction and cluster indices for non-zero clusters 
junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64).to(device)
cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64).to(device)

# %%
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                  Build andata object
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

# make coo_matrix from jc_counts.count first convert from tensor to pandas dataframe
junc_counts = final_data.junc_count.values
cluster_counts = final_data.cluster_count.values
junc_ratios = final_data.juncratio.values

# get total cell coverage using final data add up all junction counts for each cell
tot_counts = final_data.groupby("cell_id_index").agg({"junc_count": "sum"})

# Create a COO sparse matrix with data, row and column indices
data = junc_ratios
rows = cell_index_tensor.cpu().numpy()
cols = junc_index_tensor.cpu().numpy()
coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N, J))
csr_sparse = coo_matrix.tocsr()

adata = ad.AnnData(csr_sparse, dtype=np.float32)
adata.obs["cell_type"] = pd.Categorical(cell_ids_conversion.cell_type)  # Categoricals are preferred for efficiency
adata.obs["total_counts"] = tot_counts.junc_count.values  
adata.var["junction_ids"] = pd.Categorical(junction_ids_conversion.junction_id)  # Categoricals are preferred for efficiency
adata.var["junction_cluster"] = pd.Categorical(junction_ids_conversion.Cluster)  # Categoricals are preferred for efficiency

adata.obs_names = adata.obs["cell_type"]
adata.var_names = adata.var["junction_ids"] 
sc.settings.verbosity = 0             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.pp.highly_variable_genes(adata, min_mean=0.005, min_disp=0.05) #Expects logarithmized data
adata.raw = adata
adata = adata[:, adata.var.highly_variable]

sc.tl.pca(adata, svd_solver='arpack')

print(sc.pl.pca_variance_ratio(adata, log=True))
print(sc.pl.pca(adata, color="cell_type"))

sc.pl.violin(adata, ['10_100540782_100544950', '10_102152530_102157018'], groupby='cell_type')

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata)

sc.tl.paga(adata)
sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')

sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden', 'cell_type'])

sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)