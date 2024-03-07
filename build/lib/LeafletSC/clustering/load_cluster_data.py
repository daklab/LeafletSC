import torch
import torch.distributions as distributions

import pandas as pd
import numpy as np
import copy
torch.cuda.empty_cache()

from dataclasses import dataclass
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import argparse
from scipy.stats import binom
from tqdm import tqdm
import sklearn.cluster
from scipy.stats import binom

@dataclass
class IndexCountTensor():
    ycount_lookup: torch.Tensor
    tcount_lookup: torch.Tensor 
    ycount_lookup_T: torch.Tensor
    tcount_lookup_T: torch.Tensor 
        
def make_torch_data(final_data, **float_type):

    device = float_type["device"]
            
    # note these are staying on the CPU! 
    print("The number of cells going into training data is:")
    print(len(final_data.cell_id_index.unique()))

    # initiate instance of data class containing junction and cluster indices for non-zero clusters 
    junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64, device=device)
    cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64, device=device)
    print(len(cell_index_tensor.unique()))
    
    ycount = torch.tensor(final_data.junc_count.values, **float_type) 
    tcount = torch.tensor(final_data.clustminjunc.values, **float_type)

    M = len(cell_index_tensor)

    ycount_lookup = torch.sparse_coo_tensor(
        torch.stack([cell_index_tensor, junc_index_tensor]), 
        ycount).to_sparse_csr()

    ycount_lookup_T = torch.sparse_coo_tensor( # this is a hack since I can't figure out tranposing sparse matrices :( maybe will work in newer pytorch? 
        torch.stack([junc_index_tensor, cell_index_tensor]), 
        ycount).to_sparse_csr()

    tcount_lookup = torch.sparse_coo_tensor( 
        torch.stack([cell_index_tensor, junc_index_tensor]), 
        tcount).to_sparse_csr()
    
    tcount_lookup_T = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor, cell_index_tensor]), 
        tcount).to_sparse_csr()

    my_data = IndexCountTensor(ycount_lookup, tcount_lookup, ycount_lookup_T, tcount_lookup_T)
    
    return cell_index_tensor, junc_index_tensor, my_data


# load data 

def load_cluster_data(input_file=None, input_folder=None, celltypes = None, num_cells_sample = None, max_intron_count=None, remove_singletons=True, has_genes="no"):

    """
    Load and preprocess cluster data from HDF5 files, either from a single file or a directory of files.
    It filters data based on cell types, samples a specified number of cells, removes outliers based on max intron count,
    and constructs sparse matrices for junction and cluster counts.
    
    Parameters:
    - input_file (str, optional): Path to a single HDF5 file to load.
    - input_folder (str, optional): Path to a folder containing multiple HDF5 files to load and concatenate.
    - celltypes (list of str, optional): List of cell types to include in the analysis.
    - num_cells_sample (int, optional): Number of cells to randomly sample from the dataset.
    - max_intron_count (int, optional): Maximum allowable intron count for filtering outliers.
    - has_genes (str, optional): Indicates whether gene IDs are included in the dataset ('yes' or 'no').
    
    Returns:
    - final_data (DataFrame): Processed data including junction and cluster counts, cell types, and ratios.
    - coo_counts_sparse (csr_matrix): Sparse matrix of junction counts.
    - coo_cluster_sparse (csr_matrix): Sparse matrix of cluster counts.
    - cell_ids_conversion (DataFrame): Mapping of cell_id_index to cell_id and cell_type.
    - junction_ids_conversion (DataFrame): Mapping of junction_id_index to junction_id, and optionally gene_id.
    """

   # read in hdf file 
    if input_file:
        summarized_data = pd.read_hdf(input_file, 'df')

    print("Reading in data from folder ...")
    
    if input_folder:
        print(input_folder)
        # read each hdf file in folder and concatenate
        files = os.listdir(input_folder)

        df_list = []
        for file in files:
            if file.endswith(".h5"):
                path_with_quotes = input_folder + file
                fixed_path = path_with_quotes.replace("'", "")
                df = pd.read_hdf(fixed_path, 'df')
                df_list.append(df)
            else:
                pass

        # concatenate all dataframes
        summarized_data = pd.concat(df_list, ignore_index=True)
    
    print("Finished reading in data from folder ...")

    #if want to look at only specific subset of cell types 
    if celltypes:
        print("Looking at only specific cell types ..." + str(celltypes))
        summarized_data = summarized_data[summarized_data["cell_type"].isin(celltypes)]
        # redo cell id indexing in summarized data, assign cell id index to each cell id
        summarized_data["cell_id_index"] = pd.factorize(summarized_data.cell_id)[0]
        # same for junction id indexing
        summarized_data["junction_id_index"] = pd.factorize(summarized_data.junction_id)[0]

    if num_cells_sample:
        summarized_data = summarized_data.sample(n=num_cells_sample)
   
    if remove_singletons:
        print("Removing singletons ...")
        # Get unique junction-clusters pairs remove clusters that only have one junction
        junctions_per_cluster = summarized_data[["Cluster", "junction_id"]].drop_duplicates()
        junctions_per_cluster = junctions_per_cluster.groupby("Cluster").size().reset_index(name='counts')
        junctions_per_cluster = junctions_per_cluster[junctions_per_cluster["counts"] > 1]
        print("Number of junctions before removing singletons: ", summarized_data.junction_id_index.max())
        summarized_data = summarized_data[summarized_data["Cluster"].isin(junctions_per_cluster["Cluster"])]
        # redo cell id indexing in summarized data, assign cell id index to each cell id
        summarized_data["cell_id_index"] = pd.factorize(summarized_data.cell_id)[0]
        # same for junction id indexing
        summarized_data["junction_id_index"] = pd.factorize(summarized_data.junction_id)[0]
        print("Number of junctions after removing singletons: ", summarized_data.junction_id_index.max())

    print("The number of unique cell types in the data is: ", len(summarized_data["cell_type"].unique()))
    print("The number of unique cells in the data is: ", len(summarized_data["cell_id"].unique()))
    print("The number of unique junctions in the data is: ", len(summarized_data["junction_id"].unique()))

    # assert no more intron clusters of size one
    junctions_per_cluster = summarized_data[["Cluster", "junction_id"]].drop_duplicates()
    junctions_per_cluster = junctions_per_cluster.groupby("Cluster").size().reset_index(name='counts')
    junctions_per_cluster = junctions_per_cluster[junctions_per_cluster["counts"] == 1]
    assert len(junctions_per_cluster) == 0

    # remove outliers 
    if max_intron_count:
        print("The maximum junction count was initially: ", summarized_data["Cluster_Counts"].max())
        clusts_remove = summarized_data[summarized_data["Cluster_Counts"] > max_intron_count].Cluster.unique()
        # remove all clusters that these junctions belong to 
        print(len(clusts_remove))
        summarized_data = summarized_data[~summarized_data["Cluster"].isin(clusts_remove)]
        print("The maximum junction count is now: ", summarized_data["Cluster_Counts"].max())
        # renumber cell_id_index and junction_id_index
        summarized_data["cell_id_index"] = pd.factorize(summarized_data.cell_id)[0]
        summarized_data["junction_id_index"] = pd.factorize(summarized_data.junction_id)[0]

    coo = summarized_data[["cell_id_index", "junction_id_index", "junc_count", "Cluster_Counts", "Cluster", "junc_ratio"]]

    # just some sanity checks to make sure indices are in order 
    cell_ids_conversion = summarized_data[["cell_id_index", "cell_id", "cell_type"]].drop_duplicates()
    cell_ids_conversion = cell_ids_conversion.sort_values("cell_id_index")

    junction_ids_conversion = summarized_data[["junction_id_index", "junction_id", "Cluster"]].drop_duplicates()
    junction_ids_conversion = junction_ids_conversion.sort_values("junction_id_index")

    if has_genes == "yes":
        junction_ids_conversion = summarized_data[["junction_id_index", "junction_id", "Cluster", "gene_id"]].drop_duplicates()
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

    print("The number of junctions in the data is: ", len(final_data.junction_id_index.unique()))
    print("The number of cells in the data is: ", len(final_data.cell_id_index.unique()))
    print("The number of cell types in the data is: ", len(final_data.cell_type.unique()))
    
    return(final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion)
