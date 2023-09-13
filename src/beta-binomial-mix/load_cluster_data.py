
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os

# load data 

def load_cluster_data(input_file=None, input_folder=None, celltypes = None, num_cells_sample = None):

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
        summarized_data = summarized_data[summarized_data["cell_type"].isin(celltypes)]

    if num_cells_sample:
        summarized_data = summarized_data.sample(n=num_cells_sample)
   
    print(summarized_data["cell_type"].unique())
    print(len(summarized_data["cell_id"].unique()))
    print(summarized_data.junction_id_index.max())

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

    print("The number of junctions in the data is: ", len(final_data.junction_id_index.unique()))
    print("The number of cells in the data is: ", len(final_data.cell_id_index.unique()))
    print("The number of cell types in the data is: ", len(final_data.cell_type.unique()))
    
    return(final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion)
