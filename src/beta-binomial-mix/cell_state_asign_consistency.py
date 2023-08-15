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
import itertools

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, hamming

import os 
import argparse
from tqdm import tqdm

import sklearn.cluster

# this python script contains functions for evaluating how consistent the cell 
# state assignments are across random trials (with varying initializations)

# input = list of results from each trial (results) all output of betabinomo_mix_singlecells
# the third item of each result is the phi_f tensor (cell x cell_state)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for evaluating consistency of cell state assignments 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_cell_pairs(bbmix_results, plot_heatmap = False):
    '''
    This function checks the consistency of cell state assignments across random trials
    by checking the consistency of cell pairs and whether they get assigned to same state

    Input: 
        bbmix_results = list of results from each trial (results) all output of betabinomo_mix_singlecells
        the third item of each result is the phi_f tensor (num_cells x K)

    Output:

    '''

    # extract PHI_f from every trial in num_trials
    num_trials = len(bbmix_results)
    N = bbmix_results[0][3].shape[0]
    all_iters_PHI_f = [ result[3] for result in bbmix_results ]
    i = 0

    # Create an empty list to store DataFrames from each iteration
    dfs_list = []

    for PHI_var in all_iters_PHI_f:

        probability_tensor = PHI_var

        # Create an array with cell IDs (e.g., cell_0, cell_1, ..., cell_(N-1))
        cell_ids = np.arange(probability_tensor.shape[0])
        cell_ids = [cell_id for cell_id in cell_ids]

        # Get the cluster IDs for each cell based on the maximum probability
        cluster_ids = np.argmax(probability_tensor, axis=1)

        # Create a DataFrame with the cell_id, cluster_id, and probability columns
        df = pd.DataFrame({"cell_id": cell_ids, "cluster_id": cluster_ids})

        # Add column with iteration number
        df["iteration"] = i
        i += 1
        # Append the DataFrame to the list
        dfs_list.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    concatenated_df = pd.concat(dfs_list, ignore_index=True)
    print("Got all cells and their assignments based on PHI_f in each trial!")

    # initiate list to save results for each iteration

    all_iters_results = [None] * num_trials

    for trial in range(num_trials):
        # Find the unique clusters for each cell_id
        cell_by_cell_matrix = np.zeros((N, N))

        clusters = concatenated_df.loc[concatenated_df["iteration"] == trial, ["cell_id", "cluster_id"]]
        unique_clusters = clusters.set_index('cell_id')['cluster_id'].to_dict()

        # Fill the cell_by_cell_matrix using numpy indexing
        for cell_id, cluster_id in unique_clusters.items():
            cell_by_cell_matrix[cell_id, cell_id] = 1
            same_cluster_cells = clusters[clusters["cluster_id"] ==  cluster_id].cell_id.values
            cell_by_cell_matrix[cell_id, same_cluster_cells] = 1

        all_iters_results[trial] = cell_by_cell_matrix

    # get all pairs of num_trials 
    all_pairs = list(itertools.combinations(range(num_trials), 2))
    print("Got a cell by cell matrix for each trial indicating whether cells were coassigned")

    # Calculate sum over matrices across all iterations to find cell-cell pairs that are MOST misassigned 
    # These would have the lowest scores, the higher the scores (more 1s) the more consistent the cell state assignments are
    # Then do hierarchical clustering on this 
    # Create an empty list to store DataFrames from each iteration
    nonzero_dfs_list = []
    zero_dfs_list = []

    for pair in tqdm(all_pairs):
        ## assess similarity between iterations
        distance_matrix = (all_iters_results[pair[0]] - all_iters_results[pair[1]])
        # Get indices of non-zero elements
        non_zero_indices = np.argwhere(distance_matrix != 0)
        non_zero_df = pd.DataFrame(non_zero_indices, columns=["cell_1", "cell_2"])
        non_zero_df["pair"] = str(pair)
        nonzero_dfs_list.append(non_zero_df)
        
        zero_indicies = np.argwhere(distance_matrix == 0)
        zero_df = pd.DataFrame(zero_indicies, columns=["cell_1", "cell_2"])
        zero_df["pair"] = str(pair)
        zero_dfs_list.append(zero_df)

        #unique, counts = np.unique(x, return_counts=True)
        # turn unique, counts into dataframe 
        #df = pd.DataFrame({'unique': unique, 'counts': counts})
        # get percentage for counts 
        #df['percentage'] = df['counts']/df['counts'].sum()
        #df["pair"] = str(pair)
        #dfs_list.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    #concatenated_iters_comp = pd.concat(dfs_list, ignore_index=True)
    #concatenated_iters_comp = concatenated_iters_comp[["unique", "pair", "percentage"]]

    # turn into wide format for plotting heatmap
    # concatenated_iters_comp_wide = concatenated_iters_comp.pivot(index='pair', columns='unique', values='percentage')
    #concatenated_iters_comp.sort_values(by=['percentage'], inplace=True, ascending=False)

    #print(f"The minimum percentage of matching cell pairs across all trials is {concatenated_iters_comp[concatenated_iters_comp['unique'] == 0]['percentage'].min().round(2)}")
    num_matrices = len(all_iters_results)

    if plot_heatmap:
        print("Making plots!")
        for i in range(num_matrices):
            for j in range(i + 1, num_matrices):
                dist_matrix = all_iters_results[i] - all_iters_results[j]
                dist_matrix = dist_matrix[0:1000, 0:1000]
                # Create a clustermap with modified size and centering at zero
                cluster_map = sns.clustermap(data=dist_matrix, method='complete', cmap="viridis",
                             annot=False, fmt=".2f", xticklabels=False, yticklabels=False,
                             figsize=(6, 6), center=0)
                plt.title('Distance matrix between trials ' + str(i) + ' and ' + str(j))
                # Display the clustermap
                plt.show()

    print("Combining all the non-zero cell pairs = missassigned cell pairs across all trials")
    nonzero_dfs = pd.concat(nonzero_dfs_list)
    # make new column combining the first two columns 
    nonzero_dfs['cell_pair'] = nonzero_dfs['cell_1'].astype(str) + '_cell_' + nonzero_dfs['cell_2'].astype(str)

    print("Combining all the zero cell pairs = co-assigned cell pairs across all trials")
    zero_dfs = pd.concat(zero_dfs_list)
    # make new column combining the first two columns 
    zero_dfs['cell_pair'] = zero_dfs['cell_1'].astype(str) + '_cell_' + zero_dfs['cell_2'].astype(str)

    print("check_cell_pairs done running!")
    return(nonzero_dfs, zero_dfs, all_iters_results)


# generating the heatmap plot should be its own function
def plot_heatmap(all_iters_results):
    # input = list of cellxcell matrices from each trial 
    # if a matrix cell value = 1 then it is in the same cluster as the other cell it's being compared to
    # we then subtract the matrices across runs to see how oftens the cell pairs where co-assigned 
    # we then plot the heatmap of the difference matrix
    num_matrices = len(all_iters_results)
    print("The number of trials is: ", num_matrices)
    print("Making plots!")
    for i in range(num_matrices):
        for j in range(i + 1, num_matrices):
            dist_matrix = all_iters_results[i] - all_iters_results[j]
            dist_matrix = dist_matrix[0:2000, 0:2000]
            # Create a clustermap with modified size and centering at zero
            cluster_map = sns.clustermap(data=dist_matrix, method='complete', cmap="viridis",
                         annot=False, fmt=".2f", xticklabels=False, yticklabels=False,
                         figsize=(6, 6), center=0)
            plt.title('Distance matrix between trials ' + str(i) + ' and ' + str(j))
            # Display the clustermap
            plt.show()
