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

def check_cell_pairs(bbmix_results, row_colors, col_colors, cell_type_colors, num_cells_to_plot=4000):
    '''
    This function checks the consistency of cell state assignments across random trials
    by checking the consistency of cell pairs and whether they get assigned to same state

    Input: 
        bbmix_results = list of results from each trial (results) all output of betabinomo_mix_singlecells
        the third item of each result is the phi_f tensor (num_cells x K)
        num_cells_to_plot = number of cells to plot in heatmap (default = 4000)

    Output:

    '''

    # extract PHI_f from every trial in num_trials
    num_trials = len(bbmix_results)
    N = bbmix_results[0][3].shape[0]
    all_iters_PHI_f = [ result[3] for result in bbmix_results ]
    i = 0

    print("Running! will make heatmap with cell types")
    
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
    print("Got a cell by cell matrix for each trial indicating whether cells were coassigned")

    # Calculate sum over matrices across all iterations to find cell-cell pairs that are MOST misassigned 
    print("Getting sum of matrices across all iterations")
    sum_matrices = sum(all_iters_results)
    print("The number of trials is: ", num_trials)

    # Count occurrences of each value
    unique_values, counts = np.unique(sum_matrices, return_counts=True)
    
    # Create a DataFrame to store the counts
    df = pd.DataFrame({'Value': unique_values, 'Count': counts})
    print(df.sort_values(by=['Count'], ascending=False).head(10))

    cluster = sns.clustermap(
    data=sum_matrices[0:num_cells_to_plot, 0:num_cells_to_plot],
    method='complete',
    cmap="viridis",
    annot=False,
    fmt=".2f",
    xticklabels=False,
    yticklabels=False,
    figsize=(10, 8),
    center=0,
    row_colors=row_colors[0:num_cells_to_plot],  # Apply row colors
    col_colors=col_colors[0:num_cells_to_plot]   # Apply column colors
    )

    # Create a legend using custom legend handles   
    # Create a legend using custom legend handles with smaller font size
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=cell_type)
                  for cell_type, color in cell_type_colors.items()]
    legend = plt.legend(handles=legend_handles, title='Cell Types', loc='upper right', fontsize='small')

    plt.title('Cell-Cell co-assignments for all cells')
    # Show the plot
    plt.show()

    return(sum_matrices)


def consensus_clustering(results):

    num_trials = len(results)
    N = results[0][3].shape[0]
    all_iters_PHI_f = [ result[3] for result in results ]
    i = 0

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
    print("Got a cell by cell matrix for each trial indicating whether cells were coassigned")
    # Calculate sum over matrices across all iterations to find cell-cell pairs that are MOST misassigned 
    print("Getting sum of matrices across all iterations")
    sum_matrices = sum(all_iters_results)
    return(sum_matrices)