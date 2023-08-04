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

import os 
import argparse

import sklearn.cluster

# this python script contains functions for evaluating how consistent the cell 
# state assignments are across random trials (with varying initializations)

# input = list of results from each trial (results) all output of betabinomo_mix_singlecells
# the third item of each result is the phi_f tensor (cell x cell_state)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for evaluating consistency of cell state assignments 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_cell_pairs(bbmix_results):
    '''
    This function checks the consistency of cell state assignments across random trials
    by checking the consistency of cell pairs and whether they get assigned to same state

    Input: 
        bbmix_results = list of results from each trial (results) all output of betabinomo_mix_singlecells
        the third item of each result is the phi_f tensor (num_cells x K)

    Output:

    '''

    # extract PHI_f from every trial in num_trials
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
    # Create an empty list to store DataFrames from each iteration
    dfs_list = []

    for pair in tqdm(all_pairs):
        ## assess similarity between iterations
        x = (all_iters_results[pair[0]] - all_iters_results[pair[1]])
        unique, counts = np.unique(x, return_counts=True)
        # turn unique, counts into dataframe 
        df = pd.DataFrame({'unique': unique, 'counts': counts})
        # get percentage for counts 
        df['percentage'] = df['counts']/df['counts'].sum()
        df["pair"] = str(pair)
        dfs_list.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    concatenated_iters_comp = pd.concat(dfs_list, ignore_index=True)

    concatenated_iters_comp = concatenated_iters_comp[["unique", "pair", "percentage"]]

    # turn into wide format for plotting heatmap
    concatenated_iters_comp_wide = concatenated_iters_comp.pivot(index='pair', columns='unique', values='percentage')

    concatenated_iters_comp.sort_values(by=['percentage'], inplace=True, ascending=False)

    print(f"The minimum percentage of matching cell pairs across all trials is {concatenated_iters_comp[concatenated_iters_comp['unique'] == 0]['percentage'].min().round(2)}")

