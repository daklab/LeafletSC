# %%
import torch
import torch.utils.data as data 
import torch.distributions as distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import pdb

import pandas as pd
import numpy as np
import copy

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import sleep
from scipy.sparse import csr_matrix
from typing import Dict

torch.manual_seed(42)

# Path: src/beta-binomial-lda/02_simulate_cellstate_data.py
#________________________________________________________________________________________________________________

# We want to create a simulated dataset of junction and intron cluster counts across single cells 
# which resembles observed levels of coverage and splicing. We simulate it to have cell states defined by 
# differences in splice junction counts. The goal is see if the model defined in 03_betabinom_LDA_singlecells.py
# can recover the cell states.

# Define class for generative model of junction and intron cluster counts
#________________________________________________________________________________________________________________

# simulate Beta_k=1 to look very different from Beta_k=2

# Define data classes 
class Simulate_DataLoader():
    
    # data loader for csr_matrix

    def __init__(self, csr_mat1, csr_mat2):
        self.csr_mat1 = csr_mat1
        self.csr_mat2 = csr_mat2
        
    def __getitem__(self, idx):
        mat1_row = np.array(self.csr_mat1[idx])
        mat2_row = np.array(self.csr_mat2[idx])
        return mat1_row, mat2_row
    
    def __len__(self):
        return self.csr_mat1.shape[0]

class JunctionClusterCounts():
    
    def __init__(self, num_cells: int, num_junctions: int, num_states: int, state_params: Dict):
        self.num_cells = num_cells
        self.num_junctions = num_junctions
        self.num_states = num_states
        
        # sample parameters for beta distributions for each state
        self.alpha_params, self.beta_params = self.generate_beta_params(state_params)
        
        # sample proportions of cell states 
        self.theta = torch.rand(num_cells, num_states)
        
        #sample labels for junctions (which cell state they get assigned to given the proportions)
        self.Z = torch.zeros(num_cells, num_junctions, dtype=torch.long)
        
        #given probability of success for each junction given the assigned cell state, 
        # sample counts using a binomial distribution
        self.counts = torch.zeros(num_cells, num_junctions)

        # generate unique number of trials for each junction in every cell
        # most of these should be zeroes 
        num_trials = torch.zeros(num_cells, num_junctions)
        num_elements = num_cells * num_junctions
        num_non_zero = int(0.01 * num_elements)
        indices = torch.randperm(num_elements)[:num_non_zero]
        num_trials.view(-1)[indices] =torch.randint(low=1, high=10, size=(num_non_zero,)).float()
        self.num_trials = num_trials
    
    # this is not technically necessary because state_params already has this infor 
    def generate_beta_params(self, state_params: Dict):
        alpha_params = torch.zeros(self.num_states, self.num_junctions)
        beta_params = torch.zeros(self.num_states, self.num_junctions)
        for state_idx in range(self.num_states):
            alpha_params[state_idx] = state_params[state_idx]["alpha"]
            beta_params[state_idx] = state_params[state_idx]["beta"]
        return alpha_params, beta_params
    
    # rewrite simulate_counts without forloops [to-do] **

    def simulate_counts(self):
        for cell_idx in tqdm(range(self.num_cells)):
            for j_idx in (range(self.num_junctions)):
                state_probs = self.theta[cell_idx]
                state_probs /= state_probs.sum()
                #sample a cell state label for each junction
                state_idx = torch.multinomial(state_probs, 1).item()
                self.Z[cell_idx, j_idx] = state_idx
                #sample a probability(success) for junction in given cell state
                junction_probs = torch.distributions.beta.Beta(
                    self.alpha_params[state_idx, j_idx], 
                    self.beta_params[state_idx, j_idx]).sample()
                #sample junction counts given probability of success and total number of reads in cluster 
                counts = torch.distributions.binomial.Binomial(
                    total_count=self.num_trials[cell_idx, j_idx], probs=junction_probs).sample()
                self.counts[cell_idx, j_idx] = counts

# %%
# Define parameters for simulation of cell states via beta distributions
#________________________________________________________________________________________________________________

import random

num_states = 2
num_junctions = 200
num_cells = 1000

state_params = {}
for i in range(num_states):
    alpha = torch.tensor([random.uniform(0.01, 1) for _ in range(num_junctions)])
    beta = torch.tensor([random.uniform(0.01, 1) for _ in range(num_junctions)])
    state_params[i] = {'alpha': alpha, 'beta': beta}

print(state_params)

# create an instance of the JunctionClusterCounts class
jc_counts = JunctionClusterCounts(num_cells, num_junctions, num_states, state_params)

# simulate counts
jc_counts.simulate_counts()

# %% 
# import functions from 03_betabinomo_LDA_singlecells
#________________________________________________________________________________________________________________

from betabinomo_LDA_singlecells import * 

if __name__ == "__main__":

    # Load data and define global variables 

    batch_size = 2048 #should also be an argument that gets fed in
    
    #prep dataloader for training
    cell_junc_counts = data.DataLoader(Simulate_DataLoader(jc_counts.counts, jc_counts.num_trials))

    # global variables
    N = len(cell_junc_counts.dataset) # number of cells
    J = cell_junc_counts.dataset[0][0].shape[0] # number of junctions
    K = num_states
    num_trials = 1 
    num_iters = 200

    # loop over the number of trials (for now just testing using one trial but in general need to evaluate how performance is affected by number of trials)
    for t in range(num_trials):
        
        # run coordinate ascent VI
        print(K)

        ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = calculate_CAVI(J, K, N, cell_junc_counts, num_iters)
        juncs_probs = ALPHA_f / (ALPHA_f+PI_f)
        theta_f = distributions.Dirichlet(GAMMA_f).sample().numpy()
        z_f = distributions.Categorical(PHI_f).sample()

        #make theta_f a dataframe 
        theta_f_plot = pd.DataFrame(theta_f)
        print(sns.clustermap(theta_f_plot))

        theta_f = torch.tensor(theta_f)
        og_theta_plot = pd.DataFrame(jc_counts.theta.numpy())
        print(sns.clustermap(og_theta_plot))

        print("compare proportions of cell states estimated versus true")
        sns.scatterplot(x=theta_f_plot[0], y=og_theta_plot[0])
        plt.xlabel('Estimated')
        plt.ylabel('Simulated')

        sns.scatterplot(x=theta_f_plot[1], y=og_theta_plot[1])
        plt.xlabel('Estimated')
        plt.ylabel('Simulated')

# %%
