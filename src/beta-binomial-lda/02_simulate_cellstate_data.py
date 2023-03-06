# %%
import torch
import torch.utils.data as data 
import torch.distributions as distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import pdb

import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import sleep
from scipy.sparse import csr_matrix
from typing import Dict
import scanpy as sc
import anndata as ad
import scipy

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

# %% 
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
    
    def __init__(self, num_cells: int, num_junctions: int, num_states: int):
        self.num_cells = num_cells
        self.num_junctions = num_junctions
        self.num_states = num_states
        
        # sample parameters for beta distributions for each state
        self.alpha_params, self.beta_params = self.generate_beta_params()
        
        # sample proportions of cell states 
        tensor = torch.zeros((num_cells, num_states))
        indices = torch.randint(num_states, size=(num_cells,))
        # set the value of the selected index to 1
        tensor[torch.arange(num_cells), indices] = 1
        self.theta = tensor
        
        #sample labels for junctions (which cell state they get assigned to given the proportions)
        self.Z = torch.zeros(num_cells, num_junctions, dtype=torch.long)
        
        #given probability of success for each junction given the assigned cell state, 
        # sample counts using a binomial distribution
        self.counts = torch.zeros(num_cells, num_junctions)
        self.tot_counts = torch.zeros(num_cells)

        # generate unique number of trials for each junction in every cell
        # most of these should be zeroes 
        num_trials = torch.zeros(num_cells, num_junctions)
        num_elements = num_cells * num_junctions
        num_non_zero = int(0.05 * num_elements)
        indices = torch.randperm(num_elements)[:num_non_zero]
        num_trials.view(-1)[indices] =torch.randint(low=1, high=7, size=(num_non_zero,)).float()
        self.num_trials = num_trials
    
    def generate_beta_params(self):
        
        alpha_params = torch.zeros(self.num_states, self.num_junctions)
        beta_params = torch.zeros(self.num_states, self.num_junctions)
    
        # Assign junctions to each state
        junction_states = torch.randint(low=0, high=self.num_states, size=(self.num_junctions,))
    
        for state_idx in range(self.num_states):
            # Select only the junctions assigned to this state
            state_junctions = (junction_states == state_idx).nonzero(as_tuple=True)[0]

            # Set p(success) close to 1 for state_junctions
            alpha_params[state_idx, state_junctions] = torch.ones(len(state_junctions)).uniform_(0.99, 1.0)
            beta_params[state_idx, state_junctions] = torch.ones(len(state_junctions)).uniform_(0.01, 0.1)

            # Set p(success) close to 0 for non-state_junctions
            non_state_junctions = (junction_states != state_idx).nonzero(as_tuple=True)[0]
            alpha_params[state_idx, non_state_junctions] = torch.ones(len(non_state_junctions)).uniform_(0.01, 0.1)
            beta_params[state_idx, non_state_junctions] = torch.ones(len(non_state_junctions)).uniform_(0.99, 1.0)

        return alpha_params, beta_params

    # rewrite simulate_counts without forloops [to-do] **

    #def simulate_counts(self):
    #    for cell_idx in tqdm(range(self.num_cells)):
    #        for j_idx in (range(self.num_junctions)):
    #            state_probs = self.theta[cell_idx]
    #            state_probs /= state_probs.sum()
    #            #sample a cell state label for each junction
    #            state_idx = torch.multinomial(state_probs, 1).item()
    #            self.Z[cell_idx, j_idx] = state_idx
    #            #sample a probability(success) for junction in given cell state
    #            junction_probs = torch.distributions.beta.Beta(
    #                self.alpha_params[state_idx, j_idx], 
    #                self.beta_params[state_idx, j_idx]).sample()
    #            #sample junction counts given probability of success and total number of reads in cluster 
    #            counts = torch.distributions.binomial.Binomial(
    #                total_count=self.num_trials[cell_idx, j_idx], probs=junction_probs).sample()
    #            self.counts[cell_idx, j_idx] = counts

    def simulate_counts(self):

        # Calculate state probabilities for all cells
        state_probs = self.theta / self.theta.sum(dim=1, keepdim=True)

        # Sample a cell state label for each junction in each cell
        state_idx = torch.multinomial(state_probs, num_samples=1, replacement=False)

        # Calculate beta distribution parameters for all junctions and cell states
        alpha = self.alpha_params[state_idx, torch.arange(self.num_junctions)]
        beta = self.beta_params[state_idx, torch.arange(self.num_junctions)]

        # Sample junction probabilities for all junctions and cell states
        junction_probs = torch.distributions.beta.Beta(alpha, beta).sample()

        # **** To-Do! ****
        # We want to ensure that every cell (row) has at least one junction with a non-zero count  (to-do)
        # We also want to ensure that every junction (column) has at least one cell with a non-zero count (to-do)

        # Sample counts for all junctions and cell states
        counts = torch.distributions.binomial.Binomial(
            total_count=self.num_trials, probs=junction_probs).sample()

        # Store the results in the self.Z and self.counts tensors
        # This ends up being basically a simple mixed model rather than mixed membership 
        # So every junction in the same cell will have the same Z value
        # Need to fix this is simulating more complex data structures 
        self.Z = state_idx
        self.counts = counts

    def get_total_counts(self):
        tot_counts=self.counts.sum(dim=1)
        self.tot_counts = tot_counts

# %%
# Define parameters for simulation of cell states via beta distributions
#________________________________________________________________________________________________________________

num_states = 3
num_junctions = 5000
num_cells = 750

# create an instance of the JunctionClusterCounts class
jc_counts = JunctionClusterCounts(num_cells, num_junctions, num_states)

# simulate counts
jc_counts.simulate_counts()
jc_counts.get_total_counts()

# %%
#create COO matrix 
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

# make coo_matrix from jc_counts.count first convert from tensor to pandas dataframe
junc_counts = jc_counts.counts.numpy()
cluster_counts = jc_counts.num_trials.numpy()
junc_ratios = junc_counts / cluster_counts
thetas = pd.DataFrame(jc_counts.theta.numpy())
# extract cell state labels from theta
indicator_vector = thetas.idxmax(axis=1)

# Make sparse matrix out of junction ratios 
indices = np.nonzero(~np.isnan(junc_ratios))
sps = scipy.sparse.coo_matrix((junc_ratios[indices], indices), shape=junc_ratios.shape)
csr_sparse = sps.tocsr()

adata = ad.AnnData(csr_sparse, dtype=np.float32)
adata.obs["cell_state"] = pd.Categorical(indicator_vector)  # Categoricals are preferred for efficiency
adata.obs["total_counts"] = jc_counts.tot_counts  # Categoricals are preferred for efficiency
sc.settings.verbosity = 0             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.pp.highly_variable_genes(adata, min_mean=0.005, max_mean=5, min_disp=0.05) #Expects logarithmized data
adata.raw = adata

#Regress out effects of total counts per cell 
sc.pp.regress_out(adata, ['total_counts'])

sc.tl.pca(adata, svd_solver='arpack')

print(sc.pl.pca_variance_ratio(adata, log=True))
print(sc.pl.pca(adata, color="cell_state"))


# %% 
# import functions from 03_betabinomo_LDA_singlecells
#________________________________________________________________________________________________________________

from betabinomo_LDA_singlecells import * 

if __name__ == "__main__":

    # Load data and define global variables 

    batch_size = 512 #should also be an argument that gets fed in
    
    #prep dataloader for training
    cell_junc_counts = data.DataLoader(Simulate_DataLoader(jc_counts.counts, jc_counts.num_trials))

    # global variables
    N = len(cell_junc_counts.dataset) # number of cells
    J = cell_junc_counts.dataset[0][0].shape[0] # number of junctions
    K = num_states
    num_trials = 1 
    num_iters = 1000

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

# %%

print("compare proportions of cell states estimated versus true")
print(sns.jointplot(x = og_theta_plot[0],y = theta_f_plot[0]))
print(sns.jointplot(x = og_theta_plot[1],y = theta_f_plot[1]))

# plot ELBOs 
plt.plot(elbos_all[1:])
# %%
