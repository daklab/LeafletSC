# betabinomo_LDA_singlecells.py>

# %%
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

import sklearn.cluster



# %%    
#parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

#parser.add_argument('--input_file', dest='input_file', 
              #      help='name of the file that has the intron cluster events and junction information from running 01_prepare_input_coo.py')
#args = parser.parse_args()

# %%
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

# %% 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for probabilistic beta-binomial AS model 

def init_var_params(K, final_data, float_type, init_labels = None, eps = 1e-2):
    
    '''
    Function for initializing variational parameters using global variables N, J, K   
    Sample variables from prior distribtuions 
    To-Do: implement SVD for more relevant initializations
    '''
    
    print('Initialize VI params')
    
    J = final_data.junc_index.max().item() + 1
    N = final_data.cell_index.max().item() + 1

    # Cell states distributions , each junction in the FULL list of junctions get a ALPHA and PI parameter for each cell state
    ALPHA = 1. + torch.rand(J, K, **float_type)
    PI = 1. + torch.rand(J, K, **float_type)

    # Topic Proportions (cell states proportions), GAMMA ~ Dirichlet(eta) 
    if not init_labels is None:
        GAMMA = torch.full((N, K), 1./K, **float_type)
        GAMMA[torch.arange(N),init_labels] = 2.
        PHI = GAMMA[final_data.cell_index,:] # will get normalized below
    else:
        GAMMA = 1. + torch.rand(N, K, **float_type) * 0.1
        M = len(final_data.junc_index) # number of cell-junction pairs coming from non zero clusters
        #PHI = torch.ones((M, K), dtype=DTYPE).to(device) * 1/K
        PHI = torch.rand(M, K, **float_type)
    
    PHI /= PHI.sum(1, keepdim=True)
    
    # Choose random states to be close to 1 and the rest to be close to 0 
    # By intializing with one value being 100 and the rest being 1 
    # generate unique random indices for each row
    #random_indices = torch.randint(K, size=(N, 1)).to(device)

    # create a mask for the random indices
    #mask = torch.zeros((N, K)).to(device)
    #mask.scatter_(1, random_indices, 1)

    # set the random indices to 1000
    #GAMMA = GAMMA * (1 - mask) + 1000 * mask

    # Cell State Assignments, each cell gets a PHI value for each of its junctions
    # Initialized to 1/K for each junction

    
    return ALPHA, PI, GAMMA, PHI

# %%

# Functions for calculating the ELBO

def E_log_pbeta(ALPHA, PI, hypers):
    '''
    Expected log joint of our latent variable B ~ Beta(a, b)
    Calculated here in terms of its variational parameters ALPHA and PI 
    ALPHA = K x J matrix 
    PI = K x J matrix 
    alpha_prior and pi_prior = scalar are fixed priors on the parameters of the beta distribution
    '''

    E_log_p_beta_a = torch.sum(((hypers["alpha_prior"] -1)  * (torch.digamma(ALPHA) - torch.digamma(ALPHA + PI))))
    E_log_p_beta_b = torch.sum(((hypers["pi_prior"]-1) * (torch.digamma(PI) - torch.digamma(ALPHA + PI))))

    E_log_pB = E_log_p_beta_a + E_log_p_beta_b
    return(E_log_pB)


def E_log_ptheta(GAMMA, hypers):
    
    '''
    We are assigning a K vector to each cell that has the proportion of each K present in each cell 
    GAMMA is a variational parameter assigned to each cell, follows a K dirichlet
    '''

    E_log_p_theta = (hypers["eta"] - 1.) * (GAMMA.digamma() - GAMMA.sum(dim=1, keepdim=True).digamma()).sum()
    return(E_log_p_theta)

# %%
def E_log_xz(ALPHA, PI, GAMMA, PHI, final_data):
    
    '''
    sum over N cells and J junctions... where we are looking at the exp log p(z|theta)
    plus the exp log p(x|beta and z)
    '''
    ### E[log p(Z_ij|THETA_i)]    
    all_digammas = torch.digamma(GAMMA) - torch.digamma(torch.sum(GAMMA, dim=1, keepdim=True)) # shape: (N, K)
            
    # Element-wise multiplication and sum over junctions-Ks and across cells 
    #E_log_p_xz_part1_ = torch.sum(PHI * all_digammas[cell_index_tensor,:]) # memory intensive :(
    E_log_p_xz_part1 = torch.sum( (final_data.cells_lookup @ PHI) * all_digammas)  # bit better
    
    ### E[log p(Y_ij | BETA, Z_ij)] 
    alpha_pi_digamma = (ALPHA + PI).digamma()
    E_log_beta = ALPHA.digamma() - alpha_pi_digamma
    E_log_1m_beta = PI.digamma() - alpha_pi_digamma
    
    #part2 = final_data.y_count * E_log_beta[junc_index_tensor, :] + final_data.t_count * E_log_1m_beta[junc_index_tensor, :]
    #part2 *= PHI # this is lower memory use because multiplication is in place
    
    # confirmed this gives the same answer
    part2 = (final_data.ycount_lookup @ PHI) * E_log_beta + (final_data.tcount_lookup @ PHI) * E_log_1m_beta 

    E_log_p_xz_part2 = part2.sum() 
    
    E_log_p_xz = E_log_p_xz_part1 + E_log_p_xz_part2
    return(E_log_p_xz)

# %%

## Define all the entropies

def get_entropy(ALPHA, PI, GAMMA, PHI, eps = 1e-10):
    
    '''
    H(X) = E(-logP(X)) for random variable X whose pdf is P
    '''

    #1. Sum over Js, entropy of beta distribution for each K given its variational parameters     
    beta_dist = distributions.Beta(ALPHA, PI)
    #E_log_q_beta = beta_dist.entropy().mean(dim=1).sum()
    E_log_q_beta = beta_dist.entropy().sum()

    #2. Sum over all cells, entropy of dirichlet cell state proportions given its variational parameter 
    dirichlet_dist = distributions.Dirichlet(GAMMA)
    E_log_q_theta = dirichlet_dist.entropy().sum()
    
    #3. Sum over all cells and junctions, entropy of  categorical PDF given its variational parameter (PHI_ij)
    E_log_q_z = -torch.sum(PHI * torch.log(PHI + eps)) # memory intensive. eps to handle PHI==0 correctly
    
    entropy_term = E_log_q_beta + E_log_q_theta + E_log_q_z
    
    return entropy_term

# %%

def get_elbo(ALPHA, PI, GAMMA, PHI, final_data, hypers):
    
    #1. Calculate expected log joint
    E_log_pbeta_val = E_log_pbeta(ALPHA, PI, hypers)
    E_log_ptheta_val = E_log_ptheta(GAMMA, hypers)
    E_log_pzx_val = E_log_xz(ALPHA, PI, GAMMA, PHI, final_data)

    #2. Calculate entropy
    entropy = get_entropy(ALPHA, PI, GAMMA, PHI)

    #3. Calculate ELBO
    elbo = E_log_pbeta_val + E_log_ptheta_val + E_log_pzx_val + entropy
    assert(not elbo.isnan())

    return(elbo.item())


# %%

def update_z_theta(ALPHA, PI, GAMMA, PHI, final_data, hypers):

    '''
    Update variational parameters for z and theta distributions
    '''                

    alpha_pi_digamma = (ALPHA + PI).digamma()
    E_log_beta = ALPHA.digamma() - alpha_pi_digamma
    E_log_1m_beta = PI.digamma() - alpha_pi_digamma
    
    # Update PHI

    # Add the values from part1 to the appropriate indices
    E_log_theta = torch.digamma(GAMMA) - torch.digamma(torch.sum(GAMMA, dim=1, keepdim=True)) # shape: N x K
    
    #counts = torch.bincount(final_data.cell_index)
    PHI[:,:] = E_log_theta[final_data.cell_index,:] # in place
    PHI += final_data.y_count[:, None] * E_log_beta[final_data.junc_index, :] 
    PHI += final_data.t_count[:, None] * E_log_1m_beta[final_data.junc_index, :] # this is really log_PHI! memory :(

    # Compute the logsumexp of the tensor
    PHI -= torch.logsumexp(PHI, dim=1, keepdim=True) 
    
    # Compute the exponentials of the tensor
    #PHI = PHI.exp() # can this be done in place? Yes!
    torch.exp(PHI, out=PHI) # in place! 
    
    # Normalize every row in tensor so sum of row = 1
    PHI /= torch.sum(PHI, dim=1, keepdim=True) # in principle not necessary
    
    # Update GAMMA using the updated PHI
    GAMMA = hypers["eta"] + final_data.cells_lookup @ PHI
    
    return(PHI, GAMMA)    

def update_beta(PHI, final_data, hypers):
    
    '''
    Update variational parameters for beta distribution
    '''
    
    ALPHA = final_data.ycount_lookup @ PHI + hypers["alpha_prior"]
    PI = final_data.tcount_lookup @ PHI + hypers["pi_prior"]
    
    return(ALPHA, PI)

# %%   

def update_variational_parameters(ALPHA, PI, GAMMA, PHI, final_data, hypers):
    
    '''
    Update variational parameters for beta, theta and z distributions
    '''
    # TODO: is this order good? idea is to update q(beta) first since we may have a 
    # "good" initialization for GAMMA and PHI
    ALPHA, PI = update_beta(PHI, final_data, hypers)
    PHI, GAMMA = update_z_theta(ALPHA, PI, GAMMA, PHI, final_data, hypers) 

    return(ALPHA, PI, GAMMA, PHI)

# %%

def calculate_CAVI(K, my_data, float_type, hypers = None, init_labels = None, num_iterations=5):
    '''
    Calculate CAVI
    '''
    
    if hypers is None: 
        hypers = {
            "eta" : 1., # or 1/K? 
            "alpha_prior" : 1., # karin had 0.65 
            "pi_prior" : 1. 
        }

    ALPHA, PI, GAMMA, PHI = init_var_params(K, my_data, float_type, init_labels = init_labels)
    #torch.cuda.empty_cache()

    elbos = [ get_elbo(ALPHA, PI, GAMMA, PHI, my_data, hypers) ]
    #torch.cuda.empty_cache()

    print("Got the initial ELBO")
    
    for iteration in range(num_iterations):
        print("ELBO", elbos[-1],  "CAVI iteration # ", iteration+1, end = "\r")
        ALPHA, PI, GAMMA, PHI = update_variational_parameters(ALPHA, PI, GAMMA, PHI, my_data, hypers)
        elbo = get_elbo(ALPHA, PI, GAMMA, PHI, my_data, hypers)
        elbos.append(elbo)
        if elbos[-1] < elbos[-2]: break # add tolerance? 
    
    print("ELBO converged, CAVI iteration # ", iteration+1, " complete")
    return(ALPHA, PI, GAMMA, PHI, elbos)

# %%
@dataclass
class IndexCountTensor():
    junc_index: torch.Tensor
    cell_index: torch.Tensor
    y_count: torch.Tensor
    t_count: torch.Tensor
    cells_lookup: torch.Tensor # sparse cells x M matrix mapping of cells to (cell,junction) pairs
    ycount_lookup: torch.Tensor
    tcount_lookup: torch.Tensor 

def make_torch_data(final_data, **float_type):
    
    device = float_type["device"]

    # initiate instance of data class containing junction and cluster indices for non-zero clusters 
    junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64, device=device)
    cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64, device=device)

    ycount = torch.tensor(final_data.junc_count.values, **float_type) 
    tcount = torch.tensor(final_data.clustminjunc.values, **float_type)

    M = len(cell_index_tensor)
    cells_lookup = torch.sparse_coo_tensor(
        torch.stack([cell_index_tensor, torch.arange(M, device=device)]), 
        torch.ones(M, **float_type)).to_sparse_csr()
    
    ycount_lookup = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor, torch.arange(M, device=device)]), 
        ycount).to_sparse_csr()
    
    tcount_lookup = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor, torch.arange(M, device=device)]), 
        tcount).to_sparse_csr()

    return IndexCountTensor(junc_index_tensor, cell_index_tensor, ycount, tcount, cells_lookup, ycount_lookup, tcount_lookup)
