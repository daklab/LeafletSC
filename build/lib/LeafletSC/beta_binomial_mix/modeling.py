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
from scipy.stats import binom
from tqdm import tqdm
import sklearn.cluster
from scipy.stats import binom

from importlib import reload

import umap
import scanpy as sc

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for probabilistic binomial AS mixture model 

def init_var_params(K, final_data, float_type, init_labels = None, eps = 1e-2):
    
    '''
    Function for initializing variational parameters using global variables N, J, K   
    Sample variables from prior distribtuions 
    If init_labels is not None, then we will initialize the cell state proportions with these labels
    init_labels can just be cell_ids_conversion["cell_type"]
    If user wants to run the model with init_labels then final_data has to have a column for cell_type_dummy_variable
    So that we can map back the cell type assignment
    '''
    
    print('Initialize VI params')
    
    (N,J) = final_data.ycount_lookup.shape
    print("Initializing variational parameters with N =", N, "cells and J =", J, "junctions")

    # Cell states distributions , each junction in the FULL list of junctions get a ALPHA and PI parameter for each cell state
    ALPHA = 1. + torch.rand(J, K, **float_type)
    PI = 1. + torch.rand(J, K, **float_type)

    # Topic Proportions (cell states proportions), GAMMA ~ Dirichlet(eta) 
    if not init_labels is None:
        print("Initializing cell state proportions with init_labels")
        # Set GAMMA to dummy values based on the initial labels
        cell_type_dummy = pd.get_dummies(init_labels)
        PHI = torch.tensor(cell_type_dummy.values, **float_type)
        # GAMMA is the sum of the cell state proportions for each cell
        GAMMA = PHI.sum(0)/PHI.sum(0).sum() 
    else:
        GAMMA = 1. + torch.rand(K, **float_type) 
        PHI = torch.rand(N, K, **float_type)
    PHI /= PHI.sum(1, keepdim=True)
    return ALPHA, PI, GAMMA, PHI

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

    E_log_p_theta = (hypers["eta"] - 1.) * (GAMMA.digamma() - GAMMA.sum().digamma()).sum()
    return(E_log_p_theta)

# %%
def E_log_xz(ALPHA, PI, GAMMA, PHI, final_data):
    
    '''
    sum over N cells and J junctions... where we are looking at the exp log p(z|theta)
    plus the exp log p(x|beta and z)
    '''
    ### E[log p(Z_ij|THETA_i)]    
    E_log_theta = GAMMA.digamma() - GAMMA.sum().digamma() # shape: (K)
            
    # Element-wise multiplication and sum over junctions-Ks and across cells 
    #E_log_p_xz_part1_ = torch.sum(PHI * all_digammas[cell_index_tensor,:]) # memory intensive :(
    E_log_p_xz_part1 = (PHI * E_log_theta).sum()  # bit better
    
    ### E[log p(Y_ij | BETA, Z_ij)] 
    alpha_pi_digamma = (ALPHA + PI).digamma()
    E_log_beta = ALPHA.digamma() - alpha_pi_digamma # J x K
    E_log_1m_beta = PI.digamma() - alpha_pi_digamma
        
    # confirmed this gives the same answer
    part2 = PHI * (final_data.ycount_lookup @ E_log_beta + final_data.tcount_lookup @ E_log_1m_beta) 
    # ycount_lookup is N (cells) x J(unctions). 
    
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
    E_log_theta = torch.digamma(GAMMA) - torch.digamma(torch.sum(GAMMA)) # shape: K
    
    PHI = final_data.ycount_lookup @ E_log_beta + final_data.tcount_lookup @ E_log_1m_beta # N x K
    PHI += E_log_theta[None,:]

    # Compute the logsumexp of the tensor
    PHI -= torch.logsumexp(PHI, dim=1, keepdim=True) 
    
    # Compute the exponentials of the tensor
    #PHI = PHI.exp() # can this be done in place? Yes!
    torch.exp(PHI, out=PHI) # in place! 
    
    # Normalize every row in tensor so sum of row = 1
    PHI /= torch.sum(PHI, dim=1, keepdim=True) # in principle not necessary
    
    # Update GAMMA using the updated PHI
    GAMMA = hypers["eta"] + PHI.sum(0) 
    
    return(PHI, GAMMA)    

def update_beta(PHI, final_data, hypers):
    
    '''
    Update variational parameters for beta distribution
    '''
    # ycount_lookup is N x J. PHI is N x K. More efficient to do transposes differently? 
    ALPHA = final_data.ycount_lookup_T @ PHI + hypers["alpha_prior"] # J x K
    PI = final_data.tcount_lookup_T @ PHI + hypers["pi_prior"]
    
    return(ALPHA, PI)

# %%   

def update_variational_parameters(ALPHA, PI, GAMMA, PHI, final_data, hypers, fixed_cell_types = False):
    
    '''
    Update variational parameters for beta, theta and z distributions
    '''
    ALPHA, PI = update_beta(PHI, final_data, hypers)
    if not fixed_cell_types:
        print("Updating z and theta")
        PHI, GAMMA = update_z_theta(ALPHA, PI, GAMMA, PHI, final_data, hypers)
    return(ALPHA, PI, GAMMA, PHI)

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

# %%

def calculate_CAVI(K, my_data, float_type, hypers = None, init_labels = None, num_iterations=5, fixed_cell_types = False, tolerance=1e-3):
    
    '''
    Run CAVI
    '''

    if hypers is None: 
        hypers = {
            "eta" : 1/K, # 1 or 1/K
            "alpha_prior" : 1., 
            "pi_prior" : 1. 
        }
        
    
    # If fixed_cell_types is True, do not update GAMMA and PHI, 
    # fix them to the initial values
    # This is useful for differential splicing analysis

    ALPHA, PI, GAMMA, PHI = init_var_params(K, my_data, float_type, init_labels = init_labels)

    if fixed_cell_types:
        print("Running CAVI with fixed cell types")
        print("z and theta will not be updated!")
    else:
        print("Running CAVI with unknown cell types (not predefined)")
        print("z and theta will be updated!")

    elbos = [ get_elbo(ALPHA, PI, GAMMA, PHI, my_data, hypers) ] 

    print("Got the initial ELBO ^")
    
    print("The tolerance is set to ", tolerance)

    for iteration in range(num_iterations):
        print("ELBO", elbos[-1],  "CAVI iteration # ", iteration+1, end = "\r")
        ALPHA, PI, GAMMA, PHI = update_variational_parameters(ALPHA, PI, GAMMA, PHI, my_data, hypers, fixed_cell_types)
        elbo = get_elbo(ALPHA, PI, GAMMA, PHI, my_data, hypers)
        elbos.append(elbo)
        if abs(elbos[-1] - elbos[-2]) < tolerance:
            convergence_message = "ELBO converged @ {} CAVI iteration # {} complete".format(elbos[-1], iteration + 1)
            print(convergence_message)
            break

    if convergence_message:
        print(convergence_message)

    print("Finished CAVI!")
    return(ALPHA.cpu(), PI.cpu(), GAMMA.cpu(), PHI.cpu(), elbos) # move results back to CPU to avoid clogging GPU

# Functions for differential splicing analysis 
def log_beta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

def score(a, b):
    return log_beta(a,b).sum() - log_beta(a.sum(), b.sum())

# Add functionality for getting score q-value to measure significance of differential splicing

