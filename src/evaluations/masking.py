import numpy as np
import pyro
import pyro.distributions as dist
import torch
import scipy.sparse as sp

# Cluster + junction counts set to zero for those predefined indices 
# From model fit, get estimate of what those values should be given learned PSI values 
# Use Likelihood for test elements where might put more weight on juntion with higher counts 
# L1 mean absolute error can be used to evaluate imputed vs observed PSI values for J-C pairs


def generate_mask(intron_clusts, mask_percentage=0.1):

    '''
    Generate a mask for a given intron cluster matrix.

    Parameters
    ----------
    intron_clusts : 
        A C x J sparse matrix of type '<class 'numpy.int64'>'
	with stored elements in COOrdinate format> [intron cluster counts]

    mask_percentage : float
        The percentage of entries to mask. Default is 0.1.

    Returns
    -------
    mask : torch.Tensor
        A C x J matrix of 0s and 1s where 1s indicate masked entries.
    '''

    # Get number of cells and junctions
    num_cells = intron_clusts.shape[0]
    num_junctions = intron_clusts.shape[1]

    # Get number of entries to mask
    num_masked = int(num_cells * num_junctions * mask_percentage)

    # Get indices of entries to mask
    masked_indices = torch.randperm(num_cells * num_junctions)[:num_masked]

    # Create mask
    mask = np.zeros((num_cells, num_junctions))
    mask[masked_indices // num_junctions, masked_indices % num_junctions] = 1
    print("Masked indices: ", masked_indices)
    return mask

# Second function to apply mask to intron cluster matrix and junction count matrix

def apply_mask(junction_counts, intron_clusts, mask):
    
        '''
        Apply a mask to an intron cluster matrix and junction count matrix.
    
        Parameters
        ----------
        intron_clusts : 
        A C x J sparse matrix of type '<class 'numpy.int64'>'
	    with stored elements in COOrdinate format> [intron cluster counts]
    
        intron_clusts : 
        A C x J sparse matrix of type '<class 'numpy.int64'>'
	    with stored elements in COOrdinate format> [intron cluster counts]
    
        mask : torch.Tensor
            A C x J matrix of 0s and 1s where 1s indicate masked entries.
    
        Returns
        -------
        masked_intron_clusts : torch.Tensor
            A C x J matrix of intron clusters with masked entries set to 0.
    
        masked_junction_counts : torch.Tensor
            A C x J matrix of junction counts with masked entries set to 0.
        '''
    
        # Mask intron clusters
        masked_intron_clusts = intron_clusts.toarray() * (1 - mask)
    
        # Mask junction counts
        masked_junction_counts = junction_counts.toarray() * (1 - mask)
    
        return masked_junction_counts, masked_intron_clusts

def prep_model_input(masked_junction_counts, masked_intron_clusts):
   
    '''
    Prepare input for factor model.

    Parameters
    ----------
    masked_junction_counts : torch.Tensor
        A C x J matrix of junction counts with masked entries set to 0.

    masked_intron_clusts : torch.Tensor
        A C x J matrix of intron clusters with masked entries set to 0.
    
    Returns
    -------
    masked_junction_counts_tensor : torch.sparse_coo_tensor
        A sparse tensor of masked junction counts.
    
    masked_intron_clusts_tensor : torch.sparse_coo_tensor
        A sparse tensor of masked intron clusters.
    ''' 

    # First make intron cluster sparse tensor 

    #1. intron clusts 
    indices = torch.tensor(np.nonzero(masked_intron_clusts), dtype=torch.long)
    values = torch.tensor(masked_intron_clusts[np.nonzero(masked_intron_clusts)], dtype=torch.float)
    # Determine the size of the tensor
    num_cells = masked_intron_clusts.shape[0]
    num_junctions = masked_intron_clusts.shape[1]
    size = (num_cells, num_junctions)
    # Create a sparse tensor
    masked_intron_clusts_tensor = torch.sparse_coo_tensor(indices, values, size)
    masked_intron_clusts_tensor

    #2. use the same indices to make a sparse tensor from masked_junction_counts
    values_j = torch.tensor(masked_junction_counts[np.nonzero(masked_intron_clusts)], dtype=torch.float)
    # Keep same size tensor as introns 
    masked_junction_counts_tensor = torch.sparse_coo_tensor(indices, values_j, size)

    return masked_junction_counts_tensor, masked_intron_clusts_tensor

# next function shoould evaluate model fit on masked data

def evaluate_model(masked_juncs, masked_clusts, true_juncs, true_clusts, model_psi, mask):
    
    '''
    Evaluate the factor model on masked data.

    Parameters
    ----------
    masked_junction_counts_tensor : torch.sparse_coo_tensor
        A sparse tensor of masked junction counts.

    masked_intron_clusts_tensor : torch.sparse_coo_tensor
        A sparse tensor of masked intron clusters.

    psi : torch.Tensor
        A C x K matrix of cell-specific factors.

    phi : torch.Tensor
        A K x J matrix of junction-specific factors.

    mu : torch.Tensor
        A C x J matrix of mean junction counts.

    sigma : torch.Tensor
        A C x J matrix of standard deviations of junction counts.

    mask : torch.Tensor
        A C x J matrix of 0s and 1s where 1s indicate masked entries.

    Returns
    -------
    mse : float
        The mean squared error between the imputed and observed junction counts.
    '''

    # Get number of cells and junctions
    num_cells = masked_junction_counts_tensor.shape[0]
    num_junctions = masked_junction_counts_tensor.shape[1]

    # Get indices of unmasked entries
    unmasked_indices = torch.nonzero(1 - mask)

    # Get imputed junction counts
    imputed_junction_counts = torch.mm(psi, phi) * sigma + mu

    # Get MSE
    mse = torch.sum((imputed_junction_counts[unmasked_indices[:, 0], unmasked_indices[:, 1]] - masked_junction_counts_tensor[unmasked_indices[:, 0], unmasked_indices[:, 1]]) ** 2) / (num_cells * num_junctions)

    return mse