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

    # Make sure this number is less than total number of non-zero entries in intron_clusts
    assert num_masked < intron_clusts.nnz

    # Get indices of entries to mask, only from indices where values are >=1 in intron_clusts
    indices_to_mask_from = np.nonzero(intron_clusts)
    masked_indices = np.random.choice(np.arange(len(indices_to_mask_from[0])), size=num_masked, replace=False)

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

def evaluate_model(true_juncs, true_clusts, model_psi, model_assign, mask):
    
    '''
    Evaluate the factor model on masked data.

    Parameters
    ----------
    true_juncs : torch.Tensor
        A C x J matrix of junction counts with masked entries set to 0.

    true_clusts : torch.Tensor          
        A C x J matrix of intron clusters with masked entries set to 0. 

    model_psi : torch.Tensor
        A J x K matrix of cell-specific factor loadings.

    Returns
    -------
    mse : float
        The mean squared error between the imputed and observed junction counts.
    '''

    # get predicted PSI values for each cell and junction
    pred = model_assign @ model_psi # predicted PSI values for each cell and junction

    # let's look at only the masked entries
    masked_pred = pred[np.nonzero(mask)]
    true_psi = true_juncs / true_clusts
    # get true_psi values for masked indices 
    masked_true_psi = true_psi[np.nonzero(mask)]

    # get L1 absolute mean error between masked predicted and true PSI values
    mse = np.mean(np.abs(masked_pred - masked_true_psi))

    return mse