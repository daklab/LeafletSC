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

torch.manual_seed(42)

import pytest
from betabinomo_LDA_singlecells_sprase_version import update_beta

# Path: src/beta-binomial-lda/unit_tests.py
# Write out some unit tests 
#_________________________________________________________________

#%% 

# Testing beta updates 

#1. Test that the function returns two PyTorch tensors of the correct shape and dtype.

def make_input_data(N, J, size=1000):

    cell_id_index = np.random.randint(0, N, size=size)
    junction_id_index = np.random.randint(0, J, size=size)
    junc_count = np.random.randint(0, 11, size=size)
    cluster_count = np.random.randint(0, 11, size=size)
    clustminjunc = cluster_count-junc_count
    final_data = pd.DataFrame({
    'cell_id_index': cell_id_index,
    'junction_id_index': junction_id_index,
    'junc_count': junc_count,
    'cluster_count': cluster_count,
    'clustminjunc': clustminjunc
    })
    final_data = final_data[final_data["clustminjunc"]>=0]
    return(final_data)

def test_update_beta():
    N, J, K = 20, 10, 2
    PHI = torch.full((N, J, K), 1 + 1e-6, dtype=torch.double)
    #PHI = PHI / PHI.sum(dim=-1, keepdim=True) # normalize to sum to 1
    PHI = torch.softmax(PHI, dim=-1)
    final_data = make_input_data(N, J)
    alpha_prior = 0.65
    beta_prior = 0.65
    ALPHA_t, PI_t = update_beta(J, K, PHI, final_data, alpha_prior, beta_prior)
    assert isinstance(ALPHA_t, torch.Tensor)
    assert isinstance(PI_t, torch.Tensor)
    assert ALPHA_t.shape == (J, K)
    assert PI_t.shape == (J, K)
    assert ALPHA_t.dtype == torch.float64
    assert PI_t.dtype == torch.float64

#2. Test that the function returns the expected results for a simple input.

def test_update_beta_simple():
    J, K = 2, 3
    PHI = torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]], 
                        [[0.6, 0.1, 0.3], [0.2, 0.2, 0.6], [0.3, 0.5, 0.2]]], dtype=torch.float64)
    final_data = pd.DataFrame({'cell_id_index': [0, 0, 1, 1, 1], 'junction_id_index': [0, 2, 0, 1, 2], 
                               'junc_count': [1.0]*5, 'cluster_count': [1]*5})
    alpha_prior = 0.65
    beta_prior = 0.65
    ALPHA_t, PI_t = update_beta(J, K, PHI, final_data, alpha_prior, beta_prior)
    assert torch.allclose(ALPHA_t, torch.tensor([[1.3, 0.65, 1.95], [1.3, 1.3, 1.95]], dtype=torch.float64))
    assert torch.allclose(PI_t, torch.tensor([[1.95, 1.95, 0.65], [0.65, 1.95, 1.95]], dtype=torch.float64))

#3. Test that the function handles empty input correctly.

def test_update_beta_empty():
    J, K = 0, 5
    PHI = torch.randn((J, K, 3), dtype=torch.float64)
    final_data = pd.DataFrame({'cell_id_index': [], 'junction_id_index': [], 
                               'junc_count': [], 'cluster_count': []})
    alpha_prior = 0.65
    beta_prior = 0.65
    ALPHA_t, PI_t = update_beta(J, K, PHI, final_data, alpha_prior, beta_prior)
    assert torch.allclose(ALPHA_t, torch.ones((J, K), dtype=torch.float64) * alpha_prior)
    assert torch.allclose(PI_t, torch.ones((J, K), dtype=torch.float64) * beta_prior)


# %% 
# Run tests 

if __name__ == '__main__':
    test_update_beta()
    test_update_beta_simple()
    test_update_beta_empty()
    print('All tests passed!')
# %%
