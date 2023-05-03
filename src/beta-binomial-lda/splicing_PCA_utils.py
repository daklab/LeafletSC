
import numpy as np
import torch
import sklearn.manifold
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt 

from nuclear_norm_PCA import sparse_sum

import collections

def filter_junctions(junc_counts, cluster_counts, min_junc_mean = 0.005, plot = False):
    """filter for reasonably high use junctions"""
    junc_norm_sum = sparse_sum(junc_counts, 0) / junc_counts.shape[0]
    
    if plot:
        plt.hist(np.log10(junc_norm_sum), 100)
        plt.xlabel("log10(mean junction count)")
    
    to_keep = junc_norm_sum > min_junc_mean

    return junc_counts.tocsr()[:,to_keep].tocoo(), cluster_counts.tocsr()[:,to_keep].tocoo()

def simulate_junc_counts(cluster_counts, cell_types, psi_prior_shape1 = 0.5, psi_prior_shape2 = 0.5):
    """Simulate junc counts while keeping the cluster counts of observed data. 
    
    Args: 
        cluster_counts: scipy coo_matrix. 
        cell_types: pandas Categorical series
    """
    
    N,P = cluster_counts.shape

    K = len(cell_types.cat.categories)

    cell_type_psi = torch.distributions.beta.Beta(psi_prior_shape1, psi_prior_shape2).sample([P,K]) # psi for each junction in each cell type

    cell_type_labels = cell_types.cat.codes.to_numpy()

    sim_junc_counts = cluster_counts.copy() 

    sim_junc_counts.data = torch.distributions.binomial.Binomial( 
         total_count = torch.tensor(cluster_counts.data), 
         probs = cell_type_psi[
             cluster_counts.col, 
             cell_type_labels[cluster_counts.row] ] 
    ).sample().numpy()
    
    return sim_junc_counts
    
def make_Y(junc_counts, cluster_counts, float_type, rho = 0.1):
    """Prep centered PSI matrix and weights. 
    
    Args:
        junc_counts: scipy.coo_matrix of junction counts, including explicit 0 where cluster_counts are nonzero
        cluster_counts: scipy.coo_matrix of cluster counts. Indices should match with junc_counts
        float_type: dictionary specifying dtype and device
        rho: correlation parameter for beta binomial (0=binomial, 1=maximally overdispersed)
        
    Returns:
        Y_data: centered PSI at nonzero (observed) elements of w (corresponding to nonzero cluster counts). This will be the target matrix for nuclear norm PCA
        w: scipy.coo_matrix weight matrix
        """

    psi = junc_counts.copy() # junction usage ratios
    psi.data /= cluster_counts.data
    
    w = junc_counts.copy() # observation weights = inverse variances
    w.data = cluster_counts.data / (1. + (cluster_counts.data - 1) * rho) # beta binomial variance, kinda
    
    # calculate mean junction usage ratios
    w_psi = w.copy() 
    w_psi.data *= psi.data
    junc_means = sparse_sum(w_psi, 0) / sparse_sum(w, 0)
    
    # center Y
    Y_data = psi.data - junc_means[psi.col]

    return Y_data, w

def to_torch(Y_data, w, **float_type):
    """Move data to torch. 
    
    Args:
        Y_data: numpy array of Y values (e.g. centered PSI) for observed elements (i.e. w>0). 
        w: scipy.coo_matrix of weights
        float_type: dictionary specifying dtype and device
    """
    
    Y = torch.tensor(Y_data, **float_type)
    W = torch.tensor(w.data, **float_type) 
    indices_np = np.stack([w.row,w.col])
    indices = torch.tensor(indices_np, device = float_type["device"], dtype = torch.long)
    
    return Y,W,indices

def train_test(Y_data, w, float_type, prop_train = 0.7, seed = 42):
    """Split into training and test, and move data to torch. 
    
    Args:
        Y_data: numpy array of Y values (e.g. centered PSI) for observed elements (i.e. w>0). 
        w: scipy.coo_matrix of weights
        float_type: dictionary specifying dtype and device
        prop_train: proportion of data to use for training (1-prop_train is used for test)
        seed: random seed for reproducibility.
        
    Returns: 
        Y,W and indices for train and test such that torch.sparse_coo_tensor(indices, Y) is the data matrix. 
    """

    np.random.seed(seed)
    
    train = np.random.rand(len(Y_data)) < prop_train
    #train_data = final_data.iloc[train,:]
    #test_data = final_data.iloc[~train,:]

    Y_train = torch.tensor(Y_data[train], **float_type)
    W_train = torch.tensor(w.data[train], **float_type) 

    # test performance will be evaluated on CPU always
    Y_test = torch.tensor(Y_data[~train], dtype = float_type["dtype"])
    W_test = torch.tensor(w.data[~train], dtype = float_type["dtype"])
    
    indices_np = np.stack([w.row,w.col])

    indices_train = torch.tensor(
        indices_np[:,train], 
        device = float_type["device"], 
        dtype = torch.long)

    indices_test = torch.tensor(
        indices_np[:,~train], 
        device = "cpu", # handle test on CPU 
        dtype = torch.long)
    
    return Y_train, W_train, indices_train, Y_test, W_test, indices_test

def tsne_plot(E, cell_ids_conversion):

    PCs_embedded = sklearn.manifold.TSNE(
        n_components=2, 
        learning_rate='auto',
        init='random', 
        perplexity=30).fit_transform(E.numpy()) # scale by SVs?

    PC_embed_df = pd.DataFrame(PCs_embedded, columns = ["x","y"])
    PC_embed_df["cell_type"] = cell_ids_conversion["cell_type"].to_numpy()
    #p9.ggplot(X_embed_df, p9.aes(x = "x", y="y", color = "cell_type")) + p9.geom_point()

    #plt.figure(figsize=[8,6]) # for pdf
    plt.figure(figsize=[12,8])
    sns.scatterplot(x = "x",y = "y", hue="cell_type", data= PC_embed_df, edgecolor = 'none', alpha = 0.1)
    plt.xlabel("tSNE 1")
    plt.ylabel("tSNE 2")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
