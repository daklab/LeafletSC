import numpy as np
import torch
import sklearn.manifold
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt 

def sparse_sum(x, dim):
    """
    Compute the sum of a sparse matrix along a specified dimension and return a squeezed array.

    Parameters:
    x (spmatrix): A sparse matrix whose elements are to be summed.
    dim (int): The dimension along which the sum is computed. For example, `dim=0` sums along the rows, 
               while `dim=1` sums along the columns.

    Returns:
    ndarray: The resulting dense array with the sums, with single-dimensional entries removed from its shape.
    """
    return np.squeeze(np.asarray(x.sum(dim)))

def filter_junctions(junc_counts, cluster_counts, min_junc_mean = 0.005, plot = False):
    """
    Filters junctions based on their normalized usage frequency and optionally plots the distribution.

    This function filters junctions by calculating the mean usage frequency of each junction across 
    all samples (rows in the junc_counts matrix). Junctions with a mean usage frequency higher than 
    the specified threshold (`min_junc_mean`) are retained. If `plot` is set to True, it also generates 
    a histogram of the log10-transformed mean junction counts for all junctions.

    Parameters:
    junc_counts (spmatrix): A sparse matrix where rows represent samples and columns represent junctions.
                            Each element is the count of a junction in a particular sample.
    cluster_counts (Any): This parameter is currently not used in the function.
    min_junc_mean (float, optional): The minimum threshold for the mean usage frequency of a junction 
                                     to be retained. Default is 0.005.
    plot (bool, optional): If True, a histogram of the log10-transformed mean junction counts is displayed.
                           Default is False.

    Returns:
    ndarray: An array indicating which junctions (columns in `junc_counts`) are to be kept based on 
             the filtering criteria. Also prints the original indices of the junctions being retained.

    Side Effects:
    - If `plot` is True, a histogram plot is displayed using matplotlib.
    - Prints the original indices of the junctions that are being kept after filtering.
    """
        
    junc_norm_sum = sparse_sum(junc_counts, 0) / junc_counts.shape[0]
    
    if plot:
        plt.hist(np.log10(junc_norm_sum), 100)
        plt.xlabel("log10(mean junction count)")
    
    to_keep = junc_norm_sum > min_junc_mean
    print("also printing original indices of junctions we are keeping")
    
    return junc_counts.tocsr()[:,to_keep].tocoo(), cluster_counts.tocsr()[:,to_keep].tocoo(), to_keep


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
