import numpy as np
import scipy.sparse as sp
import pandas as pd

def sparse_sum(x, dim):
    return np.squeeze(np.asarray(x.sum(dim)))

def binomial_ZI(y, n, nz, min_cells = 50):
    
    nzsum = sparse_sum(nz,0) # total observations (cells) per junctions
    ysum = sparse_sum(y,0) # per junction count

    to_keep = np.logical_and( nzsum >= min_cells, ysum > 0 ) # CLUSTER is observed in at least 50 cells

    y = y[:,to_keep]
    n = n[:,to_keep]

    ysum = sparse_sum(y,0) # total count for each junction
    nsum = sparse_sum(n,0) # corresponding counts for cluster

    psi = ysum / nsum # for binomial maximum likelihood is just the ratio

    p0 = n @ sp.diags(np.log1p(-psi)) # log probability of being 0 for every element
    p0.data = np.exp(p0.data) # leaves 0 entries as 0
    expected_0s = sparse_sum(p0,0) # expected 0s for each junction

    p0.data = p0.data * (1. - p0.data) # now is the variance
    var_0s = sparse_sum(p0,0) # variance of num zeros

    upper_bound = expected_0s + 2. * np.sqrt(var_0s)

    is_zero = y.copy()
    is_zero.data = np.logical_and(n.data > 0, y.data == 0).astype(float)

    z_sum = sparse_sum(is_zero, 0)

    zi = z_sum > upper_bound

    return pd.DataFrame({
        "expected_0s":expected_0s, 
        "observed_0s":z_sum, 
        "zero_inflated":zi})
