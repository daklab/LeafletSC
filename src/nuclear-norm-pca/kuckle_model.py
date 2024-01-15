# kuckle_model.py 
# maintainer: Karin Isaev, model by David Knowles  
# date: 2024-01-12

# purpose:  
#   - define a convex nuclear norm constained linear embedding model to infer cell states driven by splicing differences across cells

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt 
import collections

# utils for nuclear norm main 
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

    psi = junc_counts.copy() # junction counts
    psi.data /= cluster_counts.data # now junction usage ratios (PSI)
    
    w = junc_counts.copy() # observation weights = inverse variances
    # weight matrix 'w' is a sparse matrix calculated using the beta-binomial variance model 
    # it represents the inverse variances (observation weights) for each PSI value 
    # rho is overdispersion parameter for beta binomial. When rho is 0, variance reduces to the binomial variance (1/p) 
    # when rho is 1, variance is maximally overdispersed 
    
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


# nuclear norm algorithm

SVD_Result = collections.namedtuple("SVD_Result", "U S V")

def power_iteration(A, its = 20, tol = 1e-3, verbose = False):

    """This is roughly equivalent to torch.svd_lowrank(A, q = 1, niter = its, M = None). 
    However, it also has a tol parameter. If the two most recent estimates of the top
    SV are within this tolerance of each other, the loop will exit. 

    Perform power iteration to approximate the largest singular value and corresponding singular vectors of a matrix.

    This function is an implementation of the power iteration method, which is an algorithm to find 
    the dominant eigenvalue and eigenvector of a matrix. It is particularly useful when dealing with large 
    matrices where computing the full SVD is computationally expensive. The algorithm iteratively updates 
    an estimate of the leading eigenvector and the corresponding eigenvalue. It also includes a tolerance 
    parameter to exit early if the estimates converge.

    Args:
        A (Tensor): A 2D tensor representing the input matrix.
        its (int, optional): The number of iterations for the power iteration. Default is 20.
        tol (float, optional): The tolerance level for convergence. If the estimates of the dominant 
                               singular value in successive iterations are within this tolerance, 
                               the loop exits. Default is 1e-3.
        verbose (bool, optional): If True, prints the iteration number and current estimates of the 
                                  dominant singular value at each iteration.

    Returns:
        u (Tensor): The approximated left singular vector corresponding to the largest singular value.
        v_norm (float): The approximated largest singular value.
        v (Tensor): The approximated right singular vector corresponding to the largest singular value.

    """
    
    n, p = A.shape
    v = torch.randn(p, 1, dtype = A.dtype, device = A.device)
    v /= v.norm() # l2
    
    A_T = torch._linalg_utils.transpose(A)
    
    for it in range(its):
        u = torch._linalg_utils.matmul(A, v)
        u_norm = u.norm() # estimate of the biggest SV
        u /= u_norm

        v = torch._linalg_utils.matmul(A_T, u)
        v_norm = v.norm() # estimate of the biggest SV
        v /= v_norm
        
        if abs(u_norm.item() - v_norm.item()) < tol: break
        if verbose: print(it, u_norm.item(), v_norm.item())
    
    return u,v_norm,v

def nuc_norm_PCA(
    indices, # 2 x nnz elements. indices[0,:] index rows of Y, indices[1,:] index cols of Y
    Y, # data at observed elements
    W, # weights at observed elements
    r, # nuclear norm bound
    size = None, # true shape of Y. Defaults to using max values in indices.
    X = None, # warm start of X
    U = [], # list of U vectors
    V = [], # list of V vectors
    phi = [], # list of "singular values"
    power_iteration_controller = (30,1.), # see docs string
    its=100, 
    rmseTol=1e-3,
    verbose = True,
    end = "\r",  # for printing
    enforce_max_stepSize = True, # turning this off means the constraint will not necessarily be satisfied.
    **float_type
):
    """Fit nuclear norm regularized matrix completion, very close in spirit to PCA. 
    
    There are two levels of convergence to consider. Firstly, the outer loop (the Franke-Wolfe) optimization stops when either `its` 
    iterations have been completed or the RMSE changes by <rmseTol. Secondly, the inner loop (power iteration (PI) to get an approximate 
    rank-1 SVD) is controlled by `power_iteration_controller`. power_iteration_controller can be a function, in which case it takes one 
    parameter (the current iteration number) and returns a tuple of pi_its (the maximum number of iterations the PI is allowed) and 
    tol (PI stops if the last two estimates of the top SV are within tol of each other). According to the theory, one should use something 
    like `power_iteration_controller = lambda it: (it+2, 0.)` or `lambda it: (100, max( 1e-1/(it+1)**2, 1e-6 ) )` but in practice time/accuracy 
    trade-off seems best with fixed `pi_its` or `tol`. Possibly this is because our analytic calculation of the stepSize makes up for inaccuracy in the SVD. 
    """

    if X is None: X = torch.zeros_like(Y) 
    oldRmse=np.inf
    
    if verbose: print("It\tRMSE\tStep\tDeltaRMSE")
    
    for it in range(its):

        grad_data = W * (X - Y)
        grad = torch.sparse_coo_tensor(
            indices, 
            grad_data, # just at observed elements! 
            size = size, 
            **float_type
        ) 

        pi_its,tol = power_iteration_controller(it) if callable(power_iteration_controller) else power_iteration_controller

        #u,s,v = sp.linalg.svds(-grad, k = 1, tol=tol) # u: Nx1, s: [1,]. v: 1xJ
        #u,s,v = torch.svd_lowrank(-grad, q = 1, niter = niter)
        u,s,v = power_iteration(-grad, its = pi_its, tol = tol)
        u = u.flatten()
        v = v.flatten()
        
        ruv = r * u[ indices[0,:] ] * v[ indices[1,:] ]
        
        # step size calculation
        #erv = (X - ruv) * W
        #stepSize=torch.dot(grad_data, erv)/torch.dot(erv, erv)
        erv = X - ruv
        stepSize=torch.dot(grad_data, erv)/torch.dot(W, erv * erv)
        
        if stepSize<0: print('Warning: step size is',stepSize,'\n')
        if enforce_max_stepSize: stepSize=min(stepSize,torch.tensor(1,**float_type))
        X = (1.0-stepSize) * X + stepSize * ruv

        #print(len(U))
        U.append(u.cpu()) # save GPU memory
        #print(len(U))

        V.append(v.cpu()) # save GPU memory
        stepSize_clone = stepSize.clone().cpu()[None]
        
        if len(phi) == 0: 
            phi = r * stepSize_clone
        else: 
            phi = torch.concat( (phi * (1.0 - stepSize_clone), r * stepSize_clone) )
        
        assert(len(U) == len(phi))
            
        er = X - Y
        rmse = (torch.dot(W, er * er) / W.sum()).sqrt().item()
        
        rmseDelta=abs(rmse-oldRmse)
        
        if verbose: print("%i\t%.4g\t%.4g\t%.4g" % (it,rmse,stepSize,rmseDelta),end = end)
        
        if rmseDelta < rmseTol: break

        oldRmse = rmse

    phi = torch.tensor(phi, dtype = float_type["dtype"], device = "cpu") # this is always on CPU

    final_svd = orthonormer(U, V, phi) 

    return X, U, V, phi, final_svd, rmse

def svd_wrapper(X): 
    """Simple wrapper for torch's full rank SVD."""
    return SVD_Result(*torch.linalg.svd(X, full_matrices = False))

def orthonormer(U, V, phi):
    """Take the list of Us, Vs and phis and return a valid SVD. """
    U_mat = torch.stack(U).T
    V_mat = torch.stack(V)
    # X ~ U diag(phi) V

    # could do this on GPU, but it's fast anyway
    SVD_U = svd_wrapper(U_mat) # U = (SVD_U[0] * SVD_U[1]) @ SVD_U[2]
    SVD_V = svd_wrapper(V_mat) 

    #SVD_U = sp.linalg.svds(U_mat, k = 9, tol=tol) 
    #SVD_V = sp.linalg.svds(V_mat, k = 9, tol=tol) 

    middle = (SVD_U.S[:,None] * SVD_U.V) * phi @ SVD_V.U * SVD_V.S
    # SVD_middle = sp.linalg.svds(middle, k = 5, tol=tol) 
    SVD_middle = svd_wrapper(middle)
    
    result = SVD_Result(
        SVD_U.U @ SVD_middle.U, # U 
        SVD_middle.S, # S
        SVD_middle.V @ SVD_V.V # V
    )
    
    #print(torch.mean(torch.abs(result.U * result.S @ result.V - U_mat * phi @ V_mat)))# 8e-22 with numpy, 7e-14 with torch

    return result


def get_predictions(final_svd, indices_test):
    """This is surprisingly memory hungry, probably because of making the big N_test x K matrices. 
    One possible solution would be to use einsum on sparse indexing matrices. """
    #X_test = (final_svd.U[indices_test[0,:],:] * final_svd.S * final_svd.V[:,indices_test[1,:]].T).sum(1)
    return torch.einsum(
        'ij,ij->i', 
        final_svd.U[indices_test[0,:],:] * final_svd.S,
        final_svd.V[:,indices_test[1,:]].T) # no faster, maybe lower memory? 


def constraint_search(
    indices_train, 
    Y_train, 
    W_train, 
    size,
    indices_test = None, 
    Y_test = None, 
    W_test = None, 
    r = 1000., 
    r_factor = 1.5, 
    max_r = np.inf, 
    warm_start = True,
    end = "\r",
    inner_verbose = False, 
    **kwargs
):
    """Sweep through nuc norm bounds optionally checking test accuracy. 

    Conducts a constraint search for nuclear norm regularization, optimizing the regularization parameter r. 
    It optionally evaluates model performance on a test dataset to find an optimal balance between 
    fitting the training data and generalizing to unseen data.

    The function iteratively increases the nuclear norm constraint (r) and performs matrix completion using 
    nuclear norm regularized PCA. It tracks the performance on both training and, if provided, test data.

    Args:
        indices_train (Tensor): 2 x nnz torch.tensor of indices for the training data such that torch.sparse_coo_tensor(indices_train, Y_train) is the target matrix. 
        Y_train (Tensor): nnz-vector of target values for training data.
        W_train (Tensor): nnz-vector of weight values for training data.
        size (Tuple[int, int]): The shape of the target matrix Y (not necessarily inferable from indices_train since some rows/columns might only appear in indices_test). 
        indices_test, Y_test, W_test (Tensor, optional): Analogues for test data.
        r (float, optional): Initial nuclear norm bound. Default is 1000.0.
        r_factor (float, optional): Factor by which 'r' will be increased in each iteration. Default is 1.5.
        max_r (float, optional): Maximum bound for 'r' to consider. Default is np.inf (infinity).
        warm_start (bool, optional): If True, initializes the solution for current 'r' using the solution 
                                     from the previous (smaller) 'r'. Default is True.
        end (str, optional): For logging output; use "\r" for single line output, "\n" for multiline. Default is "\r".
        inner_verbose (bool, optional): If True, enables verbose output in the inner nuc_norm_PCA function. Default is False.
        **kwargs: Additional arguments passed to nuc_norm_PCA.

    Returns:
        Tuple: Contains the following elements:
            - rs (List[float]): List of nuclear norm bounds used.
            - testErrors (List[float]): List of RMSEs on test data for each 'r'.
            - trainErrors (List[float]): List of RMSEs on training data for each 'r'.
            - nuc_norms (List[float]): List of nuclear norms for each completed matrix.
            - final_svd: The final singular value decomposition from nuc_norm_PCA.
            - step_times (List[float]): Time taken for each iteration.

    Note:
        - If test data is provided, the function additionally monitors the test RMSE and stops the 
          iteration if the test RMSE increases for two consecutive iterations, suggesting potential overfitting.
        - The function aims to identify a suitable 'r' value that ensures a good balance between fitting the training data and generalizing to new data.
        - A smaller value of r encourages the algorithm to find a solution with fewer non-zero singular values, leading to a lower-rank approximation. Conversely, a larger value of r allows for more non-zero singular values, potentially leading to higher-rank solutions.
        """

    X_train = None
    U = []
    V = []
    phi = []

    rs = []
    testErrors = []
    trainErrors = []
    nuc_norms = []
    step_times = []
    
    test_rmse = np.nan
    
    print("CV\tBound\tRMSE\tTestRMSE", end = end)
        
    while True: 

        rs.append(r)

        if not warm_start: 
            X_train = None
            U = []
            V = []
            phi = []
            
        start_time = time.time()

        X_train, U, V, phi, final_svd, rmse = nuc_norm_PCA(
            indices_train, 
            Y_train, 
            W_train, 
            r, 
            X = X_train, 
            U = U, 
            V = V, 
            phi = phi,
            size = size, 
            verbose = inner_verbose, 
            **kwargs
        )
        
        step_times.append( time.time() - start_time )

        trainErrors.append(rmse)

        nuc_norms.append(final_svd.S.sum().item())

        if not indices_test is None: 
            # 10s! Uses too much memory on GPU. 
            X_test = get_predictions(final_svd, indices_test)

            er = X_test - Y_test # on CPU
            test_rmse = (torch.dot(W_test, er * er) / W_test.sum()).sqrt().item()

            testErrors.append(test_rmse)
            
            if len(testErrors) >= 3:
                # stop if test error increases for two rounds
                if (testErrors[-1] >= testErrors[-2] ) and (testErrors[-2] >= testErrors[-3]):
                    break
                    
        print("CV\t%.3g\t%.3g\t%.3g" % (r,rmse,test_rmse), end = end)
        
        r *= r_factor
        
        if r > max_r: break

    return rs, testErrors, trainErrors, nuc_norms, final_svd, step_times
