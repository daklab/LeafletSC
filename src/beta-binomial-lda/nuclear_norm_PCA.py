
import numpy as np
import torch
import sklearn.manifold
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt 

import collections

SVD_Result = collections.namedtuple("SVD_Result", "U S V")

def sparse_sum(x, dim):
    return np.squeeze(np.asarray(x.sum(dim)))

def power_iteration(A, its = 20, tol = 1e-3, verbose = False):
    """This is roughly equivalent to torch.svd_lowrank(A, q = 1, niter = its, M = None). 
    However, it also has a tol parameter. If the two most recent estimates of the top
    SV are within this tolerance of each other, the loop will exit. 
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

def missing_PCA(
    indices, # 2 x nnz elements. indices[0,:] index rows of Y, indices[1,:] index cols of Y
    Y, # data at observed elements
    W, # weights at observed elements
    size = None, # true shape of Y. Defaults to using max values in indices.
    X = None, # warm start of X
    U = [], # list of U vectors
    V = [], # list of V vectors
    phi = [], # list of "singular values"
    its=100, 
    niter = 30, # for torch.svd_lowrank
    rmseTol=1e-3,
    verbose = True,
    end = "\r",  # for printing
    dtype = torch.float, 
    device = "cpu"
):
    """This doesn't seem to work, despite in principle being very similar to nuc_norm_PCA. The idea was
    to copy the Franke-Wolfe algo but with no constraint. This feels ok here because we can analytically get
    the optimal stepSize. Not you can't motivate this just as gradient descent because that doesn't give
    the SVD aspect. """

    if X is None: X = torch.zeros_like(Y) 
    oldRmse=np.inf
    
    if verbose: print("It\tRMSE\tStep\tDeltaRMSE")
    
    for it in range(its):

        er = Y - X
        grad_data = - W * er
        grad = torch.sparse_coo_tensor(
            indices, 
            grad_data, # just at observed elements! 
            size = size, 
            **float_type
        ) 

        u,s,v = torch.svd_lowrank(-grad, q = 1, niter = niter)
        u = u.flatten()
        v = v.flatten()

        uv = u[ indices[0,:] ] * v[ indices[1,:] ]
        
        stepSize = -torch.dot(grad_data, uv)/torch.dot(W, uv * uv)
        
        if stepSize<0: print('Warning: step size is',stepSize,'\n')
        X = X + stepSize * uv

        U.append(u.cpu()) # save GPU memory
        V.append(v.cpu()) # save GPU memory
        phi.append( stepSize.cpu().item() )

        er = Y - X
        rmse = (torch.dot(W, er * er) / W.sum()).sqrt().item()
        
        rmseDelta=abs(rmse-oldRmse)
        
        if verbose: print("%i\t%.4g\t%.4g\t%.4g" % (it,rmse,stepSize,rmseDelta),end = end)
        
        if rmseDelta < rmseTol: break

        oldRmse = rmse
        
    return X, U, V, phi, rmse

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
    
    Args: 
        indices_train: 2 x nnz torch.tensor of indices for the training data such that torch.sparse_coo_tensor(indices_train, Y_train) is the target matrix. 
        Y_train: nnz-vector of target values.
        W_train: nnz-vector of weight values such that torch.sparse_coo_tensor(indices_train, W_train) is the weight matrix
        size: shape of Y (not necessarily inferable from indices_train since some rows/columns might only appear in indices_test) 
        indices_test, Y_test, W_test: analogues for test data. Optional so this wrapper can be used without test data (e.g. with a previously determined max_r)
        r: initial nuc norm bound
        r_factor: factor by which r will be increased in every loop
        max_r: maximum bound to consider. 
        warm_start: whether to initialize X for the current r using the solution for the previous (smaller) r. Empirically seems to be always be a good idea. 
        end: for logging output. If "\r" then will only take one line, other use "\n"
        inner_verbose: whether the inner nuc norm PCA should print output
        **kwargs: passed to nuc_norm_PCA. 
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
