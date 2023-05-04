
from importlib import reload
from load_cluster_data import load_cluster_data
import torch
import numpy as np
import plotnine as p9
import scipy.sparse as sp

import matplotlib.pyplot as plt 

def sparse_sum(x, dim):
    return np.squeeze(np.asarray(x.sum(dim)))

input_file = '/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/PBMC_input_for_LDA.h5'
final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion = load_cluster_data(
    input_file) # , celltypes = ["B", "MemoryCD4T"])

indices = (final_data.cell_id_index, final_data.junction_id_index)

junc_counts = sp.coo_matrix((final_data.junc_count, indices))
cluster_counts = sp.coo_matrix((final_data.cluster_count, indices))

psi = sp.coo_matrix((
    final_data.junc_count / final_data.cluster_count, 
    indices))

rho = 0.1
w = sp.coo_matrix((
    final_data.cluster_count / (1. + (final_data.cluster_count - 1) * rho), 
    indices))

w_psi = sp.coo_matrix((
    w.data * psi.data, 
    indices))

junc_means = sparse_sum(w_psi, 0) / sparse_sum(w, 0)

Y = sp.coo_matrix(( # psi_centered. Cells x Junctions (N x P)
    psi.data - junc_means[final_data.junction_id_index],
    indices))

X = np.zeros_like(Y.data) # better for X to be a sparse matrix? 

np.sqrt(np.sum(w.data * Y.data * Y.data))

# now want low rank SVD of -grad
r = 3000. # nuclear norm bound
U = []
V = []

for it in range(100):
    print(it, end = "\r")
    
    grad = Y.copy() # better to avoid all these copies
    grad.data = w.data * (X - Y.data)

    tol = max( 1e-1/(it+1)**2, 1e-6 )
    u,s,v = sp.linalg.svds(-grad, k = 1, tol=tol) # u: Nx1, s: [1,]. v: 1xJ
    u = u.flatten()
    v = v.flatten()

    ruv = r * u[final_data.cell_id_index] * v[final_data.junction_id_index]
    erv = (X - ruv) * w.data

    stepSize=np.dot(grad.data, erv)/np.dot(erv, erv)
    if stepSize<0: print('Warning: step size is',stepSize,'\n')
    stepSize=min(stepSize,1)
    X = (1.0-stepSize) * X + stepSize * ruv
    er = X-Y.data
    rmse = np.sqrt(np.dot(w.data, er * er) / w.data.sum())

    U.append(u)
    V.append(v)
    if it == 0: 
        phi = np.array(stepSize)[None]
    else: 
        phi = np.concatenate( (phi * (1.0 - stepSize), np.array(stepSize)[None]) )

U_mat = np.stack(U).T
V_mat = np.stack(V)
# X ~ U diag(phi) V

SVD_U = np.linalg.svd(U_mat, full_matrices = False) # U = (SVD_U[0] * SVD_U[1]) @ SVD_U[2]
SVD_V = np.linalg.svd(V_mat, full_matrices = False) 

#SVD_U = sp.linalg.svds(U_mat, k = 9, tol=tol) 
#SVD_V = sp.linalg.svds(V_mat, k = 9, tol=tol) 
middle = (SVD_U[1][:,None] * SVD_U[2]) * phi @ SVD_V[0] * SVD_V[1]
# SVD_middle = sp.linalg.svds(middle, k = 5, tol=tol) 
SVD_middle = np.linalg.svd(middle, full_matrices = False) 

U_final = SVD_U[0] @ SVD_middle[0]
S_final = SVD_middle[1]
V_final = SVD_middle[2] @ SVD_V[2]

# np.mean(np.abs(U_final * S_final @ V_final - U_mat * phi @ V_mat)) # 8e-22, nice

U_mat.T @ U_mat

plt.scatter( U_final[:,1], U_final[:,2] )

# U_final.T @ U_final = I, good
# V_final @ V_final.T = I, good

if (!is.null(Ytest)) 
  ertest=(X-Ytest) * test_weights
rmse=sqrt(sum(er^2)/sum(weights))
rmseDelta=abs(rmse-oldRmse)
if (verbose) 
  cat(it,rmse,sqrt(mean(ertest^2)),stepSize,rmseDelta,'\n')

