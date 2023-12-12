import torch
import sklearn.cluster

def pca_kmeans_init(final_data, junc_index_tensor, cell_index_tensor, K, float_type, scale_by_sv = True):
    """This has an issue as currently implemented that all the junction ratios are positive,
    and missing==0, so it is definitely confounded by expression. Normalizing just the observed
    junctions might help. """
    
    junc_ratios_sparse = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor,cell_index_tensor]),
        torch.tensor(final_data.juncratio.values, **float_type)
    ) # to_sparse_csr() # doesn't work with CSR for some reason? 
    
    #import scipy.sparse as sp
    #junc_ratios_sp = sp.coo_matrix((final_data.juncratio.values,(final_data['junction_id_index'].values, final_data['cell_id_index'].values)))
    #junc_mean = torch.tensor(junc_ratios_sp.mean(1)**float_type)
    
    #V, pc_sd, U = torch.pca_lowrank(junc_ratios_sparse, q=20, niter=5, center = False) # , M = junc_mean.T) # out of memory trying this? 
    U, pc_sd, V = torch.svd_lowrank(junc_ratios_sparse, q=20, niter=5) #, M = junc_mean)
 # coo_matrix((data, (i, j)), [shape=(M, N)])
    
    cell_pcs = V.cpu().numpy()
    
    if scale_by_sv: cell_pcs *= pc_sd.cpu().numpy()
    
    kmeans = sklearn.cluster.KMeans(
        n_clusters=K, 
        random_state=0, 
        init='k-means++',
        n_init=10).fit(cell_pcs)

    return V.cpu().numpy(), pc_sd.cpu().numpy(), kmeans.labels_
