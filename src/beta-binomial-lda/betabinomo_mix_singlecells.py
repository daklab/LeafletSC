# betabinomo_LDA_singlecells.py>

# %%
import torch
import torch.distributions as distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import pandas as pd
import numpy as np
import copy
torch.cuda.empty_cache()

from dataclasses import dataclass
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import argparse

import sklearn.cluster

torch.manual_seed(42)

DTYPE = torch.float # save memory
ETA = 2./15 # define here to make sure it is consistent between ELBO and update
# alpha_prior=0.65, beta_prior=0.65
alpha_prior=1.
pi_prior=1.

# %%    
#parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

#parser.add_argument('--input_file', dest='input_file', 
              #      help='name of the file that has the intron cluster events and junction information from running 01_prepare_input_coo.py')
#args = parser.parse_args()

# %%
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

# load data 

def load_cluster_data(input_file, celltypes = None):

   # read in hdf file 
    summarized_data = pd.read_hdf(input_file, 'df')

    #for now just look at B and T cells
    if celltypes:
        summarized_data = summarized_data[summarized_data["cell_type"].isin(celltypes)]
    print(summarized_data.cell_type.unique())
    summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
    summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()

    coo = summarized_data[["cell_id_index", "junction_id_index", "junc_count", "Cluster_Counts", "Cluster", "junc_ratio"]]

    # just some sanity checks to make sure indices are in order 
    cell_ids_conversion = summarized_data[["cell_id_index", "cell_id", "cell_type"]].drop_duplicates()
    cell_ids_conversion = cell_ids_conversion.sort_values("cell_id_index")

    junction_ids_conversion = summarized_data[["junction_id_index", "junction_id", "Cluster"]].drop_duplicates()
    junction_ids_conversion = junction_ids_conversion.sort_values("junction_id_index")
 
    # make coo_matrix for junction counts
    coo_counts_sparse = coo_matrix((coo.junc_count, (coo.cell_id_index, coo.junction_id_index)), shape=(len(coo.cell_id_index.unique()), len(coo.junction_id_index.unique())))
    coo_counts_sparse = coo_counts_sparse.tocsr()
    juncs_nonzero = pd.DataFrame(np.transpose(coo_counts_sparse.nonzero()))
    juncs_nonzero.columns = ["cell_id_index", "junction_id_index"]
    juncs_nonzero["junc_count"] = coo_counts_sparse.data

    # do the same for cluster counts 
    cells_only = coo[["cell_id_index", "Cluster", "Cluster_Counts"]].drop_duplicates()
    merged_df = pd.merge(cells_only, junction_ids_conversion)
    coo_cluster_sparse = coo_matrix((merged_df.Cluster_Counts, (merged_df.cell_id_index, merged_df.junction_id_index)), shape=(len(merged_df.cell_id_index.unique()), len(merged_df.junction_id_index.unique())))
    coo_cluster_sparse = coo_cluster_sparse.tocsr()
    cluster_nonzero = pd.DataFrame(np.transpose(coo_cluster_sparse.nonzero()))
    cluster_nonzero.columns = ["cell_id_index", "junction_id_index"]
    cluster_nonzero["cluster_count"] = coo_cluster_sparse.data

    final_data = pd.merge(juncs_nonzero, cluster_nonzero, how='outer').fillna(0)
    final_data["clustminjunc"] = final_data["cluster_count"] - final_data["junc_count"]
    final_data["juncratio"] = final_data["junc_count"] / final_data["cluster_count"] 
    final_data = final_data.merge(cell_ids_conversion, on="cell_id_index", how="left")
    
    return(final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion)

# %% 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for probabilistic beta-binomial AS model 

def init_var_params(J, K, N, final_data, init_labels = None, eps = 1e-2):
    
    '''
    Function for initializing variational parameters using global variables N, J, K   
    Sample variables from prior distribtuions 
    To-Do: implement SVD for more relevant initializations
    '''
    
    print('Initialize VI params')

    # Cell states distributions , each junction in the FULL list of junctions get a ALPHA and PI parameter for each cell state
    ALPHA = 1. + torch.rand(J, K, dtype=DTYPE, device = device)
    PI = 1. + torch.rand(J, K, dtype=DTYPE, device = device)

    # Topic Proportions (cell states proportions), GAMMA ~ Dirichlet(eta) 
    if not init_labels is None:
        GAMMA = torch.full((N, K), 1./K, dtype=DTYPE, device = device)
        GAMMA[torch.arange(N),init_labels] = 2.
        PHI = GAMMA[final_data.cell_index,:] # will get normalized below
    else:
        GAMMA = 1. + torch.rand(K, dtype=DTYPE, device = device) 
        #M = len(final_data.junc_index) # number of cell-junction pairs coming from non zero clusters
        #PHI = torch.ones((M, K), dtype=DTYPE).to(device) * 1/K
        PHI = torch.rand(N, K, dtype=DTYPE, device = device)
    
    PHI /= PHI.sum(1, keepdim=True)
    
    # Choose random states to be close to 1 and the rest to be close to 0 
    # By intializing with one value being 100 and the rest being 1 
    # generate unique random indices for each row
    #random_indices = torch.randint(K, size=(N, 1)).to(device)

    # create a mask for the random indices
    #mask = torch.zeros((N, K)).to(device)
    #mask.scatter_(1, random_indices, 1)

    # set the random indices to 1000
    #GAMMA = GAMMA * (1 - mask) + 1000 * mask

    # Cell State Assignments, each cell gets a PHI value for each of its junctions
    # Initialized to 1/K for each junction

    
    return ALPHA, PI, GAMMA, PHI

# %%

# Functions for calculating the ELBO

def E_log_pbeta(ALPHA, PI):
    '''
    Expected log joint of our latent variable B ~ Beta(a, b)
    Calculated here in terms of its variational parameters ALPHA and PI 
    ALPHA = K x J matrix 
    PI = K x J matrix 
    alpha_prior and pi_prior = scalar are fixed priors on the parameters of the beta distribution
    '''

    E_log_p_beta_a = torch.sum(((alpha_prior -1)  * (torch.digamma(ALPHA) - torch.digamma(ALPHA + PI))))
    E_log_p_beta_b = torch.sum(((pi_prior-1) * (torch.digamma(PI) - torch.digamma(ALPHA + PI))))

    E_log_pB = E_log_p_beta_a + E_log_p_beta_b
    return(E_log_pB)


def E_log_ptheta(GAMMA):
    
    '''
    We are assigning a K vector to each cell that has the proportion of each K present in each cell 
    GAMMA is a variational parameter assigned to each cell, follows a K dirichlet
    '''

    E_log_p_theta = (ETA - 1.) * (GAMMA.digamma() - GAMMA.sum().digamma()).sum()
    return(E_log_p_theta)

# %%
def E_log_xz(ALPHA, PI, GAMMA, PHI, final_data):
    
    '''
    sum over N cells and J junctions... where we are looking at the exp log p(z|theta)
    plus the exp log p(x|beta and z)
    '''
    ### E[log p(Z_ij|THETA_i)]    
    E_log_theta = GAMMA.digamma() - GAMMA.sum().digamma() # shape: (K)
            
    # Element-wise multiplication and sum over junctions-Ks and across cells 
    #E_log_p_xz_part1_ = torch.sum(PHI * all_digammas[cell_index_tensor,:]) # memory intensive :(
    E_log_p_xz_part1 = (PHI * E_log_theta).sum()  # bit better
    
    ### E[log p(Y_ij | BETA, Z_ij)] 
    alpha_pi_digamma = (ALPHA + PI).digamma()
    E_log_beta = ALPHA.digamma() - alpha_pi_digamma # J x K
    E_log_1m_beta = PI.digamma() - alpha_pi_digamma
    
    #part2 = final_data.y_count * E_log_beta[junc_index_tensor, :] + final_data.t_count * E_log_1m_beta[junc_index_tensor, :]
    #part2 *= PHI # this is lower memory use because multiplication is in place
    
    # confirmed this gives the same answer
    part2 = PHI * (final_data.ycount_lookup @ E_log_beta + final_data.tcount_lookup @ E_log_1m_beta) 
    # ycount_lookup is N (cells) x J(unctions). 
    
    E_log_p_xz_part2 = part2.sum() 
    
    E_log_p_xz = E_log_p_xz_part1 + E_log_p_xz_part2
    return(E_log_p_xz)

# %%

## Define all the entropies

def get_entropy(ALPHA, PI, GAMMA, PHI, eps = 1e-10):
    
    '''
    H(X) = E(-logP(X)) for random variable X whose pdf is P
    '''

    #1. Sum over Js, entropy of beta distribution for each K given its variational parameters     
    beta_dist = distributions.Beta(ALPHA, PI)
    #E_log_q_beta = beta_dist.entropy().mean(dim=1).sum()
    E_log_q_beta = beta_dist.entropy().sum()

    #2. Sum over all cells, entropy of dirichlet cell state proportions given its variational parameter 
    dirichlet_dist = distributions.Dirichlet(GAMMA)
    E_log_q_theta = dirichlet_dist.entropy().sum()
    
    #3. Sum over all cells and junctions, entropy of  categorical PDF given its variational parameter (PHI_ij)
    E_log_q_z = -torch.sum(PHI * torch.log(PHI + eps)) # memory intensive. eps to handle PHI==0 correctly
    
    entropy_term = E_log_q_beta + E_log_q_theta + E_log_q_z
    
    return entropy_term

# %%

def get_elbo(ALPHA, PI, GAMMA, PHI, final_data):
    
    #1. Calculate expected log joint
    E_log_pbeta_val = E_log_pbeta(ALPHA, PI)
    E_log_ptheta_val = E_log_ptheta(GAMMA)
    E_log_pzx_val = E_log_xz(ALPHA, PI, GAMMA, PHI, final_data)

    #2. Calculate entropy
    entropy = get_entropy(ALPHA, PI, GAMMA, PHI)

    #3. Calculate ELBO
    elbo = E_log_pbeta_val + E_log_ptheta_val + E_log_pzx_val + entropy
    assert(not elbo.isnan())
    print('ELBO: {}'.format(elbo))
    return(elbo.item())


# %%

def update_z_theta(ALPHA, PI, GAMMA, PHI, final_data):

    '''
    Update variational parameters for z and theta distributions
    '''                

    alpha_pi_digamma = (ALPHA + PI).digamma()
    E_log_beta = ALPHA.digamma() - alpha_pi_digamma
    E_log_1m_beta = PI.digamma() - alpha_pi_digamma
    
    # Update PHI

    # Add the values from part1 to the appropriate indices
    E_log_theta = torch.digamma(GAMMA) - torch.digamma(torch.sum(GAMMA)) # shape: K
    
    PHI = final_data.ycount_lookup @ E_log_beta + final_data.tcount_lookup @ E_log_1m_beta # N x K
    PHI += E_log_theta[None,:]

    # Compute the logsumexp of the tensor
    PHI -= torch.logsumexp(PHI, dim=1, keepdim=True) 
    
    # Compute the exponentials of the tensor
    #PHI = PHI.exp() # can this be done in place? Yes!
    torch.exp(PHI, out=PHI) # in place! 
    
    # Normalize every row in tensor so sum of row = 1
    PHI /= torch.sum(PHI, dim=1, keepdim=True) # in principle not necessary
    
    # Update GAMMA using the updated PHI
    GAMMA = ETA + PHI.sum(0) # correct dimension to sum over? 
    
    return(PHI, GAMMA)    

def update_beta(J, K, PHI, final_data):
    
    '''
    Update variational parameters for beta distribution
    '''
    # ycount_lookup is N x J. PHI is N x K. More efficient to do transposes differently? 
    ALPHA = final_data.ycount_lookup_T @ PHI + alpha_prior # J x K
    PI = final_data.tcount_lookup_T @ PHI + pi_prior
    
    return(ALPHA, PI)

# %%   

def update_variational_parameters(ALPHA, PI, GAMMA, PHI, J, K, final_data):
    
    '''
    Update variational parameters for beta, theta and z distributions
    '''
    # TODO: is this order good? idea is to update q(beta) first since we may have a 
    # "good" initialization for GAMMA and PHI
    ALPHA, PI = update_beta(J, K , PHI, final_data)
    PHI, GAMMA = update_z_theta(ALPHA, PI, GAMMA, PHI, final_data) 

    return(ALPHA, PI, GAMMA, PHI)

# %%

def calculate_CAVI(J, K, N, my_data, init_labels = None, num_iterations=5):
    
    '''
    Calculate CAVI
    '''

    ALPHA, PI, GAMMA, PHI = init_var_params(J, K, N, my_data, init_labels = init_labels)
    torch.cuda.empty_cache()

    elbos = [ get_elbo(ALPHA, PI, GAMMA, PHI, my_data) ]
    torch.cuda.empty_cache()

    print("Got the initial ELBO ^")
    
    ALPHA, PI, GAMMA, PHI = update_variational_parameters(ALPHA, PI, GAMMA, PHI, J, K, my_data)
    print("Got the first round of updates!")
    
    elbos.append( get_elbo(ALPHA, PI, GAMMA, PHI, my_data) )
    
    print("got the first ELBO after updates ^")
    iteration = 0

    while(elbos[-1] > elbos[-2]) and (iteration < num_iterations):
    #while (iteration < num_iterations):
        print("ELBO not converged, re-running CAVI iteration # ", iteration+1)
        ALPHA, PI, GAMMA, PHI = update_variational_parameters(ALPHA, PI, GAMMA, PHI, J, K, my_data)
        elbo = get_elbo(ALPHA, PI, GAMMA, PHI, my_data)
        elbos.append(elbo)
        iteration += 1
    
    print("ELBO converged, CAVI iteration # ", iteration+1, " complete")
    return(ALPHA.cpu(), PI.cpu(), GAMMA.cpu(), PHI.cpu(), elbos) # move results back to CPU to avoid clogging GPU

def pca_kmeans_init(final_data, junc_index_tensor, cell_index_tensor, K, scale_by_sv = True):

    junc_ratios_sparse = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor,cell_index_tensor]),
        torch.tensor(final_data.juncratio.values, dtype=DTYPE, device=device)
    ) # to_sparse_csr() # doesn't work with CSR for some reason? 
    
    #import scipy.sparse as sp
    #junc_ratios_sp = sp.coo_matrix((final_data.juncratio.values,(final_data['junction_id_index'].values, final_data['cell_id_index'].values)))
    #junc_mean = torch.tensor(junc_ratios_sp.mean(1), dtype=DTYPE, device=device)
    
    #V, pc_sd, U = torch.pca_lowrank(junc_ratios_sparse, q=20, niter=5, center = False) # , M = junc_mean.T)
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

# %%
@dataclass
class IndexCountTensor():
    ycount_lookup: torch.Tensor
    tcount_lookup: torch.Tensor 
    ycount_lookup_T: torch.Tensor
    tcount_lookup_T: torch.Tensor 
        

        
# %%
# put this into main code blow after 
if True: 
    input_file = '/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/PBMC_input_for_LDA.h5'
    #input_file=args.input_file
    final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion = load_cluster_data(
        input_file) # , celltypes = ["B", "MemoryCD4T"])
    # 

    # global variables
    N = coo_cluster_sparse.shape[0]
    J = coo_cluster_sparse.shape[1]

    # initiate instance of data class containing junction and cluster indices for non-zero clusters 
    junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64, device=device)
    cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64, device=device)
    
    # note these are staying on the CPU! 
    ycount = torch.tensor(final_data.junc_count.values, dtype=DTYPE, device=device) 
    tcount = torch.tensor(final_data.clustminjunc.values, dtype=DTYPE, device=device)

    M = len(cell_index_tensor)

    ycount_lookup = torch.sparse_coo_tensor(
        torch.stack([cell_index_tensor, junc_index_tensor]), 
        ycount).to_sparse_csr()

    ycount_lookup_T = torch.sparse_coo_tensor( # this is a hack since I can't figure out tranposing sparse matrices :( 
        torch.stack([junc_index_tensor, cell_index_tensor]), 
        ycount).to_sparse_csr()

    tcount_lookup = torch.sparse_coo_tensor( 
        torch.stack([cell_index_tensor, junc_index_tensor]), 
        tcount).to_sparse_csr()
    
    tcount_lookup_T = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor, cell_index_tensor]), 
        tcount).to_sparse_csr()

    my_data = IndexCountTensor(ycount_lookup, tcount_lookup, ycount_lookup_T, tcount_lookup_T)

    K = 15 # should also be an argument that gets fed in

    if False: 
        junc_sparse = torch.sparse_coo_tensor(
            torch.stack([junc_index_tensor,cell_index_tensor]),
            torch.tensor(final_data.junc_count.values, dtype=DTYPE, device=device)
        )
        cell_tots = torch.sparse.sum(junc_sparse, 0)
        plt.hist(cell_tots.to_dense().cpu().numpy(), 100)
        plt.xlabel("Number of spliced reads per cell (post UMI collapsing)")
        plt.ylabel("Frequency")
        plt.savefig("reads_per_cell_hist.pdf")

        cell_pcs, pc_sd, init_labels = pca_kmeans_init(final_data, junc_index_tensor, cell_index_tensor, K)

import plotnine as p9
import scipy.sparse as sp
def sparse_sum(x, dim):
    return np.squeeze(np.asarray(x.sum(dim)))

ind = (final_data['cell_id_index'].values, final_data['junction_id_index'].values)
y_all = sp.coo_matrix( (final_data.junc_count.values, ind ))
n_all = sp.coo_matrix( (final_data.cluster_count.values, ind ))
nz_all = sp.coo_matrix( (np.ones(M), ind ))

y_csr = y_all.tocsr()
n_csr = n_all.tocsr()
nz_csr = nz_all.tocsr()

zi_rate = {}
for celltype in cell_ids_conversion.cell_type.unique(): 
    cell_type_rows = (cell_ids_conversion.cell_type == celltype).to_numpy()
    if cell_type_rows.sum() < 500: 
        print("Skipping", celltype)
        continue

    y = y_csr[cell_type_rows,:]
    n = n_csr[cell_type_rows,:]
    nz = nz_csr[cell_type_rows,:]
    
    y = y_csr
    n = n_csr
    nz = nz_csr

    nzsum = sparse_sum(nz,0)
    ysum = sparse_sum(y,0)

    to_keep = np.logical_and( nzsum >= 50, ysum > 0 ) # CLUSTER is observed in at least 50 cells

    y = y[:,to_keep]
    n = n[:,to_keep]

    ysum = sparse_sum(y,0)
    nsum = sparse_sum(n,0)

    psi = ysum / nsum

    p0 = n @ sp.diags(np.log1p(-psi))
    p0.data = np.exp(p0.data) # leaves 0 entries as 0
    expected_0s = sparse_sum(p0,0)

    p0.data = p0.data * (1. - p0.data) # now is the variance
    var_0s = sparse_sum(p0,0)

    upper_bound = expected_0s + 2. * np.sqrt(var_0s)

    is_zero = y.copy()
    is_zero.data = np.logical_and(n.data > 0, y.data == 0).astype(float)

    z_sum = sparse_sum(is_zero, 0)

    zi = z_sum > upper_bound
    
    zi_rate[celltype] = np.mean(zi)
    to_plot = pd.DataFrame({
        "expected_0s":expected_0s, 
        "observed_0s":z_sum, 
        "zero_inflated":zi})
    ziplot = ( p9.ggplot(to_plot, p9.aes(x="expected_0s", y="observed_0s", color="zero_inflated")) + p9.geom_point(alpha=0.2) + p9.geom_abline(intercept=0,slope=1) + p9.scale_x_log10() + p9.scale_y_log10() + p9.themes.theme_bw() + p9.ggtitle(celltype + " cells"))
    ziplot
    p9.ggsave(ziplot, "zi_binomial_%s.png" % celltype, figsize=(5,4))

zi_rate_df = pd.DataFrame({"Cell_type":zi_rate.keys(), "ZI_rate":zi_rate.values()})
zi_rate_df["ZI_rate"] = 100. * zi_rate_df["ZI_rate"]
zi_summary = p9.ggplot(zi_rate_df, p9.aes("Cell_type", "ZI_rate", label="ZI_rate")) + p9.geom_col() + p9.geom_text(format_string="{:.1f}%", nudge_y = 7) + p9.coord_flip(ylim=[0,100]) + p9.ylab("Percentage zero-inflated junctions under binomial") + p9.theme(figure_size=(5,4))
zi_summary
p9.ggsave(zi_summary, "zi_summary.pdf")

karin_bb = pd.read_csv("/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/Clusters_ZI_results.csv.gz")
karin_bb["zero_inflated"] = karin_bb.obtained_zeroes > karin_bb.expected_upper_bount
karin_cd8T = karin_bb[karin_bb.cell == "CD8T"]

ziplot = ( p9.ggplot(karin_cd8T, p9.aes(x="pois_bern_mean", y="obtained_zeroes", color="zero_inflated")) + p9.geom_point(alpha=0.2) + p9.geom_abline(intercept=0,slope=1) + p9.scale_x_log10() + p9.scale_y_log10() + p9.themes.theme_bw() + p9.xlab("expected_0s") + p9.ylab("observed_0s") + p9.ggtitle(celltype + " cells")) + p9.theme(figure_size=(5,4))
ziplot
p9.ggsave(ziplot, "zi_bb.pdf")


zi_bb_rate_df = karin_bb.groupby("cell", as_index=False).agg({"zero_inflated":"mean"})
zi_bb_rate_df["zero_inflated"] = 100. * zi_bb_rate_df["zero_inflated"]
zi_summary = p9.ggplot(zi_bb_rate_df, p9.aes("cell", "zero_inflated", label="zero_inflated")) + p9.geom_col() + p9.geom_text(format_string="{:.1f}%", nudge_y = 7) + p9.coord_flip(ylim=[0,100]) + p9.ylab("Percentage zero-inflated junctions under BB") + p9.xlab("Cell type") + p9.theme(figure_size=(5,4))
zi_summary
p9.ggsave(zi_summary, "zi_bb_summary.pdf")

# %%""
if __name__ == "__main__":

    # Load data and define global variables 
    # get input data (this is standard output from leafcutter-sc pipeline so the column names will always be the same)
    
    num_trials = 1 # should also be an argument that gets fed in
    num_iters = 50 # should also be an argument that gets fed in

    # loop over the number of trials (for now just testing using one trial but in general need to evaluate how performance is affected by number of trials)
    import time
    start_time = time.time()
    results = [ calculate_CAVI(J, K, N, my_data, init_labels = None, num_iterations = num_iters) for t in range(num_trials) ]
    print(time.time() - start_time)
        
    best = np.argmax([ g[-1][-1] for g in results ]) # final ELBO
    ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = results[best]
        # run coordinate ascent VI
    #ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = calculate_CAVI(J, K, N, my_data, init_labels = None, num_iterations = num_iters)
    elbos_all = np.array(elbos_all)
    plt.plot(elbos_all[1:]); plt.show()
    
    juncs_probs = ALPHA_f / (ALPHA_f+PI_f)    
    plt.hist(juncs_probs.cpu().numpy().flatten(), 20); plt.show()
  
    theta_f_plot = pd.DataFrame(PHI_f.cpu().numpy())
    theta_f_plot['cell_id'] = cell_ids_conversion["cell_type"].to_numpy()
    theta_f_plot_summ = theta_f_plot.groupby('cell_id').mean()
    print(theta_f_plot_summ)

        # plot ELBOs 
        

        # save the learned variational parameters
        #np.savez('variational_params.npz', ALPHA_f=ALPHA_f, PI_f=PI_f, GAMMA_f=GAMMA_f, PHI_f=PHI_f, juncs_probs=juncs_probs, theta_f=theta_f, z_f=z_f)

theta = GAMMA_f / GAMMA_f.sum()
theta = theta.cpu().numpy()
theta_sorted = np.sort(theta)
plt.bar(np.arange(K)+1,theta_sorted[::-1]); plt.show()

to_keep = theta > 0.04

# plt.hist(PHI_f.cpu().numpy().flatten(), 100)

x = PHI_f.cpu().numpy()
x = x[:,to_keep]
#x -= x.mean(1,keepdims=True)
#x /= x.std(1,keepdims=True)
plt.hist(x.flatten(),100)

ct = pd.crosstab( cell_ids_conversion["cell_type"], x.argmax(axis=1) )
ct_np = ct.to_numpy()

ct_np = ct_np / ct_np.sum(1, keepdims=True) # normalize cell-type counts
ct_np = ct_np / ct_np.sum(0, keepdims=True)

ct.iloc[:,:] = ct_np

ax = plt.figure(figsize=[10,8])
sns.clustermap(ct, dendrogram_ratio=0.15, vmin = None, figsize=(8,6), annot = True)
plt.savefig("Cluster_heatmap.pdf")

pcs_scaled = cell_pcs.copy()
pcs_scaled -= pcs_scaled.mean(1,keepdims=True)
pcs_scaled /= pcs_scaled.std(1,keepdims=True)
plt.hist(pcs_scaled.flatten(),100)

pcs_sd_scaled = cell_pcs * pc_sd

import sklearn.manifold 
import plotnine as p9

PCs_embedded = sklearn.manifold.TSNE(
    n_components=2, 
    learning_rate='auto',
    init='random', 
    perplexity=30).fit_transform(pcs_sd_scaled)

PC_embed_df = pd.DataFrame(PCs_embedded, columns = ["x","y"])
PC_embed_df["cell_type"] = cell_ids_conversion["cell_type"].to_numpy()
#p9.ggplot(X_embed_df, p9.aes(x = "x", y="y", color = "cell_type")) + p9.geom_point()

plt.figure(figsize=[8,6])
sns.scatterplot(x = "x",y = "y", hue="cell_type", data= PC_embed_df, edgecolor = 'none', alpha = 0.1)
plt.xlabel("tSNE 1")
plt.ylabel("tSNE 2")
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("pca_eig_scaled.pdf")


if False: 
    import umap

    fit = umap.UMAP(
        n_neighbors = 100, 
        unique = False)
    u = fit.fit_transform(pcs_scaled) # ok well this just never runs

PCs_embedded = sklearn.manifold.TSNE(
    n_components=2, 
    learning_rate='auto',
    init='random', 
    perplexity=100).fit_transform(pcs_scaled)

X_embed_df = pd.DataFrame(X_embedded, columns = ["x","y"])
X_embed_df["cell_type"] = cell_ids_conversion["cell_type"].to_numpy()
sns.scatterplot(x = "x",y = "y", hue="cell_type", data= X_embed_df, edgecolor = 'none', alpha = 0.1)

X_embed_df = pd.DataFrame(pcs_scaled[:,:2], columns = ["x","y"])
X_embed_df["cell_type"] = cell_ids_conversion["cell_type"].to_numpy()

plt.figure(figsize=[12,8])
sns.scatterplot(x = "x",y = "y", hue="cell_type", data= X_embed_df, edgecolor = 'none', alpha = 0.1)



# %%
#print(sns.jointplot(data=final_data, x = "junc_count",y = "juncratio", height=5, ratio=2, marginal_ticks=True))

celltypes = theta_f_plot.pop("cell_id")
lut = dict(zip(celltypes.unique(), ["r", "b", "g", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]))
print(lut)
row_colors = celltypes.map(lut)
print(sns.clustermap(theta_f_plot, row_colors=row_colors))
# %%
juncs_probs.diff()
junc_ind=2
clust=junction_ids_conversion[junction_ids_conversion["junction_id_index"]==junc_ind].Cluster
juncs_include = junction_ids_conversion[junction_ids_conversion["Cluster"]==int(clust)]
plot_clusts = final_data[final_data["junction_id_index"].isin(juncs_include["junction_id_index"].values)]
plot_clusts
#%%
sns.histplot(data=plot_clusts[plot_clusts["cell_type"]=="MemoryCD4T"], x="juncratio", hue="junction_id_index", multiple="stack")

# %%
sns.histplot(data=plot_clusts[plot_clusts["cell_type"]=="B"], x="juncratio", hue="junction_id_index", multiple="stack")

# %%
sns.violinplot(data=plot_clusts, x="junction_id_index", y="juncratio")# %%

# %%
sns.histplot(juncs_probs.diff().cpu().numpy())
# %%



if False: 
    
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the total count and three different rates
    n = 10
    rates = [0.1, 0.2, 0.5]

    # Calculate the probability mass function (PMF) for each rate
    pmfs = []
    for rate in rates:
        pmf = np.zeros(n+1)
        for k in range(n+1):
            pmf[k] = np.math.comb(n, k) * (rate ** k) * ((1 - rate) ** (n - k))
        pmfs.append(pmf)

    # Plot the PMFs
    plt.figure(figsize=(5,4))
    for i in range(len(rates)):
        plt.plot(pmfs[i], "-o", label='Rate = {}'.format(rates[i]))
    plt.legend()
    plt.xlabel('Junction count')
    plt.ylabel('Probability')
    plt.title('Binomial distribution with n = 10')
    plt.savefig("binomal_example.pdf")
    plt.show()
    
    
import numpy as np
from scipy.stats import binom, betabinom
import matplotlib.pyplot as plt

# Set the parameters
n = 10
p = 0.1
a1 = 0.1
b1 = 0.9
a2 = 1
b2 = 9

# Generate the data
x = np.arange(n+1)

# Plot the distributions
plt.plot(x, binom.pmf(x, n, p), "-o", label='Binomial(%i, %.1f)' % (n,p))
plt.plot(x, betabinom.pmf(x, n, a1, b1), "-o", label='Beta-Binomial(%i, %.1f, %.1f)' % (n,a1,b1))
plt.plot(x, betabinom.pmf(x, n, a2, b2), "-o", label='Beta-Binomial(%i, %.1f, %.1f)' % (n,a2,b2))
plt.legend( fontsize=12)
plt.xlabel('Junction counts', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.title('Binomial vs. Beta-Binomial distribution', fontsize=16)
plt.savefig("bb_example.pdf")
plt.show()





final_data[ ["cell_id_index",] ]

ind = (final_data['cell_id_index'].values, final_data['junction_id_index'].values)
y_all = sp.coo_matrix( (final_data.junc_count.values, ind ))
n_all = sp.coo_matrix( (final_data.cluster_count.values, ind ))
nz_all = sp.coo_matrix( (np.ones(M), ind ))

y_csr = y_all.tocsr()
n_csr = n_all.tocsr()
nz_csr = nz_all.tocsr()


summarized_data = pd.read_hdf(input_file, 'df')
cluster_data = summarized_data[["cell_id_index","Cluster","Cluster_Counts"]].drop_duplicates()
#summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
#summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()


cluster_counts = sp.coo_matrix( (cluster_data.Cluster_Counts.values, (cluster_data.cell_id_index.values, cluster_data.Cluster.values) ))
to_keep = sparse_sum(cluster_counts,0) > 0
cluster_counts = cluster_counts.tocsr()[:,to_keep]
cluster_counts = cluster_counts.tocoo()
cluster_counts.data = np.ones_like(cluster_counts.data)
cell_clusters = sparse_sum(cluster_counts,1) 

plt.xlabel("Percent clusters observed per cell")



junc_counts = final_data[final_data.junc_count > 0]
ind = (junc_counts['cell_id_index'].values, junc_counts['junction_id_index'].values)
junc_presence = sp.coo_matrix( (np.ones(junc_counts.shape[0]), ind ))

plt.hist(100 * cell_clusters / 11585, 30, alpha=0.5, label = "Clusters") 
plt.hist(100 * sparse_sum(junc_presence,1) / 38802, 30, alpha=0.5, label = "Junctions")
plt.xlabel("Percent observed (>0 UMI) per cell", fontsize=14)
plt.legend(fontsize=12)
plt.savefig("percent_obs.pdf")
