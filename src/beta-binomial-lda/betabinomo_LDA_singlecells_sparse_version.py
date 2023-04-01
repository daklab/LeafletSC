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

torch.manual_seed(42)

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

def init_var_params(J, K, N, final_data, eps = 1e-2):
    
    '''
    Function for initializing variational parameters using global variables N, J, K   
    Sample variables from prior distribtuions 
    To-Do: implement SVD for more relevant initializations
    '''
    
    print('Initialize VI params')

    # Cell states distributions , each junction in the FULL list of junctions get a ALPHA and PI parameter for each cell state
    ALPHA = torch.from_numpy(np.random.uniform(0.5, 50, size=(J, K))).to(device)
    PI = torch.from_numpy(np.random.uniform(0.5, 50, size=(J, K))).to(device)
    print(ALPHA)
    print(PI)

    # Topic Proportions (cell states proportions), GAMMA ~ Dirichlet(eta) 
    GAMMA = torch.ones((N, K)).double().to(device)
    # Multiple each row of Gamma by a different number 
    GAMMA = GAMMA * torch.from_numpy(np.random.uniform(0.5, 50, size=(N, 1))).double().to(device)

    # Choose random states to be close to 1 and the rest to be close to 0 
    # By intializing with one value being 100 and the rest being 1 
    # generate unique random indices for each row
    #random_indices = torch.randint(K, size=(N, 1)).to(device)

    # create a mask for the random indices
    #mask = torch.zeros((N, K)).to(device)
    #mask.scatter_(1, random_indices, 1)

    # set the random indices to 1000
    #GAMMA = GAMMA * (1 - mask) + 1000 * mask
    print(GAMMA)

    # Cell State Assignments, each cell gets a PHI value for each of its junctions
    # Initialized to 1/K for each junction
    M = len(final_data.junc_index) # number of cell-junction pairs coming from non zero clusters
    PHI = torch.ones((M, K)).double().to(device) * 1/K
    
    return ALPHA, PI, GAMMA, PHI

# %%

# Functions for calculating the ELBO

def E_log_pbeta(ALPHA, PI, alpha_prior=0.65, pi_prior=0.65):
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


def E_log_ptheta(GAMMA, eta=0.1):
    
    '''
    We are assigning a K vector to each cell that has the proportion of each K present in each cell 
    GAMMA is a variational parameter assigned to each cell, follows a K dirichlet
    '''

    E_log_p_theta = torch.sum((eta - 1) * sum(torch.digamma(GAMMA).T - torch.digamma(torch.sum(GAMMA, dim=1))))
    return(E_log_p_theta)

# %%
def E_log_xz(ALPHA, PI, GAMMA, PHI, final_data):
    
    '''
    sum over N cells and J junctions... where we are looking at the exp log p(z|theta)
    plus the exp log p(x|beta and z)
    '''

    # make copies of the variational parameters - do I need to do this here? 
    #ALPHA_t = copy.deepcopy(ALPHA)
    #PI_t = copy.deepcopy(PI)
    #PHI_t = copy.deepcopy(PHI)
    #GAMMA_t = copy.deepcopy(GAMMA)

    ALPHA_t = ALPHA
    PI_t = PI
    PHI_t = PHI
    GAMMA_t = GAMMA
    
    # Set up indicies for extracting correct values from ALPHA, PI and PHI
    junc_index_tensor = final_data.junc_index
    cell_index_tensor = final_data.cell_index

    ycount=final_data.y_count
    tcount=final_data.t_count

    ### E[log p(Z_ij|THETA_i)]    
    all_digammas = (torch.digamma(GAMMA_t) - torch.digamma(torch.sum(GAMMA_t, dim=1)).unsqueeze(1)) # shape: (N, K)
            
    # Element-wise multiplication and sum over junctions-Ks and across cells 
    PHI_t_times_all_digammas = torch.sum(PHI_t * all_digammas[cell_index_tensor])
    E_log_p_xz_part1 = PHI_t_times_all_digammas

    ### E[log p(Y_ij | BETA, Z_ij)] 
    alpha_pi_digamma = (ALPHA_t + PI_t).digamma()
    E_log_beta = ALPHA_t.digamma() - alpha_pi_digamma
    E_log_1m_beta = PI_t.digamma() - alpha_pi_digamma
    
    junc_counts_states = ycount.squeeze()[:, None] * E_log_beta[junc_index_tensor, :]
    clust_counts_states = tcount.squeeze()[:, None] * E_log_1m_beta[junc_index_tensor, :]
    part2 = junc_counts_states + clust_counts_states

    E_log_p_xz_part2 = torch.sum(PHI_t * part2) #check that summation dimension is correct****
    
    E_log_p_xz = E_log_p_xz_part1 + E_log_p_xz_part2
    return(E_log_p_xz)

# %%

## Define all the entropies

def get_entropy(ALPHA, PI, GAMMA, PHI):
    
    '''
    H(X) = E(-logP(X)) for random variable X whose pdf is P
    '''

    #1. Sum over Js, entropy of beta distribution for each K given its variational parameters     
    beta_dist = distributions.Beta(ALPHA, PI)
    E_log_q_beta = beta_dist.entropy().mean(dim=1).sum()

    #2. Sum over all cells, entropy of dirichlet cell state proportions given its variational parameter 
    dirichlet_dist = distributions.Dirichlet(GAMMA)
    E_log_q_theta = dirichlet_dist.entropy().sum()
    
    #3. Sum over all cells and junctions, entropy of  categorical PDF given its variational parameter (PHI_ij)
    E_log_q_z = torch.sum(-(PHI * torch.log(PHI)))
    
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
    
    print('ELBO: {}'.format(elbo))
    return(elbo)


# %%

def update_z_theta(ALPHA, PI, GAMMA, PHI, final_data, theta_prior=0.01):

    '''
    Update variational parameters for z and theta distributions
    '''                

    GAMMA_t = GAMMA
    ALPHA_t = ALPHA
    PI_t = PI

    # Set up indicies for extracting correct values from PHI
    junc_index_tensor = final_data.junc_index
    cell_index_tensor = final_data.cell_index

    ycount=final_data.y_count
    tcount=final_data.t_count

    alpha_pi_digamma = (ALPHA_t + PI_t).digamma()
    E_log_beta = ALPHA_t.digamma() - alpha_pi_digamma
    E_log_1m_beta = PI_t.digamma() - alpha_pi_digamma
    
    junc_counts_states = ycount.squeeze()[:, None] * E_log_beta[junc_index_tensor, :]
    clust_counts_states = tcount.squeeze()[:, None] * E_log_1m_beta[junc_index_tensor, :]
    
    # Now I need to add values across cells with the same cell_id_index with GAMMA likelihood
    #result_tensor = torch.zeros((len(junc_index_tensor), K), dtype=torch.float64, device = device)

    # Update PHI

    # Add the values from part1 to the appropriate indices
    E_log_theta = torch.digamma(GAMMA_t) - torch.digamma(torch.sum(GAMMA_t)).unsqueeze(0) # shape: N x K      
    #counts = torch.bincount(cell_index_tensor)
    #expanded_part1 = part1.repeat_interleave(counts, dim=0)
    log_PHI_t = E_log_theta[cell_index_tensor,:] + junc_counts_states + clust_counts_states

    # Compute the logsumexp of the tensor
    tensor_logsumexp = torch.logsumexp(log_PHI_t, dim=1, keepdim=True)
    # Compute the exponentials of the tensor
    PHI_t = torch.exp(log_PHI_t - tensor_logsumexp)
    # Normalize every row in tensor so sum of row = 1
    PHI_t /= torch.sum(PHI_t, dim=1, keepdim=True) # in principle not necessary
    
    # Update GAMMA_c using the updated PHI_c
    # Make sure to only add junctions that belong to the same cell
    
    GAMMA_up = theta_prior + final_data.cells_lookup @ PHI_t
    
    #cell_sums = torch.zeros(N, K, dtype=torch.float64, device = device)
    #GAMMA_up = theta_prior + cell_sums.index_add_(0, cell_index_tensor, PHI_up) # SLOW
    #stop
    return(PHI_t, GAMMA_up)    

def update_beta(J, K, PHI, final_data, alpha_prior=0.65, beta_prior=0.65):
    
    '''
    Update variational parameters for beta distribution
    '''
    
    # Re-initialize ALPHA and PI values
    #ALPHA_t = torch.ones((J, K), dtype=torch.float64, device = device) * alpha_prior
    #PI_t = torch.ones((J, K), dtype=torch.float64, device = device) * beta_prior

    # Calculate alphas and pis for each cell-junction pair 
    alphas = final_data.y_count * PHI
    pis = final_data.t_count * PHI #t_count is cluster counts minus junction counts
    
    # Create a tensor of the unique junction indices
    index_tensor = final_data.junc_index

    # Use scatter_add to sum the values for each unique index
    #ALPHA_t = torch.scatter_add(ALPHA_t, 0, index_tensor[:, None].repeat(1, alphas.shape[1]), alphas)
    #PI_t = torch.scatter_add(PI_t, 0, index_tensor[:, None].repeat(1, pis.shape[1]), pis)
    
    ALPHA_t = final_data.junctions_lookup @ alphas + alpha_prior
    PI_t = final_data.junctions_lookup @ pis + beta_prior
    
    return(ALPHA_t, PI_t)

# %%   

def update_variational_parameters(ALPHA, PI, GAMMA, PHI, J, K, final_data):
    
    '''
    Update variational parameters for beta, theta and z distributions
    '''

    PHI_up, GAMMA_up = update_z_theta(ALPHA, PI, GAMMA, PHI, final_data) 
    ALPHA_up, PI_up = update_beta(J, K , PHI_up, final_data)

    return(ALPHA_up, PI_up, GAMMA_up, PHI_up)

# %%

def calculate_CAVI(J, K, N, my_data, num_iterations=5):
    
    '''
    Calculate CAVI
    '''

    ALPHA_init, PI_init, GAMMA_init, PHI_init = init_var_params(J, K, N, my_data)
    torch.cuda.empty_cache()

    elbos_init = get_elbo(ALPHA_init, PI_init, GAMMA_init, PHI_init, my_data)
    torch.cuda.empty_cache()

    elbos = []
    elbos.append(elbos_init)
    print("Got the initial ELBO ^")
    
    ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi = update_variational_parameters(ALPHA_init, PI_init, GAMMA_init, PHI_init, J, K, my_data)
    print("Got the first round of updates!")
    
    elbo_firstup = get_elbo(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, my_data)
    elbos.append(elbo_firstup)
    
    print("got the first ELBO after updates ^")
    iter = 0

    while(elbos[-1] > elbos[-2]) and (iter < num_iterations):
        print("ELBO not converged, re-running CAVI iteration # " + str(iter+1))
        ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi = update_variational_parameters(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, J, K, my_data)
        elbo = get_elbo(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, my_data)
        elbos.append(elbo)
        iter = iter + 1
    
    print("ELBO converged, CAVI iteration # " + str(iter+1) + " complete")
    return(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, elbos)

# %%
@dataclass
class IndexCountTensor():
    junc_index: torch.Tensor
    cell_index: torch.Tensor
    y_count: torch.Tensor
    t_count: torch.Tensor
    cells_lookup: torch.Tensor # sparse cells x M matrix mapping of cells to (cell,junction) pairs
    junctions_lookup: torch.Tensor 
        
# %%
# put this into main code blow after 
if True: 
    input_file = '/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/PBMC_input_for_LDA.h5'
    #input_file=args.input_file
    final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion = load_cluster_data(
        input_file, celltypes = ["B", "MemoryCD4T"])
    # 

    # global variables

    N = coo_cluster_sparse.shape[0]
    J = coo_cluster_sparse.shape[1]
    K = 3 # should also be an argument that gets fed in

    # initiate instance of data class containing junction and cluster indices for non-zero clusters 
    junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64).to(device)
    cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64).to(device)
    ycount = torch.tensor(final_data.junc_count.values).unsqueeze(-1).to(device)
    tcount = torch.tensor(final_data.clustminjunc.values).unsqueeze(-1).to(device)

    M = len(cell_index_tensor)
    cells_lookup = torch.sparse_coo_tensor(
        torch.stack([cell_index_tensor, torch.arange(M, device=device)]), 
        torch.ones(M, device=device).double()).to_sparse_csr()
    junctions_lookup = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor, torch.arange(M, device=device)]), 
        torch.ones(M, device=device).double()).to_sparse_csr()

    my_data = IndexCountTensor(junc_index_tensor, cell_index_tensor, ycount, tcount, cells_lookup, junctions_lookup)

# %%
if __name__ == "__main__":

    # Load data and define global variables 
    # get input data (this is standard output from leafcutter-sc pipeline so the column names will always be the same)
    
    num_trials = 1 # should also be an argument that gets fed in
    num_iters = 100 # should also be an argument that gets fed in

    # loop over the number of trials (for now just testing using one trial but in general need to evaluate how performance is affected by number of trials)
    for t in range(num_trials):
        
        # run coordinate ascent VI
        print(K)

        ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = calculate_CAVI(J, K, N, my_data, num_iters)
        elbos_all = torch.FloatTensor(elbos_all).numpy()
        juncs_probs = ALPHA_f / (ALPHA_f+PI_f)
        theta_f = distributions.Dirichlet(GAMMA_f).sample()
        z_f = distributions.Categorical(PHI_f).sample()
        #make theta_f a dataframe 
        theta_f_plot = pd.DataFrame(theta_f.cpu())
        theta_f_plot['cell_id'] = cell_ids_conversion["cell_type"].to_numpy()
        theta_f_plot_summ = theta_f_plot.groupby('cell_id').median()
        print(theta_f_plot_summ)

        # plot ELBOs 
        plt.plot(elbos_all[1:])

        # save the learned variational parameters
        #np.savez('variational_params.npz', ALPHA_f=ALPHA_f, PI_f=PI_f, GAMMA_f=GAMMA_f, PHI_f=PHI_f, juncs_probs=juncs_probs, theta_f=theta_f, z_f=z_f)

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
