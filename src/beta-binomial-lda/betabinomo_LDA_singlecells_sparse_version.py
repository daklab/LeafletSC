# betabinomo_LDA_singlecells.py>

# %%
import torch
import torch.distributions as distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import pandas as pd
import numpy as np
import copy

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep
import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50" #By default, this value is set to 256 MB. 

torch.manual_seed(42)
# %%
# load data 

def load_cluster_data(input_file):

   # read in this pickl file output_test.pkl
    summarized_data = pd.read_pickle(input_file)

    #for now just look at B and T cells
    summarized_data = summarized_data[summarized_data["cell_type"].isin(["NaiveCD4T", "B"])]
    print(summarized_data.cell_type.unique())
    summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
    summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()

    # print histogram of junction ratios with x-axis label color by cell type 
    plt.figure(figsize=(10, 5))
    plt.title("Histogram of Junction Ratios")

    for cell_type in summarized_data.cell_type.unique():
        plt.hist(summarized_data[summarized_data["cell_type"] == cell_type]["junc_ratio"], bins=10, alpha=0.3, label=cell_type)
        plt.legend()
    plt.xlabel("Junction Ratio")
    plt.ylabel("Frequency")
    plt.show()

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

# Functions for probabilistic beta-binomial AS model 

def init_var_params(J, K, N, eps = 1e-2):
    
    '''
    Function for initializing variational parameters using global variables N, J, K   
    Sample variables from prior distribtuions 
    To-Do: implement SVD for more relevant initializations
    '''
    
    print('Initialize VI params')

    # Cell states distributions , each junction in the FULL list of junctions get a ALPHA and PI parameter for each cell state
    ALPHA = torch.from_numpy(np.random.uniform(1, 1, size=(J, K))).to(device)
    PI = torch.from_numpy(np.random.uniform(1, 1, size=(J, K))).to(device)

    print(ALPHA / (ALPHA+PI))

    # Topic Proportions (cell states proportions), GAMMA ~ Dirichlet(eta) 
    GAMMA = torch.ones((N, K)).double().to(device)
    
    # Choose random states to be close to 1 and the rest to be close to 0 
    # By intializing with one value being 100 and the rest being 1 
    # generate unique random indices for each row
    random_indices = torch.randint(K, size=(N, 1)).to(device)

    # create a mask for the random indices
    mask = torch.zeros((N, K)).to(device)
    mask.scatter_(1, random_indices, 1)

    # set the random indices to 1000
    GAMMA = GAMMA * (1 - mask) + 1000 * mask
    print(GAMMA)

    # Cell State Assignments, each cell gets a PHI value for each of its junctions
    PHI = torch.full((N, J, K), 1 + eps, dtype=torch.double).to(device)
    PHI = torch.softmax(PHI, dim=-1)
    
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
    ALPHA_t = copy.deepcopy(ALPHA)
    PI_t = copy.deepcopy(PI)
    PHI_t = copy.deepcopy(PHI)
    GAMMA_t = copy.deepcopy(GAMMA)

    ### E[log p(Z_ij|THETA_i)]    
    all_digammas = (torch.digamma(GAMMA_t) - torch.digamma(torch.sum(GAMMA_t, dim=1)).unsqueeze(1)) # shape: (N, K)
    E_log_p_xz_part1 = sum(torch.sum(PHI_t[c] @ all_digammas[c]) for c in range(PHI_t.shape[0])) #

    ### E[log p(Y_ij | BETA, Z_ij)] 
    
    # Set up indicies for extracting correct values from ALPHA, PI and PHI
    junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64).to(device)
    cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64).to(device)

    ycount=torch.tensor(final_data.junc_count.values).unsqueeze(-1).to(device)
    tcount=torch.tensor(final_data.clustminjunc.values).unsqueeze(-1).to(device)

    junc_counts_states = ycount.squeeze()[:, None] * (torch.digamma(ALPHA_t[junc_index_tensor, :]) - torch.digamma(ALPHA_t[junc_index_tensor, :]+PI_t[junc_index_tensor, :]))
    clust_counts_states = tcount.squeeze()[:, None] * (torch.digamma(PI_t[junc_index_tensor, :]) - torch.digamma(ALPHA_t[junc_index_tensor, :]+PI_t[junc_index_tensor, :]))
    part2 = junc_counts_states + clust_counts_states

    # Get the phi values for the current batch
    phi_batch = PHI_t[cell_index_tensor, junc_index_tensor, :]
    E_log_p_xz_part2 = torch.sum(phi_batch * part2) #check that summation dimension is correct****
    
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
    E_log_q_z = torch.sum(-(PHI * torch.log(PHI)).sum(dim=-1))
    
    entropy_term = E_log_q_beta + E_log_q_theta + E_log_q_z
    return entropy_term

# %%

def get_elbo(ALPHA, PI, GAMMA, PHI, final_data):
    
    #1. Calculate expected log joint
    E_log_pbeta_val = E_log_pbeta(ALPHA, PI)
    E_log_ptheta_val = E_log_ptheta(GAMMA)
    E_log_pzx_val = E_log_xz(ALPHA, PI, GAMMA, PHI, final_data) #**this step takes a long time

    #2. Calculate entropy
    entropy = get_entropy(ALPHA, PI, GAMMA, PHI)

    #3. Calculate ELBO
    elbo = E_log_pbeta_val + E_log_ptheta_val + E_log_pzx_val + entropy
    
    print('ELBO: {}'.format(elbo))
    return(elbo)


# %%

def update_z_theta(ALPHA, PI, GAMMA, PHI, final_data, theta_prior=0.1):

    '''
    Update variational parameters for z and theta distributions
    '''                
    
    GAMMA_t = copy.deepcopy(GAMMA)
    ALPHA_t = copy.deepcopy(ALPHA)
    PI_t = copy.deepcopy(PI)
    PHI_t = copy.deepcopy(PHI)

    # part1 should be good to go 
    part1 = torch.digamma(GAMMA_t) - torch.digamma(torch.sum(GAMMA_t)).unsqueeze(0) # shape: N x K      

    # Set up indicies for extracting correct values from PHI
    junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64).to(device)
    cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64).to(device)

    ycount=torch.tensor(final_data.junc_count.values).unsqueeze(-1).to(device)
    tcount=torch.tensor(final_data.clustminjunc.values).unsqueeze(-1).to(device)

    junc_counts_states = ycount.squeeze()[:, None] * (torch.digamma(ALPHA_t[junc_index_tensor, :]) - torch.digamma(ALPHA_t[junc_index_tensor, :]+PI_t[junc_index_tensor, :]))
    clust_counts_states = tcount.squeeze()[:, None] * (torch.digamma(PI_t[junc_index_tensor, :]) - torch.digamma(ALPHA_t[junc_index_tensor, :]+PI_t[junc_index_tensor, :]))
    part2 = junc_counts_states + clust_counts_states

    # Now I need to add values across cells with the same cell_id_index with GAMMA likelihood
    result_tensor = torch.zeros((N, J, K), dtype=torch.float64).to(device)
    
    # Add the values from part2 to the appropriate indices
    result_tensor[cell_index_tensor, junc_index_tensor, :] += part2
    
    # Add the values from part1 to the appropriate indices
    result_tensor[:, :, :] += part1[:, None, :]

    # Update PHI_t
    PHI_t[cell_index_tensor, junc_index_tensor, :] = result_tensor[cell_index_tensor, junc_index_tensor, :]
    PHI_t[cell_index_tensor, junc_index_tensor, :] = torch.exp((PHI_t[cell_index_tensor, junc_index_tensor, :] + 1e-9))
    # Renormalize every row of every cell in PHI with non zero cluster counts 
    row_sum = PHI_t[cell_index_tensor, junc_index_tensor, :].sum(dim=1)    
    PHI_t[cell_index_tensor, junc_index_tensor, :] = PHI_t[cell_index_tensor, junc_index_tensor, :] / row_sum.unsqueeze(-1)
    PHI_up = PHI_t 
        
    # Update GAMMA_c using the updated PHI_c
    GAMMA_up = theta_prior + torch.sum(PHI_up, axis=1)

    return(PHI_up, GAMMA_up)    

def update_beta(J, K, PHI, final_data, alpha_prior=0.65, beta_prior=0.65):
    
    '''
    Update variational parameters for beta distribution
    '''
    
    # Re-initialize ALPHA and PI values
    ALPHA_t = torch.ones((J, K), dtype=torch.float64).to(device) * alpha_prior
    PI_t = torch.ones((J, K), dtype=torch.float64).to(device) * beta_prior

    # Set up indicies for extracting correct values from PHI
    first_indices = final_data.cell_id_index.values
    second_indices = final_data.junction_id_index.values

    # Calculate alphas and pis for each cell-junction pair 
    alphas = (torch.tensor(final_data.junc_count.values).unsqueeze(-1).to(device) * PHI[first_indices, second_indices, :])
    pis = (torch.tensor(final_data.clustminjunc.values).unsqueeze(-1).to(device) * PHI[first_indices, second_indices, :])
    
    # Create a tensor of the unique junction indices
    index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64).to(device)

    # Use scatter_add to sum the values for each unique index
    ALPHA_t = torch.scatter_add(ALPHA_t, 0, index_tensor[:, None].repeat(1, alphas.shape[1]), alphas)
    PI_t = torch.scatter_add(PI_t, 0, index_tensor[:, None].repeat(1, pis.shape[1]), pis)
    
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

def calculate_CAVI(J, K, N, final_data, num_iterations=5):
    
    '''
    Calculate CAVI
    '''

    ALPHA_init, PI_init, GAMMA_init, PHI_init = init_var_params(J, K, N)
    elbos_init = get_elbo(ALPHA_init, PI_init, GAMMA_init, PHI_init, final_data)
    elbos = []
    elbos.append(elbos_init)
    print("Got the initial ELBO ^")
    
    ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi = update_variational_parameters(ALPHA_init, PI_init, GAMMA_init, PHI_init, J, K, final_data)
    print("Got the first round of updates!")
    
    elbo_firstup = get_elbo(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, final_data)
    elbos.append(elbo_firstup)
    
    print("got the first ELBO after updates ^")
    iter = 0

    while(elbos[-1] > elbos[-2]) and (iter < num_iterations):
        print("ELBO not converged, re-running CAVI iteration # " + str(iter+1))
        ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi = update_variational_parameters(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, J, K, final_data)
        elbo = get_elbo(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, final_data)
        elbos.append(elbo)
        iter = iter + 1
    
    print("ELBO converged, CAVI iteration # " + str(iter+1) + " complete")
    return(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, elbos)

# %%
if __name__ == "__main__":

    # Load data and define global variables 
    # get input data (this is standard output from leafcutter-sc pipeline so the column names will always be the same)
    
    input_file = '/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/junctions_full_for_LDA.pkl.pkl' #this should be an argument that gets fed in
    final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion = load_cluster_data(input_file)
    
    # global variables
    
    N = coo_cluster_sparse.shape[0]
    J = coo_cluster_sparse.shape[1]
    K = 2 # should also be an argument that gets fed in
    
    num_trials = 1 # should also be an argument that gets fed in
    num_iters = 200 # should also be an argument that gets fed in

    # loop over the number of trials (for now just testing using one trial but in general need to evaluate how performance is affected by number of trials)
    for t in range(num_trials):
        
        # run coordinate ascent VI
        print(K)

        ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = calculate_CAVI(J, K, N, final_data, num_iters)
        juncs_probs = ALPHA_f / (ALPHA_f+PI_f)
        theta_f = distributions.Dirichlet(GAMMA_f).sample().numpy()
        z_f = distributions.Categorical(PHI_f).sample()

        #make theta_f a dataframe 
        theta_f_plot = pd.DataFrame(theta_f)
        theta_f_plot['cell_id'] = cell_ids_conversion["cell_type"].to_numpy()

        celltypes = theta_f_plot.pop("cell_id")
        lut = dict(zip(celltypes.unique(), ["r", "b", "g", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]))
        print(lut)
        row_colors = celltypes.map(lut)
        print(sns.clustermap(theta_f_plot, row_colors=row_colors))
        plt.show()
        # plot ELBOs 
        plt.plot(elbos_all[1:])
        print(sns.jointplot(data=final_data, x = "junc_count",y = "juncratio", hue="cell_type", kind="kde"))
# %%
print(sns.jointplot(data=final_data, x = "junc_count",y = "juncratio", height=5, ratio=2, marginal_ticks=True))