# betabinomo_LDA_singlecells.py>

# %%
import torch
import torch.utils.data as data 
import torch.distributions as distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import pdb

import pandas as pd
import numpy as np
import copy

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import sleep

torch.manual_seed(42)

# %%
# load data 
class CSRDataLoader():
    
    # data loader for csr_matrix

    def __init__(self, csr_mat1, csr_mat2):
        self.csr_mat1 = csr_mat1
        self.csr_mat2 = csr_mat2
        
    def __getitem__(self, idx):
        mat1_row = self.csr_mat1[idx].toarray().reshape(-1)
        mat2_row = self.csr_mat2[idx].toarray().reshape(-1)
        return mat1_row, mat2_row
    
    def __len__(self):
        return self.csr_mat1.shape[0]

def load_cluster_data(input_file):

   # read in this pickl file output_test.pkl
    summarized_data = pd.read_pickle(input_file)

    #for now just look at B and T cells
    summarized_data = summarized_data[summarized_data["cell_type"].isin(["NaiveCD4T"])]
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

    # notes to self about sparse csr matrices 
    #The crow_indices tensor contains the row indices of non-zero elements in the tensor. In this case, the non-zero elements have row indices ranging from 0 to 16723683.
    #The col_indices tensor contains the column indices of the non-zero elements in the tensor.
    #The values tensor contains the values of the non-zero elements in the tensor. In this case, all the non-zero elements have the value 1.
    #The size of the tensor is (40939, 38105), meaning that the tensor has 40939 rows and 38105 columns.
    #The nnz property tells us that there are 16723683 non-zero elements in the tensor.
    
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
    ALPHA = torch.from_numpy(np.random.uniform(1, 1, size=(J, K))) 
    PI = torch.from_numpy(np.random.uniform(1, 1, size=(J, K)))

    print(ALPHA / (ALPHA+PI))

    # Topic Proportions (cell states proportions), GAMMA ~ Dirichlet(eta) 
    #GAMMA = (torch.rand((N, K)).double() + eps) * 100
    GAMMA = torch.ones((N, K)).double()
    
    # Choose random states to be close to 1 and the rest to be close to 0 
    # By intializing with one value being 100 and the rest being 1 
    # generate unique random indices for each row
    random_indices = torch.randint(K, size=(N, 1))

    # create a mask for the random indices
    mask = torch.zeros((N, K))
    mask.scatter_(1, random_indices, 1)

    # set the random indices to 1000
    GAMMA = GAMMA * (1 - mask) + 1000 * mask
    print(GAMMA)

    # Cell State Assignments, each cell gets a PHI value for each of its junctions
    #PHI = torch.rand((N, J, K)).double() + eps
    #PHI = torch.ones((N, J, K)).double() + eps
    PHI = torch.full((N, J, K), 1 + eps, dtype=torch.double)
    #PHI = PHI / PHI.sum(dim=-1, keepdim=True) # normalize to sum to 1
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

    E_log_p_beta_a = torch.sum(((alpha_prior -1 )  * (torch.digamma (ALPHA) - torch.digamma (ALPHA + PI))))
    E_log_p_beta_b = torch.sum(((pi_prior-1) * (torch.digamma (PI) - torch.digamma (ALPHA + PI))))

    E_log_pB = E_log_p_beta_a + E_log_p_beta_b
    assert(~(np.isnan(E_log_pB)))
    return(E_log_pB)


def E_log_ptheta(GAMMA, eta=0.1):
    
    '''
    We are assigning a K vector to each cell that has the proportion of each K present in each cell 
    GAMMA is a variational parameter assigned to each cell, follows a K dirichlet
    '''

    E_log_p_theta = torch.sum((eta - 1) * sum(torch.digamma(GAMMA).T - torch.digamma(torch.sum(GAMMA, dim=1))))
    assert(~(np.isnan(E_log_p_theta)))
    return(E_log_p_theta)

# %%
def E_log_xz(ALPHA, PI, GAMMA, PHI, cell_junc_counts):
    
    '''
    sum over N cells and J junctions... where we are looking at the exp log p(z|theta)
    plus the exp log p(x|beta and z)
    '''

    E_log_p_xz_part1 = 0
    E_log_p_xz_part2 = 0

    # make copies of the variational parameters - do I need to do this here? 
    ALPHA_t = copy.deepcopy(ALPHA)
    PI_t = copy.deepcopy(PI)
    PHI_t = copy.deepcopy(PHI)
    GAMMA_t = copy.deepcopy(GAMMA)

    ### E[log p(Z_ij|THETA_i)]    
    all_digammas = (torch.digamma(GAMMA_t) - torch.digamma(torch.sum(GAMMA_t, dim=1)).unsqueeze(1)) # shape: (N, K)
    E_log_p_xz_part1 += sum(torch.sum(PHI_t[c] @ all_digammas[c]) for c in range(PHI_t.shape[0])) #

    ### E[log p(Y_ij | BETA, Z_ij)] BETA here defines our probability of success for every junction given a cell state
    lnBeta = (torch.digamma(ALPHA_t) - torch.digamma(ALPHA_t + PI_t)).unsqueeze(0) # shape: (1, J, K)
    ln1mBeta = (torch.digamma(PI_t) - torch.digamma(ALPHA_t + PI_t)).unsqueeze(0) # shape: (1, J, K)

    batch_sizes = []

    for c, (juncs, clusters) in enumerate(tqdm(cell_junc_counts)): #cell_junc_counts is a dataloader object

        batch_size = juncs.size(0)
        batch_sizes.append(batch_size)
        
        start_idx = c * batch_size
        end_idx = start_idx + batch_size

        if(c>0):
            start_idx = c * batch_sizes[c-1]
            end_idx = start_idx + batch_size    

        #print("working on cells " + str(start_idx) + " to " + str(end_idx) + " of " + str(N) + " cells")

        # Broadcast the lnBeta tensor along the batch dimension
        lnBeta_t = copy.deepcopy(lnBeta)
        ln1mBeta_t = copy.deepcopy(ln1mBeta)
        
        lnBeta_t = lnBeta_t.expand(juncs.size(0), -1, -1)  # shape: (32, J, K) where 32 is batch size
        ln1mBeta_t = ln1mBeta_t.expand(juncs.size(0), -1, -1)  # shape: (32, J, K) where 32 is batch size

        # Get the phi values for the current batch
        phi_batch = PHI_t[start_idx:end_idx]  

        # Perform the element-wise multiplication - i think these steps take the longest
        second_term = torch.mul(lnBeta_t, juncs.unsqueeze(-1)).squeeze(-1) #shape: (32, J, K) if 32 is the batch size
        third_term = torch.mul(ln1mBeta_t, (clusters-juncs).unsqueeze(-1)).squeeze(-1) #shape: (32, J, K) if 32 is the batch size
     
        E_log_p_xz_part2_add = torch.sum(phi_batch * (second_term + third_term)) #check that summation dimension is correct****
        E_log_p_xz_part2 = E_log_p_xz_part2+E_log_p_xz_part2_add
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
    #print(E_log_q_beta)

    #2. Sum over all cells, entropy of dirichlet cell state proportions given its variational parameter 
    dirichlet_dist = distributions.Dirichlet(GAMMA)
    E_log_q_theta = dirichlet_dist.entropy().sum()
    #print(E_log_q_theta)
    
    #3. Sum over all cells and junctions, entropy of  categorical PDF given its variational parameter (PHI_ij)
    E_log_q_z = -(PHI * torch.log(PHI)).sum(dim=-1).sum()
    #print(E_log_q_z)
    
    entropy_term = E_log_q_beta + E_log_q_theta + E_log_q_z
    return entropy_term

# %%

def get_elbo(ALPHA, PI, GAMMA, PHI, cell_junc_counts):
    
    #1. Calculate expected log joint
    E_log_pbeta_val = E_log_pbeta(ALPHA, PI)
    E_log_ptheta_val = E_log_ptheta(GAMMA)
    E_log_pzx_val = E_log_xz(ALPHA, PI, GAMMA, PHI, cell_junc_counts) #**this step takes a long time

    #2. Calculate entropy
    entropy = get_entropy(ALPHA, PI, GAMMA, PHI)

    #3. Calculate ELBO
    elbo = E_log_pbeta_val + E_log_ptheta_val + E_log_pzx_val + entropy
    
    print('ELBO: {}'.format(elbo))
    return(elbo)


# %%

def update_z_theta(ALPHA, PI, GAMMA, PHI, cell_junc_counts, theta_prior=0.1):

    '''
    Update variational parameters for z and theta distributions
    '''                
    GAMMA_t = copy.deepcopy(GAMMA)
    ALPHA_t = copy.deepcopy(ALPHA)
    PI_t = copy.deepcopy(PI)
    PHI_t = copy.deepcopy(PHI)
    #reset PHI_t for update
    #PHI_t = torch.ones((N, J, K)).double() / K

    # Iterate through each cell 
    batch_sizes=[]

    for c, (juncs, clusters) in enumerate(tqdm(cell_junc_counts)):
        
        batch_size = juncs.size(0)
        batch_sizes.append(batch_size)
        
        start_idx = c * batch_size
        end_idx = start_idx + batch_size

        if(c>0):
            start_idx = c * batch_sizes[c-1]
            end_idx = start_idx + batch_size        

        #print("working on cells " + str(start_idx) + " to " + str(end_idx) + " of " + str(N) + " cells")

        # Get GAMMA for that cell and for this iteration so that PHI can be updated 
        GAMMA_i_t = (GAMMA_t[start_idx:end_idx]) # K-vector 
        
        # Update PHI_c to PHI_c+batch.size
        ALPHA_t_iter = ALPHA_t.expand(juncs.size(0), -1, -1)  # shape: (32, 3624, 2) where 32 is batch size
        PI_t_iter = PI_t.expand(juncs.size(0), -1, -1)  # shape: (32, 3624, 2) where 32 is batch size

        #these multiplications take the longest
        part1 = torch.digamma(GAMMA_i_t) - torch.digamma(torch.sum(GAMMA_i_t)).unsqueeze(0) #K long vector                  
        part2 = torch.mul(juncs.unsqueeze(-1), (torch.digamma(ALPHA_t_iter) - torch.digamma(ALPHA_t_iter+PI_t_iter))).squeeze(-1) #this operation might take a while
        part3 = torch.mul((clusters-juncs).unsqueeze(-1), (torch.digamma(PI_t_iter) - torch.digamma(ALPHA_t_iter+PI_t_iter))).squeeze(-1)  #this operation might take a while
        
        part1_expanded = part1.unsqueeze(1).expand(-1, part2.shape[1], -1)  # shape: (32, 3624, 2)
        log_PHI_i = part1_expanded + part2 + part3    

        # double check that this is correct!!    
        PHI_i = torch.exp(log_PHI_i - log_PHI_i.logsumexp(dim=-1).unsqueeze(-1)) + 1e-9 
        #renormalize every row of every cell in PHI_i
        PHI_i = PHI_i / PHI_i.sum(dim=-1).unsqueeze(-1)
        PHI_t[start_idx:end_idx] = PHI_i 
        #print("getting ELBO using new PHI values and previous GAMMA values")
        #get_elbo(ALPHA_t, PI_t, GAMMA_t, PHI_t)
        
        # Update GAMMA_c using the updated PHI_c
        GAMMA_i_t_up = theta_prior + torch.sum(PHI_t[start_idx:end_idx], axis=1)
        GAMMA_t[start_idx:end_idx] = GAMMA_i_t_up    
        #print("getting ELBO using new GAMMA values")
        #get_elbo(ALPHA_t, PI_t, GAMMA_t, PHI_t)

    #calculate ELBO after GAMMA update --> seems to be lower than after PHI update so something might be wrong
    #print("Get ELBO post PHI and GAMMA updates for current batch of cells")
    #get_elbo(ALPHA_t, PI_t, GAMMA_t, PHI_t)

    return(PHI_t, GAMMA_t)    

def update_beta(J, K, PHI, GAMMA, final_data, alpha_prior=0.65, beta_prior=0.65):
    
    '''
    Update variational parameters for beta distribution
    '''
    
    # Re-initialize ALPHA and PI values
    ALPHA_t = torch.ones((J, K), dtype=torch.float64) * alpha_prior
    PI_t = torch.ones((J, K), dtype=torch.float64) * beta_prior

    # Set up indicies for extracting correct values from PHI
    first_indices = final_data.cell_id_index.values
    second_indices = final_data.junction_id_index.values

    # Calculate alphas and pis for each cell-junction pair 
    alphas = (torch.tensor(final_data.junc_count.values).unsqueeze(-1) * PHI[first_indices, second_indices, :])
    pis = (torch.tensor(final_data.clustminjunc.values).unsqueeze(-1) * PHI[first_indices, second_indices, :])
    
    # Create a tensor of the unique junction indices
    index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64)

    # Use scatter_add to sum the values for each unique index
    ALPHA_t = torch.scatter_add(ALPHA_t, 0, index_tensor[:, None].repeat(1, alphas.shape[1]), alphas)
    PI_t = torch.scatter_add(PI_t, 0, index_tensor[:, None].repeat(1, pis.shape[1]), pis)
    
    return(ALPHA_t, PI_t)

# %%   

def update_variational_parameters(ALPHA, PI, GAMMA, PHI, J, K, cell_junc_counts):
    
    '''
    Update variational parameters for beta, theta and z distributions
    '''

    PHI_up, GAMMA_up = update_z_theta(ALPHA, PI, GAMMA, PHI, cell_junc_counts) #slowest part#*****
    #print("got the PHI and GAMMA updates")
        
    ALPHA_up, PI_up = update_beta(J, K , PHI_up, GAMMA_up, cell_junc_counts)
    #print("got the ALPHA and PI updates")

    return(ALPHA_up, PI_up, GAMMA_up, PHI_up)

# %%

def calculate_CAVI(J, K, N, cell_junc_counts, num_iterations=5):
    
    '''
    Calculate CAVI
    '''

    ALPHA_init, PI_init, GAMMA_init, PHI_init = init_var_params(J, K, N)
    elbos_init = get_elbo(ALPHA_init, PI_init, GAMMA_init, PHI_init, cell_junc_counts)
    elbos = []
    elbos.append(elbos_init)
    print("Got the initial ELBO ^")
    
    ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi = update_variational_parameters(ALPHA_init, PI_init, GAMMA_init, PHI_init, J, K, cell_junc_counts)
    print("Got the first round of updates!")
    
    elbo_firstup = get_elbo(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, cell_junc_counts)
    elbos.append(elbo_firstup)
    
    print("got the first ELBO after updates ^")
    iter = 0

    while(elbos[-1] > elbos[-2]) and (iter < num_iterations):
        print("ELBO not converged, re-running CAVI iteration # " + str(iter+1))
        ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi = update_variational_parameters(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, J, K, cell_junc_counts)
        elbo = get_elbo(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, cell_junc_counts)
        elbos.append(elbo)
        iter = iter + 1
    
    print("ELBO converged, CAVI iteration # " + str(iter+1) + " complete")
    return(ALPHA_vi, PI_vi, GAMMA_vi, PHI_vi, elbos)

# %%
if __name__ == "__main__":

    # Load data and define global variables 
    # get input data (this is standard output from leafcutter-sc pipeline so the column names will always be the same)
    
    input_file = '/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/junctions_full_for_LDA.pkl.pkl' #this should be an argument that gets fed in
    coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion = load_cluster_data(input_file)

    juncs = coo_counts_sparse
    clusts = coo_cluster_sparse
    batch_size = 512 #should also be an argument that gets fed in
    
    #prep dataloader for training
    
    #choose random indices to subset coo_counts_sparse and coo_cluster_sparse
    rand_ind = np.random.choice(16585, size=2000, replace=False)

    cell_junc_counts = data.DataLoader(CSRDataLoader(coo_counts_sparse[rand_ind], coo_cluster_sparse[rand_ind, ]), batch_size=batch_size, shuffle=False)

    # global variables
    #N = len(cell_junc_counts.dataset) # number of cells
    N =coo_cluster_sparse.size()[0]
    J = coo_cluster_sparse.size()[1]
    #J = cell_junc_counts.dataset[0][0].shape[0] # number of junctions
    K = 2 # should also be an argument that gets fed in
    num_trials = 1 # should also be an argument that gets fed in
    num_iters = 5 # should also be an argument that gets fed in

    # loop over the number of trials (for now just testing using one trial but in general need to evaluate how performance is affected by number of trials)
    for t in range(num_trials):
        
        # run coordinate ascent VI
        print(K)

        ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = calculate_CAVI(J, K, N, cell_junc_counts, num_iters)
        juncs_probs = ALPHA_f / (ALPHA_f+PI_f)
        theta_f = distributions.Dirichlet(GAMMA_f).sample().numpy()
        z_f = distributions.Categorical(PHI_f).sample()

        #make theta_f a dataframe 
        theta_f_plot = pd.DataFrame(theta_f)
        theta_f_plot['cell_id'] = cell_ids_conversion["cell_type"].to_numpy()[rand_ind]

        celltypes = theta_f_plot.pop("cell_id")
        lut = dict(zip(celltypes.unique(), ["r", "b", "g", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]))
        print(lut)
        row_colors = celltypes.map(lut)
        print(sns.clustermap(theta_f_plot, row_colors=row_colors))
# %%
#sort GAMMA_f 
#GAMMA_f_sort = GAMMA_f.sort(dim=1, descending=True)

# for bimodal distribution mean might not be the best way to compare them since most density lies at the two tails 

# Calculate the variances for each of the 3624 junctions
#variances = (ALPHA_f * PI_f) / ((ALPHA_f + PI_f) ** 2 * (ALPHA_f + PI_f + 1))
#variances = variances.reshape(J, K)

# use symmetric KL divergence, take mean of each calculation both ways to compare 
# Beta distributions 
# squared helinger distance between two Beta distributions  

# what if we initialize the cell states to be the cell types 
# batch size shouldn't affect results right? 
# seems like it's learning similar success of probability of junctions being expressed in every cell state

#If you're interested in exploring this further, you can try visualizing the data for that particular 
# junction in each cell state to see if there are any obvious differences in the distributions. You can also 
# try fitting other distributions to the data to see if they provide a better fit, or investigate if there are 
# any other variables that may be driving the differences in the variances.

#TO-DO: double check junction counts, maybe only include cell that have at least X counts so that can actually learn something about them
#clusters should have X counts overall across all cells 
#maybe just need to start out with a smaller number of junction/clusters


# plot candidate junctions to show read counts distributions 
# across cell states

# %%

#rand_ind

#coo_counts_sparse[rand_ind]
#coo_cluster_sparse[rand_ind]

#cell_ids_conversion 
#junction_ids_conversion

#coo_counts_sparse[rand_ind][:,27953].transpose().toarray()