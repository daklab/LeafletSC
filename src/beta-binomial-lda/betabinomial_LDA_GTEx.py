# %%
from importlib import reload
from load_cluster_data import load_cluster_data
import gc

import numpy as np
import torch
import pandas as pd 
import seaborn as sns
import collections

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

float_type = { 
    "device" : device, 
    "dtype" : torch.float, # save memory
}

hypers = {
    "eta" : 1., 
    "alpha_prior" : 1., # karin had 0.65 
    "pi_prior" : 1.
}

K = 10

import plotnine as p9
import scipy.sparse as sp
import matplotlib.pyplot as plt 
import splicing_PCA_utils
from nuclear_norm_PCA import sparse_sum

from pca_kmeans_init import pca_kmeans_init
from betabinomo_LDA_singlecells_kinit import *
import betabinomo_LDA_singlecells_kinit
reload(betabinomo_LDA_singlecells_kinit)


# %% [markdown]
# ### Load data

# %%
input_file = '/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/melted_df_subsample.pickle'
summarize_data = pd.read_pickle(input_file)
print(len(summarize_data))

# %%
# concatenate all the chunks in summarize_data into one dataframe 
summarize_data = pd.concat(summarize_data)
summarize_data.head()

# %%
summarize_data.shape

# %%
print(summarize_data.SMTS.unique())

# %%
len(summarize_data.SMTS.unique())   

# %%
summarize_data.head()

# %%
# remove junctions that have multiple gene names associated with them (multiple rows per junction)
summarize_data['junction_id_index'] = summarize_data.groupby('Name').ngroup()
junction_ids_conversion = summarize_data[["junction_id_index", "Name", "Cluster", "Description"]].drop_duplicates()
junction_ids_conversion = junction_ids_conversion.sort_values("junction_id_index")

# %%
# remove junctions if they appear with multiple gene names
juncs_keep = summarize_data[["junction_id_index", "Name", "Cluster", "Description", "gene_name"]].drop_duplicates()

# %%
# find junctions in juncs_keep that appear twice because they are associated with multiple gene_names 
clusts_remove = juncs_keep[juncs_keep.duplicated(subset="junction_id_index")].Cluster.unique()
summarize_data = summarize_data[~summarize_data.Cluster.isin(clusts_remove)]

# %%
summarize_data['sample_id_index'] = summarize_data.groupby('Sample').ngroup()
summarize_data['junction_id_index'] = summarize_data.groupby('Name').ngroup()
coo = summarize_data[["sample_id_index", "junction_id_index", "Count", "Cluster_Counts", "Cluster", "JunctionUsageRatio"]]

# %%
# just some sanity checks to make sure indices are in order 
cell_ids_conversion = summarize_data[["sample_id_index", "Sample", "SMTS", "SMTSD"]].drop_duplicates()
cell_ids_conversion = cell_ids_conversion.sort_values("sample_id_index")

junction_ids_conversion = summarize_data[["junction_id_index", "Name", "Cluster", "Description", "gene_name"]].drop_duplicates()
junction_ids_conversion = junction_ids_conversion.sort_values("junction_id_index")

# %% [markdown]
# ### Prep data as input into LDA 

# %%
# rename summarize_data ample_id_index to cell_id_index and Count to junc_count
summarize_data = summarize_data.rename(columns={"junction_id_index" : "junction_id_index", "sample_id_index": "cell_id_index", "Count": "junc_count"})

# %%
summarize_data["clustminjunc"] = summarize_data["Cluster_Counts"] - summarize_data["junc_count"]
summarize_data.sort_values("JunctionUsageRatio", ascending=True).head()

# %% [markdown]
# ### Run LDA

# %%
float_type["device"]

# %%
# final prep of data 
final_data = make_torch_data(summarize_data, **float_type)

# %%
final_data

# %%
K

# %%
num_trials = 1 # can't currently run more than 1 or overflow GPU memory :( 
num_iters = 300 # should also be an argument that gets fed in

# loop over the number of trials (for now just testing using one trial but in general need to evaluate how performance is affected by number of trials)
for t in range(num_trials):

    # run coordinate ascent VI
    print(K)

    ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = calculate_CAVI(K, final_data, float_type, hypers = hypers, num_iterations = num_iters)
    elbos_all = np.array(elbos_all)
    juncs_probs = ALPHA_f / (ALPHA_f+PI_f)
    
    theta_f = GAMMA_f / GAMMA_f.sum(1,keepdim=True)
    theta_f_plot = pd.DataFrame(theta_f.cpu())
    theta_f_plot['SMTS'] = cell_ids_conversion["SMTS"].to_numpy()
    theta_f_plot_summ = theta_f_plot.groupby('SMTS').mean()
    print(theta_f_plot_summ)
    
    # plot ELBO
    plt.plot(elbos_all[2:]); plt.show()

# %%
x = theta_f.cpu().numpy()
x -= x.mean(1,keepdims=True)
x /= x.std(1,keepdims=True)
plt.hist(x.flatten(),100)
pd.crosstab( cell_ids_conversion["cell_type"], x.argmax(axis=1) )


