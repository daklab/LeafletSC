# %% 
import pandas as pd

import torch
print(torch.cuda.is_available())

import numpy as np
import argparse
from datetime import datetime
from scipy.stats import betabinom

from ast import literal_eval

from multiprocessing import Pool
import itertools  

import warnings
import math
warnings.filterwarnings("ignore") #temp solution

# Set seed for reproducibility.
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

startTime = datetime.now()

from pathlib import Path
Path("Clusters_ZI_results").mkdir(parents=True, exist_ok=True)

# %% 
parser = argparse.ArgumentParser(description='Read in junction files')

parser.add_argument('--intron_cluster', dest='introns',
                    help='intron clusters obtained by running leafcutter clustering on regtools output') #if single cell mode then should also contain column with individual cell ids and counts

parser.add_argument('--vglm_script_path', dest='run_vglm_bb_path', 
                    help='name of the output file to save junction count matrix to')

parser.add_argument('--output_directory', dest='outdir', 
                    help='name of the output file to save junction count matrix to')

parser.add_argument('--cell', dest='cell_type', 
                    help='cell type name to run script for to generate cluster files with expected and observed junction counts')

args = parser.parse_args()
print(args.cell_type)
cell_type=args.cell_type
path = args.run_vglm_bb_path

# %%
#required for R functionality
import rpy2.robjects as ro
path = "/gpfs/commons/home/kisaev/zoo-splice/data-summary/run_VGLM_BB.R"
import rpy2.robjects.numpy2ri 
rpy2.robjects.numpy2ri.activate()

# %%
#+++++++++++++++++++++++++++++++++++++++++
#Check for Zero Inflated junction counts
#Beta Binomial (BB)
#+++++++++++++++++++++++++++++++++++++++++

#Fit BB for each cell type independently
#so that one isoform not being expressed in a given cell 
#type doesn't end up looking like zero inflation

#cluster total counts 
#have a random variable z for each junction-cell pair (across individual cell types)
#z denotes if the count is zero (binomial)
#capture the distribution of sum(z) over all cells for a given junction
#and compare that to the observed sum(z)

#fit a and b for beta binomial using all cells 
#n_ij = cluster counts 
#junction j in cell i, junction count is y_ij

def get_missing_junctions(clust, cell, clusts):

    '''
    construct a count matrix for all cells and junctions
    recovers junctions for which a cell had zero counts and wasn't reported for example
    '''

    all_cluster_juncs = clusts[(clusts["Cluster"]==clust)].junction_id.unique()

    #clust_dat=clusts[(clusts["Cluster"]==clust)& (clusts["file_name"]==cell)][["Cluster", "junction_id", "cell_id", "count"]]
    clust_dat=clusts[(clusts["Cluster"]==clust)][["Cluster", "junction_id", "cell_id", "count"]]

    all_cells = clusts[clusts["file_name"]==cell].cell_id.unique() 
    all_cells=pd.Series(all_cells) 
    missing_cells=all_cells[~all_cells.isin(clust_dat.cell_id)]
    num_missing=len(missing_cells)
    
    missing_cells = pd.DataFrame({'Cluster': [clust] *num_missing, 'junction_id': ["missing"] *num_missing, 'cell_id': all_cells[~all_cells.isin(clust_dat.cell_id)], 'count': [0] *num_missing})
    all_cells_clust = pd.concat([clust_dat, missing_cells])
    all_cells_clust = all_cells_clust.sort_values(by=['cell_id'])

    #pivot table make wide matrix 
    count_matrix = pd.pivot_table(all_cells_clust, index='cell_id', columns='junction_id', values='count', fill_value=0)
    #remove temporary 'missing' value in the junction_id column 
    count_matrix=count_matrix.drop(columns="missing")
    #keep only cells of specific cell type
    idx=count_matrix.index.isin(all_cells) 
    count_matrix=count_matrix.iloc[idx]

    #cell type specific matrix with all junctions in cluster 
    return(count_matrix)

def get_alpha_beta(s, N):

    '''
    load R script that run the vglm betabinomialff to estimate the a and b parameters
    '''

    r=ro.r
    r.source(path)
    p=r.get_alpha_beta(s, N)
    return p 

def get_junc_beta_params(clust, cell, clusts):
    '''
    estimate a and b parameters for beta binomial for each junction individually (cell types looked at individually for now)
    then fit betabinomial to extract probability of observing zero counts for junction-cell pair
    '''

    #by individual cell types
    clust_matrix=get_missing_junctions(clust, cell, clusts) #do this for every cell type

    #remove cells that have 0 counts across cluster
    clust_matrix = clust_matrix.loc[(clust_matrix.sum(axis=1) != 0)]
    print("ready to estimate parameters for cluster " + str(clust))

    #should have at least 50 cells that have non-zero cluster count (can change this parameter)
    num_cells=clust_matrix.shape[0]

    if num_cells >= 50:

        juncs_cells_clusters = []

        print("Able to evaluate cluster: " + str(clust) + " in cell type " + str(cell))
    
        #total counts across all junctions in cluster for given cell
        tot_cells_counts=clust_matrix.sum(axis=1).ravel()

        #iterate over junctions 
        for j in range(len(list(clust_matrix))):
            juncs_cells_zij = []
            #junction counts across cells
            s=clust_matrix.iloc[:,j].ravel()
            try:
                #get alpha and beta (concentration = alpha+beta)
                res = get_alpha_beta(s, tot_cells_counts)
                a, b = np.array_split(res, 2)
                a, b = float(a), float(b)
                for i in range(len(tot_cells_counts)):
                    cell_count=clust_matrix.iloc[i,j]
                    tot_cell_count=tot_cells_counts[i]
                    try:
                        prob_count = betabinom.pmf(0, tot_cell_count, a, b) #want to get p when count is 0
                        cell_id = clust_matrix.index[i]
                        junc_id = list(clust_matrix)[j]
                        res = [clust, cell, cell_id, junc_id, cell_count, tot_cell_count, prob_count, a, b ]
                        juncs_cells_zij.append(res)
                    except:
                        pass            
            except:
                pass
            if(len(juncs_cells_zij) > 0):
                #currently assuming at least 1 junction in cluster will work
                juncs_cells_zij = pd.DataFrame(juncs_cells_zij, columns=["cluster", "cell", "cell_id", "junction_id", "cell_count", "tot_cell_count", "prob_count", "alpha", "beta"])
                juncs_cells_zij["z_truth"] = 1
                juncs_cells_zij.z_truth[juncs_cells_zij.cell_count == 0] = 1 
                juncs_cells_zij.z_truth[juncs_cells_zij.cell_count > 0]  =  0
                pois_bern_mean = juncs_cells_zij["prob_count"].sum()
                pois_bern_variance = (juncs_cells_zij["prob_count"] * (1-juncs_cells_zij["prob_count"])).sum()
                expected_upper_bount = (pois_bern_mean + (math.sqrt(pois_bern_variance) * 2))
                obtained = juncs_cells_zij.z_truth.sum()
                res = [clust, cell, junc_id, pois_bern_mean, pois_bern_variance, expected_upper_bount, obtained, num_cells, a, b]
                juncs_cells_clusters.append(res)  

        if(len(juncs_cells_clusters) > 0):
            # flatten, create df and drop duplicates
            juncs_cells_clusters = pd.DataFrame(juncs_cells_clusters, columns=["cluster", "cell", "junction_id", "pois_bern_mean", "pois_bern_variance", "expected_upper_bount", "obtained_zeroes", "num_cells", "alpha", "beta"])
            print(juncs_cells_clusters)
            file_name="Clusters_ZI_results/" + str(clust) + '_' + cell + '_junctions_prob_of_being_zero.txt.gz'
            juncs_cells_clusters.to_csv(file_name, index=False, sep="}")  #find alterantive more efficient way to save this file
            print("done estimating BB PMF for " + str(clust))

def main():
    
    #get alpha beta parameters for each junctions (within each cell type seperatley)
    pool=Pool(processes=4)
    #clust_file = '/gpfs/commons/home/kisaev/leafcutter-sc/test.h5' #this should be an argument that gets fed in
    clust_file = args.introns   
    clusts = pd.read_hdf(clust_file, 'df')

    all_cells = clusts.cell_id.unique() #cells across different cell types can have the same cell barcode, came from different sublibraries 
    all_cells=pd.Series(all_cells) 
    print("The number of total cells evaluated is " + str(len(all_cells))) 

    cells_types = clusts[["cell", "file_name", "cell_id"]].drop_duplicates()
    print(cells_types.groupby(["file_name"])["file_name"].count())
    unique_cells = cells_types.file_name.unique()

    all_clusters=list(clusts.Cluster.unique())
    
    #get list of all possible clust-cell inputs
    cell_types = cells_types.file_name.unique()
    cell_types=list(cell_types)
    pool.starmap(get_junc_beta_params, zip(all_clusters, itertools.repeat(cell_types)), clusts)

    print("Done running all clusters")

    #write log file 
    wd_path = args.outdir
    file_name="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/Clusters_ZI_results/" + str(cell_type) +".log" 
    pd.DataFrame({'cell_type' : [cell_type]}).to_csv(file_name, index=False) 
    print(datetime.now() - startTime)

if __name__=="__main__":
    main()