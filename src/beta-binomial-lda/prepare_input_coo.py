import pandas as pd
import numpy as np
import argparse
from ast import literal_eval
import scipy
import scipy.special
from scipy.special import gamma, digamma, loggamma
import copy
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'

import delayedsparse
import scipy.sparse
from scipy.sparse import coo_matrix

import anndata as ad
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--intron_clusters', dest='intron_clusters',
                    help='path to the file that has the intron cluster events and junction information from running intron_clustering.py')
parser.add_argument('--output_file', dest='output_file',
                    help='how you want to name the output file, this will be the input for the beta-binomial LDA model')

args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main(intron_clusters, output_file):
    """ 
    Create a COO sparse matrix that we can just feed in later to our 
    beta-binomial LDA model 
    """

    clusts=pd.read_csv(intron_clusters, sep="}", header=0, low_memory=False)

    #seperate individual cells (this step is pretty slow)
    clusts.split_up = clusts.split_up.apply(literal_eval) 
    clusts=clusts.explode('split_up')
    clusts[['cell', 'count']] = clusts['split_up'].str.split(':', 1, expand=True)
    clusts=clusts.drop(['split_up'], axis=1)
    clusts["cell_id"] = clusts.file_name + "_" + clusts.cell
    clusts = clusts.reset_index(drop=True)

    print("The number of intron clusters evaluated is " + str(len(clusts.Cluster.unique())))

    summarized_data = clusts[["cell_id", "junction_id", "count", "Cluster", "file_name", "score"]]
    summarized_data["count"] = summarized_data["count"].astype(int)
    
    #need to get total cluster counts for each cell 
    clust_cell_counts= summarized_data.groupby(["cell_id", "Cluster"])["count"].sum().reset_index()
    clust_cell_counts.style.hide_index()
    clust_cell_counts.columns = ['cell_id', 'Cluster', 'Cluster_Counts']    

    all_cells = clusts.cell_id.unique()  
    all_cells=pd.Series(all_cells) 
    print("The number of total cells evaluated is " + str(len(all_cells))) 

    cells_types = clusts[["cell", "file_name", "cell_id"]].drop_duplicates()
    print(cells_types.groupby(["file_name"])["file_name"].count())
    unique_cells = cells_types.file_name.unique()
 
    summarized_data = summarized_data.drop_duplicates(subset=['cell_id', 'junction_id'], keep='last') #double check if this is still necessary
    summarized_data = clust_cell_counts.merge(summarized_data)
    juncs = summarized_data[["junction_id"]].drop_duplicates()

    #save file and use as input for LDA script 
    summarized_data["junc_ratio"] = summarized_data["count"] / summarized_data["Cluster_Counts"]
    summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
    summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()

    summarized_data.to_csv(output_file, index=False, compression="gzip")

if __name__ == '__main__':
    intron_clusters=args.intron_clusters
    output_file=args.output_file
    main(intron_clusters, output_file)