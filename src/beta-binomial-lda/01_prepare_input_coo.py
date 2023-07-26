import pandas as pd
import argparse
from ast import literal_eval
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--intron_clusters', dest='intron_clusters',
                    help='path to the file that has the intron cluster events and junction information from running intron_clustering.py')
parser.add_argument('--output_file', dest='output_file',
                    help='how you want to name the output file, this will be the input for the beta-binomial LDA model')
parser.add_argument('--has_genes', dest='has_genes',
                    help='Yes if intron clustering was done with a gtf file, No if intron clustering was done in an annotation free manner')
args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main(intron_clusters, output_file, has_genes):
    """ 
    Create a COO sparse matrix that we can just feed in later to our 
    beta-binomial LDA model 
    """

    clusts=pd.read_csv(intron_clusters, sep="}", header=0, low_memory=False)

    #seperate individual cells (this step is pretty slow! how can I speed it up?)
    clusts.split_up = clusts.split_up.apply(literal_eval) 
    clusts=clusts.explode('split_up')
    clusts[['cell', 'junc_count']] = clusts['split_up'].str.split(':', 1, expand=True) # this step may be fairly memory intensive
    clusts=clusts.drop(['split_up'], axis=1)
    clusts["cell_id"] = clusts.file_name + "_" + clusts.cell
    clusts = clusts.reset_index(drop=True)

    print("The number of intron clusters evaluated is " + str(len(clusts.Cluster.unique())))
    print("The number of junctions evaluated is " + str(len(clusts.junction_id.unique())))
    
    if(has_genes=="yes"):
        print("A gtf file was used to generate intron clusters")
        summarized_data = clusts[["cell_id", "junction_id", "gene_id", "junc_count", "Cluster", "file_name", "score"]] #the score is the original column from regtools results, indicating total counts for junction across all single cells from that cell type
    if(has_genes=="no"):
        print("No gtf file was used to generate intron clusters")
        summarized_data = clusts[["cell_id", "junction_id", "junc_count", "Cluster", "file_name", "score"]]
    
    summarized_data["junc_count"] = summarized_data["junc_count"].astype(int)
    
    #need to get total cluster counts for each cell 
    clust_cell_counts= summarized_data.groupby(["cell_id", "Cluster"])["junc_count"].sum().reset_index()
    clust_cell_counts.style.hide_index()
    clust_cell_counts.columns = ['cell_id', 'Cluster', 'Cluster_Counts']    

    all_cells = clusts.cell_id.unique()  
    all_cells=pd.Series(all_cells) 

    print("The number of total cells evaluated is " + str(len(all_cells))) 

    cells_types = clusts[["cell", "file_name", "cell_id"]].drop_duplicates()
    print(cells_types.groupby(["file_name"])["file_name"].count())
 
    summarized_data = summarized_data.drop_duplicates(subset=['cell_id', 'junction_id'], keep='last') #double check if this is still necessary
    summarized_data = clust_cell_counts.merge(summarized_data)

    #save file and use as input for LDA script 
    summarized_data["junc_ratio"] = summarized_data["junc_count"] / summarized_data["Cluster_Counts"]
    summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
    summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()

    #Create a cell type column (file_name) will be used as that 
    summarized_data["cell_type"] = summarized_data["file_name"]
    summarized_data.drop(["file_name"], axis=1, inplace=True)

    #save as hdf file 
    summarized_data.to_hdf(output_file + ".h5", key='df', mode='w')

    print("Done generating input file for beta-binomial LDA model")

if __name__ == '__main__':
    intron_clusters=args.intron_clusters
    output_file=args.output_file
    has_genes=args.has_genes
    main(intron_clusters, output_file, has_genes)


# to run 
#cluster_file="/gpfs/commons/scratch/kisaev/ss_tabulamuris_test/Leaflet/clustered_junctions_noanno_anno_free_50_100000_10_5_0.1_single_cell.gz"
#cd Leaflet
#python src/beta-binomial-lda/01_prepare_input_coo.py --intron_clusters $cluster_file --output_file "/gpfs/commons/scratch/kisaev/ss_tabulamuris_test/Leaflet/BBmixture_mm10ss2_input" --has_genes "no"