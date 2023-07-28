import pandas as pd
import argparse
from ast import literal_eval
import numpy as np
from tqdm import tqdm
import concurrent.futures
import time

pd.options.mode.chained_assignment = None  # default='warn'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.strings")

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--intron_clusters', dest='intron_clusters',
                    help='path to the file that has the intron cluster events and junction information from running intron_clustering.py')
parser.add_argument('--output_file', dest='output_file',
                    help='how you want to name the output file, this will be the input for the beta-binomial LDA model')
parser.add_argument('--has_genes', dest='has_genes',
                    help='Yes if intron clustering was done with a gtf file, No if intron clustering was done in an annotation free manner')
parser.add_argument('--chunk_size', dest='chunk_size', default=5000,
                    help='how many lines to read in at a time, default is 5000')
args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_chunk(chunk):
    chunk = chunk.copy()
    chunk['cell_readcounts'] = chunk['cell_readcounts'].str.split(',')
    chunk = chunk.explode('cell_readcounts')
    chunk[['cell', 'junc_count']] = chunk['cell_readcounts'].str.split(pat=':', n=1, expand=True)
    chunk = chunk.drop(['cell_readcounts'], axis=1)
    return chunk

def main(intron_clusters, output_file, has_genes, chunk_size):
    """ 
    Create a COO sparse matrix that we can just feed in later to our 
    beta-binomial LDA model 
    """

    start_time = time.time()

    print("The intron clusters file you provided is " + str(intron_clusters) + ", reading in chunks of " + str(chunk_size) + " lines")
    clusts_chunks = pd.read_csv(intron_clusters, sep="}", header=0, low_memory=False, chunksize=int(chunk_size))

    print("Processing intron clusters")
    clusts_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for chunk in tqdm(clusts_chunks):
            clusts_list.append(executor.submit(process_chunk, chunk))

    clusts = pd.concat([future.result() for future in clusts_list])
    clusts["cell_id"] = clusts.cell
    clusts = clusts.drop(['cell'], axis=1)

    print("The number of intron clusters evaluated is " + str(len(clusts.Cluster.unique())))
    print("The number of junctions evaluated is " + str(len(clusts.junction_id.unique())))

    if(has_genes=="yes"):
        print("A gtf file was used to generate intron clusters")
        summarized_data = clusts[["cell_id", "junction_id", "gene_id", "junc_count", "Cluster", "file_name"]] 
    if(has_genes=="no"):
        print("No gtf file was used to generate intron clusters")
        summarized_data = clusts[["cell_id", "junction_id", "junc_count", "Cluster", "cell_type"]]
    
    summarized_data = summarized_data.copy()
    summarized_data["junc_count"] = summarized_data["junc_count"].astype(int)
    
    #Need to get total cluster counts for each cell 
    clust_cell_counts= summarized_data.groupby(["cell_id", "Cluster"])["junc_count"].sum().reset_index()
    clust_cell_counts.columns = ['cell_id', 'Cluster', 'Cluster_Counts']

    all_cells = clusts.cell_id.unique()  
    all_cells=pd.Series(all_cells) 

    print("The number of total cells evaluated is " + str(len(all_cells))) 

    cells_types = clusts[["cell_type", "cell_id"]].drop_duplicates()
    print(cells_types.groupby(["cell_type"])["cell_type"].count())
 
    print("Ensuring that each cell-junction pair appears only once")
    summarized_data = summarized_data.drop_duplicates(subset=['cell_id', 'junction_id'], keep='last') #double check if this is still necessary

    print("Merge cluster counts with summarized data")
    summarized_data = clust_cell_counts.merge(summarized_data)
    print("DOne merging cluster counts with summarized data")

    #save file and use as input for LDA script 
    summarized_data["junc_ratio"] = summarized_data["junc_count"] / summarized_data["Cluster_Counts"]
    summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
    summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()

    print("Done making the input file for beta-binomial mixture model, now saving splitting it up by cell type and saving as hdf file")

    # split summarized_data file by cell_type and save each one as a hdf file with output_file + cell type name
    summarized_data_split = summarized_data.groupby('cell_type')
    for name, group in summarized_data_split:
        group.to_hdf(output_file + "_" + name + ".h5", key='df', mode='w', complevel=9, complib='zlib')

    print("Done generating input file for beta-binomial LDA model. This process took " + str(round(time.time() - start_time)) + " seconds")

if __name__ == '__main__':
    intron_clusters=args.intron_clusters
    output_file=args.output_file
    has_genes=args.has_genes
    chunk_size=args.chunk_size
    main(intron_clusters, output_file, has_genes, chunk_size)

# to run 
#cluster_file="/gpfs/commons/scratch/kisaev/ss_tabulamuris_test/Leaflet/clustered_junctions_noanno_anno_free_100_50000_20_10_0.1_single_cell.gz"
#output_file="/gpfs/commons/scratch/kisaev/ss_tabulamuris_test/Leaflet/BBmixture_mm10ss2_input_fulldata"

#for testing
#cluster_file="/gpfs/commons/home/kisaev/slurm/test_clusters"
#output_file="/gpfs/commons/scratch/kisaev/ss_tabulamuris_test/Leaflet/BBmixture_mm10ss2_test"

#cd Leaflet

#sbatch --wrap "python /gpfs/commons/home/kisaev/Leaflet/src/beta-binomial-lda/01_prepare_input_coo.py --intron_clusters $cluster_file --output_file $output_file --has_genes "no" --chunk_size 10000" --mem 1024G -p bigmem -J "BBmixture"
