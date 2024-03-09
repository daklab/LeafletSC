import pandas as pd
import argparse
from ast import literal_eval
import numpy as np
from tqdm import tqdm
import concurrent.futures
import time
import tables  

#pd.options.mode.chained_assignment = None  # default='warn'
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.strings")

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--intron_clusters', dest='intron_clusters',
                    help='path to the file that has the intron cluster events and junction information from running intron_clustering.py')

parser.add_argument('--output_file', dest='output_file', 
                    default="output_file",
                    help='how you want to name the output file, this will be the input for all Leaflet models')

parser.add_argument('--has_genes', dest='has_genes',
                    default="no",
                    help='yes if intron clustering was done with a gtf file, No if intron clustering was done in an annotation free manner')

parser.add_argument('--chunk_size', dest='chunk_size', 
                    default=5000,
                    help='how many lines to read in at a time, default is 5000')

parser.add_argument('--metadata', dest='metadata',
                    default=None,
                    help='path to the metadata file, if provided, the output file will have cell type information')

args, unknown = parser.parse_known_args()

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

def main(intron_clusters, output_file, has_genes, chunk_size, metadata):
    
    """ 
    Create input for sparse representation that we will feed into Leaflets models 

    Required columns in metadata file are: bam_file_name (cell id), free_annotation (cell type information)
    If has genes is yes, then gene_id column is also required

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
    print("Done processing intron clusters")

    # if metadata file is provided, add cell type information to the output file
    if args.metadata is not None:

        print("Reading in metadata file")
        metadata = pd.read_csv(args.metadata)
        
        # Create a dictionary to store the mapping
        mappings = []

        # Create a subset of clusts dataframe just dataframe with unique cell
        clusts_cells = clusts.drop_duplicates(subset=['cell'])

        # Iterate through the rows in the metadata dataframe to find the matching cell
        for index, row in tqdm(metadata.iterrows()):
            bam_file_name = row['bam_file_name']
            # Check for partial matches in the 'cell' column of the clusts dataframe
            matching_cell = clusts_cells[clusts_cells['cell'].str.contains(bam_file_name, case=False)]['cell'].tolist()
            if len(matching_cell) == 1:
                matching_cell = matching_cell[0]
                mappings.append({'bam_file_name': bam_file_name, 'cell': matching_cell})

        # Create a dataframe from the list of mappings
        mapping_df = pd.DataFrame(mappings)
        metadata = metadata.merge(mapping_df, on='bam_file_name', how='left')
        # Merge the metadata dataframe with the clusts dataframe
        clusts = clusts.merge(metadata, on='cell', how='left')
        clusts["cell_id"] = clusts["bam_file_name"]
        clusts["cell_type"] = clusts["free_annotation"]

    if args.metadata is None: 
        # Necessary to do this because it's possible (though rare) that cells get same ID especially in SS2 data
        clusts["cell_id"] = clusts.cell + "_" + clusts.cell_type
    
    clusts = clusts.drop(['cell'], axis=1)

    print("The number of intron clusters evaluated is " + str(len(clusts.Cluster.unique())))
    print("The number of junctions evaluated is " + str(len(clusts.junction_id.unique())))
    # numer of uniqiue cells 
    print("The number of cells evaluated is " + str(len(clusts.cell_id.unique())))
    # print the number of clusters with only one junction
    print("The number of clusters with only one junction is " + str(len(clusts[clusts.Count==1].Cluster.unique())))

    if(has_genes=="yes"):
        print("A gtf file was used to generate intron clusters")
        summarized_data = clusts[["cell_id", "junction_id", "gene_id", "junc_count", "Cluster", "cell_type"]] 
    if(has_genes=="no"):
        print("No gtf file was used to generate intron clusters")
        summarized_data = clusts[["cell_id", "junction_id", "junc_count", "Cluster", "cell_type"]]
    
    summarized_data = summarized_data.copy()
    summarized_data["junc_count"] = summarized_data["junc_count"].astype(int)
    
    #Need to get total cluster counts for each cell 
    clust_cell_counts= summarized_data.groupby(["cell_id", "Cluster"])["junc_count"].sum().reset_index()
    clust_cell_counts.columns = ['cell_id', 'Cluster', 'Cluster_Counts']

    all_cells = clusts.cell_id.unique()  
    all_cells = pd.Series(all_cells) 

    print("The number of total cells evaluated is " + str(len(all_cells))) 

    cells_types = clusts[["cell_type", "cell_id"]].drop_duplicates()
    print("The number of cells per cell type is:")
    print(cells_types.groupby(["cell_type"])["cell_type"].count())
 
    summarized_data = summarized_data.drop_duplicates(subset=['cell_id', 'junction_id'], keep='last') #double check if this is still necessary
    summarized_data = clust_cell_counts.merge(summarized_data)

    print(np.unique(summarized_data['cell_id'].values))
    summarized_data["junc_ratio"] = summarized_data["junc_count"] / summarized_data["Cluster_Counts"]

    #save file and use as input Leaflet models 
    summarized_data['cell_id_index'] = summarized_data.groupby('cell_id').ngroup()
    summarized_data['junction_id_index'] = summarized_data.groupby('junction_id').ngroup()
    
    if metadata is not None:
        print("Done making the input file for Leaflet models, now saving splitting it up by cell type and saving as hdf file")
        # split summarized_data file by cell_type and save each one as a hdf file with output_file + cell type name
        summarized_data_split = summarized_data.groupby('cell_type')
        for name, group in summarized_data_split:
            # if "/" detected in name (cell_type) replace it with "_"
            if "/" in name:
                name = name.replace("/", "_")
            group.to_hdf(output_file + "_" + name + ".h5", key='df', mode='w', complevel=9, complib='zlib')
            print("You can find the resulting file at " + output_file + "_" + name + ".h5")

    if metadata is None:
        # save summarized_data as hdf file
        summarized_data.to_hdf(output_file + ".h5", key='df', mode='w', complevel=9, complib='zlib')    
        print("You can find the resulting file at " + output_file + ".h5")

    print("Done generating input file for Leaflet model. This process took " + str(round(time.time() - start_time)) + " seconds")

if __name__ == '__main__':

    intron_clusters=args.intron_clusters
    output_file=args.output_file
    has_genes=args.has_genes
    chunk_size=args.chunk_size
    metadata=args.metadata

    main(intron_clusters, output_file, has_genes, chunk_size, metadata)

