# %%
import pandas as pd
import argparse
from ast import literal_eval
import numpy as np
import itertools
from io import BytesIO
import tqdm
import dask.dataframe as dd
from dask import delayed
import matplotlib.pyplot as plt
import dask
dask.config.set(scheduler='threads')

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
# load the clustered data /gpfs/commons/groups/knowles_lab/Karin/data/GTEx/clustered_junctions.h5
clusts = pd.read_hdf("/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/clustered_junctions_minjunccounts.h5", key='df') # these have start-1 coordinates compared to original GTEx matrix

# make Name column to match GTEx file by first need to add "chr" before Chromosome column and subtract 1 from Start column 
clusts["Name"] = "chr" + clusts["Chromosome"].astype(str) + "_" + (clusts["Start"]+1).astype(str) + "_" + clusts["End"].astype(str)

# %%
print(clusts.head())

# %%
# Remove singleton clusters where Count == 1
clusts = clusts[clusts["Count"] > 1]
len(clusts.Name.unique())

# %%
# order clusts by descending count
clusts = clusts.sort_values(by="Count", ascending=False)
clusts.head()

# remove clusters with more than 3 junctions
clusts = clusts[clusts["Count"] <= 3]
len(clusts.Name.unique())

# %%
# gtex sample annotations 
samples = pd.read_csv("/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
samples = samples[["SAMPID", "SMTS", "SMTSD"]].drop_duplicates()
print(samples.head())

# %%
# make a dataframe for each tissue type in SMTS column that has each sample ID and the tissue type with corresponding junctions and their counts 

# %%
clusts_simple = clusts[["Name", "Cluster", "gene_name"]].drop_duplicates()
# reset index in the dataframe
clusts_simple = clusts_simple.reset_index(drop=True)
print(clusts_simple.head())

# %%
len(clusts_simple.Cluster.unique())

# %%
len(clusts_simple.Name.unique())

# %%
# subsample 10000 Cluster IDs for a test run
clusts_sample = clusts_simple.sample(n=10000, random_state=1)
print(len(clusts_sample.Cluster.unique()))
print(len(clusts_sample.Name.unique()))

# %%
clusts_sample.head()

# %%
import dask.dataframe as dd
from tqdm import tqdm

gtex_juncs = '/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'

class MeltedJunctionsCSV:
    
    def __init__(self, file_name, clusts_names, clusts, samples, tot_juncs):
        self.file_name = file_name
        self.clusts_names = clusts_names
        self.clusts = clusts
        self.samples = samples
        self.tot_juncs = tot_juncs
        
    def melt_junctions(self):
        # Create an empty list to store the melted rows
        melted_dfs = []
        reader = pd.read_csv(self.file_name, sep="\t", chunksize=512, header=0)
        # Get total number of reads using just one column (don't read whole file)
        total_rows = 357747
        for chunk in tqdm(reader, total=total_rows/512, unit=" chunk"):
            # check if junction is in clusts_names otherwise skip 
            filtered_chunk = chunk[chunk["Name"].isin(self.clusts_names)]
            if not filtered_chunk.empty:
                print("Processing junctions:")
                sample_df = filtered_chunk.melt(
                    id_vars=['Name', 'Description'],
                    var_name='Sample',
                    value_name='Count')
                # Remove rows with zero counts, turn count column into int64
                sample_df['Count'] = sample_df['Count'].astype('int64')
                sample_df = sample_df[sample_df['Count'] > 0]
                # Concatenate sample_df to the list of melted dataframes
                melted_dfs.append(sample_df)
        return melted_dfs
    
# %%
import multiprocessing

class MeltedJunctionsCSV:
    def __init__(self, file_name, clusts_names, clusts, samples, tot_juncs):
        self.file_name = file_name
        self.clusts_names = clusts_names
        self.clusts = clusts
        self.samples = samples
        self.tot_juncs = tot_juncs
        
    def process_chunk(self, chunk):
        filtered_chunk = chunk[chunk["Name"].isin(self.clusts_names)]
        if not filtered_chunk.empty:
            print("Processing junctions:")
            sample_df = filtered_chunk.melt(
                id_vars=['Name', 'Description'],
                var_name='Sample',
                value_name='Count')
            # Remove rows with zero counts, turn count column into int64
            sample_df['Count'] = sample_df['Count'].astype('int64')
            sample_df = sample_df[sample_df['Count'] > 0]
            return sample_df
        else:
            return None
        
    def melt_junctions(self):
        # Create an empty list to store the melted rows
        melted_dfs = []
        print("Reading file...")
        reader = pd.read_csv(self.file_name, sep="\t", chunksize=512, header=0)
        # Get total number of reads using just one column (don't read the whole file)
        print("Initializing multiprocessing...")
        num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
        print(f"Number of processes: {num_processes}")
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(self.process_chunk, reader)
            
        melted_dfs = [df for df in results if df is not None]
        
        return melted_dfs

# %% 
class MeltedJunctions:
    def __init__(self, file_name, clusts_names, clusts, samples, tot_juncs):
        self.file_name = file_name
        self.clusts_names = clusts_names
        self.clusts = clusts
        self.samples = samples
        self.tot_juncs = tot_juncs
        
    def melt_junctions(self):
        melted_dfs = []
        
        # Read in the file as a Dask DataFrame
        dask_df = dd.read_csv(self.file_name, sample=1000000, sep="\t", blocksize=25e6)
        
        # Extract the header (sample names)
        header = list(dask_df.columns[2:])

        print("Number of samples in the file: ", len(header))
        # Group the samples by tissue
        samples_df = self.samples

        # Keep only samples that are found in the header 
        samples_df = samples_df[samples_df['SAMPID'].isin(header)]
        grouped_samples = samples_df.groupby('SMTS')['SAMPID'].apply(list)
        # Iterate over the tissues and split the count matrix
        print("Iterating over tissues...")

        for tissue, samples in grouped_samples.items():
            print("Processing tissue: ", tissue)
            # Get the column indices for the samples in the current tissue
            sample_indices = [header.index(sample) for sample in samples]
            print("Number of samples in the current tissue: ", str(len(sample_indices)))
            for sample in samples:
                print("Processing sample:", sample)
                index_sample=header.index(sample)+2
                # read in file as pandas dataframe for just this column 
                sample_df = pd.read_csv(self.file_name, usecols=['Name', 'Description', sample], header=0, sep="\t")

                sample_df = dask_df[['Name', 'Description', sample]]
                sample_df = sample_df[sample_df['Name'].isin(self.clusts_names)]
                sample_df['Tissue'] = tissue

                # Merge with cluster info to get Cluster ID
                sample_df = sample_df.merge(self.clusts, on="Name", how="left")

                # Melt the dataframe
                sample_df = sample_df.melt(
                    id_vars=['Name', 'Description', 'Tissue', 'gene_name', 'Cluster'],
                    var_name='Sample',
                    value_name='Count')                

                # Remove rows with zero counts, turn count column into int64
                sample_df['Count'] = sample_df['Count'].astype('int64')
                sample_df = sample_df[sample_df['Count'] > 0]

                # Get total counts per cluster in sample
                cluster_counts = sample_df.groupby(["Cluster"])["Count"].sum().reset_index()
                cluster_counts.columns = ['Cluster', 'Cluster_Counts']
                sample_df = sample_df.merge(cluster_counts, on=["Cluster"], how="left")

                # Append the melted dataframe to the list
                melted_dfs.append(sample_df)
        
        print("Concatenating melted dataframes...")
        return melted_dfs

# %%
# create an instance of the class with the file name and clusts names as arguments
tot_juncs = len(clusts_sample.Name.unique())
melted_junctions = MeltedJunctions(gtex_juncs, clusts_sample.Name, clusts_sample, samples, tot_juncs)
melted_df = melted_junctions.melt_junctions()

# %%
#cluster_counts= test.groupby(["Sample", "Cluster"])["Count"].sum().reset_index()
#cluster_counts.columns = ['Sample', 'Cluster', 'Cluster_Counts']    
#cluster_counts
#test.merge(cluster_counts, on=["Sample", "Cluster"], how="left")

# %%
#save file and use as input for LDA script 
# Need to get total cluster counts for each sample-junction pair  (figure out how to do this later it's too much operation for single dask?)
#cluster_counts= tissue_df.groupby(["Sample", "Cluster"])["Count"].sum().reset_index()
#cluster_counts.columns = ['Sample', 'Cluster', 'Cluster_Counts']    
#tissue_df = tissue_df.merge(clust_counts, on=["Sample", "Cluster"], how="left")
#print(cluster_counts.head())
#summarized_data["junc_ratio"] = summarized_data["junc_count"] / summarized_data["Cluster_Counts"]
#summarized_data['sample_id_index'] = summarized_data.groupby('SAMPID').ngroup()
#summarized_data['junction_id_index'] = summarized_data.groupby('Name').ngroup()

# %%
#summarized_data.to_hdf("/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_junction_cluster_counts" + ".h5", key='df', mode='w', format="table")
