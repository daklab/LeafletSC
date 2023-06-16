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
# rename first column Sample 
samples = samples.rename(columns={"SAMPID": "Sample"})
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
#clusts_sample = clusts_simple.sample(n=10000, random_state=1)
#print(len(clusts_sample.Cluster.unique()))
#print(len(clusts_sample.Name.unique()))
clusts_sample = clusts_simple

# %%
clusts_sample.head()

# %%
import dask.dataframe as dd
from tqdm import tqdm
import multiprocessing

gtex_juncs = '/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'
    
# %%
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
            # Merge with clusts df to get gene names
            sample_df = sample_df.merge(self.clusts, on="Name", how="left")
            # Get total cluster counts per sample and merge back with sample_df
            cluster_counts = sample_df.groupby(["Sample", "Cluster"])["Count"].sum().reset_index()
            # Rename last column to TotalCount
            cluster_counts.columns = ['Sample', 'Cluster', 'Cluster_Counts']    
            sample_df = sample_df.merge(cluster_counts, on=["Sample", "Cluster"], how="left")
            # Get junction usage ratio for each sample 
            sample_df["JunctionUsageRatio"] = sample_df["Count"] / sample_df["Cluster_Counts"]
            # Merge with sample annotations only for samples in the sample_df
            sample_df = sample_df.merge(self.samples)
            return sample_df
        else:
            print("No junctions found in chunk")
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
# create an instance of the class with the file name and clusts names as arguments
tot_juncs = len(clusts_sample.Name.unique())

print("Running using ALL junctions... and ALL samples!")
melted_junctions = MeltedJunctionsCSV(gtex_juncs, clusts_sample.Name, clusts_sample, samples, tot_juncs)
melted_df = melted_junctions.melt_junctions()

# save object as pickle file
import pickle
with open('/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/melted_df_subsample.pickle', 'wb') as handle:
    pickle.dump(melted_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("done!!!")

# %%
#save file and use as input for LDA script 
#summarized_data['sample_id_index'] = summarized_data.groupby('SAMPID').ngroup()
#summarized_data['junction_id_index'] = summarized_data.groupby('Name').ngroup()

# %%
#summarized_data.to_hdf("/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_junction_cluster_counts" + ".h5", key='df', mode='w', format="table")

#cd /gpfs/commons/home/kisaev/leafcutter-sc
#python_script=src/beta-binomial-lda/GTEx_prepare_input_job.py
#sbatch --wrap "python $python_script" --mem 150G -c 20 -J GTExFULL