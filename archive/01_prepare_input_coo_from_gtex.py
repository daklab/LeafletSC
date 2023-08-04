import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

# load the clustered data /gpfs/commons/groups/knowles_lab/Karin/data/GTEx/clustered_junctions.h5
clusts = pd.read_hdf("/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/clustered_junctions.h5", key='df') # these have start-1 coordinates compared to original GTEx matrix

# make Name column to match GTEx file by first need to add "chr" before Chromosome column and subtract 1 from Start column 
clusts["Name"] = "chr" + clusts["Chromosome"].astype(str) + "_" + (clusts["Start"]+1).astype(str) + "_" + clusts["End"].astype(str)

# Remove singleton clusters where Count == 1
clusts = clusts[clusts["Count"] > 1]
print(len(clusts.Name.unique()))

# load the GTEx junctions file
gtex_juncs = '/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'
start_col = 2

melted_dfs = []

with open(gtex_juncs) as f:

    # Skip the first two rows
    next(f)
    next(f)

    header = f.readline().strip().split("\t")
    
    for i, line in enumerate(f):
        if i % 100 == 0:
            print(f"Processing line {i}")

        # Split the line by tabs to create a list of values
        row = line.strip().split("\t")
        df = pd.DataFrame([row])
        df.columns = header

        # Check if junction_id 'Name' is in the clustered junctions
        if df["Name"].isin(clusts.Name).any():
            print(df["Name"])
            # Melt the DataFrame so that the values become a column
            df = df.melt(id_vars=['Name', 'Description'], var_name='SAMPID', value_name='junc_count')
            # Filter out the rows with zero values.
            df['junc_count'] = df['junc_count'].astype(int)
            df = df[df['junc_count'] > 0]
            melted_dfs.append(df)

result_df = pd.concat(melted_dfs)

# save file as h5 file
result_df.to_hdf('/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/gtex_junc_counts_processed.h5', key='df', mode='w')

