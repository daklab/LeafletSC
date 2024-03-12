import pandas as pd
import numpy as np
import argparse
import pyranges as pr
from gtfparse import read_gtf #initially tested with version 1.3.0)
from tqdm import tqdm
import time
import warnings
import glob 
import time
import gzip
from pathlib import Path
import concurrent.futures
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore", category=FutureWarning, module="pyranges")

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, \
                                 one file per line and no header')

parser.add_argument('--junc_files', dest='junc_files',
                    help='path that has all junction files along with counts in single cells or bulk samples, \
                    make sure path ends in "/" Can also be a comma seperated list of paths. If you have a complex folder structure, \
                        provide the most root folder that contains all the junction files. The script will recursively search for junction files with the suffix provided in the next argument.')

parser.add_argument('--sequencing_type', dest='sequencing_type',
                    default='single_cell',
                    help='were the junction obtained using data from single cell or bulk sequencing? \
                        options are "single_cell" or "bulk". Default is "single_cell"')

parser.add_argument('--gtf_file', dest='gtf_file', 
                    default = None,
                    help='a path to a gtf file to annotate high confidence junctions, \
                    ideally from long read sequencing, if not provided, then the script will not \
                        annotate junctions based on gtf file')

parser.add_argument('--output_file', dest='output_file', 
                    default='intron_clusters.txt',
                    help='name of the output file to save intron cluster file to')

parser.add_argument('--junc_bed_file', dest='junc_bed_file', 
                    default='juncs.bed',
                    help='name of the output bed file to save final list of junction coordinates to')

parser.add_argument('--threshold_inc', dest='threshold_inc',
                    default=0.005,
                    help='threshold to use for removing clusters that have junctions with low read counts \
                        (proportion of reads relative to intron cluster) at either end, default is 0.01')

parser.add_argument('--min_intron_length', dest='min_intron_length',
                    default=50,
                    help='minimum intron length to consider, default is 50')

parser.add_argument('--max_intron_length', dest='max_intron_length',
                    default=500000,
                    help='maximum intron length to consider, default is 500000')

parser.add_argument('--min_junc_reads', dest='min_junc_reads',
                    default=1,
                    help='minimum number of reads to consider a junction, default is 1')

parser.add_argument('--keep_singletons', dest='keep_singletons', 
                    default=False,
                    help='Indicate whether you would like to keep "clusters" composed of just one junction.\
                          Default is False which means do not keep singletons')

parser.add_argument('--junc_suffix', dest='junc_suffix', #set default param to *.junc, 
                    default='*.juncs', 
                    help='suffix of junction files')

parser.add_argument('--min_num_cells_wjunc', dest='min_num_cells_wjunc',
                    default=1,
                    help='minimum number of cells that have a junction to consider it, default is 1')

parser.add_argument('--run_notebook', dest='run_notebook',
                    default=False,
                    help='Indicate whether you would like to run the script in a notebook and return the table in session.\
                          Default is False')

args = parser.parse_args(args=[])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_gtf(gtf_file): #make this into a seperate script that processes the gtf file into gr object that can be used in the main scriptas input 
    """
    Process the GTF file into a pyranges object.

    Parameters:
    - gtf_file (str): Path to the GTF file.

    Returns:
    - gtf_exons_gr (pyranges.GenomicRanges): Processed pyranges object.
    """

    print("The gtf file you provided is " + gtf_file)
    print("Reading the gtf may take a while depending on the size of your gtf file")

    # calculate how long it takes to read gtf_file and report it 
    start_time = time.time()
    #[1] extract all exons from gtf file provided 
    gtf = read_gtf(gtf_file, result_type="pandas") #to reduce the speed of this, can just get rows with exon in the feature column (preprocess this before running package)? check if really necessary
    end_time = time.time()

    print("Reading gtf file took " + str(round((end_time-start_time), 2)) + " seconds")
    # assert that gtf is a non empty dataframe otherwise return an error
    if gtf.empty or type(gtf) != pd.DataFrame:
        raise ValueError("The gtf file provided is empty or not a pandas DataFrame. Please provide a valid gtf file and ensure you have the \
                         latest version of gtfparse installed by running 'pip install gtfparse --upgrade'")
    
    # Convert the seqname column to a string in gtf 
    gtf["seqname"] = gtf["seqname"].astype(str)

    # Make a copy of the DataFrame
    gtf_exons = gtf[(gtf["feature"] == "exon")].copy()

    if gtf_exons['seqname'].str.contains('chr').any():
        gtf_exons.loc[gtf_exons['seqname'].str.contains('chr'), 'seqname'] = gtf_exons['seqname'].map(lambda x: x.lstrip('chr').rstrip('chr'))

    if not set(['seqname', 'start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'transcript_id', 'exon_id']).issubset(gtf_exons.columns):
        # print the columns that the file is missing
        missing_cols = set(['seqname', 'start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'transcript_id', 'exon_id']).difference(gtf_exons.columns)
        print("Your gtf file is missing the following columns: " + str(missing_cols))

        # if the missing column is just exon_id, we can generate it
        if "exon_id" in missing_cols:
            # add exon_id to gtf_exons
            print("Adding exon_id column to gtf file")
            gtf_exons.loc[:, "exon_id"] = gtf_exons["transcript_id"] + "_" + gtf_exons["start"].astype(str) + "_" + gtf_exons["end"].astype(str)
        else:
            pass

    # Convert the DataFrame to a PyRanges object
    gtf_exons_gr = pr.from_dict({"Chromosome": gtf_exons["seqname"], "Start": gtf_exons["start"], "End": gtf_exons["end"], "Strand": gtf_exons["strand"], "gene_id": gtf_exons["gene_id"], "gene_name": gtf_exons["gene_name"], "transcript_id": gtf_exons["transcript_id"], "exon_id": gtf_exons["exon_id"]})

    # Remove rows where exon start and end are the same or when gene_name is empty
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.Start == gtf_exons_gr.End)]
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.gene_name == "")]

    # When do I need to do this? depends on gtf file used? base 0 or 1? probably need this to be a parameter 
    gtf_exons_gr.Start = gtf_exons_gr.Start-1

    # Drop duplicated positions on same strand 
    gtf_exons_gr = gtf_exons_gr.drop_duplicate_positions(strand=True) # Why are so many gone after this? 

    # Print the number of unique exons, transcript ids, and gene ids
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("The number of unique exons is " + str(len(gtf_exons_gr.exon_id.unique())))
    print("The number of unique transcript ids is " + str(len(gtf_exons_gr.transcript_id.unique())))
    print("The number of unique gene ids is " + str(len(gtf_exons_gr.gene_id.unique())))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return(gtf_exons_gr)

def preprocess_data(dataset):
    """
    Preprocess the junction data.

    Parameters:
    - dataset (pd.DataFrame): Input DataFrame.

    Returns:
    - juncs_dat_summ (pd.DataFrame): Preprocessed DataFrame.
    """
        
    # Assuming 'score' is a column in 'dataset' that you want to summarize
    juncs_dat_summ = dataset.groupby(["chrom", "chromStart", "chromEnd", "junction_id"], as_index=False).score.sum()
    juncs_dat_summ = juncs_dat_summ.merge(
        juncs_dat_summ.groupby(['chromStart'])['score'].sum().reset_index().rename(columns={'score': 'total_5SS_counts'}),
        on='chromStart'
    ).merge(
        juncs_dat_summ.groupby(['chromEnd'])['score'].sum().reset_index().rename(columns={'score': 'total_3SS_counts'}),
        on='chromEnd'
    )
    juncs_dat_summ['5SS_usage'] = juncs_dat_summ['score'] / juncs_dat_summ['total_5SS_counts']
    juncs_dat_summ['3SS_usage'] = juncs_dat_summ['score'] / juncs_dat_summ['total_3SS_counts']
    return juncs_dat_summ

def refine_cluster(cluster, clusters_df, preprocessed_data):
    """
    Refine a single cluster.

    Parameters:
    - cluster (int): Cluster ID.
    - clusters_df (pd.DataFrame): DataFrame of clusters.
    - preprocessed_data (pd.DataFrame): Preprocessed data.

    Returns:
    - List: Cluster details.
    """
    clust_dat = clusters_df[clusters_df.Cluster == cluster]
    juncs_dat_all = preprocessed_data[preprocessed_data.junction_id.isin(clust_dat.junction_id)]
    ss_score = juncs_dat_all[["5SS_usage", "3SS_usage"]].min().min()
    junc = juncs_dat_all[(juncs_dat_all["5SS_usage"] == ss_score) | (juncs_dat_all["3SS_usage"] == ss_score)].junction_id.values[0]
    return [cluster, junc, ss_score]

def refine_clusters(clusters, clusters_df, dataset):
    """
    Refine multiple clusters.

    Parameters:
    - clusters (list): List of cluster IDs.
    - clusters_df (pd.DataFrame): DataFrame of clusters.
    - dataset (pd.DataFrame): Input DataFrame.

    Returns:
    - list: List of refined clusters.
    """
        
    preprocessed_data = preprocess_data(dataset)
    all_juncs_scores = []
    # start time 
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: refine_cluster(x, clusters_df, preprocessed_data), clusters))
    for result in results:
        all_juncs_scores.append(result)

    # end time
    end_time = time.time()
    print("Refining clusters took " + str(round((end_time-start_time), 2)) + " seconds")
    return all_juncs_scores

def filter_junctions_by_shared_splice_sites(df):
    """
    Filter junctions by shared splice sites.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    # Function to apply to each group (cluster)
    def filter_group(group):
        # Find duplicated start and end positions within the group
        duplicated_starts = group['Start'].duplicated(keep=False)
        duplicated_ends = group['End'].duplicated(keep=False)
        
        # Keep rows where either start or end position is duplicated (this results in at least two junctions in every cluster)
        return group[duplicated_starts | duplicated_ends]
    
    # Group by 'Cluster' and apply the filtering function
    filtered_df = df.groupby('Cluster').apply(filter_group).reset_index(drop=True)
    return filtered_df.Cluster.unique()

def read_junction_files(junc_files, junc_suffix):
    """
    Read junction files.

    Parameters:
    - junc_files (list): List of paths to junction files.
    - junc_suffix (str): Suffix of junction files.

    Returns:
    - pd.DataFrame: Concatenated DataFrame of junction files.
    """
    all_juncs_list = []

    for junc_path in junc_files:
        junc_path = Path(junc_path)
        print(f"Reading in junction files from {junc_path}")

        junc_files_in_path = list(junc_path.rglob(junc_suffix))
        if not junc_files_in_path:
            print(f"No junction files found in {junc_path} with suffix {junc_suffix}")
            continue

        print(f"The number of junction files to be processed is {len(junc_files_in_path)}")

        files_not_read = []

        for junc_file in tqdm(junc_files_in_path):
            try:
                juncs = pd.read_csv(junc_file, sep="\t", header=None)
                juncs['file_name'] = junc_file  # Add the file name as a new column
                juncs['cell_type'] = junc_file
                all_juncs_list.append(juncs)  # Append the DataFrame to the list
            except Exception as e:
                print(f"Could not read in {junc_file}: {e}")
                files_not_read.append(junc_file)

    if len(files_not_read) > 0:
        print("The total number of files that could not be read is " + str(len(files_not_read)) + " as these had no junctions")

    # Concatenate all DataFrames into a single DataFrame
    all_juncs = pd.concat(all_juncs_list, ignore_index=True) if all_juncs_list else pd.DataFrame()

    return all_juncs

def clean_up_juncs(all_juncs, col_names, min_intron, max_intron):
    
    # Apply column names to the DataFrame
    all_juncs.columns = col_names
    
    # Split 'blockSizes' into two new columns and convert them to integers (this step takes a while)
    all_juncs[['block_add_start', 'block_subtract_end']] = all_juncs["blockSizes"].str.split(',', expand=True).astype(int)

    # Adjust 'chromStart' and 'chromEnd' based on 'block_add_start' and 'block_subtract_end'
    all_juncs["chromStart"] += all_juncs['block_add_start']
    all_juncs["chromEnd"] -= all_juncs['block_subtract_end']

    # Calculate 'intron_length' and filter based on 'min_intron' and 'max_intron'
    all_juncs["intron_length"] = all_juncs["chromEnd"] - all_juncs["chromStart"]
    mask = (all_juncs["intron_length"] >= min_intron) & (all_juncs["intron_length"] <= max_intron)
    all_juncs = all_juncs[mask]

    # Filter for 'chrom' column to handle "chr" prefix
    all_juncs = all_juncs.copy()

    # New filter for 'chrom' column to handle "chr" prefix, using .loc for safe in-place modification
    standard_chromosomes_pattern = r'^(?:chr)?(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)$'
    all_juncs = all_juncs[all_juncs['chrom'].str.match(standard_chromosomes_pattern)]

    print("Cleaning up 'chrom' column")
    # Remove "chr" prefix from 'chrom' column
    all_juncs['chrom'] = all_juncs['chrom'].str.replace(r'^chr', '', regex=True)
    
    # Add 'junction_id' column
    all_juncs['junction_id'] = all_juncs['chrom'] + '_' + all_juncs['chromStart'].astype(str) + '_' + all_juncs['chromEnd'].astype(str)
    
    # Get total score for each junction and merge with all_juncs with new column "total_counts"
    all_juncs = all_juncs.groupby('junction_id').agg({'score': 'sum'}).reset_index().merge(all_juncs, on='junction_id', how='left')

    # rename score_x and score_y to total_junc_counts and score 
    all_juncs.rename(columns={'score_x': 'counts_total', 'score_y': 'score'}, inplace=True)

    return(all_juncs)

def mapping_juncs_exons(juncs_gr, gtf_exons_gr, singletons):
    print("Annotating junctions with known exons based on input gtf file")
    
    # for each junction, the start of the junction should equal end of exons and end of junction should equal start of exon 
    juncs_gr = juncs_gr.k_nearest(gtf_exons_gr, strandedness = "same", ties="different", k=2, overlap=False)
    # ensure distance parameter is still 1 
    juncs_gr = juncs_gr[abs(juncs_gr.Distance) == 1]

    # group juncs_gr by gene_id and ensure that each junction has Start and End aligning with at least one End_b and Start_b respectively
    grouped_gr = juncs_gr.df.groupby("gene_id")
    juncs_keep = []
    for name, group in grouped_gr:
        group = group[(group.Start.isin(group.End_b)) & (group.End.isin(group.Start_b))]
        # save junctions that are found here after filtering for matching start and end positions
        juncs_keep.append(group.junction_id.unique())

    # flatten the list of lists
    juncs_keep = [item for sublist in juncs_keep for item in sublist]
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(juncs_keep)]
    
    print("The number of junctions after assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())))
    if len(juncs_gr.junction_id.unique()) < 5000:
        print("There are less than 5000 junctions after assessing distance to exons. Please check your gtf file and ensure that it is in the correct format (start and end positions are not off by 1).", flush=True)
    
    print("Clustering intron splicing events by gene_id")
    juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id', 'gene_id']].drop_duplicate_positions()
    clusters = juncs_coords_unique.cluster(by="gene_id", slack=-1, count=True)
    print("The number of clusters after clustering by gene_id is " + str(clusters.Cluster.max())) 

    if singletons == False:
        # remove singletons 
        clusters = clusters[clusters.Cluster > 1]
        print("The number of clusters after removing singletons is " + str(clusters.Cluster.max()))
        # update juncs_gr to only include junctions that are part of clusters
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
        # update juncs_coords_unique to only include junctions that are part of clusters
        juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
        print("The number of junctions after removing singletons is " + str(len(juncs_coords_unique.junction_id.unique())))
        return juncs_gr, juncs_coords_unique, clusters
    else:
        return juncs_gr, juncs_coords_unique, clusters

def visualize_junctions(dat, junc_id):
    # Filter data for the specific junction ID
    dat = dat[dat.Cluster == dat[dat.junction_id == junc_id].Cluster.values[0]]

    # Get junctions
    juncs = dat[["chromStart", "chromEnd", "strand"]]
    juncs = juncs.drop_duplicates()

    # Sort junctions based on strand
    if juncs.strand.values[0] == "+":
        juncs = juncs.sort_values("chromStart")
    else:
        juncs = juncs.sort_values("chromEnd", ascending=False)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, len(juncs) * 0.5))

    # Plot junctions as lines
    for i, (_, junc) in enumerate(juncs.iterrows()):
        ax.plot([junc["chromStart"], junc["chromEnd"]], [i, i], color="red")

    # Set labels and title
    ax.set_xlabel("Genomic Position")
    ax.set_yticks(list(range(len(juncs))))
    ax.set_title(f"Visualization of Junctions in Cluster {dat.Cluster.values[0]} in the Gene {dat.gene_id.values[0]}")
    print("The junction of interest is " + junc_id)
    plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#        Run analysis and obtain intron clusters
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main(junc_files, gtf_file, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, junc_suffix, min_num_cells_wjunc, run_notebook):
    
    #1. Check format of junc_files and convert to list if necessary
    # Can either be a list of folders with junction files or a single folder with junction files

    if "," in junc_files:
        junc_files = junc_files.split(",")
    else:
        # if junc_files is a single file, then convert it to a list
        junc_files = [junc_files]

    #2. run read_junction_files function to read in all junction files
    all_juncs = read_junction_files(junc_files, junc_suffix)

    #3. If gtf_file is not empty, read it in and process it
    if gtf_file is not None:
        gtf_exons_gr = process_gtf(gtf_file)
        print("Done extracting exons from gtf file")
    else:
        pass

    #4. Convert parameters to integers outside the loop
    min_intron = int(min_intron)
    max_intron = int(max_intron)
    min_junc_reads = int(min_junc_reads)
    min_num_cells_wjunc = int(min_num_cells_wjunc)

    #5. Define column names based on sequencing type
    col_names = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", 
             "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    if sequencing_type == "single_cell":
        col_names += ["num_cells_wjunc", "cell_readcounts"]
    col_names += ["file_name", "cell_type"]
    print(f"Loading files obtained from {sequencing_type} sequencing")
    
    # 6. Clean up junctions and filter for intron length
    all_juncs = clean_up_juncs(all_juncs, col_names, min_intron, max_intron)

    # 7. Make gr object from ALL junctions across all cell types 
    print("Making gr object from all junctions across all cell types")
    juncs_gr = pr.from_dict({"Chromosome": all_juncs["chrom"], "Start": all_juncs["chromStart"], "End": all_juncs["chromEnd"], "Strand": all_juncs["strand"], "Cell": all_juncs["cell_type"], "junction_id": all_juncs["junction_id"], "counts_total": all_juncs["counts_total"]})
    # Unique set of junction coordinates across all samples (or cells) 
    juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id", "counts_total"]].drop_duplicate_positions()

    # if min_junc_reads is not none then remove junctions with less than min_junc_reads
    if min_junc_reads is not None:
        juncs_gr = juncs_gr[juncs_gr.counts_total > min_junc_reads]

    #keep only junctions that could be actually related to isoforms that we expect in our cells (via gtf file provided)
    print("The number of junctions prior to assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())))

    # 8. Annotate junctions based on gtf file (if gtf_file is not empty)
    if gtf_file is not None:
        juncs_gr, juncs_coords_unique, clusters = mapping_juncs_exons(juncs_gr, gtf_exons_gr, singleton) 
    else:
        print("Clustering intron splicing events by coordinates")
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id', 'counts_total']].drop_duplicate_positions()
        clusters = juncs_coords_unique.cluster(slack=-1, count=True)
        print("The number of clusters after clustering by coordinates is " + str(len(clusters.Cluster.unique())))
        if singleton == False:
            clusters = clusters[clusters.Count > 1]
            # update juncs_gr to include only clusters that are in clusters
            juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
            juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
            print("The number of clusters after removing singletons is " + str(len(clusters.Cluster.unique())))

    # 9. Now for each cluster we want to check that each junction shares a splice site with at least one other junction in the cluster
    clusts_keep = filter_junctions_by_shared_splice_sites(clusters.df)
    # update clusters, juncs_gr, and juncs_coords_unique to only include clusters
    clusters = clusters[clusters.Cluster.isin(clusts_keep)]
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
    juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
    print("The number of clusters after filtering for shared splice sites is " + str(len(clusters.Cluster.unique())))
    print("The number of junctions after filtering for shared splice sites is " + str(len(juncs_coords_unique.junction_id.unique())))

    # 10. Update our all_juncs file to only include junctions that are part of clusters
    all_juncs = all_juncs[all_juncs.junction_id.isin(juncs_coords_unique.junction_id)]

    # 11. Refine intron clusters based on splice sites found in them -> remove low confidence junctions basically a filter to see which junctions to keep
    all_juncs_scores = refine_clusters(clusters.Cluster.unique(), clusters, all_juncs)
    junc_scores_all = pd.DataFrame(all_juncs_scores, columns=["Cluster", "junction_id", "junction_score"])
    junc_scores_all = junc_scores_all[junc_scores_all.junction_score < threshold_inc]
    # remove junctions that are in junc_scores_all from juncs_gr, clusters, all_juncs and juncs_coords_unique
    juncs_gr = juncs_gr[~juncs_gr.junction_id.isin(junc_scores_all.junction_id)]
    clusters = clusters[~clusters.junction_id.isin(junc_scores_all.junction_id)]
    all_juncs = all_juncs[~all_juncs.junction_id.isin(junc_scores_all.junction_id)]
    juncs_coords_unique = juncs_coords_unique[~juncs_coords_unique.junction_id.isin(junc_scores_all.junction_id)]
    print("The number of clusters after removing low confidence junctions is " + str(len(clusters.Cluster.unique())))

    # 12. given junctions that remain, see if need to recluster introns (low confidence junctions removed)
    print("Reclustering intron splicing events after low confidence junction removal")
    # check if there are any duplicate entried in pyranges object 
    juncs_gr = juncs_gr.drop_duplicate_positions()
    # drop original cluster column and add new one
    clusters = juncs_gr.cluster(by="gene_id", slack=-1, count=True)
    
    # 13. remove singletons if there are new ones 
    if((singleton) == False):
        clusters = clusters[clusters.Count > 1]
    
    # update juncs_gr to only include junctions that are part of clusters and update juncs_coords_unique to only include junctions that are part of clusters
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
    juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
    # update all_juncs 
    all_juncs = all_juncs[all_juncs.junction_id.isin(juncs_coords_unique.junction_id)]
    print("The number of clusters after removing singletons is " + str(len(clusters.Cluster.unique())))

    # 14. After re-clustering above, need to confirm that junctions still share splice sites  
    print("Confirming that junctions in each cluster share splice sites")
    clusts_keep = filter_junctions_by_shared_splice_sites(clusters.df)
    # update clusters, juncs_gr, and juncs_coords_unique to only include clusters
    clusters = clusters[clusters.Cluster.isin(clusts_keep)]
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
    juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
    all_juncs = all_juncs[all_juncs.junction_id.isin(juncs_coords_unique.junction_id)]
    print("The number of clusters after filtering for shared splice sites is " + str(len(clusters.Cluster.unique())))

    if gtf_file is not None:
        clusts_unique = clusters.df[["Cluster", "junction_id", "gene_id", "Count"]].drop_duplicates()
    else:
        clusts_unique = clusters.df[["Cluster", "junction_id", "Count"]].drop_duplicates()
    
     # merge juncs_gr with corresponding cluster id
    all_juncs_df = all_juncs.merge(clusts_unique, how="left")

    # get final list of junction coordinates and save to bed file for visualization
    juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id"]]
    juncs_gr = juncs_gr.drop_duplicate_positions()
    juncs_gr.to_bed(junc_bed_file, chain=True) #add option to add prefix to file name

    print("The number of clusters to be finally evaluated is " + str(len(all_juncs_df.Cluster.unique()))) 
    print("The number of junctions to be finally evaluated is " + str(len(all_juncs_df.junction_id.unique())))

    # assert unique number of junctions and clusters in all_juncs_df and clusters_df is the same 
    assert len(all_juncs_df.junction_id.unique()) == len(clusters.df.junction_id.unique())
    assert len(all_juncs_df.Cluster.unique()) == len(clusters.df.Cluster.unique()) 
    
    # 12. Save the final list of intron clusters to a file
    # to the output file add the parameters that was used so user can easily tell how they generated this file 
    date = time.strftime("%Y%m%d")
    output = output_file + "_" + str(min_intron) + "_" + str(max_intron) + "_" + str(min_junc_reads) + "_" + date + "_" + str(sequencing_type) 
    with gzip.open(output + '.gz', mode='wt', encoding='utf-8') as f:
        all_juncs_df.to_csv(f, index=False, sep="}")
    print("You can find the output file here: " + output + ".gz")
    print("Finished obtaining intron cluster files!")

    # also return the final list of intron clusters if running in notebook 
    if run_notebook:
        return all_juncs_df

if __name__ == '__main__':
    
    gtf_file=args.gtf_file
    junc_files=args.junc_files
    output_file=args.output_file
    sequencing_type=args.sequencing_type
    junc_bed_file=args.junc_bed_file
    threshold_inc = args.threshold_inc
    min_intron=args.min_intron_length
    max_intron=args.max_intron_length
    min_junc_reads=args.min_junc_reads
    junc_suffix=args.junc_suffix #'*.juncswbarcodes'
    min_num_cells_wjunc=args.min_num_cells_wjunc
    # ensure singleton is boolean
    if args.keep_singletons == "True":
        singleton=True
    else:
        singleton=False
    run_notebook = args.run_notebook
    
    # print out all user defined arguments that were chosen 
    print("The following arguments were chosen:")
    print("gtf_file: " + str(gtf_file))
    print("junc_files: " + junc_files)
    print("output_file: " + output_file)
    print("sequencing_type: " + sequencing_type)
    print("junc_bed_file: " + junc_bed_file)
    print("threshold_inc: " + str(threshold_inc))
    print("min_intron: " + str(min_intron))
    print("max_intron: " + str(max_intron))
    print("min_junc_reads: " + str(min_junc_reads))
    print("junc_suffix: " + junc_suffix)
    print("min_num_cells_wjunc: " + str(min_num_cells_wjunc))
    print("singleton: " + str(singleton))

    main(junc_files, gtf_file, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, junc_suffix, min_num_cells_wjunc, run_notebook)
