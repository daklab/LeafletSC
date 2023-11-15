import pandas as pd
import numpy as np
import argparse
import glob
import os
import pyranges as pr
from gtfparse import read_gtf #initially tested with version 1.3.0 
from tqdm import tqdm
import gzip
import time
import concurrent.futures
import warnings
import sys
import builtins
import concurrent.futures
import datetime
from numba import njit

warnings.filterwarnings("ignore", category=FutureWarning, module="pyranges")

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--junc_files', dest='junc_files',
                    help='path that has all junction files along with counts in single cells or bulk samples, make sure path ends in "/" Can also be a comma seperated list of paths.')

parser.add_argument('--sequencing_type', dest='sequencing_type',
                    help='were the junction obtained using data from single cell or bulk sequencing? options are "single_cell" or "bulk". Note if sequencing was done using smart-seq2, then use "bulk" option')

parser.add_argument('--gtf_file', dest='gtf_file',
                    help='a path to a gtf file to annotate high confidence junctions, ideally from long read sequencing')

parser.add_argument('--output_file', dest='output_file', 
                    default='intron_clusters.txt',
                    help='name of the output file to save intron cluster file to')

parser.add_argument('--junc_bed_file', dest='junc_bed_file', 
                    default='juncs.bed',
                    help='name of the output bed file to save final list of junction coordinates to')

parser.add_argument('--threshold_inc', dest='threshold_inc',
                    default=0.005,
                    help='threshold to use for removing clusters that have junctions with low read counts at either end, default is 0.01')

parser.add_argument('--min_intron_length', dest='min_intron_length',
                    default=50,
                    help='minimum intron length to consider, default is 50')

parser.add_argument('--max_intron_length', dest='max_intron_length',
                    default=500000,
                    help='maximum intron length to consider, default is 500000')

parser.add_argument('--min_junc_reads', dest='min_junc_reads',
                    default=5,
                    help='minimum number of reads to consider a junction, default is 5')

parser.add_argument('--keep_singletons', dest='keep_singletons', 
                    default=False,
                    help='Indicate whether you would like to keep "clusters" composed of just one junction. Default is False which means do not keep singletons')

parser.add_argument('--junc_suffix', dest='junc_suffix', #set default param to *.junc, 
                    default='*.juncs', 
                    help='suffix of junction files')

parser.add_argument('--min_num_cells_wjunc', dest='min_num_cells_wjunc',
                    default=1,
                    help='minimum number of cells that have a junction to consider it, default is 1')

parser.add_argument('--filter_low_juncratios_inclust', dest='filter_low_juncratios_inclust',
                    default="no",
                    help='yes if want to remove lowly used junctions in clusters, default is no')

parser.add_argument('--strict_filter', dest='strict_filter',
                    default=True,
                    help='default is True, this means that only clusters with less junctions that the mean junction count per cluster is included. This is meant to remove very complex splicing events that might be hard to make sense of in the single cell context especially.')

args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_gtf(gtf_file): #make this into a seperate script that processes the gtf file into gr object that can be used in the main scriptas input 

    print("The gtf file you provided is " + gtf_file)
    print("This step may take a while depending on the size of your gtf file")

    # calculate how long it takes to read gtf_file and report it 
    start_time = time.time()
    #[1] extract all exons from gtf file provided 
    gtf = read_gtf(gtf_file) #to reduce the speed of this, can just get rows with exon in the feature column (preprocess this before running package)? check if really necessary
    end_time = time.time()
    print("Reading gtf file took " + str(round((end_time-start_time), 2)) + " seconds")

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
    
    gtf_exons_gr = pr.from_dict({"Chromosome": gtf_exons["seqname"], "Start": gtf_exons["start"], "End": gtf_exons["end"], "Strand": gtf_exons["strand"], "gene_id": gtf_exons["gene_id"], "gene_name": gtf_exons["gene_name"], "transcript_id": gtf_exons["transcript_id"], "exon_id": gtf_exons["exon_id"]})
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.Start == gtf_exons_gr.End)]
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.gene_name == "")]

    # When do I need to do this? depends on gtf file used? base 0 or 1? probably need this to be a parameter 
    gtf_exons_gr.Start = gtf_exons_gr.Start-1

    # Drop duplicated positions on same strand 
    gtf_exons_gr = gtf_exons_gr.drop_duplicate_positions(strand=True) # Why are so many gone after this? 

    # Print the number of unique exons, transcript ids, and gene ids
    print("The number of unique exons is " + str(len(gtf_exons_gr.exon_id.unique())))
    print("The number of unique transcript ids is " + str(len(gtf_exons_gr.transcript_id.unique())))
    print("The number of unique gene ids is " + str(len(gtf_exons_gr.gene_id.unique())))
    return(gtf_exons_gr)

def read_file(filename, sequencing_type, col_names, junc_suff, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads):
    try:
        if not os.path.exists(filename):
            print(f"File '{filename}' does not exist.")
            return None
        
        juncs = pd.read_csv(filename, sep="\t", header=None, low_memory=False)    
  
        # Check if the DataFrame is not empty
        if not juncs.empty:
            juncs = juncs.copy()
            juncs = juncs.set_axis(col_names, axis=1, copy=False)

            juncs[['block_add_start', 'block_subtract_end']] = juncs["blockSizes"].str.split(',', expand=True)
            juncs[['block_add_start', 'block_subtract_end']] = juncs[['block_add_start', 'block_subtract_end']].astype(int)

            juncs["chromStart"] = juncs["chromStart"] + juncs['block_add_start']
            juncs["chromEnd"] = juncs["chromEnd"] - juncs['block_subtract_end']

            juncs["intron_length"] = juncs["chromEnd"] - juncs["chromStart"]

            min_intron = int(min_intron)
            max_intron = int(max_intron)

            mask = (juncs["intron_length"] >= min_intron) & (juncs["intron_length"] <= max_intron)
            juncs = juncs[mask]

            filename = filename.split("/")[-1]
            cell_type = ""

            if sequencing_type == "single_cell":
                mask = juncs["num_cells_wjunc"] >= min_num_cells_wjunc
                juncs = juncs[mask]

                filename = filename.split(junc_suff)[0]
                if 'batch' in filename:
                    cell_type = filename.split(".batch")[0]
                elif 'pseudobulk' in filename:
                    cell_type = filename.split("_pseudobulk")[0]
                else:
                    cell_type = filename+'_cell'
            elif sequencing_type == "bulk":
                cell_type = filename.split('.junc')[0]

            # if not juncs.shape[0] == 0:
            if juncs.shape[0] > 0:
                juncs['cell_type'] = cell_type
                # should do this filter once all cells are read below during load_files...
                mask = juncs["score"] >= min_junc_reads
                juncs = juncs[mask]
                return juncs
            else:
                print(juncs)
                print(f"No valid data in file '{filename}'.")
    except pd.errors.EmptyDataError:
        # Handle the specific case where the file is empty (skip, print a message, etc.)
        pass
        
def load_files(filenames, sequencing_type, junc_suffix, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads):
    start_time = time.time()
    junc_suff = junc_suffix.split("*")[1]

    # Convert parameters to integers outside the loop
    min_intron = int(min_intron)
    max_intron = int(max_intron)
    min_junc_reads = int(min_junc_reads)

    # Use set for faster membership check
    col_names = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]

    if sequencing_type == "single_cell":
        min_num_cells_wjunc = int(min_num_cells_wjunc)
        col_names.append("num_cells_wjunc")
        col_names.append("cell_readcounts")
    
    print("Loading files obtained from " + sequencing_type + " sequencing")  

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: read_file(x, sequencing_type, col_names, junc_suff, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads), filenames))

    all_juncs = pd.concat(results)
    print("Reading all the junctions took " + str(round((time.time()-start_time), 2)) + " seconds")
    return all_juncs

def preprocess_data(dataset):
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
    clust_dat = clusters_df[clusters_df.Cluster == cluster]
    juncs_dat_all = preprocessed_data[preprocessed_data.junction_id.isin(clust_dat.junction_id)]
    ss_score = juncs_dat_all[["5SS_usage", "3SS_usage"]].min().min()
    junc = juncs_dat_all[(juncs_dat_all["5SS_usage"] == ss_score) | (juncs_dat_all["3SS_usage"] == ss_score)].junction_id.values[0]
    return [cluster, junc, ss_score]

def refine_clusters(clusters, clusters_df, dataset):
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

def filter_junctions_in_cluster(group_df):
    # Find the rows that share the same start or end position
    # Account for the fact that duplicates are possible if maps to multiple transcript_ids
    matches = group_df[["Start", "End", "junction_id"]].drop_duplicates()
    # Identify rows that have a duplicated start or end value
    duplicated_starts = matches['Start'].duplicated(keep=False)
    duplicated_ends = matches['End'].duplicated(keep=False)
    duplicated_df = matches[duplicated_starts | duplicated_ends]
    return(duplicated_df.junction_id.values)

def main(junc_files, gtf_file, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, strict_filter, junc_suffix, min_num_cells_wjunc, filter_low_juncratios_inclust):
    """
    Intersect junction coordinates with up/downstream exons in the canonical setting based on gtf file 
    and then obtain intron clusters using overlapping junctions.
    """

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #        Run analysis and obtain intron clusters
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++

    junc_files = junc_files.split(',')
    print(junc_files)

    # Redirect standard output to the log file
    sys.stdout = log_file

    #[2] collect all junctions across all cell types 
    start_time = time.time()
    all_juncs = []

    # make sure that junc_files is a list 
    if isinstance(junc_files, str):
        junc_files = [junc_files]

    for junc_path in junc_files:
        junc_files_in_path = glob.glob(os.path.join(junc_path, junc_suffix))
        print("The number of regtools junction files to be processed is " + str(len(junc_files_in_path)))
        # Call load_files for the current path
        juncs = load_files(junc_files_in_path, sequencing_type, junc_suffix, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads)
        all_juncs.append(juncs)   

    end_time = time.time()
    print("Reading all junction files took " + str(round((end_time-start_time), 2)) + " seconds", flush=True)
    print("Done extracting junctions!", flush=True)
   
    # combine all_juncs into one dataframe 
    if len(all_juncs) > 1:
        print("More than one Donor with cells of this organ")
        juncs = pd.concat(all_juncs)
    else:
        print("Just one Donor with cells of this organ")
        juncs = all_juncs[0]    
    juncs = juncs.copy()

    # if "chr" appears in the chrom column 
    if juncs['chrom'].str.contains("chr").any():
        juncs = juncs[juncs['chrom'].str.contains("chr")]
        juncs['chrom'] = juncs['chrom'].map(lambda x: x.lstrip('chr').rstrip('chr'))
    
    # add unique value to each junction name (going to appear multiple times otherwise once for each sample)
    juncs["name"] = juncs["name"] + juncs.groupby("name").cumcount().astype(str)
    juncs['junction_id'] = juncs['chrom'] + '_' + juncs['chromStart'].astype(str) + '_' + juncs['chromEnd'].astype(str)

    #make gr object from ALL junctions across all cell types  
    juncs_gr = pr.from_dict({"Chromosome": juncs["chrom"], "Start": juncs["chromStart"], "End": juncs["chromEnd"], "Strand": juncs["strand"], "Cell": juncs["cell_type"], "junction_id": juncs["junction_id"], "counts_total": juncs["score"]})

    # we don't actually care about cell types anymore, we just want to obtain a list of junctions to include 
    # in the final analysis and to group them into alternative splicing events
    juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id"]].drop_duplicate_positions()

    #keep only junctions that could be actually related to isoforms that we expect in our cells (via gtf file provided)
    print("The number of junctions prior to assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())), flush=True)

    #[1] load gtf coordinates into pyranges object 
    gtf_exons_gr = process_gtf(gtf_file)
    print("Done extracting exons from gtf file")

    #[3] annotate each junction with nearbygenes 
    print("Annotating junctions with known exons based on input gtf file", flush=True)
    juncs_gr = juncs_gr.k_nearest(gtf_exons_gr, strandedness = "same", ties="different", k=2, overlap=False)
    # ensure distance parameter is still 1 
    juncs_gr = juncs_gr[abs(juncs_gr.Distance) == 1]
    # for each junction, the start of the junction should equal end of exons and end of junction should equal start of exon 
    juncs_gr = juncs_gr[(juncs_gr.Start.isin(juncs_gr.End_b)) & (juncs_gr.End.isin(juncs_gr.Start_b))]
    juncs_gr = juncs_gr[juncs_gr.Start == juncs_gr.End_b]
    print("The number of junctions after assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())), flush=True) 
    if len(juncs_gr.junction_id.unique()) < 5000:
        print("There are less than 5000 junctions after assessing distance to exons. Please check your gtf file and ensure that it is in the correct format (start and end positions are not off by 1).", flush=True)
    print("Clustering intron splicing events by gene_id", flush=True)
    juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id', 'gene_id']].drop_duplicate_positions()
    clusters = juncs_coords_unique.cluster(by="gene_id", slack=-1, count=True)

    # filter intron singleton "clusters" (remove those with only one intron and those with only a single splicing site event (one SS))
    if((singleton) == False):
        print("Removing singleton clusters", flush=True)
        clusters = clusters[clusters.Count > 1]
        print("The number of junctions after removing singletons " + str(len(clusters.junction_id.unique())), flush=True)
    
    # check if any Clusters in pyranges object have more than one unique gene_id
    grouped = clusters.df.groupby('Cluster')['gene_id'].nunique().reset_index()
    print("Checking if any intron clusters have more than one gene_id")
    print(grouped[grouped['gene_id'] > 1]['Cluster'])

    if((strict_filter) == True):
        print("Removing clusters with more than mean number of junctions", flush=True)
        clusters = clusters[clusters.Count < clusters.Count.mean()]
        print("The number of junctions after strict filtering " + str(len(clusters.junction_id.unique())), flush=True)

    print("The number of clusters to be initially evaluated is " + str(len(clusters.Cluster.unique())), flush=True)
    print("The number of junctions to be initially evaluated is " + str(len(clusters.junction_id.unique())), flush=True)
    clusters_df = clusters.df

    #[5]  additional removal of low confidence junctions under canonical setting 
    if filter_low_juncratios_inclust == "yes":
        
        # refine intron clusters based on splice sites found in them -> remove low confidence junctions basically a filter to see which junctions to keep
        print("Ensuring that junction usage ratios across cluster is not super imabalnced", flush=True)
        print("This step may take a while, will implement parallelization soon!")
        junc_scores_all = refine_clusters(clusters_df.Cluster.unique(), clusters_df, juncs) 
        junc_scores_all = pd.DataFrame(junc_scores_all, columns=["Cluster", "junction_id", "junction_score"])
        # save full dataset
        juncs_all = junc_scores_all.copy()
        # remove junctions that have low scores via threshold_inc 
        junc_scores_all = junc_scores_all[junc_scores_all.junction_score < threshold_inc]
        print("The number of low confidence junctions is " + str(len(junc_scores_all.junction_id.unique())), flush=True)

        # given junctions that remain, see if need to recluster introns (low confidence junctions removed)
        print("Reclustering intron splicing events after low confidence junction removal", flush=True)
        # filter juncs_gr such that it does not contain the junctions in junc_scores_all
        juncs_gr = juncs_gr[~juncs_gr.junction_id.isin(junc_scores_all.junction_id)]
        # check if there are any duplicate entried in pyranges object 
        juncs_gr = juncs_gr.drop_duplicate_positions()
        # drop original cluster column and add new one
        clusters = juncs_gr.cluster(by="gene_id", slack=-1, count=True)
        
        #remove singletons if there are new ones 
        if((singleton) == False):
            clusters = clusters[clusters.Count > 1]
            
    clusters_df = clusters.df
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters_df["junction_id"])]     
    juncs = juncs[juncs.junction_id.isin(clusters_df["junction_id"])]     

    # again confirm that now cluster doesn't just have one unique junction 
    if((singleton) == False):
        # ensure that in every cluster, we only keep junctions that share splice sites 
        print("Confirming that junctions in each cluster share splice sites", flush=True)
        keep_junction_ids = clusters_df.groupby('Cluster').apply(filter_junctions_in_cluster)
        keep_junction_ids = np.concatenate(keep_junction_ids.values)        
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(keep_junction_ids)]
        clusters_df = clusters_df[clusters_df.junction_id.isin(keep_junction_ids)]
        juncs = juncs[juncs.junction_id.isin(clusters_df["junction_id"])]     

    assert((clusters_df.groupby(['Cluster'])["gene_id"].nunique().reset_index().gene_id.unique() == 1))
    clusts_unique = clusters.df[["Cluster", "junction_id", "gene_id", "Count"]].drop_duplicates()

    # merge juncs_gr with corresponding cluster id
    juncs = juncs.merge(clusts_unique, how="left")

    #[6]  Get final list of junction coordinates and save to bed file (easy visualization in IGV)
    juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id"]]
    juncs_gr = juncs_gr.drop_duplicate_positions()
    juncs_gr.to_bed(junc_bed_file, chain=True) #add option to add prefix to file name

    # check if junction doesn't belong to more than 1 cluster 
    clusters = clusters.drop_duplicate_positions()
    juncs_clusts = clusters.df.groupby("junction_id")["Cluster"].count().reset_index()

    # check how many cells in each cell type have at least one read mapping to each junction 
    if sequencing_type == "single_cell":
        grouped_data = juncs.groupby(['junction_id', 'cell_type'])['num_cells_wjunc'].sum().reset_index()
        junction_summary = pd.pivot_table(grouped_data, values='num_cells_wjunc', index='junction_id', columns='cell_type', fill_value=0)

        #   get row sums for each junction
        junction_summary['total_cells_wjunc'] = junction_summary.sum(axis=1)
        junction_summary = junction_summary.sort_values(by=['total_cells_wjunc'], ascending=False)

        # generate quick summary of values in total_cells_wjunc column
        print(junction_summary.total_cells_wjunc.describe(), flush=True)

    # for now just report them first so user knows to be more careful with them, the clustering is also done on gene level
    print("Found junctions that belong to more than one cluster, these are:", flush=True)
    print(juncs_clusts[juncs_clusts["Cluster"] > 1], flush=True)
    print("These are removed from the final results", flush=True)

    # remove clusters that have junctions that belong to more than one cluster
    clusters = clusters.df
    clusters = clusters[clusters.Cluster.isin(juncs_clusts[juncs_clusts["Cluster"] > 1].Cluster) == False]

    # combine cell junction counts with info on junctions and clusters 
    print("The number of clusters to be finally evaluated is " + str(len(juncs.Cluster.unique())), flush=True) 
    print("The number of junctions to be finally evaluated is " + str(len(juncs.junction_id.unique())), flush=True) 
    
    # to the output file add the parameters that was used so user can easily tell how they generated this file 
    output = output_file + "_" + str(min_intron) + "_" + str(max_intron) + "_" + str(min_junc_reads) + "_" + str(min_num_cells_wjunc) + "_" + str(threshold_inc) + "_" + str(sequencing_type) 
    with gzip.open(output + '.gz', mode='wt', encoding='utf-8') as f:
        juncs.to_csv(f, index=False, sep="}")
    print("You can find the output file here: " + output + ".gz", flush=True)
    print("Finished obtaining intron cluster files!")

if __name__ == '__main__':

    # create log file to store everything that gets printed to the console add date and time to the log_file name 
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"Leaflet_log_file_{formatted_time}.log"

    log_file = open(log_file_name, 'a')

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
    filter_low_juncratios_inclust=args.filter_low_juncratios_inclust
    # ensure singleton is boolean
    if args.keep_singletons == "True":
        singleton=True
    else:
        singleton=False
    # ensure strict_filter is boolean
    if args.strict_filter == "True":
        strict_filter=True
    else:
        strict_filter=False
    # print out all user defined arguments that were chosen 
    print("The following arguments were chosen:" , flush=True)
    print("gtf_file: " + gtf_file, flush=True)
    print("junc_files: " + junc_files, flush=True)
    print("output_file: " + output_file, flush=True)
    print("sequencing_type: " + sequencing_type, flush=True)
    print("junc_bed_file: " + junc_bed_file, flush=True)
    print("threshold_inc: " + str(threshold_inc), flush=True)
    print("min_intron: " + str(min_intron), flush=True)
    print("max_intron: " + str(max_intron), flush=True)
    print("min_junc_reads: " + str(min_junc_reads), flush=True)
    print("junc_suffix: " + junc_suffix, flush=True)
    print("min_num_cells_wjunc: " + str(min_num_cells_wjunc), flush=True)
    print("singleton: " + str(singleton), flush=True)
    print("strict_filter: " + str(strict_filter), flush=True)
    print("filter_low_juncratios_inclust: " + (filter_low_juncratios_inclust), flush=True)

    main(junc_files, gtf_file, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, strict_filter, junc_suffix, min_num_cells_wjunc, filter_low_juncratios_inclust)
    # Close the log file when finished
    log_file.close()

