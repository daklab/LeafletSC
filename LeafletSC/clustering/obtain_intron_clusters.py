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

parser.add_argument('--filter_low_juncratios_inclust', dest='filter_low_juncratios_inclust',
                    default="no",
                    help='yes if want to remove lowly used junctions in clusters, default is no')

parser.add_argument('--strict_filter', dest='strict_filter',
                    default=True,
                    help='default is True, this means that only clusters with less junctions that the mean \
                        junction count per cluster is included. This is meant to remove very complex \
                        splicing events that might be hard to make sense of in the single cell context especially.')

args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_gtf(gtf_file): #make this into a seperate script that processes the gtf file into gr object that can be used in the main scriptas input 

    print("The gtf file you provided is " + gtf_file)
    print("Now reading gtf file using gtfparse")
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

def filter_junctions_by_shared_splice_sites(df):
    # Function to apply to each group (cluster)
    def filter_group(group):
        # Find duplicated start and end positions within the group
        duplicated_starts = group['Start'].duplicated(keep=False)
        duplicated_ends = group['End'].duplicated(keep=False)
        
        # Keep rows where either start or end position is duplicated
        return group[duplicated_starts | duplicated_ends]
    
    # Group by 'Cluster' and apply the filtering function
    filtered_df = df.groupby('Cluster').apply(filter_group).reset_index(drop=True)
    
    return filtered_df

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#        Run analysis and obtain intron clusters
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main(junc_files, gtf_file, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, strict_filter, junc_suffix, min_num_cells_wjunc, filter_low_juncratios_inclust):
    
    #1. Check format of junc_files and convert to list if necessary
    # Can either be a list of folders with junction files or a single folder with junction files

    if "," in junc_files:
        junc_files = junc_files.split(",")
    else:
        # if junc_files is a single file, then convert it to a list
        junc_files = [junc_files]

    # Initialize an empty list to store DataFrames
    all_juncs_list = []

    # 3. Process each path
    for junc_path in junc_files:
        
        # make sure junc_path has "/" at the end
        #if not junc_path.endswith("/"):
        #    junc_path = junc_path + "/"
        
        junc_path = Path(junc_path)
        print(f"Reading in junction files from {junc_path}")
        
        # Using rglob to recursively find files matching the suffix
        junc_files_in_path = list(junc_path.rglob(junc_suffix))
        if not junc_files_in_path:
            print(f"No junction files found in {junc_path} with suffix {junc_suffix}")
            continue

        #junc_files_in_path = glob.glob(junc_path + "*" + junc_suffix)  # Adjusted to correctly form the glob pattern
        #if not junc_files_in_path:
        #    print(f"No junction files found in {junc_path} with suffix {junc_suffix}")
        #    continue
        
        print(f"The number of regtools junction files to be processed is {len(junc_files_in_path)}")

        files_not_read = []

        # 4. Read and process each file
        for junc_file in tqdm(junc_files_in_path):
            try:
                juncs = pd.read_csv(junc_file, sep="\t", header=None)
                juncs['file_name'] = junc_file  # Add the file name as a new column
                #juncs['cell_type'] = junc_file.split("/")[-1]
                juncs['cell_type'] = junc_file
                all_juncs_list.append(juncs)  # Append the DataFrame to the list
            except Exception as e:
                print(f"Could not read in {junc_file}: {e}")  
                files_not_read.append(junc_file)

    print("The total number of files that could not be read is " + str(len(files_not_read)) + " as these had no junctions")

    # 5. Concatenate all DataFrames into a single DataFrame
    all_juncs = pd.concat(all_juncs_list, ignore_index=True) if all_juncs_list else pd.DataFrame()

    #1. If gtf_file is not empty, read it in and process it

    if gtf_file is not None:
        gtf_exons_gr = process_gtf(gtf_file)
        print("Done extracting exons from gtf file")
    else:
        pass

    # 6. Convert parameters to integers outside the loop

    min_intron = int(min_intron)
    max_intron = int(max_intron)
    min_junc_reads = int(min_junc_reads)
    min_num_cells_wjunc = int(min_num_cells_wjunc)

    # Define column names based on sequencing type
    col_names = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", 
             "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    if sequencing_type == "single_cell":
        col_names += ["num_cells_wjunc", "cell_readcounts"]
    col_names += ["file_name", "cell_type"]

    print(f"Loading files obtained from {sequencing_type} sequencing")
    
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
    print("Filtering based on intron length")

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

    # 7. Make gr object from ALL junctions across all cell types 
    print("Making gr object from all junctions across all cell types")
    juncs_gr = pr.from_dict({"Chromosome": all_juncs["chrom"], "Start": all_juncs["chromStart"], "End": all_juncs["chromEnd"], "Strand": all_juncs["strand"], "Cell": all_juncs["cell_type"], "junction_id": all_juncs["junction_id"], "counts_total": all_juncs["counts_total"]})

    # we don't actually care about cell types anymore, we just want to obtain a list of junctions to include 
    # in the final analysis and to group them into alternative splicing events
    juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id", "counts_total"]].drop_duplicate_positions()

    # print summary stats for counts_total 
    print("The summary statistics for counts_total across junctions are: ")
    print(juncs_gr.counts_total.describe())
    
    # print the number of junctions with only 1 read across the board is:
    print("The number of junctions with only 1 read across the board is: ")
    print(len(juncs_gr[juncs_gr.counts_total == 1]))

    # if min_junc_reads is not none then remove junctions with less than min_junc_reads
    if min_junc_reads is not None:
        juncs_gr = juncs_gr[juncs_gr.counts_total > min_junc_reads]

    print("The number of junctions after filtering for minimum junction reads is " + str(len(juncs_gr.junction_id.unique())))

    #keep only junctions that could be actually related to isoforms that we expect in our cells (via gtf file provided)
    print("The number of junctions prior to assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())), flush=True)

    # 8. Annotate junctions based on gtf file (if gtf_file is not empty)
    if gtf_file is not None:
        print("Annotating junctions with known exons based on input gtf file")
        juncs_gr = juncs_gr.k_nearest(gtf_exons_gr, strandedness = "same", ties="different", k=2, overlap=False)
        # ensure distance parameter is still 1 
        juncs_gr = juncs_gr[abs(juncs_gr.Distance) == 1]
        # for each junction, the start of the junction should equal end of exons and end of junction should equal start of exon 
        juncs_gr = juncs_gr[(juncs_gr.Start.isin(juncs_gr.End_b)) & (juncs_gr.End.isin(juncs_gr.Start_b))]
        juncs_gr = juncs_gr[juncs_gr.Start == juncs_gr.End_b]
        print("The number of junctions after assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())), flush=True) 
        if len(juncs_gr.junction_id.unique()) < 5000:
            print("There are less than 5000 junctions after assessing distance to exons. Please check your gtf file and ensure that it is in the correct format (start and end positions are not off by 1).", flush=True)
        print("Clustering intron splicing events by gene_id")
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id', 'gene_id']].drop_duplicate_positions()
        clusters = juncs_coords_unique.cluster(by="gene_id", slack=-1, count=True)
        print("The number of clusters after clustering by gene_id is " + str(clusters.Cluster.max()))        
    else:
        print("Clustering intron splicing events by coordinates")
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id', 'counts_total']].drop_duplicate_positions()
        clusters = juncs_coords_unique.cluster(slack=-1, count=True)
        print("The number of clusters after clustering by coordinates is " + str(clusters.Cluster.max()))

    # 9. if singleton is False, remove clusters with only one junction
    if singleton == False:
        print(clusters.Count.value_counts())
        clusters = clusters[clusters.Count > 1]
        print("The number of clusters after removing singletons is " + str(len(clusters.Cluster.unique())))

    # 10. If filter_low_juncratios_inclust is yes, remove junctions with low read counts in clusters
    if filter_low_juncratios_inclust == "yes":
        #print("Ensuring that junction usage ratios across cluster is not super imabalnced")
        print("This option is currently not available")
    else:
        pass

    clusters_df = clusters.df
    # update juncs_gr to include only clusters that are in clusters_df
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters_df.junction_id)]
    all_juncs_df = all_juncs[all_juncs.junction_id.isin(clusters_df["junction_id"])] 

    # 11. Confirming that in every cluster, we only keep junctions taht share splice sites 
    
    print("Confirming that junctions in each cluster share splice sites")
    # Assuming 'clusters_df' is your DataFrame
    filtered_clusters_df = filter_junctions_by_shared_splice_sites(clusters_df)
    print("The number of clusters after filtering for shared splice sites is " + str(len(filtered_clusters_df.Cluster.unique())))

    # Check if any clusters are singletons now and remove if have singleton == False
    # need to update Count column in clusters_df to reflect the new number of junctions in each cluster
    # recalculate Count column adn add it filtered_clusters_df
    filtered_clusters_df['Count'] = filtered_clusters_df.groupby('Cluster')['junction_id'].transform('count')

    # check if any clusters are singletons now and remove if have singleton == False
    if singleton == False:
        print(filtered_clusters_df.Count.value_counts())
        filtered_clusters_df = filtered_clusters_df[filtered_clusters_df.Count > 1]
        print("The number of clusters after removing singletons is " + str(len(filtered_clusters_df.Cluster.unique())))

    # update juncs_gr, clusters_df, and all_juncs_df to include only junctions that are in keep_junction_ids
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(filtered_clusters_df.junction_id)]
    clusters_df = filtered_clusters_df
    all_juncs_df = all_juncs_df[all_juncs_df.junction_id.isin(clusters_df["junction_id"])] 

    if gtf_file is not None:
        clusts_unique = clusters_df[["Cluster", "junction_id", "gene_id", "Count"]].drop_duplicates()
    else:
        clusts_unique = clusters_df[["Cluster", "junction_id", "Count"]].drop_duplicates()
    
     # merge juncs_gr with corresponding cluster id
    all_juncs_df = all_juncs_df.merge(clusts_unique, how="left")

    # get final list of junction coordinates and save to bed file for visualization
    juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id"]]
    juncs_gr = juncs_gr.drop_duplicate_positions()
    juncs_gr.to_bed(junc_bed_file, chain=True) #add option to add prefix to file name

    print("The number of clusters to be finally evaluated is " + str(len(all_juncs_df.Cluster.unique()))) 
    print("The number of junctions to be finally evaluated is " + str(len(all_juncs_df.junction_id.unique())))

    # to add: double check if junctions belonging to multiple clusters, shouldn't really happen 
    # those ones probably need to be pruned -> combined to one intron cluster
    # double check if intron clusters mapping to multiple genes that also shouldn't really happen

    # assert unique number of junctions and clusters in all_juncs_df and clusters_df is the same 
    assert len(all_juncs_df.junction_id.unique()) == len(clusters_df.junction_id.unique())
    assert len(all_juncs_df.Cluster.unique()) == len(clusters_df.Cluster.unique()) 
    
    # 12. Save the final list of intron clusters to a file
    # to the output file add the parameters that was used so user can easily tell how they generated this file 
    # add date also 
    date = time.strftime("%Y%m%d")
    output = output_file + "_" + str(min_intron) + "_" + str(max_intron) + "_" + str(min_junc_reads) + "_" + date + "_" + str(sequencing_type) 
    with gzip.open(output + '.gz', mode='wt', encoding='utf-8') as f:
        all_juncs_df.to_csv(f, index=False, sep="}")
    print("You can find the output file here: " + output + ".gz", flush=True)
    print("Finished obtaining intron cluster files!")

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
    print("gtf_file: " + str(gtf_file), flush=True)
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
