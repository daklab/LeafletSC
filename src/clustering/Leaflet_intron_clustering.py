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
                    help='path that has all junction files along with counts in single cells or bulk samples, make sure path ends in "/" ')

parser.add_argument('--sequencing_type', dest='sequencing_type',
                    help='were the junction obtained using data from single cell or bulk sequencing? options are "single_cell" or "bulk". Note if sequencing was done using smart-seq2, then use "bulk" option')

parser.add_argument('--setting', dest='setting', 
                    help='indicate whether analysis should be done in "canonical" mode or "anno_free" where can expect rare events that may be true outlier events')

parser.add_argument('--gtf_file', dest='gtf_file',
                    help='a path to a gtf file to annotate high confidence junctions, ideally from long read sequencing')

parser.add_argument('--output_file', dest='output_file', 
                    default='intron_clusters.txt',
                    help='name of the output file to save intron cluster file to')

parser.add_argument('--junc_bed_file', dest='junc_bed_file', 
                    default='juncs.bed',
                    help='name of the output bed file to save final list of junction coordinates to')

parser.add_argument('--threshold_inc', dest='threshold_inc',
                    default=0.01,
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
                    default=5,
                    help='minimum number of cells that have a junction to consider it, default is 5')

parser.add_argument('--filter_low_juncratios_inclust', dest='filter_low_juncratios_inclust',
                    default="yes",
                    help='yes if want to remove lowly used junctions in clusters, default is yes')

parser.add_argument('--strict_filter', dest='strict_filter',
                    default=True,
                    help='default is True, this means that only clusters with less junctions that the mean junction count per cluster is included. This is meant to remove very complex splicing events that might be hard to make sense of in the single cell context especially.')

args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_gtf(gtf_file): #make this into a seperate script that processes the gtf file into gr object that can be used in the main scriptas input 

    print("The gtf file you provided is " + gtf_file)
    
    #[1] extract all exons from gtf file provided 
    gtf = read_gtf(gtf_file) #to reduce the speed of this, can just get rows with exon in the feature column (preprocess this before running package)? check if really necessary
    
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

    juncs = pd.read_csv(filename, sep="\t", header=None, low_memory=False)
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

    print(cell_type , flush=True)
    juncs['cell_type'] = cell_type
    mask = juncs["score"] >= min_junc_reads
    juncs = juncs[mask]
    yield juncs


def load_files(filenames, sequencing_type, junc_suffix, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads):
    
    start_time = time.time()
    junc_suff = junc_suffix.split("*")[1]
    print(junc_suff)

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
        print(sequencing_type, junc_suff, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads)
    
    print("Loading files obtained from " + sequencing_type + " sequencing")  
    print(col_names)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: read_file(x, sequencing_type, col_names, junc_suff, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads), filenames))

    print("Concatenating all the junctions")
    all_juncs = pd.concat(results)
    print("Reading all the junctions took " + str(round((time.time()-start_time), 2)) + " seconds")
    return all_juncs

@njit
def compute_ss_score(junc_count, total5_counts, total3_counts):
    ss5_prop = junc_count / total5_counts
    ss3_prop = junc_count / total3_counts
    return min(ss5_prop, ss3_prop)

def refine_clusters(clusters, clusters_df, dataset, threshold_inc=0.01):
    all_juncs_scores = []

    for clust in tqdm(clusters):
        clust_dat = clusters_df[clusters_df.Cluster == clust]
        juncs_dat = clust_dat.junction_id.unique()

        # Preprocess data
        juncs_dat_all = dataset[dataset.junction_id.isin(juncs_dat)]
        juncs_dat_summ = juncs_dat_all.groupby(["chrom", "chromStart", "chromEnd", "junction_id"], as_index=False).score.sum()

        for junc in juncs_dat:
            junc_row = juncs_dat_summ[juncs_dat_summ.junction_id == junc].iloc[0]
            junc_chr, junc_start, junc_end, _, junc_count = junc_row[0:5]

            ss5_dat = juncs_dat_all[(juncs_dat_all["chromStart"] == junc_start) & (juncs_dat_all["chrom"] == junc_chr)]
            total5_counts = ss5_dat["score"].sum()

            ss3_dat = juncs_dat_all[(juncs_dat_all["chromEnd"] == junc_end) & (juncs_dat_all["chrom"] == junc_chr)]
            total3_counts = ss3_dat["score"].sum()

            ss_score = compute_ss_score(junc_count, total5_counts, total3_counts)
            if ss_score > threshold_inc:
                all_juncs_scores.append([clust, junc, ss_score])
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

def main(junc_files, gtf_file, setting, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, strict_filter, junc_suffix, min_num_cells_wjunc, filter_low_juncratios_inclust):
    """
    Intersect junction coordinates with up/downstream exons in the canonical setting based on gtf file 
    and then obtain intron clusters using overlapping junctions.
    """

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #        Run analysis and obtain intron clusters
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # Redirect standard output to the log file
    sys.stdout = log_file

    #[1] load gtf coordinates into pyranges object only if canonical setting is used
    if setting == "canonical":
        gtf_exons_gr = process_gtf(gtf_file)
        print("Done extracting exons from gtf file")

    #[2] collect all junctions across all cell types 
    all_files = glob.glob(os.path.join(junc_files, junc_suffix)) #this suffix should be user defined in case they used something else when running Regtools 
       
    print("The number of regtools junction files to be processed is " + str(len(all_files)), flush=True)
    print(all_files, flush=True)
    
    # concat all junction files by reading them in parallel first
    juncs = load_files(all_files, sequencing_type, junc_suffix, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads)

    print("Done extracting junctions!", flush=True)

    juncs = juncs.copy()

    # if "chr" appears in the chrom column 
    if juncs['chrom'].str.contains("chr").any():
        juncs = juncs[juncs['chrom'].str.contains("chr")]
        juncs['chrom'] = juncs['chrom'].map(lambda x: x.lstrip('chr').rstrip('chr'))
    
    # add unique value to each junction name (going to appear multiple times otherwise once for each sample)
    # check if this takes forever with tons of samples 
    juncs["name"] = juncs["name"] + juncs.groupby("name").cumcount().astype(str)
    juncs['junction_id'] = juncs['chrom'] + '_' + juncs['chromStart'].astype(str) + '_' + juncs['chromEnd'].astype(str)

    #make gr object from ALL junctions across all cell types  
    juncs_gr = pr.from_dict({"Chromosome": juncs["chrom"], "Start": juncs["chromStart"], "End": juncs["chromEnd"], "Strand": juncs["strand"], "Cell": juncs["cell_type"], "junction_id": juncs["junction_id"], "counts_total": juncs["score"]})

    #keep only junctions that could be actually related to isoforms that we expect in our cells (via gtf file provided)
    print("The number of junctions prior to assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())), flush=True)

    #[3] annotate each junction with nearbygenes 
    if setting == "canonical":
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

    #[4]  "cluster" introns annotation free  
    # Each junction should only be related to two exons (one upstream and one downstream)
    if setting == "anno_free":
        print("Clustering intron splicing events via junction coordinates, annotation free!", flush=True)
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id']].drop_duplicate_positions()
        print("The number of junctions prior to clustering " + str(len(juncs_coords_unique.junction_id.unique())), flush=True)
        clusters = juncs_coords_unique.cluster(slack=-1, count=True)
        print("The number of junctions after clustering " + str(len(clusters.junction_id.unique())), flush=True)
    
    print(clusters.head())

    # filter intron singleton "clusters" (remove those with only one intron and those with only a single splicing site event (one SS))
    if((singleton) == False):
        print("Removing singleton clusters", flush=True)
        clusters = clusters[clusters.Count > 1]
        print("The number of junctions after removing singletons " + str(len(clusters.junction_id.unique())), flush=True)
    
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
        junc_scores_all = refine_clusters(clusters_df.Cluster.unique(), clusters_df, juncs, threshold_inc) 
        junc_scores_all = pd.DataFrame(junc_scores_all, columns=["Cluster", "junction_id", "junction_score"])
        print("The number of junctions after filtering low confidence junctions is " + str(len(junc_scores_all.junction_id.unique())), flush=True)

        # given junctions that remain, see if need to recluster introns (low confidence junctions removed)
        print("Reclustering intron splicing events after low confidence junction removal", flush=True)
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(junc_scores_all["junction_id"])]
        if(setting == "canonical"):
            # drop original cluster column and add new one
            clusters = juncs_gr.cluster(by="gene_id", slack=-1, count=True)
        if(setting == "anno_free"):
            clusters = juncs_gr.cluster(slack=-1, count=True)
        
        #remove singletons if there are new ones 
        if((singleton) == False):
            clusters = clusters[clusters.Count > 1]
            
    clusters_df = clusters.df
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters_df["junction_id"])]     
    juncs = juncs[juncs.junction_id.isin(clusters_df["junction_id"])]     
    
    if setting == "canonical": # this part needs fixing... can be shortened

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

    if setting == "anno_free": #annotation free setting dont need to overlap with exons 
        clusts_unique = clusters.df[["Cluster", "junction_id", "Count"]].drop_duplicates()

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
    output = output_file + "_" + setting + "_" + str(min_intron) + "_" + str(max_intron) + "_" + str(min_junc_reads) + "_" + str(min_num_cells_wjunc) + "_" + str(threshold_inc) + "_" + str(sequencing_type) 
    with gzip.open(output + '.gz', mode='wt', encoding='utf-8') as f:
        juncs.to_csv(f, index=False, sep="}")
    print("You can find the output file here: " + output + ".gz", flush=True)

if __name__ == '__main__':

    # create log file to store everything that gets printed to the console add date and time to the log_file name 
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"Leaflet_log_file_{formatted_time}.log"

    log_file = open(log_file_name, 'a')

    gtf_file=args.gtf_file
    junc_files=args.junc_files
    setting=args.setting
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
    print("setting: " + setting, flush=True)
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

    main(junc_files, gtf_file, setting, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, strict_filter, junc_suffix, min_num_cells_wjunc, filter_low_juncratios_inclust)
    # Close the log file when finished
    log_file.close()

#alt gtf file
#gtf_file="/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf"
#gtf_file="/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/kallisto/Mus_musculus.GRCm38.102.gtf"
#path = "/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/"
#setting = "annotated"
#sequencing_type = "bulk"