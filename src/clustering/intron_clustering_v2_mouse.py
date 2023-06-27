import pandas as pd
import numpy as np
import argparse
import glob
import os
import pyranges as pr
from gtfparse import read_gtf #initially tested with version 1.3.0 
from tqdm import tqdm
import gzip
import torch
import time 
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import IterableDataset

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--junc_files', dest='junc_files',
                    help='path that has all junction files along with counts in single cells or bulk samples')

parser.add_argument('--sequencing_type', dest='sequencing_type',
                    help='were the junction obtained using data from single cell or bulk sequencing? options are "single_cell" or "bulk". Note if sequencing was done using smart-seq2, then use "bulk" option')

parser.add_argument('--setting', dest='setting', 
                    help='indicate whether analysis should be done in "canonical" mode or "cryptic" where can expect rare events that may be true outlier events')

parser.add_argument('--gtf_file', dest='gtf_file',
                    help='a path to a gtf file to annotate high confidence junctions, ideally from long read sequencing')

parser.add_argument('--output_file', dest='output', 
                    help='name of the output file to save intron cluster file to')

parser.add_argument('--junc_bed_file', dest='junc_bed_file', 
                    help='name of the output bed file to save final list of junction coordinates to')

parser.add_argument('--threshold_inc', dest='threshold_inc',
                    help='threshold to use for removing clusters that have junctions with low read counts at either end, default is 0.01')

parser.add_argument('--min_intron_length', dest='min_intron_length',
                    help='minimum intron length to consider, default is 50')

parser.add_argument('--max_intron_length', dest='max_intron_length',
                    help='maximum intron length to consider, default is 50000')

parser.add_argument('--min_junc_reads', dest='min_junc_reads',
                    help='minimum number of reads to consider a junction, default is 2')

parser.add_argument('--keep_singletons', dest='keep_singletons', 
                    help='Keep "clusters" composed of just one junction. These might represent alternative first or last exon in some cases which may be worthwhile to consider. Default is False')

parser.add_argument('--junc_suffix', dest='junc_suffix',
                    help='suffix of junction files')

parser.add_argument('--min_num_cells_wjunc', dest='min_num_cells_wjunc',
                    help='minimum number of cells that have a junction to consider it, default is 5')

parser.add_argument('--filter_low_juncratios_inclust', dest='filter_low_juncratios_inclust',
                    help='yes if want to remove lowly used junctions in clusters, default is yes')

args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_files(filenames, sequencing_type, min_intron=50, max_intron=50000, min_num_cells_wjunc=5, min_junc_reads=2):
    
    """
    Read in every junction file (ideally associated with a-priori known cell types) and clean up column names 
    and filter out introns that are too long or too short
    """
    
    for filename in filenames:
        print("reading the junction file" + filename)
        juncs=pd.read_csv(filename, sep="\t", header=None, low_memory=False)

        #give columns proper names and keep only those that we care about
        col_names = ["chrom", 	#The name of the chromosome.
            "chromStart", 	#The starting position of the junction-anchor. This includes the maximum overhang for the junction on the left. For the exact junction start add blockSizes[0].
            "chromEnd", 	#The ending position of the junction-anchor. This includes the maximum overhang for the juncion on the left. For the exact junction end subtract blockSizes[1].
            "name", 	#The name of the junctions, the junctions are just numbered JUNC1 to JUNCn.
            "score", 	#The number of reads supporting the junction.
            "strand", 	#Defines the strand - either '+' or '-'. This is calculated using the XS tag in the BAM file.
            "thickStart", 	#Same as chromStart.
            "thickEnd", 	#Same as chromEnd.
            "itemRgb", 	#RGB value - "255,0,0" by default.
            "blockCount", 	#The number of blocks, 2 by default.
            "blockSizes",	#A comma-separated list of the block sizes. The number of items in this list should correspond to blockCount.
            "blockStarts"] 	#A comma-separated list of block starts. All of the blockStart positions should be calculated relative to chromStart. The number of items in this list should correspond to blockCount.] 
        if sequencing_type == "single_cell":
            col_names.append("num_cells_wjunc")
            col_names.append("cell_readcounts")
        juncs = juncs.set_axis(col_names, axis=1, copy=False)
        
        # print("finished reading the junction file" + filename)

        #add and subtract block values
        juncs[['block_add_start','block_subtract_end']] = juncs["blockSizes"].str.split(',', expand=True)
        juncs["chromStart"] = juncs["chromStart"] + juncs['block_add_start'].astype(int)
        juncs["chromEnd"] = juncs["chromEnd"] - juncs['block_subtract_end'].astype(int)

        #get the length of the intron between two exons that make up the junction 
        juncs["intron_length"] = juncs["chromEnd"] - juncs["chromStart"]

        #remove introns longer than 50kb for now (these should be a parameter that can be changed by user)
        juncs = juncs[juncs.intron_length < int(max_intron)]
        juncs = juncs[juncs.intron_length > int(min_intron)]

        #remove junctions that only got less than 5 (default) cells mapping to it 
        if sequencing_type == "single_cell":
            juncs['split_up'] = juncs["cell_readcounts"].str.split(',')
            juncs=juncs.drop(["cell_readcounts"], axis=1)
            juncs = juncs[juncs.num_cells_wjunc >= int(min_num_cells_wjunc)] # junction has at least one read count in at least x different cells 
        
        #remove junctions that have less than "min_junc_reads" total read counts covering it 
        juncs = juncs[juncs["score"] >= int(min_junc_reads)] 
        
        #extract name of cells 
        filename = filename.split("/")[-1]
        if sequencing_type == "single_cell":
            filename = filename.split('.juncswbarcodes')[0]
        elif sequencing_type == "bulk":
            filename = filename.split('.junc')[0]
        juncs['file_name'] = filename
        
        print("The number of junctions found in the sample " + filename + "is: " + str(len(juncs["name"].unique())))
        yield juncs 

def refine_clusters(clusters, clusters_df, dataset, threshold_inc=0.01): #need to improve the speed of this 
    
    """
    Look at a given cluster and all its junctions, check every junction's counts at its 5' end and 3' end. 
    If either is < "threshold_inc" percentage of the total reads then consider removing marked cluster
    (if considering canonical setting). In cryptic case, it could be a real rare event? 
    """
    
    all_juncs_scores=[]
    for clust in tqdm(clusters):
        clust_dat=clusters_df[clusters_df.Cluster==clust]
        #if not #unique start and end ==1 (junctions are the same = only one event in cluster)
        juncs=clust_dat.junction_id.unique()
        juncs_dat_all=dataset[dataset.junction_id.isin(juncs)]
        #combine counts 
        juncs_dat_summ = juncs_dat_all.groupby(["chrom", "chromStart", "chromEnd", "junction_id"], as_index=False).score.sum()
        for junc in juncs:
            junc_chr, junc_start, junc_end, junction_id, junc_count = juncs_dat_summ[juncs_dat_summ.junction_id == junc].iloc[0][0:5]
            # 5'SS  
            ss5_dat=juncs_dat_all[(juncs_dat_all["chromStart"] == junc_start) & (juncs_dat_all["chrom"] == junc_chr)]
            total5_counts=ss5_dat["score"].sum()
            ss5_prop=junc_count/total5_counts
            # 3'SS
            ss3_dat=juncs_dat_all[(juncs_dat_all["chromEnd"] == junc_end) & (juncs_dat_all["chrom"] == junc_chr)]
            total3_counts=ss3_dat["score"].sum()
            ss3_prop=junc_count/total3_counts
            #overal score
            ss_score=min(ss3_prop, ss5_prop)
            if not ss_score <= int(float(threshold_inc)):
                res=[clust, junc, ss_score]
                all_juncs_scores.append(res)
    return(all_juncs_scores)
  
def filter_junctions_in_cluster(group_df):
    # Find the rows that share the same start or end position
    # Account for the fact that duplicates are possible if maps to multiple transcript_ids
    matches = group_df[["Start", "End", "junction_id"]].drop_duplicates()
    # Identify rows that have a duplicated start or end value
    duplicated_starts = matches['Start'].duplicated(keep=False)
    duplicated_ends = matches['End'].duplicated(keep=False)
    duplicated_df = matches[duplicated_starts | duplicated_ends]
    return(duplicated_df.junction_id.values)

def process_gtf(gtf_file): #make this into a seperate script that processes the gtf file into gr object that can be used in the main scriptas input 

    #[1] extract all exons from gtf file provided 
    gtf = read_gtf(gtf_file) #to reduce the speed of this, can just get rows with exon in the feature column (preprocess this before running package)? check if really necessary
    gtf_exons = gtf[(gtf["feature"] == "exon")]
    #remove "chr" from chromosome names only if present
    if gtf_exons['seqname'].str.contains('chr').any():
        gtf_exons['seqname'] = gtf_exons['seqname'].map(lambda x: x.lstrip('chr').rstrip('chr'))
    #make pyranges object
    gtf_exons_gr = pr.from_dict({"Chromosome": gtf_exons["seqname"], "Start": gtf_exons["start"], "End": gtf_exons["end"], "Strand": gtf_exons["strand"], "gene_id": gtf_exons["gene_id"], "gene_name": gtf_exons["gene_name"], "transcript_name": gtf_exons["transcript_name"], "transcript_id": gtf_exons["transcript_id"], "exon_id": gtf_exons["exon_id"]})
    #remove those exons where exon start == exon end 
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.Start == gtf_exons_gr.End)]
    #remove genes with no gene names (most likely novel genes?/not fully annotated)
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.gene_name == "")]

    # When do I need to do this? depends on gtf file used? base 0 or 1? probably need this to be a parameter 
    gtf_exons_gr.Start = gtf_exons_gr.Start-1
    gtf_exons_gr.End = gtf_exons_gr.End.astype("int64")
    gtf_exons_gr.Start = gtf_exons_gr.Start.astype("int64")

    # Drop duplicated positions on same strand 
    gtf_exons_gr = gtf_exons_gr.drop_duplicate_positions(strand=True) #why are so many gone after this? 
    return(gtf_exons_gr)

def main(junc_files, gtf_file, setting, output, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, junc_suffix, min_num_cells_wjunc, filter_low_juncratios_inclust):
    """
    Intersect junction coordinates with up/downstream exons in the canonical setting based on gtf file 
    and then obtain intron clusters using overlapping junctions.
    """

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #        Run analysis and obtain intron clusters
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #[1] load gtf coordinates into pyranges object only if canonical setting is used
    if setting == "canonical":
        gtf_exons_gr = process_gtf(gtf_file)
        print("done extracting exons from gtf file")

    #[2] collect all junctions across all cell types 
    if sequencing_type == "single_cell":
        all_files = glob.glob(os.path.join(junc_files, junc_suffix)) #this suffix should be user defined in case they used something else when running Regtools 
    elif sequencing_type == "bulk":
        all_files = glob.glob(os.path.join(junc_files, "*.junc"))
        #if all_files is empty try the same command with *.juncs instead
        if len(all_files) == 0:
            all_files = glob.glob(os.path.join(junc_files, "*.juncs"))

    # what if have 100s of files here? ... 
    juncs = pd.concat(load_files(all_files, sequencing_type, min_intron, max_intron, min_num_cells_wjunc, min_junc_reads))
    print("Done extracting junctions!")

    # keep only standard chromosomes
    # if "chr" appears in the chrom column 
    if juncs['chrom'].str.contains("chr").any():
        juncs = juncs[juncs['chrom'].str.contains("chr")]
        juncs['chrom'] = juncs['chrom'].map(lambda x: x.lstrip('chr').rstrip('chr'))
    
    # add unique value to each junction name (going to appear multiple times otherwise once for each sample)
    # check if this takes forever with tons of samples 
    juncs["name"] = juncs["name"] + juncs.groupby("name").cumcount().astype(str)
    juncs['junction_id'] = juncs['chrom'] + '_' + juncs['chromStart'].astype(str) + '_' + juncs['chromEnd'].astype(str)

    #make gr object from ALL junctions across all cell types  
    juncs_gr = pr.from_dict({"Chromosome": juncs["chrom"], "Start": juncs["chromStart"], "End": juncs["chromEnd"], "Strand": juncs["strand"], "Cell": juncs["file_name"], "junction_id": juncs["junction_id"], "counts_total": juncs["score"]})

    #keep only junctions that could be actually related to isoforms that we expect in our cells (via gtf file provided)
    print("The number of junctions prior to assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())))

    #[3] annotate each junction with nearbygenes 
    juncs_gr.Start = juncs_gr.Start.astype("int64")
    juncs_gr.End = juncs_gr.End.astype("int64")

    if setting == "canonical":
        print("Annotating junctions with known exons based on input gtf file")
        juncs_gr = juncs_gr.k_nearest(gtf_exons_gr, strandedness = "same", ties="different", k=2, overlap=False)
        #ensure distance parameter is still 1 
        juncs_gr = juncs_gr[abs(juncs_gr.Distance) == 1]
        #for each junction, the start of the junction should equal end of exons and end of junction should equal start of exon 
        juncs_gr = juncs_gr[(juncs_gr.Start.isin(juncs_gr.End_b)) & (juncs_gr.End.isin(juncs_gr.Start_b))]
        juncs_gr = juncs_gr[juncs_gr.Start == juncs_gr.End_b]
        print("The number of junctions after assessing distance to exons is " + str(len(juncs_gr.junction_id.unique()))) 
        if len(juncs_gr.junction_id.unique()) < 5000:
            print("There are less than 5000 junctions after assessing distance to exons. Please check your gtf file and ensure that it is in the correct format (start and end positions are not off by 1).")
        print("Clustering intron splicing events by gene_id")
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id', 'gene_id']].drop_duplicate_positions()
        clusters = juncs_coords_unique.cluster(by="gene_id", slack=-1, count=True)

    #[4]  "cluster" introns annotation free  
    # Each junction should only be related to two exons (one upstream and one downstream)
    # cluster the introns using just the unique set of junction coordinates 
    if setting == "anno_free":
        print("Clustering intron splicing events via junction coordinates, annotation free!")
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id']].drop_duplicate_positions()
        clusters = juncs_coords_unique.cluster(slack=-1, count=True)
    
    #filter intron singleton "clusters" (remove those with only one intron and those with only a single splicing site event (one SS))
    if((singleton) == False):
        print("Removing singleton clusters")
        clusters = clusters[clusters.Count > 1]
    
    print("Removing clusters with more than mean number of junctions")
    clusters = clusters[clusters.Count < clusters.Count.mean()]

    print("The number of clusters to be initially evaluated is " + str(len(clusters.Cluster.unique())))
    print("The number of junctions to be initially evaluated is " + str(len(clusters.junction_id.unique())))
    
    clusters_df = clusters.df
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters_df["junction_id"])]     
    juncs = juncs[juncs.junction_id.isin(clusters_df["junction_id"])]     
    
    #[5]  additional removal of low confidence junctions under canonical setting 
    if filter_low_juncratios_inclust == "yes":
        
        #refine intron clusters based on splice sites found in them -> remove low confidence junctions basically a filter to see which junctions to keep
        junc_scores_all=refine_clusters(clusters_df.Cluster.unique(), clusters_df, juncs, threshold_inc) 
        junc_scores_all = pd.DataFrame(junc_scores_all, columns=["Cluster", "junction_id", "junction_score"])

        #given junctions that remain, see if need to recluster introns (low confidence junctions removed)
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(junc_scores_all["junction_id"])]
        if(setting == "canonical"):
            juncs_gr = juncs_gr.cluster(by="gene_id", slack=-1, count=True)
        if(setting == "anno_free"):
            juncs_gr = juncs_gr.cluster(slack=-1, count=True)
        #remove singletons if there are new ones 
        if((singleton) == False):
            juncs_gr = juncs_gr[juncs_gr.Count > 1]
        
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id']].drop_duplicate_positions()
        clusters = juncs_coords_unique.cluster(slack=-1, count=True)
        clusters_df = clusters.df
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters_df["junction_id"])]     
        juncs = juncs[juncs.junction_id.isin(clusters_df["junction_id"])]     
    
    if setting == "canonical":

        #again confirm that now cluster doesn't just have one unique junction 
        clusters_df=juncs_gr.df
        if((singleton) == False):
            clusters_wnomultiple_events=clusters_df.groupby(['Cluster'])['Chromosome', 'Start', 'End'].nunique().reset_index()
            clusters_wnomultiple_events=clusters_wnomultiple_events[(clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.Start) & (clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.End)].Cluster
            clusters_df = clusters_df[clusters_df['Cluster'].isin(clusters_wnomultiple_events) == False]
            clusters_list = clusters_df.Cluster.unique()
            # keep only clusters that pass 
            juncs_gr = juncs_gr[juncs_gr.Cluster.isin(clusters_df.Cluster.unique())]
            
            # ensure that in every cluster, we only keep junctions that share splice sites 
            print("Confirming that junctions in each cluster share splice sites")
            keep_junction_ids = clusters_df.groupby('Cluster').apply(filter_junctions_in_cluster)
            keep_junction_ids = np.concatenate(keep_junction_ids.values)
            
            # Check if need to recluster again in case we removed junctions that were previously in the same cluster
            juncs_gr = juncs_gr[juncs_gr.junction_id.isin(keep_junction_ids)]
            juncs_gr = juncs_gr.drop("Cluster")
            juncs_gr = juncs_gr.cluster(by="gene_id", slack=-1, count=True)
    
            #again confirm that now cluster doesn't just have one unique junction 
            clusters_df=juncs_gr.df
            clusters_wnomultiple_events=clusters_df.groupby(['Cluster'])['Chromosome', 'Start', 'End'].nunique().reset_index()
            clusters_wnomultiple_events=clusters_wnomultiple_events[(clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.Start) & (clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.End)].Cluster
            clusters_df = clusters_df[clusters_df['Cluster'].isin(clusters_wnomultiple_events) == False]
            juncs_gr = juncs_gr[juncs_gr.Cluster.isin(clusters_df.Cluster.unique())]
            # ensure that every junction is only attributed to one gene's intron cluster 
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
    juncs_clusts = clusters.df.groupby("junction_id")["Cluster"].count().reset_index()

    # for now just report them first so user knows to be more careful with them, the clustering is also done on gene level
    print("Found junctions that belong to more than one cluster, these are:")
    print(juncs_clusts[juncs_clusts["Cluster"] > 1])
    print("These are removed from the final results")

    # remove clusters that have junctions that belong to more than one cluster
    clusters = clusters.df
    clusters = clusters[clusters.Cluster.isin(juncs_clusts[juncs_clusts["Cluster"] > 1].Cluster) == False]

    # combine cell junction counts with info on junctions and clusters 
    print("The number of clusters to be finally evaluated is " + str(len(juncs.Cluster.unique()))) 
    print("The number of junctions to be finally evaluated is " + str(len(juncs.junction_id.unique()))) 
    
    # to the output file add the parameters that was used so user can easily tell how they generated this file 
    output = output + "_" + setting + "_" + str(min_intron) + "_" + str(max_intron) + "_" + str(min_junc_reads) + "_" + str(min_num_cells_wjunc) + "_" + str(threshold_inc) + "_" + str(sequencing_type) 
    with gzip.open(output + '.gz', mode='wt', encoding='utf-8') as f:
        juncs.to_csv(f, index=False, sep="}")

if __name__ == '__main__':
    gtf_file=args.gtf_file
    path=args.junc_files
    setting=args.setting
    output_file=args.output
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
    # print out all user defined arguments that were chosen 
    print("The following arguments were chosen:")
    print("gtf_file: " + gtf_file)
    print("path: " + path)
    print("setting: " + setting)
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
    print("filter_low_juncratios_inclust: " + (filter_low_juncratios_inclust))

    main(path, gtf_file, setting, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, junc_suffix, min_num_cells_wjunc, filter_low_juncratios_inclust)

# to test run 
#gtf_file="/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/genes/genes.gtf"
#junc_files="/gpfs/commons/scratch/kisaev/ss_tabulamuris_test"
#output_file="/gpfs/commons/scratch/kisaev/ss_tabulamuris_test/Leaflet/clustered_junctions_noanno.txt" 
#setting="anno_free"
#sequencing_type="single_cell"
#singleton="False"
#junc_bed_file="/gpfs/commons/scratch/kisaev/ss_tabulamuris_test/Leaflet/clustered_junctions.bed"
#min_junc_reads=10
#min_num_cells_wjunc=5
#junc_suffix="*.juncswbarcodes"
#filter_low_juncratios_inclust="yes"
#python src/clustering/intron_clustering_v2_mouse.py --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --min_intron=50 --max_intron=500000 --min_junc_reads=$min_junc_reads --threshold_inc=0.1 --junc_bed_file $junc_bed_file --keep_singletons $singleton --junc_suffix $junc_suffix --min_num_cells_wjunc=$min_num_cells_wjunc --filter_low_juncratios_inclust $filter_low_juncratios_inclust
