# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import glob
import os
import pyranges as pr
from gtfparse import read_gtf
from tqdm import tqdm
import time
import anndata as ad
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--junc_files', dest='junc_files',
                    help='path that has all junction files along with counts in single cells')
parser.add_argument('--setting', dest='setting', 
                    help='indicate whether analysis should be done in "canonical" mode or "cryptic" where can expect rare events that may be true outlier events')
parser.add_argument('--gtf_file', dest='gtf_file',
                    help='a gtf file to annotate high confidence junctions, ideally from long read sequencing')
parser.add_argument('--output_file', dest='output', 
                    help='name of the output file to save intron cluster file to')
args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_files(filenames):
    
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
        "blockStarts", 	#A comma-separated list of block starts. All of the blockStart positions should be calculated relative to chromStart. The number of items in this list should correspond to blockCount.
        "num_cells_wjunc", 
        "cell_readcounts"] 

        juncs = juncs.set_axis(col_names, axis=1, inplace=False)
        #add and subtract block values
        juncs[['block_add_start','block_subtract_end']] = juncs["blockSizes"].str.split(',', expand=True)
        juncs["chromStart"] = juncs["chromStart"] + juncs['block_add_start'].astype(int)
        juncs["chromEnd"] = juncs["chromEnd"] - juncs['block_subtract_end'].astype(int)

        #get the length of the intron between two exons that make up the junction 
        juncs["intron_length"] = juncs["chromEnd"] - juncs["chromStart"]

        #remove introns longer than 50kb for now (these should be a parameter that can be changed by user)
        juncs = juncs[juncs.intron_length < 50000]
        juncs = juncs[juncs.intron_length > 50]

        #remove junctions that only got less than 3 cell mapping to it 
        juncs['split_up'] = juncs["cell_readcounts"].str.split(',')
        juncs=juncs.drop(["cell_readcounts"], axis=1)
        juncs = juncs[juncs.num_cells_wjunc >= 5] #junction has at least one read count in at least three different cells (should also be changeable parameter)
        
        #remove junctions that have less than 3 total read counts covering it 
        juncs = juncs[juncs["score"] > 5] 
        
        #extract name of cells 
        filename = filename.split("/")[-1]
        filename = filename.split('.wbarcode.junc')[0]
        juncs['file_name'] = filename
        
        print("The number of junctions found is " + str(len(juncs["name"].unique())))
        yield juncs 

def refine_clusters(clusters, clusters_df, dataset, threshold_inc): #need to improve the speed of this 
    
    """
    Look at a given cluster and all its junctions, check every junction's counts at its 5' end and 3' end. 
    If either is < "threshold_inc" percentage of the total reads then consider removing marked cluster
    (if considering canonical setting). In cryptic case, it could be a real rare event? 
    """
    
    all_juncs_scores=[]
    for clust in tqdm(clusters):
        clust_dat=clusters_df[clusters_df.Cluster==clust]
        #if not #unique start and end ==1 (junctions are the same = only one event in cluster)
        juncs_low_conf=[]
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
            if not ss_score <= threshold_inc:
                res=[clust, junc, ss_score]
                all_juncs_scores.append(res)
    return(all_juncs_scores)


def main(junc_files, gtf_file, setting, output):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    startTime = datetime.now()

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #        Run analysis and obtain intron clusters
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #[1] extract all exons from gtf file provided 
    gtf = read_gtf(gtf_file)
    #keep only exons so can look at junctions
    gtf_exons = gtf[gtf["feature"] == "exon"]
    #remove "chr" from chromosome names 
    gtf_exons['seqname'] = gtf_exons['seqname'].map(lambda x: x.lstrip('chr').rstrip('chr'))
    #make pyranges object
    gtf_exons_gr = pr.from_dict({"Chromosome": gtf_exons["seqname"], "Start": gtf_exons["start"], "End": gtf_exons["end"], "Strand": gtf_exons["strand"], "gene_id": gtf_exons["gene_id"], "transcript_id": gtf_exons["transcript_id"], "exon_number": gtf_exons["exon_number"]})
    #remove those exons where exon start == exon end 
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.Start == gtf_exons_gr.End)]
    gtf_exons_gr.Start = gtf_exons_gr.Start-1

    #[2] collect all junctions across all cell types 
    all_files = glob.glob(os.path.join(junc_files, "*.wbarcode.junc"))
    df = pd.concat(load_files(all_files))
    print("done collecting junctions")

    #add unique value to each junction name (going to appear multiple times otherwise once for each sample)
    df["name"] = df["name"] + df.groupby("name").cumcount().astype(str)
    df['junction_id'] = df['chrom'] + '_' + df['chromStart'].astype(str) + '_' + df['chromEnd'].astype(str)

    #make gr object from ALL junctions across all cell types  
    gr = pr.from_dict({"Chromosome": df["chrom"], "Start": df["chromStart"], "End": df["chromEnd"], "Strand": df["strand"], "Cell": df["file_name"], "junction_id": df["junction_id"], "counts_total": df["score"]})

    #keep only junctions that could be actually related to isoforms that we expect in our cells (via gtf file provided)
    print("The number of junctions prior to assessing distance to exons is " + str(len(gr.junction_id.unique())))
    junctions_only_orig=gr #save a copy of just gr object for junctions

    #[3] annotate each junction with genes 
    gr = gr.k_nearest(gtf_exons_gr, strandedness = "same", ties="different", k=2, overlap=False)
    #for each junction, the start of the junction should equal end of exons and end of junction should equal start of exon 
    gr = gr[(gr.Start.isin(gr.End_b)) & (gr.End.isin(gr.Start_b))]
    #ensure distance parameter is still 1 
    gr = gr[gr.Distance == 1]
    print("The number of junctions after assessing distance to exons is " + str(len(gr.junction_id.unique()))) #what's up with all the junctions that aren't within 10bp of exons? check where they are appearing in the genome? (noise? non human genes / not mapping to human PBMC reference)

    #[4]  "cluster" introns events by gene  
    clusters = gr.cluster(by="gene_id", slack=-1, count=True)

    #filter intron "clusters" (remove those with only one intron and those with only a single splicing site event (one SS))
    clusters_df=clusters.df
    clusters_wnomultiple_events=clusters_df.groupby(['Cluster'])['Chromosome', 'Start', 'End'].nunique().reset_index()
    clusters_wnomultiple_events=clusters_wnomultiple_events[(clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.Start) & (clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.End)].Cluster
    clusters_df = clusters_df[clusters_df['Cluster'].isin(clusters_wnomultiple_events) == False]
    clusters_list=clusters_df.Cluster.unique() 

    print("The number of clusters to be initially evaluated is " + str(len(clusters_list)))
    df=df[df.junction_id.isin(clusters_df["junction_id"])] 

    #[5]  additional removal of low confidence junctions under canonical setting 
    if setting == "canonical":
        #refine intron clusters based on splice sites found in them -> remove low confidence junctions basically a filter to see which junctions to keep
        junc_scores_all=refine_clusters(clusters_list, clusters_df, df, 0.01) #this is very slow, need to make this into dataloader
        junc_scores_all = pd.DataFrame(junc_scores_all, columns=["Cluster", "junction_id", "junction_score"])

        #given junctions that remain, see if need to recluster introns (low confidence junctions removed)
        gr = gr[gr.junction_id.isin(junc_scores_all["junction_id"])]
        gr = gr.cluster(by="gene_id", slack=-1, count=True)

        #again confirm that now cluster doesn't just have one unique junction 
        clusters_df=gr.df
        clusters_wnomultiple_events=clusters_df.groupby(['Cluster'])['Chromosome', 'Start', 'End'].nunique().reset_index()
        clusters_wnomultiple_events=clusters_wnomultiple_events[(clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.Start) & (clusters_wnomultiple_events.Chromosome == clusters_wnomultiple_events.End)].Cluster
        clusters_df = clusters_df[clusters_df['Cluster'].isin(clusters_wnomultiple_events) == False]
    
    #ensure that every junction is only attributed to one gene's intron cluster 
    assert((clusters_df.groupby(['Cluster'])["gene_id"].nunique().reset_index().gene_id.unique() == 1))

    df=df[df.junction_id.isin(clusters_df["junction_id"])] 
    clusters=clusters_df[["junction_id", "Cluster", "gene_id", "transcript_id"]]

    #[6]  Get final list of junction coordinates and save to bed file (easy visualization in IGV)
    gr = gr[gr.junction_id.isin(df["junction_id"])]
    gr = gr[["Chromosome", "Start", "End", "Strand", "junction_id", "Cluster"]]
    gr = gr.drop_duplicate_positions()
    gr.to_bed("extracted_junction_coordinates.bed", chain=True)

    #summary number of junctions per cluster 
    summ_clusts_juncs=clusters[["Cluster", "junction_id"]].drop_duplicates().groupby("Cluster")["junction_id"].count().reset_index()
    summ_clusts_juncs = summ_clusts_juncs.sort_values("junction_id", ascending=False)

    #remove transcript_id columns
    clusters=clusters.drop("transcript_id", axis=1)
    clusters=clusters.drop_duplicates()

    #ensure junction doesn't belong to more than 1 cluster 
    juncs_clusts = clusters.groupby("junction_id")["Cluster"].count().reset_index()

    #for now keep those junctions, might be in regions where there are multiple 
    # overlapping genes so would be hard to decipher anyhow 
    #can look in more detail at this later at some point 
    #for now just report them so user knows to be more careful with them, the clustering is also done on gene level
    print(juncs_clusts[juncs_clusts["Cluster"] > 1])

    #combine cell junction counts with info on junctions and clusters 
    df=pd.merge(clusters, df, on = "junction_id")
    print("The number of clusters to be finally evaluated is " + str(len(df.Cluster.unique()))) 

    #save this file and return (main output from script)
    df.to_csv(output, index=False, sep="}")  #find alterantive more efficient way to save this file, pickl file?

    print(datetime.now() - startTime)

if __name__ == '__main__':
    gtf_file=args.gtf_file
    path=args.junc_files
    setting=args.setting
    output_file=args.output
    main(path, gtf_file, setting, output_file)