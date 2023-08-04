import pandas as pd
import numpy as np
import glob
import os
import pyranges as pr
from gtfparse import read_gtf #initially tested with version 1.3.0 
import gzip
import argparse

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, one file per line and no header')

parser.add_argument('--gtf_file', dest='gtf_file',
                    help='a path to a gtf file to annotate high confidence junctions, ideally from long read sequencing')

parser.add_argument('--output_file', dest='output', 
                    help='name of the output file to save intron cluster file to')

parser.add_argument('--junc_bed_file', dest='junc_bed_file', 
                    help='name of the output bed file to save final list of junction coordinates to')

parser.add_argument('--min_junc_counts', dest='min_junc_counts', 
                    help='minimum number of junction counts to keep junction in the analysis', default=10, type=int)

args = parser.parse_args()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_gtf(gtf_file): #make this into a seperate script that processes the gtf file into gr object that can be used in the main scriptas input 

    #[1] extract all exons from gtf file provided 
    gtf = read_gtf(gtf_file) #to reduce the speed of this, can just get rows with exon in the feature column (preprocess this before running package)? check if really necessary
    gtf_exons = gtf[(gtf["feature"] == "exon")]
    #remove "chr" from chromosome names only if present
    if gtf_exons['seqname'].str.contains('chr').any():
        gtf_exons['seqname'] = gtf_exons['seqname'].map(lambda x: x.lstrip('chr').rstrip('chr'))
    #make pyranges object
    gtf_exons_gr = pr.from_dict({"Chromosome": gtf_exons["seqname"], "Start": gtf_exons["start"], "End": gtf_exons["end"], "Strand": gtf_exons["strand"], "gene_id": gtf_exons["gene_id"], "gene_name": gtf_exons["gene_name"], "transcript_name": gtf_exons["transcript_name"], "transcript_id": gtf_exons["transcript_id"], "exon_number": gtf_exons["exon_number"]})
    #remove those exons where exon start == exon end 
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.Start == gtf_exons_gr.End)]
    #remove genes with no gene names (most likely novel genes?/not fully annotated)
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.gene_name == "")]
    # When do I need to do this? depends on gtf file used? base 0 or 1?
    gtf_exons_gr.Start = gtf_exons_gr.Start-1
    gtf_exons_gr.End = gtf_exons_gr.End.astype("int64")
    # Drop duplicated positions on same strand 
    gtf_exons_gr = gtf_exons_gr.drop_duplicate_positions(strand=True) #why are so many gone after this? 
    return(gtf_exons_gr)

def main(gtf_file, junc_bed_file, output_file, min_junc_counts=9000):
    """
    Intersect junction coordinates with up/downstream exons in the canonical setting based on gtf file 
    and then obtain intron clusters using overlapping junctions.
    """

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #        Run analysis and obtain intron clusters
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # just loading GTEx junction summary file here, generated using the script /gpfs/commons/home/kisaev/leafcutter-sc/data/external/GTEx_tot_junc_counts.sh
    file_name = "/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_juncs_total_counts.txt"
    junc_genes = "/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_juncs_genes.txt" # generated with command: tail -n +3 GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct | cut -f1,2 > GTEx_juncs_genes.txt

    # load summarized GTEx junctions (total counts across all samples for each junction)
    gtex = pd.read_csv(file_name, sep='\t')
    # change column names to "Name", "Junc_Counts"
    gtex.columns = ["Name", "Junc_Counts"]

    # load file with gene names for each junction
    gtex_genes = pd.read_csv(junc_genes, sep='\t')

    # merge with gtex_genes to add on gene name 
    gtex = gtex.merge(gtex_genes, on="Name", how="left")

    #[1] load gtf coordinates into pyranges object
    gtf_exons_gr = process_gtf(gtf_file)
    print("------------------Done extracting exons from GTF file------------------")

    #[2] seperate junctions in GTEx files into chr, start, end and strand columns... 
    gtex[['chr', 'start', 'end']] = gtex['Name'].str.split('_', expand=True)   
    gtex['gene_id'] = gtex['Description'].fillna('').str.split('.').str[0] #get rid of version number in gene_id
    
    # remove junctions with less than min_junc_counts counts
    # remove junctions before clustering that have very few counts (to reduce noise) for now at least 
    # those junctions could still be interesting to look at later but right now need to minimize the size of the dataset 
    gtex = gtex[gtex.Junc_Counts > min_junc_counts]
    print("------------------" + str(len(gtex.gene_id.unique())) + " unique genes in GTEx junction file------------------")

    #[3] need to get strand info using gtf file and merge it with gtex 
    gtex = gtex.merge(gtf_exons_gr.df[["Strand", "gene_id"]], on="gene_id", how="left")
    # if "chr" appears in the chrom column
    if gtex['chr'].str.contains("chr").any():
        gtex = gtex[gtex['chr'].str.contains("chr")]
        gtex['chr'] = gtex['chr'].map(lambda x: x.lstrip('chr').rstrip('chr'))
    # remove junctions with no strand info 
    gtex = gtex[gtex['Strand'].notnull()]
    print("------------------" + str(len(gtex.gene_id.unique())) + " unique genes in GTEx junction file after merging with gtf file------------------")

    # keep junction_id to map back later 
    gtex['junction_id'] = gtex['chr'] + '_' + gtex['start'].astype(str) + '_' + gtex['end'].astype(str)
    # keep unique entries 
    gtex = gtex.drop_duplicates()

    # remove junctions that appear across multiple rows 
    gtex = gtex.drop_duplicates(subset=['junction_id'])
    
    # convert start and end columns to int 
    gtex.start = gtex.start.astype("int64")
    gtex.end = gtex.end.astype("int64")

    # add a column with the size of the junction
    gtex["size"] = gtex.end - gtex.start

    # remove junctions that are too small or too large
    gtex = gtex[(gtex["size"] > 100) & (gtex["size"] < 250000)]

    # make a gr object from gtex junctions 
    gtex_gr = pr.from_dict({"Chromosome": gtex["chr"], "Start": gtex["start"], "End": gtex["end"], "Strand": gtex["Strand"], "gene_id": gtex["gene_id"], "junction_id": gtex["junction_id"]})

    # keep only junctions that could be actually related to isoforms that we expect in our cells (via gtf file provided)
    print("------------------The number of junctions prior to assessing distance to exons is " + str(len(gtex_gr.junction_id.unique())))

    #[4] annotate each junction with nearby genes 
    gtf_exons_gr.Start = gtf_exons_gr.Start.astype("int64")
    gtf_exons_gr.End = gtf_exons_gr.End.astype("int64")
    # if a gene_id is associated with multiple gene names, keep only the first gene name 
    gtf_exons_gr = gtf_exons_gr[["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"]].drop_duplicate_positions()

    gtex_gr.Start = gtex_gr.Start.astype("int64")
    gtex_gr.End = gtex_gr.End.astype("int64")
    gtex_gr.Start = gtex_gr.Start-1

    gtex_gr_full = gtex_gr.k_nearest(gtf_exons_gr, strandedness = "same", ties="different", k=2, overlap=False)
    # ensure distance parameter is still 1 
    gtex_gr = gtex_gr_full[abs(gtex_gr_full.Distance) == 1]
    print("------------------The number of junctions after assessing distance to exons is " + str(len(gtex_gr.junction_id.unique()))) 

    # for each junction, the start of the junction should equal end of exons and end of junction should equal start of exon 
    gtex_gr = gtex_gr[(gtex_gr.Start.isin(gtex_gr.End_b)) & (gtex_gr.End.isin(gtex_gr.Start_b))]
    print("------------------The number of junctions after assessing distance to exons is " + str(len(gtex_gr.junction_id.unique()))) 

    gtex_gr = gtex_gr[gtex_gr.End == gtex_gr.Start_b]
    print("------------------The number of junctions after assessing distance to exons is " + str(len(gtex_gr.junction_id.unique()))) 

    # just need unique combinations of chr, start, end, strand, gene_id, junction_id 
    gtex_gr = gtex_gr[["Chromosome", "Start", "End", "Strand", "gene_id", "junction_id", "gene_name"]]
    gtex_gr = gtex_gr.df.drop_duplicates()
    gtex_gr = pr.PyRanges(gtex_gr)

    print("------------------The number of junctions after assessing distance to exons is " + str(len(gtex_gr.junction_id.unique()))) 

    #[5]  "cluster" introns events by gene  
    clusters = gtex_gr.cluster(by="gene_id", slack=-1, count=True) #check if need to subtract 1 from junc start or end
    clusters_df = clusters.df

    # Order by Count 
    clusters_df = clusters_df.sort_values("Count", ascending=False)
    
    # Remove singletons where Count == 1 and those where count is greater than 20 
    clusters_df = clusters_df[(clusters_df.Count > 1) & (clusters_df.Count < 10)]
    print("------------------The number of clusters to be initially evaluated is " + str(len(clusters_df.Cluster.unique())))

    # ensure that every junction is only attributed to one gene's intron cluster 
    assert((clusters_df.groupby(['Cluster'])["gene_id"].nunique().reset_index().gene_id.unique() == 1))
    clusters_df = clusters_df[["junction_id", "Cluster", "gene_id", "gene_name"]].drop_duplicates()

    #[6]  Get final list of junction coordinates and save to bed file (easy visualization in IGV)
    gtex_gr = clusters[["Chromosome", "Start", "End", "Strand", "junction_id", "Cluster", "gene_name", "gene_id"]]
    gtex_gr = gtex_gr.drop_duplicate_positions()

    # keep only the junctions remaining in clusters_df
    gtex_gr = gtex_gr[gtex_gr.junction_id.isin(clusters_df.junction_id)]
    print("------------------The number of junctions is " + str(len(gtex_gr.junction_id.unique()))) 

    print("------------MAKING BED FILE------------------")
    gtex_gr.to_bed(junc_bed_file, chain=True) #add option to add prefix to file name

    # check if junction doesn't belong to more than 1 cluster 
    juncs_clusts = clusters_df.groupby("junction_id")["Cluster"].count().reset_index()

    print("------------------Found junctions that belong to more than one cluster, these are:")
    print(juncs_clusts[juncs_clusts["Cluster"] > 1])
    print("------------------These are removed from the final results")

    # remove clusters that have junctions that belong to more than one cluster
    clusters = clusters[clusters.Cluster.isin(juncs_clusts[juncs_clusts["Cluster"] > 1].Cluster) == False]

    # combine cell junction counts with info on junctions and clusters 
    print("------------------The number of clusters to be finally evaluated is " + str(len(clusters.Cluster.unique()))) 
    
    #save this file and return (main output from script)
    # save the DataFrame with compression
    file_name = output_file + ".h5"
    print("------------------" + file_name + "------------------")
    clusters = clusters.df
    # rename the Cluster from 1 to n 
    clusters["Cluster"] = clusters.groupby("Cluster").ngroup() + 1
    clusters.to_hdf(file_name, key='df', mode='w', complevel=5, complib='blosc', format='table')

if __name__ == '__main__':
    gtf_file=args.gtf_file
    print(gtf_file)
    output_file=args.output
    print(output_file)
    junc_bed_file=args.junc_bed_file
    print(junc_bed_file)
    min_junc_counts=args.min_junc_counts
    print(min_junc_counts)
    main(gtf_file, junc_bed_file, output_file, min_junc_counts)

# to test run 
#gtf_file=/gpfs/commons/groups/knowles_lab/Karin/genome_files/Homo_sapiens.GRCh38.108.chr.gtf
#gtf_file="/gpfs/commons/groups/knowles_lab/Karin/genome_files/Homo_sapiens.GRCh38.108.chr.gtf"
#output_file=/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/clustered_junctions_minjunccounts
#junc_bed_file=/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/clustered_filtered_junctions.bed
#min_junc_counts=10000

#python src/clustering/intron_clustering_GTEx.py --gtf_file $gtf_file --output_file $output_file --junc_bed_file $junc_bed_file --min_junc_counts $min_junc_counts
