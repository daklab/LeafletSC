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
    and filter out rare junction events 
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

        #remove introns longer than 500kb for now (these should be a parameter that can be changed by user)
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
    If either end has less than threshold_inc pecentage of the reads then consider removing cluster 
    if considering canonical setting. In cryptic case, it could be a real rare event? 
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
    gr = pr.from_dict({"Chromosome": df["chrom"], "Start": df["chromStart"], "End": df["chromEnd"], "Strand": df["strand"], "Cell": df["file_name"], "jutions]$ 

]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ echo %g[Kr[K[K$gtf_file

]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ gtf_file=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/longread/SRR9944890/collapse/SRR9944890_isoform_g 
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ junc_files=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ echo $setting

]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ setting=canonical
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ echo setting
setting
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ echo settingsetting=canonical"canonical[C[C[C[C[C[C[C[C[C"
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ setting="canonical"[7Pecho setting
setting
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ output=test.txt
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ output=test.txt[3Pecho settingsetting="canonical"[7Pecho settingsetting=canonical[4Pecho $settingjunc_files=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions
/gpfs/commons/home/kisaev/leafcutter-sc/src/clustering/intron_clustering.py
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ echo $scriptoutput=test.txt[3Pecho settingsetting="canonical"[7Pecho settingsetting=canonical[4Pecho $settingjunc_files=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions

  File "/gpfs/commons/home/kisaev/leafcutter-sc/src/clustering/intron_clustering.py", line 1
    Script started on Mon 23 Jan 2023 04:06:59 PM EST
                 ^
SyntaxError: invalid syntax
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ python $script $junc_files $gtf_file $setting $output
usage: intron_clustering.py [-h] [--junc_files JUNC_FILES] [--setting SETTING]
                            [--gtf_file GTF_FILE] [--output_file OUTPUT]
intron_clustering.py: error: unrecognized arguments: /gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions /gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/longread/SRR9944890/collapse/SRR9944890_isoform_gid.gtf canonical test.txt
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ python $script $junc_files $gtf_file $setting $output[1@-[1@-[1@j[1@u[1@n[1@c[1@_[1@f[1@i[1@l[1@e[1@s[1@ [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1@-[1@-[1@g[1@t[1@f[1@_[1@f[1@i[1@l[1@e[1@ [C[C[C[C[C[C[C[C[C[C[1@-[1@-[1@s[1@e[1@t[1@t[1@i[1@n[1@g[1@ [C[C[C[C[C[C[C[C[C-$output-$outputi$outputu$outputt$output[1P$output[1P$output[1P$outputo$outputu$outputt$outputp$outputu$outputt$output_$outputf$outputi$outputl$outpute$output
usage: intron_clustering.py [-h] [--junc_files JUNC_FILES] [--setting SETTING]
                            [--gtf_file GTF_FILE] [--output_file OUTPUT]
intron_clustering.py: error: unrecognized arguments: --output_filetest.txt
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ python $script --junc_files $junc_files --gtf_file $gtf_file --setting $setting --output_file$output $output
/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/longread/SRR9944890/collapse/SRR9944890_isoform_gid.gtf
/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions
canonical
test.txt
/gpfs/commons/home/kisaev/miniconda3/envs/myenv/lib/python3.7/site-packages/gtfparse/read_gtf.py:154: FutureWarning: The error_bad_lines argument has been deprecated and will be removed in a future version.


  features=features)
/gpfs/commons/home/kisaev/miniconda3/envs/myenv/lib/python3.7/site-packages/gtfparse/read_gtf.py:154: FutureWarning: The warn_bad_lines argument has been deprecated and will be removed in a future version.


  features=features)
INFO:root:Extracted GTF attributes: ['gene_id', 'transcript_id', 'exon_number']
/gpfs/commons/home/kisaev/leafcutter-sc/src/clustering/intron_clustering.py:138: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  gtf_exons['seqname'] = gtf_exons['seqname'].map(lambda x: x.lstrip('chr').rstrip('chr'))
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/DC.wbarcode.junc
The number of junctions found is 53023
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/B.wbarcode.junc
The number of junctions found is 113032
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/IGHA.wbarcode.junc
The number of junctions found is 31501
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/CD8T.wbarcode.junc
The number of junctions found is 88070
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/NK.wbarcode.junc
The number of junctions found is 117124
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/FCGR3A.wbarcode.junc
The number of junctions found is 90929
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/CD14Mono.wbarcode.junc
The number of junctions found is 61356
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/NaiveCD4T.wbarcode.junc
The number of junctions found is 101279
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/ZNF385D.wbarcode.junc
The number of junctions found is 3935
reading the junction file/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/MemoryCD4T.wbarcode.junc
The number of junctions found is 114036
done collecting junctions
The number of junctions prior to assessing distance to exons is 150958
The number of junctions after assessing distance to exons is 19425
/gpfs/commons/home/kisaev/leafcutter-sc/src/clustering/intron_clustering.py:174: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  clusters_wnomultiple_events=clusters_df.groupby(['Cluster'])['Chromosome', 'Start', 'End'].nunique().reset_index()
The number of clusters to be initially evaluated is 1318

/gpfs/commons/home/kisaev/leafcutter-sc/src/clustering/intron_clustering.py:194: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  clusters_wnomultiple_events=clusters_df.groupby(['Cluster'])['Chromosome', 'Start', 'End'].nunique().reset_index()
                junction_id  Cluster
1132   17_46929584_46931098        2
1134   17_46929584_46935028        2
1135   17_46931207_46932066        2
1136   17_46932199_46935028        2
1223     17_7574308_7574545        3
1224     17_7574308_7575118        3
1225     17_7574678_7575118        3
1226     17_7574678_7576523        3
1227     17_7575258_7576523        3
2071   21_33252830_33276595        2
2537  3_134495360_134507108        2
2538  3_134495364_134507108        2
The number of clusters to be finally evaluated is 1145
0:01:15.865468
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ ls
B.barcodes                              [0m[01;31mFCGR3A.junction.leafcutter.sorted.gz[0m      NK.junction
B.junction                              FCGR3A.wbarcode.junc                      [01;31mNK.junction.leafcutter.sorted.gz[0m
[01;31mB.junction.leafcutter.sorted.gz[0m         groups_file.txt                           NK.wbarcode.junc
B.wbarcode.junc                         IGHA.barcodes                             [01;31mout.txt.gz[0m
CD14Mono.barcodes                       IGHA.junction                             [01;31mPBMC_intron_clusters_bygene_out.txt.gz[0m
CD14Mono.junction                       [01;31mIGHA.junction.leafcutter.sorted.gz[0m        [01;35mPBMC_pseudobulk_mapped_reads.jpg[0m
[01;31mCD14Mono.junction.leafcutter.sorted.gz[0m  IGHA.wbarcode.junc                        PBMC_pseudobulk_mapped_reads.pdf
CD14Mono.wbarcode.junc                  [01;31minputfile_for_betabino_mixedmodel.txt.gz[0m  [1;35mscanpy_res[0m
CD8T.barcodes                           [01;31mleafcutter_perind.counts.gz[0m               stdout_28750310.log
CD8T.junction                           [01;31mleafcutter_perind_numers.counts.gz[0m        stdout_28753157.log
[01;31mCD8T.junction.leafcutter.sorted.gz[0m      leafcutter_pooled                         test.bed
CD8T.wbarcode.junc                      leafcutter_refined                        test_cluster.txt
[1;35mClusters_ZI_results[0m                     leafcutter_sortedlibs                     test_juncfiles.txt
DC.barcodes                             MemoryCD4T.barcodes                       [01;35mtest_nb.png[0m
DC.junction                             MemoryCD4T.junction                       [01;35mtest.png[0m
[01;31mDC.junction.leafcutter.sorted.gz[0m        [01;31mMemoryCD4T.junction.leafcutter.sorted.gz[0m  test.txt
DC.wbarcode.junc                        MemoryCD4T.wbarcode.junc                  var_params_121422.pkl
error_28750310.log                      NaiveCD4T.barcodes                        ZNF385D.barcodes
error_28753157.log                      NaiveCD4T.junction                        ZNF385D.junction
extracted_junction_coordinates.bed      [01;31mNaiveCD4T.junction.leafcutter.sorted.gz[0m   [01;31mZNF385D.junction.leafcutter.sorted.gz[0m
FCGR3A.barcodes                         NaiveCD4T.wbarcode.junc                   ZNF385D.wbarcode.junc
FCGR3A.junction                         NK.barcodes
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ ls -lt
total 1971716
-rw-rw-r-- 1 kisaev dklab  20124178 Jan 23 16:13 test.txt
-rw-rw-r-- 1 kisaev dklab    188630 Jan 23 16:13 extracted_junction_coordinates.bed
-rw-rw-r-- 1 kisaev dklab 100641449 Jan 23 15:13 NaiveCD4T.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  64800129 Jan 23 15:13 NaiveCD4T.barcodes
-rw-rw-r-- 1 kisaev dklab  35841320 Jan 23 15:13 NaiveCD4T.junction
-rw-rw-r-- 1 kisaev dklab 148830417 Jan 23 15:13 NK.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  99094212 Jan 23 15:13 NK.barcodes
-rw-rw-r-- 1 kisaev dklab  49736205 Jan 23 15:13 NK.junction
-rw-rw-r-- 1 kisaev dklab 134090446 Jan 23 15:13 MemoryCD4T.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  88787270 Jan 23 15:13 MemoryCD4T.barcodes
-rw-rw-r-- 1 kisaev dklab  45303176 Jan 23 15:13 MemoryCD4T.junction
-rw-rw-r-- 1 kisaev dklab  73079611 Jan 23 15:08 CD8T.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  44611846 Jan 23 15:08 CD8T.barcodes
-rw-rw-r-- 1 kisaev dklab  28467765 Jan 23 15:08 CD8T.junction
-rw-rw-r-- 1 kisaev dklab  30144265 Jan 23 15:08 DC.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  14535822 Jan 23 15:08 DC.barcodes
-rw-rw-r-- 1 kisaev dklab  15608443 Jan 23 15:08 DC.junction
-rw-rw-r-- 1 kisaev dklab  84672632 Jan 23 15:08 FCGR3A.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  53828768 Jan 23 15:08 FCGR3A.barcodes
-rw-rw-r-- 1 kisaev dklab  30843864 Jan 23 15:08 FCGR3A.junction
-rw-rw-r-- 1 kisaev dklab 134630631 Jan 23 15:08 B.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  86168387 Jan 23 15:08 B.barcodes
-rw-rw-r-- 1 kisaev dklab  48462244 Jan 23 15:08 B.junction
-rw-rw-r-- 1 kisaev dklab  38903902 Jan 23 14:58 CD14Mono.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab  20767508 Jan 23 14:58 CD14Mono.barcodes
-rw-rw-r-- 1 kisaev dklab  18136394 Jan 23 14:58 CD14Mono.junction
-rw-rw-r-- 1 kisaev dklab   4883981 Jan 23 14:57 ZNF385D.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab   1196812 Jan 23 14:57 ZNF385D.barcodes
-rw-rw-r-- 1 kisaev dklab   3687169 Jan 23 14:57 ZNF385D.junction
-rw-rw-r-- 1 kisaev dklab  17798432 Jan 23 14:57 IGHA.wbarcode.junc
-rw-rw-r-- 1 kisaev dklab   7332974 Jan 23 14:57 IGHA.barcodes
-rw-rw-r-- 1 kisaev dklab  10465458 Jan 23 14:57 IGHA.junction
-rw-rw-r-- 1 kisaev dklab  19618278 Jan 23 09:14 [0m[01;31minputfile_for_betabino_mixedmodel.txt.gz[0m
drwxrwsr-x 2 kisaev dklab      4096 Jan 20 16:46 [1;35mscanpy_res[0m
-rw-rw-r-- 1 kisaev dklab   5419744 Jan 20 15:09 [01;31mPBMC_intron_clusters_bygene_out.txt.gz[0m
-rw-rw-r-- 1 kisaev dklab  17341077 Jan 12 17:57 test.bed
-rw-rw-r-- 1 kisaev dklab      1835 Jan 11 11:53 test_cluster.txt
-rw-rw-r-- 1 kisaev dklab  97302737 Dec 15 22:48 var_params_121422.pkl
drwxrwsr-x 3 kisaev dklab   2097152 Oct 17 13:07 [1;35mClusters_ZI_results[0m
-rw-rw-r-- 1 kisaev dklab     63313 Oct 14 12:43 error_28753157.log
-rw-rw-r-- 1 kisaev dklab       333 Oct 14 12:33 stdout_28753157.log
-rw-rw-r-- 1 kisaev dklab     48018 Oct 14 12:27 error_28750310.log
-rw-rw-r-- 1 kisaev dklab       333 Oct 14 12:20 stdout_28750310.log
-rw-rw-r-- 1 kisaev dklab 305564248 Sep 28 12:41 [01;31mout.txt.gz[0m
-rw-rw-r-- 1 kisaev dklab     19796 Aug 22 17:32 [01;35mtest_nb.png[0m
-rw-rw-r-- 1 kisaev dklab     12282 Aug 22 17:18 [01;35mtest.png[0m
-rw-rw-r-- 1 kisaev dklab    310858 Aug 18 15:48 [01;31mleafcutter_perind_numers.counts.gz[0m
-rw-rw-r-- 1 kisaev dklab    513590 Aug 18 15:48 [01;31mleafcutter_perind.counts.gz[0m
-rw-rw-r-- 1 kisaev dklab      1193 Aug 18 15:48 leafcutter_sortedlibs
-rw-rw-r-- 1 kisaev dklab    129902 Aug 18 15:48 [01;31mZNF385D.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    162201 Aug 18 15:48 [01;31mNK.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    159025 Aug 18 15:48 [01;31mNaiveCD4T.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    161557 Aug 18 15:48 [01;31mMemoryCD4T.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    143534 Aug 18 15:48 [01;31mIGHA.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    156448 Aug 18 15:48 [01;31mFCGR3A.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    147921 Aug 18 15:48 [01;31mDC.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    156231 Aug 18 15:48 [01;31mCD8T.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    149888 Aug 18 15:48 [01;31mCD14Mono.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    161576 Aug 18 15:48 [01;31mB.junction.leafcutter.sorted.gz[0m
-rw-rw-r-- 1 kisaev dklab    379588 Aug 18 15:48 leafcutter_refined
-rw-rw-r-- 1 kisaev dklab   8582978 Aug 18 15:48 leafcutter_pooled
-rw-rw-r-- 1 kisaev dklab       973 Aug 18 15:47 test_juncfiles.txt
-rw-rw-r-- 1 kisaev dklab    203966 Aug 11 19:55 [01;35mPBMC_pseudobulk_mapped_reads.jpg[0m
-rw-rw-r-- 1 kisaev dklab      4977 Aug  8 10:24 PBMC_pseudobulk_mapped_reads.pdf
-rw-rw-r-- 1 kisaev dklab        78 Jul 25 13:24 groups_file.txt
]0;kisaev@pe2dc5-005:/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions(myenv) [kisaev@pe2dc5-005 junctions]$ less test.txt
[?1049h[?1h=
1_23030634_23044426}1}ENSG00000004487.18}1}23030634}23044426}JUNC000092830}26}+}23030523}23044486}255,0,0}2}111,60}0,13903}19}111}60}13792}['39_42_15:1', '13_ 74_48:1', '43_18_35:1', '34_95_94:2', '33_26_11:2', '40_27_21:3', '1_20_70:1', '45_53_30:2', '12_77_35:2', '34_54_3:1', '27_42_31:1', '5_35_27:1', '13_39_52:2 ', '10_41_64:1', '38_15_91:1', '34_19_50:1', '32_74_92:1', '32_92_93:1', '16_65_51:1']}B
1_23030634_23044426}1}ENSG00000004487.18}1}23030634}23044426}JUNC000056282}10}+}23030529}23044488}255,0,0}2}105,62}0,13897}8}105}62}13792}['30_25_80:1', '34_4 3_88:1', '16_87_58:1', '12_63_81:2', '41_6_54:1', '22_10_91:1', '10_20_7:1', '9_14_52:2']}CD8T
1_23030634_23044426}1}ENSG00000004487.18}1}23030634}23044426}JUNC000099191}22}+}23030528}23044489}255,0,0}2}106,63}0,13898}19}106}63}13792}['17_69_94:1', '42_ 39_46:1', '34_35_62:1', '37_4_24:1', '43_7_41:1', '38_13_84:2', '23_7_15:1', '32_92_55:1', '14_74_96:1', '22_90_65:1', '38_35_38:1', '17_56_59:1', '43_74_48:2 ', '11_12_79:1', '35_71_18:1', '45_13_69:1', '16_90_20:1', '35_17_5:1', '34_21_48:2']}NK
1_23030634_23044426}1}ENSG00000004487.18}1}23030634}23044426}JUNC000070511}28}+}23030523}23044487}255,0,0}2}111,61}0,13903}16}111}61}13792}['15_68_9:1', '17_7 _4:1', '16_57_24:1', '48_65_48:3', '9_35_91:3', '43_56_12:4', '17_52_94:1', '31_5_78:1', '15_30_36:3', '38_71_1:3', '31_6_28:1', '41_64_67:1', '39_52_85:1', ' 47_89_8:1', '34_92_66:2', '37_12_29:1']}NaiveCD4T
1_23030634_23044426}1}ENSG00000004487.18}1}23030634}23044426}JUNC000089672}33}+}23030523}23044488}255,0,0}2}111,62}0,13903}17}111}62}13792}['33_49_83:2', '16_ 41_46:6', '14_81_81:3', '16_36_80:1', '10_14_72:1', '5_73_95:1', '32_85_77:2', '18_74_92:2', '34_38_47:1', '19_57_14:2', '9_13_68:1', '1_27_35:1', '22_65_43:6 ', '1_4_64:1', '2_91_12:1', '32_95_60:1', '35_69_38:1']}MemoryCD4T
1_23030634_23050386}1}ENSG00000004487.18}1}23030634}23050386}JUNC000092840}27}+}23030525}23050480}255,0,0}2}109,94}0,19861}15}109}94}19752}['21_81_9:1', '44_3 9_29:1', '43_17_67:1', '13_31_83:1', '43_2_76:2', '27_38_61:1', '43_94_81:2', '15_85_6:3', '40_95_79:3', '30_42_47:2', '34_57_35:1', '27_61_53:2', '3_82_39:3' , '41_41_87:2', '31_33_35:2']}B
1_23030634_23050386}1}ENSG00000004487.18}1}23030634}23050386}JUNC000099180}31}+}23030523}23050495}255,0,0}2}111,109}0,19863}22}111}109}19752}['16_40_34:2', '1 4_86_86:1', '13_2_33:2', '9_24_57:1', '34_40_70:1', '13_60_11:3', '40_45_34:2', '22_64_18:1', '41_23_36:1', '23_62_41:1', '34_71_89:2', '17_52_61:1', '31_89_8 8:1', '46_80_11:1', '25_73_42:1', '38_54_25:3', '25_80_45:1', '40_72_69:1', '1_52_57:1', '29_84_15:2', '25_87_45:1', '36_11_37:1']}NK
1_23030634_23050386}1}ENSG00000004487.18}1}23030634}23050386}JUNC000070503}28}+}23030523}23050497}255,0,0}2}111,111}0,19863}16}111}111}19752}['32_85_8:1', '33 [7mtest.txt[27m[K
]0;kisaev@pe2dc5-005:~(myenv) [kisaev@pe2dc5-005 ~]$ cd leafcutter-sc
]0;kisaev@pe2dc5-005:~/leafcutter-sc(myenv) [kisaev@