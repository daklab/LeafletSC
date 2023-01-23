leafcutter-sc
==============================

Alternative splicing quantification in single cells

Project Organization
------------

How to run leafcutter-sc? 

1. First, obtain pseudo-bulk BAM files that represent known cell types. If cell types are not well defined, cluster your single cells based on gene expression and save cell labels. Seperate your cells into those clusters such that there is one BAM file for each cluster label. 

2. Run the snakemake pipeline using the scripts in [snakemake], change all paths and parameters to match your setup. The pipeline will output several useful files but the main steps include:
   - Realigning BAM file so that the -XS strand is present in the BAM file to indicate important strand information
   - Deduplicating UMIs in BAM file + Indexing BAM file
   - Calculate UMI counts across individual cells and genes 
   - Extract junctions using regtools while maintaining single cell information