leafcutter-sc
==============================

Alternative splicing quantification in single cells

Project Organization
------------

How to run leafcutter-sc? 

1. First, obtain pseudo-bulk BAM files that represent known cell types. If cell types are not well defined, cluster your single cells based on gene expression and save cell labels. Seperate your cells into those clusters such that there is one BAM file for each cluster label. 

2. Run the snakemake pipeline using the scripts in [snakemake](snakemake), change all paths and parameters to match your setup. The pipeline will output several useful files but the main steps include:
   - Realigning BAM file so that the -XS strand is present in the BAM file to indicate important strand information
   - Deduplicating UMIs in BAM file + Indexing BAM file
   - Calculate UMI counts across individual cells and genes 
   - Extract junctions using regtools while maintaining single cell information

3. Cluster your junctions to obtain intron cluster events that can be further analyzed for differential splicing. The script for this is found [here](src/clustering/intron_clustering.py). You will need to define the following parameters:
   - A path containing your junction files produced in step 2 
   - A path to a gtf that you want to use to asign genes to junctions (make sure it's the same genome build as what was used during alignment)
   - Whether you want to focus on events that are most likely canonical rather than cryptic or rare 
   - How you want to name your output file 

4. Run a beta-binomial LDA model to extract cell states and junctions that may be differentially spliced between them. To run this, first prepare an input file from your output in step 3 by running the script in [here](src/clustering/)