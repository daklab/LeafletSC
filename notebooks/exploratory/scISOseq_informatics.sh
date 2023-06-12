module load samtools 
module load sambamba

dir=/gpfs/commons/groups/knowles_lab/Karin/data/isoform_cortex_longread_SC
cd $dir 

#[1.] first need to sort the BAM files that Anoushka gave me (both ONT) aligned with minimap2

#[2.] PCDH exons we are interested in are on chromosome 5 
samtools view -b Run1.sorted.bam "chr5" > Run1.sorted.chr5.bam
sambamba index  Run1.sorted.chr5.bam

#[3.] saf file I made using Erin's exon coordinates 
fas=/gpfs/commons/groups/knowles_lab/Karin/data/isoform_cortex_longread_SC/PCDH_gene_clusters_exon_coords.saf

#[5.] want to look at exon expression across cell types! 

#[4.] run FeatureCounts (this is in total)
featureCounts -t exon -g gene_id -a PCDH_gene_clusters_exon_coords.saf \
-o Run1_PCDH_exons_counts.txt Run1.sorted.chr5.bam -F SAF -L -O -M --verbose -s 1

# isoquant also does exon with a custom reference file 
# how to figure which cell type is which cell barcode coming from? 