#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH --mem=5000M
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -J cell_barcodes # <-- name of job
#SBATCH --array=1-44950 # <-- number of jobs to run (number of unique cell ID samples)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

# Get list of bam files to run array jobs
cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/merged_bams
output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues

# Define the directory containing the BAM files
BAM_DIR=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues

# Read in the list of sample IDs from the file
SAMPLE_LIST=($(cut -f1 /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/d8a0c006-2991-5ddb-8ed8-7b7e8c187fe3/all_tabula_muris_samples_w_celltype_column.txt))

# Read in the tissue that they belong to 
TISSUE_CELL_LIST=($(cut -f7 /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/d8a0c006-2991-5ddb-8ed8-7b7e8c187fe3/all_tabula_muris_samples_w_celltype_column.txt))

# Get the current sample ID from the array job index
SAMPLE_ID=${SAMPLE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $SAMPLE_ID

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_CELL_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE

# Define the filenames for the R1 and R2 BAM files
SAMPLE_BAM_merged="$BAM_DIR/$TISSUE/${SAMPLE_ID}_merged.mus.Aligned.out.sorted.sorted.bam"
SAMPLE_BAM_R1="$BAM_DIR/$TISSUE/${SAMPLE_ID}_R1.mus.Aligned.out.sorted.bam" #these were the samples that didn't have replicated and their original names were kept 

# Check if the merged BAM file is non-empty and add cell barcodes 
if [[ -s "$SAMPLE_BAM_merged" ]]; then
    echo "The merged BAM file is non-empty."
    # add barcode -CB to BAM file with SAMPLE_ID to represent individual cell barcode
    samtools view -h $SAMPLE_BAM_merged | awk -v cb=$SAMPLE_ID -F '\t' 'BEGIN {OFS="\t"} {$NF = $NF"\tCB:Z:"cb; print}' | samtools view -bS - > $output_dir/$TISSUE/${SAMPLE_ID}_merged.mus.Aligned.out.sorted.CB.bam
    # index this new file 
    samtools index $output_dir/$TISSUE/${SAMPLE_ID}_merged.mus.Aligned.out.sorted.CB.bam
    echo "CB tag added to reads!."
else
    echo "The merged BAM file is empty or does not exist."
fi

# Check if the R1 BAM file is non-empty
if [[ -s "$SAMPLE_BAM_R1" ]]; then
    echo "The R1 BAM file is non-empty."
    # add barcode -CB to BAM file with SAMPLE_ID to represent individual cell barcode
    samtools view -h $SAMPLE_BAM_R1 | awk -v cb=$SAMPLE_ID -F '\t' 'BEGIN {OFS="\t"} {$NF = $NF"\tCB:Z:"cb; print}' | samtools view -bS - > $output_dir/$TISSUE/${SAMPLE_ID}_merged.mus.Aligned.out.sorted.CB.bam
    # index this new file 
    samtools index $output_dir/$TISSUE/${SAMPLE_ID}_merged.mus.Aligned.out.sorted.CB.bam
    echo "CB tag added to reads!."
else
    echo "The R1 BAM file is empty or does not exist."
fi

