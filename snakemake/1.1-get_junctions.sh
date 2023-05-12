#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J JUNCS # <-- name of job
#SBATCH --array=1-44950 # <-- number of jobs to run (number of unique cell ID samples)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

# Get list of bam files to run array jobs
cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq
output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/junctions

# Define the directory containing the BAM files
BAM_DIR=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/merged_bams
cd $BAM_DIR 

#find . -name "*.bam" > all_merged_BAM_tabula_muris.txt

# Read in the list of sample IDs from the file
SAMPLE_LIST=($(cut -f1 all_merged_BAM_tabula_muris.txt))

# Get the current sample ID from the array job index
SAMPLE_ID=${SAMPLE_LIST[$SLURM_ARRAY_TASK_ID - 1]}

sample_id=$(echo $SAMPLE_ID | sed 's/^\.\/\(.*\)_merged\.mus.*/\1/')
echo $sample_id

regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools
$regtools_run junctions extract -a 6 -m 50 -M 500000 $SAMPLE_ID -o $output_dir/$sample_id.juncs -s XS

