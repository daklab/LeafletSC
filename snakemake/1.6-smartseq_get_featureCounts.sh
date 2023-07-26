#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J SSfeatureCounts # <-- name of job
#SBATCH --array=1-120 # <-- number of jobs to run (number of tissue-cell type pairs)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load subread

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues
gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls -d */ | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE

output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/featureCounts

featureCounts -T 12 -a $gtf_file -p -B -C -o $output_dir/${TISSUE}_counts.txt ${TISSUE}/*.Aligned.out.sorted.CB.bam --verbose
