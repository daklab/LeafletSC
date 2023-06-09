#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J pseudobulks # <-- name of job
#SBATCH --array=1-20 # <-- number of jobs to run (number of tissues)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

input_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues"

cd $input_dir

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls -d */ | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE

output_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/PseudoBulk"

# Set the output pseudobulk BAM file name
output_file="${output_dir}/${TISSUE}_pseudobulk.bam"

# Check if the output file already exists
if [ -f ${output_file} ]; then
    echo "Output file ${output_file} already exists. Skipping..."
    exit 0
fi

# Merge all the BAM files for the current tissue into a single BAM file
#samtools merge -f ${output_file} ${input_dir}/${TISSUE}/*.sorted.CB.bam
samtools merge ${output_file} ${input_dir}/${TISSUE}/*.sorted.CB.bam

echo "Merged ${TISSUE} BAM files into ${output_file}"

# Index the new BAM file
#samtools index ${output_file}

sambamba index ${output_file}
echo "Indexed ${output_file}"