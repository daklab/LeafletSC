#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J merge_BAM # <-- name of job
#SBATCH --array=1-44950 # <-- number of jobs to run (number of unique cell ID samples)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

# Get list of bam files to run array jobs
cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq
output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/merged_bams

# Define the directory containing the BAM files
BAM_DIR=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/facs_bam_files

# Read in the list of sample IDs from the file
SAMPLE_LIST=($(cut -f1 /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/d8a0c006-2991-5ddb-8ed8-7b7e8c187fe3/all_tabula_muris_samples.txt))

# Get the current sample ID from the array job index
SAMPLE_ID=${SAMPLE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $SAMPLE_ID

# Define the filenames for the R1 and R2 BAM files
R1_FILE="${SAMPLE_ID}_R1.mus.Aligned.out.sorted.bam"
R2_FILE="${SAMPLE_ID}_R2.mus.Aligned.out.sorted.bam"

# Check if both R1 and R2 BAM files exist in the directory
if [ -f "${BAM_DIR}/${R1_FILE}" ] && [ -f "${BAM_DIR}/${R2_FILE}" ]; then
  # Merge the R1 and R2 files
  MERGED_FILE="${SAMPLE_ID}_merged.mus.Aligned.out.sorted.bam"
  samtools merge "${output_dir}/${MERGED_FILE}" "${BAM_DIR}/${R1_FILE}" "${BAM_DIR}/${R2_FILE}"
  samtools sort "${output_dir}/${MERGED_FILE}" -o "${output_dir}/${MERGED_FILE%.bam}.sorted.bam"
  samtools index "${output_dir}/${MERGED_FILE%.bam}.sorted.bam"
  echo "Merged ${R1_FILE} and ${R2_FILE} into ${MERGED_FILE%.bam}.sorted.bam"
  echo "Removing intermediate file"
  rm "${output_dir}/${MERGED_FILE}"
elif [ -f "${BAM_DIR}/${R1_FILE}" ]; then
  # Only R1 file exists, move it to the output directory
  mv "${BAM_DIR}/${R1_FILE}" "${output_dir}" #forgot to move the .bai files also so need to remake index files 
  echo "Moved ${R1_FILE} to ${output_dir}"
  # Also move the index! 
  mv "${BAM_DIR}/${R1_FILE}.bai" "${output_dir}"
else
  # Neither R1 nor R2 file exists
  echo "Could not find both ${R1_FILE} and ${R2_FILE} in ${BAM_DIR}"
fi

# post running script fix R1 BAM files (those without replicates, i forgot to move their index files into the merged folder)
#for file in *R1*; do
#  echo "Indexing $file"
#  samtools index $file
#done

# ah sigh doing this the second time around I moved the original file to the merge folder and
# so they are no longer in the facs_bam_files folder...
# the R1 non merged files were already in tissue subdirectories so i moved all those into 
# merged folder manually 

