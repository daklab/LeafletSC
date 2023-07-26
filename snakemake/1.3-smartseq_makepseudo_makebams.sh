#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p bigmem
#SBATCH -c 2
#SBATCH --mem=1000G
#SBATCH -t 5-10:00 # Runtime in D-HH:MM
#SBATCH -J pseudobulks # <-- name of job
#SBATCH --array=1 # <-- number of jobs to run (number of tissue-cell type pairs)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

input_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues"

cd $input_dir

# Get a list of all the unique tissues that we have 
#TISSUE_LIST=($(ls -d */ | awk -F/ '{print $1}'))

# just need a list of two Brain_Non-Myeloid_oligodendrocyte and Brain_Myeloid_microglial_cell
TISSUE_LIST=(Brain_Myeloid_microglial_cell)

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
#samtools merge ${output_file} ${input_dir}/${TISSUE}/*.sorted.CB.bam

# use sambamba instead of samtools with increased threads and memory
# only combine 500 BAM files a time insude the tissue directory
# if tissue is Brain_Myeloid_microglial_cell make multiple pseudobulks 

#if $TISSUE == "Brain_Myeloid_microglial_cell"; then
# Use 'find' to get a list of all the sorted CB BAM files for the specified tissue

# Use 'find' to get a list of all the sorted CB BAM files for the specified tissue
find "${input_dir}/${TISSUE}" -maxdepth 1 -type f -name '*.sorted.CB.bam' | xargs -n 128 bash -c 'batch_output="batch_$$.bam"; sambamba merge "${batch_output}" "$@"; echo "Merged files: $@"' bash   

# just print out file names 
#find "${input_dir}/${TISSUE}" -maxdepth 1 -type f -name '*.sorted.CB.bam' | xargs -n 128 bash -c 'batch_output="batch_$$.bam"; echo "Merged files: $@"' bash   

# move the batch_*.bam files to the pseudobulk directory (these files will be run through regtools)

# for every BAM file in the tissues directory that starts wtih batch, index it 
#for bam_file in ${input_dir}/batch_*.bam; do
#    echo $bam_file
#    sambamba index $bam_file
#done

# i think sambamba automatically generates an index file

#sambamba merge ${output_file} ${input_dir}/${TISSUE}/*.sorted.CB.bam

#echo "Merged ${TISSUE} BAM files into ${output_file}"

# Index the new BAM file
#samtools index ${output_file}
#echo "Indexed ${output_file}"

#sbatch --wrap="sambamba merge ${output_file} ${input_dir}/${TISSUE}/*.sorted.CB.bam" --mem=600G

