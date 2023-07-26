#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J RSEM_STAR # <-- name of job
#SBATCH --array=1-120 # <-- number of organs 

#load required modules
module purge                                                                                                                                                                         
#module load gcc/9.2.0 
#module load star/2.7.10b
#module load samtools
#module load sambamba
#module load perl 

module load gcc/9.2.0 
module load star/2.7.10b


# FINISH THIS!!!!

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/RSEM

SAMPLE_LIST=($(cat /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues/bam_list.txt))
SAMPLE_LIST=($(cat bam_list.txt))
output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/RSEM #make sure to create SJ_files folder within this directory once before submitting jobs

# Get the current tissue for the SAMPLE ID
CELL_BAM=${SAMPLE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $CELL_BAM

# Extract tissue name from the SAMPLE ID
TISSUE=$(echo $CELL_BAM | awk -F_ '{print $1}') 

# Only keep the part of TISSUE after './'
TISSUE=${TISSUE#./}

# Check if $output_dir/SJ_files/$TISSUE exists, if not, create it
if [ ! -d "${output_dir}/${TISSUE}" ]; then
    mkdir -p "${output_dir}/${TISSUE}"
fi

cellID=$(basename "$CELL_BAM" _merged.mus.Aligned.out.sorted.CB.bam)
echo $cellID

gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf
star_index=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/star2.7.10b
fasta_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/fasta/genome.fa
inputs=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/STAR/SJ_files

#--bam: Indicates that the input is in BAM format.
#--paired-end: Specifies that the data is paired-end.
#--no-bam-output: Prevents the generation of the abundance BAM file.
#--estimate-rspd: Estimates the read start position distribution.
#--append-names: Appends names to result files based on the input filename.

rsem-calculate-expression --bam --paired-end --no-bam-output --estimate-rspd --append-names --seed 12345 C9-D042105-3_11_M-1-1.Aligned.toTranscriptome.out.bam \
 /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/RSEM/ Diaphragm

# multiple BAM file inputs 
rsem-calculate-expression --bam --paired-end --no-bam-output --estimate-rspd --append-names --seed 12345 *.bam /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/RSEM/ Diaphragm
