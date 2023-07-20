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
module load kallisto

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues
gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/genes/genes.gtf

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls -d */ | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE

output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/kallisto

# First need to generate the index 
#sbatch --wrap "kallisto index -i /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/kallisto/index/transcriptome_index.idx /gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/fasta/genome.fa" --mem 64G    

# Loop over each tissue and run kallisto
for TISSUE in "${TISSUE_LIST[@]}"; do
    echo "Processing tissue: $TISSUE"
    
    output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/kallisto/${TISSUE}_counts
    mkdir -p $output_dir
    
    kallisto quant -i $index_file -o $output_dir -b 100 ${TISSUE}/*.Aligned.out.sorted.CB.bam
done
