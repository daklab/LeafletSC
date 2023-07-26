#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p dev
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-6:00 # Runtime in D-HH:MM
#SBATCH -J SSregtools # <-- name of job
#SBATCH --array=1-35 # <-- number of microglia batches

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

input_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/PseudoBulk/Brain_Myeloid_microglial_cell"

cd $input_dir

# one job will run for every pseudobulk batch
TISSUE_LIST=($(ls *.bam | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}

echo $TISSUE
input_bam=$TISSUE

output_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/Leaflet/junctions"

tissue_type="Brain_Myeloid_microglial_cell"

# Save tissue without *.bam extension Trachea_mesenchymal_cell_pseudobulk.bam
batch=$(echo $TISSUE | awk -F. '{print $1}')

# Set the output pseudobulk BAM file name
output_file="${output_dir}/${tissue_type}.${batch}.juncs"
output_barcodes="${output_dir}/${tissue_type}.${batch}.barcodes"
output_juncswbarcodes="${output_dir}/${tissue_type}.${batch}.juncswbarcodes"

# otherwise run regtools if file doesn't exist
echo Extracting junctions with regtools!

regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools
$regtools_run junctions extract -a 6 -m 50 -M 500000 $input_bam -o $output_file -s XS -b $output_barcodes
paste --delimiters='\t' $output_file $output_barcodes > $output_juncswbarcodes

echo Done!