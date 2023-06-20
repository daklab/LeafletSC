#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J SSregtools # <-- name of job
#SBATCH --array=1-120 # <-- number of jobs to run (number of tissue-cell type pairs)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

input_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/PseudoBulk"

cd $input_dir

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls *.bam | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE
input_bam=$TISSUE
output_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/Leaflet/junctions"

# Save tissue without *.bam extension Trachea_mesenchymal_cell_pseudobulk.bam
TISSUE=$(echo $TISSUE | awk -F. '{print $1}')

# Set the output pseudobulk BAM file name
output_file="${output_dir}/${TISSUE}.juncs"
output_barcodes="${output_dir}/${TISSUE}.barcodes"
output_juncswbarcodes="${output_dir}/${TISSUE}.juncswbarcodes"
echo Checking if junctions exist!

#check if output_juncswbarcodes exists and if so, skip
if [ -f $output_juncswbarcodes ]; then
    echo "File $output_juncswbarcodes exists, skipping"
    exit 0
fi

# otherwise run regtools if file doesn't exist
echo Extracting junctions with regtools!

regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools
$regtools_run junctions extract -a 6 -m 50 -M 500000 $input_bam -o $output_file -s XS -b $output_barcodes
paste --delimiters='\t' $output_file $output_barcodes > $output_juncswbarcodes