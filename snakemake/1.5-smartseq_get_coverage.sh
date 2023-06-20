#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J SSDeeptools # <-- name of job
#SBATCH --array=1-120 # <-- number of jobs to run (number of tissue-cell type pairs)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba
module load deeptools 

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/PseudoBulk
gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/genes/genes.gtf
out_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/Leaflet/Deeptools

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls *.bam | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE
input_bam=$TISSUE
TISSUE=$(echo $TISSUE | awk -F. '{print $1}')

#1. get bigwig file (default --normalizeUsing == "None")

## should I increase the bin size? 
## should I run this on one strand at a time? --filterRNAstrand with forward or reverse 

#All reads
bamCoverage -b $input_bam -o $out_dir/$TISSUE.bw --verbose --binSize 20 --numberOfProcessors "max" 
bamCoverage -b $input_bam -o $out_dir/$TISSUE.bedgraph --verbose --binSize 20 --numberOfProcessors "max" --outFileFormat bedgraph

#2. get computeMatrix (for now, don't skip zeroes)

## all reads 
computeMatrix scale-regions -S $out_dir/$TISSUE.bw \
                                --regionBodyLength 1000 --verbose \
                              --afterRegionStartLength 0 --transcriptID transcript --numberOfProcessors "max" \
                            -o $out_dir/matrix_${TISSUE}.gz --outFileNameMatrix $out_dir/matrix_${TISSUE}.tab --outFileSortedRegions $out_dir/matrix_${TISSUE}.bed \
                            -R $gtf_file \
                              --beforeRegionStartLength 0 

#3. plot profile 

## all reads 
plotProfile -m $out_dir/matrix_${TISSUE}.gz \
              -out $out_dir/${TISSUE}_transcript_coverage.png --startLabel "5'" --endLabel "3'" --plotType fill