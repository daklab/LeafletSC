#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 2 
#SBATCH --mem=45000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J PBMC_parse

module load samtools 
module load deeptools 
module load sambamba

cd /gpfs/commons/groups/knowles_lab/Karin/genome_files/

out_dir=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/deeptools/trans_lengths
genome_file=/gpfs/commons/groups/knowles_lab/Karin/genome_files/bed_files_transcripts.txt

# get bed file for current array
bed_files=$genome_file #via ls *.bed > bed_files_transcripts.txt 
names=($(cat $bed_files))

# full PBMC parse file? figure out when/how i made this file? 
bam_input=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/bam_process/bam_splits/PBMC_parse_full_dedup.bam
sample=PBMC_ParseBio

#1. get bigwig file (default --normalizeUsing == "None") should I increase the bin size? should I run this on one strand at a time? --filterRNAstrand with forward or reverse 
bamCoverage -b $bam_input -o $out_dir/$sample.bw --verbose --binSize 300 --numberOfProcessors "max" 
bamCoverage -b $bam_input -o $out_dir/$sample.bedgraph --verbose --binSize 300 --numberOfProcessors "max" --outFileFormat bedgraph

#2. get computeMatrix (for now, don't skip zeroes)
# use for loop to cycle through bed files and run computeMatrix and plotProfile for each one

for file_num in {0..5}; do
    echo $file_num
    bed_file=${names[$file_num]} 
    echo $bed_file
    bed_name=$(echo $bed_file | cut -f 1 -d '.')
    echo $bed_name
    computeMatrix scale-regions -S $out_dir/$sample.bw \
                                --regionBodyLength 500 --verbose \
                              --afterRegionStartLength 0 --numberOfProcessors "max" \
                            -o $out_dir/matrix_${sample}_${bed_name}.gz --outFileNameMatrix $out_dir/matrix_${sample}_${bed_name}.tab \
                            --outFileSortedRegions $out_dir/matrix_${sample}_${bed_name}.bed \
                            -R $bed_file \
                            --beforeRegionStartLength 0 

    plotProfile -m $out_dir/matrix_${sample}_${bed_name}.gz \
              -out $out_dir/${sample}_${bed_name}_transcript_coverage.png --startLabel "5'" --endLabel "3'" --plotType fill

    echo "done with $bed_name"
done
