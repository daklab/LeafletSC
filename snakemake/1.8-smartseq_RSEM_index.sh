#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J RSEM_index # <-- name of job

#load required modules
module purge                                                                                                                                                                         
#module load gcc/9.2.0 
#module load star/2.7.10b
#module load samtools
#module load sambamba
module load perl 

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/RSEM

# generate list of ALL bam files across tissues that end in *_merged.mus.Aligned.out.sorted.CB.bam
# find . -name "*_merged.mus.Aligned.out.sorted.CB.bam" > bam_list.txt

gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf
star_index=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/star2.7.10b
fasta_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/fasta/genome.fa

# Build an RSEM index for wrapping bowtie2 alignment to genome annotations

# $1 == path to gtf annotation file
# $2 == genome fasta file
# $3 == name of index, e.g. the genome name without the .fa/.fasta file extension
# The --bowtie2 argument tells RSEM to build a bowtie2 index for the alignment step.
# On Odyssey, loading the RSEM module also loads a compatible version of bowtie 2.

rsem-prepare-reference -p 12 --bowtie2 --gtf $gtf_file $fasta_file MM10PLUS