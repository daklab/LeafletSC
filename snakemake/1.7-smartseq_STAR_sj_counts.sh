#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J SS_STAR # <-- name of job
#SBATCH --array=1-120 # <-- number of jobs to run (number of unique cell ID samples)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load star/2.7.10b

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues
gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/genes/genes.gtf
star_index=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/star2.7.10b

output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/STAR

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls -d */ | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE

#generate index first 
#star --runThreadN 4 --runMode genomeGenerate \
#    --genomeDir /gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/star2.7.10b \
#    --genomeFastaFiles /gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/fasta/genome.fa \
#    --sjdbGTFfile $gtf_file \
#    --sjdbOverhang 100 

for bam_file in ${TISSUE}/*.Aligned.out.sorted.CB.bam; do
    cellID=$(basename "$bam_file" .Aligned.out.sorted.CB.bam)
    echo cellID
    star --genomeDir $star_index \
         --readFilesType SAM PE --readFilesCommand samtools view --readFilesIn "$bam_file" \
         --outFileNamePrefix "${output_dir}/${cellID}." \
         --outSAMtype None  --limitBAMsortRAM 44006670219 \
         --quantMode GeneCounts --runThreadN 4
done
