#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -J SS_STAR # <-- name of job
#SBATCH --array=1-39345%32 # <-- number of jobs to run 

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load star/2.7.10b
module load samtools
module load sambamba

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues

# generate list of ALL bam files across tissues that end in *_merged.mus.Aligned.out.sorted.CB.bam
# find . -name "*_merged.mus.Aligned.out.sorted.CB.bam" > bam_list.txt

gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf
star_index=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/star2.7.10b
output_dir=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/STAR #make sure to create SJ_files folder within this directory once before submitting jobs

# Get a list of all the unique tissues that we have 
#TISSUE_LIST=($(ls -d */ | awk -F/ '{print $1}'))

# use bam_list.txt to generate sample list 
SAMPLE_LIST=($(cat bam_list.txt))

# Get the current tissue for the SAMPLE ID
CELL_BAM=${SAMPLE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $CELL_BAM

# Extract tissue name from the SAMPLE ID
TISSUE=$(echo $CELL_BAM | awk -F_ '{print $1}') 

# Only keep the part of TISSUE after './'
TISSUE=${TISSUE#./}

# Check if $output_dir/SJ_files2/$TISSUE exists, if not, create it
if [ ! -d "${output_dir}/SJ_files2/${TISSUE}" ]; then
    mkdir -p "${output_dir}/SJ_files2/${TISSUE}"
fi

cellID=$(basename "$CELL_BAM" _merged.mus.Aligned.out.sorted.CB.bam)
echo $cellID

# sort the bam file first to ensure that paired-end reads are always consecutive lines 
sambamba sort -n -M -p $CELL_BAM -o ${output_dir}/resorted_BAM2/$cellID.PE.sorted.bam
bam_input=${output_dir}/resorted_BAM2/$cellID.PE.sorted.bam

star --genomeDir $star_index \
     --readFilesType SAM PE --readFilesCommand samtools view --readFilesIn "$bam_input" \
     --outFileNamePrefix "${output_dir}/SJ_files2/${TISSUE}/${cellID}." \
     --outSAMtype None  --limitBAMsortRAM 44006670219 \
     --runThreadN 4 \
     --outSJtype Standard

# now do second pass for TranscriptomeSAM 
star --runThreadN 4 --genomeDir $star_index --quantMode TranscriptomeSAM \
    --sjdbFileChrStartEnd "${output_dir}/SJ_files2/${TISSUE}/${cellID}.SJ.out.tab" \
    --readFilesType SAM PE --readFilesCommand samtools view --readFilesIn "$bam_input" \
   --outSAMtype None \
   --outFileNamePrefix "${output_dir}/SJ_files2/${TISSUE}/${cellID}."

# remove some files that we don't need (log)
rm "${output_dir}/SJ_files2/${TISSUE}/${cellID}.Log.out"
rm "${output_dir}/SJ_files2/${TISSUE}/${cellID}.Log.progress.out"
rm "${output_dir}/SJ_files2/${TISSUE}/${cellID}.Log.final.out"
# remove folder with cell STARgenome
rm -r "${output_dir}/SJ_files2/${TISSUE}/${cellID}._STARgenome"

#---------------------------------------------------------
#generate index first (only need to do this once [done])
#---------------------------------------------------------

#star --runThreadN 4 --runMode genomeGenerate \
#    --genomeDir /gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/star2.7.10b \
#    --genomeFastaFiles /gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/fasta/genome.fa \
#    --sjdbGTFfile $gtf_file \
#    --sjdbOverhang 100 
