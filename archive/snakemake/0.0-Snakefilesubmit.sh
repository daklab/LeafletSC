#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -o /gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/slurm_outputs/%x-%j.out #redirect job output 

#load required modules
module purge

source /gpfs/commons/home/kisaev/miniconda3/bin/activate myenv
python --version 

module load leafcutter/0.2.9
module load sambamba

snakefile=/gpfs/commons/home/kisaev/leafcutter-sc/snakemake/Snakefile

bam=$1
samples=$2
juncs=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions/

echo $bam 
echo $samples
echo $juncs

snakemake -s $snakefile --config bamDir="$bam" juncDir="$juncs" sample_file="$samples" --prioritize get_junctions -j12 --latency-wait 432000 --cluster-config /gpfs/commons/home/kisaev/leafcutter-sc/snakemake/config/cluster.yaml --cluster-sync "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -o {cluster.out}" --rerun-incomplete #--unlock
