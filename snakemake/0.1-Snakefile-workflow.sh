#--------------------------------------------------------------------------------------

#Final workflow 

cd /gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter
module purge

#main pipeline script
pipeline=/gpfs/commons/home/kisaev/leafcutter-sc/snakemake/0.0-Snakefilesubmit.sh

##to get lists of samples -> go to directory with bam files and do the following:  
##ls -1 *.bam | sed -e 's/\.bam$//' > samples_list.csv 

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

#PBMCs

module load sambamba
bamfiles=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/bam_process/bam_splits/
samples=/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/bam_process/bam_splits/samples_list.csv

sbatch $pipeline $bamfiles $samples 


