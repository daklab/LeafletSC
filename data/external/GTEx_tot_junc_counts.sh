#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J junc_Counts # <-- name of job

file_name=/gpfs/commons/groups/knowles_lab/Karin/data/GTEx/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct
echo $file_name
awk -F'\t' 'NR>2{sum=0; for(i=3;i<=NF;i++) sum+= $i}; NR>2 {printf("%s\t%d\n",$1,sum)}' $file_name > GTEx_juncs_total_counts.txt
echo "Done"