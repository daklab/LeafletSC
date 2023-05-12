library(data.table)
library(dplyr)

setwd("/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/d8a0c006-2991-5ddb-8ed8-7b7e8c187fe3")

# Read in the metadata
samples = fread("annotations_facs.csv")
samples = samples %>% select("cell", "cell_ontology_class", "free_annotation", "mouse.id", "subtissue", "tissue")

# Samples should look like this to match BAM files 
# A7-B002771-3_39_F-1-1_R1.mus.Aligned.out.sorted.bam instead of A1.B000610.3_56_F.1.1

# Replace "." in cell column with "-"
samples$cell = gsub("\\.", "-", samples$cell)

# Save file 
write.table(samples, file = "all_tabula_muris_samples.txt", sep = "\t", row.names = FALSE, quote = FALSE)