# Leaflet

Leaflet consists of a Bayesian mixture model that identifies latent cell states (or cell types) related to underlying alternative splicing differences in junctions. 

## Note: This repostiory is being actively modified, please feel free to submit issues if you have any questions.

## Compatibility with sequencing platforms 

- Smart-Seq2 (for now)

## Installation

Leaflet is a Python module which currently runs on Python version 3.9 or higher. Leaflet can be directly installed from github using the following command:

```pip install git+https://github.com/daklab/Leaflet.git``` 

This will automatically install any missing package dependencies.

Alternatively, Leaflet can be installed via conda (in the future). 

## Leaflet workflow 

### Input files
1. Required input files are junctions extracted using Regtools with the CB tag for single cells
2. We recommend running Regtools on a pseudobulk file containing all cells with "CB" present in the read
3. Use junction files generated for each pseudobulk sample as input into Leaflet's intron clustering functionality
4. A GTF file for your organism should be provided to intron clustering to identify alternatively spliced Leaflet events
 


