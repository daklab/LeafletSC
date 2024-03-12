# LeafletSC

LeafletSC is a binomial mixture model designed for the analysis of alternative splicing events in single-cell RNA sequencing data. The model facilitates understanding and quantifying splicing variability at the single-cell level. Below is the graphical abstract of our approach:

<p align="center">
  <img src="https://github.com/daklab/Leaflet/assets/23510936/2c7981fe-91ec-4830-b010-b74ac4140940">
</p>

## Compatibility with sequencing platforms 
LeafletSC supports analysis from the following single-cell RNA sequencing platforms:
- Smart-Seq2 
- Split-seq
- 10X 

## Getting Started

LeafletSC is implemented in Python and requires Python version 3.10 (3.11 has not been tested yet). We recommend the following approach:

```bash
# create a conda environment with python 3.10 
conda create -n "LeafletSC" python=3.10 ipython
# activate environment 
conda activate LeafletSC
# install latest version of LeafletSC into this environment
pip install LeafletSC
```

Once the package is installed, you can load it in python as follows:
```python
import LeafletSC 

# or specific submodules 
from LeafletSC.utils import *
from LeafletSC.clustering import *
```

## Requirements 
Prior to using LeafletSC, please run **regtools** on your single-cell BAM files. Here is an example of what this might look like in a Snakefile:

```Snakemake
{params.regtools_path} junctions extract -a 6 -m 50 -M 500000 {input.bam_use} -o {output.juncs} -s XS -b {output.barcodes}
# Combine junctions and cell barcodes
paste --delimiters='\t' {output.juncs} {output.barcodes} > {output.juncswbarcodes}
```
- Once you have your junction files, you can try out the mixture model tutorial under [Tutorials](Tutorials/01_run_intron_clustering.ipynb)
- While optional, we recommend running LeafletSC intron clustering with a gtf file so that junctions can be first mapped to annotated splicing events. 

## Capabilities
With LeafletSC, you can:

- Infer cell states influenced by alternative splicing and identify differentially spliced regions.
- Conduct differential splicing analysis between specific cell groups if cell identities are known.
- Generate synthetic alternative splicing datasets for robust analysis testing.

## How does it work? 
The full method can be found in our [paper](https://www.biorxiv.org/content/10.1101/2023.10.17.562774v3) while the graphical model is shown below:
<p align="center">
  <img src="https://github.com/daklab/Leaflet/assets/23510936/3e147ba5-7ee8-47ae-b84c-5e99e0551acf">
</p>

## If you use Leaflet, please cite our [paper](https://www.biorxiv.org/content/10.1101/2023.10.17.562774v3)

```
@unpublished{Isaev2023-bf,
  title    = "Investigating RNA splicing as a source of cellular diversity using a binomial mixture model",
  author   = "Isaev, Keren and Knowles, David A",
  journal  = "bioRxiv",
  pages    = "2023.10.17.562774",
  month    = oct,
  year     = 2023,
  language = "en"
}
```

### To-do: 

1. Add documentation and some tests for how to run the simulation code 
2. Add 10X/split-seq mode in addition to smart-seq2
3. Extend framework to seurat/scanpy anndata objects
4. Add notes on generative model and inference method
5. Clean up dependencies 
