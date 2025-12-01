# MethylAI: A Deep Learning Model for Predicting and Interpreting DNA Methylation from genomic Sequence

[![license](https://img.shields.io/badge/python_-3.10.15_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/PyTorch_-2.4.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/Captum_-0.6.8_-purple)](https://captum.ai/)
[![license](https://img.shields.io/badge/R_-4.3.3_-red)](https://www.r-project.org/)

> 🚧 **Repository Under Active Development** - Full release coming soon! We're currently finalizing the codebase and documentation.  

MethylAI is a convolutional neural network (CNN) based model that predicts DNA methylation levels at CpG sites from one-hot encoded DNA sequences. MethylAI was pre-trained on the most comprehensive multi-species WGBS dataset, including 1,574 human samples across 52 tissues and other 11 mammals. The model leverages a multi-scale CNN architecture and exponential activations for high accuracy and improved interpretability. Its key applications include decoding the cis-regulatory logic of DNA methylation via integration with [DeepSHAP](https://github.com/shap/shap) and predicting the impact of genetic variants on methylation landscapes.

## Key Features & Highlights

### 1. Comprehensive and Multi-Species Training Data

**Largest Human WGBS Dataset:** Trained on the most extensive collection of human whole-genome bisulfite sequencing (WGBS) data to date, comprising 1,574 samples spanning 52 tissues and 238 cell types.

**Cross-Species Pre-training:** Enhanced model accuracy through pre-training on WGBS data from **human** and **11 mammalian species**, including mouse (*Mus musculus*), rat (*Rattus norvegicus*), macaque (*Macaca fascicularis* and *Macaca mulatta*), chimpanzee (*Pan troglodytes*), gorilla (*Gorilla gorilla*), cow (*Bos taurus*), sheep (*Ovis aries*), dog (*Canis lupus familiaris*), pig (*Sus scrofa*), giant panda (*Ailuropoda melanoleuca*).

### 2. Advanced Model Architecture

**Multi-scale CNN Module:** Captures sequence features at varying resolutions to improve predictive accuracy.

**Exponential Activation Function:** Increases model interpretability by improving representations of genomic sequence motifs ([ref](https://www.nature.com/articles/s42256-020-00291-x)).

### 3. Sophisticated Training Strategy

**Pre-training + Fine-tuning:** Leverages cross-species data for pre-training, followed by human-specific fine-tuning, resulting in superior prediction performance.

**Multi-task Prediction:** Simultaneously predicts methylation levels for the 1,574 human samples:

- at each CpG site (raw and smoothed site methylation level)

- average methylation levels over genomic regions of different lengths (200 bp, 500 bp, and 1 kb).

---

## Downstream Applications

MethylAI enables a wide range of functional genomics analyses:

### Genome-wide DNA methylation Prediction:

Predict the DNA methylation level for any input DNA sequence across all 1,574 human samples, providing a comprehensive methylation profile.

### Identification of DNA methylation linked active motifs:

Integrated with the DeepSHAP algorithm, MethylAI can quantify the contribution of each nucleotide to the methylation prediction. This allows for the identification of key sequence features, such as transcription factor (TF) binding motifs, that drive methylation changes.

### Interpreting GWAS Variants:

MethylAI can predict the impact of traits/disease-associated genetic variants on DNA methylation patterns. This capability provides mechanistic insights into how non-coding genetic variations may contribute to disease pathogenesis by altering the epigenetic landscape.

---

## Preparation

This section provides the dependencies and data acquisition for MethylAI.

### Environment Setup

We recommend using [Conda](https://www.anaconda.com/) for environment management to ensure reproducibility.

#### 1. Clone the Repository

```bash
git clone https://github.com/Yu-Lab-Genomics/MethylAI.git
cd MethylAI
```

#### 2. Create Conda Environment and Install Dependencies

```bash
# Create and activate the conda environment
conda create -n methylai python=3.10 mamba
conda activate methylai

# Install dependencies
pip install -r requirements.txt

# or install necessary dependencies
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Download Required Files

#### 1. Download MethylAI Checkpoints

You can download the model checkpoints below. We recommend downloading the checkpoints to the `checkpoint` directory:

- [Pre-trained model](https://backend.aigenomicsyulab.com/files/model-download/multi_species_pretrain): pre-trained with human dataset and other 11 mammalian species
- [Fine-tuned model with complete human dataset](https://backend.aigenomicsyulab.com/files/model-download/human_complete): 1574 human samples
- [Fine-tuned model with ENCODE dataset](https://backend.aigenomicsyulab.com/files/model-download/human_encode): 96 human samples from [ENCODE project](https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=WGBS&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens) (127 samples were available from ENCODE project, 96 samples passed our quality control)
- [Fine-tuned model with human cell type dataset](https://backend.aigenomicsyulab.com/files/model-download/human_cell_type): 207 human samples from a [nature paper](https://www.nature.com/articles/s41586-022-05580-6)
- [Fine-tuned model with HEK293T WGBS data](https://backend.aigenomicsyulab.com/files/model-download/hek293t) a WGBS of HEK293T cell line generated in this study
- Corresponding sample tables are available in our website: https://methylai.aigenomicsyulab.com/

#### 2. Download human reference genome (hg38)

Obtain the reference genome for sequence extraction and coordinate mapping:

```bash
wget -P data/genome -i data/genome/hg38_genome_link.txt
gunzip data/genome/hg38.fa.gz
```

#### 3. Download CpG Site Coordinates for hg38

```bash
wget -P data/genome https://backend.aigenomicsyulab.com/files/model-download/cpg_coordinate_hg38_chr1_22
```
Note: This `cpg_coordinate_hg38.chr1-22.sort.bed.gz` was generated using [wgbs_tools](https://github.com/nloyfer/wgbs_tools).

---

## Quick Start: Model Inference Demo

Run a quick demo to ensure your preparation is correct. This will predict methylation levels for a set of CpG site within test set.

### Run the Demo script

```bash
python demo/demo.py \
--cpg_coordinate demo/demo_data/cpg_coordinate.txt \
--genome_fasta data/genome/hg38.fa \
--config_file configs/methylai_finetune_human_cell_type.py \
--config_dict_name methylai_config_dict \
--model_ckpt checkpoint/MethylAI_finetune_human_cell_type.pth \
--gpu_id 0 \
--batch_size 200 \
--num_workers 8 \
--output_folder demo/demo_result/ \
--output_prefix demo \
--reverse_complement_augmentation \
--output_bedgraph
```
**Arguments (required)**:  
`--cpg_coordinate`: BED-formatted file (zero-base) containing CpG site coordinates. Contextual sequences will be extracted for model input.  
`--genome_fasta`: Reference genome FASTA file for sequence extraction.  
`--config_file`:  Python configuration file defining model architecture and hyperparameters.  
`--config_dict_name`: Name of the Python dictionary variable containing configuration parameters.  
`--model_ckpt`: Path to the model checkpoint file.  
`--gpu_id`: GPU device for computation.  
`--batch_size`: Batch size for DataLoader during inference.  
`--num_workers`: Number of parallel workers for DataLoader.  
`--output_folder`: Directory for saving prediction results.  

**Arguments (optional)**:  
`--output_prefix`: Custom prefix for output files.  
`--reverse_complement_augmentation`: Enable reverse complement data augmentation.  
`--output_bedgraph`: Generate methylation tracks in bedGraph format for genome browser visualization.  

**⚠️ Technical Note**: The MethylAI model is designed to predict both site-specific and regional methylation patterns. Consequently, the program does not validate whether input coordinates correspond to canonical CpG dinucleotides. We caution that prediction accuracy for non-CpG sites has not been systematically evaluated and may not reflect biological reality.

### Expected Output

The following output files will be generated in the specified output directory:

**1. Primary Output File: `demo/demo_result/demo_prediction_dataframe.txt`**
- **Format**: Tab-separated file with header row
- **Structure**:
  - Columns 1-3: BED-format coordinates (chr, start, end)
  - Subsequent columns: Methylation predictions formatted as `prediction_{index}`
- **Prediction Types**:
  - **Smoothed site methylation level** (indices 0-206)
  - **Raw site methylation level** (indices 207-413)
  - **Regional methylation estimates**:
    - 1kb window (indices 414-620)
    - 500bp window (indices 621-827)
    - 200bp window (indices 828-1034)
- **Note**: CpG coordinate order corresponds exactly to the input `cpg_coordinate.txt` file. For detailed interpretation of methylation level predictions, please refer to our publication (see Citation section).

#### 2. Visualization Files (Optional)

When `--output_bedgraph` is specified:

**Directory**: `demo/demo_result/bedgraph/`  
**Files**: Multiple bedGraph files named according to prediction columns (e.g., `prediction_0.bedgraph`, `prediction_1.bedgraph`)  
**Purpose**: Genome browser-compatible tracks for visualizing methylation patterns across genomic regions

---

## Fine-tuning Tutorial 1: Using a ENCODE Dataset

This tutorial guides you through fine-tuning MethylAI on a public ENCODE dataset.

### 1: Download ENCODE WGBS Data
```bash
wget -c -P data/encode -i data/encode/encode_wgbs_link.txt
```

### 2. Prepare train/validation/test dataset files
#### 2.1. Preprocess Data
Preprocessing to extract coverage and mc values from WGBS data.
```bash
python scripts/preprocess_encode_data.py \
  --input_folder data/encode \
  --output_folder data/encode_preprocessed \
  --cpg_coordinate data/genome/cpg_coordinate_hg38.chr1-22.sort.bed.gz
```
**Arguments:**
- `--input_folder`: Input directory with ENCODE WGBS datasets
- `--output_folder`: Output directory for processed data
- `--cpg_coordinate`: hg38 CpG coordinate BED file for methylation data integration

#### 2.2. Obtain Raw and Smoothed Methylation Values
The R script applies the bsmooth algorithm from the bsseq R package to generate both raw and smoothed methylation values for downstream analysis.
```bash
Rscript src/script/bsmooth_human_wgbs.R \
  data/encode_preprocessed \
  .preprocessed.txt \
  64 \
  sample_index.txt \
  smoothed_methylation.txt.gz \
  35 \
  500
```
Arguments (positional):  
`1`: Directory containing preprocessed ENCODE files (output from previous step)  
`2`: Suffix pattern to identify preprocessed files (default: .preprocessed.txt)  
`3`: Number of CPU cores to utilize for parallel processing (adjust based on available hardware)  
`4`: Mapping file linking filenames to sample indices in the output  
`5`: Output file containing both raw and smoothed methylation values (compressed)  
`6`: Minimum coverage threshold for methylation calling (bsmooth parameter)  
`7`: Smoothing window size for the loess regression (bsmooth parameter)

#### 2.3. Generate train/validation/test dataset files
```bash
python scripts/generate_dataset_files.py \
  --smoothed_methylation_file data/encode_preprocessed/smoothed_methylation.txt.gz \
  --output_folder data/encode_dataset
```

### 3. Fine-tune the Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 nohup torchrun --standalone --nproc_per_node=gpu
```

## Fine-tuning Tutorial 2: Using Your Own WGBS Dataset

If you have your own WGBS data processed with Bismark, you can fine-tune MethylAI as follows.

Prerequisites:

- Your data should be in a format including columns: chromosome, start, end, methylated_reads, total_reads.

### 1. Data Preprocessing

```bash
# Download a sample ENCODE WGBS dataset (e.g., from ENCSR000***)
bash scripts/download_encode_data.sh

# Preprocess the data into the format required by MethylAI
python scripts/preprocess_encode_data.py \
  --input_bam data/encode_sample.bam \
  --reference_genome hg38.fa \
  --output_file data/encode_processed.h5
```

### 2. Fine-tune the Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 nohup torchrun --standalone --nproc_per_node=gpu
```

---

## Downstream Analysis

### 1. Decoding Cis-Regulatory Logic with DeepSHAP

Identify DNA methylation linked TF motifs.

```bash
python script/run_deepshap.py \
  --model models/pretrained_model.h5 \
  --input_fasta data/regions_of_interest.fa \
  --output_dir results/deepshap/
```

This script will generate:

- results/deepshap/contributions.tsv: Nucleotide-level contribution scores.
- results/deepshap/motifs.html: An interactive visualization of identified motifs.

### 2. Interpreting GWAS Variants

Predict the impact of genetic variants on DNA methylation.

```bash
python scripts/predict_variant_effect.py \
  --model models/pretrained_model.h5 \
  --reference_fasta data/reference_sequence.fa \
  --variant_vcf data/disease_associated_variants.vcf \
  --output_file results/variant_effects.tsv
```

The output will show the predicted change in methylation level for each variant, helping prioritize functionally relevant non-coding variants.

## Citation

If you use MethylAI in your research, please cite our preprint/publication:  
- https://www.biorxiv.org/content/10.1101/2025.11.20.689274v1

