# MethylAI: A Deep Learning Model for Predicting and Interpreting DNA Methylation from genomic Sequence

[![license](https://img.shields.io/badge/python_-3.10.15_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/PyTorch_-2.4.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/Captum_-0.6.8_-purple)](https://captum.ai/)
[![license](https://img.shields.io/badge/R_-4.3.3_-red)](https://www.r-project.org/)

MethylAI is a convolutional neural network (CNN) based model that predicts DNA methylation levels at CpG sites from one-hot encoded DNA sequences. MethylAI was pre-trained on the most comprehensive multi-species WGBS dataset, including 1,574 human samples across 52 tissues and other 11 mammals. The model leverages a multi-scale CNN architecture and exponential activations for high accuracy and improved interpretability. Its key applications include decoding the cis-regulatory logic of DNA methylation via integration with [DeepSHAP](https://github.com/shap/shap) and predicting the impact of genetic variants on methylation landscapes.

## Key Features & Highlights

### Comprehensive and Multi-Species Training Data

**Largest Human WGBS Dataset:** Trained on the most extensive collection of human whole-genome bisulfite sequencing (WGBS) data to date, comprising 1,574 samples spanning 52 tissues and 238 cell types.

**Cross-Species Pre-training:** Enhanced model accuracy through pre-training on WGBS data from human and **11 mammalian species**, including mouse (*Mus musculus*), rat (*Rattus norvegicus*), macaque (*Macaca fascicularis* and *Macaca mulatta*), chimpanzee (*Pan troglodytes*), gorilla (*Gorilla gorilla*), cow (*Bos taurus*), sheep (*Ovis aries*), dog (*Canis lupus familiaris*), pig (*Sus scrofa*), giant panda (*Ailuropoda melanoleuca*).

### Advanced Model Architecture

**Multi-scale CNN Module:** Captures sequence features at varying resolutions to improve predictive accuracy.

**Exponential Activation Function:** Increases model interpretability by improving representations of genomic sequence motifs ([ref](https://www.nature.com/articles/s42256-020-00291-x)).

### Sophisticated Training Strategy

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

Integrated with the DeepSHAP algorithm, MethylAI can quantify the contribution of each nucleotide to the methylation prediction. This allows for the identification of key sequence features, such as transcription factor binding motifs, that drive methylation changes.

### Interpreting GWAS Variants:

MethylAI can predict the impact of traits/disease-associated genetic variants on DNA methylation patterns. This capability provides mechanistic insights into how non-coding genetic variations may contribute to disease pathogenesis by altering the epigenetic landscape.

---

## Usage

This section provides a step-by-step guide to installing MethylAI and running its core functionalities.

### Installation

We recommend using [Conda](https://www.anaconda.com/) to manage the environment.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Yu-Lab-Genomics/MethylAI.git
cd MethylAI
```

#### Step 2: Create Conda Environment and Install Dependencies

```bash
# Create and activate the conda environment
conda create -n methylai python=3.11 mamba
conda activate methylai

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Download model checkpoints

You can download the pretrained model checkpoints below. Please download these files to the checkpoint folder to ensure our sample code runs smoothly.

- [Pre-trained model](https://methylai.aigenomicsyulab.com/): pre-trained with human dataset and other 11 mammalian species

- [Fine-tuned model with complete human dataset](https://methylai.aigenomicsyulab.com/): 1574 human samples

- [Fine-tuned model with ENCODE dataset](https://methylai.aigenomicsyulab.com/): 96 human samples from [ENCODE project](https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=WGBS&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens) (127 samples were available from ENCODE project, 96 samples passed our quality control)

- [Fine-tuned model with human cell type dataset](https://methylai.aigenomicsyulab.com/): 207 human samples from a [nature paper](https://www.nature.com/articles/s41586-022-05580-6)

---

### Quick Start: Model Inference Demo

Run a quick demo to ensure your installation is correct. This will predict methylation levels for a set of example DNA sequences.

#### Run the Inference Pipeline

```bash
python ./04.scPIT_inference/scPIT_inference.py --gpu 5 \
--model_ckpt ./02.checkpoint/model_weights.pth \
--output_path ./04.scPIT_inference/demo_output
```

Arguments:  
--gpu: ID of the GPU to use. Default is GPU 5.  
--model_ckpt: Path to the trained scPIT model checkpoint (already provided in ./02.checkpoint/).  

#### Expected Output

---

### Fine-tuning Tutorial 1: Using a Public ENCODE Dataset

This tutorial guides you through fine-tuning MethylAI on a public dataset.

#### Step 1: Download and Preprocess Data

```bash
# Download a sample ENCODE WGBS dataset (e.g., from ENCSR000***)
bash scripts/download_encode_data.sh

# Preprocess the data into the format required by MethylAI
python scripts/preprocess_encode_data.py \
  --input_bam data/encode_sample.bam \
  --reference_genome hg38.fa \
  --output_file data/encode_processed.h5
```

#### Step 2: Fine-tune the Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 nohup torchrun --standalone --nproc_per_node=gpu
```

### Fine-tuning Tutorial 2: Using Your Own WGBS Dataset

If you have your own WGBS data processed with Bismark, you can fine-tune MethylAI as follows.

Prerequisites:

- Your data should be in a format including columns: chromosome, start, end, methylated_reads, total_reads.

#### Step 1: Data Preprocessing

```bash
# Download a sample ENCODE WGBS dataset (e.g., from ENCSR000***)
bash scripts/download_encode_data.sh

# Preprocess the data into the format required by MethylAI
python scripts/preprocess_encode_data.py \
  --input_bam data/encode_sample.bam \
  --reference_genome hg38.fa \
  --output_file data/encode_processed.h5
```

#### Step 2: Fine-tune the Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 nohup torchrun --standalone --nproc_per_node=gpu
```

---

## Downstream Analysis

#### Decoding Cis-Regulatory Logic with DeepSHAP

Identify key sequence motifs influencing methylation predictions.

```bash
python scripts/run_deepshap.py \
  --model models/pretrained_model.h5 \
  --input_fasta data/regions_of_interest.fa \
  --output_dir results/deepshap/
```

This script will generate:

- results/deepshap/contributions.tsv: Nucleotide-level contribution scores.
- results/deepshap/motifs.html: An interactive visualization of identified motifs.

#### Interpreting Disease-Associated Genetic Variants

Predict the impact of a genetic variant (e.g., a SNP from GWAS) on DNA methylation.

```bash
python scripts/predict_variant_effect.py \
  --model models/pretrained_model.h5 \
  --reference_fasta data/reference_sequence.fa \
  --variant_vcf data/disease_associated_variants.vcf \
  --output_file results/variant_effects.tsv
```

The output will show the predicted change in methylation level for each variant, helping prioritize functionally relevant non-coding variants.

## Citation

If you use MethylAI in your research, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2025.11.20.689274v1)/publication.

