# MethylAI: A Deep Learning Model for Predicting and Interpreting DNA Methylation from genomic Sequence

[![license](https://img.shields.io/badge/python_-3.10.15_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/PyTorch_-2.4.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/Captum_-0.6.8_-purple)](https://captum.ai/)
[![license](https://img.shields.io/badge/R_-4.3.3_-red)](https://www.r-project.org/)

> 🚧 **Repository Under Active Development** - Full release coming soon! We're currently finalizing the codebase and documentation.  

MethylAI is a convolutional neural network (CNN) based model that predicts DNA methylation levels at CpG sites from one-hot encoded DNA sequences. MethylAI was pre-trained on the most comprehensive multi-species WGBS dataset, including 1,574 human samples across 52 tissues and other 11 mammals. The model leverages a multi-scale CNN architecture and exponential activations for high accuracy and improved interpretability. Its key applications include decoding the cis-regulatory logic of DNA methylation via integration with DeepSHAP and predicting the impact of genetic variants on methylation landscapes.

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
mamba install pandas==2.2.3 numpy==2.1.3 scipy polars==1.14.0 pyarrow==18.1.0 captum
mamba install r-base==4.3.3 r-data.table r-r.utils r-glue bioconductor-bsseq bioconductor-biocparallel
mamba install bedtools
```

### Download Required Files

#### 1. Download MethylAI Checkpoints

You can download the model checkpoints below. We recommend downloading the checkpoints to the `checkpoint` directory:

- [Pre-trained model](https://backend.aigenomicsyulab.com/files/model-download/multi_species_pretrain): pre-trained with human dataset and other 11 mammalian species
- [Fine-tuned model with complete human dataset](https://backend.aigenomicsyulab.com/files/model-download/human_complete): 1574 human samples
- [Fine-tuned model with ENCODE dataset](https://backend.aigenomicsyulab.com/files/model-download/human_encode): 96 human samples from [ENCODE project](https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=WGBS&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens) (127 samples were available from ENCODE project, 96 samples passed our sample quality control)
- [Fine-tuned model with human cell type dataset](https://backend.aigenomicsyulab.com/files/model-download/human_cell_type): 207 human samples from a [nature paper](https://www.nature.com/articles/s41586-022-05580-6)
- [Fine-tuned model with HEK293T WGBS data](https://backend.aigenomicsyulab.com/files/model-download/hek293t) a WGBS of HEK293T cell line generated in this study
- Corresponding sample tables are available in our website: https://methylai.aigenomicsyulab.com/

#### 2. Download human reference genome hg38

Obtain the reference genome for sequence extraction and coordinate mapping:

```bash
wget -c -P data/genome -i data/genome/hg38_genome_link.txt
gunzip data/genome/hg38.fa.gz
```

#### 3. Download CpG Site Coordinates for hg38

```bash
wget -c -P data/genome https://backend.aigenomicsyulab.com/files/model-download/cpg_coordinate_hg38_chr1_22
```
Note: This `cpg_coordinate_hg38.chr1-22.sort.bed.gz` was generated using [wgbs_tools](https://github.com/nloyfer/wgbs_tools).

#### 4. Download UCSC Genome Browser Tools Dependency

```bash
wget -c -P ucsc_tools -i ucsc_tools/ucsc_tools_link.txt
chmod u+x ucsc_tools/bedGraphToBigWig ucsc_tools/bigBedToBed
```

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

**Arguments (required)**  
`--cpg_coordinate`: BED-formatted file (zero-base) containing CpG site coordinates. Contextual sequences will be extracted for model input.  
`--genome_fasta`: Reference genome FASTA file for sequence extraction.  
`--config_file`:  Python configuration file defining model architecture and hyperparameters.  
`--config_dict_name`: Name of the Python dictionary variable containing configuration parameters.  
`--model_ckpt`: Path to the model checkpoint file.  
`--gpu_id`: GPU device for computation.  
`--batch_size`: Batch size for DataLoader during inference.  
`--num_workers`: Number of parallel workers for DataLoader.  
`--output_folder`: Directory for saving prediction results.  

**Arguments (optional)**  
`--output_prefix`: Custom prefix for output files.  
`--reverse_complement_augmentation`: Enable reverse complement data augmentation.  
`--output_bedgraph`: Generate methylation tracks in bedGraph format for genome browser visualization.  

**⚠️ Technical Note**: The MethylAI model is designed to predict both site-specific and regional methylation levels. Consequently, the program does not validate whether input coordinates correspond to canonical CpG dinucleotides. We caution that prediction accuracy for non-CpG sites has not been systematically evaluated and may not reflect biological reality.


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
  - **Regional methylation levels**:
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
  --input_file_suffix .bed.gz \
  --output_folder data/encode_preprocessed \
  --output_log_file preprocess.log \
  --reference_cpg_coordinate_file data/genome/cpg_coordinate_hg38.chr1-22.sort.bed.gz
```
**Arguments (required)**
- `--input_folder`: Input directory with ENCODE WGBS datasets
- `--input_file_suffix`:  
- `--output_folder`: Output directory for processed data
- `--output_log_file`:  

**Arguments (optional)**
- `--reference_cpg_coordinate_file`: reference CpG coordinate BED file for methylation data integration

#### 2.2. Obtain Raw and Smoothed Methylation Values
The R script applies the BSmooth algorithm from the [bsseq](https://bioconductor.org/packages/release//bioc/html/bsseq.html) R package to generate both raw and smoothed methylation values for downstream analysis.
```bash
Rscript src/script/bsmooth.R \
  data/encode_preprocessed \
  .preprocessed.txt \
  64 \
  smooth_methylation_info.txt \
  smooth_methylation_data.txt.gz \
  35 \
  500
```
**Arguments (positional)**  
`1`: Directory containing preprocessed ENCODE files (output from previous step)  
`2`: Suffix pattern to identify preprocessed files (default: .preprocessed.txt)  
`3`: Number of CPU cores to utilize for parallel processing (adjust based on available hardware)  
`4`: Mapping file linking filenames to sample indices in the output  
`5`: Output file containing both raw and smoothed methylation values (compressed)  
`6`: The minimum number of methylation loci in a smoothing window. (BSmooth parameter)  
`7`: The minimum smoothing window, in bases. (BSmooth parameter)

#### 2.3. Generate train/validation/test dataset files
```bash
python src/preprocess/generate_dataset.py \
--smooth_methylation_file data/encode_preprocess/smoothed_methylation_data.txt.gz \
--data_info_filedata/encode_preprocess/smoothed_methylation_info.txt \
--genome_fasta_file data/genome/hg38.fa \
--chrom_size_file data/genome/hg38.chrom.sizes \
--output_folder data/encode_dataset \
--output_prefix encode
```
**Expected output:**  
Upon successful execution, the following files will be generated in the specified output folder (`data/encode_dataset` in this example):

1. Complete Dataset File  
**File:** `encode_complete_dataset.txt`  
**Format:** Tab-separated values with header row  
**Contents:**  
- Columns 1-3: BED-format CpG site coordinates (`chr`, `start`, `end`), sorted by chromosome and start position.
- Subsequent columns represent methylation levels, organized in three sections:
  - **Smoothed site methylation levels**: Columns labeled as `smooth_{dataset_index}`
  - **Raw site methylation levels**: Columns labeled as `raw_{dataset_index}`
  - **Sequencing coverage**: Columns labeled as `coverage_{dataset_index}`
  - **Regional methylation levels**: Columns labeled as `window_{window_size}_{dataset_index}` (window sizes: 1000, 500, 200)  
**Purpose:** This comprehensive file contains data for all CpG sites and is intended for downstream analyses.

2. Model Training Datasets  
**Files:** `encode_train_set.pkl`, `encode_validation_set.pkl`, `encode_test_set.pkl`  
**Format:** Python pickle objects.  
**Contents:** These files contain the training, validation, and test splits, respectively, partitioned by chromosome as specified by the `--training_chr`, `--validation_chr`, and `--test_chr` arguments.  
**Purpose:** Direct input for model training and evaluation pipelines.

3.Dataset Information  
**File:** `encode_data_info.txt`  
**Format:** Tab-separated values with metadata.  
**Contents:**  
- Sample quality control (QC) statistics.  
- Mapping between `dataset_index`, `model_output_index`, and original filenames.

**Purpose:** Provides traceability between processed data and original samples, along with QC metrics for downstream interpretation.

**Arguments (required)**  
`--smooth_methylation_file`: File containing raw and smoothed methylation values (output from bsmooth.R).  
`--data_info_file`: Sample information file (output from bsmooth.R).  
`--genome_fasta_file`: Reference genome FASTA file (must match the coordinate system used in methylation files).  
`--chrom_size_file`: Chromosome sizes file for the reference genome.  
`--output_folder`: Directory for storing generated dataset files.  
`--output_prefix`: Prefix for output filenames.  

**Arguments (optional)**  
`--model_input_dna_length`: Length of DNA sequence used as model input (default: 18432). **Do not modify for tutorial or reproducibility**.  
`--threshold_min_coverage`: Minimum coverage threshold for defining high-quality CpG sites (default: `5`). **Do not modify for tutorial or reproducibility**.  
`--threshold_max_missing_cpg_ratio`: Sample QC threshold; samples with low-quality CpG ratio exceeding this value are excluded (default: `0.5`). **Do not modify for tutorial or reproducibility**.  
`--threshold_max_n_base_ratio`: CpG site QC threshold; sites with N-base ratio above this value in the extracted sequence are excluded (default: `0.02`). **Do not modify for tutorial or reproducibility**.  
`--threshold_max_missing_sample_ratio`: CpG site QC threshold; sites with low-quality calls across samples exceeding this ratio are excluded (default: `0.5`). **Do not modify for tutorial or reproducibility**.  
`--calculate_regional_methylation`: Window sizes (in bp) for regional methylation calculation (default: `1000 500 200`). Set `--calculate_regional_methylation 0` to disable this calculation. **Note: Computing regional methylation for all ~27 million CpG sites requires approximately 24 hours. Do not modify for tutorial or reproducibility**.  
`--quiet`: Suppress runtime messages when set.  
`--output_format`: Dataset output format; options: pickle or feather (default: `pickle`). **Do not modify for tutorial or reproducibility**.  
`--output_sampled_training_set`: Generates randomly sampled training subsets for rapid experimentation. Accepts one or more floating-point values between 0 and 1 (e.g., `0.1 0.2 0.5`) representing the sampling proportions relative to the full training set. This feature is disabled by default. **Do not set for full reproducibility**.  
`--output_slice_training_set`: Output each CpG site as a separate file to handle memory constraints. **Note: This mode requires >6 hours to complete**.  
`--training_chr`: Chromosomes for training set (default: `chr1, chr2, chr3, chr4, chr5, chr6, chr7, chr8, chr9, chr12, chr13, chr14, chr15, chr16, chr17, chr18, chr19, chr20, chr21, chr22`). **Do not modify for tutorial or reproducibility**.  
`--validation_chr`: Chromosomes for validation set (default: `chr10`). **Do not modify for tutorial or reproducibility**.  
`--test_chr`: Chromosomes for test set (default: `chr11`). **Do not modify for tutorial or reproducibility**.

### 3. Fine-tune the Model

To fine-tune MethylAI on your processed dataset, you need to modify the configuration file. Follow these steps:

1. **Open the configuration file**  
   Locate and open `config/finetune_tutorial_encode.py` in a text editor.

2. **Modify the configuration dictionary**  
   Update the following keys in the `methylai_config_dict` dictionary:

   | Key | Description                                                                                                                                                   | Example Value                                                     |
   |-----|---------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
   | `'output_folder'` | **Absolute path** to the directory where all fine‑tuning outputs (checkpoints, logs, etc.) will be saved.                                                     | `/absolute/path/to/your/output_folder`                            |
   | `'output_result_file'` | Filename (within `output_folder`) that will store per‑epoch training and validation loss, validation Pearson correlation (PCC), and Spearman correlation (SCC). | `fine_tune_results.txt`                                           |
   | `'pretrain_snapshot_path'` | **Absolute path** to the pre‑trained checkpoint downloaded in the Preparation step (`MethylAI_pretrain_12_species.pth`).                                      | `/absolute/path/to/checkpoint/MethylAI_pretrain_12_species.pth`   |
   | `'train_set_file'` | **Absolute path** to the training set file generated in the previous step (`encode_train_set.pkl`).                                                           | `/absolute/path/to/data/encode_dataset/encode_train_set.pkl`      |
   | `'validation_set_file'` | **Absolute path** to the validation set file (`encode_validation_set.pkl`).                                                                                   | `/absolute/path/to/data/encode_dataset/encode_validation_set.pkl` |
   | `'genome_fasta_file'` | **Absolute path** to the reference genome FASTA file (`hg38.fa`).                                                                                             | `/absolute/path/to/data/genome/hg38.fa`                           |
   | `'batch_size'` | Batch size for training. Default is 50 (for RTX4090 24GB GPU). Adjust according to your GPU memory capacity.                                        | `50`                                                              |

3. **Save the configuration file**  
   After making the changes, save the file.

**Important Notes:**
- We strongly recommend using **absolute paths** to avoid path‑related errors.
- The `'batch_size'` should be tuned based on your GPU’s available memory. Reduce it if you encounter out‑of‑memory errors.
- Ensure that all input files (checkpoint, dataset, reference genome) are accessible at the specified paths.

After configuring the parameters in `config/finetune_tutorial_encode.py`, execute the following command to fine‑tune the MethylAI model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
nohup torchrun --standalone --nproc_per_node=gpu \
src/script/finetune.py \
--config_file configs/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--print_loss_step 500 \
--print_model_output_step 5000 \
>> result/finetune_tutorial_encode.log 2>&1 &
```
**Environment Setting:**  
`CUDA_VISIBLE_DEVICES`: Specifies the GPU devices available for the job (default: 0‑3). Adjust this variable to match your hardware configuration.  
`OMP_NUM_THREADS`: Sets the number of OpenMP threads for CPU‑parallel operations (default: 4). Modify based on your CPU core count.

**Arguments (required)**  
`--config_file`: Path to the Python configuration file defining the model and training setup.  
`--config_dict_name`:  Name of the Python dictionary variable (within the config file) that holds the configuration.

**Arguments (optional)**  
`--print_loss_step`: Interval (in training steps) for printing loss values to monitor training stability and detect NaN issues (default: 500).  
`--print_model_output_step`: Interval (in training steps) for logging model outputs to verify numerical stability (default: 5000).

**Logging**  
`result/finetune_tutorial_encode.log`: All runtime messages (stdout and stderr) are redirected to this log file. You can change the path and filename as needed.  
The `nohup` command allows the process to continue running after disconnecting from the terminal.

**Expected output**  
Upon successful fine-tuning, the following files will be generated in the directory specified by the `'output_folder'` key in `methylai_config_dict`:

1. Training Results File
File: `{output_result_file}` (as defined in the configuration)  
Format: Tab-separated values  
Contents: Per‑epoch training and validation metrics, including:
- Training loss
- Validation loss
- Pearson correlation coefficient (PCC) on the validation set
- Spearman correlation coefficient (SCC) on the validation set

2. Model Snapshots
Directory: `{output_folder}/snapshot/`
- Files: Checkpoint files named `snapshot_epoch_{epoch_number}.pth` (one per epoch)
- Purpose: These snapshots allow you to resume training from any epoch or for downstream tasks.

---
## Fine-tuning Tutorial 2: Using Your Own WGBS Dataset

If you have your own WGBS data processed with Bismark, you can fine-tune MethylAI as follows.

### 1. Prepare train/validation/test dataset files
1.1 Preprocessing to extract coverage and mc values from Bismark output.

```bash
python scripts/preprocess_bismark_data.py \
  --input_folder data/bismark \
  --input_file_suffix bedGraph.gz.bismark.zero.cov \
  --output_folder data/bismark_preprocess \
  --output_log_file preprocess.log \
  --reference_cpg_coordinate_file data/genome/cpg_coordinate_hg38.chr1-22.sort.bed.gz
```

1.2 The R script applies the BSmooth algorithm from the bsseq R package to generate both raw and smoothed methylation values for downstream analysis.

```bash
Rscript src/script/bsmooth.R \
  data/bismark_preprocess \
  .preprocessed.txt \
  64 \
  smooth_methylation_info.txt \
  smooth_methylation_data.txt.gz \
  35 \
  500
```

1.3 Generate train/validation/test dataset files

```bash
python src/preprocess/generate_dataset.py \
--smooth_methylation_file data/bismark_preprocess/smoothed_methylation_data.txt.gz \
--data_info_file data/bismark_preprocess/smoothed_methylation_info.txt \
--genome_fasta_file data/genome/hg38.fa \
--chrom_size_file data/genome/hg38.chrom.sizes \
--output_folder data/bismark_dataset \
--output_prefix bismark
```

### 2. Fine-tune the Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
nohup torchrun --standalone --nproc_per_node=gpu \
src/script/finetune.py \
--config_file configs/finetune_tutorial_bismark.py \
--config_dict_name methylai_config_dict \
--print_loss_step 500 \
--print_model_output_step 5000 \
>> result/finetune_tutorial_bismark.log 2>&1 &
```
---

## Downstream Analysis 1: Identification of DNA Methylation Linked Active TF Motif Sites

### 1. Preparation

Download JASPAR Transcription Factors Track from UCSC Genome Browser and retain TF motif sites with a motif match score > 400:
```bash
wget -c -P data/genome https://hgdownload.soe.ucsc.edu/gbdb/hg38/jaspar/JASPAR2024.bb
ucsc_tools/bigBedToBed data/genome/JASPAR2024.bb data/genome/JASPAR2024.bed
awk -F'\t' '$5 > 400' data/genome/JASPAR2024.bed > data/genome/JASPAR2024_400.bed
```

### 2. Selection of Representative CpG Sites
```bash
python -u src/analysis_motif/get_low_me_region_representative_cpg.py \
--complete_dataset_file data/encode_dataset/encode_complete_dataset.txt \
--col_index_number 1 \
--output_folder data/encode_motif \
--output_prefix encode
```

### 3. Prediction Accuracy Evaluation of Representative CpG Sites
```bash
python -u src/analysis_motif/evaluate_representative_cpg.py --representative_cpg_file data/encode_motif/encode_smooth_1_low_methylation_region_representative_cpg.txt \
--dataset_info_file data/encode_dataset/encode_dataset_info.txt \
--config_file configs/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--model_ckpt result/finetune_tutorial_encode/snapshot/snapshot_epoch_2.pth \
--gpu_id 0 --batch_size 200 --num_workers 8 \
--col_index_number 1 --output_folder result/finetune_tutorial_encode/motif_analysis \
--output_prefix encode \
--reverse_complement_augmentation
```

### 4. Obtain DNA Sequence Attribution Score with DeepSHAP
```bash
python -u src/analysis_motif/get_sequence_attribution.py \
--representative_cpg_file data/encode_motif/encode_smooth_1_low_methylation_region_representative_cpg.txt \
--config_file configs/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--model_ckpt result/finetune_tutorial_encode/snapshot/snapshot_epoch_2.pth \
--gpu_id 1 \
--analyze_name col_1 \
--analyze_output_index 0 \
--n_permutation 80 --output_folder result/finetune_tutorial_encode/motif_analysis \
> result/finetune_tutorial_encode/motif_analysis/get_sequence_attribution_col_1.log 2>&1 &
```

### Motif Attribution Score Statistic
```bash
python -u src/analysis_motif/get_motif_statistic.py --sequence_attribution_folder result/finetune_tutorial_encode/motif_analysis/col_1_target0 \
--jaspar_bed_file data/genome/JASPAR2024_400.bed \
--output_folder result/finetune_tutorial_encode/motif_analysis/col_1_target0/motif_statistic \
--output_prefix encode_col_1 \
> result/finetune_tutorial_encode/motif_analysis/get_motif_statistic_col_1.log 2>&1 &
```

### Analysis of Active Motif Site
```bash

```

---
## Downstream Analysis 2: Interpreting GWAS Variants

Predict the impact of genetic variants on DNA methylation.

```bash
```


## Citation

If you use MethylAI in your research, please cite our preprint/publication:  
- https://www.biorxiv.org/content/10.1101/2025.11.20.689274v1

