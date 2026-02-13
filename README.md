# MethylAI: A Deep Learning Model for Predicting and Interpreting DNA Methylation from genomic Sequence

[![license](https://img.shields.io/badge/python_-3.10.15_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/PyTorch_-2.4.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/Captum_-0.6.8_-purple)](https://captum.ai/)
[![license](https://img.shields.io/badge/R_-4.3.3_-red)](https://www.r-project.org/)

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

We recommend using Conda (https://www.anaconda.com/) for environment management to ensure reproducibility.

#### 1. Clone the Repository

```bash
git clone https://github.com/Yu-Lab-Genomics/MethylAI.git
cd MethylAI
```

#### 2. Create Conda Environment and Install Dependencies

```bash
# Create and activate the conda environment
conda create -n methylai python=3.10
conda activate methylai

# Install necessary dependencies
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 captum==0.6.0 -c pytorch -c nvidia
conda install pandas==2.3.3 polars==1.14.0 scipy
conda install bedtools -c bioconda

conda install r-base=4.3.3 r-data.table r-r.utils r-glue bioconductor-bsseq bioconductor-biocparallel -c conda-forge -c bioconda
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
wget -c -O data/genome/cpg_coordinate_hg38.chr1-22.sort.bed.gz https://backend.aigenomicsyulab.com/files/model-download/cpg_coordinate_hg38_chr1_22
```
Note: This `cpg_coordinate_hg38.chr1-22.sort.bed.gz` was generated using [wgbstools](https://github.com/nloyfer/wgbs_tools).

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
--config_file config/finetune_human_cell_type.py \
--config_dict_name methylai_config_dict \
--model_ckpt checkpoint/MethylAI_finetune_human_cell_type.pth \
--gpu_id 0 \
--batch_size 100 \
--num_workers 4 \
--output_folder demo/demo_result/ \
--output_prefix demo \
--reverse_complement_augmentation \
--output_bedgraph
```
**Expected Output**  

The following output files will be generated in the specified output directory:

**1. Primary Output File**: `demo/demo_result/demo_prediction_dataframe.txt`
- **Format**: Tab-separated file with header
- **Structure**:
  - Columns 1-3: BED-format coordinates (chr, start, end)
  - Subsequent columns: Methylation predictions formatted as `prediction_{index}`
- **Prediction Types**:
  - **Smoothed site methylation level** (index 0-206)
  - **Raw site methylation level** (index 207-413)
  - **Regional methylation levels**:
    - 1kb window (index 414-620)
    - 500bp window (index 621-827)
    - 200bp window (index 828-1034)
- **Note**: CpG coordinate order corresponds exactly to the input `cpg_coordinate.txt` file. For detailed interpretation of methylation level predictions, please refer to our publication (see **Citation** section).

**2. Visualization Files (Optional)**

When `--output_bedgraph` is specified:  

**Directory**: `demo/demo_result/bedgraph/`  
**Files**: Multiple bedGraph files named according to prediction columns (e.g., `prediction_0.bedgraph`, `prediction_1.bedgraph`)  
**Purpose**: Genome browser-compatible tracks for visualizing methylation patterns across genomic regions

**Arguments (required)**  
`--cpg_coordinate`: BED-formatted file (zero-base) containing CpG site coordinates. Contextual sequences will be extracted for model input.  
`--genome_fasta`: Reference genome FASTA file for sequence extraction.  
`--config_file`:  Path to the Python configuration file.  
`--config_dict_name`:  Name of the Python dictionary variable (within the config file) that holds the configuration.  
`--model_ckpt`: Path to the model checkpoint file.  
`--gpu_id`: GPU device for computation.  
`--batch_size`: Batch size for DataLoader during inference.  
`--num_workers`: Number of parallel workers for DataLoader.  
`--output_folder`: Directory for saving prediction results.  

**Arguments (optional)**  
`--output_prefix`: Custom prefix for output files.  
`--reverse_complement_augmentation`: Enable reverse complement data augmentation.  
`--output_bedgraph`: Generate methylation tracks in bedGraph format for genome browser visualization.

### Note

MethylAI was pre-trained and fine-tuned using human chr1–9 and chr12–22 for training, chr10 for validation, and chr11 for testing (all coordinates are based on the human hg38 genome assembly). If you wish to use MethylAI as a baseline model in your research, you can quickly evaluate its prediction accuracy on ENCODE or human cell‑type datasets by following these steps:

1. Replace the demo CpG coordinate file  
Substitute demo/demo_data/cpg_coordinate.txt with the coordinates of chromosome 11 CpG sites (hg38, see **Preparation** Section).
2. Configure the model parameters  
Ensure that the --config_file, --config_dict_name, and --model_ckpt arguments point to the appropriate configuration and checkpoint files for the model you want to evaluate (e.g., the fine‑tuned model on ENCODE dataset of human cell type dataset, see **Preparation** Section).
3. Run the demo script  
Execute demo/demo.py as described in the Demo section. The output will provide MethylAI’s predicted methylation levels for the chromosome 11 test set.
4. Fine‑tune on your own data (optional)  
To adapt MethylAI to your own dataset, follow the Fine‑tuning Tutorial 1 and 2 to fine‑tune a model that can serve as a baseline for your specific biological context.

**⚠️ Technical Note**: The MethylAI model is designed to predict both site-specific and regional methylation levels. Consequently, the program does not validate whether input coordinates correspond to canonical CpG dinucleotides. We caution that prediction accuracy for non-CpG sites has not been systematically evaluated and may not reflect biological reality.

---

## Fine-tuning Tutorial 1: Using a ENCODE Dataset

This tutorial guides you through fine‑tuning a MethylAI model using an example ENCODE Whole Genome Bisulfite Sequencing (WGBS) dataset.

### 1: Download ENCODE WGBS Data

Download processed ENCODE WGBS data (output type: methylation state at CpG, in .bed.gz format). For this tutorial, we have selected 4 samples that pass our quality‑control criteria.

```bash
wget -c -P data/encode -i data/encode/encode_wgbs_link.txt
```

### 2. Prepare train/validation/test dataset files

This step creates the dataset files required for fine-tuning MethylAI.

#### 2.1. Preprocess Data

This script processes the ENCODE WGBS data by extracting coverage and methylated read counts (mc) for each CpG site.

```bash
python src/preprocess/preprocess_encode_data.py \
--input_folder data/encode \
--input_file_suffix .bed.gz \
--output_folder data/encode_preprocess \
--output_log_file preprocess.log \
--reference_cpg_coordinate_file data/genome/cpg_coordinate_hg38.chr1-22.sort.bed.gz 
```
**Expected Output:**

The script generates `.preprocessed.txt` files in the specified `--output_folder`, retaining the original filename as the base name.

- **Format**: Tab-separated values with header.
- **Columns:**
  - 1-3: BED-format coordinates (chr, start, end). If `--reference_cpg_coordinate_file` is provided, coordinates are unified with this reference; otherwise, they are derived from merging forward and reverse strands.
  - 4: mc (number of reads showing methylation at the position). 
  - 5: cov (total read coverage at the position).

**Arguments (required)**  
`--input_folder`: Directory containing the downloaded ENCODE WGBS files.  
`--input_file_suffix`: Suffix to identify the input files (e.g., .bed.gz or .bed).  
`--output_folder`: Directory where preprocessed files will be saved.  
`--output_log_file`: Filename for the processing log.  

**Arguments (optional)**  
`--reference_cpg_coordinate_file`: Reference CpG coordinate BED file (.bed or .bed.gz). If provided, methylation data is aligned to these coordinates for downstream analysis integration.

#### 2.2. Obtain Raw and Smoothed Methylation Values

This step uses the BSmooth algorithm from the R package bsseq to calculate raw and smoothed methylation values for each CpG site from the mc and cov columns in the preprocessed files.

```bash
Rscript src/preprocess/bsmooth.R \
  data/encode_preprocess \
  .preprocessed.txt \
  64 \
  smooth_methylation_info.txt \
  smooth_methylation_data.txt.gz \
  35 \
  500
```

**Expected Output:**

Two files will be generated in the `data/encode_preprocess` directory:

1. `smooth_methylation_info.txt`: tab‑separated file that maps each `dataset_index` to the original filename.
2. `smooth_methylation_data.txt.gz`: compressed, tab‑separated file with a header. The columns are:
   - Columns 1‑3: CpG coordinates (chr, start, end).
   - Subsequent columns: For each sample (indexed by `dataset_index`), three columns are produced:
     - `smooth_{dataset_index}`: Smoothed site methylation level.
     - `raw_{dataset_index}`: Raw site methylation level.
     - `coverage_{dataset_index}`: Read coverage at the CpG site.

**Arguments (positional)**  
`1`: Directory containing preprocessed `.preprocessed.txt` files (output from previous step).  
`2`: Suffix pattern to identify preprocessed files (default: `.preprocessed.txt`).  
`3`: Number of CPU cores to utilize for parallel processing (adjust based on available hardware).  
`4`: Output file that maps filenames to dataset indices.  
`5`: Output file (gzipped) containing the raw and smoothed methylation values.  
`6`: The minimum number of methylation loci in a smoothing window. (BSmooth parameter). **Do not modify for tutorial or reproducibility**.  
`7`: The minimum smoothing window, in bases. (BSmooth parameter). **Do not modify for tutorial or reproducibility**.  

#### 2.3. Generate train/validation/test dataset files

This step performs quality control (QC) for each sample, calculates regional methylation levels, and produces the dataset files required for model training. **Note:** Because this step calculates regional methylation levels centered on ~27 million human CpG sites, it requires approximately 24 hours to complete.

```bash
python -u src/preprocess/generate_dataset.py \
--smooth_methylation_file data/encode_preprocess/smooth_methylation_data.txt.gz \
--data_info_file data/encode_preprocess/smooth_methylation_info.txt \
--genome_fasta_file data/genome/hg38.fa \
--chrom_size_file data/genome/hg38.chrom.sizes \
--output_folder data/encode_dataset \
--output_prefix encode
> data/encode_preprocess/generate_dataset.log 2>&1 &
```

**Expected output:**  
Upon successful execution, the following files will be generated in the specified output folder (`data/encode_dataset` in this example):

1. Complete Dataset File  
- **File:** `encode_complete_dataset.txt`  
- **Format:** Tab-separated values with header  
- **Contents:**  
  - Columns 1-3: BED-format CpG site coordinates (`chr`, `start`, `end`), sorted by chromosome and start position.
  - Subsequent columns represent methylation levels, organized in three sections:
    - **Smoothed site methylation levels**: Columns labeled as `smooth_{dataset_index}`
    - **Raw site methylation levels**: Columns labeled as `raw_{dataset_index}`
    - **Sequencing coverage**: Columns labeled as `coverage_{dataset_index}`
    - **Regional methylation levels**: Columns labeled as `window_{window_size}_{dataset_index}` (window sizes: 1000 bp, 500 bp, 200 bp)  
- **Purpose:** This comprehensive file contains data for all CpG sites and is intended for downstream analyses.

2. Model Training Datasets  
- **Files:** `encode_train_set.pkl`, `encode_validation_set.pkl`, `encode_test_set.pkl`  
- **Format:** Python pickle objects.  
- **Contents:** These files contain the training, validation, and test splits, respectively, partitioned by chromosome as specified by the `--training_chr`, `--validation_chr`, and `--test_chr` arguments.  
- **Purpose:** Direct input for model training and evaluation pipelines.

3. Dataset Information  
- **File:** `encode_dataset_info.txt`  
- **Format:** Tab-separated values with metadata.  
- **Contents:**  
  - Sample quality control (QC) statistics.  
  - Mapping between `dataset_index`, `model_output_index`, and original filenames.
- **Purpose:** Provides traceability between processed data and original samples, along with QC metrics for downstream interpretation.

**Arguments (required)**  
- `--smooth_methylation_file`: File containing raw and smoothed methylation values (output from previous step).  
- `--data_info_file`: Sample information file (output from previous step).  
- `--genome_fasta_file`: Reference genome FASTA file (must match the coordinate system used in methylation files).  
- `--chrom_size_file`: Chromosome sizes file for the reference genome.  
- `--output_folder`: Directory for storing generated dataset files.  
- `--output_prefix`: Prefix for output filenames.  

**Arguments (optional)**  
- `--model_input_dna_length`: Length of DNA sequence used as model input (default: 18432). **Do not modify for tutorial or reproducibility**.  
- `--threshold_min_coverage`: Minimum coverage threshold for defining high-quality CpG sites (default: 5). **Do not modify for tutorial or reproducibility**.  
- `--threshold_max_missing_cpg_ratio`: Sample QC threshold; samples with low-quality CpG ratio exceeding this value are excluded (default: 0.5). **Do not modify for tutorial or reproducibility**.  
- `--threshold_max_n_base_ratio`: CpG site QC threshold; sites with N-base ratio above this value in the extracted sequence are excluded (default: 0.02). **Do not modify for tutorial or reproducibility**.  
- `--threshold_max_missing_sample_ratio`: CpG site QC threshold; sites with low-quality calls across samples exceeding this ratio are excluded (default: 0.5). **Do not modify for tutorial or reproducibility**.  
- `--calculate_regional_methylation`: Window sizes (in bp) for regional methylation calculation (default: `1000 500 200`). Set `--calculate_regional_methylation 0` to disable this calculation. Note: Computing regional methylation for all ~27 million CpG sites requires approximately 24 hours. **Do not modify for tutorial or reproducibility**.  
- `--quiet`: Suppress runtime messages when set.  
- `--output_format`: Dataset output format; options: pickle or feather (default: `pickle`). **Do not modify for tutorial or reproducibility**.  
- `--output_sampled_training_set`: Generates randomly sampled training subsets for rapid experimentation. Accepts one or more floating-point values between 0 and 1 (e.g., `0.1 0.2 0.5`) representing the sampling proportions relative to the full training set. This feature is disabled by default. **Do not set for full reproducibility**.  
- `--output_slice_training_set`: Output each CpG site as a separate file to handle memory constraints. **Note: This mode requires >4 hours to complete**.  
- `--training_chr`: Chromosomes for training set (default: `chr1  chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22`). **Do not modify for tutorial or reproducibility**.  
- `--validation_chr`: Chromosomes for validation set (default: `chr10`). **Do not modify for tutorial or reproducibility**.  
- `--test_chr`: Chromosomes for test set (default: `chr11`). **Do not modify for tutorial or reproducibility**.

### 3. Fine-tune the Model

To fine-tune MethylAI on your processed dataset, you need to modify the configuration file. Follow these steps:

1. **Open the configuration file**  
   Locate and open `config/finetune_tutorial_encode.py` in a text editor.

2. **Modify the configuration dictionary**  
   Update the following keys in the `methylai_config_dict` dictionary:

   | Key                          | Description                                                                                                                                                     | Example Value                                                     |
   |------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
   | `'output_folder'`            | **Absolute path** to the directory where all fine‑tuning outputs (checkpoints, logs, etc.) will be saved.                                                       | `/absolute/path/to/your/output_folder`                            |
   | `'output_result_file'`       | Filename (within `output_folder`) that will store per‑epoch training and validation loss, validation Pearson correlation (PCC), and Spearman correlation (SCC). | `fine_tune_results.txt`                                           |
   | `'pretrain_checkpoint_path'` | **Absolute path** to the pre‑trained checkpoint downloaded in the Preparation step (`MethylAI_pretrain_12_species.pth`).                                        | `/absolute/path/to/checkpoint/MethylAI_pretrain_12_species.pth`   |
   | `'train_set_file'`           | **Absolute path** to the training set file generated in the previous step (`encode_train_set.pkl`).                                                             | `/absolute/path/to/data/encode_dataset/encode_train_set.pkl`      |
   | `'validation_set_file'`      | **Absolute path** to the validation set file (`encode_validation_set.pkl`).                                                                                     | `/absolute/path/to/data/encode_dataset/encode_validation_set.pkl` |
   | `'genome_fasta_file'`        | **Absolute path** to the reference genome FASTA file (`hg38.fa`).                                                                                               | `/absolute/path/to/data/genome/hg38.fa`                           |
   | `'batch_size'`               | Batch size for training. Default is 50 (for RTX4090 24GB GPU). Adjust according to your GPU memory capacity.                                                    | `50`                                                              |

3. **Save the configuration file**  
   After making the changes, save the file.

**Important Notes**
- We strongly recommend using **absolute paths** to avoid path‑related errors.
- The `'batch_size'` should be tuned based on your GPU’s available memory. Reduce it if you encounter out‑of‑memory errors.
- Ensure that all input files (checkpoint, dataset, reference genome) are accessible at the specified paths.

After configuring the parameters in `config/finetune_tutorial_encode.py`, execute the following command to fine‑tune the MethylAI model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
nohup torchrun --standalone --nproc_per_node=gpu \
src/training/finetune.py \
--config_file config/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--print_loss_step 500 \
--print_model_output_step 5000 \
> result/finetune_tutorial_encode.log 2>&1 &
```

**Environment Setting**  
`CUDA_VISIBLE_DEVICES`: Specifies the GPU devices available for the job (default: `0,1,2,3`). Adjust this variable to match your hardware configuration.  
`OMP_NUM_THREADS`: Sets the number of OpenMP threads for CPU‑parallel operations (default: 4). Modify based on your CPU core count.

**Arguments (required)**  
`--config_file`: Path to the Python configuration file defining the model and training setup.  
`--config_dict_name`:  Name of the Python dictionary variable (within the config file) that holds the configuration.

**Arguments (optional)**  
`--print_loss_step`: Interval (in training steps) for printing loss values to monitor training stability and detect NaN issues (default: 500).  
`--print_model_output_step`: Interval (in training steps) for logging model outputs to monitor training stability and detect NaN issues (default: 5000).

**Logging**  
The `nohup` command allows the process to continue running after disconnecting from the terminal.
`result/finetune_tutorial_encode.log`: All runtime messages (stdout and stderr) are redirected to this log file. You can change the path and filename as needed.  

**Expected output:**  
Upon successful fine-tuning, the following files will be generated in the directory specified by the `'output_folder'` key in `methylai_config_dict`:

1. Training Results File  
- **File:** `{output_result_file}` (as defined in the configuration)  
- **Format:** Tab-separated values  
- **Contents:** Per‑epoch training and validation metrics, including:
  - Training loss
  - Validation loss
  - Pearson correlation coefficient (PCC) on the validation set
  - Spearman correlation coefficient (SCC) on the validation set

2. Model Checkpoint
- **Directory:** `{output_folder}/checkpoint/`
- **Files:** Checkpoint files named `checkpoint_epoch_{epoch_number}.pth` (one per epoch)
- **Purpose:** These checkpoints allow you to resume training from any epoch or for downstream tasks.

---

## Fine-tuning Tutorial 2: Using Your Own WGBS Dataset

If you have WGBS data that has been processed with Bismark, you can fine‑tune MethylAI by following this tutorial.

### 1. Recommendations for Bismark Processing

1. Use the hg38 reference genome (or the same genome version that your model expects).
2. When running `bismark_methylation_extractor`, include the `--zero_based` flag. This will produce zero‑based coordinate files (with the suffix `.bedGraph.gz.bismark.zero.cov`) that are compatible with our preprocessing scripts.

### 2. Prepare train/validation/test dataset files

Move your Bismark output files (`.bedGraph.gz.bismark.zero.cov`) to a directory (e.g., `data/bismark`), or adjust the input folder path accordingly. Then run the following steps, which are analogous to those in Fine‑tuning Tutorial 1. For detailed descriptions of each argument and output, please refer to the corresponding sections in Tutorial 1.

#### 2.1 Preprocess Data

```bash
python src/preprocess/preprocess_bismark_data.py \
--input_folder data/bismark \
--input_file_suffix .bedGraph.gz.bismark.zero.cov \
--output_folder data/bismark_preprocess \
--output_log_file preprocess.log \
--reference_cpg_coordinate_file data/genome/cpg_coordinate_hg38.chr1-22.sort.bed.gz
```

#### 2.2 Obtain Raw and Smoothed Methylation Values

```bash
Rscript src/preprocess/bsmooth.R \
  data/bismark_preprocess \
  .preprocessed.txt \
  64 \
  smooth_methylation_info.txt \
  smooth_methylation_data.txt.gz \
  35 \
  500
```

#### 2.3 Generate train/validation/test dataset files

```bash
python src/preprocess/generate_dataset.py \
--smooth_methylation_file data/bismark_preprocess/smooth_methylation_data.txt.gz \
--data_info_file data/bismark_preprocess/smooth_methylation_info.txt \
--genome_fasta_file data/genome/hg38.fa \
--chrom_size_file data/genome/hg38.chrom.sizes \
--output_folder data/bismark_dataset \
--output_prefix bismark
```

**Note:** After completing these steps, you will have a processed dataset (in `data/bismark_dataset`) that can be used for fine‑tuning MethylAI.

### 3. Fine-tune the Model

#### 3.1 Copy and adapt the configuration file

Duplicate `config/finetune_tutorial_encode.py` as `config/finetune_tutorial_bismark.py`. Then update the following keys in the `methylai_config_dict` dictionary:

| Key                      | Description                                                                                          |
|--------------------------|------------------------------------------------------------------------------------------------------|
| 'output_folder'          | Absolute path to the directory where all fine‑tuning outputs will be saved.                          |
| 'output_result_file'     | Filename (within `output_folder`) that will store per‑epoch training/validation metrics.             |
| 'pretrain_snapshot_path' | Absolute path to the pre‑trained checkpoint (`MethylAI_pretrain_12_species.pth`).                    |
| 'train_set_file'         | Absolute path to your training set file (e.g., `data/bismark_dataset/bismark_train_set.pkl`).        |
| 'validation_set_file'    | Absolute path to your validation set file (e.g., `data/bismark_dataset/bismark_validation_set.pkl`). |
| 'genome_fasta_file'      | Absolute path to the reference genome FASTA file (`hg38.fa`).                                        |
| 'batch_size'             | Batch size, adjusted according to your GPU memory (default `50` is for RTX 4090 24GB).               |

**Crucial adjustment for sample count:**
Modify the `'output_block_dims'` key to match the number of samples in your Bismark dataset. The original value `(4 * 80, 4 * 5)` is designed for 4 samples. Replace the multiplier `4` with your actual sample count.  
Example: If you have `8` samples, set `'output_block_dims'`: `(8 * 80, 8 * 5)`.  
For a detailed explanation of all configuration parameters, refer to the **MethylAI Configuration Description** section.

#### 3.2 Run the fine‑tuning script

After saving the modified configuration file, execute the following command to start fine‑tuning:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
nohup torchrun --standalone --nproc_per_node=gpu \
src/training/finetune.py \
--config_file config/finetune_tutorial_bismark.py \
--config_dict_name methylai_config_dict \
--print_loss_step 500 \
--print_model_output_step 5000 \
> result/finetune_tutorial_bismark.log 2>&1 &
```

---

## Downstream Analysis Tutorial 1: Identification of DNA Methylation Linked Active TF Motif Sites

**Prerequisite**: Ensure you have completed Fine-tuning Tutorial 1 to generate the required dataset and fine-tuned model.

### 1. Preparation

Download the JASPAR Transcription Factor Binding Sites track from the UCSC Genome Browser and filter for high-confidence motif sites (match score > 400).  

**Note**: This step requires approximately 1 TB of free disk space.

```bash
# Download the JASPAR2024 track in bigBed format
wget -c -P data/genome https://hgdownload.soe.ucsc.edu/gbdb/hg38/jaspar/JASPAR2024.bb

# Convert bigBed to BED format
ucsc_tools/bigBedToBed data/genome/JASPAR2024.bb data/genome/JASPAR2024.bed

# Filter motifs with a match score > 400
awk -F'\t' '$5 > 400' data/genome/JASPAR2024.bed > data/genome/JASPAR2024_400.bed
```

### 2. Selection of Representative CpG Sites

This step selects representative CpG sites from hypomethylated regions for subsequent attribution score analysis.

```bash
python -u src/analysis_motif/get_low_me_region_representative_cpg.py \
--complete_dataset_file data/encode_dataset/encode_complete_dataset.txt \
--dataset_index 1 \
--output_folder data/encode_motif \
--output_prefix encode
```

**Expected Output**:

The following files will be generated in the specified `--output_folder`:

1. `encode_smooth_1_low_methylation_region.txt`  
- **Format**: Tab-separated values with header  
- **Columns**:
  - Columns 1–3: BED‑format CpG coordinates (`chr`, `start`, `end`)
  - Subsequent columns: Smoothed, raw, and regional methylation levels and coverage values
  - Region metadata: `region_id`, `region_cg_num`, `region_len`
  - Final column `represent_cpg`: Binary flag (0 or 1) indicating whether the site is a representative CpG (1 = selected)
2. `encode_smooth_1_low_methylation_region_representative_cpg.txt`
- Subset of the above file containing only rows where `represent_cpg` = 1.

**Arguments (required)**  
`--complete_dataset_file`: Path to the complete dataset file (e.g., `encode_complete_dataset.txt` generated in Fine‑tuning Tutorial 1).  
`--dataset_index`: Index of the sample to analyze. The mapping between sample filenames and indices is available in `data/encode_dataset/encode_dataset_info.txt`.  
`--output_folder`: Directory for output files.  
`--output_prefix`: Prefix for output filenames.  

**Arguments (optional)**  
`--threshold_low_methylation`: CpG sites with smoothed methylation level below this threshold are classified as hypomethylated. Default: 0.25  
`--threshold_min_cpg_number`: Hypomethylated regions containing fewer CpG sites than this value are filtered out. Default: 5  
`--threshold_min_region_length`: Hypomethylated regions shorter than this length (bp) are filtered out. Default: 50  
`--window_interval`: For regions shorter than this length (bp), the CpG site nearest the region center is selected. For longer regions, non‑overlapping windows of this length are defined, and the CpG nearest each window center is chosen. Default: 1000

### 3. Calculate DNA Sequence Attribution Scores Using DeepSHAP
This step uses the MethylAI model combined with the DeepSHAP algorithm to estimate sequence attribution scores for the DNA sequences corresponding to the representative CpG sites. These scores quantify the contribution of each nucleotide position to the predicted methylation level.  

**Note**: Based on our testing, processing ~100,000 representative CpG sites with default parameters on an RTX 4090 GPU requires approximately 24 hours.

**Full Analysis**
```bash
nohup python -u src/analysis_motif/get_sequence_attribution.py \
--representative_cpg_file data/encode_motif/encode_smooth_1_low_methylation_region_representative_cpg.txt \
--config_file configs/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--model_ckpt result/finetune_tutorial_encode/checkpoint/checkpoint_epoch_2.pth \
--gpu_id 0 \
--sample_name col_1 \
--model_output_index 0 \
--n_permutation 40 \
--output_folder result/finetune_tutorial_encode/motif_analysis \
> result/finetune_tutorial_encode/motif_analysis/get_sequence_attribution_smooth_1.log 2>&1 &
```

**Quick Test (First 1000 Sites)**  
For a quick pipeline test, you can run the analysis on a subset of 1000 representative CpG sites:

```bash
# Extract the first 1000 representative CpG sites (including header)
head -n 1001 data/encode_motif/encode_smooth_1_low_methylation_region_representative_cpg.txt \
  > data/encode_motif/encode_smooth_1_low_methylation_region_representative_cpg_head_1001.txt

nohup python -u src/analysis_motif/get_sequence_attribution.py \
--representative_cpg_file data/encode_motif/encode_smooth_1_low_methylation_region_representative_cpg_head_1001.txt \
--config_file configs/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--model_ckpt result/finetune_tutorial_encode/checkpoint/checkpoint_epoch_2.pth \
--gpu_id 0 \
--sample_name smooth_1 \
--model_output_index 0 \
--n_permutation 40 \
--output_folder result/finetune_tutorial_encode/motif_analysis \
> result/finetune_tutorial_encode/motif_analysis/get_sequence_attribution_smooth_1.log 2>&1 &
```

**Expected Output**:  
The script creates a folder named `{sample_name}_index_{model_output_index}` within the specified `--output_folder`. This folder contains:  
1. `representative_cpg_dataframe.txt`: Records the order of CpG sites processed for attribution scores.
2. `numpy` directory containing numerical results:  
- `attribution.npy`: Attribution scores.
- `delta.npy`: delta value. See (https://captum.ai/api/deep_lift_shap.html) for details.
- `dna_attribution.npy`: DNA‑level attribution.
- `dna_one_hot.npy`: One‑hot encoded DNA sequences.  

**Note**: `attribution.npy` and `dna_one_hot.npy` are formatted for direct input to TF‑MoDISco (https://github.com/kundajelab/tfmodisco).  

3. `bedgraph` directory (if --output_bedgraph is set):  
Contains per‑sequence attribution scores in bedGraph format. Files are named as `index_{model_output_index}_{chr}_{sequence_start}_{cpg_start}.bedgraph`. These can be converted to BigWig format (using `ucsc_tools/bedGraphToBigWig`) for visualization in genome browsers such as the WashU Epigenome Browser (https://epigenomegateway.wustl.edu/).

**Arguments (required)**  
`--representative_cpg_file`: Path to the `low_methylation_region_representative_cpg.txt` file generated in the step 2.  
`--config_file`: MethylAI configuration file (Python script). Note: This analysis also uses the `genome_fasta_file` path specified in the config.  
`--config_dict_name`: Name of the Python dictionary variable in the config file that holds the configuration.  
`--model_ckpt`: Path to the MethylAI checkpoint file.  
`--gpu_id`: ID of the GPU to use for computation.  
`--sample_name`: Descriptive name for the sample (used for naming output folders).  
`--model_output_index`: Index of the model output to compute attribution scores for. **Must correspond to the `dataset_index` used in step 2**. Mapping between sample filenames, dataset_index, and model_output_index is available in `data/encode_dataset/encode_dataset_info.txt`.  
`--n_permutation`: DeepSHAP parameter: number of permuted baseline sequences. This value is limited by GPU memory (40 recommended for RTX 4090 24 GB). Larger values increase computation time with minimal effect on results.  
`--output_folder`: Directory to store output files.

**Arguments (optional)**  
`--print_per_step`: Print progress every N sequences. Default: 500  
`--output_bedgraph`: If set, output attribution scores in bedGraph format for visualization.

### 4. Prediction Accuracy Evaluation of Representative CpG Sites

This step evaluates the prediction accuracy of MethylAI for the representative CpG sites. The evaluation is necessary because the calculation of sequence attribution scores using DeepSHAP assumes that MethylAI's predictions are accurate.

```bash
nohup python -u src/analysis_motif/evaluate_representative_cpg.py \
--representative_cpg_file data/encode_motif/encode_smooth_1_low_methylation_region_representative_cpg.txt \
--dataset_info_file data/encode_dataset/encode_dataset_info.txt \
--config_file configs/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--model_ckpt result/finetune_tutorial_encode/checkpoint/checkpoint_epoch_2.pth \
--gpu_id 0 \
--batch_size 200 \
--num_workers 8 \
--dataset_index 1 \
--output_folder result/finetune_tutorial_encode/motif_analysis \
--output_prefix encode \
--reverse_complement_augmentation
```

**Expected Output**  
- **File**: `encode_smooth_1_evaluation_dataframe.txt`  
- **Format**: Tab‑separated values with header  
- **Columns**:  
  - Columns 1‑3: BED‑format coordinates of representative CpG sites (`chr`, `start`, `end`)
  - Smoothed, raw, and regional methylation levels, and coverage values (as in the input file)
  - MethylAI predictions, with column names prefixed by `prediction_`

**Arguments (required)**  
`--representative_cpg_file`: Path to the representative CpG sites file generated in Step 2.  
`--dataset_info_file`: Path to the dataset information file (e.g., `encode_dataset_info.txt` from Fine‑tuning Tutorial 1).  
`--config_file`: MethylAI configuration file (Python script).  
`--config_dict_name`: Name of the Python dictionary variable in the config file that holds the configuration.  
`--model_ckpt`: Path to the MethylAI checkpoint file.  
`--gpu_id`: ID of the GPU to use for computation.  
`--batch_size`: Batch size for the DataLoader during inference.
`--num_workers`: Number of parallel workers for the DataLoader.  
`--dataset_index`: Index of the sample to analyze (must match the dataset_index used in Step 2).  
`--output_folder`: Directory to store output files.  
`--output_prefix`: Prefix for output filenames.  

**Arguments (optional)**  
`--print_per_step`: Print progress every N steps. Default: 200  
`--reverse_complement_augmentation`: If set, enable reverse complement data augmentation during inference.

### 5. Motif Attribution Score Statistic
This step calculates motif attribution scores by aggregating sequence attribution scores across all representative CpG sites that overlap with transcription factor binding motifs (from `JASPAR2024_400.bed` generated in Step 1).

```bash
nohup python -u src/analysis_motif/get_motif_statistic.py \
--sequence_attribution_folder result/finetune_tutorial_encode/motif_analysis/smooth_1_index_0 \
--jaspar_bed_file data/genome/JASPAR2024_400.bed \
--output_folder result/finetune_tutorial_encode/motif_analysis/smooth_1_index_0/motif_statistic \
--output_prefix encode_smooth_1 \
> result/finetune_tutorial_encode/motif_analysis/get_motif_statistic_smooth_1.log 2>&1 &
```

**Expected Output**  
- **File**: `encode_smooth_1_motif_statistic_dataframe.txt`  
- **Format**: Tab‑separated values with header  
- **Columns**:  
  - Columns 1‑8: Information from JASPAR2024_400.bed:
    - `chr`, `motif_start`, `motif_end`: motif coordinates
    - `motif_id`: JASPAR motif ID
    - `motif_score`: motif match score
    - `motif_strand`: motif strand
    - `motif_name`: motif name
    - `motif_len`: motif length
  - Next columns: Relationship between the motif and the representative CpG site:
    - `cg_start`, `cg_end`: CpG site coordinates
    - `input_dna_start`, `input_dna_end`: start and end positions of the input DNA sequence
    - `motif_cg_distance`: distance between the motif and the CpG site
  - Final columns: Attribution score statistics:
    - `motif_attribution_sum`: sum of attribution scores over the motif site
    - `motif_attribution_mean`: mean attribution score over the motif site
    - `motif_attribution_abs_sum`: sum of absolute attribution scores over the motif site
    - `motif_attribution_abs_mean`: mean absolute attribution score over the motif site

**Arguments (required)**  
`--sequence_attribution_folder`: Path to the folder containing sequence attribution results from Step 3 (e.g., `smooth_1_index_0`).  
`--jaspar_bed_file`: Path to the filtered JASPAR motif file (`JASPAR2024_400.bed`) generated in Step 1.  
`--output_folder`: Directory to store output files.  
`--output_prefix`: Prefix for output filenames.  

**Arguments (optional)**  
`--print_per_step`: Print progress every N steps. Default: 500  
`--captum_cpg_file`: Specify a different file for representative CpG information (if changed from default). Default: `representative_cpg_dataframe.txt` (in sequence_attribution_folder)  
`--dna_attribution_file`: Specify a different file for DNA attribution scores (if changed from default). Default: `numpy/dna_attribution.npy` (in sequence_attribution_folder)|

### 6. Identification of Active Motif Sites

This step filters for active motif sites based on a set of thresholds to identify transcription factor binding motifs that exhibit significant attribution scores within hypomethylated regions.

```bash
python -u src/analysis_motif/get_active_motif.py \
--motif_statistic_file result/finetune_tutorial_encode/motif_analysis/smooth_1_index_0/motif_statistic/encode_smooth_1_motif_statistic_dataframe.txt \
--captum_cpg_file result/finetune_tutorial_encode/motif_analysis/smooth_1_index_0/captum_cpg_dataframe.txt \
--evaluation_file result/finetune_tutorial_encode/motif_analysis/encode_col_1_evaluation_dataframe.txt \
--dataset_index 1 \
--output_folder result/finetune_tutorial_encode/motif_analysis/smooth_1_index_0/active_motif \
--output_prefix smooth_1 \
--bedtools_path `which bedtools` > result/finetune_tutorial_encode/motif_analysis/get_active_motif.log 2>&1 &
```

**Expected Output**  

The following files will be generated in the specified `--output_folder`:  

1. `evaluation_dataframe.txt`: Prediction accuracy metrics for each representative CpG site.  

2. `smooth_1_motif_statistic.txt`: Tab‑separated file containing unfiltered motif statistics, including:  
   - Representative CpG site information: `cg_chr_start` (CpG ID), `low_me_region_id`, `cg_start`, `cg_end`, `input_dna_start`, `input_dna_end`, `smooth`, `prediction_smooth`, `coverage`, `abs_diff_smooth`, `smooth_index`.  
   - Motif information: `motif_id_name`, `motif_id`, `motif_name`, `motif_start`, `motif_end`, `motif_len`, `motif_strand`, `motif_relative_start`, `motif_relative_end`, `motif_cg_distance`, `motif_score`.  
   - Attribution score statistics: `motif_attribution_sum`, `motif_attribution_abs_sum`, `motif_attribution_mean`, `motif_attribution_abs_mean`, `motif_activation_score`.  

3. `smooth_1_motif_statistic_filtered.txt`: Filtered version of the above, after applying `--threshold_max_motif_cpg_distance` and `--threshold_max_prediction_error`.  

4. `smooth_1_active_motif_statistic.txt`: Further filtered to retain only motifs with attribution scores passing `--threshold_motif_attribution_mean`.  
5. `smooth_1_active_motif_summary.txt`: Summary per TF motif (`motif_id_name`), including mean motif_activation_score and counts of hypomethylated regions/windows where the motif is active.  

6. `smooth_1_all_motif.bed`: BED file of all motif sites from smooth_1_motif_statistic_filtered.txt.  
7. `smooth_1_active_motif.bed`: BED file of active motif sites from smooth_1_active_motif_statistic.txt.  
8. `smooth_1_inactive_motif.bed`: BED file of inactive motif sites, generated by subtracting active motifs from all motifs (bedtools intersect -v).

**Arguments (required)**  
`--motif_statistic_file`: Path to the motif statistics file (`motif_statistic_dataframe.txt`) from Step 5.  
`--captum_cpg_file`: Path to the representative CpG file (`representative_cpg_dataframe.txt`) from Step 3.  
`--evaluation_file`: Path to the prediction accuracy file (`evaluation_dataframe.txt`) from Step 4.  
`--dataset_index`: Sample index (must match the index used in Step 2).
`--output_folder`: Directory to store output files.  
`--output_prefix`: Prefix for output filenames.  

**Arguments (optional)**  
`--methylation_type`: Type of methylation region to analyze: `low` (hypomethylated) or `high` (hypermethylated). Default: `low`  
`--threshold_max_motif_cpg_distance`: Maximum allowed distance (bp) between the motif site and the representative CpG site. Default: 1000  
`--threshold_max_prediction_error`: Maximum allowed prediction error for the representative CpG site. Set to a large value (e.g., 1.0) to disable this filter. Default: 0.2  
`--threshold_motif_attribution_mean`: Threshold on the mean attribution score. Use negative values for hypomethylation analysis and positive values for hypermethylation analysis. Default: -0.02  
`--bedtools_path`: Path to the bedtools executable. Required to generate the `inactive_motif.bed` file.

### Note
In this tutorial, we provide a complete workflow for analyzing active TF motif sites within hypomethylated regions. This pipeline has been extensively validated in our research, ensuring its robustness and reliability. Several identified active motifs and their corresponding TFs have been experimentally confirmed to regulate DNA methylation in hypomethylated regions (see our MethylAI bioRxiv preprint in the **Citation** section).

We plan to extend this framework in future releases to include:
- Analysis of hypermethylated regions
- Custom analysis of user‑specified genomic elements for DNA‑methylation‑linked active TF motif discovery

---
## Downstream Analysis Tutorial 2: Interpreting GWAS Variants
**Prerequisite:** Complete **Fine‑tuning Tutorial 1** to generate the required dataset and fine‑tuned model, and **Downstream Analysis Tutorial 1** to obtain active motif sites.

**Rationale:** Variants that intersect with active motif sites are more biological interpretable (e.g., a variant may affect DNA methylation by altering a transcription factor binding site). Furthermore, our mQTL validation shows that MethylAI achieves >87% accuracy in predicting the direction of methylation changes for variants located within active motif sites (see the MethylAI bioRxiv preprint in the **Citation** section).

This tutorial guides you through identifying variants that lie within active motif sites of hypomethylated regions and using MethylAI to screen for variants predicted to increase DNA methylation in these regions.

### 1. Preparation

We use the dbSNP database file `00-common_all.vcf.gz` as an example. You can replace this with your own set of GWAS variants.

```bash
# Create a directory for variant data
mkdir -p data/variant

# Download the dbSNP common variants VCF
wget -c -P data/variant https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz

# Decompress the files
gunzip data/variant/00-common_all.vcf.gz
gunzip data/genome/cpg_coordinate_hg38.chr1-22.sort.bed.gz

# Add 'chr' prefix to chromosome column in the VCF (to match BED format)
awk 'BEGIN{OFS="\t"} {if($0 !~ /^#/) $1="chr"$1; print}' data/variant/00-common_all.vcf \
> data/variant/00-common_all_withchr.vcf
```

### 2. Generate Variant Dataset

This step identifies variants overlapping active motif sites (obtained in Downstream Analysis Tutorial 1) and extracts nearby CpG sites to create a dataset for variant effect prediction.

#### 2.1 Intersect Variants with Active Motif Sites

Use `bedtools intersect` to select variants that overlap active motif sites:

```bash
# Extract variants overlapping active motif sites (unique records)
bedtools intersect -a data/variant/00-common_all.vcf \
-b result/finetune_tutorial_encode/motif_analysis/smooth_1_index_0/active_motif/smooth_1_active_motif.bed \
-wa -u -header \
> data/variant/00-common_all_intersect_smooth_1_active_motif.vcf

# Extract variants with detailed overlap information (including motif coordinates)
bedtools intersect -a data/variant/00-common_all_withchr.vcf \
-b result/finetune_tutorial_encode/motif_analysis/smooth_1_index_0/active_motif/smooth_1_active_motif.bed \
-wa -wb \
> data/variant/00-common_all_intersect_smooth_1_active_motif_detail.txt
```

#### 2.2 Extract CpG Sites within ±1 kb of Variants

Use `bedtools window` to identify CpG sites within a 1-kb window of each variant:

```bash
bedtools window -a data/variant/00-common_all_intersect_smooth_1_active_motif.vcf \
-b data/genome/cpg_coordinate_hg38.chr1-22.sort.bed \
-w 1000 \
> 00-common_all_intersect_smooth_1_active_motif_1k_cpg.txt
```

#### 2.3 Create Variant‑CpG Dataset

Generate a processed dataset suitable for variant effect prediction:

```bash
mkdir -p data/variant_dataset

python src/analysis_variant/get_variant_cpg_dataset.py \
--input_variant_cpg_file data/variant/00-common_all_intersect_smooth_1_active_motif_1k_cpg.txt \
--output_variant_cpg_dataset_file data/variant_dataset/00-common_all_intersect_smooth_1_active_motif_1k_cpg_dataset.txt
```

**Expected Output:**  
- `data/variant_dataset/00-common_all_intersect_smooth_1_active_motif_1k_cpg_dataset.txt`
  - **Format**: Tab‑separated values with header
  - **Purpose**: Processed dataset ready for variant effect calculation in the next step

**Arguments (required)**  
`--input_variant_cpg_file`: Path to the variant‑CpG file generated in step 2.2 (`00-common_all_intersect_smooth_1_active_motif_1k_cpg.txt`).  
`--output_variant_cpg_dataset_file`: Path for the output variant dataset file.

### 3. Predict Variant Effects on DNA Methylation

This step uses MethylAI to predict the effect of variants on DNA methylation levels by comparing predictions for reference and alternative alleles.

```bash
# Create output directory
mkdir result/finetune_tutorial_encode/variant_analysis

nohup python -u src/analysis_variant/get_variant_effect.py \
--variant_dataset_file data/variant_dataset/00-common_all_intersect_active_motif_1k_cpg_dataset.txt \
--dataset_info_file data/encode_dataset/encode_dataset_info.txt \
--config_file configs/finetune_tutorial_encode.py \
--config_dict_name methylai_config_dict \
--model_ckpt result/finetune_tutorial_encode/checkpoint/checkpoint_epoch_2.pth \
--gpu_id 0 \
--batch_size 100 \
--num_workers 8 \
--dataset_index 1 \
--output_folder result/finetune_tutorial_encode/variant_analysis/smooth_1 \
--output_prefix 00-common_all_intersect_smooth_1_active_motif \
--reverse_complement_augmentation \
> result/finetune_tutorial_encode/variant_analysis/get_variant_effect.log 2>&1 &
```
**Expected Output:**  
`00-common_all_intersect_smooth_1_active_motif_variant_prediction_dataframe.txt`
- **Format**: Tab-separated values with header
- **Contents**:
  - Columns 1‑12: Variant and CpG information from the input dataset:
chr, POS, RSID, REF, ALT, ALT_split, variant_start, variant_ref_len, variant_alt_len, variant_cg_distance, cg_start, cg_end
  - Subsequent columns, MethylAI predictions for the reference and alternative alleles. 
    - `ref_prediction_*`: Methylation predictions for the reference allele
    - `alt_prediction_*`: Methylation predictions for the alternative allele
  - Final column: `cg_change` (binary flag, 0 or 1) indicates whether the variant disrupts a canonical CpG dinucleotide sequence (1 = disrupted). Note: Rows with `cg_change = 1` should be filtered out in downstream analyses.

**Arguments (required)**  
`--variant_dataset_file`: Path to the variant dataset file generated in the step 2 (`00-common_all_intersect_smooth_1_active_motif_1k_cpg_dataset.txt`).  
`--dataset_info_file`: Path to the dataset information file (e.g., `encode_dataset_info.txt` from Fine‑tuning Tutorial 1).  
`--config_file`: MethylAI configuration file (Python script). Note: This analysis also uses the `genome_fasta_file` path specified in the config.  
`--config_dict_name`: Name of the Python dictionary variable in the config file that holds the configuration.  
`--model_ckpt`: Path to the MethylAI checkpoint file.  
`--gpu_id`: ID of the GPU to use for computation.  
`--batch_size`: Batch size for the DataLoader during inference.  
`--num_workers`: Number of parallel workers for the DataLoader.  
`--dataset_index`: Index of the sample to analyze (must match the dataset index used for active motif identification). For other analyses, you may specify multiple indices separated by spaces, or use `0` to analyze all samples.  
`--output_folder`: Directory to store output files.  
`--output_prefix`: Prefix for output filenames.

**Arguments (optional)**  
`--print_per_step`: Print progress every N steps. Default: 500  
`--reverse_complement_augmentation`: Enable reverse complement data augmentation during inference.

### 4. Analyze Variant Effects

This step filters for variants that are predicted to increase DNA methylation in hypomethylated regions and associates them with overlapping motif information.

```bash
python -u src/analysis_variant/analyze_variant_effect.py --variant_effect_file result/finetune_tutorial_encode/variant_analysis/00-common_all_intersect_smooth_1_active_motif_variant_prediction_dataframe.txt \
--variant_active_motif_detail_file data/variant/00-common_all_intersect_smooth_1_active_motif_detail.txt --dataset_index 1 \
--output_folder result/finetune_tutorial_encode/variant_analysis/smooth_1 \
--output_prefix 00-common_all_intersect_smooth_1_active_motif \
> result/finetune_tutorial_encode/variant_analysis/analyze_variant_effect.log 2>&1 &
```

**Note**: Variants that disrupt CpG dinucleotides (i.e., rows with cg_change = 1) are automatically filtered out in this step, as the focus is on non‑CpG‑altering variants that may modulate methylation levels.

**Expected Output:**  
`00-common_all_intersect_smooth_1_active_motif_variant_analysis_result.txt`  
- **Format**: Tab‑separated values with header
- **Contents**: 
  - Columns 1‑12: Variant and CpG information from the input dataset (same as the previous step): chr, POS, RSID, REF, ALT, ALT_split, variant_start, variant_ref_len, variant_alt_len, variant_cg_distance, cg_start, cg_end
  - Next columns: MethylAI predictions for reference and alternative alleles (same as input):
    - `ref_prediction_*`, `alt_prediction_*`
  - New columns:
    - `effect_*`: Predicted variant effect, calculated as `MethylAI(ALT allele) – MethylAI(REF allele)`
    - motif_start, motif_end, motif_id_name: Information of the active motif that overlaps the variant (from `variant_active_motif_detail.txt`)

For each variant, the CpG site with the largest absolute effect size is retained as the representative site.

**Arguments (required)**  
`--variant_effect_file`: Path to the variant effect file (`00-common_all_intersect_smooth_1_active_motif_variant_prediction_dataframe.txt`) generated in the step 3.  
`--variant_active_motif_detail_file`: Path to the variant‑motif overlap detail file (`00-common_all_intersect_smooth_1_active_motif_detail.txt`) generated in Step 2.1.  
`--dataset_index`: Index of the sample to analyze (**must match the dataset index used for active motif identification**).  
`--output_folder`: Directory to store output files.  
`--output_prefix`: Prefix for output filenames.  

**Arguments (optional)**  
`--methylation_type`: Type of methylation region to analyze: `low` (hypomethylated) or `high` (hypermethylated). Default: `low`  
`--threshold_min_variant_effect`: Minimum predicted variant effect threshold. Use **positive** values for hypomethylation analysis (to select variants that increase methylation) and **negative** values for hypermethylation analysis. Default: 0.01

### Note

This tutorial provides a workflow for analyzing variants that intersect active motif sites within hypomethylated regions.

If you wish to analyze **all variants** (without filtering for active motif overlap), you can skip step 2.1 (variant‑motif intersection) and use the original VCF file (00-common_all_withchr.vcf) directly in step 2.2. In this case, you should adjust the --threshold_min_variant_effect in step 4 to a more stringent value to maintain reliability. We recommend:

- Hypomethylation analysis: `--threshold_min_variant_effect 0.1` 
- Hypermethylation analysis: `--threshold_min_variant_effect -0.1`

Skipping step 2.1 means that the motif association step (in step 4) will not be performed, and the output will lack motif‑related columns (`motif_start`, `motif_end`, `motif_id_name`). To disable motif association in step 4, set the argument --variant_active_motif_detail_file to an empty string (`""`).

We plan to extend this framework in future releases to include:
- Analysis of variants in hypermethylated regions
- More flexible variant‑effect analysis pipelines for broader genomic contexts

## MethylAI Configuration Description

The configuration dictionary (`methylai_config_dict`) controls all aspects of model architecture, training, and data handling. Below is a detailed description of each parameter, grouped by function.

### Trainer Parameters

| Parameter                            | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                 |
|--------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| total_epoch_number                   | 3          | Total number of training epochs. Recommended: 3 for fine‑tuning, 5 for training from scratch.                                                                                                                                                                                                                                                                                               |
| max_step_per_epoch                   | 10_000_000 | Maximum number of steps per epoch. Set to a large value (e.g., 10_000_000) to disable early stopping within an epoch.                                                                                                                                                                                                                                                                       |
| learning_rate                        | 0.0001     | Learning rate for all parts of MethylAI except the output block. Recommended: 0.0001 for fine‑tuning, 0.0006 for training from scratch.                                                                                                                                                                                                                                                     |
| output_block_learning_rate           | 0.0006     | Learning rate for the output block. Recommended: 0.0006.                                                                                                                                                                                                                                                                                                                                    |
| weight_decay                         | 0.01       | Weight decay (L2 regularization) coefficient.                                                                                                                                                                                                                                                                                                                                               |
| batch_size                           | 50         | Batch size. Set to 50 for an RTX 4090 (24 GB); scale proportionally for GPUs with more memory.                                                                                                                                                                                                                                                                                              |
| output_folder                        | –          | Absolute path to the directory where all training outputs (checkpoints, logs, etc.) are saved.                                                                                                                                                                                                                                                                                              |
| output_result_file                   | –          | Filename (within output_folder) that records per‑epoch training and validation metrics.                                                                                                                                                                                                                                                                                                     |
| pretrain_checkpoint_path             | –          | Absolute path to the pre‑trained checkpoint file (e.g., MethylAI_pretrain_12_species.pth).                                                                                                                                                                                                                                                                                                  |
| is_load_output_block_pretrain_weight | False      | Whether to load pre‑trained weights for the output block. Set to True only when pre‑training and fine‑tuning use the same dataset; in that case, also set output_block_learning_rate to 0.0001.                                                                                                                                                                                             |
| checkpoint_path                      | None       | Absolute path to a checkpoint from which to resume training (e.g., after an interruption). Note: If both pretrain_checkpoint_path and checkpoint_path are set, only checkpoint_path takes effect. checkpoint_path also restores optimizer and scheduler states, while pretrain_checkpoint_path loads only model weights. If both are set to `None`, the model will be trained from scratch. |
| is_run_validation_at_first           | False      | Whether to run validation before the first training epoch (useful for debugging).                                                                                                                                                                                                                                                                                                           |

### Model Architecture Parameters

| Parameter                          | Default                        | Description                                                                                                                                                                                                                                                                                                                                                                                  |
|------------------------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_block_channel                | [20, 190, 30]                  | Number of channels allocated to each kernel size in the first multi‑scale convolutional layer.                                                                                                                                                                                                                                                                                               |
| input_block_kernel_size            | [3, 9, 21]                     | Kernel sizes in the first multi‑scale convolutional layer. Must have the same length as input_block_channel. The i‑th element of input_block_channel corresponds to the i‑th kernel size (e.g., 20 channels for kernel size 3).                                                                                                                                                              |
| input_block_additional_conv_layer  | (9, 9)                         | Kernel sizes of additional convolutional layers after the multi‑scale layer. (9, 9) adds two convolutional layers with kernel size 9, each followed by batch normalization and GELU activation.                                                                                                                                                                                              |
| input_block_exponential_activation | True                           | Whether to use exponential activation after the first multi‑scale convolutional layer. If False, GELU is used instead.                                                                                                                                                                                                                                                                       |
| width                              | [300, 360, 420, 480, 540, 600] | Channel counts for each multi‑scale CNN block.                                                                                                                                                                                                                                                                                                                                               |
| depth                              | [2, 2, 2, 2, 2, 2]             | Number of repeated convolutional layers in each multi‑scale CNN block.                                                                                                                                                                                                                                                                                                                       |
| kernel_size                        | [9, 9, 9, 9, 9, 9]             | Kernel sizes for each multi‑scale CNN block.                                                                                                                                                                                                                                                                                                                                                 |
| stride                             | [4, 4, 4, 4, 4, 2]             | Stride for each multi‑scale CNN block. The total input DNA length is the product of all strides multiplied by the last kernel size (e.g., 4×4×4×4×4×2×9 = 9×2¹¹ = 18,432).                                                                                                                                                                                                                   |
| output_block_dims                  | (4 * 80, 4 * 5)                | Dimensions of the output block. Each tuple element specifies the number of neurons in a hidden layer; the last element is the output layer size. Additional hidden layers can be added by extending the tuple (e.g., (4×80, 4×40, 4×5)). The default (320, 20) corresponds to a hidden layer of 320 neurons and an output layer of 20 neurons (4 samples × 5 methylation levels per sample). |

### Dataset & DataLoader Parameters

| Parameter                          | Default | Description                                                                                                                                                                                                                                                     |
|------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_dna_length                   | 9*2**11 | Length of the DNA sequence input to MethylAI (must match the product of strides and the last kernel size).                                                                                                                                                      |
| train_num_workers                  | 2       | Number of workers for the training DataLoader. Recommended: 2.                                                                                                                                                                                                  |
| validation_num_workers             | 4       | Number of workers for the validation DataLoader. Recommended: 4.                                                                                                                                                                                                |
| loss_weight_factor                 | 5       | Factor used to scale loss weights based on CpG coverage.                                                                                                                                                                                                        |
| max_loss_weight_factor             | 1.0     | Maximum allowed loss weight.                                                                                                                                                                                                                                    |
| minimal_coverage                   | 5       | Minimum coverage required for a CpG site to contribute to the loss. Sites with coverage below this threshold receive a loss weight of 0. With the default settings, loss weights are binary: 0 for coverage <5, 1 otherwise.                                    |
| is_keep_raw_methylation            | True    | Whether to use raw methylation levels during training (mainly for experimental purposes).                                                                                                                                                                       |
| is_reverse_complement_augmentation | True    | Whether to use reverse‑complement sequences for data augmentation. When True, the training set is doubled by adding the reverse‑complement of every CpG sequence. During validation, predictions for the forward and reverse‑complement sequences are averaged. |

### Learning Rate Scheduler Parameters

| Parameter                | Default | Description                                                                                                                                                                              |
|--------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| warmup_lr_epoch_number   | 1       | Number of epochs for linear learning‑rate warm‑up. Values can be floats (e.g., 0.5 for half an epoch).                                                                                   |
| constant_lr_epoch_number | 1       | Number of epochs with a constant learning rate after warm‑up. Values can be floats (e.g., 0.5 for half an epoch). After these epochs, the CosineAnnealingWarmRestarts scheduler is used. |

### Training/Validation Set Paths

| Parameter                    | Default | Description                                                                                                                                                                                                                                                             |
|------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| train_set_file               | –       | Absolute path to the training set file (e.g., encode_train_set.pkl).                                                                                                                                                                                                    |
| validation_set_file          | –       | Absolute path to the validation set file (e.g., encode_validation_set.pkl).                                                                                                                                                                                             |
| genome_fasta_file            | –       | Absolute path to the reference genome FASTA file (e.g., hg38.fa).                                                                                                                                                                                                       |
| cpg_index_to_repetition_dict | {}      | If provided, enables oversampling of hard‑to‑learn CpG sites (e.g., cell‑type‑specific sites). The dictionary maps {path_to_cpg_file: repetition_count}. Multiple entries are allowed. Code for generating the oversampling files will be provided in a future release. |

**Important**: Always use absolute paths for file paths to avoid path‑related errors.

## Reference
This project utilizes and/or references the following libraries and packages:
- R package bsseq: https://bioconductor.org/packages/release/bioc/html/bsseq.html
- Bismark: https://github.com/FelixKrueger/Bismark
- Captum (implementation of DeepSHAP): https://captum.ai/
- DeepSHAP: https://arxiv.org/abs/1705.07874
- wgbstools: https://github.com/nloyfer/wgbs_tools
- bedtools: https://github.com/arq5x/bedtools2

## Citation
If you use MethylAI in your research, please cite our preprint/publication:  
- https://www.biorxiv.org/content/10.1101/2025.11.20.689274v1

