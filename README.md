# MethylAI: A Deep Learning Model for Predicting DNA Methylation from DNA Sequence

[![license](https://img.shields.io/badge/python_-3.10.15_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/PyTorch_-2.4.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/Captum_-0.6.8_-purple)](https://captum.ai/)
[![license](https://img.shields.io/badge/R_-4.3.3_-red)](https://www.r-project.org/)

MethylAI is a convolutional neural network (CNN) model designed to predict DNA methylation levels at CpG sites and genomic regions directly from one-hot encoded DNA sequences centered on the target CpG sites. Beyond accurate prediction, MethylAI is engineered for interpretability to uncover the biological mechanisms underlying DNA methylation regulation.

## Key Features & Highlights

### Comprehensive and Multi-Species Training Data

**Largest Human WGBS Dataset:** Trained on the most extensive collection of human whole-genome bisulfite sequencing (WGBS) data to date, comprising 1,574 samples spanning 52 tissues and 171 cell types.

**Cross-Species Pre-training:** Enhanced model accuracy through pre-training on WGBS data from **11 mammalian species**, including mouse (*Mus musculus*), rat (*Rattus norvegicus*), macaque (*Macaca fascicularis* and *Macaca mulatta*), chimpanzee (*Pan troglodytes*), gorilla (*Gorilla gorilla*), cow (*Bos taurus*), sheep (*Ovis aries*), dog (*Canis lupus familiaris*), pig (*Sus scrofa*), giant panda (*Ailuropoda melanoleuca*).

### Advanced Model Architecture

**Multi-scale CNN Module:** Captures sequence features at varying resolutions to improve predictive accuracy.

**Exponential Activation Function:** Increases model interpretability by providing more biologically meaningful intermediate representations.

### Sophisticated Training Strategy

**Pre-training + Fine-tuning:** Leverages cross-species data for pre-training, followed by human-specific fine-tuning, resulting in superior prediction performance.

**Multi-task Prediction:** Simultaneously predicts methylation levels for the 1,574 human samples:

- at each CpG site,

- average methylation levels over genomic regions of different lengths (200 bp, 500 bp, and 1 kb).


## Main Applications

MethylAI can be leveraged for advanced functional genomics analyses:

### Identification of Cis-Regulatory Elements:

Integrated with the DeepSHAP algorithm, MethylAI can identify key sequence features (e.g., transcription factor binding motifs) that influence DNA methylation levels, providing insights into the cis-regulatory code of DNA methylation.

### Predicting the Regulatory Impact of Genetic Variants:

By introducing sequence variations in silico, MethylAI can predict the effect of genetic variants (e.g., SNPs) or allelic differences on local DNA methylation patterns, aiding in the functional interpretation of non-coding genomic variants.

## Installation Guide

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Yu-Lab-Genomics/MethylAI.git
cd MethylAI
```

#### Step 2: Create Environment and Install Dependencies

```bash
conda create -n methylai python=3.10
conda activate methylai
pip install -r requirements.txt
```

#### Step 3: Download model checkpoints

You can download the pretrained model checkpoints below. Place the downloaded model directory in the main path

- [Pretrained Model](https://sctp4m.aigenomicsyulab.com/): pretrained with human dataset and other 11 mammalian species

- [Full model](https://sctp4m.aigenomicsyulab.com/): 1574 human samples

- [ENCODE model](https://sctp4m.aigenomicsyulab.com/): 96 human samples from [ENCODE project](https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=WGBS&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens) (127 samples were available from ENCODE project, but only 96 samples passed our quality control)

- [Human DNA methylation atlas model](https://sctp4m.aigenomicsyulab.com/): 207 human samples from a [nature paper](https://www.nature.com/articles/s41586-022-05580-6)

## Demo

### Run the Inference Pipeline

```bash
python ./04.scPIT_inference/scPIT_inference.py --gpu 5 \
--model_ckpt ./02.checkpoint/model_weights.pth \
--output_path ./04.scPIT_inference/demo_output
```

Arguments:  
--gpu: ID of the GPU to use. Default is GPU 5.  
--model_ckpt: Path to the trained scPIT model checkpoint (already provided in ./02.checkpoint/).  

### Expected Output



## Fine-tuning with your WGBS-seq data



### Data Pre-Processing & Dataset Splitting



### Model Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 nohup torchrun --standalone --nproc_per_node=gpu
```

## Downstream Applications

### Interpretability Analysis

### Variants effect prediction



