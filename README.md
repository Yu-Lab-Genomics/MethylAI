# MethylAI: comprehensive sequence-based DNA methylation prediction model
===========================================================================

[![license](https://img.shields.io/badge/python_-3.10.15_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/PyTorch_-2.4.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/Captum_-0.6.8_-purple)](https://captum.ai/)
[![license](https://img.shields.io/badge/R_-4.3.3_-red)](https://www.r-project.org/)

## Installation

MethyAI is implemented based on Pytorch. We use pytorch-2.4.1 and cuda-12.8. Other version could be also compatible. We highly recommend using Anaconda to manage your Python environment. This ensures a consistent and reproducible setup for running our model. To create the recommended environment, please follow these steps:

1.  **Install Anaconda:** If you haven't already, download and install Anaconda from the official website: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)

2.  **Create the environment:** Navigate to the root directory of this repository in your terminal or Anaconda Prompt. Then, execute the following command to create the CAPTAIN environment:

    ```bash
    conda env create -n methyai
    ```

3.  **Activate the environment:** Once the environment is created, activate it using the following command:

    ```bash
    conda activate methyai
    ```
4.  **Install requried packages:** We have included a `requirements.txt` file. This file lists the necessary Python packages required to run the model:

    ```bash
    pip install -r requirements.txt
    ```



