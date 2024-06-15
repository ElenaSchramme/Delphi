<center width="30%" style="padding:10px">
    <p>
        <img src=".github/delphi-logo-white-bg.svg" width="300" style="weight:150px"/>
    </p>
</center>

# Delphi

## Learning the natural history of human disease with generative transformers

[[`Paper`](https://www.medrxiv.org/content/10.1101/2024.06.07.24308553v1)] [[`BibTeX`](#Citation)]

Artem Shmatko*, Alexander Wolfgang Jung*, Kumar Gaurav*, Søren Brunak, Laust Mortensen, Ewan Birney, Tom Fitzgerald, Moritz Gerstung (*Equal Contribution)

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


## Repository Overview

This repository contains code of the modified GPT-2 model used in the paper "Learning the natural history of human disease with generative transformers", along with the training code and analysis notebooks.

The implementation is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Installation

1. Download the repository:

```bash
git clone https://github.com/gerstung-lab/Delphi.git
cd Delphi
```

2. Create a virtual conda environment and install the requirements:
```bash
conda create -n delphi python=3.11
conda activate delphi
pip install -r requirements.txt
```

Installing the requirements normally takes a few minutes.

## Data availability

Delphi-2M is trained on 500K patient health trajectories from the UK Biobank data, which is available to researchers upon [application](https://www.ukbiobank.ac.uk/).
In addition, we provide a synthetic dataset that was generated by the trained model itself. The synthetic data is statistically similar to the real data, while not disclosing any patient information.

## Training

To train the model, run:

```bash
python train.py config/train_delphi_demo.py --device=cuda --out_dir=Delphi-2M
```

If you want to train the model on a CPU, remove the `--device=cuda` argument.
For more information on the training configuration, check the `config/train_delphi_demo.py` file.

Training a demo model takes around 10 minutes on a single GPU.

Training the original model took 1 GPU-hour (NVIDIA V100, CentOS 7). Training on M1 Macbook Pro's MPS takes around 10 hours.

## Notebooks

After training the model, the obtailed checkpoints can be used to run the analysis notebooks.
Please note that in order to reproduce the results from the paper, it is necessary to train the model on the UK Biobank data. 

The `notebooks` folder contains the following notebooks:

`evaluate_delphi.ipynb`: The notebook is used to evaluate Delphi-2M in terms of prediction accuracy compared to age-sex-based epidemiological baseline and calibration of the predicted risks. Furthermore, it explores the attention mechanism of the model and the structure of the latent space using UMAP of the learned disease embeddings.

`shap_analysis.ipynb`: The notebook is used to analyse the SHAP values of the model to understand the importance of the input disease events for the model's predictions for a given patient. Then, using SHAP values aggregated over the entire dataset, we identify the diseases that are most important for each possible disease in the future.

`sampling_trajectories.ipynb`: The notebook is used to analyse the synthetic data generated by Delphi-2M. It compares the synthetic data to the real data in terms of the distribution of disease events, ages and disease rates.

## Citation

```bibtex
@article{Shmatko2024.06.07.24308553,
	title = {Learning the natural history of human disease with generative transformers},
    author = {Shmatko, Artem and Jung, Alexander Wolfgang and Gaurav, Kumar and Brunak, S{\o}ren and Mortensen, Laust and Birney, Ewan and Fitzgerald, Tom and Gerstung, Moritz},
	doi = {10.1101/2024.06.07.24308553},
	journal = {medRxiv},
	publisher = {Cold Spring Harbor Laboratory Press},
	year = {2024}
}
```