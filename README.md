# T4E-Miner

T4E-Miner is a structure- and function-guided weak-label expansion framework for mining tagatose 4-epimerases from metagenomic sequence space.

## Overview

Protein language models have shown great potential for enzyme discovery, but their application to rare enzyme families is often limited by the scarcity of experimentally validated data and the presence of annotation noise.

T4E-Miner addresses this limitation by integrating global structural similarity with conserved function-related catalytic-site features to generate functionally restricted weak labels for task-specific protein language model adaptation. The framework was developed for the discovery of tagatose 4-epimerases, which catalyze the one-step conversion of D-fructose to the rare sugar D-tagatose.

This repository provides the core scripts used for model training, candidate retrieval and inference in T4E-Miner.

## Repository structure

```text
T4E-Miner/
├── ckpt/                  # placeholder for model checkpoints
├── data/                  # placeholder for input data
├── dataset/               # dataset loading modules
├── examples/              # example input files
├── model/                 # model definition
├── result/                # placeholder for output results
├── scripts/               # training, retrieval and inference scripts
├── environment.yml        # conda environment file
├── LICENSE
└── README.md
Dataset and checkpoint

Large datasets and model checkpoint files are not included in this repository.

Please place the required input data under the data/ directory and model checkpoints under the ckpt/ directory before running T4E-Miner.

The expected checkpoint files include:

ckpt/
├── pretrained protein language model weights
└── fine-tuned T4E-Miner checkpoint

The processed datasets and trained checkpoints will be released through an external archive before publication.

Installation

Clone this repository:

git clone https://github.com/Xiaoyuan916/T4E-Miner.git
cd T4E-Miner

Create the conda environment:

conda env create -f environment.yml
conda activate T4E_find
Training

The model can be fine-tuned using:

python scripts/train.py

Please modify the data paths, checkpoint paths and training parameters in the script according to your local file organization.

Inference

Candidate sequences can be predicted using:

python scripts/inference.py \
    --model_path ckpt/esmc_300m_2024_12_v0.pth \
    --checkpoint_path ckpt/t4e_miner_finetuned.pth \
    --inference_data examples/example_sequences.fasta \
    --output_path result/

The inference script outputs prediction scores and predicted labels for all input sequences.

Retrieval

Candidate retrieval and downstream screening can be performed using:

python scripts/retrieval.py

Please modify the input and output paths in the script before running.

Example input

An example FASTA file is provided in:

examples/example_sequences.fasta

Input sequences should be provided in FASTA format:

>sequence_id
MSEQUENCE...
Notes

This repository contains the core implementation of T4E-Miner. Large-scale metagenomic datasets, intermediate files, trained checkpoints and prediction results are not stored in the GitHub repository because of file-size limitations.

Citation

Citation information will be added after publication.

License

This project is released under the MIT License. See the LICENSE file for details.

Part of the code structure was adapted from the ESM-Ezy project under the MIT License.
