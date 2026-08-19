# T4E-Miner

T4E-Miner is a structure- and function-guided weak-label expansion framework for mining tagatose 4-epimerases (T4Es) from metagenomic sequence space.

T4Es catalyze the one-step conversion of D-fructose to the rare sugar D-tagatose. T4E-Miner combines structure- and catalytic-site-guided weak-label construction with task-specific fine-tuning of the ESM-C protein language model to prioritize candidate T4Es for experimental validation.

This repository provides the code used for:

- ESM-C model fine-tuning;
- model inference on candidate protein sequences; and
- sequence-identity-based candidate binning.

Processed datasets and trained model checkpoints are provided separately through Zenodo.

## Workflow

The T4E-Miner workflow consists of four main stages:

1. **Weak-label construction:** experimentally characterized T4Es are used as seed enzymes for structure-guided homolog retrieval, followed by catalytic-site conservation filtering.
2. **Model fine-tuning:** the curated positive and negative datasets are used to fine-tune the terminal layers of ESM-C 300M together with a binary classification head.
3. **Metagenomic inference:** the fine-tuned model predicts and scores candidate T4Es from metagenomic protein sequences.
4. **Candidate stratification:** predicted candidates are grouped according to their sequence identity to the experimentally characterized T4E A0A662GG58 to guide representative selection for experimental validation.

## Repository structure

```text
T4E-Miner/
├── ckpt/                              # directory for model checkpoints
├── data/                              # directory for processed datasets
├── dataset/                           # dataset loading modules
├── examples/
│   └── example_sequences.fasta        # example inference input
├── model/                             # ESM-C classification model
├── result/                            # default output directory
├── scripts/
│   ├── train.py                       # five-fold model fine-tuning
│   ├── inference.py                   # sequence-level model inference
│   ├── sequence_identity_binning.py   # candidate identity binning
│   └── retrieval.py                   # representation-based retrieval utility
├── utils/                             # I/O, metrics and reproducibility utilities
├── environment.yml                    # Conda environment specification
├── LICENSE
└── README.md
```

## Data and model checkpoint

Large processed datasets and model checkpoint files are not stored directly in this GitHub repository because of their file sizes. They are archived in Zenodo:

> **Zenodo:** [processed datasets and trained checkpoint] (https://doi.org/10.5281/zenodo.21290155)

After downloading the archive, place the files under `data/` and `ckpt/`, respectively. The expected dataset layout for model training is:

```text
data/dataset_split/
├── folds/
│   ├── fold_1/
│   │   ├── train_pos.fasta
│   │   ├── train_neg.fasta
│   │   ├── val_pos.fasta
│   │   └── val_neg.fasta
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── fold_5/
└── test_set/
    ├── test_pos.fasta
    └── test_neg.fasta
```

## Weak-label construction

The structure-similarity search used for weak-label construction was performed through the **Foldseek web server** rather than a custom local search script.

Briefly, structures of 17 experimentally characterized T4Es were individually submitted as queries. Experimentally resolved structures or structures obtained from the AlphaFold Protein Structure Database were used when available; AlphaFold predictions were used otherwise. Structures with an average pLDDT below 70 were excluded. Foldseek hits satisfying a TM-score threshold of 0.7 were collected and merged.

The retrieved candidates were subsequently examined for conservation of the catalytically important Glu-Asp-His-His (EDHH) residues. Because bacterial and archaeal T4Es differ in their local sequence contexts and residue numbering, catalytic-site conservation was inspected separately according to taxonomic origin. After structural and catalytic-site filtering, 282 weak-positive homologs were combined with the 17 seed T4Es to produce the final positive dataset of 299 sequences.

The processed weak-positive dataset and associated dataset splits are provided in the Zenodo archive. No custom Foldseek search script is included because the structural searches were performed through the web server.

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/Xiaoyuan916/T4E-Miner.git
cd T4E-Miner

conda env create -f environment.yml
conda activate T4E-Miner
```

The model requires access to the pretrained ESM-C 300M weights. These weights may be supplied as a local model path when running the scripts.

## Model fine-tuning

The final T4E-Miner configuration fine-tunes the last four ESM-C transformer layers together with the classification head. The model was trained using a batch size of 8, a maximum of 50 epochs and an early-stopping patience of 15.

```bash
python scripts/train.py \
    --folds_dir data/dataset_split \
    --model_path ckpt/esmc_300m_2024_12_v0.pth \
    --batch_size 8 \
    --epoch 50 \
    --last_layers 4 \
    --patience 15 \
    --save_path result/training
```

The training script runs the five cluster-separated development folds and evaluates each fold-specific model on the fixed held-out test set. The best checkpoint for each fold is selected according to validation F1 score.

## Model inference

Input sequences must be provided in FASTA format:

```text
>sequence_id
MSEQUENCE...
```

An example input file is available at `examples/example_sequences.fasta`.

Run inference using:

```bash
python scripts/inference.py \
    --model_path ckpt/esmc_300m_2024_12_v0.pth \
    --checkpoint_path ckpt/t4e_miner_finetuned.pth \
    --inference_data examples/example_sequences.fasta \
    --output_path result/inference \
    --batch_size 64
```

The output file `all_inference_scores.tsv` contains:

- the sequence identifier;
- logits for classes 0 and 1;
- class probabilities; and
- the predicted class label.

## Sequence-identity binning

Candidate stratification was performed according to sequence identity to the experimentally characterized T4E A0A662GG58. Sequence identities were calculated separately using MAFFT and supplied to the binning script in a CSV column named `Identity(%)`.

The binning script does **not** calculate sequence identity. It assigns candidates with precomputed identity values to the following intervals:

```text
<20%, 20-30%, 30-40%, 40-50%,
50-60%, 60-70%, 70-80%, 80-100%
```

Example input:

```csv
Sequence_ID,Identity(%)
candidate_1,28.4
candidate_2,36.7
candidate_3,81.2
```

Run candidate binning using:

```bash
python scripts/sequence_identity_binning.py \
    --input-csv data/candidates_with_identity.csv \
    --output-dir result/sequence_bins \
    --identity-column "Identity(%)"
```

The script produces a complete binned table and a separate CSV file for every non-empty identity interval.

## Reproducibility notes

- The random seed used by the training script is 24 unless otherwise specified.
- Positive samples and hard negatives were clustered at 40% sequence identity before cluster-separated dataset partitioning.
- The fixed held-out test set was reserved before model training and was not used for model selection.
- The processed dataset splits and trained checkpoint should be downloaded from the accompanying Zenodo record.
- Large metagenomic input files, predicted structures and intermediate Foldseek outputs are not stored in this repository.

## Citation

If you use T4E-Miner, please cite the associated article. 

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

Part of the code structure was adapted from the ESM-Ezy project under the MIT License.
