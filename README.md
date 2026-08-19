# T4E-Miner

T4E-Miner is a structure- and function-guided weak-label expansion framework for mining tagatose 4-epimerases (T4Es) from metagenomic protein sequences.

The framework combines structure-guided homolog retrieval, catalytic-site conservation filtering and task-specific fine-tuning of the ESM-C 300M protein language model.

## Data and checkpoints

Processed datasets and trained model checkpoints are available from Zenodo:

> [T4E-Miner data and checkpoints](https://doi.org/10.5281/zenodo.21290156)

Download and extract the two archives in the repository root:

```bash
unzip T4E-Miner_data.zip
unzip T4E-Miner_ckpt.zip
```

The training dataset will be located at:

```text
data/train_data_new/dataset_split/
```

The checkpoint archive contains five fold-specific models:

```text
ckpt/t4e_miner_fold1_best.pth
ckpt/t4e_miner_fold2_best.pth
ckpt/t4e_miner_fold3_best.pth
ckpt/t4e_miner_fold4_best.pth
ckpt/t4e_miner_fold5_best.pth
```

The Fold 1 checkpoint was used for the reported MGnify inference.

## Installation

```bash
git clone https://github.com/Xiaoyuan916/T4E-Miner.git
cd T4E-Miner

conda env create -f environment.yml
conda activate T4E-Miner
```

T4E-Miner uses the official ESM-C 300M model. Use the model-registry name `esmc_300m`; do not pass a direct `.pth` path as the base model.

The official pretrained weights are downloaded automatically on first use and reused from the local Hugging Face cache.

## Model training

The final configuration fine-tunes the last four ESM-C transformer layers using a batch size of 8, a maximum of 50 epochs, early-stopping patience of 15 and random seed 24.

```bash
python scripts/train.py \
    --folds_dir data/train_data_new/dataset_split \
    --model_path esmc_300m \
    --batch_size 8 \
    --epoch 50 \
    --last_layers 4 \
    --patience 15 \
    --random_seed 24 \
    --save_path result/training
```

The script trains five fold-specific models and evaluates each model on the fixed held-out test set.

Small numerical differences may occur between independent GPU training runs because of CUDA floating-point operations and early stopping.

## Model inference

Run inference using the published Fold 1 checkpoint:

```bash
python scripts/inference.py \
    --model_path esmc_300m \
    --checkpoint_path ckpt/t4e_miner_fold1_best.pth \
    --inference_data examples/example_sequences.fasta \
    --output_path result/inference \
    --batch_size 64
```

Results are saved to:

```text
result/inference/all_inference_scores.tsv
```

The output contains sequence identifiers, class logits, class probabilities and predicted labels.

## Sequence-identity binning

Candidates with precomputed sequence identities can be grouped using:

```bash
python scripts/sequence_identity_binning.py \
    --input-csv data/candidates_with_identity.csv \
    --output-dir result/sequence_bins \
    --identity-column "Identity(%)"
```


## Citation

If you use T4E-Miner, please cite the associated article and Zenodo record:

```text
Geng, X. T4E-Miner processed datasets and trained model checkpoints.
Zenodo. https://doi.org/10.5281/zenodo.21290156
```

## License

This project is released under the MIT License.
