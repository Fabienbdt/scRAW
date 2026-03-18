# scRAW

scRAW is a clustering pipeline for single-cell RNA-seq data stored as `AnnData` objects.

This repository follows a single execution path:

1. preprocessing of a raw `.h5ad` matrix with normalization, `log1p`, HVG selection, and scaling;
2. MLP autoencoder training with a warm-up phase and a weighted phase;
3. pseudo-label updates in latent space and dynamic cell weighting;
4. final clustering on the learned embedding;
5. metric computation and figure export.

It is driven by:

- a JSON file for hyperparameters and paths;
- a notebook to launch and inspect one run;
- a small Python API in `src/scraw/`.

## Repository layout

- `configs/default_scraw.json`: default experiment configuration.
- `notebooks/scraw_demo.ipynb`: notebook example to run the pipeline.
- `src/scraw/config.py`: JSON-backed configuration objects.
- `src/scraw/preprocessing.py`: default preprocessing path.
- `src/scraw/model.py`: autoencoder definition and device helpers.
- `src/scraw/trainer.py`: training loop, masking, triplet loss, and dynamic weighting.
- `src/scraw/clustering.py`: pseudo-label and final clustering helpers.
- `src/scraw/metrics.py`: evaluation metrics.
- `src/scraw/plots.py`: plotting helpers.
- `src/scraw/pipeline.py`: end-to-end execution entry point for notebooks/scripts.

## Installation

```bash
cd scRAW
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configuration

The main parameters are defined in `configs/default_scraw.json`:

- `data`: input `.h5ad` path, output directory, optional label column;
- `runtime`: random seed and device selection;
- `preprocessing`: cell/gene filters, target sum, HVG count, scaling;
- `model`: hidden layers, latent dimension, dropout;
- `training`: epochs, warm-up, batch size, learning rate, masking;
- `weighting`: dynamic rare-cell weighting parameters;
- `triplet`: optional rare-cell triplet loss;
- `clustering`: pseudo-label method and final HDBSCAN settings;
- `outputs`: figure and model export flags.

## Notebook usage

Open `notebooks/scraw_demo.ipynb`. The notebook is structured as follows:

- it defines a version of the scRAW training procedure, including the autoencoder, pseudo-label update, dynamic weighting, and rare-cell triplet loss;
- it loads and preprocesses one input dataset;
- it runs the training procedure and displays the evaluation metrics and the UMAP comparison of ground-truth and predicted labels.

For a shorter programmatic entry point, the equivalent high-level usage is:

```python
from scraw import load_config, run_pipeline

config = load_config("configs/default_scraw.json")
result = run_pipeline(config)
result["metrics"]
```

## Outputs

A run writes the following files by default:

- `config/config_used.json`
- `results/results.json`
- `results/analysis_results.csv`
- `results/embeddings.npy`
- `results/final_labels.npy`
- `results/pseudo_labels.npy`
- `results/cell_weights.npy`
- `models/autoencoder.pt`
- `figures/loss_history.png`
- `figures/latent_clusters.png`
- `figures/latent_weights.png`
- `figures/latent_ground_truth.png` when labels are available
