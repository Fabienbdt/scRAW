# scRAW

scRAW is a single-cell RNA-seq clustering pipeline for `AnnData` datasets.
It combines preprocessing, representation learning, pseudo-label refinement,
cell reweighting, optional batch-adversarial training, and final clustering in
latent space.

## Pipeline Overview

The default pipeline follows these steps:

1. Load a `.h5ad` dataset.
2. Apply cell and gene filtering.
3. Run normalization, `log1p`, HVG selection, and scaling when the input looks
   like raw counts.
4. Train an MLP autoencoder with masking, dynamic cell weighting, and optional
   triplet and batch-adversarial objectives.
5. Update pseudo-labels in latent space during training.
6. Run final clustering on the learned embedding.
7. Compute metrics and export results, arrays, and figures.

If the input matrix already contains negative values, scRAW assumes that the
data has already been preprocessed and skips normalization, `log1p`, and HVG
selection.

## Repository Layout

- `configs/default_scraw.json`: default experiment configuration.
- `notebooks/scraw_demo.ipynb`: notebook example for one complete run.
- `src/scraw/config.py`: configuration dataclasses and JSON loading.
- `src/scraw/preprocessing.py`: preprocessing path for raw or preprocessed data.
- `src/scraw/model.py`: autoencoder and embedding utilities.
- `src/scraw/trainer.py`: training loop, pseudo-label updates, weighting, and
  adversarial branch.
- `src/scraw/clustering.py`: pseudo-label and final clustering helpers.
- `src/scraw/metrics.py`: evaluation metrics.
- `src/scraw/plots.py`: plotting helpers.
- `src/scraw/pipeline.py`: end-to-end execution entry point.

## Installation

For the validated Baron run, use the same environment as the reference local
run:

- Python `3.12.3`
- Linux `x86_64`
- NVIDIA CUDA `12.4`
- dependency versions from `requirements.txt`

This repository is meant to be run directly from the source tree. Install the
pinned dependencies from `requirements.txt`, then add the repository `src/`
directory to `PYTHONPATH` with an absolute path.

```bash
cd scRAW
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The absolute path is intentional: using a relative `src` path can fail if
Python is launched from a different working directory.

## Run scRAW

The default configuration in `configs/default_scraw.json` is the recommended
entry point. It uses:

- `data/baron_human_pancreas.h5ad`
- `results/default_run`
- `seed = 42`
- `device = "cuda"`
- `strict_repro = true`

Run scRAW from the repository root with:

```bash
cd scRAW
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from scraw import load_config, run_pipeline; config = load_config("configs/default_scraw.json"); result = run_pipeline(config); print(result["metrics"]); print(result["output_dir"])'
```

## Configuration

The default configuration lives in `configs/default_scraw.json`.
The main sections are:

- `data`: input dataset, output directory, optional label key.
- `runtime`: random seed and device selection.
  The default runtime also enables strict deterministic PyTorch/CUDA settings.
- `preprocessing`: filtering, target sum, HVG selection, and scaling.
- `model`: hidden layers, latent dimension, and dropout.
- `training`: epochs, batch size, learning rate, masking, and gradient clip.
- `weighting`: dynamic rare-cell weighting parameters.
- `triplet`: optional triplet-loss settings.
- `clustering`: pseudo-label method and final HDBSCAN settings.
- `batch_correction`: optional adversarial batch correction settings.
- `outputs`: control figure and model export.

Label and batch columns can be provided explicitly in the configuration. When
they are not set, scRAW tries to detect common column names automatically.

## Outputs

By default, one run writes:

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
- `figures/latent_ground_truth.png` when ground-truth labels are available

## Notes

- The default configuration is set up for the Baron human pancreas example
  dataset included in `data/`.
