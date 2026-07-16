# scRAW

scRAW is a rare-cell-aware clustering pipeline for single-cell RNA-seq data
stored as `AnnData` (`.h5ad`). It learns an autoencoder representation, updates
pseudo-labels and cell weights during training, optionally removes batch signal
with an adversarial objective, and performs final clustering in latent space.

The first priority of this repository is to reproduce the Baron human pancreas
reference experiment. Use the pinned Linux/CUDA environment and the unchanged
default configuration below for that purpose. A portable installation and a
short CPU smoke test are provided for development convenience, but they are not
reference-result protocols.

## Reference protocol at a glance

| Goal | Environment | Configuration | Command |
| --- | --- | --- | --- |
| Reproduce one reference seed | Linux x86_64, Python 3.12.3, CUDA 12.4, pinned dependencies | `configs/default_scraw.json` | `scraw run --config configs/default_scraw.json --device cuda` |
| Reproduce the 10-seed experiment | Same pinned environment | Same default configuration; seeds set by the script | `bash scripts/run_default_10seeds.sh` |
| Check that the pipeline works | Any supported portable environment | `configs/smoke_cpu.json` | `scraw run --config configs/smoke_cpu.json` |
| Develop or test the package | Portable environment | Test fixtures | `python -m pytest` |

Two protocol details are intentional and must be retained when comparing with
the reference results:

1. **Ground-truth class count is an intended prior.** When `pseudo_k` is unset
   (`0`) and the configured label column is available, scRAW deliberately uses
   the number of unique ground-truth classes as the target number of
   pseudo-clusters. For the bundled Baron data, `Group` contains 14 classes.
   Only the class count is used by this step, not the per-cell class identity.
   This is part of the reference protocol, not a bug.
2. **The current HDBSCAN noise policy is intentional.** With the default
   `hdbscan_reassign_noise: false`, raw HDBSCAN noise points (`-1`) are retained
   together as one additional output cluster and all labels are then remapped to
   contiguous integers. They are not dropped from evaluation. Setting the flag
   to `true` instead assigns noise points to the nearest non-noise centroid.
   Do not change this setting when reproducing the reference results.

More detail is available in [the architecture guide](docs/architecture.md).

## Exact Baron reference reproduction

### 1. Use the authoritative platform

The validated reference stack is:

- Linux `x86_64`
- Python `3.12.3`
- NVIDIA CUDA `12.4`
- the exact package versions in `requirements.txt`

`requirements.txt` is the authoritative lock for numerical reproduction.
`requirements-cuda124.txt` is only a compatibility alias for the same lock.
Do not substitute `requirements-portable.txt` for a reference run.

Start from a clean clone and record the exact revision:

```bash
git clone https://github.com/Fabienbdt/scRAW.git
cd scRAW
git rev-parse HEAD
```

For an already cloned repository, inspect local work before synchronizing it.
A destructive reset discards local changes:

```bash
git status --short
git fetch --prune origin
git reset --hard origin/main
git rev-parse HEAD
```

### 2. Verify the reference inputs

The repository includes the combined Baron human pancreas dataset. Verify both
the data and experiment configuration before running:

```bash
sha256sum \
  data/baron_human_pancreas.h5ad \
  configs/default_scraw.json \
  requirements.txt
```

Expected values:

```text
aa784472d90c2a6cdb99cdd076525b2dbe83a7254543092eaccca25bc60e359e  data/baron_human_pancreas.h5ad
f2a6550ffae5c14f800b5cbdfdae41a91be8f0ef671d825448867155792dd767  configs/default_scraw.json
fcb0eb645938dce438406c2a1d6ff7f212d400cce274681c7b37de26baee9d1f  requirements.txt
```

The input contains 8,569 cells, 20,125 genes, 14 `Group` labels, and four donor
batches. See [the data record](data/README.md) for provenance and columns.

### 3. Create the pinned environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The final `--no-deps` is important: it installs the `scraw` command without
allowing package metadata to replace the already pinned reference stack.

Run the strict built-in reference diagnostic before starting training:

```bash
scraw doctor --reference
```

It verifies the operating system, machine architecture, exact Python/CUDA and
package versions, and SHA256 values of the dataset, default configuration, and
reference dependency lock. It prints a JSON report and exits with status `1` if
any reference check fails. Resolve every failed item before comparing numerical
results. `scraw doctor` without `--reference` is informational and does not fail
on a portable environment.

Check the environment before a long run:

```bash
python --version
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
scraw --version
scraw show-config --config configs/default_scraw.json > /tmp/scraw-default-normalized.json
```

For traceability, also save the commit, driver information, and resolved Python
environment next to your results or in a separate run manifest:

```bash
mkdir -p reproducibility
git rev-parse HEAD > reproducibility/git-commit.txt
nvidia-smi > reproducibility/nvidia-smi.txt
python -m pip freeze > reproducibility/pip-freeze.txt
```

### 4. Run one reference seed

The default configuration uses seed `42`, the bundled dataset, strict
deterministic settings, and the published reference hyperparameters:

```bash
scraw run \
  --config configs/default_scraw.json \
  --device cuda \
  --output-dir results/reference_seed42
```

The CLI refuses to write into an existing non-empty output directory. This
protects completed experiments. Use a new directory for each run; use
`--overwrite` only when replacement is deliberate.

### 5. Run the reference 10-seed experiment

The provided launcher runs seeds `1,42,43,44,45,46,47,48,49,50`, writes one
directory per seed, reuses already completed seeds, and records a combined log:

```bash
SCRAW_PYTHON="$PWD/.venv/bin/python" \
SCRAW_DEVICE=cuda \
bash scripts/run_default_10seeds.sh
```

Useful optional variables are `SCRAW_OUTPUT_ROOT`, `SCRAW_LOG_DIR`,
`SCRAW_MACHINE_TAG`, `SCRAW_DATA_PATH`, `SCRAW_CONFIG`, and
`SCRAW_SEEDS_CSV`. Changing the dataset, configuration, seed list, or device
creates a different experiment and must be reported as such.

Strict deterministic PyTorch settings reduce run-to-run variation. Exact
bitwise equality can still depend on the GPU model, NVIDIA driver, and system
libraries, so use the same machine class when exact numerical comparison is
required. Compare the saved `config_used.json`, input SHA256, environment
manifest, and per-seed `results.json` before interpreting a discrepancy.

## Portable installation (convenience only)

For CPU, macOS, Linux without the reference CUDA stack, or routine development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the compatible version ranges in `requirements-portable.txt`.
It is useful for onboarding and functional testing, but it does not reproduce
the authoritative reference environment.

To add tests and development tools:

```bash
python -m pip install -r requirements-dev.txt
```

## Functional CPU smoke test

`configs/smoke_cpu.json` copies the reference values and changes only operational
scale/output choices: CPU, two epochs, one warmup epoch, a smoke output path, and
disabled figures/checkpoint. The short schedule does not reach the reference
triplet or adversarial start epochs. It answers only “does the base training,
weight refresh, clustering, metrics, and artifact path execute?” It cannot be
used to compare scientific results.

```bash
scraw run --config configs/smoke_cpu.json
```

## Command-line interface

```text
scraw run         train the model and write clustering outputs
scraw infer       replay preprocessing, encoding, and clustering from a checkpoint
scraw show-config print built-in defaults or normalize a JSON configuration
scraw doctor      inspect the host, dependencies, and reference-file hashes
scraw --version   print the installed package version
```

Run `scraw COMMAND --help` for all overrides. `python -m scraw` exposes the same
interface.

Examples:

```bash
# Built-in defaults
scraw run

# Explicit dataset and label/batch columns
scraw run \
  --config configs/default_scraw.json \
  --data-path data/baron_human_pancreas.h5ad \
  --label-key Group \
  --batch-key batch \
  --output-dir results/baron_explicit \
  --device cuda

# Inspect the effective defaults
scraw show-config
```

## Checkpoint replay

The saved checkpoint contains the trained autoencoder weights. Replay uses the
saved training configuration to recompute preprocessing, encode cells, run
final clustering, and recompute metrics:

```bash
scraw infer \
  --config results/reference_seed42/config/config_used.json \
  --checkpoint results/reference_seed42/models/autoencoder.pt \
  --output-dir results/reference_seed42_replay \
  --device cuda
```

Checkpoint replay is intended for the **same dataset and configuration** as the
training run. Preprocessing state is not embedded in the checkpoint; it is
recomputed. A different dataset may have a different filtered/HVG feature space
and is not a valid reference replay even when dimensions happen to match.

## Pipeline

1. Load an `.h5ad` dataset.
2. Filter cells and genes.
3. For count-like input, normalize totals, apply `log1p`, select highly variable
   genes, standardize, and clip values. Negative input values signal an already
   transformed matrix; normalization, `log1p`, and HVG selection are skipped.
4. Train a symmetric MLP autoencoder with masked reconstruction.
5. After warmup, refresh latent pseudo-labels and rare-cell weights at the
   configured interval.
6. Add the optional rare-cell triplet objective and batch-adversarial branch.
7. Run final HDBSCAN clustering with the documented fallback and noise policy.
8. Compute metrics and export arrays, summaries, figures, and the optional model.

## Configuration

The built-in defaults live in `src/scraw/config.py`; the authoritative reference
copy is `configs/default_scraw.json`.

| Section | Purpose |
| --- | --- |
| `data` | Input `.h5ad`, output directory, and evaluation label key |
| `runtime` | Seed, device, and strict deterministic settings |
| `preprocessing` | Input mode, cell/gene filters, normalization, HVGs, scaling, and clipping |
| `model` | Hidden layers, latent dimension, and dropout |
| `training` | Epoch schedule, batch size, learning rate, masking, and clipping |
| `weighting` | Cluster-frequency and latent-density cell weights |
| `triplet` | Rare-cell-focused semi-hard triplet objective |
| `clustering` | Pseudo-label method, target K, and final HDBSCAN policy |
| `batch_correction` | Optional gradient-reversal batch adversary |
| `outputs` | Figure and checkpoint persistence |

JSON files may be partial: omitted fields use the built-in defaults. For a
scientific run, save and compare the fully resolved
`config/config_used.json` written by the pipeline.

### Label-count prior

The `clustering.pseudo_k` resolution order is:

1. an explicit value greater than `1`;
2. otherwise, the number of unique labels when a label column is available;
3. otherwise, a bounded heuristic based on the number of cells.

The second case is the expected Baron reference behavior. It provides the known
number of biological classes as a clustering prior. Removing that prior changes
the protocol and will not reproduce the reference experiment.

### HDBSCAN final labels

HDBSCAN is the default final clustering method. If it raises an error or finds
at most one usable non-noise cluster, scRAW falls back to Leiden and then KMeans
if needed. In a successful HDBSCAN run:

- `hdbscan_reassign_noise: false` groups all raw `-1` observations into one
  additional cluster before contiguous remapping (the reference behavior);
- `hdbscan_reassign_noise: true` assigns each raw `-1` observation to its nearest
  non-noise centroid before contiguous remapping.

## Outputs

A full training run writes this pipeline-managed tree:

```text
results/reference_seed42/
├── config/
│   └── config_used.json
├── results/
│   ├── results.json
│   ├── analysis_results.csv
│   ├── embeddings.npy
│   ├── final_labels.npy
│   ├── pseudo_labels.npy
│   ├── cell_weights.npy
│   ├── cell_ids.npy
│   └── feature_ids.npy
├── models/
│   └── autoencoder.pt
└── figures/
    ├── loss_history.png
    ├── latent_clusters.png
    ├── latent_weights.png
    └── latent_ground_truth.png
```

`results.json` records the resolved label/batch keys, known label count,
effective pseudo K, processed matrix size, resolved device, dependency/runtime
provenance, input path/SHA256, metric bundle, and loss history.
`analysis_results.csv` contains the scalar metrics in one row. `cell_ids.npy`
and `feature_ids.npy` preserve the identity and order of the processed matrix.
Figure/model files are omitted when disabled.

## Repository layout

```text
configs/                  reference and smoke configurations
data/                     bundled dataset and provenance record
docs/architecture.md      data flow and module responsibilities
notebooks/scraw_demo.ipynb explanatory, non-reference notebook
scripts/                  reference multi-seed and checkpoint launchers
src/scraw/                installable package
tests/                    unit and interface tests
requirements.txt          authoritative pinned CUDA 12.4 reference stack
requirements-cuda124.txt  compatibility alias for requirements.txt
requirements-portable.txt portable package dependencies
requirements-test.txt     test dependencies
requirements-dev.txt      development dependencies
```

The notebook is designed to explain the training procedure and may use a
shortened configuration. Its displayed metrics are not the canonical reference
acceptance target. Use the CLI, default JSON, and pinned environment above for
result reproduction.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, tests, code style,
and the rules for protocol-sensitive changes.

```bash
python -m pytest
python -m ruff check src tests
python -m build
python -m twine check dist/*
```

When reporting a reproducibility issue, include the commit SHA, dataset and
configuration SHA256 values, `pip freeze`, GPU/driver information, command,
complete log, and generated `config_used.json`.
