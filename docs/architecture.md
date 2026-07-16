# scRAW architecture

This guide describes the execution path, the responsibility of each module,
and the protocol-sensitive behavior that must remain stable for result
reproduction. The public entry points are the `scraw` command and the
`scraw.run_pipeline` Python API.

## End-to-end data flow

```text
JSON / CLI overrides
        │
        ▼
ScRAWConfig ──► runtime validation and deterministic seeding
        │
        ▼
AnnData load ──► label and batch-column resolution
        │
        ▼
preprocessing ──► dense float32 cells × selected features
        │
        ▼
MLP autoencoder warmup
        │
        ▼
pseudo-label refresh ──► cluster-frequency + density cell weights
        │                         │
        ├─────────────────────────┘
        ▼
weighted reconstruction + optional triplet + optional batch adversary
        │
        ▼
final latent embedding ──► HDBSCAN (Leiden/KMeans fallback)
        │
        ├──► evaluation metrics
        └──► arrays, JSON/CSV, checkpoint, and figures
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed configuration sections, default values, JSON merge/load/save |
| `cli.py` | `run`, `infer`, and `show-config`; overrides; output safety; concise errors |
| `preprocessing.py` | Filtering, input-mode resolution, normalization/HVG path, scaling |
| `model.py` | Symmetric MLP autoencoder, gradient reversal, device resolution, batched encoding |
| `trainer.py` | Deterministic setup, warmup/weighted phases, dynamic weights, losses, optimization |
| `clustering.py` | Pseudo-label K resolution helpers, Leiden/KMeans pseudo-labels, final HDBSCAN |
| `metrics.py` | Clustering, class-balanced, rare-class, neighborhood, and silhouette metrics |
| `plots.py` | Loss and 2D latent-space figures |
| `pipeline.py` | AnnData orchestration, metadata/provenance, artifacts, checkpoint replay |

Imports in `scraw.__init__` remain lightweight; the scientific stack is loaded
only when a pipeline entry point is called.

## Configuration resolution

`ScRAWConfig.from_dict` merges each JSON section with the built-in dataclass
defaults. This permits small operational configurations while preserving a
single source of defaults. The resolved configuration is always written to
`config/config_used.json` and should be retained with published results.

The CLI loads that object and applies only overrides explicitly passed by the
caller. It refuses a non-empty output directory unless `--overwrite` is given.
This check is an artifact-safety mechanism, not a training parameter.

### Input and metadata columns

`data.label_key` controls the biological labels used for the known class count,
metrics, and ground-truth figures. `batch_correction.key` controls donor/batch
identities used by the adversarial branch.

- A non-null explicit key must exist in `adata.obs`; a misspelled or absent key
  is an error.
- A JSON `null` requests automatic detection from the supported conventional
  column names.
- The reference JSON explicitly uses `Group` and `batch`.

This distinction prevents a typo from silently selecting a different protocol.

### Preprocessing input mode

`preprocessing.input_mode` has three values:

- `raw`: force the count-data path (filter, normalize totals, `log1p`, select
  HVGs, then standardize and clip);
- `preprocessed`: skip normalization, `log1p`, and HVG selection, then
  standardize and clip the supplied features;
- `auto`: preserve the historical behavior: negative values indicate already
  transformed input; otherwise the raw-count path is used.

The reference JSON intentionally remains byte-for-byte stable and omits this
newer field; omission resolves to the implicit built-in default `auto`. Set the
mode explicitly for a new dataset when its representation is known, and record
the resolved config.
The matrix supplied to the trainer is dense `float32`; its row and feature IDs
are saved with the output arrays.

## Training state machine

### 1. Runtime setup

The trainer resolves `auto`, `cpu`, `cuda`, `cuda:N`, or `mps`, seeds Python,
NumPy, and PyTorch, and enables deterministic PyTorch/CUDA behavior when
`strict_repro` is true. Invalid device strings fail instead of silently using
CPU. An unavailable requested accelerator may fall back to CPU with a warning;
check the recorded device before comparing results.

### 2. Autoencoder warmup

The symmetric MLP encoder is:

```text
input → [Linear → BatchNorm → LeakyReLU → Dropout] × hidden layers → latent
```

The decoder mirrors the hidden layers and returns to the processed feature
dimension. Random feature masking creates the denoising reconstruction task.
Before `warmup_epochs`, all cells have neutral weight `1.0`.

### 3. Known class-count prior and pseudo-labels

The effective pseudo-cluster target is resolved once at the start of training:

1. use an explicit `clustering.pseudo_k > 1`;
2. otherwise, when ground-truth labels are available, use their number of
   unique classes;
3. otherwise, use the bounded cell-count heuristic in `clustering.py`.

The second path is intentional. For the bundled Baron dataset it sets the target
to 14 from the `Group` column. scRAW uses this known **count** to guide Leiden or
KMeans pseudo-labeling; it does not substitute ground-truth cell identities for
pseudo-label assignments. This prior is part of the reference scientific
protocol and must not be removed as “label leakage” when reproducing the
reported experiment.

At the warmup boundary and each configured update interval, the trainer encodes
all cells and refreshes pseudo-labels. Leiden scans resolution toward the target
count and falls back to KMeans if Leiden fails.

### 4. Dynamic rare-cell weighting

Two per-cell signals are computed from the current embedding:

- inverse pseudo-cluster frequency, which emphasizes smaller pseudo-clusters;
- latent k-nearest-neighbor distance, which emphasizes lower-density cells.

The configured additive or multiplicative fusion is normalized, clipped, and
smoothed across refreshes with momentum. Weighted reconstruction begins after
warmup. The final pseudo-label and weight arrays are persisted for inspection.

### 5. Optional objectives

The rare-cell triplet objective starts at `triplet.start_epoch`, selects
high-weight anchors, and uses semi-hard negatives from other pseudo-clusters.
Its coefficient is ramped after activation.

When batch correction is enabled and at least two batch values are present, a
classification head receives the latent embedding through gradient reversal.
The adversarial coefficient ramps from `batch_correction.start_epoch`. A
non-zero `mmd_weight` is rejected because an MMD objective is not implemented;
this avoids silently claiming a different scientific method.

The optimized loss is the sum of weighted reconstruction, ramped triplet, and
weighted adversarial classification terms. Adam, cosine annealing, and gradient
clipping follow the resolved configuration.

## Final clustering and noise behavior

Final labels come from HDBSCAN on the learned embedding. The reference settings
are `min_cluster_size=8`, `min_samples=6`, selection method `eom`, and
`hdbscan_reassign_noise=false`.

The current noise policy is part of the reference implementation:

- If HDBSCAN finds more than one usable non-noise cluster and reassignment is
  disabled, every raw `-1` observation is retained and all such observations
  form one additional output cluster. Labels are then remapped to contiguous
  non-negative integers.
- If reassignment is enabled, each raw `-1` observation is assigned to the
  nearest non-noise cluster centroid before contiguous remapping.
- If HDBSCAN fails or finds at most one usable cluster, scRAW falls back to
  Leiden using the effective pseudo K, then KMeans if Leiden also fails.

Consequently, the reference output does not discard HDBSCAN noise observations
from the result or metric sample count. Altering this policy changes the
scientific protocol.

## Metrics and artifacts

When labels are available, scRAW computes NMI, ARI, Hungarian-aligned accuracy,
macro F1, balanced accuracy, rare-class accuracy, balanced rare-class accuracy,
class-wise scores, and class-balanced kNN purity. Silhouette uses predicted
clusters and does not require ground truth.

The pipeline writes:

- the resolved config;
- a strict-JSON summary (non-finite metrics are represented as `null`);
- a scalar metrics CSV;
- embeddings, final labels, pseudo-labels, cell weights, processed cell IDs, and
  processed feature IDs;
- autoencoder weights when enabled;
- training and latent-space figures when enabled.

`results.json` also records `known_label_count`, `effective_pseudo_k`, processed
dimensions, selected label/batch keys, package/runtime versions, resolved device,
and input path/SHA256. This metadata is the first place to inspect when two runs
differ.

## Checkpoint replay boundary

`scraw infer` rebuilds the autoencoder from the saved configuration, loads its
state dictionary, and then reruns preprocessing, encoding, final clustering,
metrics, and inference figures. It is intended to verify or regenerate outputs
for the **same input dataset and configuration**.

The checkpoint is not a frozen preprocessing pipeline and is not a general
transfer-learning artifact. In particular, it does not contain fitted feature
selection/scaling state. Supplying a new dataset can select a different feature
space and is outside reference replay semantics.

## Reproducibility boundary

The authoritative reference contract consists of:

- the repository commit;
- `configs/default_scraw.json` and its SHA256;
- `data/baron_human_pancreas.h5ad` and its SHA256;
- Linux x86_64, Python 3.12.3, CUDA 12.4, and `requirements.txt`;
- the selected seed(s), device, and generated `config_used.json`;
- the intentionally known ground-truth class count and unchanged HDBSCAN noise
  policy described above.

`configs/smoke_cpu.json`, the notebook, portable dependencies, alternate
devices, and checkpoint use on another dataset are outside that boundary.

## Safe extension points

- Add configuration fields in `config.py` with conservative defaults and ensure
  partial JSON merging remains backward compatible.
- Add an objective as a small trainer helper with an explicit activation field,
  saved config metadata, and focused tests.
- Add metrics in `metrics.py`; use strict JSON-safe values at the pipeline
  boundary.
- Add figures in `plots.py`; keep computation independent of plotting.
- Add a CLI override only when it maps directly to a typed configuration field
  and cannot silently change unrelated values.

Scientific changes to pseudo-K resolution, preprocessing, weighting, loss
composition, final HDBSCAN/noise handling, or metric definitions require a new
protocol designation and side-by-side regression results. See
[CONTRIBUTING.md](../CONTRIBUTING.md).
