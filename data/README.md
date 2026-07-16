# Bundled Baron human pancreas dataset

`baron_human_pancreas.h5ad` is the reference input used by the default scRAW
experiment. Keep this exact file when reproducing the reported results.

## Identity

| Field | Value |
| --- | --- |
| File | `data/baron_human_pancreas.h5ad` |
| SHA256 | `aa784472d90c2a6cdb99cdd076525b2dbe83a7254543092eaccca25bc60e359e` |
| Shape | 8,569 cells × 20,125 genes |
| Biological labels | 14 unique values in `obs["Group"]` |
| Batches | 4 human donors in `obs["batch"]` |
| Stored dataset name | `Baron Human Pancreas` |
| Stored source | `GSE84133_RAW (local human donor H5AD files)` |

Verify the artifact on Linux with:

```bash
sha256sum data/baron_human_pancreas.h5ad
```

Expected output:

```text
aa784472d90c2a6cdb99cdd076525b2dbe83a7254543092eaccca25bc60e359e  data/baron_human_pancreas.h5ad
```

Do not run a reference experiment if this value differs.

## Source and scope

The file combines the four human donors from the Baron pancreas study,
associated with NCBI GEO accession
[GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133).
Its embedded `uns` metadata identifies the source as
`GSE84133_RAW (local human donor H5AD files)` and lists donors `human1` through
`human4`.

This repository distributes a combined H5AD artifact, not the original GEO
archive and not a scripted reconstruction from every upstream raw file. The
SHA256 above therefore defines the exact scRAW reference input. Users who
rebuild the dataset from GEO must treat that as a new data preparation unless
the resulting H5AD is byte-identical.

Consult the GEO record and original publication for upstream terms, study
design, and citation requirements. Inclusion in this repository does not alter
the upstream dataset's terms.

## Observation metadata

The file contains these `obs` columns:

| Column | Meaning in this repository |
| --- | --- |
| `barcode` | Original cell barcode |
| `assigned_cluster` | Stored source cell annotation |
| `Group` | Authoritative reference label used by `default_scraw.json` |
| `label` | Duplicate/convenience label field |
| `cell_type` | Duplicate/convenience cell-type field |
| `labels` | Duplicate/convenience label field |
| `batch` | Donor identifier used for adversarial batch correction |

The `Group` distribution is:

| Group | Cells |
| --- | ---: |
| acinar | 958 |
| activated_stellate | 284 |
| alpha | 2,326 |
| beta | 2,525 |
| delta | 601 |
| ductal | 1,077 |
| endothelial | 252 |
| epsilon | 18 |
| gamma | 255 |
| macrophage | 55 |
| mast | 25 |
| quiescent_stellate | 173 |
| schwann | 13 |
| t_cell | 7 |

The donor distribution is:

| Batch | Cells |
| --- | ---: |
| human1 | 1,937 |
| human2 | 1,724 |
| human3 | 3,605 |
| human4 | 1,303 |

## Reference preprocessing

`configs/default_scraw.json` explicitly selects `Group` and `batch`. The
reference preprocessing filters cells/genes, normalizes count-like input to a
total of 20,000, applies `log1p`, selects up to 2,000 highly variable genes,
standardizes each selected feature, and clips at ±10. The processed reference
matrix has 8,569 cells and 2,000 selected features with the current input and
configuration.

Because `pseudo_k` is unset, the 14 unique `Group` values intentionally supply
the target pseudo-cluster count. This known class-count prior is part of the
reference protocol. Per-cell `Group` identities are used for evaluation and
ground-truth plots, not substituted for predicted assignments.

The pipeline writes processed `cell_ids.npy` and `feature_ids.npy` alongside
embeddings and labels. Preserve those arrays when comparing or joining outputs;
filtering and HVG selection mean positional assumptions alone are fragile.

## Using another dataset

For a new H5AD, create a new configuration and explicitly identify its label and
batch columns (or set the keys to JSON `null` to request auto-detection). Also
record the source, dimensions, processing state, and SHA256. A new dataset is
not expected to reproduce the Baron reference metrics.
