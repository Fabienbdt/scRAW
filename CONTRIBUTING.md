# Contributing to scRAW

Thank you for helping improve scRAW. Contributions should make the repository
easier to reproduce, understand, and maintain without silently changing the
scientific protocol.

## Choose the right environment

Use the portable development environment for code and tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

Use the authoritative pinned environment only when validating reference
results. On Linux x86_64 with Python 3.12.3 and CUDA 12.4:

```bash
python3.12 -m venv .venv-reference
source .venv-reference/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
scraw doctor --reference
```

`requirements.txt` is the exact reference lock.
`requirements-portable.txt` supports ordinary installation and is consumed by
the package metadata. `requirements-cuda124.txt` is a compatibility alias for
the reference lock. Do not start a reference comparison until
`scraw doctor --reference` reports `reference_compatible: true`.

## Before editing

1. Start from a clean branch based on current `main`.
2. Read [README.md](README.md), especially the reference protocol.
3. Read [docs/architecture.md](docs/architecture.md) before changing the data,
   model, trainer, clustering, or output path.
4. Decide whether the change is operational, backward-compatible, or a new
   scientific protocol. State that classification in the change description.

Do not include local datasets, result directories, checkpoints, virtual
environments, caches, or IDE state in a change.

## Run the checks

The normal pre-review suite is:

```bash
python -m compileall -q src tests
python -m pytest
python -m ruff check src tests
python -m build
python -m twine check dist/*
git diff --check
```

The CPU smoke test exercises the complete pipeline but does not validate
scientific results:

```bash
scraw run --config configs/smoke_cpu.json
```

Use a fresh output directory, or explicitly pass `--overwrite` when replacing a
previous smoke result.

For a protocol-sensitive change, also run the pinned Baron reference protocol
on the reference GPU platform. Preserve the commit SHA, dataset/config SHA256,
`pip freeze`, GPU/driver information, full logs, resolved configs, and per-seed
metrics. Compare all ten reference seeds, not one convenient seed.

## Code style

- Write identifiers, docstrings, comments, log messages, exceptions, CLI help,
  and documentation in clear English.
- Prefer small functions with one responsibility and typed configuration
  boundaries over hidden global state.
- Add a focused docstring for public APIs and non-obvious scientific helpers.
- Keep imports lightweight at package import time; load the scientific stack at
  execution boundaries when practical.
- Raise a clear error for invalid explicit input. Do not silently reinterpret a
  misspelled column, device, method, or unimplemented objective.
- Catch only exceptions for which the code provides a documented fallback.
- Use `pathlib.Path` for paths and preserve platform-neutral behavior outside
  the explicitly Linux/CUDA reference protocol.
- Keep JSON outputs strict: represent unavailable/non-finite values as `null`,
  not non-standard `NaN` literals.
- Avoid unrelated formatting or generated-file churn in a focused change.

## Scientific protocol invariants

The following behaviors are intentional:

- When `pseudo_k` is unset and labels are present, scRAW uses the number of
  unique ground-truth classes as the pseudo-cluster target. The known class count
  is part of the reference protocol; it is not an implementation defect.
- With reference `hdbscan_reassign_noise: false`, raw HDBSCAN noise observations
  are grouped into one additional final cluster and retained in evaluation.
  Do not change this behavior as a cleanup or bug fix.
- `preprocessing.input_mode: auto` preserves the historical negative-value
  heuristic used by the reference configuration.
- Checkpoint replay recomputes preprocessing and is intended for the same data
  and configuration, not arbitrary transfer to a new feature space.

A proposal to change any of these, or to change preprocessing, loss terms,
weighting, clustering fallback, seed handling, or metric definitions, must:

1. use a new explicit configuration/protocol designation rather than changing
   reference defaults in place;
2. explain the scientific rationale;
3. include focused unit tests;
4. provide side-by-side ten-seed reference results and full manifests;
5. update README and architecture documentation;
6. state migration and comparability consequences.

An option exposed by configuration must be implemented or rejected clearly.
For example, a non-zero MMD weight must not be accepted unless an MMD loss is
actually part of the optimized objective and covered by tests.

## Tests to add

- Configuration change: default, partial JSON, invalid value, and round-trip
  tests.
- CLI change: parser/help, override behavior, error exit, and strict JSON output
  tests.
- Preprocessing change: raw, preprocessed, sparse/dense, row/feature identity,
  and empty-result cases.
- Training change: deterministic tiny matrices and activation-boundary tests.
- Clustering change: normal clusters, all/partial HDBSCAN noise, fallback, and
  stable contiguous output labels.
- Metric change: perfect, mismatched, rare-class, missing-label, and non-finite
  cases.
- Artifact change: exact filename, shape/dtype, identity mapping, and provenance
  tests.

Keep unit tests small. The bundled Baron file and reference GPU experiment are
integration/reproducibility checks and should not be required for every local
unit-test iteration.

## Documentation

Update the closest user-facing document in the same change. Commands must be
copy-pasteable from the repository root, file paths must match actual outputs,
and examples must say whether they reproduce reference results or only test
functionality.

When adding a dataset, include a `README.md` recording source, accession,
dimensions, important `obs`/`var` fields, processing status, license/usage
constraints, and SHA256. Do not call a transformed dataset “raw” without a
precise explanation.

## Change checklist

- [ ] The change has one clear scope.
- [ ] User-facing text and code comments are English.
- [ ] Unit tests cover success and failure paths.
- [ ] The standard check suite passes.
- [ ] New output or CLI behavior is documented.
- [ ] Reference defaults and hashes are unchanged, or a new protocol is named.
- [ ] Ground-truth class-count and HDBSCAN noise semantics remain explicit.
- [ ] Reproduction evidence accompanies every protocol-sensitive change.
