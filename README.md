# AGCD: Anonymous Reproducibility Release

This repository contains the code and compact diagnostic artifacts for the
anonymous submission **AGCD: Auditing Failure-Driven Policy Improvement in
World--Action Models**.

AGCD tests whether an outcome predictor uses an action--geometry interaction or
the easier shortcut of action amplitude. The paired protocol keeps the action
vector element-identical while varying task geometry. The repository includes
the synthetic factor generator, baseline predictors, calibration utilities,
dynamic-contact diagnostics, public-data audit utilities, and unit tests.

## Reproduce the core diagnostics

From the repository root:

```bash
python -m pytest -q
python scripts/mujoco_factor_task.py --n 900 --multitask \
  --visual-mode color --output data/diagnostics/mujoco_multitask_900
python scripts/audit_multitask_statistics.py --n 900 --seeds 5 \
  --output data/diagnostics/mujoco_multitask_900/statistics.json
```

The generated records use layout-disjoint train/test splits. Do not replace
these with random row splits: the three actions from one layout are deliberately
paired.

## Repository map

* `models/`, `baselines/`, `policies/`: predictor and decision modules.
* `scripts/`: data generation, audits, calibration, and aggregation entrypoints.
* `configs/`: tensor contracts and experiment defaults.
* `data/diagnostics/`: small JSON summaries used in the paper.
* `tests/`: tensor-contract and end-to-end smoke tests.
* `paper/`: anonymized paper source and the IEEE conference class.

Large raw videos, simulator installations, model checkpoints, and private
machine paths are intentionally excluded. Public-data utilities accept a local
export and never silently crawl an unbounded archive. Public trajectory labels
are explicitly called `progress_proxy` when no reliable terminal reward exists;
they are not presented as verified success/failure labels.

The real-tabletop $3\times3\times15$ crossing is recorded in
`data/diagnostics/real_hardware_crossing_135.json`, including the complete cell
counts, matched/off-diagonal rates, and statistical test.

## Optional dependencies

The minimal tests use `requirements-minimal.txt`. Public Parquet/RLDS auditing
may additionally use `requirements-public.txt`. Simulator-specific packages
are environment-dependent and are not vendored here.

## Anonymous review policy

No author names, affiliations, email addresses, server hostnames, user names,
absolute paths, private checkpoints, or identifying metadata are required by
this release. The repository is intended to be uploaded to an anonymous
hosting account for review; author information can be restored after the
review period.

## License and data

The research code is released for review. Third-party datasets and simulator
assets retain their original licenses; this repository contains only compact
derived summaries and does not redistribute those datasets.
