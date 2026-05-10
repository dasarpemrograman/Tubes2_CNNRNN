# Tubes2_CNNRNN

CNN and RNN/LSTM image learning project for Tubes 2. The repository is managed
with `uv`; the authoritative dependency files are `pyproject.toml` and
`uv.lock`.

## Setup

Install `uv`, then run all commands from the repository root.

```powershell
uv sync
```

The project targets Python 3.11 through `.python-version`.

## Repository Layout

```text
src/common/       shared paths, I/O, logging, plotting, seed utilities
src/cnn/          CNN implementation owned by Person A
src/captioning/   RNN/LSTM image captioning implementation owned by Person B
src/scratch/      from-scratch forward propagation and bonus backward work
src/experiments/  experiment orchestration
src/notebooks/    testing notebooks
artifacts/        generated results, metrics, figures, and bonus outputs
scripts/          repo audit and report-support tooling
doc/report.pdf    final Google Docs PDF export only
```

`doc/` must contain exactly one file: `report.pdf`. Do not store report source
files, figures, or drafts in `doc/`.

## Data Placement

Keep datasets out of git. Use `data/` or `datasets/` locally for Intel Image
Classification and Flickr8k files, then write generated indexes and experiment
outputs to `artifacts/`.

## Standard Commands

```powershell
uv sync
uv run python scripts/audit_artifacts.py
uv run python scripts/generate_report_tables.py artifacts/cnn/results.csv artifacts/captioning/results.csv
uv run python scripts/validate_figures.py artifacts/cnn artifacts/captioning artifacts/bonus
uv run pytest
uv run jupyter lab
```

## CNN Commands

Placeholders for Person A to replace with final script paths:

```powershell
uv run python -m experiments.train_cnn
uv run python -m experiments.evaluate_cnn
uv run python -m scratch.cnn_inference
```

## RNN/LSTM Captioning Commands

Placeholders for Person B to replace with final script paths:

```powershell
uv run python -m experiments.train_captioning
uv run python -m experiments.evaluate_captioning
uv run python -m scratch.captioning_inference
```

## Bonus Commands

Placeholders for bonus work:

```powershell
uv run python -m experiments.cnn_feature_maps
uv run python -m experiments.cnn_grad_cam
uv run python -m experiments.cnn_batch_inference
uv run python -m experiments.cnn_gradient_check
uv run python -m experiments.captioning_beam_search
uv run python -m experiments.captioning_batch_inference
uv run python -m experiments.captioning_gradient_check
```

## Report

The report is authored in Google Docs outside this repository and exported to
`doc/report.pdf` for submission. Required Google Docs sections include cover,
problem description, implementation explanation, class/attribute/method
explanation, CNN/RNN/LSTM forward propagation, required results, bonus,
conclusion, task division, references, and AI usage form.

## Task Division

| Person | Ownership |
| --- | --- |
| Person A | CNN track |
| Person B | RNN/LSTM image captioning track |
| Person C | Repo setup, uv tooling, audits, README, Google Docs coordination, release, compliance, submission |

## Compliance Checklist

- No plagiarism.
- Responsible AI use only.
- No cross-group collaboration.
- Group has exactly 3 people.
- Group is not cross-class.
- Smallest NIM member submits the GitHub link to Edunex before 15 May 2026.
