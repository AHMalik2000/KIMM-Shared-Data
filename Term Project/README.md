# LSTM vs Transformer Encoder for AG News Classification

This repository contains the initial scaffold for an introductory deep learning term project. The project compares an LSTM classifier and a Transformer Encoder classifier trained from scratch on the AG News four-class text classification task.

## Assignment Goal

The goal is not only to maximize accuracy. The project should compare recurrent sequence modeling and self-attention-based modeling in terms of classification performance, training behavior, convergence, failure cases, and sensitivity to experimental settings.

## Dataset

- Source: Hugging Face `ag_news`
- Classes: `World`, `Sports`, `Business`, `Sci/Tech`
- Split plan:
  - Training: 90% of official training split
  - Validation: 10% of official training split, created with `seed=42`
  - Test: official test split, used only for final evaluation

The project must use one dataset source only. The default source is Hugging Face; TorchText or raw CSV should only be used if Hugging Face access fails, and sources must not be mixed.

## Planned Models

- LSTM classifier trained from scratch
  - Embedding layer
  - One-layer bidirectional LSTM
  - Dropout
  - Classification layer

- Transformer Encoder classifier trained from scratch
  - Embedding layer
  - Positional encoding
  - Two Transformer Encoder layers
  - Dropout
  - Classification layer

The models do not need identical parameter counts, but their sizes should be reasonably comparable and reported.

## Experiments

1. Main comparison using the same data split, preprocessing, vocabulary, maximum sequence length, metrics, and training budget.
2. Required ablation: dataset size at 25%, 50%, and 100%.
3. Optional extension: sequence length 128 vs 256.

## Required Outputs

- Initial project plan
- Clean, runnable Jupyter notebook
- Final report, 4-6 pages excluding AI Usage Appendix
- AI Usage Appendix, maximum 1 page
- Final presentation, 8-10 minutes plus Q&A

## Project Documents

- `docs/initial_project_plan.md`: one-page course-facing plan
- `docs/report_outline.md`: final report structure
- `docs/ai_usage_appendix_template.md`: required AI use appendix template
- `docs/presentation_outline.md`: presentation plan
- `docs/superpowers/specs/2026-05-21-lstm-transformer-ag-news-design.md`: design/spec
- `docs/superpowers/plans/2026-05-21-lstm-transformer-ag-news.md`: implementation plan
