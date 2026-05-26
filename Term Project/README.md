# LSTM vs Transformer Encoder for AG News Classification

This repository contains the working scaffold for an introductory deep learning term project. The project compares an LSTM classifier and a Transformer Encoder classifier trained from scratch on the AG News four-class text classification task.

Current status: the experiment notebook has been run and exported result artifacts are available in `outputs/`. The next project step is to convert the verified notebook tables, figures, and selected misclassifications into the final report and presentation.

## Assignment Goal

The goal is not only to maximize accuracy. The project should compare recurrent sequence modeling and self-attention-based modeling in terms of classification performance, training behavior, convergence, failure cases, and sensitivity to experimental settings.

## Dataset

- Source: Hugging Face `fancyzhx/ag_news`
- Access method: `datasets.load_dataset("fancyzhx/ag_news")`
- Classes: `World`, `Sports`, `Business`, `Sci/Tech`
- Split plan:
  - Training: 90% of official training split
  - Validation: 10% of official training split, created with `seed=42`
  - Test: official test split, used only for final evaluation

The project must use one dataset source only. This project uses Hugging Face `fancyzhx/ag_news`; TorchText or raw CSV should not be mixed into the same experiment.

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

## Required Outputs And Status

- Initial project plan: drafted in `docs/initial_project_plan.md`
- Clean, runnable Jupyter notebook: implemented at `notebooks/ag_news_lstm_transformer.ipynb`
- Generated tables and figures: exported to `outputs/`
- Final report, 4-6 pages excluding AI Usage Appendix: outline updated in `docs/report_outline.md`
- AI Usage Appendix, maximum 1 page: working draft in `docs/ai_usage_appendix_template.md`
- Final presentation, 8-10 minutes plus Q&A: outline updated in `docs/presentation_outline.md`

Final claims, tables, figures, and conclusions should remain tied to the exported notebook outputs.

## Project Documents

- `docs/initial_project_plan.md`: one-page course-facing plan
- `docs/report_outline.md`: final report structure with current result notes
- `docs/ai_usage_appendix_template.md`: required AI use appendix template with current planning/review entries
- `docs/presentation_outline.md`: presentation plan
- `docs/superpowers/specs/2026-05-21-lstm-transformer-ag-news-design.md`: design/spec
- `docs/superpowers/plans/2026-05-21-lstm-transformer-ag-news.md`: implementation plan
