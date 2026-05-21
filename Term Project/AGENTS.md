# Project-level Codex instructions

This project is for an introductory deep learning term project comparing an LSTM classifier and a Transformer Encoder classifier on AG News text classification.

## Core constraints

- Use Python in Jupyter notebooks with PyTorch.
- Train both comparison models from scratch.
- Do not use pretrained language models such as BERT, DistilBERT, RoBERTa, MiniLM, GPT-style models, or pretrained embeddings unless the instructor explicitly approves it.
- Use exactly one AG News dataset source for all experiments. Default to Hugging Face `datasets.load_dataset("ag_news")`.
- Do not mix Hugging Face, TorchText, and raw CSV variants in the same experiment.
- Build the tokenizer/vocabulary from the training split only.
- Use the official test split only for final evaluation, not model selection, early stopping, or hyperparameter tuning.
- Keep labels as integer class indices `0, 1, 2, 3` for `torch.nn.CrossEntropyLoss`.

## Experiment defaults

- Create validation data from the official training split using `train_test_split(test_size=0.1, seed=42)`.
- Use the same preprocessing, vocabulary, max sequence length, metrics, and training budget for both models.
- Main comparison: LSTM classifier vs Transformer Encoder classifier.
- Required ablation: dataset size at 25%, 50%, and 100%.
- Optional extension: sequence length 128 vs 256.
- Required metrics and artifacts: accuracy, macro F1-score, train/validation loss curves, confusion matrix, parameter counts, hyperparameter table, ablation table, and at least five misclassified examples.

## Documentation expectations

- Track AI assistance in the AI Usage Appendix as work happens.
- Keep claims in the report tied to verified notebook outputs.
- Prefer reproducible, clearly controlled experiments over extra model complexity.
- Leave unrelated parent-repository changes untouched.
