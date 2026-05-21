# LSTM vs Transformer Encoder AG News Design

## Summary

Build a reproducible PyTorch/Jupyter project that compares an LSTM classifier and a Transformer Encoder classifier on AG News text classification. Both models are trained from scratch and evaluated under the same split, preprocessing, vocabulary, maximum sequence length, metrics, and training budget.

The project emphasizes controlled experimental design and interpretation rather than maximum accuracy. Required outputs include the notebook, main comparison, dataset-size ablation, failure analysis, final report, AI Usage Appendix, and final presentation.

## Dataset And Preprocessing

Use Hugging Face `ag_news` as the only dataset source. Create validation data from the official training split with `train_test_split(test_size=0.1, seed=42)`, and reserve the official test split for final evaluation only.

The preprocessing pipeline will:

- inspect and document label values and label mapping
- tokenize text reproducibly
- build vocabulary from training data only
- reserve IDs for padding and unknown tokens
- pad or truncate sequences to a fixed maximum length
- produce identical processed inputs for both comparison models
- verify labels are in the range `0..3`

The main comparison uses maximum sequence length 128. A sequence-length extension may compare 128 vs 256 after the required work is complete.

## Model Design

The LSTM classifier uses an embedding layer, one-layer bidirectional LSTM, dropout, masked pooling or final-state pooling, and a linear classification head.

The Transformer Encoder classifier uses an embedding layer, learned positional encoding, two Transformer Encoder layers, four attention heads, feedforward dimension 512, dropout, masked mean pooling, and a linear classification head.

Both models use `torch.nn.CrossEntropyLoss` and are trained from scratch. Pretrained language models and pretrained embeddings are out of scope unless the instructor explicitly approves them.

## Experiments And Evaluation

The main comparison trains both models with Adam, learning rate `1e-3`, batch size `64`, embedding dimension `128`, dropout `0.2`, and approximately 6 epochs. Final settings must be documented if changed.

The required ablation tests dataset size at 25%, 50%, and 100% of the training data. The hypothesis is that the LSTM may be more stable under limited data, while the Transformer Encoder may benefit more from the full dataset because of greater capacity. During the ablation, all other preprocessing and training settings remain controlled.

Evaluation artifacts:

- model hyperparameter table
- trainable parameter count table
- accuracy and macro F1-score table
- training and validation loss curves
- confusion matrices
- ablation result table
- at least five misclassified examples with true label, both model predictions, confidence if available, and possible error cause

## Deliverables

The repository will contain:

- a clean, runnable notebook at `notebooks/ag_news_lstm_transformer.ipynb`
- a 4-6 page final report based on `docs/report_outline.md`
- a maximum one-page AI Usage Appendix based on `docs/ai_usage_appendix_template.md`
- an 8-10 minute presentation based on `docs/presentation_outline.md`

## Risks And Controls

The main risks are test-set leakage, vocabulary leakage, mismatched preprocessing, padding influencing pooling or attention, uncontrolled hyperparameter changes, and unsupported claims in the final report. Controls are encoded in `AGENTS.md`, the notebook checklist, and the implementation plan.
