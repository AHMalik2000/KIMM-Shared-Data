# Outputs

Generated result artifacts from `notebooks/ag_news_lstm_transformer.ipynb`.

## Tables

- `parameter_counts.csv`: trainable parameters for both models.
- `main_results.csv`: validation and final test loss, accuracy, and macro F1.
- `dataset_size_ablation.csv`: required 25%, 50%, and 100% training-fraction ablation.
- `sequence_length_ablation.csv`: optional 128-token vs 256-token sequence-length ablation.
- `selected_misclassifications.csv`: representative test-set failure-analysis examples.
- `artifact_manifest.csv`: generated artifact list.

## Figures

- `main_loss_curves.png`: training and validation loss curves for the main comparison.
- `lstm_confusion_matrix.png`: LSTM final test confusion matrix.
- `transformer_confusion_matrix.png`: Transformer Encoder final test confusion matrix.

## Report Notes

Use these files as the evidence source for the report and presentation. The current run used the official AG News test split only for final evaluation, with validation data created from the official training split.
