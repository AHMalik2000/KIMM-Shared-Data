# Final Report Outline

Use this outline after the experiment notebook has been rerun in order. Do not replace placeholders with results until the corresponding notebook tables, figures, and examples have been inspected.

## 1. Introduction

- Define AG News text classification.
- State the main research question: how do LSTM and Transformer Encoder classifiers differ under controlled training from scratch?
- Briefly motivate the comparison between recurrent modeling and self-attention.
- Avoid claiming which model performs better until the verified result table is available.

## 2. Dataset

- Dataset source: Hugging Face `fancyzhx/ag_news`.
- Access method: `datasets.load_dataset("fancyzhx/ag_news")`.
- Train/validation/test split and sample counts.
- Four labels: `World`, `Sports`, `Business`, `Sci/Tech`.
- Label mapping and class distribution.
- Tokenization, vocabulary construction from training data only, padding, truncation, maximum sequence length, and leakage prevention.
- Placeholder to fill from notebook: exact train/validation/test sample counts and class distribution.

## 3. Models

- LSTM classifier architecture:
  - vocabulary size
  - embedding dimension
  - hidden size
  - number of layers
  - bidirectionality
  - dropout
  - pooling or final-state strategy
  - trainable parameter count
- Transformer Encoder classifier architecture:
  - vocabulary size
  - embedding dimension
  - positional encoding
  - number of heads
  - number of encoder layers
  - feedforward dimension
  - dropout
  - pooling or CLS-token strategy
  - trainable parameter count
- Explain how the model sizes are reasonably comparable.
- Placeholder to fill from notebook: final hyperparameter table and trainable parameter counts.

## 4. Experiments

- Main comparison setup.
- Shared preprocessing and controlled variables.
- Optimizer, learning rate, batch size, epochs, seed, device, and loss function.
- Required dataset-size ablation:
  - hypothesis
  - changed variable
  - controlled variables
  - result table
  - interpretation
- Optional sequence-length ablation if completed.
- Placeholder to fill from notebook: actual device, runtime-relevant settings, final epoch count, and any deviations from the initial plan.

## 5. Results

- Accuracy and macro F1-score table.
- Training and validation loss curves.
- Confusion matrices.
- Convergence analysis.
- Overfitting or underfitting discussion.
- Class-level performance observations.
- Placeholder to fill from notebook: final validation/test accuracy and macro F1, loss figures, and confusion matrices.

## 6. Failure Analysis

- At least five misclassified examples.
- Include examples missed by both models where possible.
- Include examples missed only by one model where possible.
- For each example, report true label, LSTM prediction, Transformer prediction, confidence if available, and likely error cause.
- Compare whether the models fail similarly or differently.
- Placeholder to fill from notebook: at least five selected test-set errors, preferably covering both-wrong, LSTM-only-wrong, and Transformer-only-wrong cases where available.

## 7. Conclusion

- Summarize the main empirical findings.
- State whether the ablation supported the hypothesis.
- Identify lessons learned about LSTM and Transformer Encoder behavior.
- Propose one reasonable next experiment.
- Tie every conclusion to a result table, figure, confusion matrix, or failure example from the notebook.

## 8. AI Usage Appendix

- Attach the one-page appendix after the main report.
- Keep the appendix outside the 4-6 page main report length.
