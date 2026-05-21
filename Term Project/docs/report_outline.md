# Final Report Outline

## 1. Introduction

- Define AG News text classification.
- State the main research question: how do LSTM and Transformer Encoder classifiers differ under controlled training from scratch?
- Briefly motivate the comparison between recurrent modeling and self-attention.

## 2. Dataset

- Dataset source: Hugging Face `ag_news`.
- Access method: `datasets.load_dataset("ag_news")`.
- Train/validation/test split and sample counts.
- Four labels: `World`, `Sports`, `Business`, `Sci/Tech`.
- Label mapping and class distribution.
- Tokenization, vocabulary construction from training data only, padding, truncation, maximum sequence length, and leakage prevention.

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

## 5. Results

- Accuracy and macro F1-score table.
- Training and validation loss curves.
- Confusion matrices.
- Convergence analysis.
- Overfitting or underfitting discussion.
- Class-level performance observations.

## 6. Failure Analysis

- At least five misclassified examples.
- Include examples missed by both models where possible.
- Include examples missed only by one model where possible.
- For each example, report true label, LSTM prediction, Transformer prediction, confidence if available, and likely error cause.
- Compare whether the models fail similarly or differently.

## 7. Conclusion

- Summarize the main empirical findings.
- State whether the ablation supported the hypothesis.
- Identify lessons learned about LSTM and Transformer Encoder behavior.
- Propose one reasonable next experiment.

## 8. AI Usage Appendix

- Attach the one-page appendix after the main report.
- Keep the appendix outside the 4-6 page main report length.
