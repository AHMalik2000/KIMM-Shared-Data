# Final Report Outline

Use this outline to write the final 4-6 page report. The placeholders have been replaced with the current notebook outputs exported in `outputs/`.

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
- Current split sizes from notebook: train 108,000; validation 12,000; test 7,600.
- Test split class distribution: 1,900 examples per class.
- Training split is nearly balanced: World 26,991; Sports 26,966; Business 27,100; Sci/Tech 26,943.

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
- Parameter counts from `outputs/parameter_counts.csv`: LSTM 2,825,220 trainable parameters; Transformer Encoder 2,973,444 trainable parameters.
- Both models used the same vocabulary size of 20,000, maximum sequence length of 128 for the main comparison, batch size 64, Adam optimizer, learning rate 1e-3, dropout 0.2, and 6 training epochs.

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
- Notebook device: CPU.
- Final epoch count: 6.
- Required dataset-size ablation completed at 25%, 50%, and 100%.
- Optional sequence-length ablation completed at 128 and 256 tokens.

## 5. Results

- Accuracy and macro F1-score table.
- Training and validation loss curves.
- Confusion matrices.
- Convergence analysis.
- Overfitting or underfitting discussion.
- Class-level performance observations.
- Main validation metrics: LSTM accuracy 0.9113 and macro F1 0.9108; Transformer Encoder accuracy 0.9118 and macro F1 0.9115.
- Final test metrics: LSTM accuracy 0.9120 and macro F1 0.9118; Transformer Encoder accuracy 0.9125 and macro F1 0.9126.
- Main result: the Transformer Encoder was slightly better on final test accuracy and macro F1, but the margin was small.
- Training behavior: both models continued reducing training loss while validation loss stopped improving and then increased, which indicates overfitting by the final epoch. The LSTM showed a sharper train/validation loss gap by epoch 6.
- Confusion-matrix observation: both models mostly classified Sports correctly. The largest visible confusion for both models was between Business and Sci/Tech.
- Dataset-size ablation: LSTM led at 25% and 50%; Transformer Encoder led at 100%. This supports the hypothesis that the LSTM was more stable with less data while the Transformer benefited more from the full training set.
- Sequence-length ablation: 256 tokens reduced validation accuracy and macro F1 for both models relative to 128 tokens, so longer context did not help in this setup.

## 6. Failure Analysis

- At least five misclassified examples.
- Include examples missed by both models where possible.
- Include examples missed only by one model where possible.
- For each example, report true label, LSTM prediction, Transformer prediction, confidence if available, and likely error cause.
- Compare whether the models fail similarly or differently.
- Current selected examples in `outputs/selected_misclassifications.csv` include six cases: two both-wrong examples, two LSTM-only wrong examples, and two Transformer-only wrong examples.
- Likely error themes to write up: sports event wording inside a World item, technology-company stories overlapping Business and Sci/Tech, wildfire prediction mixing science and world-news language, and short science/technology items with business-like organization names.

## 7. Conclusion

- Summarize that both from-scratch models reached similar performance near 91.2% test accuracy.
- State that the Transformer Encoder was slightly better in the main final test result, while the LSTM was stronger in lower-data ablation settings.
- State that the dataset-size ablation supported the original data-efficiency hypothesis.
- Note that longer sequence length did not improve validation metrics under the current training budget.
- Propose a reasonable next experiment: tune early stopping or regularization using validation loss, because both models show overfitting by epoch 6.

## 8. AI Usage Appendix

- Attach the one-page appendix after the main report.
- Keep the appendix outside the 4-6 page main report length.
