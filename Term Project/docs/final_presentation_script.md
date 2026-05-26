# Final Presentation Slide Plan

## Slide 1 - Research Question

Claim: We compare recurrent sequence modeling and self-attention under one controlled AG News pipeline.

Speaker note: The goal is not just high accuracy. The project asks whether LSTM and Transformer Encoder models behave differently in performance, convergence, data efficiency, and failure cases.

## Slide 2 - Dataset And Split

Claim: AG News is balanced and controlled enough for a fair four-class comparison.

- Source: Hugging Face `fancyzhx/ag_news`
- Train/validation/test: 108,000 / 12,000 / 7,600
- Labels: World, Sports, Business, Sci/Tech

## Slide 3 - Shared Pipeline

Claim: The comparison uses the same inputs for both models.

- Train-only vocabulary, 20,000 tokens
- Regex tokenizer
- Max length 128 for main comparison
- Padding masks used during pooling and attention

## Slide 4 - Model Architectures

Claim: The two models are different architectures but comparable in size.

- LSTM: bidirectional, one layer, masked mean pooling, 2.83M parameters
- Transformer Encoder: two layers, four heads, learned positions, 2.97M parameters

## Slide 5 - Main Test Result

Claim: Both models are close, with a slight Transformer Encoder edge.

- LSTM test accuracy 0.9120, macro F1 0.9118
- Transformer test accuracy 0.9125, macro F1 0.9126

## Slide 6 - Training Behavior

Claim: Both models overfit by epoch 6.

- Training loss keeps falling
- Validation loss rises late in training
- Early stopping or stronger regularization is the next experiment

## Slide 7 - Confusion Matrices

Claim: Business and Sci/Tech are the main source of class-level errors.

- Sports is easiest for both models
- Business/Sci-Tech overlap is frequent because many company stories have both market and technology cues

## Slide 8 - Dataset-Size Ablation

Claim: LSTM is more stable with less data, while Transformer benefits from the full dataset.

- 25%: LSTM 0.8719 vs Transformer 0.8699 validation accuracy
- 50%: LSTM 0.8944 vs Transformer 0.8901
- 100%: Transformer 0.9146 vs LSTM 0.9089

## Slide 9 - Sequence-Length Extension And Failures

Claim: More context did not help in this run, and errors are mostly boundary cases.

- 256 tokens reduced validation metrics for both models
- Failure examples show Olympics/World ambiguity and Business/Sci-Tech overlap

## Slide 10 - Conclusion

Claim: Architecture differences appear most clearly in data efficiency and failure patterns, not in a large final accuracy gap.

- Transformer Encoder slightly wins the final test comparison
- LSTM performs better with limited data
- Future work: early stopping, regularization, and class-level error reduction
