# One-Page Project Plan And Result Summary

## Project Question

This project compares a from-scratch LSTM classifier and a from-scratch Transformer Encoder classifier on AG News text classification. The core question is how recurrent sequence modeling and self-attention differ in classification performance, convergence behavior, data efficiency, and failure patterns under a controlled PyTorch/Jupyter workflow.

## Dataset And Preprocessing

- Dataset source: Hugging Face `fancyzhx/ag_news`, loaded with `datasets.load_dataset("fancyzhx/ag_news")`.
- Split: 108,000 train, 12,000 validation from the official training split with `seed=42`, and 7,600 official test examples for final evaluation only.
- Labels: `World`, `Sports`, `Business`, `Sci/Tech`, encoded as integer classes `0, 1, 2, 3`.
- Pipeline: regex tokenization, vocabulary built from training data only, 20,000-token vocabulary, padding/truncation to 128 tokens for the main comparison, and padding masks for pooling/attention.

## Models And Training

- LSTM: embedding layer, one-layer bidirectional LSTM, masked mean pooling, dropout, and linear classifier; 2,825,220 trainable parameters.
- Transformer Encoder: embedding layer, learned positional embeddings, two encoder layers, four attention heads, feedforward dimension 512, masked mean pooling, dropout, and linear classifier; 2,973,444 trainable parameters.
- Shared training setup: Adam, learning rate `1e-3`, batch size 64, dropout 0.2, 6 epochs, seed 42, CPU runtime.

## Experiments And Final Findings

- Main test result: LSTM accuracy 0.9120 and macro F1 0.9118; Transformer Encoder accuracy 0.9125 and macro F1 0.9126. The Transformer Encoder is slightly ahead, but the gap is small.
- Required ablation: dataset size at 25%, 50%, and 100%. LSTM led at 25% and 50%, while Transformer Encoder led at 100%, supporting the hypothesis that LSTM is more stable with limited data and Transformer benefits more from full data.
- Optional extension: sequence length 128 vs 256. Longer 256-token inputs reduced validation metrics for both models, so 128 remains the better setting in this run.
- Failure analysis: six selected test errors cover both-wrong, LSTM-only wrong, and Transformer-only wrong cases. Most errors reflect Business/Sci-Tech overlap, news-section ambiguity, or short texts with misleading surface cues.
