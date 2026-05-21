# Initial Project Plan

## Project Question

This project compares an LSTM classifier and a Transformer Encoder classifier on AG News text classification. The main research question is: under a controlled preprocessing and training setup, how do recurrent sequence modeling and self-attention differ in accuracy, macro F1-score, convergence behavior, and failure patterns?

## Dataset And Split

The project will use the Hugging Face `ag_news` dataset through `datasets.load_dataset("ag_news")`. No other AG News source will be mixed into the experiments. The official training split will be divided into training and validation sets using `train_test_split(test_size=0.1, seed=42)`. The official test split will be reserved for final evaluation only.

The report will document the dataset source, sample counts for train/validation/test, label mapping, class distribution, tokenization method, vocabulary construction method, padding and truncation strategy, and maximum sequence length. Labels will be verified as integer class indices `0, 1, 2, 3` for `torch.nn.CrossEntropyLoss`.

## Preprocessing

The same preprocessing pipeline will be used for both models. Text will be tokenized with a simple reproducible tokenizer, and the vocabulary will be built only from the training split to avoid vocabulary leakage. Tokens will be mapped to integer IDs with reserved indices for padding and unknown tokens. Sequences will be padded or truncated to a fixed maximum length of 128 for the main comparison.

## Models

The LSTM classifier will use an embedding layer, a one-layer bidirectional LSTM, dropout, a pooling or final-state strategy, and a final linear classification layer. The Transformer Encoder classifier will use an embedding layer, positional encoding, two Transformer Encoder layers, four attention heads, dropout, masked pooling, and a final linear classification layer.

Both models will be trained from scratch. Pretrained language models and pretrained embeddings will not be used unless explicitly approved by the instructor. Trainable parameter counts and all final hyperparameters will be reported.

## Experiments

The main comparison will train both models using the same dataset split, vocabulary, maximum sequence length, evaluation metrics, optimizer family, and comparable training budget. The initial training setup will use Adam, learning rate `1e-3`, batch size `64`, embedding dimension `128`, dropout `0.2`, and approximately 6 epochs.

The required ablation study will test dataset size at 25%, 50%, and 100% of the training data. The hypothesis is that the LSTM may be more stable with limited data, while the Transformer Encoder may benefit more from the full dataset because self-attention has higher capacity. All other settings will be controlled during this ablation.

If time permits, a second ablation will compare maximum sequence length 128 vs 256 to test whether longer context improves classification performance or mainly increases training cost.

## Evaluation And Analysis

The project will report accuracy, macro F1-score, training and validation loss curves, confusion matrices, model hyperparameters, trainable parameter counts, and ablation results. Interpretation will focus on convergence speed, overfitting behavior, class-level errors, and whether the ablation supports the stated hypothesis.

Failure analysis will include at least five misclassified examples. The examples will include cases where possible that are missed by both models, missed only by the LSTM, and missed only by the Transformer Encoder. Each example will list the input text, true label, both predictions, confidence scores if available, and a possible reason for the error.

## Deliverables

The final submission will include a clean reproducible Jupyter notebook, a 4-6 page final report, a maximum one-page AI Usage Appendix, and an 8-10 minute presentation with Q&A.
