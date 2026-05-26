# LSTM vs Transformer Encoder for AG News Classification

## 1. Introduction

This project compares two neural text classifiers trained from scratch on the AG News four-class classification task: a bidirectional LSTM classifier and a Transformer Encoder classifier. The goal is to compare recurrent sequence modeling and self-attention under a controlled setup, not to maximize accuracy with pretrained language models.

Both models use the same dataset source, split procedure, tokenizer, vocabulary, maximum sequence length, optimizer family, training budget, and evaluation metrics. The main research question is how the two architectures differ in final accuracy, macro F1-score, training behavior, data efficiency, and failure patterns when tested on the same news classification task.

## 2. Dataset And Preprocessing

The project uses Hugging Face `fancyzhx/ag_news` as the only dataset source. The official training split was divided into 108,000 training examples and 12,000 validation examples using `train_test_split(test_size=0.1, seed=42)`. The official test split contains 7,600 examples and was used only for final evaluation.

The four labels are `World`, `Sports`, `Business`, and `Sci/Tech`. The test split is balanced, with 1,900 examples per class. The training split is also nearly balanced: 26,991 World, 26,966 Sports, 27,100 Business, and 26,943 Sci/Tech examples.

Texts were tokenized with a reproducible regex tokenizer. The vocabulary was built only from the training split, with reserved indices for padding and unknown tokens. The main comparison used a vocabulary size of 20,000 and maximum sequence length of 128. Both models received the same padded and truncated integer-token inputs, and padding was masked during model pooling and Transformer attention.

## 3. Models

The LSTM classifier uses an embedding layer, a one-layer bidirectional LSTM, masked mean pooling, dropout, and a final linear classifier. The Transformer Encoder classifier uses an embedding layer, learned positional embeddings, two Transformer Encoder layers, four attention heads, masked mean pooling, dropout, and a final linear classifier.

| Model | Trainable parameters |
|---|---:|
| LSTM | 2,825,220 |
| Transformer Encoder | 2,973,444 |

The models are not exactly the same size, but their trainable parameter counts are close enough for a fair introductory comparison. Both were trained from scratch, without pretrained embeddings or pretrained language models.

## 4. Experiments

The main comparison trained both models for 6 epochs using Adam, learning rate `1e-3`, batch size 64, dropout 0.2, seed 42, and CPU execution. Accuracy and macro F1-score were used as the main metrics.

The required ablation changed the amount of training data while keeping the rest of the setup fixed. The tested fractions were 25%, 50%, and 100% of the training split. The hypothesis was that the LSTM might be more stable with less data, while the Transformer Encoder might benefit more from the full dataset because of its higher-capacity self-attention architecture.

An optional sequence-length ablation compared maximum sequence lengths of 128 and 256 tokens.

## 5. Results

| Model | Split | Loss | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| LSTM | Validation | 0.3402 | 0.9113 | 0.9108 |
| Transformer Encoder | Validation | 0.3103 | 0.9118 | 0.9115 |
| LSTM | Test | 0.3459 | 0.9120 | 0.9118 |
| Transformer Encoder | Test | 0.3066 | 0.9125 | 0.9126 |

Both models reached similar final performance. The Transformer Encoder was slightly better on the final test split, with 0.9125 accuracy and 0.9126 macro F1 compared with the LSTM's 0.9120 accuracy and 0.9118 macro F1. Because the gap is small, the result should be interpreted as a slight advantage rather than a decisive difference.

The loss curves show that both models continued to reduce training loss while validation loss stopped improving and then increased. This indicates overfitting by the final epoch. The LSTM had a sharper training-validation loss gap by epoch 6, while the Transformer Encoder retained slightly lower validation and test loss.

The confusion matrices show that Sports was the easiest class for both models. The largest recurring confusion was between Business and Sci/Tech, which is reasonable because many technology-company news items contain both business and technical language.

## 6. Ablation Results

| Model | Training fraction | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|
| LSTM | 25% | 0.8719 | 0.8713 |
| Transformer Encoder | 25% | 0.8699 | 0.8696 |
| LSTM | 50% | 0.8944 | 0.8940 |
| Transformer Encoder | 50% | 0.8901 | 0.8895 |
| LSTM | 100% | 0.9089 | 0.9082 |
| Transformer Encoder | 100% | 0.9146 | 0.9142 |

The dataset-size ablation supports the original hypothesis. The LSTM performed slightly better at 25% and 50% of the training data, while the Transformer Encoder performed better when trained on the full dataset. This suggests that the LSTM was more data-efficient in the lower-data settings, while the Transformer Encoder benefited more from the full training split.

The optional sequence-length ablation did not support using longer sequences. At 256 tokens, validation accuracy and macro F1 decreased for both models compared with the 128-token main setting. Under this training budget, longer context added cost without improving performance.

## 7. Failure Analysis

The selected misclassified examples include cases where both models were wrong, cases where only the LSTM was wrong, and cases where only the Transformer Encoder was wrong. Several errors reflect natural label overlap. For example, Intel and IBM stories can be labeled as Sci/Tech while also containing strong Business signals. Other errors involve surface words that point toward a different section than the dataset label, such as an Olympics item labeled World but predicted Sports by both models.

These examples suggest that the remaining errors are mostly semantic boundary cases rather than simple preprocessing failures. The failure analysis should include the exported examples from `outputs/selected_misclassifications.csv`, along with short explanations for each error.

## 8. Conclusion

Both from-scratch models performed well on AG News under the controlled setup, reaching about 91.2% final test accuracy. The Transformer Encoder achieved the best final test metrics, but only by a small margin. The LSTM was more competitive in lower-data ablation settings, while the Transformer Encoder benefited more from the full training set.

The dataset-size ablation supports the hypothesis that the LSTM is more stable with limited data and the Transformer Encoder scales better with more training examples. The sequence-length ablation showed that simply increasing input length from 128 to 256 did not improve validation performance.

A reasonable next experiment would be to tune regularization or early stopping using the validation split, because both models showed overfitting by epoch 6.
