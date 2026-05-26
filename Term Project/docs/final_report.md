# LSTM vs Transformer Encoder for AG News Classification

## 1. Introduction

This project compares two neural text classifiers trained from scratch on the AG News four-class classification task: a bidirectional LSTM classifier and a Transformer Encoder classifier. The goal is not simply to maximize accuracy. The project evaluates how recurrent sequence modeling and self-attention differ in classification performance, training behavior, convergence patterns, data efficiency, and failure cases.

Both models use the same dataset source, split procedure, tokenizer, vocabulary, maximum sequence length, optimizer family, training budget, and evaluation metrics. This controlled setup makes the comparison more informative than a raw leaderboard-style accuracy result. The main research question is: under a shared preprocessing and training pipeline, how do the LSTM and Transformer Encoder differ on AG News classification?

## 2. Dataset And Preprocessing

The project uses Hugging Face `fancyzhx/ag_news` as the only dataset source. The official training split was divided into 108,000 training examples and 12,000 validation examples using `train_test_split(test_size=0.1, seed=42)`. The official test split contains 7,600 examples and was reserved for final evaluation only.

The four labels are `World`, `Sports`, `Business`, and `Sci/Tech`. The official test split is balanced, with 1,900 examples per class. The training split is also nearly balanced: 26,991 World, 26,966 Sports, 27,100 Business, and 26,943 Sci/Tech examples. Labels were verified as integer class indices `0, 1, 2, 3`, which matches the requirement for `torch.nn.CrossEntropyLoss`.

Texts were tokenized with a reproducible regular-expression tokenizer. The vocabulary was built only from the training split to avoid vocabulary leakage. The vocabulary size was capped at 20,000 tokens, with reserved indices for padding and unknown tokens. The main comparison used a maximum sequence length of 128 tokens. Both models received the same padded and truncated token IDs, and padding masks were used during LSTM pooling and Transformer attention.

## 3. Models

The LSTM classifier uses an embedding layer, a one-layer bidirectional LSTM, masked mean pooling, dropout, and a final linear classifier. The Transformer Encoder classifier uses an embedding layer, learned positional embeddings, two Transformer Encoder layers, four attention heads, feedforward dimension 512, masked mean pooling, dropout, and a final linear classifier. Both models were trained from scratch, without pretrained embeddings or pretrained language models.

| Model | Trainable parameters |
| --- | --- |
| LSTM | 2,825,220 |
| Transformer Encoder | 2,973,444 |

| Setting | LSTM | Transformer Encoder |
| --- | --- | --- |
| Vocabulary size | 20,000 | 20,000 |
| Embedding dimension | 128 | 128 |
| Encoder depth | 1 bidirectional LSTM layer | 2 Transformer Encoder layers |
| Hidden / feedforward size | Hidden size 128 per direction | Feedforward dimension 512 |
| Attention heads | Not applicable | 4 |
| Pooling strategy | Masked mean pooling | Masked mean pooling |
| Dropout | 0.2 | 0.2 |
| Optimizer / learning rate | Adam / 1e-3 | Adam / 1e-3 |
| Epochs / batch size | 6 / 64 | 6 / 64 |

The parameter counts are reasonably comparable for an introductory architecture comparison. The Transformer Encoder has about 5.2% more trainable parameters than the LSTM, so the report treats its small final metric advantage cautiously.

## 4. Experiments

The main comparison trained both models for 6 epochs using Adam, learning rate `1e-3`, batch size 64, dropout 0.2, seed 42, and CPU execution. The evaluation metrics were accuracy and macro F1-score. Macro F1 was included because AG News has multiple classes and the assignment requires class-sensitive evaluation, even though the dataset is nearly balanced.

The required ablation changed one major factor: the fraction of training data. The tested fractions were 25%, 50%, and 100% of the training split. The hypothesis was that the LSTM would be more stable with less data, while the Transformer Encoder would benefit more from the full dataset because self-attention has greater capacity.

An optional extension compared maximum sequence lengths of 128 and 256 tokens. This tested whether longer context improved validation performance enough to justify the additional training cost.

## 5. Results

| Model | Split | Loss | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| LSTM | valid | 0.3402 | 0.9113 | 0.9108 |
| Transformer Encoder | valid | 0.3103 | 0.9118 | 0.9115 |
| LSTM | test | 0.3459 | 0.9120 | 0.9118 |
| Transformer Encoder | test | 0.3066 | 0.9125 | 0.9126 |

Both models reached similar final performance. On the official test split, the LSTM achieved 0.9120 accuracy and 0.9118 macro F1. The Transformer Encoder achieved 0.9125 accuracy and 0.9126 macro F1. The Transformer Encoder therefore performed slightly better on the final test metrics, but the difference is small and should not be overinterpreted as a decisive architectural win.

The loss curves show that both models continued to reduce training loss while validation loss stopped improving and then increased by the final epoch. This indicates overfitting by epoch 6. The LSTM had a sharper training-validation loss gap by the final epoch, while the Transformer Encoder retained slightly lower validation and test loss. A reasonable next experiment would tune early stopping or regularization using the validation set.

The confusion matrices show that Sports was the easiest class for both models. The largest recurring confusion was between Business and Sci/Tech. For the LSTM, 177 Business items were predicted as Sci/Tech and 121 Sci/Tech items were predicted as Business. For the Transformer Encoder, 200 Business items were predicted as Sci/Tech and 93 Sci/Tech items were predicted as Business. This pattern is expected because technology-company stories often contain both market and technical language.

Figures used for the final report:

![Training and validation loss curves](../outputs/main_loss_curves.png)

![LSTM final test confusion matrix](../outputs/lstm_confusion_matrix.png)

![Transformer Encoder final test confusion matrix](../outputs/transformer_confusion_matrix.png)

## 6. Ablation Study

Hypothesis: the LSTM will be more stable with limited training data, while the Transformer Encoder will benefit more from the full dataset. The changed variable was training-set fraction. The controlled variables were dataset source, train/validation split, tokenizer, vocabulary, maximum sequence length, optimizer, learning rate, batch size, epoch count, metrics, and validation evaluation.

| Model | Training fraction | Validation accuracy | Validation macro F1 |
| --- | --- | --- | --- |
| LSTM | 25% | 0.8719 | 0.8713 |
| Transformer Encoder | 25% | 0.8699 | 0.8696 |
| LSTM | 50% | 0.8944 | 0.8940 |
| Transformer Encoder | 50% | 0.8901 | 0.8895 |
| LSTM | 100% | 0.9089 | 0.9082 |
| Transformer Encoder | 100% | 0.9146 | 0.9142 |

The ablation supports the hypothesis. At 25% training data, the LSTM slightly outperformed the Transformer Encoder. At 50%, the LSTM again had higher validation accuracy and macro F1. At 100%, the Transformer Encoder moved ahead. This suggests that the LSTM was more data-efficient in lower-data settings, while the Transformer Encoder benefited more from access to the full training split.

The optional sequence-length ablation is shown below.

| Model | Max length | Validation loss | Validation accuracy | Validation macro F1 |
| --- | --- | --- | --- | --- |
| LSTM | 128 | 0.3402 | 0.9113 | 0.9108 |
| Transformer Encoder | 128 | 0.3103 | 0.9118 | 0.9115 |
| LSTM | 256 | 0.3535 | 0.9065 | 0.9058 |
| Transformer Encoder | 256 | 0.3304 | 0.9062 | 0.9059 |

Increasing maximum sequence length from 128 to 256 did not improve validation performance. Both models performed worse at 256 tokens. Under the current training budget and architecture sizes, longer context added cost without improving classification quality.

## 7. Failure Analysis

The table below gives six representative misclassified test examples. It includes cases missed by both models, cases missed only by the LSTM, and cases missed only by the Transformer Encoder.

| # | Input excerpt | True label | LSTM prediction | Transformer prediction | LSTM conf. | Transformer conf. | Likely reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Live: Olympics day four Richard Faulds and Stephen Parry are going for gold for Great Britain... | World | Sports | Sports | 0.9369 | 0.9983 | Sports terms dominate the short text even though the dataset section is World. |
| 2 | Intel to delay product aimed for high-definition TVs SAN FRANCISCO -- In the latest of a seri... | Business | Sci/Tech | Sci/Tech | 0.9941 | 0.9977 | A technology company story overlaps Business and Sci/Tech vocabulary. |
| 3 | Prediction Unit Helps Forecast Wildfires (AP) AP - It's barely dawn when Mike Fitzpatrick sta... | Sci/Tech | World | Sci/Tech | 0.4821 | 0.9785 | Weather prediction language mixes science terms with world-event framing. |
| 4 | Some People Not Eligible to Get in on Google IPO Google has billed its IPO as a way for every... | Sci/Tech | Business | Sci/Tech | 0.8170 | 0.8252 | Finance and technology cues conflict; the IPO framing points toward Business. |
| 5 | Teenage T. rex's monster growth Tyrannosaurus rex achieved its massive size due to an enormou... | Sci/Tech | Sci/Tech | Business | 0.4429 | 0.5615 | Very short science item gives limited context; the Transformer overweights business-like wording. |
| 6 | IBM to hire even more new workers By the end of the year, the computing giant plans to have i... | Sci/Tech | Sci/Tech | Business | 0.8355 | 0.6214 | Company hiring story overlaps labor/business cues and computing-company Sci/Tech label. |

The selected errors show that the two models often fail on semantic boundary cases. Some examples contain surface words strongly associated with a different news section, such as an Olympics item labeled World but predicted as Sports by both models. Other examples involve technology companies or IPOs, where Business and Sci/Tech labels overlap. The models therefore fail similarly when the text has ambiguous section cues, but they also show different single-model errors: the LSTM misread some science/world-style items, while the Transformer misread some short Sci/Tech items as Business.

## 8. Conclusion

Both from-scratch models performed well on AG News under the controlled setup, reaching about 91.2% final test accuracy. The Transformer Encoder achieved the best final test metrics, but only by a small margin. The LSTM was stronger in lower-data ablation settings, while the Transformer Encoder benefited more from the full training set.

The required dataset-size ablation supports the original data-efficiency hypothesis. The optional sequence-length ablation showed that simply increasing maximum sequence length from 128 to 256 did not improve validation metrics. The most reasonable next experiment would be to tune early stopping or regularization, because both models show overfitting by the final epoch.
