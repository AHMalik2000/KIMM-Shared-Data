# AG News LSTM vs Transformer Encoder Project Manual

## 1. Purpose Of This Manual

This manual is a technical guide to the project in this repository. It explains the project background, the dataset, the two neural network models, the notebook workflow, the output artifacts, and the results that have been achieved so far.

The project compares two from-scratch PyTorch text classifiers on AG News:

- A bidirectional LSTM classifier.
- A Transformer Encoder classifier.

The goal is not just to choose the model with the highest final accuracy. The stronger goal is to understand how recurrent sequence modeling and self-attention behave under the same dataset source, preprocessing pipeline, validation split, training budget, metrics, and report constraints.

The current repository state shows that the experiment notebook has been run, output tables and figures have been exported to `outputs/`, and final submission-facing deliverables have been generated in `deliverables/`.

## 2. Project Scope And Constraints

The project is an introductory deep learning term project using Python, Jupyter notebooks, and PyTorch. The main comparison is LSTM vs Transformer Encoder on AG News text classification.

The important constraints are:

- Both models are trained from scratch.
- No pretrained language models, pretrained embeddings, BERT-style models, GPT-style models, or other pretrained text encoders are used.
- One dataset source is used for all experiments: Hugging Face `fancyzhx/ag_news`, loaded with `datasets.load_dataset("fancyzhx/ag_news")` [1, 2].
- The official AG News test split is reserved for final evaluation only.
- A validation set is created from the official training split with `train_test_split(test_size=0.1, seed=42)`, following the same idea as a reproducible random train/test split [9].
- The vocabulary is built only from the training split.
- Labels remain integer class indices `0`, `1`, `2`, and `3`, which is the format expected by `torch.nn.CrossEntropyLoss` for class-index targets.
- Both models use the same preprocessing, vocabulary, maximum sequence length, metrics, and main training budget.
- The required ablation is dataset size at 25%, 50%, and 100% of the training data.
- The optional extension completed here is maximum sequence length 128 vs 256.

These constraints are important because a model comparison is only meaningful if the data pipeline and evaluation rules are controlled. If the models used different vocabularies, different tokenization, different data sources, or test-set feedback during development, then final accuracy would be hard to attribute to the model architecture itself.

## 3. Dataset Background

AG News is a news topic classification dataset. Each example is a short text item, and the label is one of four classes:

| Label index | Label name |
| --- | --- |
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

The Hugging Face dataset card describes AG News as a text classification dataset derived from a large collection of news articles [1]. The AG News topic classification benchmark used here is associated with Zhang, Zhao, and LeCun's 2015 text classification work [3].

The implemented notebook uses these splits:

| Split | Source | Samples | Use |
| --- | --- | ---: | --- |
| Train | 90% of official training split | 108,000 | Model fitting and train-only vocabulary construction |
| Validation | 10% of official training split, seed 42 | 12,000 | Model selection, ablation comparison, and interpretation |
| Test | Official test split | 7,600 | Final evaluation only |

The notebook verifies that labels are exactly `{0, 1, 2, 3}`. The final report also records that the official test split is balanced with 1,900 examples per class. The training split is nearly balanced:

| Class | Training count |
| --- | ---: |
| World | 26,991 |
| Sports | 26,966 |
| Business | 27,100 |
| Sci/Tech | 26,943 |

Because the dataset is nearly balanced, accuracy is informative. Macro F1 is still reported because it gives each class equal weight and is more sensitive to class-specific weaknesses.

## 4. Model Background

### 4.1 Text Classification Pipeline

Both models receive text after the same preprocessing pipeline:

1. Raw text is lowercased and tokenized with a regular expression tokenizer.
2. A vocabulary is built from training text only.
3. The vocabulary is capped at 20,000 entries.
4. Two reserved tokens are used:
   - `<pad>` at index 0.
   - `<unk>` at index 1.
5. Tokens are converted to integer IDs.
6. Sequences are truncated or padded to a fixed maximum length.
7. An attention mask marks real tokens as `1` and padding positions as `0`.

The main comparison uses `MAX_LEN = 128`. The optional sequence-length extension also trains at `MAX_LEN = 256`.

The attention mask has two jobs:

- In the LSTM model, it prevents padded positions from contributing to masked mean pooling.
- In the Transformer Encoder model, it prevents padded positions from being treated as meaningful tokens during attention and pooling.

### 4.2 Embeddings

Both models start with an `nn.Embedding` layer. This layer maps each token ID to a learned dense vector. In this project, embeddings are learned from scratch during AG News training. They are not initialized from GloVe, word2vec, FastText, BERT, or any pretrained source.

The shared embedding dimension is 128. With a vocabulary near 20,000 tokens, most model parameters are in the embedding table. This is normal for small text classifiers.

### 4.3 LSTM Classifier

An LSTM is a recurrent neural network architecture designed to process sequences step by step while maintaining a hidden state and a cell state. The original LSTM design was introduced by Hochreiter and Schmidhuber to address long-term dependency problems in standard recurrent neural networks [4]. PyTorch's `nn.LSTM` implements a multi-layer LSTM recurrent module with the standard input, forget, candidate, and output gate structure [6].

In practical terms, an LSTM uses gates to decide:

- what new information should enter memory,
- what old information should be forgotten,
- what candidate information should be written,
- what information should be exposed as the hidden state.

The project uses a bidirectional LSTM. That means the sequence is processed in both directions:

- forward, from the first token to the last token;
- backward, from the last token to the first token.

The two directions are concatenated, so a hidden size of 128 per direction produces 256 features per token position.

The implemented LSTM classifier contains:

| Component | Setting |
| --- | --- |
| Embedding dimension | 128 |
| LSTM layers | 1 |
| Direction | Bidirectional |
| Hidden size | 128 per direction |
| Pooling | Masked mean pooling |
| Dropout | 0.2 before classifier |
| Classifier | Linear layer to 4 classes |
| Trainable parameters | 2,825,220 |

The forward pass is:

1. Convert `input_ids` to embeddings.
2. Run the embeddings through the bidirectional LSTM.
3. Multiply token outputs by the mask so padding contributes zero.
4. Average over real token positions.
5. Apply dropout.
6. Apply a linear classifier to produce four logits.

The model outputs logits, not probabilities. Probabilities are computed only for prediction confidence and analysis using `torch.softmax(logits, dim=1)`.

### 4.4 Transformer Encoder Classifier

The Transformer architecture was introduced by Vaswani et al. in "Attention Is All You Need" [5]. Its key idea is self-attention: each token representation can directly attend to other token representations in the same sequence. Unlike a recurrent model, a Transformer layer does not need to process tokens one at a time in sequence order.

The project uses only the encoder side of the Transformer family, because the task is classification rather than text generation. A Transformer Encoder layer contains self-attention and a feedforward network, matching the role of PyTorch's `nn.TransformerEncoderLayer` reference implementation [7]. In this project, learned positional embeddings are added to token embeddings so the model has information about token order.

The implemented Transformer Encoder classifier contains:

| Component | Setting |
| --- | --- |
| Embedding dimension | 128 |
| Positional encoding | Learned positional embeddings |
| Encoder layers | 2 |
| Attention heads | 4 |
| Feedforward dimension | 512 |
| Pooling | Masked mean pooling |
| Dropout | 0.2 |
| Classifier | Linear layer to 4 classes |
| Trainable parameters | 2,973,444 |

The forward pass is:

1. Convert `input_ids` to token embeddings.
2. Add learned positional embeddings.
3. Build a padding mask from the attention mask.
4. Run the sequence through two Transformer Encoder layers.
5. Apply masked mean pooling over real token positions.
6. Apply dropout.
7. Apply a linear classifier to produce four logits.

The Transformer has about 5.2% more trainable parameters than the LSTM. The difference is small enough for this course comparison, but it is still reported so the final result is not presented as a perfectly equal-capacity comparison.

## 5. Loss, Metrics, And Evaluation

### 5.1 Cross-Entropy Loss

The notebook uses `torch.nn.CrossEntropyLoss`. In PyTorch, this loss expects unnormalized logits as model input. For the class-index case used here, the target should contain class indices in the range `[0, C)`, where `C` is the number of classes [8]. Since AG News has four classes, valid target labels are `0`, `1`, `2`, and `3`.

This is why the notebook explicitly checks the label set before training.

### 5.2 Accuracy

Accuracy is the fraction of predictions that match the true label:

```text
accuracy = correct_predictions / total_examples
```

Because AG News is balanced in this run, accuracy is a useful headline metric.

### 5.3 Macro F1

F1 combines precision and recall for a class. Macro F1 computes F1 separately for each class and then averages the class scores without weighting by class frequency [10].

This matters because a model could have strong overall accuracy but still perform worse on one class. Macro F1 makes that weakness more visible.

### 5.4 Confusion Matrix

A confusion matrix shows which true classes are predicted as which output classes. Rows represent true labels, and columns represent predictions. A strong classifier has large diagonal entries and small off-diagonal entries.

In this project, the confusion matrices are used to identify class-level error patterns. The main observed pattern is Business vs Sci/Tech confusion.

## 6. Repository Structure

The important project files are:

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview, current status, dataset, model, experiment, and deliverable summary |
| `AGENTS.md` | Project-specific instructions and constraints for future agent work |
| `requirements.txt` | Python dependencies |
| `References/2026_term_project.pdf` | Assignment source document |
| `notebooks/ag_news_lstm_transformer.ipynb` | Main experiment notebook |
| `outputs/` | Generated CSV tables and PNG figures from the notebook |
| `docs/initial_project_plan.md` | Initial course-facing project plan |
| `docs/one_page_project_summary.md` | Final one-page project summary with results |
| `docs/final_report.md` | Final report source in Markdown |
| `docs/ai_usage_appendix_final.md` | Final AI Usage Appendix source |
| `docs/final_presentation_script.md` | Slide plan and presenter script |
| `deliverables/` | Final DOCX and PPTX submission artifacts |
| `tools/build_final_deliverables.py` | Script used to generate final deliverable files |
| `docs/project_manual.md` | This manual |

The repository is inside a larger parent checkout. Future commits should stage only project-relevant files from `Term Project/` unless the user explicitly asks to modify parent-repository files.

## 7. Notebook Workflow

The main notebook is organized as a linear experiment pipeline.

### 7.1 Setup And Reproducibility

The notebook imports PyTorch, Hugging Face Datasets, pandas, NumPy, matplotlib, seaborn, and scikit-learn metrics. It sets:

- `SEED = 42`
- deterministic CuDNN flags where relevant
- `device = "cuda"` if available, otherwise CPU

The current run used CPU.

### 7.2 Load Dataset

The notebook loads:

```python
raw_dataset = load_dataset("fancyzhx/ag_news")
```

It then splits `raw_dataset["train"]` into train and validation:

```python
split_dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=SEED)
```

The official test split is kept separate as `test_data`.

### 7.3 Inspect Labels And Splits

The notebook creates tables for split sizes and class distributions. It asserts that all labels across train, validation, and test are exactly `{0, 1, 2, 3}`.

This step is a guard against label-format mistakes before `CrossEntropyLoss` is used.

### 7.4 Tokenization And Vocabulary

The tokenizer is a regular expression tokenizer:

```python
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
```

The tokenizer lowercases text and extracts alphanumeric word-like tokens. The vocabulary is built from `train_data["text"]` only. This avoids leakage from validation or test text.

Vocabulary settings:

| Setting | Value |
| --- | --- |
| Vocabulary cap | 20,000 |
| Padding token | `<pad>` |
| Unknown token | `<unk>` |
| Padding index | 0 |
| Unknown index | 1 |
| Main maximum length | 128 |

### 7.5 Dataset And DataLoader

The notebook wraps Hugging Face examples in a PyTorch `Dataset`. Each item returns:

- `input_ids`: token ID tensor,
- `attention_mask`: Boolean mask where real tokens are true,
- `label`: integer class label,
- `text`: original raw text for later failure analysis.

The batch size is 64. Training loaders use shuffling; validation and test loaders do not need shuffling.

### 7.6 Model Definitions

The notebook defines:

- `LSTMClassifier`
- `TransformerEncoderClassifier`
- `count_trainable_parameters`

The parameter count table is exported as `outputs/parameter_counts.csv`.

### 7.7 Training And Evaluation Utilities

The notebook defines a shared `run_epoch` function used for both training and evaluation. It:

1. switches the model into train or eval mode,
2. computes logits,
3. computes cross-entropy loss,
4. performs backward pass and optimizer update during training,
5. stores labels, predictions, and probabilities,
6. returns loss, accuracy, macro F1, labels, predictions, and probabilities.

The shared evaluation path is important because it prevents accidental metric differences between models.

### 7.8 Main Comparison

The main comparison trains the LSTM and Transformer Encoder for 6 epochs each with:

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 6 |
| Dropout | 0.2 |
| Max sequence length | 128 |
| Seed | 42 |
| Runtime | CPU |

After training, both models are evaluated on validation and official test splits. The output table is `outputs/main_results.csv`.

### 7.9 Dataset-Size Ablation

The required ablation tests how performance changes with less training data. The tested fractions are:

- 25% of the training split,
- 50% of the training split,
- 100% of the training split.

Each model is retrained for each fraction and evaluated on the same validation split. The official test split is not used for ablation comparisons.

The 100% ablation result is a retraining inside the ablation loop, so it should be interpreted as the 100% point within the ablation experiment, not as a byte-for-byte duplicate of the main comparison run.

### 7.10 Optional Sequence-Length Ablation

The optional extension compares maximum sequence length 128 vs 256. Both models are retrained at each length and evaluated on validation data.

This tests whether allowing longer input context improves performance enough to justify the extra cost.

### 7.11 Failure Analysis

The notebook builds a table of test-set examples with:

- raw text,
- true label,
- LSTM prediction,
- Transformer prediction,
- LSTM confidence,
- Transformer confidence.

It selects six examples:

- two cases both models got wrong,
- two cases only the LSTM got wrong,
- two cases only the Transformer got wrong.

The selected examples are exported to `outputs/selected_misclassifications.csv`.

### 7.12 Export Tables And Figures

The notebook exports result tables and figures to `outputs/`. These outputs are the evidence base for the final report, presentation, and this manual.

## 8. Output Artifact Guide

### 8.1 Tables

| File | Meaning |
| --- | --- |
| `outputs/parameter_counts.csv` | Trainable parameter count for each model |
| `outputs/main_results.csv` | Validation and final test loss, accuracy, and macro F1 |
| `outputs/dataset_size_ablation.csv` | Validation accuracy and macro F1 at 25%, 50%, and 100% training data |
| `outputs/sequence_length_ablation.csv` | Validation loss, accuracy, and macro F1 at max lengths 128 and 256 |
| `outputs/selected_misclassifications.csv` | Six representative failure-analysis examples |
| `outputs/artifact_manifest.csv` | Compact inventory of generated output artifacts |

### 8.2 Figures

| File | Meaning |
| --- | --- |
| `outputs/main_loss_curves.png` | Training and validation loss curves for the main comparison |
| `outputs/lstm_confusion_matrix.png` | LSTM confusion matrix on the official test split |
| `outputs/transformer_confusion_matrix.png` | Transformer Encoder confusion matrix on the official test split |

The loss-curve figure is used to discuss convergence and overfitting. The confusion matrices are used to discuss class-level mistakes, especially Business vs Sci/Tech.

### 8.3 Final Deliverables

| File | Meaning |
| --- | --- |
| `deliverables/ag_news_one_page_summary.docx` | One-page project plan and result summary |
| `deliverables/ag_news_final_report.docx` | Final report without AI Usage Appendix |
| `deliverables/ag_news_ai_usage_appendix.docx` | Standalone AI Usage Appendix |
| `deliverables/ag_news_final_report_with_ai_appendix.docx` | Combined report and appendix |
| `deliverables/ag_news_lstm_transformer_presentation.pptx` | Final presentation deck |

The file `deliverables/completion_audit.md` records that the generated Word and PowerPoint artifacts were structurally and visually checked, including page counts for the report and appendix and slide export checks for the presentation.

## 9. Results Achieved

### 9.1 Parameter Counts

| Model | Trainable parameters |
| --- | ---: |
| LSTM | 2,825,220 |
| Transformer Encoder | 2,973,444 |

The Transformer Encoder has 148,224 more trainable parameters, about a 5.2% increase over the LSTM. This is close enough for a course-level comparison, but the report correctly treats the comparison as "reasonably comparable" rather than exactly parameter-matched.

### 9.2 Main Validation And Test Results

| Model | Split | Loss | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| LSTM | Validation | 0.3402 | 0.9113 | 0.9108 |
| Transformer Encoder | Validation | 0.3103 | 0.9118 | 0.9115 |
| LSTM | Test | 0.3459 | 0.9120 | 0.9118 |
| Transformer Encoder | Test | 0.3066 | 0.9125 | 0.9126 |

The Transformer Encoder has the best final test metrics, but the margin is small:

- Accuracy lead: 0.0005, or 0.05 percentage points.
- Macro F1 lead: about 0.0008.

The correct interpretation is that both models perform very similarly on the final test split. The Transformer Encoder is slightly ahead in this run, but the result should not be overstated as a decisive architectural win.

### 9.3 Training Behavior

The loss curves show that both models continue reducing training loss while validation loss stops improving and then increases by the final epoch. This indicates overfitting by epoch 6.

Observed pattern:

- The LSTM reaches strong validation accuracy early but develops a larger train-validation loss gap.
- The Transformer Encoder has slightly lower validation and test loss in the final main comparison.
- A future improvement would be to use early stopping or stronger regularization based on validation loss.

### 9.4 Confusion Matrix Findings

Both confusion matrices show that Sports is the easiest class for both models. The most persistent confusion is between Business and Sci/Tech.

The final report records these examples of Business/Sci-Tech confusion:

- LSTM: 177 Business examples predicted as Sci/Tech, and 121 Sci/Tech examples predicted as Business.
- Transformer Encoder: 200 Business examples predicted as Sci/Tech, and 93 Sci/Tech examples predicted as Business.

This is plausible because many news stories about technology companies contain both market language and technology language.

### 9.5 Dataset-Size Ablation

| Model | Training fraction | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: |
| LSTM | 25% | 0.8719 | 0.8713 |
| Transformer Encoder | 25% | 0.8699 | 0.8696 |
| LSTM | 50% | 0.8944 | 0.8940 |
| Transformer Encoder | 50% | 0.8901 | 0.8895 |
| LSTM | 100% | 0.9089 | 0.9082 |
| Transformer Encoder | 100% | 0.9146 | 0.9142 |

The ablation supports the original hypothesis:

- With 25% training data, the LSTM is slightly better.
- With 50% training data, the LSTM is still better.
- With 100% training data, the Transformer Encoder becomes better.

The technical interpretation is that the LSTM is more stable in lower-data settings, while the Transformer Encoder benefits more from the full dataset. The differences are modest, so the result should be described as a trend rather than as proof of universal superiority.

### 9.6 Optional Sequence-Length Ablation

| Model | Max length | Validation loss | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 128 | 0.3402 | 0.9113 | 0.9108 |
| Transformer Encoder | 128 | 0.3103 | 0.9118 | 0.9115 |
| LSTM | 256 | 0.3535 | 0.9065 | 0.9058 |
| Transformer Encoder | 256 | 0.3304 | 0.9062 | 0.9059 |

Increasing the maximum length from 128 to 256 did not improve validation performance. Both models performed worse at 256 tokens.

A likely explanation is that AG News texts are short enough that 128 tokens already capture the useful signal for most examples. Longer sequences also increase computation and may introduce more noisy or weakly useful tokens under the same six-epoch training budget.

### 9.7 Failure Analysis

The selected failure cases show three recurring themes:

- Surface cues can dominate the true news section. For example, an Olympics-related text labeled World was predicted as Sports by both models.
- Business and Sci/Tech overlap heavily when a story discusses a technology company, IPO, chip delay, software company, or computing-related business event.
- Short science or technology items can be ambiguous because there is less context for the model to average over.

The failure analysis is useful because it shows that the remaining errors are mostly semantic boundary cases, not obvious preprocessing failures or label-format bugs.

## 10. What Has Been Achieved

The current project has achieved the required core objectives:

- Built a reproducible Jupyter/PyTorch notebook.
- Loaded one AG News dataset source through Hugging Face Datasets.
- Created a validation split from the official training split.
- Reserved the official test split for final evaluation.
- Built a train-only vocabulary.
- Implemented an LSTM classifier from scratch.
- Implemented a Transformer Encoder classifier from scratch.
- Trained both models under the same main preprocessing and training budget.
- Reported trainable parameter counts.
- Reported accuracy and macro F1.
- Exported loss curves and confusion matrices.
- Completed the required dataset-size ablation.
- Completed an optional sequence-length ablation.
- Exported six selected misclassified examples.
- Drafted final report, AI Usage Appendix, one-page summary, and presentation artifacts.
- Audited final deliverables for structure, page counts, slide counts, and rendered content.

The most important empirical findings are:

- Both models reach about 91.2% final test accuracy.
- The Transformer Encoder is slightly better on final test accuracy and macro F1.
- The LSTM is stronger at 25% and 50% training data.
- The Transformer Encoder benefits more at 100% training data.
- Increasing max sequence length from 128 to 256 hurts validation performance in this run.
- The main error pattern is Business vs Sci/Tech ambiguity.
- Both models show overfitting by the final epoch.

## 11. Reproducibility Notes

### 11.1 Environment

The project dependencies are listed in `requirements.txt`:

```text
datasets
jupyter
matplotlib
numpy
pandas
pip-system-certs
scikit-learn
seaborn
torch
```

The notebook can be run from the project root or from the `notebooks/` directory. It computes `PROJECT_ROOT` so outputs are written to the project-level `outputs/` directory.

### 11.2 Re-running The Notebook

To reproduce the experiment:

1. Install the dependencies.
2. Open `notebooks/ag_news_lstm_transformer.ipynb`.
3. Run all cells in order.
4. Inspect generated outputs in `outputs/`.
5. Confirm report claims still match the generated CSV files and figures.

Because training is stochastic and uses CPU/GPU kernels, exact values can vary by environment. The notebook sets seeds and deterministic flags to reduce variance, but future runs should still treat the generated CSV outputs as the source of truth for report claims.

### 11.3 Regenerating Deliverables

The deliverables can be regenerated with:

```powershell
python tools\build_final_deliverables.py
```

After regeneration, check:

- `deliverables/ag_news_final_report.docx`
- `deliverables/ag_news_ai_usage_appendix.docx`
- `deliverables/ag_news_final_report_with_ai_appendix.docx`
- `deliverables/ag_news_lstm_transformer_presentation.pptx`

The existing completion audit records that the report, appendix, combined report, and presentation were checked on 2026-05-27.

## 12. Interpretation Guidelines For Future Work

Use these rules when modifying or extending the project:

- Do not compare against pretrained models unless the instructor explicitly allows it.
- Do not mix TorchText AG News, raw CSV AG News, and Hugging Face AG News in one experiment.
- Do not tune hyperparameters on the official test split.
- Do not rebuild vocabulary on validation or test text.
- Do not claim the Transformer is decisively superior from the current metrics; the final advantage is small.
- Do not claim 256-token context is better; the current result shows the opposite.
- Do treat Business vs Sci/Tech as the main class-boundary issue.
- Do treat early stopping and regularization as reasonable next experiments.
- Do preserve output CSV files as the evidence source for report and presentation claims.

Good next technical extensions would be:

- early stopping based on validation loss,
- dropout or weight decay tuning,
- multiple random seeds for more robust conclusions,
- a parameter-matched model-size sensitivity study,
- per-class precision, recall, and F1 tables,
- calibration analysis for confidence scores,
- runtime comparison between the two architectures.

## 13. Reference Notes

The manual uses two kinds of evidence:

- Local project evidence: the notebook, exported CSV tables, exported PNG figures, report source, and completion audit.
- External references: dataset cards, library documentation, and foundational model papers.

## 14. References

1. Hugging Face dataset card for `fancyzhx/ag_news`: https://huggingface.co/datasets/fancyzhx/ag_news
2. Hugging Face Datasets loading documentation: https://huggingface.co/docs/datasets/loading
3. Zhang, X., Zhao, J., and LeCun, Y. (2015). "Character-level Convolutional Networks for Text Classification." arXiv:1509.01626. https://arxiv.org/abs/1509.01626
4. Hochreiter, S., and Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation, 9(8), 1735-1780. DOI: https://doi.org/10.1162/neco.1997.9.8.1735
5. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). "Attention Is All You Need." arXiv:1706.03762. https://arxiv.org/abs/1706.03762
6. PyTorch `torch.nn.LSTM` documentation: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
7. PyTorch `torch.nn.TransformerEncoderLayer` documentation: https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html
8. PyTorch `torch.nn.CrossEntropyLoss` documentation: https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html
9. scikit-learn `train_test_split` documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
10. scikit-learn `f1_score` documentation: https://sklearn.org/1.8/modules/generated/sklearn.metrics.f1_score.html
