# Final Presentation Outline

Target length: 8-10 minutes plus Q&A.

Use this outline as a slide plan. Result placeholders have been replaced with the current notebook outputs exported in `outputs/`.

## Slide 1: Title And Research Question

- Project title.
- Team members.
- Research question: controlled comparison of LSTM and Transformer Encoder classifiers on AG News.
- Suggested title: "LSTM vs Transformer Encoder for AG News Classification".
- Add presenter names before submission.

## Slide 2: Task And Dataset

- AG News four-class text classification.
- Class labels: `World`, `Sports`, `Business`, `Sci/Tech`.
- Dataset source and train/validation/test split.
- Split sizes: train 108,000; validation 12,000; test 7,600.
- Test split is balanced with 1,900 examples per class.

## Slide 3: Shared Data Pipeline

- Tokenization.
- Train-only vocabulary construction.
- Padding and truncation.
- Maximum sequence length.
- Same processed inputs for both models.
- Mention train-only vocabulary construction to show leakage control.

## Slide 4: Model Architectures

- LSTM classifier summary.
- Transformer Encoder classifier summary.
- Trainable parameter counts.
- Key hyperparameters.
- Parameter counts: LSTM 2,825,220; Transformer Encoder 2,973,444.
- Shared setup: vocabulary size 20,000, maximum sequence length 128, dropout 0.2.

## Slide 5: Main Experimental Setup

- Optimizer, learning rate, batch size, epochs, seed, device.
- Metrics: accuracy and macro F1-score.
- Fair comparison controls.
- Current settings: Adam, learning rate 1e-3, batch size 64, 6 epochs, seed 42, CPU.

## Slide 6: Main Results

- Validation: LSTM accuracy 0.9113, macro F1 0.9108; Transformer Encoder accuracy 0.9118, macro F1 0.9115.
- Test: LSTM accuracy 0.9120, macro F1 0.9118; Transformer Encoder accuracy 0.9125, macro F1 0.9126.
- Interpretation: Transformer Encoder was slightly better, but the main-result margin was small.

## Slide 7: Training Behavior

- Training and validation loss curves.
- Convergence speed.
- Overfitting or underfitting observations.
- Use `outputs/main_loss_curves.png`.
- Observation: training loss keeps falling while validation loss rises after earlier epochs, showing overfitting by epoch 6.

## Slide 8: Confusion Matrix And Class-Level Errors

- Confusion matrices.
- Most frequently confused classes.
- Possible reasons for confusion.
- Use `outputs/lstm_confusion_matrix.png` and `outputs/transformer_confusion_matrix.png`.
- Main class-level observation: Sports is easiest for both models; Business and Sci/Tech are the most frequent confusion pair.

## Slide 9: Ablation Study

- Dataset-size hypothesis.
- 25%, 50%, and 100% training data results.
- Interpretation of data efficiency and sensitivity.
- Optional sequence-length result if completed.
- Dataset-size ablation validation accuracy:
  - 25%: LSTM 0.8719; Transformer Encoder 0.8699.
  - 50%: LSTM 0.8944; Transformer Encoder 0.8901.
  - 100%: LSTM 0.9089; Transformer Encoder 0.9146.
- Interpretation: LSTM is stronger with less data, while Transformer Encoder benefits more from the full training set.
- Optional sequence-length result: 256 tokens reduced validation metrics for both models compared with 128 tokens.

## Slide 10: Failure Analysis And Conclusion

- Representative misclassified examples.
- Whether models failed similarly or differently.
- Main takeaways.
- One reasonable next experiment.
- Brief AI-use transparency note.
- Use `outputs/selected_misclassifications.csv`.
- Suggested takeaway: errors often come from overlap between Business and Sci/Tech, or news items whose surface words point to a different section than the dataset label.
