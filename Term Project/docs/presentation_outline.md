# Final Presentation Outline

Target length: 8-10 minutes plus Q&A.

Use this outline as a slide plan. Replace result placeholders only after the notebook has generated verified tables, figures, and failure-analysis examples.

## Slide 1: Title And Research Question

- Project title.
- Team members.
- Research question: controlled comparison of LSTM and Transformer Encoder classifiers on AG News.
- Status to fill: final project title and presenter names.

## Slide 2: Task And Dataset

- AG News four-class text classification.
- Class labels: `World`, `Sports`, `Business`, `Sci/Tech`.
- Dataset source and train/validation/test split.
- Figure/table to fill: split sizes and class distribution.

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
- Table to fill: final model settings and parameter counts.

## Slide 5: Main Experimental Setup

- Optimizer, learning rate, batch size, epochs, seed, device.
- Metrics: accuracy and macro F1-score.
- Fair comparison controls.
- Keep this slide factual; do not discuss final performance here.

## Slide 6: Main Results

- Accuracy and macro F1 table.
- Short interpretation of which model performed better and why.
- Table to fill: validation and final test metrics for both models.

## Slide 7: Training Behavior

- Training and validation loss curves.
- Convergence speed.
- Overfitting or underfitting observations.
- Figure to fill: loss curves for both models.

## Slide 8: Confusion Matrix And Class-Level Errors

- Confusion matrices.
- Most frequently confused classes.
- Possible reasons for confusion.
- Figure to fill: confusion matrices for final evaluated models.

## Slide 9: Ablation Study

- Dataset-size hypothesis.
- 25%, 50%, and 100% training data results.
- Interpretation of data efficiency and sensitivity.
- Optional sequence-length result if completed.
- Table/figure to fill: ablation results by train fraction and model.

## Slide 10: Failure Analysis And Conclusion

- Representative misclassified examples.
- Whether models failed similarly or differently.
- Main takeaways.
- One reasonable next experiment.
- Brief AI-use transparency note.
- Table to fill: at least five representative misclassified examples.
