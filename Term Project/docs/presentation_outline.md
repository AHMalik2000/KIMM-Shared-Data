# Final Presentation Outline

Target length: 8-10 minutes plus Q&A.

## Slide 1: Title And Research Question

- Project title.
- Team members.
- Research question: controlled comparison of LSTM and Transformer Encoder classifiers on AG News.

## Slide 2: Task And Dataset

- AG News four-class text classification.
- Class labels: `World`, `Sports`, `Business`, `Sci/Tech`.
- Dataset source and train/validation/test split.

## Slide 3: Shared Data Pipeline

- Tokenization.
- Train-only vocabulary construction.
- Padding and truncation.
- Maximum sequence length.
- Same processed inputs for both models.

## Slide 4: Model Architectures

- LSTM classifier summary.
- Transformer Encoder classifier summary.
- Trainable parameter counts.
- Key hyperparameters.

## Slide 5: Main Experimental Setup

- Optimizer, learning rate, batch size, epochs, seed, device.
- Metrics: accuracy and macro F1-score.
- Fair comparison controls.

## Slide 6: Main Results

- Accuracy and macro F1 table.
- Short interpretation of which model performed better and why.

## Slide 7: Training Behavior

- Training and validation loss curves.
- Convergence speed.
- Overfitting or underfitting observations.

## Slide 8: Confusion Matrix And Class-Level Errors

- Confusion matrices.
- Most frequently confused classes.
- Possible reasons for confusion.

## Slide 9: Ablation Study

- Dataset-size hypothesis.
- 25%, 50%, and 100% training data results.
- Interpretation of data efficiency and sensitivity.
- Optional sequence-length result if completed.

## Slide 10: Failure Analysis And Conclusion

- Representative misclassified examples.
- Whether models failed similarly or differently.
- Main takeaways.
- One reasonable next experiment.
- Brief AI-use transparency note.
