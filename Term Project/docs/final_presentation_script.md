# Final Presentation Slide Plan

Reference format: 4:3 academic seminar deck, white background, blue title rule, top-left slide numbering, bottom-left course/lab footer, and bottom-right page count.

## Slide 1 - Title

LSTM vs Transformer Encoder for AG News Text Classification.

## Slide 2 - Table of Contents

Introduction and Motivation; Dataset and Preprocessing; Model Development; Main Comparison Results; Ablation Study; Failure Analysis; Conclusion and Future Work.

## Slide 3 - Introduction

Claim: Both models are common sequence classifiers, but they process text differently.

## Slide 4 - Dataset

Claim: AG News provides a controlled four-class classification task.

## Slide 5 - Preprocessing

Claim: Both models use the same processed input pipeline.

## Slide 6 - Model Development

Claim: The two models are different but comparable in size.

## Slide 7 - Main Results

Claim: Both models perform similarly, with a slight Transformer Encoder advantage.

## Slide 8 - Training Behavior

Claim: Both models show overfitting by the final epoch.

## Slide 9 - Confusion Matrices

Claim: Business and Sci/Tech are the main source of class-level errors.

## Slide 10 - Ablation Study

Claim: LSTM is more stable with less data; Transformer benefits from the full dataset.

## Slide 11 - Conclusion

Claim: Architecture differences appear most clearly in data efficiency and failure patterns.

## Anticipated Q&A

Q1: Why is the Transformer Encoder better with more data?
A1: Its self-attention layers have higher capacity to model token interactions, so they appear to benefit more once the full training split is available.

Q2: Would early stopping change the conclusion?
A2: It might improve both models because validation loss rises by epoch 6, but the final test gap is already small, so the conservative conclusion would likely remain.

Q3: Why use macro F1 if AG News is balanced?
A3: Macro F1 is required by the assignment and still confirms that performance is not hiding a severe class-specific failure.

Q4: Why did 256 tokens hurt performance?
A4: Longer inputs add cost and noise under the same architecture and epoch budget; the key AG News cues usually appear early.
