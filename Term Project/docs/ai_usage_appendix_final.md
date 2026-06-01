# AI Usage Appendix

## Tools Used

Codex was used for project planning, code drafting support, debugging review, document review, notebook-result summarization, report editing, presentation drafting, and final deliverable packaging.

## What AI Was Used For

AI was used to translate the assignment requirements into a controlled experiment plan, check the scaffold against dataset-source and test-set rules, draft and review PyTorch notebook components, draft report and presentation text from verified notebook outputs, and identify concise interpretations of the metric tables, ablation results, loss curves, confusion matrices, and selected misclassifications. For code-level work, AI assistance was used to draft or review the model class structure, shared `run_epoch` training/evaluation utility, plotting functions, and confusion-matrix/failure-analysis table generation. All generated code was inspected and run cell-by-cell, and the masked mean pooling and Transformer padding-mask logic were manually checked against the attention masks.

## Incorrect, Incomplete, Or Misleading AI Outputs

Some early scaffold text still described outputs as pending after the notebook had already been run. That wording was corrected in the project README, report notes, and presentation notes. Some notebook markdown cells also contained interpretation placeholders after outputs were generated; those were replaced with result-aware interpretations. During code review, no instances were found of three common AI-assisted code errors: applying `softmax` before `CrossEntropyLoss`, normalizing confusion matrices when raw error counts were needed, or using the official test split during model selection or ablation. The team also corrected interpretation wording so that no claim treated the small Transformer Encoder metric lead as a large or decisive result.

## Team Modifications And Decisions

The team kept the project constrained to one dataset source, Hugging Face `fancyzhx/ag_news`, and required the official test split to be used only for final evaluation. The team selected dataset-size ablation as the required ablation and kept sequence length as an optional extension. The team decided to interpret the results conservatively: the Transformer Encoder was slightly better on the final test split, while the LSTM was more data-efficient in the 25% and 50% training-data settings. The team also added explicit checks for integer labels, train-only vocabulary construction, padding masks, and reproducible seeds rather than relying on generated code without verification.

## Effect Of AI Use

AI reduced the time needed to organize requirements, audit deliverables, draft code scaffolds, and write clear explanations. The final responsibility remains with the team: all notebook outputs, figures, tables, code, and claims must be reviewed and approved before submission.
