# AI Usage Appendix

## Tools Used

Codex was used for project planning, document review, notebook-result summarization, report drafting, presentation drafting, and final deliverable packaging.

## What AI Was Used For

AI was used to translate the assignment requirements into a controlled experiment plan, check the scaffold against dataset-source and test-set rules, draft report and presentation text from verified notebook outputs, and identify concise interpretations of the metric tables, ablation results, loss curves, confusion matrices, and selected misclassifications.

## Incorrect, Incomplete, Or Misleading AI Outputs

Some early scaffold text still described outputs as pending after the notebook had already been run. That wording was corrected in the project README, report notes, and presentation notes. Some notebook markdown cells also contained interpretation placeholders after outputs were generated; those were replaced with result-aware interpretations. The team also had to ensure that no claim treated the small Transformer Encoder metric lead as a large or decisive result.

## Team Modifications And Decisions

The team kept the project constrained to one dataset source, Hugging Face `fancyzhx/ag_news`, and required the official test split to be used only for final evaluation. The team selected dataset-size ablation as the required ablation and kept sequence length as an optional extension. The team decided to interpret the results conservatively: the Transformer Encoder was slightly better on the final test split, while the LSTM was more data-efficient in the 25% and 50% training-data settings.

## Effect Of AI Use

AI reduced the time needed to organize requirements, audit deliverables, and draft clear explanations. The final responsibility remains with the team: all notebook outputs, figures, tables, code, and claims must be reviewed and approved before submission.
