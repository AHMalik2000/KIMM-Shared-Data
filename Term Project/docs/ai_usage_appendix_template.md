# AI Usage Appendix

Maximum length: 1 page.

## Tools Used

- Codex was used for project planning, document review, and document refinement.
- Add any other AI tools used by the team before submission.

## What AI Was Used For

- Drafting and reviewing the project plan, report outline, presentation outline, and AI Usage Appendix structure.
- Checking the documents against the assignment requirements, including dataset-source consistency, test-set use, ablation requirements, failure analysis, and AI transparency.
- Drafting report- and presentation-facing summaries from verified notebook outputs after the notebook was run.
- Identifying result takeaways: small Transformer Encoder test advantage, lower-data LSTM advantage, overfitting by epoch 6, and Business/Sci-Tech confusion.

## Incorrect, Incomplete, Or Misleading AI Outputs

Record any AI output that required correction. Current known issues and examples to watch for:

- The team must reject or correct any code that uses the test set during model selection.
- The team must reject or correct any code that mixes Hugging Face, TorchText, and raw CSV dataset sources.
- The team must check for tensor shape errors, missing padding masks, and unsupported result interpretations.
- Early scaffold text still described outputs as pending after the notebook had been run; this was corrected in the project README and deliverable outlines.
- Some notebook markdown cells still contained interpretation placeholders after outputs were generated; these were replaced with concise interpretations for the report draft.

## Team Modifications

Describe what the team changed after reviewing AI output. Current and expected examples:

- Kept the experiment constrained to one dataset source: Hugging Face `fancyzhx/ag_news`.
- Required train-only vocabulary construction and official test-set use only for final evaluation.
- Required both models to use the same preprocessing, vocabulary, metrics, and comparable training budget.
- Verified that labels are integer class indices 0, 1, 2, 3 and that the train/validation/test split was documented.
- Interpreted the ablation results conservatively rather than overclaiming a large overall model difference.

## Team Decisions

List decisions made by the team, not by AI. Current decisions to confirm before submission:

- Main comparison: LSTM classifier vs Transformer Encoder classifier trained from scratch.
- Dataset source: Hugging Face `datasets.load_dataset("fancyzhx/ag_news")`.
- Required ablation: dataset size at 25%, 50%, and 100%.
- Main maximum sequence length: 128.
- Optional extension completed: 128 vs 256 sequence length.
- Final interpretation: both models performed similarly on the final test split, with the Transformer Encoder slightly ahead; the LSTM was more data-efficient in the 25% and 50% ablation settings.
- Failure-analysis examples were selected from the official test split and exported to `outputs/selected_misclassifications.csv`.

## Effect Of AI Use

AI helped organize the project requirements, identify required deliverables, turn the assignment into a controlled experiment plan, and draft concise interpretations from notebook outputs. The team remains responsible for verifying the notebook run, checking all exported artifacts, correcting mistakes, and approving final claims before submission.
