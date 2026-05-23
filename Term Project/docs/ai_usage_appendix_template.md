# AI Usage Appendix

Maximum length: 1 page.

## Tools Used

- Codex was used for project planning, document review, and document refinement.
- Add any other AI tools used by the team before submission.

## What AI Was Used For

- Drafting and reviewing the project plan, report outline, presentation outline, and AI Usage Appendix structure.
- Checking the documents against the assignment requirements, including dataset-source consistency, test-set use, ablation requirements, failure analysis, and AI transparency.
- Future entries to add if used: PyTorch code drafting, tensor-shape debugging, metric code, visualization code, concept explanations, or report editing.

## Incorrect, Incomplete, Or Misleading AI Outputs

Record any AI output that required correction. Current known issues and examples to watch for:

- The team must reject or correct any code that uses the test set during model selection.
- The team must reject or correct any code that mixes Hugging Face, TorchText, and raw CSV dataset sources.
- The team must check for tensor shape errors, missing padding masks, and unsupported result interpretations.
- Add specific corrections encountered during notebook implementation.

## Team Modifications

Describe what the team changed after reviewing AI output. Current and expected examples:

- Kept the experiment constrained to one dataset source: Hugging Face `fancyzhx/ag_news`.
- Required train-only vocabulary construction and official test-set use only for final evaluation.
- Required both models to use the same preprocessing, vocabulary, metrics, and comparable training budget.
- Add later: corrected data split logic, verified label indices, adjusted hyperparameters, or rewrote explanations using course concepts.

## Team Decisions

List decisions made by the team, not by AI. Current decisions to confirm before submission:

- Main comparison: LSTM classifier vs Transformer Encoder classifier trained from scratch.
- Dataset source: Hugging Face `datasets.load_dataset("fancyzhx/ag_news")`.
- Required ablation: dataset size at 25%, 50%, and 100%.
- Main maximum sequence length: 128, with 128 vs 256 kept as an optional extension.
- Add later: final model configurations, final interpretation of results, and examples selected for failure analysis.

## Effect Of AI Use

AI helped organize the project requirements, identify required deliverables, and turn the assignment into a controlled experiment plan. The team remains responsible for running the notebook, verifying outputs, correcting mistakes, and writing final claims based only on inspected results.
