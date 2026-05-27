# Completion Audit

## Assignment Requirements

| Requirement | Evidence |
|---|---|
| LSTM classifier trained from scratch | `notebooks/ag_news_lstm_transformer.ipynb`; summarized in `docs/final_report.md` Section 3 |
| Transformer Encoder classifier trained from scratch | `notebooks/ag_news_lstm_transformer.ipynb`; summarized in `docs/final_report.md` Section 3 |
| One dataset source only | `docs/final_report.md` Section 2 documents Hugging Face `fancyzhx/ag_news` only |
| Train/validation/test split documented | `docs/final_report.md` Section 2 and `docs/one_page_project_summary.md` |
| Vocabulary built from training data only | `docs/final_report.md` Section 2 and notebook preprocessing section |
| Accuracy and macro F1 reported | `outputs/main_results.csv`; `docs/final_report.md` Section 5 |
| Training and validation loss curves | `outputs/main_loss_curves.png`; embedded in `deliverables/ag_news_final_report.docx` |
| Confusion matrices | `outputs/lstm_confusion_matrix.png` and `outputs/transformer_confusion_matrix.png`; embedded in `deliverables/ag_news_final_report.docx` |
| Model hyperparameters and parameter counts | `outputs/parameter_counts.csv`; `docs/final_report.md` Section 3 |
| One focused ablation with hypothesis and interpretation | `outputs/dataset_size_ablation.csv`; `docs/final_report.md` Section 6 |
| At least five misclassified examples | `outputs/selected_misclassifications.csv`; `docs/final_report.md` Section 7 |
| AI Usage Appendix | `docs/ai_usage_appendix_final.md`; `deliverables/ag_news_ai_usage_appendix.docx` |
| One-page plan/summary | `docs/one_page_project_summary.md`; `deliverables/ag_news_one_page_summary.docx` |
| Final report | `docs/final_report.md`; `deliverables/ag_news_final_report.docx` |
| Final presentation | `docs/final_presentation_script.md`; `deliverables/ag_news_lstm_transformer_presentation.pptx` |
| Reference report/deck format and tone | Report follows the reference report pattern: Table of Figures, Motivation, Background, Model Development, Experimental Verification, Interpretation, Conclusion, Appendix/References. Presentation follows the reference deck pattern: 4:3 academic seminar slides, title slide, Table of Contents, top blue rule, small slide/page markers, and concise technical bullets. |

## Generated Deliverables

- `ag_news_one_page_summary.docx`
- `ag_news_final_report.docx`
- `ag_news_ai_usage_appendix.docx`
- `ag_news_final_report_with_ai_appendix.docx`
- `ag_news_lstm_transformer_presentation.pptx`

## Verification Notes

- DOCX/PPTX files were structurally verified with `python-docx` and `python-pptx`.
- The final report DOCX contains 6 tables and 3 embedded figures.
- The presentation PPTX contains 11 slides and 3 embedded result figures.
- Microsoft PowerPoint rendered the updated presentation to PNG successfully for visual QA.
- Microsoft Word COM verification on 2026-05-27 reported the final report as 6 pages, the AI Usage Appendix as 1 page, and the combined report plus appendix as 7 pages.
- Microsoft PowerPoint COM verification on 2026-05-27 reported 11 slides at 4:3 size and exported all 11 slides to PNG at 1280x960.
- Exported slide PNGs were checked for nonblank rendered content.

## Sections 16-18 Final Audit

| PDF section | Requirement or grading area | Final status |
|---|---|---|
| 16 Deliverables | Initial project plan: one page describing split, preprocessing, model configuration, ablation plan, and expected analysis | Satisfied by `docs/initial_project_plan.md` and `deliverables/ag_news_one_page_summary.docx`. |
| 16 Deliverables | Source code as a clean, runnable, reproducible Jupyter notebook | Satisfied by `notebooks/ag_news_lstm_transformer.ipynb`; notebook JSON parsed successfully and contains the controlled AG News pipeline, model definitions, training loop, evaluation, and artifact generation. |
| 16 Deliverables | Final report: 4-6 pages excluding AI Usage Appendix | Satisfied by `deliverables/ag_news_final_report.docx`; Word reported 6 pages. |
| 16 Deliverables | AI Usage Appendix: maximum 1 page | Satisfied by `deliverables/ag_news_ai_usage_appendix.docx`; Word reported 1 page. |
| 16 Deliverables | Final presentation: 8-10 minute presentation plus Q&A | Satisfied by `deliverables/ag_news_lstm_transformer_presentation.pptx` and `docs/final_presentation_script.md`; deck has 11 concise slides and a presenter script for the expected talk flow. |
| 17 Minimum Requirements | Train one LSTM classifier | Satisfied by notebook and reported results. |
| 17 Minimum Requirements | Train one Transformer Encoder classifier | Satisfied by notebook and reported results. |
| 17 Minimum Requirements | Conduct one ablation study | Satisfied by dataset-size ablation at 25%, 50%, and 100% in `outputs/dataset_size_ablation.csv`. |
| 17 Minimum Requirements | Report accuracy and macro F1-score | Satisfied by `outputs/main_results.csv` and report tables. |
| 17 Minimum Requirements | Provide training and validation loss curves | Satisfied by `outputs/main_loss_curves.png` and embedded report figure. |
| 17 Minimum Requirements | Provide a confusion matrix | Satisfied by LSTM and Transformer confusion matrices in `outputs/` and embedded report figures. |
| 17 Minimum Requirements | Analyze at least five misclassified examples | Satisfied by 6 examples in `outputs/selected_misclassifications.csv` and the report failure-analysis table. |
| 17 Minimum Requirements | Submit an AI Usage Appendix | Satisfied by Markdown and DOCX appendix artifacts. |
| 17 Minimum Requirements | Submit final report, code, and presentation | Satisfied by DOCX report, Jupyter notebook, and PPTX deck. |
| 17 Optional Extensions | Optional extension coverage | Included sequence-length ablation at 128 vs 256 in `outputs/sequence_length_ablation.csv`. |
| 18 Grading Rubric | Problem definition and dataset understanding | Covered in report Motivation/Background and dataset documentation. |
| 18 Grading Rubric | Correct implementation and reproducibility | Covered by notebook, fixed seed, one dataset source, train-only vocabulary, reported hyperparameters, and output artifacts. |
| 18 Grading Rubric | Fair model comparison | Covered by shared split, preprocessing, vocabulary, max length, metrics, and comparable training budget. |
| 18 Grading Rubric | Ablation study | Covered by clear dataset-size hypothesis, controlled variables, result table, and interpretation. |
| 18 Grading Rubric | Result interpretation | Covered by metric, loss-curve, confusion-matrix, and model-behavior discussion. |
| 18 Grading Rubric | Failure analysis | Covered by concrete misclassified examples, confidence scores, and likely error reasons. |
| 18 Grading Rubric | AI usage transparency and verification | Covered by the AI Usage Appendix. |
| 18 Grading Rubric | Presentation quality | Covered by a concise 4:3 academic seminar deck, rendered with PowerPoint for visual QA. |
