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

## Generated Deliverables

- `ag_news_one_page_summary.docx`
- `ag_news_final_report.docx`
- `ag_news_ai_usage_appendix.docx`
- `ag_news_final_report_with_ai_appendix.docx`
- `ag_news_lstm_transformer_presentation.pptx`

## Verification Notes

- DOCX/PPTX files were structurally verified with `python-docx` and `python-pptx`.
- The final report DOCX contains 6 tables and 3 embedded figures.
- The presentation PPTX contains 10 slides and 3 embedded result figures.
- Full visual rendering with the bundled DOCX renderer could not be completed because LibreOffice/`soffice` is not installed in the current environment.
