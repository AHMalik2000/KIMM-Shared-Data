from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PptRGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches as PptInches, Pt as PptPt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
DELIVERABLES = ROOT / "deliverables"

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

LSTM_CM = [
    [1717, 62, 66, 55],
    [14, 1859, 15, 12],
    [58, 13, 1652, 177],
    [58, 18, 121, 1703],
]

TRANSFORMER_CM = [
    [1731, 34, 64, 71],
    [31, 1820, 35, 14],
    [58, 8, 1634, 200],
    [46, 11, 93, 1750],
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str | float) -> str:
    return f"{float(value):.4f}"


def table_md(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def failure_reason(row: dict[str, str]) -> str:
    text = row["text"]
    true_label = row["true_label_name"]
    lstm = row["lstm_pred_name"]
    transformer = row["transformer_pred_name"]
    if "Olympics" in text:
        return "Sports terms dominate the short text even though the dataset section is World."
    if "Intel" in text:
        return "A technology company story overlaps Business and Sci/Tech vocabulary."
    if "Wildfires" in text:
        return "Weather prediction language mixes science terms with world-event framing."
    if "Google IPO" in text:
        return "Finance and technology cues conflict; the IPO framing points toward Business."
    if "T. rex" in text:
        return "Very short science item gives limited context; the Transformer overweights business-like wording."
    if "IBM" in text:
        return "Company hiring story overlaps labor/business cues and computing-company Sci/Tech label."
    return f"Boundary case between {true_label}, {lstm}, and {transformer} news language."


def excerpt(text: str, limit: int = 96) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def build_markdown() -> dict[str, str]:
    main = read_csv("main_results.csv")
    params = read_csv("parameter_counts.csv")
    dataset_ablation = read_csv("dataset_size_ablation.csv")
    seq_ablation = read_csv("sequence_length_ablation.csv")
    failures = read_csv("selected_misclassifications.csv")

    main_rows = [
        [r["model"], r["split"], fmt(r["loss"]), fmt(r["accuracy"]), fmt(r["macro_f1"])]
        for r in main
    ]
    param_rows = [[r["model"], f'{int(r["trainable_parameters"]):,}'] for r in params]
    hyperparam_rows = [
        ["Vocabulary size", "20,000", "20,000"],
        ["Embedding dimension", "128", "128"],
        ["Encoder depth", "1 bidirectional LSTM layer", "2 Transformer Encoder layers"],
        ["Hidden / feedforward size", "Hidden size 128 per direction", "Feedforward dimension 512"],
        ["Attention heads", "Not applicable", "4"],
        ["Pooling strategy", "Masked mean pooling", "Masked mean pooling"],
        ["Dropout", "0.2", "0.2"],
        ["Optimizer / learning rate", "Adam / 1e-3", "Adam / 1e-3"],
        ["Epochs / batch size", "6 / 64", "6 / 64"],
    ]
    ablation_rows = [
        [r["model"], f'{float(r["train_fraction"]):.0%}', fmt(r["valid_accuracy"]), fmt(r["valid_macro_f1"])]
        for r in dataset_ablation
    ]
    seq_rows = [
        [r["model"], r["max_sequence_length"], fmt(r["valid_loss"]), fmt(r["valid_accuracy"]), fmt(r["valid_macro_f1"])]
        for r in seq_ablation
    ]
    failure_rows = [
        [
            str(i + 1),
            excerpt(r["text"]),
            r["true_label_name"],
            r["lstm_pred_name"],
            r["transformer_pred_name"],
            fmt(r["lstm_confidence"]),
            fmt(r["transformer_confidence"]),
            failure_reason(r),
        ]
        for i, r in enumerate(failures)
    ]

    one_pager = f"""# One-Page Project Plan And Result Summary

## Project Question

This project compares a from-scratch LSTM classifier and a from-scratch Transformer Encoder classifier on AG News text classification. The core question is how recurrent sequence modeling and self-attention differ in classification performance, convergence behavior, data efficiency, and failure patterns under a controlled PyTorch/Jupyter workflow.

## Dataset And Preprocessing

- Dataset source: Hugging Face `fancyzhx/ag_news`, loaded with `datasets.load_dataset("fancyzhx/ag_news")`.
- Split: 108,000 train, 12,000 validation from the official training split with `seed=42`, and 7,600 official test examples for final evaluation only.
- Labels: `World`, `Sports`, `Business`, `Sci/Tech`, encoded as integer classes `0, 1, 2, 3`.
- Pipeline: regex tokenization, vocabulary built from training data only, 20,000-token vocabulary, padding/truncation to 128 tokens for the main comparison, and padding masks for pooling/attention.

## Models And Training

- LSTM: embedding layer, one-layer bidirectional LSTM, masked mean pooling, dropout, and linear classifier; 2,825,220 trainable parameters.
- Transformer Encoder: embedding layer, learned positional embeddings, two encoder layers, four attention heads, feedforward dimension 512, masked mean pooling, dropout, and linear classifier; 2,973,444 trainable parameters.
- Shared training setup: Adam, learning rate `1e-3`, batch size 64, dropout 0.2, 6 epochs, seed 42, CPU runtime.

## Experiments And Final Findings

- Main test result: LSTM accuracy 0.9120 and macro F1 0.9118; Transformer Encoder accuracy 0.9125 and macro F1 0.9126. The Transformer Encoder is slightly ahead, but the gap is small.
- Required ablation: dataset size at 25%, 50%, and 100%. LSTM led at 25% and 50%, while Transformer Encoder led at 100%, supporting the hypothesis that LSTM is more stable with limited data and Transformer benefits more from full data.
- Optional extension: sequence length 128 vs 256. Longer 256-token inputs reduced validation metrics for both models, so 128 remains the better setting in this run.
- Failure analysis: six selected test errors cover both-wrong, LSTM-only wrong, and Transformer-only wrong cases. Most errors reflect Business/Sci-Tech overlap, news-section ambiguity, or short texts with misleading surface cues.
"""

    report = f"""# LSTM vs Transformer Encoder for AG News Classification

## 1. Introduction

This project compares two neural text classifiers trained from scratch on the AG News four-class classification task: a bidirectional LSTM classifier and a Transformer Encoder classifier. The goal is not simply to maximize accuracy. The project evaluates how recurrent sequence modeling and self-attention differ in classification performance, training behavior, convergence patterns, data efficiency, and failure cases.

Both models use the same dataset source, split procedure, tokenizer, vocabulary, maximum sequence length, optimizer family, training budget, and evaluation metrics. This controlled setup makes the comparison more informative than a raw leaderboard-style accuracy result. The main research question is: under a shared preprocessing and training pipeline, how do the LSTM and Transformer Encoder differ on AG News classification?

## 2. Dataset And Preprocessing

The project uses Hugging Face `fancyzhx/ag_news` as the only dataset source. The official training split was divided into 108,000 training examples and 12,000 validation examples using `train_test_split(test_size=0.1, seed=42)`. The official test split contains 7,600 examples and was reserved for final evaluation only.

The four labels are `World`, `Sports`, `Business`, and `Sci/Tech`. The official test split is balanced, with 1,900 examples per class. The training split is also nearly balanced: 26,991 World, 26,966 Sports, 27,100 Business, and 26,943 Sci/Tech examples. Labels were verified as integer class indices `0, 1, 2, 3`, which matches the requirement for `torch.nn.CrossEntropyLoss`.

Texts were tokenized with a reproducible regular-expression tokenizer. The vocabulary was built only from the training split to avoid vocabulary leakage. The vocabulary size was capped at 20,000 tokens, with reserved indices for padding and unknown tokens. The main comparison used a maximum sequence length of 128 tokens. Both models received the same padded and truncated token IDs, and padding masks were used during LSTM pooling and Transformer attention.

## 3. Models

The LSTM classifier uses an embedding layer, a one-layer bidirectional LSTM, masked mean pooling, dropout, and a final linear classifier. The Transformer Encoder classifier uses an embedding layer, learned positional embeddings, two Transformer Encoder layers, four attention heads, feedforward dimension 512, masked mean pooling, dropout, and a final linear classifier. Both models were trained from scratch, without pretrained embeddings or pretrained language models.

    {table_md(["Model", "Trainable parameters"], param_rows)}

    {table_md(["Setting", "LSTM", "Transformer Encoder"], hyperparam_rows)}

The parameter counts are reasonably comparable for an introductory architecture comparison. The Transformer Encoder has about 5.2% more trainable parameters than the LSTM, so the report treats its small final metric advantage cautiously.

## 4. Experiments

The main comparison trained both models for 6 epochs using Adam, learning rate `1e-3`, batch size 64, dropout 0.2, seed 42, and CPU execution. The evaluation metrics were accuracy and macro F1-score. Macro F1 was included because AG News has multiple classes and the assignment requires class-sensitive evaluation, even though the dataset is nearly balanced.

The required ablation changed one major factor: the fraction of training data. The tested fractions were 25%, 50%, and 100% of the training split. The hypothesis was that the LSTM would be more stable with less data, while the Transformer Encoder would benefit more from the full dataset because self-attention has greater capacity.

An optional extension compared maximum sequence lengths of 128 and 256 tokens. This tested whether longer context improved validation performance enough to justify the additional training cost.

## 5. Results

{table_md(["Model", "Split", "Loss", "Accuracy", "Macro F1"], main_rows)}

Both models reached similar final performance. On the official test split, the LSTM achieved 0.9120 accuracy and 0.9118 macro F1. The Transformer Encoder achieved 0.9125 accuracy and 0.9126 macro F1. The Transformer Encoder therefore performed slightly better on the final test metrics, but the difference is small and should not be overinterpreted as a decisive architectural win.

The loss curves show that both models continued to reduce training loss while validation loss stopped improving and then increased by the final epoch. This indicates overfitting by epoch 6. The LSTM had a sharper training-validation loss gap by the final epoch, while the Transformer Encoder retained slightly lower validation and test loss. A reasonable next experiment would tune early stopping or regularization using the validation set.

The confusion matrices show that Sports was the easiest class for both models. The largest recurring confusion was between Business and Sci/Tech. For the LSTM, 177 Business items were predicted as Sci/Tech and 121 Sci/Tech items were predicted as Business. For the Transformer Encoder, 200 Business items were predicted as Sci/Tech and 93 Sci/Tech items were predicted as Business. This pattern is expected because technology-company stories often contain both market and technical language.

Figures used for the final report:

![Training and validation loss curves](../outputs/main_loss_curves.png)

![LSTM final test confusion matrix](../outputs/lstm_confusion_matrix.png)

![Transformer Encoder final test confusion matrix](../outputs/transformer_confusion_matrix.png)

## 6. Ablation Study

Hypothesis: the LSTM will be more stable with limited training data, while the Transformer Encoder will benefit more from the full dataset. The changed variable was training-set fraction. The controlled variables were dataset source, train/validation split, tokenizer, vocabulary, maximum sequence length, optimizer, learning rate, batch size, epoch count, metrics, and validation evaluation.

{table_md(["Model", "Training fraction", "Validation accuracy", "Validation macro F1"], ablation_rows)}

The ablation supports the hypothesis. At 25% training data, the LSTM slightly outperformed the Transformer Encoder. At 50%, the LSTM again had higher validation accuracy and macro F1. At 100%, the Transformer Encoder moved ahead. This suggests that the LSTM was more data-efficient in lower-data settings, while the Transformer Encoder benefited more from access to the full training split.

The optional sequence-length ablation is shown below.

{table_md(["Model", "Max length", "Validation loss", "Validation accuracy", "Validation macro F1"], seq_rows)}

Increasing maximum sequence length from 128 to 256 did not improve validation performance. Both models performed worse at 256 tokens. Under the current training budget and architecture sizes, longer context added cost without improving classification quality.

## 7. Failure Analysis

The table below gives six representative misclassified test examples. It includes cases missed by both models, cases missed only by the LSTM, and cases missed only by the Transformer Encoder.

{table_md(["#", "Input excerpt", "True label", "LSTM prediction", "Transformer prediction", "LSTM conf.", "Transformer conf.", "Likely reason"], failure_rows)}

The selected errors show that the two models often fail on semantic boundary cases. Some examples contain surface words strongly associated with a different news section, such as an Olympics item labeled World but predicted as Sports by both models. Other examples involve technology companies or IPOs, where Business and Sci/Tech labels overlap. The models therefore fail similarly when the text has ambiguous section cues, but they also show different single-model errors: the LSTM misread some science/world-style items, while the Transformer misread some short Sci/Tech items as Business.

## 8. Conclusion

Both from-scratch models performed well on AG News under the controlled setup, reaching about 91.2% final test accuracy. The Transformer Encoder achieved the best final test metrics, but only by a small margin. The LSTM was stronger in lower-data ablation settings, while the Transformer Encoder benefited more from the full training set.

The required dataset-size ablation supports the original data-efficiency hypothesis. The optional sequence-length ablation showed that simply increasing maximum sequence length from 128 to 256 did not improve validation metrics. The most reasonable next experiment would be to tune early stopping or regularization, because both models show overfitting by the final epoch.
"""
    report = report.replace("\n    |", "\n|")

    appendix = """# AI Usage Appendix

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
"""

    slides_md = """# Final Presentation Slide Plan

## Slide 1 - Research Question

Claim: We compare recurrent sequence modeling and self-attention under one controlled AG News pipeline.

Speaker note: The goal is not just high accuracy. The project asks whether LSTM and Transformer Encoder models behave differently in performance, convergence, data efficiency, and failure cases.

## Slide 2 - Dataset And Split

Claim: AG News is balanced and controlled enough for a fair four-class comparison.

- Source: Hugging Face `fancyzhx/ag_news`
- Train/validation/test: 108,000 / 12,000 / 7,600
- Labels: World, Sports, Business, Sci/Tech

## Slide 3 - Shared Pipeline

Claim: The comparison uses the same inputs for both models.

- Train-only vocabulary, 20,000 tokens
- Regex tokenizer
- Max length 128 for main comparison
- Padding masks used during pooling and attention

## Slide 4 - Model Architectures

Claim: The two models are different architectures but comparable in size.

- LSTM: bidirectional, one layer, masked mean pooling, 2.83M parameters
- Transformer Encoder: two layers, four heads, learned positions, 2.97M parameters

## Slide 5 - Main Test Result

Claim: Both models are close, with a slight Transformer Encoder edge.

- LSTM test accuracy 0.9120, macro F1 0.9118
- Transformer test accuracy 0.9125, macro F1 0.9126

## Slide 6 - Training Behavior

Claim: Both models overfit by epoch 6.

- Training loss keeps falling
- Validation loss rises late in training
- Early stopping or stronger regularization is the next experiment

## Slide 7 - Confusion Matrices

Claim: Business and Sci/Tech are the main source of class-level errors.

- Sports is easiest for both models
- Business/Sci-Tech overlap is frequent because many company stories have both market and technology cues

## Slide 8 - Dataset-Size Ablation

Claim: LSTM is more stable with less data, while Transformer benefits from the full dataset.

- 25%: LSTM 0.8719 vs Transformer 0.8699 validation accuracy
- 50%: LSTM 0.8944 vs Transformer 0.8901
- 100%: Transformer 0.9146 vs LSTM 0.9089

## Slide 9 - Sequence-Length Extension And Failures

Claim: More context did not help in this run, and errors are mostly boundary cases.

- 256 tokens reduced validation metrics for both models
- Failure examples show Olympics/World ambiguity and Business/Sci-Tech overlap

## Slide 10 - Conclusion

Claim: Architecture differences appear most clearly in data efficiency and failure patterns, not in a large final accuracy gap.

- Transformer Encoder slightly wins the final test comparison
- LSTM performs better with limited data
- Future work: early stopping, regularization, and class-level error reduction
"""

    return {
        "one_pager": one_pager,
        "report": report,
        "appendix": appendix,
        "slides": slides_md,
    }


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(9)
    run.bold = bold
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def style_doc(doc: Document, title: str) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(10.5)
    for style_name, size, color in [
        ("Heading 1", 15, RGBColor(46, 116, 181)),
        ("Heading 2", 12.5, RGBColor(46, 116, 181)),
        ("Heading 3", 11.5, RGBColor(31, 77, 120)),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = color
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = RGBColor(11, 37, 69)
    p.paragraph_format.space_after = Pt(4)


def add_para(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.08


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.add_run(item)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, header in enumerate(headers):
        set_cell_text(hdr[i], header, bold=True)
        set_cell_shading(hdr[i], "F2F4F7")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph()


def add_image(doc: Document, path: Path, caption: str, width: float = 6.1) -> None:
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(5)
        for run in p.runs:
            run.font.size = Pt(8.5)
            run.italic = True


def write_docx_files(markdown: dict[str, str]) -> None:
    DELIVERABLES.mkdir(exist_ok=True)
    main = read_csv("main_results.csv")
    params = read_csv("parameter_counts.csv")
    dataset_ablation = read_csv("dataset_size_ablation.csv")
    seq_ablation = read_csv("sequence_length_ablation.csv")
    failures = read_csv("selected_misclassifications.csv")

    # One-pager
    doc = Document()
    style_doc(doc, "One-Page Project Plan And Result Summary")
    add_para(doc, "Research question: compare a from-scratch LSTM classifier and Transformer Encoder classifier on AG News under one controlled PyTorch/Jupyter pipeline.")
    add_bullets(doc, [
        "Dataset: Hugging Face fancyzhx/ag_news only; 108,000 train, 12,000 validation, 7,600 official test examples.",
        "Preprocessing: train-only 20,000-token vocabulary, regex tokenizer, max length 128, padding masks for pooling and attention.",
        "Models: bidirectional LSTM with 2,825,220 parameters; Transformer Encoder with 2,973,444 parameters.",
        "Training: Adam, learning rate 1e-3, batch size 64, dropout 0.2, 6 epochs, seed 42.",
        "Final test result: Transformer Encoder slightly leads, 0.9125 accuracy and 0.9126 macro F1 vs LSTM 0.9120 and 0.9118.",
        "Ablation: LSTM leads at 25% and 50% data; Transformer Encoder leads at 100%, supporting the data-efficiency hypothesis.",
        "Failure analysis: six selected test errors show Business/Sci-Tech overlap and ambiguous news-section cues.",
    ])
    doc.save(DELIVERABLES / "ag_news_one_page_summary.docx")

    # Final report
    doc = Document()
    style_doc(doc, "LSTM vs Transformer Encoder for AG News Classification")
    sections = markdown["report"].split("\n## ")
    for index, section_text in enumerate(sections):
        if index == 0:
            continue
        lines = section_text.splitlines()
        doc.add_heading(lines[0], level=1)
        body = "\n".join(lines[1:]).strip()
        if lines[0].startswith("3. Models"):
            add_para(doc, "The LSTM uses an embedding layer, one-layer bidirectional LSTM, masked mean pooling, dropout, and a linear classifier. The Transformer Encoder uses embeddings, learned positional embeddings, two encoder layers, four heads, masked mean pooling, dropout, and a classifier.")
            add_table(doc, ["Model", "Trainable parameters"], [[r["model"], f'{int(r["trainable_parameters"]):,}'] for r in params], [3.0, 2.0])
            add_table(
                doc,
                ["Setting", "LSTM", "Transformer Encoder"],
                [
                    ["Vocabulary size", "20,000", "20,000"],
                    ["Embedding dimension", "128", "128"],
                    ["Encoder depth", "1 bidirectional LSTM layer", "2 Transformer Encoder layers"],
                    ["Hidden / feedforward", "128 per direction", "512 feedforward"],
                    ["Attention heads", "Not applicable", "4"],
                    ["Pooling", "Masked mean pooling", "Masked mean pooling"],
                    ["Dropout", "0.2", "0.2"],
                    ["Optimizer / LR", "Adam / 1e-3", "Adam / 1e-3"],
                    ["Epochs / batch", "6 / 64", "6 / 64"],
                ],
                [1.7, 2.2, 2.3],
            )
            add_para(doc, "Both models were trained from scratch with no pretrained embeddings or pretrained language models.")
        elif lines[0].startswith("5. Results"):
            add_table(doc, ["Model", "Split", "Loss", "Accuracy", "Macro F1"], [[r["model"], r["split"], fmt(r["loss"]), fmt(r["accuracy"]), fmt(r["macro_f1"])] for r in main], [2.2, 0.8, 0.8, 0.9, 0.9])
            add_para(doc, "Both models reached similar final performance. The Transformer Encoder was slightly better on final test metrics, but the gap was small.")
            add_image(doc, OUT / "main_loss_curves.png", "Figure 1. Training and validation loss curves.")
            add_image(doc, OUT / "lstm_confusion_matrix.png", "Figure 2. LSTM final test confusion matrix.", 5.3)
            add_image(doc, OUT / "transformer_confusion_matrix.png", "Figure 3. Transformer Encoder final test confusion matrix.", 5.3)
            add_para(doc, "Both models overfit by epoch 6. Sports was easiest for both models, and Business/Sci/Tech confusion was the main class-level error pattern.")
        elif lines[0].startswith("6. Ablation"):
            add_para(doc, "Hypothesis: LSTM is more stable with limited data, while Transformer Encoder benefits more from full data. The changed variable was training-set fraction; all other major settings were controlled.")
            add_table(doc, ["Model", "Train fraction", "Accuracy", "Macro F1"], [[r["model"], f'{float(r["train_fraction"]):.0%}', fmt(r["valid_accuracy"]), fmt(r["valid_macro_f1"])] for r in dataset_ablation], [2.4, 1.0, 1.0, 1.0])
            add_para(doc, "The ablation supports the hypothesis: LSTM led at 25% and 50%, while Transformer Encoder led at 100%.")
            add_table(doc, ["Model", "Max length", "Loss", "Accuracy", "Macro F1"], [[r["model"], r["max_sequence_length"], fmt(r["valid_loss"]), fmt(r["valid_accuracy"]), fmt(r["valid_macro_f1"])] for r in seq_ablation], [2.2, 0.9, 0.8, 0.9, 0.9])
            add_para(doc, "The optional sequence-length extension showed that 256-token inputs reduced validation performance for both models.")
        elif lines[0].startswith("7. Failure"):
            add_para(doc, "The selected examples cover both-wrong, LSTM-only wrong, and Transformer-only wrong cases.")
            rows = []
            for i, r in enumerate(failures):
                rows.append([
                    str(i + 1),
                    excerpt(r["text"], 72),
                    r["true_label_name"],
                    r["lstm_pred_name"],
                    r["transformer_pred_name"],
                    f'{float(r["lstm_confidence"]):.2f}',
                    f'{float(r["transformer_confidence"]):.2f}',
                    failure_reason(r),
                ])
            add_table(doc, ["#", "Input excerpt", "True", "LSTM", "Transformer", "LSTM c.", "Trans. c.", "Likely reason"], rows, [0.25, 1.45, 0.6, 0.7, 0.75, 0.45, 0.5, 2.0])
            add_para(doc, "Most remaining errors are semantic boundary cases involving class overlap or misleading surface cues, not label-format or preprocessing failures.")
        else:
            for para in body.split("\n\n"):
                clean = para.strip()
                if clean and not clean.startswith("|") and not clean.startswith("- `"):
                    add_para(doc, clean)
    doc.save(DELIVERABLES / "ag_news_final_report.docx")

    doc.add_section(WD_SECTION.NEW_PAGE)
    doc.add_heading("AI Usage Appendix", level=1)
    for para in markdown["appendix"].split("\n\n")[1:]:
        if para.startswith("## "):
            doc.add_heading(para[3:], level=2)
        else:
            add_para(doc, para)
    doc.save(DELIVERABLES / "ag_news_final_report_with_ai_appendix.docx")

    # Standalone appendix
    doc = Document()
    style_doc(doc, "AI Usage Appendix")
    for para in markdown["appendix"].split("\n\n")[1:]:
        if para.startswith("## "):
            doc.add_heading(para[3:], level=1)
        else:
            add_para(doc, para)
    doc.save(DELIVERABLES / "ag_news_ai_usage_appendix.docx")


def add_ppt_title(slide, kicker: str, title: str) -> None:
    tx = slide.shapes.add_textbox(PptInches(0.55), PptInches(0.35), PptInches(11.8), PptInches(0.35))
    p = tx.text_frame.paragraphs[0]
    p.text = kicker.upper()
    p.font.size = PptPt(10)
    p.font.bold = True
    p.font.color.rgb = PptRGBColor(54, 96, 146)
    box = slide.shapes.add_textbox(PptInches(0.55), PptInches(0.72), PptInches(11.8), PptInches(0.8))
    p = box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = PptPt(28)
    p.font.bold = True
    p.font.color.rgb = PptRGBColor(11, 37, 69)


def add_ppt_bullets(slide, items: list[str], x=0.75, y=1.65, w=5.7, h=4.7, size=17) -> None:
    box = slide.shapes.add_textbox(PptInches(x), PptInches(y), PptInches(w), PptInches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.size = PptPt(size)
        p.font.color.rgb = PptRGBColor(35, 35, 35)
        p.space_after = PptPt(8)


def add_metric(slide, x: float, y: float, value: str, label: str, color=(46, 116, 181)) -> None:
    shape = slide.shapes.add_shape(1, PptInches(x), PptInches(y), PptInches(2.45), PptInches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PptRGBColor(242, 244, 247)
    shape.line.color.rgb = PptRGBColor(210, 216, 224)
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = value
    p.font.size = PptPt(24)
    p.font.bold = True
    p.font.color.rgb = PptRGBColor(*color)
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = label
    p2.font.size = PptPt(11)
    p2.font.color.rgb = PptRGBColor(70, 70, 70)
    p2.alignment = PP_ALIGN.CENTER


def add_footer(slide, num: int) -> None:
    box = slide.shapes.add_textbox(PptInches(11.9), PptInches(6.95), PptInches(0.7), PptInches(0.2))
    p = box.text_frame.paragraphs[0]
    p.text = str(num)
    p.font.size = PptPt(8)
    p.font.color.rgb = PptRGBColor(120, 120, 120)
    p.alignment = PP_ALIGN.RIGHT


def write_pptx() -> None:
    prs = Presentation()
    prs.slide_width = PptInches(13.333)
    prs.slide_height = PptInches(7.5)
    blank = prs.slide_layouts[6]
    slides = [
        ("Research Question", "One controlled AG News pipeline compares recurrent modeling and self-attention.", ["Compare LSTM and Transformer Encoder classifiers trained from scratch.", "Analyze accuracy, macro F1, convergence, ablation behavior, and failure cases.", "No pretrained language models or pretrained embeddings."]),
        ("Dataset", "AG News provides a balanced four-class classification task.", ["Source: Hugging Face fancyzhx/ag_news.", "Split: 108,000 train, 12,000 validation, 7,600 official test.", "Labels: World, Sports, Business, Sci/Tech."]),
        ("Shared Pipeline", "Both models receive the same processed inputs.", ["Regex tokenizer and train-only vocabulary.", "Vocabulary size: 20,000 tokens.", "Maximum sequence length: 128 for the main comparison.", "Padding masks used during pooling and attention."]),
        ("Architectures", "Different sequence models, comparable parameter scale.", ["LSTM: bidirectional one-layer recurrent encoder with masked mean pooling.", "Transformer Encoder: two layers, four heads, learned positional embeddings.", "Parameter counts are close enough for a fair introductory comparison."]),
        ("Main Result", "Both models are close; Transformer Encoder has a slight final edge.", ["LSTM test accuracy 0.9120 and macro F1 0.9118.", "Transformer test accuracy 0.9125 and macro F1 0.9126.", "The margin is small, so the conclusion should be cautious."]),
        ("Training Behavior", "Both models show overfitting by epoch 6.", ["Training loss continues falling.", "Validation loss rises late in training.", "Early stopping or stronger regularization is the most direct next experiment."]),
        ("Class-Level Errors", "Business and Sci/Tech are the main confusion pair.", ["Sports is easiest for both models.", "Business/Sci-Tech stories often mix company, market, and technology cues.", "Confusion matrices support the failure-analysis examples."]),
        ("Dataset-Size Ablation", "LSTM is stronger with less data; Transformer benefits from full data.", ["25% data: LSTM 0.8719 vs Transformer 0.8699 validation accuracy.", "50% data: LSTM 0.8944 vs Transformer 0.8901.", "100% data: Transformer 0.9146 vs LSTM 0.9089."]),
        ("Sequence Length And Failures", "Longer context did not help, and errors are semantic boundary cases.", ["256 tokens reduced validation metrics for both models.", "Selected errors include Olympics/World ambiguity and Business/Sci-Tech overlap.", "The models fail similarly on ambiguous section cues but differ on some single-model errors."]),
        ("Conclusion", "Architecture differences appear most clearly in data efficiency and failure patterns.", ["Transformer Encoder slightly wins the final test comparison.", "LSTM performs better in limited-data settings.", "Future work: early stopping, regularization, and class-level error reduction."]),
    ]
    image_map = {
        5: OUT / "main_loss_curves.png",
        6: OUT / "lstm_confusion_matrix.png",
        7: OUT / "transformer_confusion_matrix.png",
    }
    for idx, (kicker, title, bullets) in enumerate(slides, start=1):
        slide = prs.slides.add_slide(blank)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = PptRGBColor(250, 250, 248)
        add_ppt_title(slide, kicker, title)
        if idx == 1:
            add_metric(slide, 0.75, 2.05, "91.25%", "best test accuracy")
            add_metric(slide, 3.45, 2.05, "6", "training epochs", (89, 89, 89))
            add_metric(slide, 6.15, 2.05, "3", "ablation sizes", (31, 77, 120))
            add_ppt_bullets(slide, bullets, x=0.85, y=3.65, w=11.0, h=2.0, size=18)
        elif idx == 4:
            add_metric(slide, 0.75, 2.0, "2.83M", "LSTM parameters")
            add_metric(slide, 3.45, 2.0, "2.97M", "Transformer parameters")
            add_ppt_bullets(slide, bullets, x=0.85, y=3.55, w=11.0, h=2.0, size=18)
        elif idx in image_map:
            img = image_map[idx]
            if img.exists():
                slide.shapes.add_picture(str(img), PptInches(6.8), PptInches(1.65), width=PptInches(5.45))
            add_ppt_bullets(slide, bullets, x=0.75, y=1.75, w=5.6, h=4.7, size=17)
        else:
            add_ppt_bullets(slide, bullets, x=0.9, y=1.85, w=11.2, h=4.8, size=20)
        add_footer(slide, idx)
    prs.save(DELIVERABLES / "ag_news_lstm_transformer_presentation.pptx")


def main() -> None:
    markdown = build_markdown()
    DELIVERABLES.mkdir(exist_ok=True)
    (DOCS / "one_page_project_summary.md").write_text(markdown["one_pager"], encoding="utf-8")
    (DOCS / "final_report.md").write_text(markdown["report"], encoding="utf-8")
    (DOCS / "ai_usage_appendix_final.md").write_text(markdown["appendix"], encoding="utf-8")
    (DOCS / "final_presentation_script.md").write_text(markdown["slides"], encoding="utf-8")
    write_docx_files(markdown)
    write_pptx()


if __name__ == "__main__":
    main()
