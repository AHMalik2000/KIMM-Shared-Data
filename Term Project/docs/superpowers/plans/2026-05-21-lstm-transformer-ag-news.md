# LSTM vs Transformer Encoder AG News Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PyTorch/Jupyter project comparing an LSTM classifier and a Transformer Encoder classifier trained from scratch on AG News.

**Architecture:** Keep the notebook organized as a linear experiment pipeline: setup, data loading, preprocessing, model definitions, training utilities, main comparison, ablation, failure analysis, and exportable report figures/tables. Shared preprocessing and evaluation utilities must be used by both models to preserve a fair comparison.

**Tech Stack:** Python, Jupyter, PyTorch, Hugging Face `datasets`, scikit-learn metrics, pandas, matplotlib, seaborn.

---

## File Structure

- Create: `notebooks/ag_news_lstm_transformer.ipynb`
  - Main reproducible experiment notebook.
- Create: `outputs/README.md`
  - Documents generated result artifacts without committing large binary outputs unless needed.
- Modify as results become available: `docs/initial_project_plan.md`, `docs/report_outline.md`, `docs/ai_usage_appendix_template.md`, and `docs/presentation_outline.md`
  - Keep these aligned with final experimental decisions and verified outputs.

## Task 1: Environment And Notebook Skeleton

**Files:**
- Create: `notebooks/ag_news_lstm_transformer.ipynb`
- Create: `outputs/README.md`

- [ ] **Step 1: Create notebook sections**

Create markdown sections in this order:

```markdown
# AG News LSTM vs Transformer Encoder
## 1. Setup And Reproducibility
## 2. Load Dataset
## 3. Inspect Labels And Splits
## 4. Tokenization And Vocabulary
## 5. Dataset And DataLoader
## 6. Model Definitions
## 7. Training And Evaluation Utilities
## 8. Main Comparison
## 9. Dataset-Size Ablation
## 10. Optional Sequence-Length Ablation
## 11. Failure Analysis
## 12. Export Tables And Figures
## 13. AI Usage Notes
```

- [ ] **Step 2: Add setup code**

```python
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

- [ ] **Step 3: Create outputs documentation**

Create `outputs/README.md`:

```markdown
# Outputs

Generated figures, tables, and intermediate result files should be written here during notebook execution.

Do not treat generated outputs as verified report evidence until the notebook cell that produced them has been rerun in order and inspected.
```

- [ ] **Step 4: Run notebook setup cells**

Run the setup cells and confirm the selected device prints without error.

## Task 2: Data Loading And Preprocessing

**Files:**
- Modify: `notebooks/ag_news_lstm_transformer.ipynb`

- [ ] **Step 1: Load the dataset**

```python
raw_dataset = load_dataset("ag_news")
split_dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=SEED)

train_data = split_dataset["train"]
valid_data = split_dataset["test"]
test_data = raw_dataset["test"]

label_names = raw_dataset["train"].features["label"].names
label_names
```

- [ ] **Step 2: Verify split sizes and label range**

```python
split_summary = pd.DataFrame(
    {
        "split": ["train", "valid", "test"],
        "samples": [len(train_data), len(valid_data), len(test_data)],
    }
)

all_labels = set(train_data["label"]) | set(valid_data["label"]) | set(test_data["label"])
assert all_labels == {0, 1, 2, 3}, all_labels
split_summary
```

- [ ] **Step 3: Add tokenizer and train-only vocabulary**

```python
import re

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1
VOCAB_SIZE = 20_000
MAX_LEN = 128

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

counter = Counter()
for text in train_data["text"]:
    counter.update(tokenize(text))

most_common = counter.most_common(VOCAB_SIZE - 2)
itos = [PAD_TOKEN, UNK_TOKEN] + [token for token, _ in most_common]
stoi = {token: idx for idx, token in enumerate(itos)}

def encode(text: str, max_len: int = MAX_LEN) -> tuple[list[int], list[int]]:
    ids = [stoi.get(token, UNK_IDX) for token in tokenize(text)]
    ids = ids[:max_len]
    attention_mask = [1] * len(ids)
    pad_count = max_len - len(ids)
    if pad_count > 0:
        ids = ids + [PAD_IDX] * pad_count
        attention_mask = attention_mask + [0] * pad_count
    return ids, attention_mask
```

- [ ] **Step 4: Confirm vocabulary and encoded sample**

```python
assert itos[PAD_IDX] == PAD_TOKEN
assert itos[UNK_IDX] == UNK_TOKEN
assert len(itos) <= VOCAB_SIZE
sample_ids, sample_mask = encode(train_data[0]["text"])
assert len(sample_ids) == MAX_LEN
assert len(sample_mask) == MAX_LEN
```

## Task 3: Dataset And DataLoader

**Files:**
- Modify: `notebooks/ag_news_lstm_transformer.ipynb`

- [ ] **Step 1: Define PyTorch dataset**

```python
class AGNewsTorchDataset(Dataset):
    def __init__(self, hf_dataset, max_len: int = MAX_LEN):
        self.texts = hf_dataset["text"]
        self.labels = hf_dataset["label"]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = encode(self.texts[idx], self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.texts[idx],
        }
```

- [ ] **Step 2: Define DataLoader helper**

```python
BATCH_SIZE = 64

def make_loader(hf_dataset, batch_size: int = BATCH_SIZE, shuffle: bool = False, max_len: int = MAX_LEN) -> DataLoader:
    dataset = AGNewsTorchDataset(hf_dataset, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(train_data, shuffle=True)
valid_loader = make_loader(valid_data)
test_loader = make_loader(test_data)
```

- [ ] **Step 3: Verify a batch**

```python
batch = next(iter(train_loader))
assert batch["input_ids"].shape[1] == MAX_LEN
assert batch["attention_mask"].shape == batch["input_ids"].shape
assert batch["label"].min().item() >= 0
assert batch["label"].max().item() <= 3
```

## Task 4: Model Definitions

**Files:**
- Modify: `notebooks/ag_news_lstm_transformer.ipynb`

- [ ] **Step 1: Define parameter counter**

```python
def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 2: Implement LSTM classifier**

```python
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))
```

- [ ] **Step 3: Implement Transformer Encoder classifier**

```python
class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.2,
        max_len: int = MAX_LEN,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        padding_mask = ~attention_mask
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))
```

- [ ] **Step 4: Instantiate and count parameters**

```python
vocab_size = len(itos)
lstm_model = LSTMClassifier(vocab_size).to(device)
transformer_model = TransformerEncoderClassifier(vocab_size).to(device)

parameter_table = pd.DataFrame(
    [
        {"model": "LSTM", "trainable_parameters": count_trainable_parameters(lstm_model)},
        {"model": "Transformer Encoder", "trainable_parameters": count_trainable_parameters(transformer_model)},
    ]
)
parameter_table
```

## Task 5: Training And Evaluation Utilities

**Files:**
- Modify: `notebooks/ag_news_lstm_transformer.ipynb`

- [ ] **Step 1: Implement train/evaluate loops**

```python
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }

def train_model(model, train_loader, valid_loader, epochs: int = 6, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        valid_metrics = run_epoch(model, valid_loader)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_accuracy": valid_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "valid_macro_f1": valid_metrics["macro_f1"],
        }
        history.append(row)
        print(row)
    return pd.DataFrame(history)
```

- [ ] **Step 2: Implement final evaluation helper**

```python
def evaluate_model(model, loader):
    metrics = run_epoch(model, loader)
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "labels": metrics["labels"],
        "preds": metrics["preds"],
        "probs": metrics["probs"],
    }
```

## Task 6: Main Comparison

**Files:**
- Modify: `notebooks/ag_news_lstm_transformer.ipynb`

- [ ] **Step 1: Train both models**

```python
set_seed(SEED)
lstm_model = LSTMClassifier(vocab_size).to(device)
lstm_history = train_model(lstm_model, train_loader, valid_loader, epochs=6, lr=1e-3)

set_seed(SEED)
transformer_model = TransformerEncoderClassifier(vocab_size).to(device)
transformer_history = train_model(transformer_model, train_loader, valid_loader, epochs=6, lr=1e-3)
```

- [ ] **Step 2: Evaluate on validation and final test**

```python
lstm_valid = evaluate_model(lstm_model, valid_loader)
transformer_valid = evaluate_model(transformer_model, valid_loader)

lstm_test = evaluate_model(lstm_model, test_loader)
transformer_test = evaluate_model(transformer_model, test_loader)

main_results = pd.DataFrame(
    [
        {"model": "LSTM", "split": "valid", "accuracy": lstm_valid["accuracy"], "macro_f1": lstm_valid["macro_f1"]},
        {"model": "Transformer Encoder", "split": "valid", "accuracy": transformer_valid["accuracy"], "macro_f1": transformer_valid["macro_f1"]},
        {"model": "LSTM", "split": "test", "accuracy": lstm_test["accuracy"], "macro_f1": lstm_test["macro_f1"]},
        {"model": "Transformer Encoder", "split": "test", "accuracy": transformer_test["accuracy"], "macro_f1": transformer_test["macro_f1"]},
    ]
)
main_results
```

- [ ] **Step 3: Plot loss curves and confusion matrices**

Use `matplotlib` and `seaborn` to plot train/validation loss curves and confusion matrices for both models. Save figures into `outputs/`.

## Task 7: Required Dataset-Size Ablation

**Files:**
- Modify: `notebooks/ag_news_lstm_transformer.ipynb`

- [ ] **Step 1: Define subset helper**

```python
def make_fraction_dataset(hf_dataset, fraction: float, seed: int = SEED):
    indices = np.arange(len(hf_dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    keep = int(len(indices) * fraction)
    return hf_dataset.select(indices[:keep].tolist())
```

- [ ] **Step 2: Train both models at each fraction**

```python
ablation_rows = []
for fraction in [0.25, 0.50, 1.00]:
    subset = make_fraction_dataset(train_data, fraction)
    subset_loader = make_loader(subset, shuffle=True)

    set_seed(SEED)
    ab_lstm = LSTMClassifier(vocab_size).to(device)
    train_model(ab_lstm, subset_loader, valid_loader, epochs=6, lr=1e-3)
    ab_lstm_valid = evaluate_model(ab_lstm, valid_loader)
    ablation_rows.append({"model": "LSTM", "train_fraction": fraction, "valid_accuracy": ab_lstm_valid["accuracy"], "valid_macro_f1": ab_lstm_valid["macro_f1"]})

    set_seed(SEED)
    ab_transformer = TransformerEncoderClassifier(vocab_size).to(device)
    train_model(ab_transformer, subset_loader, valid_loader, epochs=6, lr=1e-3)
    ab_transformer_valid = evaluate_model(ab_transformer, valid_loader)
    ablation_rows.append({"model": "Transformer Encoder", "train_fraction": fraction, "valid_accuracy": ab_transformer_valid["accuracy"], "valid_macro_f1": ab_transformer_valid["macro_f1"]})

ablation_results = pd.DataFrame(ablation_rows)
ablation_results
```

- [ ] **Step 3: Interpret the ablation**

Write a markdown cell answering whether the results support the hypothesis that the LSTM is more stable with limited data and the Transformer benefits more from the full dataset.

## Task 8: Failure Analysis

**Files:**
- Modify: `notebooks/ag_news_lstm_transformer.ipynb`

- [ ] **Step 1: Build prediction table**

```python
failure_df = pd.DataFrame(
    {
        "text": test_data["text"],
        "true_label": lstm_test["labels"],
        "lstm_pred": lstm_test["preds"],
        "transformer_pred": transformer_test["preds"],
        "lstm_confidence": [max(p) for p in lstm_test["probs"]],
        "transformer_confidence": [max(p) for p in transformer_test["probs"]],
    }
)
failure_df["true_label_name"] = failure_df["true_label"].map(lambda x: label_names[x])
failure_df["lstm_pred_name"] = failure_df["lstm_pred"].map(lambda x: label_names[x])
failure_df["transformer_pred_name"] = failure_df["transformer_pred"].map(lambda x: label_names[x])
failure_df["lstm_wrong"] = failure_df["lstm_pred"] != failure_df["true_label"]
failure_df["transformer_wrong"] = failure_df["transformer_pred"] != failure_df["true_label"]
failure_df.head()
```

- [ ] **Step 2: Select at least five examples**

Select examples covering:

```python
both_wrong = failure_df[failure_df["lstm_wrong"] & failure_df["transformer_wrong"]].head(2)
lstm_only_wrong = failure_df[failure_df["lstm_wrong"] & ~failure_df["transformer_wrong"]].head(2)
transformer_only_wrong = failure_df[~failure_df["lstm_wrong"] & failure_df["transformer_wrong"]].head(2)

selected_failures = pd.concat([both_wrong, lstm_only_wrong, transformer_only_wrong]).head(6)
selected_failures[
    ["text", "true_label_name", "lstm_pred_name", "transformer_pred_name", "lstm_confidence", "transformer_confidence"]
]
```

- [ ] **Step 3: Add explanations**

Create a markdown table with at least five selected examples and a possible reason for each error: ambiguous wording, class overlap, insufficient context, truncation, tokenization issue, overfitting, underfitting, or model limitation.

## Task 9: Update Deliverable Documents

**Files:**
- Modify: `docs/report_outline.md`
- Modify: `docs/presentation_outline.md`
- Modify: `docs/ai_usage_appendix_template.md`

- [ ] **Step 1: Fill report with verified results**

Use notebook-generated tables and figures only after rerunning the notebook in order.

- [ ] **Step 2: Fill AI Usage Appendix**

Record AI tools used, what was generated, what was wrong or incomplete, what the team modified, and which decisions were made by the team.

- [ ] **Step 3: Fill presentation outline**

Replace outline bullets with concrete final results, plots, and takeaways.

## Verification Checklist

- [ ] Hugging Face `ag_news` is the only dataset source.
- [ ] Vocabulary is built from training data only.
- [ ] Official test split is used only for final evaluation.
- [ ] Both models use the same processed inputs in the main comparison.
- [ ] Labels are verified as `0..3`.
- [ ] Padding is masked during pooling and Transformer attention.
- [ ] `model.train()` and `model.eval()` are used correctly.
- [ ] Accuracy and macro F1-score are reported.
- [ ] Training and validation loss curves are generated.
- [ ] Confusion matrices are generated.
- [ ] Dataset-size ablation changes one major variable at a time.
- [ ] At least five misclassified examples are analyzed.
- [ ] AI Usage Appendix is completed and limited to one page.
