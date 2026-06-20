# Presentation Script — LSTM vs Transformer Encoder for AG News Text Classification

**Course:** Introduction to Deep Learning · Term Project
**Team:** Malik Muhammad Abdullah Hassan (02621070) · Syed Muhammad Shabih Ahmad (02623026) · Zain Shabbir (02623024)
**Target length:** 8–10 minutes (≈9 minutes spoken) plus Q&A
**Aligned to:** the 13-slide `Final_Presentation` deck

**Presenter map**
- **Part 1 — Abdullah Hassan** (Slides 1–5): Title, Contents, Introduction & Motivation, Dataset, Preprocessing — *≈3 min*
- **Part 2 — Shabih Ahmad** (Slides 6–9): Model Development, Main Results, Training Behavior, Confusion Matrices — *≈3.5 min*
- **Part 3 — Zain Shabbir** (Slides 10–13): Ablation Studies, Failure Analysis, Conclusion, Q&A — *≈3 min*

---

## Part 1 — Abdullah Hassan *(≈3 minutes)*

### Title (Slide 1)
Good morning, everyone. Thank you for being here. Our term project for Introduction to Deep Learning is titled *"LSTM versus Transformer Encoder for AG News Text Classification."* I'm Abdullah Hassan, and I'm presenting today together with Shabih Ahmad and Zain Shabbir. Over the next few minutes we'll walk you through how two very different sequence models behave on the same text classification task when everything else is held constant.

### Table of Contents (Slide 2)
Here is the path we'll take. I'll begin with our motivation, the dataset, and how we prepared the text. Shabih will then cover the two model architectures and our main comparison results. Finally, Zain will present the ablation studies, our failure analysis, and the conclusions we draw from the project.

### Introduction and Motivation (Slide 3)
So, why compare an LSTM with a Transformer Encoder at all? Both are standard sequence classifiers, but they read text in fundamentally different ways. The LSTM is recurrent: it processes tokens one at a time and carries information forward through a hidden state. The Transformer Encoder, by contrast, uses self-attention, which lets every token interact directly with every other token in the sequence.

The key point of our project is in the last line on this slide: our goal was *not* simply to find the higher-accuracy model. The more interesting question is how recurrence and self-attention differ in their training behavior, their data efficiency, and the kinds of mistakes they make. To answer that fairly, we trained both models entirely from scratch — no pretrained embeddings, no BERT or GPT-style components — under one controlled experimental setup.

### Dataset (Slide 4)
We used the AG News dataset, a standard four-class news topic classification benchmark. Each example is a short news item labelled as one of four categories: World, Sports, Business, or Sci/Tech. Following the assignment's rule, we drew the data from a single source — Hugging Face's `fancyzhx/ag_news` — to avoid mixing different versions of the same benchmark.

As the table shows, we split the official training set into 108,000 training examples and 12,000 validation examples using a fixed seed of 42. The official test set of 7,600 examples — balanced at 1,900 per class — was reserved strictly for final evaluation. We never used it for model selection or tuning. We also confirmed that the labels are integer indices zero through three, which is exactly what PyTorch's cross-entropy loss expects.

### Preprocessing (Slide 5)
Both models receive text through the same preprocessing pipeline — and this is central to making the comparison fair. Raw text is lowercased and tokenized with a regular-expression tokenizer. We then build a vocabulary *from the training text only* — this prevents any leakage from validation or test data — and cap it at 20,000 entries. Index zero is reserved for padding and index one for unknown tokens. Sequences are converted to integer IDs and padded or truncated to a fixed maximum length, and an attention mask marks which positions are real tokens and which are padding. Because both models share this identical pipeline, any difference in results can be attributed to the architecture rather than the data.

**With that foundation in place, I'll hand over to Shabih to describe the two models.**

---

## Part 2 — Shabih Ahmad *(≈3.5 minutes)*

### Model Development (Slide 6)
Thank you, Abdullah. Let's look at the two architectures. The first is the LSTM classifier: an embedding layer, a single-layer *bidirectional* LSTM, masked mean pooling over the real tokens, dropout, and a final linear classifier. The second is the Transformer Encoder classifier: an embedding layer with learned positional encodings, two encoder layers with four attention heads each, masked mean pooling, dropout, and a linear classifier.

Crucially, both were trained from scratch. As the boxes show, the LSTM has about 2.83 million trainable parameters and the Transformer about 2.97 million — roughly a five percent difference, which the assignment considers reasonably comparable. Most of those parameters actually live in the shared embedding table, so the encoders themselves are close in size. To keep the comparison clean, both models used identical settings: the Adam optimizer, a learning rate of one times ten-to-the-minus-three, six epochs, dropout of 0.2, batch size 64, and cross-entropy loss, all run on CPU with seed 42.

### Main Results (Slide 7)
Here are the headline numbers. On the held-out test set, the LSTM reached 91.20 percent accuracy and the Transformer Encoder 91.25 percent. The picture is the same for macro F1 — about 0.9118 versus 0.9126. We report macro F1 in addition to accuracy because, even though AG News is balanced, macro F1 weights every class equally and would expose any hidden class-specific weakness.

So the Transformer Encoder does have the best final metrics — but the lead is only about 0.0005 in accuracy. Our honest interpretation is that both models perform almost identically; the Transformer is marginally ahead in this run, but this is not a decisive architectural win.

### Training Behavior (Slide 8)
The loss curves tell a more interesting story than the final numbers alone. Training loss keeps falling steadily across all six epochs for both models, but the *validation* loss stops improving and begins to rise in the later epochs — that's the region circled on the graph. That divergence is a clear sign of overfitting by epoch six. We also noticed that the LSTM reaches a strong validation accuracy slightly faster in the first epoch. The practical takeaway is that early stopping or stronger regularization would be the logical next improvement for both models.

### Confusion Matrices (Slide 9)
Breaking the errors down by class confirms where the difficulty lies. Sports is the easiest class for both models — it has very distinctive vocabulary. The persistent problem is the overlap between Business and Sci/Tech, because so many stories about technology companies contain both market language and technical language. For the LSTM, 177 Business items were predicted as Sci/Tech and 121 the other way; for the Transformer, 200 Business items went to Sci/Tech and 93 came back. Together that single class pair accounts for roughly 44 percent of all test errors for both models, and this pattern lines up exactly with the failure cases Zain will show you next.

**Zain, over to you for the deeper analysis.**

---

## Part 3 — Zain Shabbir *(≈3 minutes)*

### Ablation Studies (Slide 10)
Thanks, Shabih. We ran two ablation studies, each changing one factor at a time. The first, and the required one, varied the *amount of training data* — 25, 50, and 100 percent — to ask which model is more data-efficient. Our hypothesis was that the LSTM would be more stable with little data, while the Transformer would benefit more from the full dataset. Everything else — the split, tokenizer, vocabulary, sequence length, and training budget — was held fixed.

The results support that hypothesis. At 25 and 50 percent of the data, the LSTM is slightly ahead. At 100 percent, the Transformer Encoder moves in front. In other words, the Transformer gained more from the extra data — about 4.5 points versus the LSTM's 3.7. The differences are modest, though, so we describe this as a trend rather than universal proof. The second ablation varied sequence length, 128 versus 256 tokens. Counter to what you might expect, the longer length actually *hurt* both models. AG News items are short enough that 128 tokens already capture the useful signal, and the extra length only adds computation and noise.

### Failure Analysis (Slide 11)
We then examined the misclassified examples in detail, deliberately including cases that both models got wrong, cases only the LSTM missed, and cases only the Transformer missed. Three themes emerge. First, surface cues can override the true section — an Olympics story labelled World was called Sports by both models, and this is also a case where truncation may play a role. Second, Business and Sci/Tech overlap heavily whenever a story involves a tech company, an IPO, or a chip delay — example two, an Intel product-delay story, fooled both models. Third, very short science items give the models too little context, which is where the two models diverge: the Transformer over-weighted business-like wording on the T. rex and IBM items, while the LSTM stumbled on some science-and-world cases. So the models fail similarly on genuinely ambiguous text, but each also has its own distinct error profile.

### Conclusion (Slide 12)
To summarize: both models reach about 91.2 percent test accuracy. The Transformer Encoder is very slightly better on the final metrics; the LSTM is stronger in low-data settings while the Transformer benefits more from the full training set; longer sequences did not help; the main error pattern is Business versus Sci/Tech; and both models overfit by the final epoch. Our recommended next steps are early stopping, stronger regularization, and repeating the study across multiple seeds for more robust conclusions.

I'll also note briefly, in the interest of transparency, that we used AI assistance for drafting code, debugging, and visualization, but the team verified every piece of code, every result, and every claim ourselves — the details are documented in our AI Usage Appendix.

### Thank You (Slide 13)
That concludes our presentation. Thank you all for your attention — on behalf of Abdullah, Shabih, and myself, we'd be glad to take your questions.
