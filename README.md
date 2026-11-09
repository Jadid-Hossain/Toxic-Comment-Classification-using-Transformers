# Assignment 03 — Toxic Comment Classification
**Transfer Learning, Fine-Tuning, Hyperparameter Tuning, and Interpretability**

| | |
|---|---|
| **Name** | MD. JADID HOSSAIN SANIM |
| **Student ID** | 1000129287 |


---

## 1. Dataset and Preprocessing

* **Dataset:** [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments), loaded via Hugging Face `datasets` (`load_dataset("google/civil_comments")`).


* **Toxicity Threshold:** The continuous `toxicity` score is converted into a binary label using a threshold of 0.5:
  * `toxicity >= 0.5` $\rightarrow$ **`1` (Toxic)**
  * `toxicity < 0.5` $\rightarrow$ **`0` (Not Toxic)**


* **Subsampling & Splitting:** To fit within Google Colab T4 GPU compute limits while preserving the natural class distribution (~8% toxic), a stratified random subset of 30,000 comments was used:
  * **Train:** 20,000 rows (7.99% toxic)
  * **Validation:** 5,000 rows (8.00% toxic)
  * **Test:** 5,000 rows (8.00% toxic)


* **Evaluation Metric:** Because ~92% of the comments are non-toxic, accuracy alone is misleading (a majority-class classifier achieves 92.00% accuracy with 0.0000 F1). Therefore,  F1-score (on the toxic class) is used as the primary evaluation metric, supported by Precision, Recall, ROC-AUC, and PR-AUC.

![Data Overview](figures/01_data_overview.png)

---

## 2. Choice of BERT-Based Model

**Model Used:** distilbert-base-uncased


**Why this model was chosen:**
1. **BERT Architecture:** DistilBERT is a distilled version of bert-base-uncased trained via knowledge distillation, retaining ~97% of BERT's language understanding capabilities while sharing the same core transformer encoder design.
2. **Suitability for User Comments:** Online comments frequently contain irregular capitalization. The uncased tokenizer normalizes casing, reducing vocabulary sparsity on noisy text.
3. **Computational Efficiency for Tuning:** With 40% fewer parameters and ~60% faster training than full BERT, DistilBERT makes it feasible to run an extensive 18-trial hyperparameter search (Grid + Random Search) alongside full end-to-end retraining on a single T4 GPU.

---

## 3. Experimental Results: Frozen Baseline, Fine-Tuning, and Hyperparameter Tuning

### Summary of Test Set Performance:

| Pipeline Stage | Accuracy | Precision | Recall | **F1-Score (Primary)** | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| 0. Majority Class (Floor) | 0.9200 | 0.0000 | 0.0000 | **0.0000** | 0.5000 | 0.0800 |
| 1. Frozen DistilBERT + Linear Head (Baseline) | 0.9230 | 0.5524 | 0.1975 | **0.2910** | 0.8524 | 0.3911 |
| 2. End-to-End Fine-Tuned (Default Config) | 0.9428 | 0.6676 | 0.5675 | **0.6135** | 0.9377 | 0.6907 |
| 3. End-to-End Fine-Tuned (Tuned Config) | 0.9358 | 0.6010 | 0.5875 | **0.5942** | 0.8898 | 0.6003 |

---

### Stage 1 & 2: Frozen Feature Extractor vs. End-to-End Fine-Tuning

* **Stage 1 (Frozen Baseline):** All 66.4M DistilBERT backbone parameters were frozen, and only a linear classification head (Dropout + Linear, around 1.5K parameters) was trained on cached 768-dimensional CLS embeddings.
* **Stage 2 (Default Fine-Tuning):** All backbone layers were unfrozen and trained end-to-end using standard default hyperparameters (learning_rate = 2e-5, batch_size = 32, epochs = 3, warmup_ratio = 0.06, weight_decay = 0.01, head_dropout = 0.1).

![Frozen Head Training](figures/02_frozen_head.png)
![Fine-Tuning Default Curves](figures/03_finetune_default.png)

---

### Stage 3: Hyperparameter Tuning & Screenshot of Results

To optimize the fine-tuned model, **18 trials** were conducted on an 8,000-row stratified training subset (2 epochs per trial) and evaluated on the 5,000-row validation set:
* **Grid Search (6 trials):** Searched combinations of learning_rate $ \{1\text{e-}5, 2\text{e-}5, 5\text{e-}5\}$ and batch_size $\ \{16, 32\}$. Best Validation F1: **0.5916**.


* **Random Search (12 trials):** Sampled across a broader space: log-uniform learning_rate $\ [5\text{e-}6, 1\text{e-}4]$, batch_size $\ \{16, 32, 64\}$, warmup_ratio $\ \{0.0, 0.06, 0.15\}$, weight_decay $\ \{0.0, 0.01, 0.1\}$, and head_dropout $\ \{0.1, 0.3, 0.5\}$. Best Validation F1: **0.5958**.

**Winning Hyperparameter Configuration:**
* `learning_rate`: `9.02e-5`
* `batch_size`: `32`
* `warmup_ratio`: `0.06`
* `weight_decay`: `0.0`
* `head_dropout`: `0.3`

#### Screenshot of Tuning Results
![Tuning Results](figures/04_tuning_results.png)

---

### Justification of Results (Without Tuning vs. With Tuning)

1. **Frozen Baseline vs. Untuned Fine-Tuning (F1: 0.2910 $\rightarrow$ 0.6135):**
   * Unfreezing the backbone more than doubled the F1-score (**+0.3225**) andnearly tripled Recall (0.1975 $\rightarrow$ 0.5675).
   * **Why:** Pre-trained DistilBERT embeddings are optimized for general masked-language modeling, not sentiment or hostility detection. A linear classifier on frozen CLS vectors cannot separate subtle toxic cues from neutral text. End-to-end fine-tuning allows the self-attention layers to re-weight contextual representations specifically around insults, profanity, and aggressive phrasing.

2. **Untuned Fine-Tuning vs. Tuned Fine-Tuning (F1: 0.6135 vs. 0.5942):**
   * The tuned configuration achieved the highest **Recall (0.5875 vs. 0.5675)**, meaning it successfully caught more toxic comments on the test set than the default model, though a slight drop in Precision (0.6010 vs. 0.6676) resulted in a marginally lower overall Test F1 (0.5942).

   * **Why:** During the 2-epoch, 8,000-sample search phase, a higher learning rate (9.02e-5) converged faster and achieved the top validation F1 (0.5958). However, when retrained over the full 20,000-sample dataset, this aggressive learning rate with zero weight decay caused slight overfitting compared to the conservative textbook learning rate (2e-5 with 0.01 weight decay). This demonstrates an important practical insight: hyperparameter configurations optimized on smaller, short-horizon proxy subsets favor higher learning rates that can slightly overfit when scaled to full training runs.

![Stage Comparison](figures/06_stage_comparison.png)
![Confusion Matrices](figures/07_confusion_matrices.png)
![ROC and PR Curves](figures/08_roc_pr_curves.png)

---

## 4. Test on Real-World Text

Ten custom real-world comments (outside the Civil Comments dataset) were passed through the final tuned model using the 0.5 decision threshold:

| # | Input Comment | $P(\text{toxic})$ | Predicted Class | Short Check & Analysis |
|---|---|---|---|---|
| 1 | *"This article really helped me understand the topic, thank you for writing it!"* | `0.0001` | **NOT TOXIC** | **Correct.** Polite, appreciative comment. |
| 2 | *"I disagree with the author, but the argument is well made and clearly explained."* | `0.0000` | **NOT TOXIC** | **Correct.** Constructive, civil disagreement. |
| 3 | *"This is the stupidest thing I have read all week."* | `0.9997` | **TOXIC** | **Correct.** Direct insult ("stupidest") strongly triggers the classifier. |
| 4 | *"You clearly have no idea what you are talking about."* | `0.0000` | **NOT TOXIC** | **False Negative (Borderline).** Condescending tone, but lacks explicit profanity; model treats it as non-toxic debate. |
| 5 | *"What a complete waste of time this video was."* | `0.0000` | **NOT TOXIC** | **Correct.** Harsh criticism of content, but not abusive or toxic. |
| 6 | *"Great work as always, looking forward to the next part!"* | `0.0000` | **NOT TOXIC** | **Correct.** Friendly, encouraging feedback. |
| 7 | *"The author should be ashamed of publishing this nonsense."* | `0.0001` | **NOT TOXIC** | **False Negative (Implicit Hostility).** Hostile attack using formal vocabulary ("ashamed", "nonsense") without explicit slurs. |
| 8 | *"Can someone explain step three? I did not quite follow it."* | `0.0000` | **NOT TOXIC** | **Correct.** Neutral question asking for help. |
| 9 | *"Honestly, this changed my mind about the whole topic. Well done."* | `0.0000` | **NOT TOXIC** | **Correct.** Positive reflection. |
| 10 | *"Stop spreading lies, you are misleading everyone who reads this."* | `0.0002` | **NOT TOXIC** | **False Negative (Borderline).** Accusatory tone ("lies", "misleading"), but missed because it contains no explicit profanity or slurs. |

**Takeaway from Real-World Testing:** The model is highly accurate on civil comments, constructive criticism, and overt insults (e.g., *"stupidest"* $\rightarrow P = 0.9997$). However, because it was trained on a 30,000-row imbalanced subset (where only ~8% of examples are toxic), it relies heavily on explicit lexical toxicity (profanity/slurs) and struggles with subtle condescension or polite hostility (#4, #7, #10).

---

## 5. Interpretability Analysis

### Explainability Methods Used and Why

1. **SHAP (Primary — Partition Explainer):**
   * **Why:** Computes local, additive token attributions grounded in cooperative game theory. Because it perturbs input text through hierarchical masking directly on the fine-tuned model, it fairly distributes credit among multi-word phrases and insults.
2. **LIME (Cross-Check):**
   * **Why:** Fits an interpretable local linear surrogate model on randomly perturbed text samples. Using LIME alongside SHAP verifies that the highlighted toxic tokens are genuine model features rather than artifacts of a single explanation algorithm.
3. **Transformer Attention Weights (Internal View):**
   * **Why:** Visualizes how strongly the CLS classification token attends to individual input words in the final DistilBERT layer (averaged across attention heads), providing a direct look into the model's internal feature aggregation.

---

### Analysis of Selected Test Set Predictions

Four representative test-set cases were analyzed across the interpretability methods:

| Case Type | True Label | Predicted $P(\text{toxic})$ | Key Findings from Explainability Analysis |
|---|---|---|---|
| **1. Confident Toxic** | Toxic (1) | 0.9999 | SHAP assigns large positive weights to explicit slurs and insults ("imbecile?", "racist", "Pocahontas?"), while surrounding words remain near zero. LIME and final-layer CLS attention focus on the exact same hostility tokens. |
| **2. Confident Non-Toxic** | Not Toxic (0) | 0.0000 | All token attributions are small and neutral/negative. No single word pushes the log-odds toward the toxic class. |
| **3. Borderline Prediction** | Not Toxic (0) | 0.4895 | Competing signals cancel out: a sharp/confrontational word pushes the score positively toward toxicity, while surrounding civil context pulls it back down, landing right near the 0.5 decision boundary. |
| **4. False Negative (Error)** | Toxic (1) | 0.0000 | SHAP and LIME show near-zero positive token contributions. The comment's toxicity is conveyed through sarcasm and implicit context rather than overt slur words, explaining why the token-driven classifier missed it. |

![SHAP Explanations](figures/10_shap_toxic_confident.png)
![Attention Weights](figures/11_attention_weights.png)
![LIME Cross-Check](figures/12_lime_crosscheck.png)

---
