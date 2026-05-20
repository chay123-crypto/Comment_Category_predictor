# Comment Category Classification based on Toxicity Levels

A Multi-class Text Classification NLP pipeline that categorizes toxic comments into 4 classes using a stacking ensemble of classical ML models. Built for a Kaggle competition on an imbalanced dataset of ~190k comments in the training sample.

---

## Problem

Given a comment along with metadata (votes, emoticons, timestamps, demographic signals), predict one of 4 toxicity categories. The dataset is heavily imbalanced — class 3 accounts for only ~2.7% of samples — making macro F1 the critical metric.

**Dataset:** ~190k rows | 4-class imbalanced | Kaggle competition

---

## Approach

### Feature Engineering (20+ features)

Raw text, vote signals, timestamps, and emoticons were transformed into a rich feature matrix:

| Feature Group | Examples |
|---|---|
| Text (TF-IDF) | Word unigrams/bigrams + char `wb` subword n-grams |
| Temporal | Year, month, day, hour + cyclical sin/cos encodings |
| Vote-based | `total_votes`, `net_votes`, `engagement_ratio`, `controversial_score` |
| Text style | `caps_word_count`, `avg_word_length`, `unique_word_ratio`, `elongated_count`, `emoji_count` |
| Emoticon | `emoticon_total`, `emoticon_presence`, `dominant` emoticon type |

### Models Trained

| Model | Role | Notes |
|---|---|---|
| SGD (log loss) | Base learner | Elastic net regularization, adaptive LR, class weights |
| LightGBM | Base learner | Leaf-wise growth, 500 estimators, GPU-accelerated |
| ComplementNB | Base learner | Text-only (TF-IDF columns), suited for sparse features |
| XGBoost | Reference | Histogram method, sample weights (commented out in final stack) |
| MLP (2×256) | Meta-learner | Trained on stacked OOF probabilities from all base models |

### Stacking

Probability outputs from SGD, LightGBM, and CNB are horizontally stacked as meta-features. The MLP meta-learner is trained on these 12-dimensional vectors (3 models × 4 classes) and learns optimal combination weights.

```
meta_features = [SGD_proba | CNB_proba | LGBM_proba]  →  MLP  →  final prediction
```

### Threshold Tuning

Default 0.5 threshold performs poorly under class imbalance. For each model, per-class thresholds are tuned by maximizing F1 on the precision-recall curve:

```python
precision, recall, thresh = precision_recall_curve(y_true_class, y_scores_class)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_thresh = thresh[np.argmax(f1_scores)]
```

---

## Results (Validation Set)

| Model | Macro F1 | 
|---|---|
| **Stacking Ensemble** | **0.835** | 
| SGD | 0.828|
| LightGBM | 0.8202 | 
| XGBoost | 0.818 |
| ComplementNB | 0.614 |

The stacking ensemble outperforms all individual models by learning to combine their complementary strengths.

---

## Stack

- **Python** — pandas, NumPy, scikit-learn, LightGBM, XGBoost
- **Feature extraction** — `TfidfVectorizer` (word + char n-grams), `PowerTransformer`, `OrdinalEncoder`
- **Sparse matrix ops** — `scipy.sparse.hstack`
- **Environment** — Kaggle (GPU-enabled)

---

## Repository Structure

```
├── comment_category_predictor.ipynb   # Full pipeline: EDA → features → models → stacking
└── README.md
```

> **Note:** Notebook is pushed with outputs cleared. Markdown cells contain inline analysis notes and result tables for each step.

---

## Key Takeaways

- Cyclical time encoding (`sin`/`cos`) outperforms raw hour/weekday integers for temporal features
- ComplementNB works best when restricted to TF-IDF columns only — mixing numeric features hurts it
- Per-class threshold tuning on the PR curve gives meaningful F1 gains over default 0.5, especially for the minority class (class 3)
- XGBoost did not improve the final stack and was excluded — gradient boosted trees overlap too much with LightGBM to add diversity
