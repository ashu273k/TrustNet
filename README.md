# 🛡️ TrustNet: Twitter Bot Detection (Cresci 2017 Subset)

Lightweight, explainable experimentation on social bot detection using a curated slice of the Cresci 2017 datasets.

![Status: Active](https://img.shields.io/badge/status-active-success?style=flat-square)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)
![License: MIT](https://img.shields.io/badge/license-MIT-purple?style=flat-square)
![Domain: Social Bots](https://img.shields.io/badge/Domain-Social%20Bots-orange?style=flat-square)

**TrustNet** is a lightweight experimentation space for detecting automated (bot / spam) accounts on Twitter/X using a carefully curated *small subset* of the well‑known **Cresci 2017** datasets. This repository focuses on approach clarity, feature engineering transparency, and reproducible evaluation.

---

## ✨ Motivation
Modern social platforms face coordinated inauthentic behavior: spambots, follow churners, fake amplifiers. Research datasets (like Cresci 2017) are large; for teaching, prototyping, or rapid iteration a smaller, well-structured sample accelerates idea > model cycles. TrustNet aims to:

1. Provide a concise, explainable baseline pipeline.
2. Showcase modular feature extraction (profile, content, network, temporal).
3. Encourage experimentation with classical ML before jumping to deep architectures.

---

## 📂 Dataset Overview
We extract a reduced slice from the original Cresci 2017 Twitter bot collections:

| File | Purpose | Rows | Label Distribution* |
|------|---------|------|---------------------|
| `Datasets/genuine_users.csv` | Human / legitimate accounts | (small subset) | label = genuine |
| `Datasets/spam_user.csv` | Spam / bot accounts (selected) | (small subset) | label = spam |

*Exact counts intentionally minimized for lightweight experimentation; expand by substituting full Cresci 2017 data.*

### 🔐 Licensing & Ethics
The original dataset belongs to the authors of the Cresci 2017 paper. Use responsibly; comply with Twitter's TOS and data redistribution norms. This repository only contains *derived / subset* CSVs for educational purposes.

### 📝 Suggested Citation (Original Work)
If you build upon this, cite the Cresci paper:

> Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017). The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race. *Proceedings of the 26th International Conference on World Wide Web Companion*.

---

## 🧱 Data Schema (Expected Columns)
Although the tiny subset may include fewer, typical Cresci-style fields you can expect / extend:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| `user_id` | Unique user identifier | string/int | 1234567890 |
| `screen_name` | Handle | string | someuser |
| `name_len` | Length of display name | int | 12 |
| `description_len` | Bio length (chars) | int | 87 |
| `followers_count` | Followers | int | 240 |
| `friends_count` | Following | int | 310 |
| `statuses_count` | Total tweets | int | 1543 |
| `favourites_count` | Likes given | int | 420 |
| `listed_count` | Public lists membership | int | 3 |
| `default_profile_image` | Avatar default? | bool/int | 0 |
| `created_at` | Account creation time | datetime | 2016-02-11 |
| `avg_tweets_per_day` | Temporal normalized activity | float | 5.3 |
| `spam_ratio` | Heuristic: spam-like tokens / total | float | 0.18 |
| `has_url` | Profile URL flag | int | 1 |
| `label` | Target class | categorical | genuine / spam |

You can generate additional engineered features (see next section) from raw tweet timelines if you integrate more data later.

---

## 🛠️ Feature Engineering Modules

| Category | Examples | Rationale |
|----------|----------|-----------|
| Profile Metadata | age days, profile image flag, bio length, name entropy | Bots often reuse templates, have lower entropy |
| Activity / Temporal | tweets per day, burstiness, inter-tweet std | Automation yields uniform or extreme bursts |
| Network | followers/friends ratio, reciprocal rate | Spam accounts follow aggressively to gain traction |
| Content (Optional) | URL proportion, hashtag density, lexicon similarity | Promotional / malicious payload density |
| Linguistic (Optional) | average token length, emoji rate | Synthetic text differs in distribution |

> Keep features explainable; avoid leaking future knowledge (no post-classification timeline stats).

---

## 🧪 Modeling Approach

Baseline recommendation:

1. Clean & impute: handle nulls (median for numeric, mode for binary).
2. Scale numeric features (StandardScaler or RobustScaler).
3. Train several classical classifiers: Logistic Regression, Random Forest, Gradient Boosting, XGBoost / LightGBM (optional), SVM (RBF for non-linear patterns).
4. Perform stratified 5-fold cross-validation (avoid accuracy obsession; report precision/recall/F1, macro + per-class; add ROC-AUC).
5. Calibrate probabilities if deployment requires risk ranking (e.g., `CalibratedClassifierCV`).

### 🧪 Example Metric Set

| Metric | Why |
|--------|-----|
| Recall (spam) | Catch more malicious accounts |
| Precision (spam) | Limit false accusations |
| F1 (macro) | Balanced view across skew |
| ROC-AUC | Ranking quality |
| PR-AUC (spam) | Robust under class imbalance |

### 🔄 Iteration Loop

Extract features → Evaluate → Inspect misclassifications → Add/Refine features → Re-train.

---

## ⚡ Quickstart

### 1. Clone

```bash
git clone https://github.com/ashu273k/TrustNet.git
cd "TrustNet dataset"
```

### 2. (Optional) Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows (WSL) / Linux
```

### 3. Install Core Dependencies (example)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Load & Inspect

```python
import pandas as pd
genuine = pd.read_csv('Datasets/genuine_users.csv')
spam = pd.read_csv('Datasets/spam_user.csv')
df = pd.concat([genuine.assign(label='genuine'), spam.assign(label='spam')])
print(df.head())
```

### 5. Train a Simple Baseline

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

feature_cols = [c for c in df.columns if c not in ['label','user_id','screen_name','created_at']]
X = df[feature_cols]
y = df['label']

pipe = Pipeline([
	('scaler', StandardScaler()),
	('clf', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
pipe.fit(X_train, y_train)
from sklearn.metrics import f1_score
print('Test F1 (spam class):', f1_score(y_test, pipe.predict(X_test), pos_label='spam'))
```

---

## 📊 Visualization Ideas

| Plot | Insight |
|------|---------|
| Followers vs Friends scatter | Aggressive following patterns |
| Distribution of account age | Newborn clusters of bots |
| Boxplot of tweets/day by class | Activity intensity |
| ROC curve ensemble | Comparative classifier tradeoffs |

---

## 🚀 Roadmap

- [ ] Add notebook with full feature pipeline
- [ ] Integrate tweet-level content features
- [ ] Provide benchmarking script (multi-model comparison)
- [ ] Add model persistence + inference script
- [ ] Experiment with anomaly detection (e.g., Isolation Forest)

---

## 🤝 Contributing

Contributions welcome! Feel free to:

1. Open an issue (bug / idea / enhancement)
2. Fork & branch (`feat/<short-description>`)
3. Submit PR with concise description & before/after metrics

Keep code modular, add comments for any non-obvious feature transformations, and prefer deterministic seeds where possible.

---

## 🧾 License

Released under the MIT License. See `LICENSE` (add one if missing) for details. Original dataset governed by its own terms—respect them.

---

## 🙏 Acknowledgements

Cresci 2017 authors for releasing foundational bot datasets. The open-source ML community for tooling. You for exploring ethical automation detection.

---

## 💡 Tips for Extension

| Idea | Description |
|------|------------|
| Semi-supervised refinement | Use confident predictions to pseudo-label unlabeled accounts |
| Graph features | Build small ego networks and compute clustering coeff, assortativity |
| Temporal signatures | FFT / spectral density of posting intervals |
| Text embeddings | Add sentence-transformer vectors; beware leakage / overfitting |
| Model stacking | Blend linear + tree + anomaly detectors |

---

---

Made with curiosity and caution — automate detection, not judgment.

