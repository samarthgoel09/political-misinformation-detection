# Political Misinformation Detection

An AI-based system for detecting political misinformation in U.S. news and online content using supervised learning.

## Overview

This project classifies text as credible or misleading using linguistic and structural features learned from labeled datasets. It implements and compares multiple classification approaches:

- **Baseline Models**: Logistic Regression, Multinomial Naive Bayes, Linear SVM (using TF-IDF features)
- **Deep Learning**: Fine-tuned DistilBERT for multi-class text classification
- **Cross-Dataset Evaluation**: Models trained on one dataset and tested on another to measure how well they generalize beyond their training data

## Datasets

| Dataset | Source | Labels | Setup |
|---------|--------|--------|-------|
| [Fakeddit](https://github.com/entitize/Fakeddit) | Reddit posts (1M+ samples) | 6-way: true, satire, misleading, imposter, false connection, manipulated | Manual download → `data/fakeddit/` (see instructions below) |
| [NELA-GT](https://doi.org/10.7910/DVN/CHMUYZ) | News articles with source credibility ratings | 3-way: reliable, mixed, unreliable | Included in `data/nela/` |
| [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) | PolitiFact statements (12,800 samples) | 6/3/2-way credibility labels | Auto-downloaded on first run |

---

## Setup

### Step 1 — Clone the repo and enter the folder

```bash
git clone https://github.com/samarthgoel09/political-misinformation-detection
cd political-misinformation-detection
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download NLTK data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

---

## How to Run

### Quick test with sample data (no downloads needed)

```bash
python main.py --use-sample-data
```

---

### LIAR dataset (auto-downloads, easiest to run)

```bash
python main.py --dataset liar --label-scheme 2way
```

With error analysis:

```bash
python main.py --dataset liar --label-scheme 2way --error-analysis
```

With BERT:

```bash
python main.py --dataset liar --label-scheme 2way --use-bert --bert-epochs 5
```

Full run with everything:

```bash
python main.py --dataset liar --label-scheme 2way --use-smote --use-bert --bert-epochs 5 --error-analysis --bias-analysis
```

---

### Fakeddit dataset (requires manual download)

The Fakeddit TSV files are too large to include in this repository (train.tsv is 218MB). To use them:

1. Go to https://github.com/entitize/Fakeddit
2. Click the download link in the README to access the Google Drive folder
3. Open the `all_samples` folder
4. Download `all_train.tsv`, `all_validate.tsv`, and `all_test_public.tsv`
5. Rename them to `train.tsv`, `validate.tsv`, and `test.tsv`
6. Place them in `data/fakeddit/`
7. Run:

```bash
python main.py --dataset fakeddit --label-scheme 2way --error-analysis
```

---

### NELA-GT dataset

The data file is included in the repository under `data/nela/`. Just run:

```bash
python main.py --dataset nela --label-scheme 3way
```

---

### Cross-Dataset Evaluation (main contribution)

This trains on one dataset and tests on another to measure generalization.
Requires both LIAR (auto-downloads) and Fakeddit TSV files in `data/fakeddit/`.

```bash
# Train on LIAR, test on Fakeddit
python main.py --dataset liar --label-scheme 2way --cross-dataset

# Run in both directions (LIAR->Fakeddit and Fakeddit->LIAR)
python main.py --dataset liar --label-scheme 2way --cross-dataset --bidirectional
```

---

### Hyperparameter Tuning

```bash
python main.py --dataset liar --label-scheme 2way --tune
```

---

## CLI Options

### Data options

| Flag | Default | Description |
|------|---------|-------------|
| `--use-sample-data` | off | Use synthetic data for quick testing |
| `--dataset` | `fakeddit` | Which dataset to use: `fakeddit`, `nela`, `liar`, `both` |
| `--label-scheme` | `6way` | Classification granularity: `2way`, `3way`, `6way` |
| `--n-samples` | `2000` | Number of samples to load |
| `--fakeddit-dir` | `data/fakeddit` | Path to Fakeddit TSV files |
| `--nela-dir` | `data/nela` | Path to NELA-GT data |

### Model options

| Flag | Default | Description |
|------|---------|-------------|
| `--use-smote` | off | Apply SMOTE oversampling to handle class imbalance |
| `--no-class-weight` | off | Disable balanced class weights in LR and SVM |
| `--tune` | off | Run GridSearchCV hyperparameter tuning |
| `--max-features` | `10000` | TF-IDF vocabulary size |

### BERT options

| Flag | Default | Description |
|------|---------|-------------|
| `--use-bert` | off | Also train DistilBERT model |
| `--bert-epochs` | `5` | Number of training epochs |
| `--bert-batch-size` | `32` | Batch size |
| `--bert-max-length` | `128` | Max token length |
| `--bert-max-samples` | `5000` | Max training samples for BERT |
| `--bert-early-stopping` | `2` | Patience in epochs before early stopping |
| `--bert-lr-schedule` | `cosine` | Learning rate schedule: `cosine` or `linear` |
| `--bert-grad-accum` | `1` | Gradient accumulation steps |
| `--bert-no-class-weight` | off | Disable class weighting in BERT loss |

### Analysis options

| Flag | Default | Description |
|------|---------|-------------|
| `--error-analysis` | off | Generate misclassification visualizations |
| `--bias-analysis` | off | Generate bias and fairness analysis |
| `--results-dir` | `results/` | Output directory for all plots and metrics |

### Cross-dataset options

| Flag | Default | Description |
|------|---------|-------------|
| `--cross-dataset` | off | Run cross-dataset generalization evaluation |
| `--source-dataset` | `liar` | Dataset to train on |
| `--target-dataset` | `fakeddit` | Dataset to test on |
| `--bidirectional` | off | Run eval in both directions |

---

## Project Structure

```
political-misinformation-detection/
├── main.py                      # End-to-end pipeline runner — start here
├── requirements.txt             # All Python dependencies
│
├── data/
│   ├── fakeddit/                # Place downloaded Fakeddit TSV files here
│   ├── nela/                    # NELA-GT data files (included in repo)
│   ├── liar/                    # LIAR files (auto-downloaded on first run)
│   ├── sample_data.py           # Generates synthetic data for testing
│   ├── download_fakeddit.py     # Loads Fakeddit TSV files
│   ├── download_nela.py         # Loads NELA-GT (CSV, SQLite, or JSON)
│   └── download_liar.py         # Downloads and loads LIAR automatically
│
├── src/
│   ├── preprocess.py            # Text cleaning: URLs, HTML, stopwords, lemmatization
│   ├── features.py              # TF-IDF feature extraction + optional SMOTE
│   ├── models.py                # Logistic Regression, Naive Bayes, SVM
│   ├── bert_model.py            # DistilBERT fine-tuning with early stopping
│   ├── evaluate.py              # Metrics, confusion matrices, comparison charts
│   ├── error_analysis.py        # Per-class error rates, heatmaps, bias analysis
│   ├── tuning.py                # GridSearchCV hyperparameter tuning
│   └── cross_dataset_eval.py    # Cross-dataset generalization evaluation
│
└── results/                     # Auto-created — all outputs saved here
    ├── metrics_summary.csv      # Model performance table
    ├── model_comparison.png     # Bar chart comparing all models
    ├── confusion_matrix_*.png   # Per-model confusion matrices
    ├── error_analysis/          # Error heatmaps and misclassified samples
    ├── bias_analysis/           # Label distribution and disparity plots
    └── cross_dataset/           # Cross-dataset results and charts
```

---

## Evaluation Metrics

All models are evaluated on:

- **Accuracy** — overall correct predictions
- **Precision** (macro and weighted) — how many predicted positives are correct
- **Recall** (macro and weighted) — how many actual positives were caught
- **F1-Score** (macro and weighted) — harmonic mean of precision and recall
- **Confusion matrices** — counts and normalized, showing exact error patterns
- **Per-class error rates** — which classes the model struggles with most
- **Misclassification heatmaps** — which class pairs get confused most often
- **Text length analysis** — whether text length affects classification accuracy
- **Label distribution analysis** — class imbalance visualization
- **Cross-dataset generalization gap** — drop in F1 when testing on an unseen dataset

All results are saved to the `results/` folder as PNG charts and CSV files.

---

## References

- Nakamura, K., Levy, S., & Wang, W. Y. (2020). r/Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection. *LREC 2020*, 6149–6157.
- Nørregaard, J., Horne, B. D., & Adalı, S. (2019). NELA-GT-2018: A large multi-labelled news dataset for the study of fake news. *ICWSM 2019*.
- Wang, W. Y. (2017). "Liar, liar pants on fire": A new benchmark dataset for fake news detection. *ACL 2017*, 422–426.
