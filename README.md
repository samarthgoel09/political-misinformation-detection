# Political Misinformation Detection

An AI-based system for detecting political misinformation in U.S. news and online content using supervised learning.

## Overview

This project classifies text as credible or misleading using linguistic and structural features learned from labeled datasets. It implements and compares multiple classification approaches:

- **Baseline Models**: Logistic Regression, Multinomial Naive Bayes, Linear SVM (using TF-IDF features)
- **Deep Learning**: Fine-tuned DistilBERT for multi-class text classification

## Datasets

| Dataset | Source | Labels | Setup |
|---------|--------|--------|-------|
| [Fakeddit](https://github.com/entitize/Fakeddit) | Reddit posts (1M+ samples) | 6-way: true, satire, misleading, imposter, false connection, manipulated | Manual download → `data/fakeddit/` |
| [NELA-GT](https://doi.org/10.7910/DVN/CHMUYZ) | News articles with source credibility | 3-way: reliable, mixed, unreliable | Manual download → `data/nela/` |
| [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) | PolitiFact statements (12,800 samples) | 6/3/2-way credibility labels | Auto-downloaded on first run |

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## Usage

### With Sample Data (for testing)
```bash
python main.py --use-sample-data
```

### With LIAR Dataset (auto-downloads)
```bash
python main.py --dataset liar --label-scheme 2way --n-samples 5000
```

### With Real Datasets
1. Download Fakeddit from [GitHub](https://github.com/entitize/Fakeddit) and place TSV files in `data/fakeddit/`
2. Download NELA-GT from [Harvard Dataverse](https://doi.org/10.7910/DVN/CHMUYZ) and place in `data/nela/`
3. Run the pipeline:
```bash
python main.py --dataset fakeddit --label-scheme 6way
python main.py --dataset nela --label-scheme 3way
python main.py --dataset both --label-scheme 2way
```

### Full Pipeline with All Features
```bash
python main.py --dataset liar --label-scheme 2way --n-samples 5000 --use-smote --use-bert --error-analysis --bias-analysis
```

### Hyperparameter Tuning
```bash
python main.py --dataset liar --label-scheme 2way --tune
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--use-sample-data` | off | Use synthetic data for quick testing |
| `--dataset` | `fakeddit` | Dataset: `fakeddit`, `nela`, `liar`, `both` |
| `--label-scheme` | `6way` | Classification granularity: `2way`, `3way`, `6way` |
| `--n-samples` | `2000` | Number of samples to load |
| `--use-smote` | off | Apply SMOTE oversampling for class imbalance |
| `--tune` | off | Run GridSearchCV hyperparameter tuning |
| `--use-bert` | off | Also train DistilBERT model |
| `--bert-epochs` | `3` | Number of BERT training epochs |
| `--bert-batch-size` | `32` | BERT batch size |
| `--bert-max-samples` | `5000` | Max samples for BERT training |
| `--error-analysis` | off | Generate misclassification visualizations |
| `--bias-analysis` | off | Generate bias and fairness analysis |
| `--results-dir` | `results/` | Output directory for plots and metrics |

## Project Structure

```
political-misinfo-detection/
├── main.py                  # End-to-end pipeline runner
├── requirements.txt         # Python dependencies
├── data/
│   ├── sample_data.py       # Synthetic data generator
│   ├── download_fakeddit.py # Fakeddit loader
│   ├── download_nela.py     # NELA-GT loader (CSV, SQLite, JSON)
│   └── download_liar.py     # LIAR loader (auto-download)
├── src/
│   ├── preprocess.py        # Text cleaning & normalization
│   ├── features.py          # TF-IDF extraction + SMOTE
│   ├── models.py            # Baseline classifiers (LR, NB, SVM)
│   ├── bert_model.py        # DistilBERT fine-tuning
│   ├── evaluate.py          # Metrics & visualizations
│   ├── error_analysis.py    # Misclassification & bias analysis
│   └── tuning.py            # Hyperparameter grid search
├── results/                 # Generated plots & metrics (auto-created)
└── notebooks/               # Jupyter exploration (optional)
```

## Evaluation Metrics

- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1-Score (macro & weighted)
- Confusion matrices (counts + normalized)
- Per-class error rates and misclassification heatmaps
- Label distribution and performance disparity analysis

## References

- Nakamura, K., Levy, S., & Wang, W. Y. (2020). r/Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection. *LREC 2020*, 6149–6157.
- Nørregaard, J., Horne, B. D., & Adalı, S. (2019). NELA-GT-2018: A large multi-labelled news dataset for the study of fake news. *ICWSM 2019*.
- Wang, W. Y. (2017). "Liar, liar pants on fire": A new benchmark dataset for fake news detection. *ACL 2017*, 422–426.
