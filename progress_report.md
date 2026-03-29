# Progress Report: Political Misinformation Detection

**Course:** CS 5100 — Foundations of Artificial Intelligence
**Student:** Samarth Goel
**Date:** March 29, 2026

---

## What Has Been Achieved

We have built a complete supervised learning pipeline for classifying political misinformation. The system supports multiple datasets (Fakeddit, NELA-GT, LIAR, and synthetic), text preprocessing, TF-IDF feature extraction, three baseline classifiers, DistilBERT fine-tuning, hyperparameter tuning, error analysis, and bias analysis.

---

## Phase 1: Initial Baseline (Binary Classification — Fakeddit)

Four classifiers were trained and evaluated on 3,492 politically-filtered Fakeddit samples using the binary label scheme:

| Model | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
|-------|----------|------------|-------------------|----------------|
| Logistic Regression | 70.1% | 0.604 | 0.703 | 0.610 |
| Multinomial Naive Bayes | 70.9% | 0.609 | 0.729 | 0.616 |
| Linear SVM | 71.1% | 0.637 | 0.701 | 0.634 |
| **DistilBERT** | **76.7%** | **0.750** | **0.747** | **0.753** |

DistilBERT outperformed all TF-IDF baselines by ~6 percentage points in accuracy and ~12 points in macro F1.

---

## Phase 2: Extended Evaluation (Multi-Dataset, All Label Schemes)

### Binary Classification Results (LIAR + NELA combined)

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Train Time |
|-------|----------|------------|---------------|------------|
| Logistic Regression | 57.0% | 0.567 | 0.572 | 0.02s |
| Multinomial Naive Bayes | 56.5% | 0.562 | 0.567 | 0.003s |
| Linear SVM | 59.25% | 0.455 | 0.495 | 0.02s |
| DistilBERT | 58.0% | 0.569 | 0.579 | 643s |

Note: SVM achieves the highest accuracy but lowest macro F1 (0.455), indicating it is predicting majority classes while ignoring minority ones. Logistic Regression achieves the best balance between accuracy and macro F1.

### 6-Way Classification Results (LIAR)

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| Logistic Regression | 22.0% | 0.214 | 0.220 |
| Multinomial Naive Bayes | 22.75% | 0.210 | 0.223 |
| Linear SVM | 21.25% | 0.128 | 0.147 |

6-way classification on LIAR performs near random chance (16.7%), which is expected: the LIAR labels (`true`, `mostly-true`, `half-true`, `barely-true`, `false`, `pants-fire`) form a nuanced credibility spectrum rather than discrete categories, making them extremely difficult to separate with bag-of-words features.

---

## Phase 3: Class Imbalance Handling

To address class imbalance (which caused 0% recall on minority classes in initial 6-way experiments), the following techniques were implemented:

- **Balanced class weights** for Logistic Regression and Linear SVM (`class_weight='balanced'`)
- **Balanced sample weights** for Multinomial Naive Bayes (via `compute_sample_weight`)
- **SMOTE oversampling** (optional, via `--use-smote` flag) using `imbalanced-learn`
- **Class-weighted CrossEntropyLoss** for DistilBERT, computed from training label distribution

---

## Phase 4: Error Analysis and Bias Analysis

A comprehensive error analysis module (`src/error_analysis.py`) was built:

- **Per-class error rates** and top confusion pairs
- **Error heatmaps** showing misclassification patterns across all class pairs
- **Text length distribution** comparing correct vs. misclassified samples
- **CSV export** of misclassified examples for manual inspection

A bias and ethical analysis module was also implemented:

- **Label distribution analysis** with imbalance ratio computation
- **Per-class performance disparities** (precision, recall, F1 per class)
- **Text length bias** examining length patterns across classes

Key finding from error analysis: on LIAR 6-way, models confuse adjacent labels most (e.g., `barely-true` predicted as `false`, `mostly-true` predicted as `half-true`), which is semantically reasonable but still incorrect.

---

## Phase 5: Hyperparameter Tuning

Grid search with cross-validation (`src/tuning.py`) was added for all baseline models:

| Model | Parameter | Search Space |
|-------|-----------|-------------|
| Logistic Regression | C | {0.01, 0.1, 1.0, 10.0} |
| Linear SVM | C | {0.01, 0.1, 1.0, 10.0} |
| Multinomial Naive Bayes | alpha | {0.01, 0.1, 0.5, 1.0, 2.0} |

Optimization metric: macro F1-score (appropriate for imbalanced classes).

---

## Phase 6: Dataset Expansion

Three real-world datasets are now supported:

| Dataset | Samples | Labels | Status |
|---------|---------|--------|--------|
| Fakeddit | 1M+ (filtered to ~3.5K political) | 2/3/6-way | Requires manual download |
| NELA-GT | Varies (10GB CSV available) | 3-way (reliable/mixed/unreliable) | Loader implemented; source name mismatch identified (see below) |
| LIAR | 12,800 PolitiFact statements | 2/3/6-way | Auto-downloads on first run |

### NELA-GT Issue

The available `nela_ps_newsdata.csv` file contains local/regional U.S. news sources (e.g., `auburntimes`, `baldwintimes`) that do not appear in the built-in source credibility mapping, which covers major national outlets. This results in all samples being assigned to one class, making classification impossible. Resolution options:

1. Obtain the official NELA-GT `labels.csv` file and place it in `data/nela/` — the loader will use it automatically
2. Expand the `SOURCE_CREDIBILITY` mapping in `data/download_nela.py` to cover regional sources

---

## Pipeline Architecture

```
main.py (CLI entry point)
  |
  +-- load_data()           -> Fakeddit / NELA-GT / LIAR / Synthetic
  +-- run_baseline_pipeline()
  |     +-- preprocess_dataframe()
  |     +-- extract_tfidf_features()
  |     +-- [optional] apply_smote()
  |     +-- train_all_models() OR tune_all_models()
  |     +-- evaluate_all_models()
  |     +-- [optional] run_error_analysis()
  |     +-- [optional] run_bias_analysis()
  |
  +-- run_bert_pipeline()
        +-- Class-weighted DistilBERT training
        +-- Evaluate + confusion matrix
        +-- [optional] error + bias analysis
```

---

## CLI Usage Examples

```bash
# Quick test with sample data
python main.py --use-sample-data

# LIAR 2-way (recommended for best accuracy)
python main.py --dataset liar --label-scheme 2way --n-samples 5000 --use-smote --use-bert --error-analysis --bias-analysis

# Hyperparameter tuning on LIAR
python main.py --dataset liar --label-scheme 2way --tune

# Fakeddit with BERT and class balancing
python main.py --dataset fakeddit --use-bert --label-scheme 6way
```

---

## Challenges and Adjustments

**Class imbalance** remains the most significant challenge. Initial 6-way experiments showed 0% recall on minority classes, which led to implementing class weights, SMOTE, and weighted loss functions. 6-way classification on LIAR still performs near random chance due to the label spectrum nature of the task — 2-way classification is significantly more tractable (~57–77% accuracy depending on dataset).

**NELA-GT source mapping** was a practical challenge: the available CSV uses regional source names not covered by the built-in credibility mapping, requiring either the official `labels.csv` or a manual mapping expansion.

**BERT compute cost vs. gain**: DistilBERT takes ~643 seconds to train and achieves 58% accuracy on 2-way LIAR, comparable to Logistic Regression (57%) trained in 0.02 seconds. More epochs and samples are needed to justify the compute cost.

---

## Remaining Work

- Resolve NELA-GT source mapping issue and run evaluation on NELA data
- Run LIAR 2-way with more epochs (5+) and more BERT samples to improve BERT results
- Cross-dataset evaluation (train on LIAR, test on Fakeddit)
- Final report writeup and submission

**Code:** [github.com/samarthgoel09/political-misinformation-detection](https://github.com/samarthgoel09/political-misinformation-detection)
