# Progress Report: Political Misinformation Detection

**Course:** CS 5100 — Foundations of Artificial Intelligence
**Student:** Samarth Goel
**Date:** March 29, 2026

---

## What Has Been Achieved

We have built a complete supervised learning pipeline for classifying political misinformation using the Fakeddit dataset. The system includes text preprocessing (URL/HTML removal, stopword removal, lemmatization via NLTK), TF-IDF feature extraction with bigram support, and stratified train/validation/test splitting. A synthetic data generator was also built for rapid development testing.

### Phase 1: Initial Baseline (Binary Classification)

Four classifiers were trained and evaluated on 3,492 politically-filtered Fakeddit samples using the binary label scheme:

| Model | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
|-------|----------|------------|-------------------|----------------|
| Logistic Regression | 70.1% | 0.604 | 0.703 | 0.610 |
| Multinomial Naive Bayes | 70.9% | 0.609 | 0.729 | 0.616 |
| Linear SVM | 71.1% | 0.637 | 0.701 | 0.634 |
| **DistilBERT** | **76.7%** | **0.750** | **0.747** | **0.753** |

DistilBERT outperformed all TF-IDF baselines by approximately 6 percentage points in accuracy and 12 points in macro F1-score.

### Phase 2: Class Imbalance Handling and Extended Features

To address the critical class imbalance problem (which caused 0% recall on minority classes in 6-way classification), the following techniques were implemented:

- **Balanced class weights** for Logistic Regression and Linear SVM (`class_weight='balanced'`)
- **Balanced sample weights** for Multinomial Naive Bayes (via `compute_sample_weight`)
- **SMOTE oversampling** (optional, via `--use-smote` flag) using `imbalanced-learn`
- **Class-weighted CrossEntropyLoss** for DistilBERT, computed from training label distribution

### Phase 3: Error Analysis and Bias Analysis

A comprehensive error analysis module was built to identify patterns in misclassifications:

- **Per-class error rates** and top confusion pairs
- **Error heatmaps** showing misclassification patterns
- **Text length distribution** comparing correct vs. misclassified samples
- **CSV export** of misclassified samples for manual inspection

A bias and ethical analysis module was also implemented:

- **Label distribution analysis** with imbalance ratio computation
- **Per-class performance disparities** (precision, recall, F1 per class)
- **Text bias analysis** examining length patterns across classes

### Phase 4: Hyperparameter Tuning

Grid search with cross-validation was added for all baseline models:

- Logistic Regression: C in {0.01, 0.1, 1.0, 10.0}
- Linear SVM: C in {0.01, 0.1, 1.0, 10.0}
- Multinomial Naive Bayes: alpha in {0.01, 0.1, 0.5, 1.0, 2.0}

Optimization metric: macro F1-score (appropriate for imbalanced classes).

### Phase 5: Additional Dataset Support

- **LIAR dataset loader** added with automatic download from UCSB
- Supports 2-way, 3-way, and 6-way label schemes
- PolitiFact political statements with truth labels (12.8K samples)

## Pipeline Architecture

```
main.py (CLI entry point)
  |
  +-- load_data()          -> Fakeddit / NELA-GT / LIAR / Synthetic
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
        +-- evaluate + confusion matrix
        +-- [optional] error + bias analysis
```

## CLI Usage Examples

```bash
# Quick test with sample data
python main.py --use-sample-data

# Full pipeline with all features
python main.py --use-sample-data --use-bert --use-smote --error-analysis --bias-analysis

# Hyperparameter tuning
python main.py --use-sample-data --tune

# LIAR dataset with 6-way classification
python main.py --dataset liar --label-scheme 6way --error-analysis

# Fakeddit with BERT and class balancing
python main.py --dataset fakeddit --use-bert --label-scheme 6way
```

## Challenges and Adjustments

**Class imbalance** was the most significant challenge. Initial 6-way experiments showed 0% recall on minority classes (e.g., imposter and misleading content), which led to implementing multiple balancing strategies: class weights, SMOTE, and weighted loss functions. Satire detection remains difficult even in the 3-way scheme (3-47% recall), consistent with the Fakeddit paper's findings.

Our scope remains text-only as planned, which limits performance compared to multimodal approaches but keeps the system focused and interpretable.

## Remaining Work

- Run full-scale evaluation on complete Fakeddit dataset (1M+ samples) with GPU acceleration
- Integrate NELA-GT dataset for source-credibility-based classification
- Final visualizations, analysis, and report writeup

**Code:** [github.com/samarthgoel09/political-misinformation-detection](https://github.com/samarthgoel09/political-misinformation-detection)
