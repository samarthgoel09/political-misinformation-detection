# Progress Report: Political Misinformation Detection

**Course:** CS 5100 — Foundations of Artificial Intelligence  
**Student:** Samarth Goel  
**Date:** March 14, 2026

---

## What Has Been Achieved

We have built a complete supervised learning pipeline for classifying political misinformation using the Fakeddit dataset. The system includes text preprocessing (URL/HTML removal, stopword removal, lemmatization via NLTK), TF-IDF feature extraction with bigram support, and stratified train/validation/test splitting. A synthetic data generator was also built for rapid development testing.

Four classifiers were trained and evaluated on 3,492 politically-filtered Fakeddit samples using the binary label scheme:

| Model | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
|-------|----------|------------|-------------------|----------------|
| Logistic Regression | 70.1% | 0.604 | 0.703 | 0.610 |
| Multinomial Naive Bayes | 70.9% | 0.609 | 0.729 | 0.616 |
| Linear SVM | 71.1% | 0.637 | 0.701 | 0.634 |
| **DistilBERT** | **76.7%** | **0.750** | **0.747** | **0.753** |

DistilBERT outperformed all TF-IDF baselines by approximately 6 percentage points in accuracy and 12 points in macro F1-score. Data loading scripts for NELA-GT have also been written but not yet run on the full dataset.

## Immediate Next Steps

- Expand evaluation to the 6-way label scheme (our primary classification target) and address class imbalance through oversampling and class weighting
- Acquire and process the NELA-GT dataset for source-credibility-based classification
- Tune hyperparameters across all models, particularly DistilBERT with GPU-accelerated training

## Challenges and Adjustments

**Class imbalance** is the most significant challenge. Initial 6-way experiments showed 0% recall on minority classes (e.g., imposter and misleading content), so we began with binary classification to validate the pipeline. Satire detection remains difficult even in the 3-way scheme (3–47% recall), consistent with the Fakeddit paper's findings. Our scope remains text-only as planned.

## Plan for the Next Month

**Weeks 1–2:** Scale to 6-way classification with class balancing techniques; integrate NELA-GT.  
**Week 3:** GPU-accelerated BERT training on larger subsets; error analysis on misclassified samples.  
**Week 4:** Bias and ethical implications analysis; final visualizations and report.

**Code:** [github.com/samarthgoel09/political-misinformation-detection](https://github.com/samarthgoel09/political-misinformation-detection)
