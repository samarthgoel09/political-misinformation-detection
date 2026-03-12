# Progress Report: Political Misinformation Detection

**Course:** [Your Course Name]
**Student:** Samarth Goel
**Date:** March 12, 2026

---

## What Has Been Achieved

We have built and tested a complete supervised learning pipeline for detecting political misinformation in text-based content. The system has been evaluated on real data from the Fakeddit dataset.

1. **Data Infrastructure**: Built data loading scripts for both the Fakeddit and NELA-GT datasets, including political content filtering (by subreddit and keyword matching). A synthetic data generator was also created for development testing, mirroring Fakeddit's 6-way label structure.

2. **Text Preprocessing Pipeline**: Implemented NLP preprocessing including text cleaning (URL/HTML removal, special character stripping), stopword removal, and lemmatization using NLTK.

3. **Feature Extraction**: Built a TF-IDF vectorization module with configurable vocabulary size and bigram support, with stratified train/validation/test splitting.

4. **Model Training and Evaluation**: Trained and compared four classifiers on 3,492 politically-filtered Fakeddit samples (binary scheme):

| Model | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
|-------|----------|------------|-------------------|----------------|
| Logistic Regression | 70.1% | 0.604 | 0.703 | 0.610 |
| Multinomial Naive Bayes | 70.9% | 0.609 | 0.729 | 0.616 |
| Linear SVM | 71.1% | 0.637 | 0.701 | 0.634 |
| **DistilBERT** | **76.7%** | **0.750** | **0.747** | **0.753** |

DistilBERT outperformed all baselines by ~6% accuracy and ~12 points in F1-score, with notably more balanced precision and recall on the "fake" class.

## Immediate Next Steps

- Experiment with GPU-accelerated BERT training for faster iteration and larger sample sizes
- Acquire and integrate the NELA-GT dataset for source-credibility-based classification
- Address class imbalance (the 6-way scheme showed models struggling on minority classes like satire and imposter content, consistent with findings from the Fakeddit literature)
- Tune hyperparameters to improve baseline model performance

## Challenges and Adjustments

- **Class imbalance**: In the 6-way classification, minority classes (e.g., imposter content: 40 samples, misleading content: 25 samples in test) showed 0% recall for baseline models. Binary classification produced more balanced results.
- **Satire detection**: As predicted by the Fakeddit paper, satire remains one of the hardest categories to classify, with the 3-way scheme showing only 3–47% recall depending on the model.
- **Scope**: As planned, we focus exclusively on text-based models without image data, keeping the project manageable while producing meaningful results.

## Overall Plan (Next Month)

1. **Week 1**: Increase training data size, run experiments with class balancing techniques (oversampling, class weights)
2. **Week 2**: Integrate NELA-GT dataset, compare model performance across both datasets
3. **Week 3**: GPU-accelerated BERT training on larger subsets, error analysis on misclassified samples
4. **Week 4**: Complete bias/ethical implications analysis, finalize visualizations, write final report
