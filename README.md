# Political Misinformation Detection

An AI-based system for detecting political misinformation in U.S. news and online content using supervised learning.

## Overview

This project classifies text as credible or misleading using linguistic and structural features learned from labeled datasets. It implements and compares multiple classification approaches:

- **Baseline Models**: Logistic Regression, Multinomial Naive Bayes, Linear SVM (using TF-IDF features)
- **Deep Learning**: Fine-tuned DistilBERT for multi-class text classification

## Datasets

| Dataset | Source | Labels |
|---------|--------|--------|
| [Fakeddit](https://github.com/entitize/Fakeddit) | Reddit posts (1M+ samples) | 6-way: true, satire, misleading, imposter, false connection, manipulated |
| [NELA-GT](https://doi.org/10.7910/DVN/CHMUYZ) | News articles with source credibility | 3-way: reliable, mixed, unreliable |

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

### With Real Datasets
1. Download the Fakeddit dataset from [GitHub](https://github.com/entitize/Fakeddit) and place TSV files in `data/fakeddit/`
2. Download NELA-GT from [Harvard Dataverse](https://doi.org/10.7910/DVN/CHMUYZ) and place in `data/nela/`
3. Run the pipeline:
```bash
python main.py --dataset fakeddit    # Fakeddit only
python main.py --dataset nela        # NELA-GT only
python main.py --dataset both        # Both datasets
```

### Train BERT Model
```bash
python main.py --use-sample-data --use-bert
```

## Project Structure

```
political-misinfo-detection/
├── main.py              # End-to-end pipeline runner
├── requirements.txt     # Python dependencies
├── data/
│   ├── sample_data.py       # Synthetic data generator
│   ├── download_fakeddit.py # Fakeddit loader
│   └── download_nela.py     # NELA-GT loader
├── src/
│   ├── preprocess.py    # Text cleaning & tokenization
│   ├── features.py      # TF-IDF feature extraction
│   ├── models.py        # Baseline classifiers
│   ├── bert_model.py    # DistilBERT fine-tuning
│   └── evaluate.py      # Metrics & visualizations
├── results/             # Generated plots & metrics
└── notebooks/           # Jupyter exploration (optional)
```

## Evaluation Metrics

- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1-Score (macro & weighted)
- Confusion Matrices

## References

- Nakamura, K., Levy, S., & Wang, W. Y. (2020). r/Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection. *LREC 2020*, 6149–6157.
- Nørregaard, J., Horne, B. D., & Adalı, S. (2019). NELA-GT-2018: A large multi-labelled news dataset for the study of fake news. *ICWSM 2019*.
