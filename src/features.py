"""
Feature extraction module.
Provides TF-IDF vectorization for baseline models.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def extract_tfidf_features(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Extract TF-IDF features and create train/val/test splits.

    Args:
        df: DataFrame with text and labels.
        text_col: Column name containing preprocessed text.
        max_features: Maximum vocabulary size.
        ngram_range: N-gram range for TF-IDF.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        random_state: Random seed.

    Returns:
        Dictionary with train/val/test feature matrices and labels.
    """
    texts = df[text_col].values
    labels = df["label_id"].values
    label_names = df["label"].values

    # Split: train -> (train + val), test
    X_temp, X_test, y_temp, y_test, ln_temp, ln_test = train_test_split(
        texts, labels, label_names,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Split train into train + val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ln_train, ln_val = train_test_split(
        X_temp, y_temp, ln_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    # Fit TF-IDF on training data
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF Features:")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Train: {X_train_tfidf.shape}")
    print(f"  Val:   {X_val_tfidf.shape}")
    print(f"  Test:  {X_test_tfidf.shape}")

    # Get unique labels  
    unique_labels = sorted(set(label_names))

    return {
        "X_train": X_train_tfidf,
        "X_val": X_val_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_names_train": ln_train,
        "label_names_test": ln_test,
        "label_names_val": ln_val,
        "vectorizer": vectorizer,
        "unique_labels": unique_labels,
    }


def get_top_features(vectorizer, model, class_names, n_top: int = 15):
    """
    Get the top TF-IDF features for each class (works with linear models).

    Args:
        vectorizer: Fitted TfidfVectorizer.
        model: Trained sklearn model with coef_ attribute.
        class_names: List of class labels.
        n_top: Number of top features per class.

    Returns:
        Dictionary mapping class name to list of top features.
    """
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}

    if hasattr(model, "coef_"):
        for i, class_name in enumerate(class_names):
            if model.coef_.shape[0] > 1:
                coef = model.coef_[i]
            else:
                coef = model.coef_[0]
            top_idx = np.argsort(coef)[-n_top:][::-1]
            top_features[class_name] = [
                (feature_names[j], round(coef[j], 4))
                for j in top_idx
            ]

    return top_features
