"""
Baseline classification models.
Trains and evaluates Logistic Regression, Naive Bayes, and SVM classifiers.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def get_models() -> dict:
    """
    Return a dictionary of baseline models to train.

    Returns:
        Dictionary mapping model name to sklearn estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0,
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(
                max_iter=2000,
                C=1.0,
                random_state=42,
            ),
            cv=3,
        ),
    }


def train_model(model, X_train, y_train, model_name: str = "Model"):
    """
    Train a single model and return it with training time.

    Args:
        model: sklearn estimator.
        X_train: Training feature matrix.
        y_train: Training labels.
        model_name: Name for logging.

    Returns:
        Tuple of (trained model, training time in seconds).
    """
    print(f"\nTraining {model_name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.2f}s")
    return model, train_time


def train_all_models(X_train, y_train) -> dict:
    """
    Train all baseline models.

    Args:
        X_train: Training feature matrix (TF-IDF).
        y_train: Training labels.

    Returns:
        Dictionary mapping model name to (trained model, training time).
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        trained_model, train_time = train_model(model, X_train, y_train, name)
        results[name] = {
            "model": trained_model,
            "train_time": train_time,
        }

    return results


def predict(model, X) -> np.ndarray:
    """Generate predictions from a trained model."""
    return model.predict(X)


def predict_proba(model, X) -> np.ndarray:
    """Generate probability predictions if supported."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None
