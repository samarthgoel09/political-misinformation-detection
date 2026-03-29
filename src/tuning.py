"""
Hyperparameter tuning module.
Uses GridSearchCV to find optimal parameters for baseline models.
"""

import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def get_tuning_configs():
    """
    Return models and their parameter grids for tuning.

    Returns:
        Dictionary mapping model_name -> (model, param_grid).
    """
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            {"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
        ),
        "Linear SVM": (
            CalibratedClassifierCV(
                LinearSVC(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
                cv=3,
            ),
            {"estimator__C": [0.01, 0.1, 1.0, 10.0]},
        ),
    }


def tune_model(model, param_grid, X_train, y_train, cv=3, scoring="f1_macro"):
    """
    Run grid search for a single model.

    Args:
        model: sklearn estimator.
        param_grid: Dictionary of parameters to search.
        X_train: Training feature matrix.
        y_train: Training labels.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for optimization.

    Returns:
        Tuple of (best_estimator, best_params, best_score).
    """
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_all_models(X_train, y_train, cv=3, scoring="f1_macro"):
    """
    Tune all baseline models with grid search.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        cv: Number of cross-validation folds.
        scoring: Scoring metric.

    Returns:
        Dictionary matching the format expected by evaluate_all_models:
        {model_name: {"model": best_model, "train_time": seconds}}.
    """
    configs = get_tuning_configs()
    results = {}

    print("\n--- Hyperparameter Tuning ---")

    for name, (model, param_grid) in configs.items():
        print(f"\nTuning {name}...")
        print(f"  Search space: {param_grid}")

        start = time.time()
        best_model, best_params, best_score = tune_model(
            model, param_grid, X_train, y_train, cv=cv, scoring=scoring,
        )
        elapsed = time.time() - start

        print(f"  Best params: {best_params}")
        print(f"  Best CV {scoring}: {best_score:.4f}")
        print(f"  Tuning time: {elapsed:.2f}s")

        results[name] = {
            "model": best_model,
            "train_time": elapsed,
            "best_params": best_params,
            "best_cv_score": best_score,
        }

    return results
