"""
Cross-dataset evaluation module.
Trains models on one dataset and tests them on another to measure
how well misinformation detectors generalize beyond their training data.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# Label harmonization
def _harmonize_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse any label scheme down to 2-way (0=fake, 1=true) so that
    models trained on one dataset can be evaluated on another.

    Fakeddit 6-way:
        0 (true)          -> 1
        1-5 (all fake)    -> 0

    LIAR 6-way:
        pants-fire (0), false (1), barely-true (2) -> 0 (fake)
        half-true (3), mostly-true (4), true (5)   -> 1 (true)

    NELA 3-way:
        0 (reliable)   -> 1
        1 (mixed)      -> 0   (treated as unreliable for binary)
        2 (unreliable) -> 0

    Returns a copy of df with label_id remapped to {0, 1} and label
    set to 'fake' / 'true'.
    """
    df = df.copy()

    # Detect which scheme is in use based on unique label_ids
    unique_ids = sorted(df["label_id"].unique())

    if set(unique_ids) <= {0, 1}:
        # Make sure names are canonical
        df["label"] = df["label_id"].map({0: "fake", 1: "true"})
        return df

    # Fakeddit 6-way: 0=true, 1-5=fake
    if max(unique_ids) == 5 and "satire" in " ".join(df["label"].unique()).lower():
        df["label_id"] = df["label_id"].apply(lambda x: 1 if x == 0 else 0)
        df["label"] = df["label_id"].map({0: "fake", 1: "true"})
        return df

    # LIAR 6-way: 0,1,2=fake -> 0 / 3,4,5=true -> 1
    if max(unique_ids) == 5:
        df["label_id"] = df["label_id"].apply(lambda x: 0 if x <= 2 else 1)
        df["label"] = df["label_id"].map({0: "fake", 1: "true"})
        return df

    # NELA 3-way: 0=reliable -> 1, 1/2=unreliable -> 0
    if set(unique_ids) <= {0, 1, 2}:
        df["label_id"] = df["label_id"].apply(lambda x: 1 if x == 0 else 0)
        df["label"] = df["label_id"].map({0: "fake", 1: "true"})
        return df

    raise ValueError(
        f"Cannot auto-harmonize label_ids {unique_ids} to binary. "
        "Please preprocess labels manually before calling cross-dataset eval."
    )

# TF-IDF helpers
def _fit_tfidf(train_texts, max_features: int = 10_000):
    """Fit a TF-IDF vectorizer on training texts and return it."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X = vec.fit_transform(train_texts)
    return vec, X

# Model helpers
def _get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=42),
            cv=3,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    }


def _train_model(model, X_train, y_train, name: str):
    """Train a single model, using sample weights for NB."""
    start = time.time()
    if isinstance(model, MultinomialNB):
        sw = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Trained {name} in {elapsed:.2f}s")
    return model, elapsed

# Metrics
def _compute_metrics(y_true, y_pred, label_names):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

# Plotting
def _plot_cross_dataset_results(
    results: dict,
    source_name: str,
    target_name: str,
    save_path: str = None,
):
    """
    Bar chart comparing cross-dataset vs. in-distribution performance.
    results: {model_name: {"in_dist": metrics, "cross": metrics}}
    """
    model_names = list(results.keys())
    metrics = ["accuracy", "f1_macro"]
    display = {"accuracy": "Accuracy", "f1_macro": "F1 (Macro)"}

    n_models = len(model_names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.35

    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = {"in_dist": "#2196F3", "cross": "#FF7043"}

    for ax, metric in zip(axes, metrics):
        in_vals = [results[m]["in_dist"].get(metric, 0) for m in model_names]
        cross_vals = [results[m]["cross"].get(metric, 0) for m in model_names]

        bars_in = ax.bar(x - width / 2, in_vals, width, label=f"In-dist ({source_name})",
                         color=colors["in_dist"], alpha=0.85)
        bars_cross = ax.bar(x + width / 2, cross_vals, width,
                            label=f"Cross-dataset ({source_name} → {target_name})",
                            color=colors["cross"], alpha=0.85)

        for bars in [bars_in, bars_cross]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(f"{display[metric]} — In-dist vs Cross-dataset", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Cross-Dataset Generalization: Train on {source_name}, Test on {target_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved cross-dataset comparison chart to {save_path}")
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, label_names, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title_suffix in [
        (axes[0], cm, "d", "Counts"),
        (axes[1], cm_norm, ".2f", "Normalized"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(f"{title} ({title_suffix})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix to {save_path}")
    plt.close()

# Main entry point
def run_cross_dataset_evaluation(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str = "Source",
    target_name: str = "Target",
    max_features: int = 10_000,
    results_dir: str = "results",
) -> dict:
    """
    Train on source_df, evaluate on both source (in-distribution) and
    target (cross-dataset) test sets using a shared binary label space.

    Args:
        source_df:    DataFrame with 'text', 'label', 'label_id' from source dataset.
        target_df:    DataFrame from target dataset (same column schema).
        source_name:  Human-readable name for the source (e.g. "LIAR").
        target_name:  Human-readable name for the target (e.g. "Fakeddit").
        max_features: TF-IDF vocabulary size.
        results_dir:  Root directory for saving outputs.

    Returns:
        Nested dict {model_name: {"in_dist": metrics, "cross": metrics}}.
    """
    print("\n" + "=" * 60)
    print(f"  CROSS-DATASET EVALUATION")
    print(f"  Train: {source_name}  |  Test: {target_name}")
    print("=" * 60)

    # Harmonize labels to binary
    print("\nHarmonizing labels to binary (fake=0 / true=1)...")
    source_df = _harmonize_to_binary(source_df)
    target_df = _harmonize_to_binary(target_df)
    label_names = ["fake", "true"]

    print(f"  {source_name} distribution:\n{source_df['label'].value_counts().to_string()}")
    print(f"  {target_name} distribution:\n{target_df['label'].value_counts().to_string()}")

    from src.preprocess import preprocess_dataframe
    print(f"\nPreprocessing {source_name} texts...")
    source_df = preprocess_dataframe(source_df)
    print(f"Preprocessing {target_name} texts...")
    target_df = preprocess_dataframe(target_df)

    # train and test
    from sklearn.model_selection import train_test_split
    X_src_train, X_src_test, y_src_train, y_src_test = train_test_split(
        source_df["clean_text"].to_numpy(),
        source_df["label_id"].to_numpy(dtype=int),
        test_size=0.2,
        random_state=42,
        stratify=source_df["label_id"].to_numpy(dtype=int),
    )

    # Using all of target as the cross-dataset test set
    X_tgt = target_df["clean_text"].to_numpy()
    y_tgt = target_df["label_id"].to_numpy(dtype=int)

    print("\nFitting TF-IDF on source training set...")
    vectorizer, X_src_train_tfidf = _fit_tfidf(X_src_train, max_features=max_features)
    X_src_test_tfidf = vectorizer.transform(X_src_test)
    X_tgt_tfidf = vectorizer.transform(X_tgt)

    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Source train: {X_src_train_tfidf.shape}")
    print(f"  Source test (in-dist): {X_src_test_tfidf.shape}")
    print(f"  Target test (cross):   {X_tgt_tfidf.shape}")

    # Training and evaluating
    models = _get_models()
    all_results = {}
    cross_dir = os.path.join(results_dir, "cross_dataset")
    os.makedirs(cross_dir, exist_ok=True)

    summary_rows = []

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        trained_model, train_time = _train_model(model, X_src_train_tfidf, y_src_train, model_name)

        # In-distribution performance
        y_pred_in = trained_model.predict(X_src_test_tfidf)
        in_metrics = _compute_metrics(y_src_test, y_pred_in, label_names)
        in_metrics["train_time"] = train_time

        print(f"  In-distribution ({source_name} → {source_name} test):")
        print(f"    Accuracy: {in_metrics['accuracy']:.4f}  |  F1 Macro: {in_metrics['f1_macro']:.4f}")
        print(classification_report(y_src_test, y_pred_in, target_names=label_names,
                                    zero_division=0))

        # Cross-dataset performance
        y_pred_cross = trained_model.predict(X_tgt_tfidf)
        cross_metrics = _compute_metrics(y_tgt, y_pred_cross, label_names)
        cross_metrics["train_time"] = 0  # same model, no extra training

        print(f"  Cross-dataset ({source_name} → {target_name}):")
        print(f"    Accuracy: {cross_metrics['accuracy']:.4f}  |  F1 Macro: {cross_metrics['f1_macro']:.4f}")
        print(classification_report(y_tgt, y_pred_cross, target_names=label_names,
                                    zero_division=0))

        # Generalization gap
        gap = in_metrics["f1_macro"] - cross_metrics["f1_macro"]
        print(f"  Generalization gap (F1 macro): {gap:+.4f}")

        # Confusion matrix for cross-dataset
        safe_name = model_name.lower().replace(" ", "_")
        _plot_confusion_matrix(
            y_tgt, y_pred_cross, label_names,
            title=f"{model_name}: {source_name} → {target_name}",
            save_path=os.path.join(cross_dir, f"cm_cross_{safe_name}.png"),
        )

        all_results[model_name] = {
            "in_dist": in_metrics,
            "cross": cross_metrics,
        }

        summary_rows.append({
            "model": model_name,
            "source": source_name,
            "target": target_name,
            "in_dist_accuracy": in_metrics["accuracy"],
            "in_dist_f1_macro": in_metrics["f1_macro"],
            "cross_accuracy": cross_metrics["accuracy"],
            "cross_f1_macro": cross_metrics["f1_macro"],
            "generalization_gap_f1": gap,
        })

    # Summary
    _plot_cross_dataset_results(
        all_results,
        source_name=source_name,
        target_name=target_name,
        save_path=os.path.join(cross_dir, f"cross_dataset_{source_name.lower()}_to_{target_name.lower()}.png"),
    )

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(cross_dir, "cross_dataset_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Saved cross-dataset summary to {csv_path}")
    print(summary_df.to_string(index=False))

    return all_results


def run_bidirectional_cross_dataset(
    liar_df: pd.DataFrame,
    fakeddit_df: pd.DataFrame,
    max_features: int = 10_000,
    results_dir: str = "results",
) -> dict:
    """
    Convenience wrapper that runs cross-dataset eval in BOTH directions:
        LIAR -> Fakeddit
        Fakeddit -> LIAR

    Returns combined results dict.
    """
    print("\n" + "=" * 60)
    print("  BIDIRECTIONAL CROSS-DATASET EVALUATION")
    print("=" * 60)

    results = {}

    # Direction 1: LIAR -> Fakeddit
    r1 = run_cross_dataset_evaluation(
        source_df=liar_df,
        target_df=fakeddit_df,
        source_name="LIAR",
        target_name="Fakeddit",
        max_features=max_features,
        results_dir=results_dir,
    )
    results["LIAR→Fakeddit"] = r1

    # Direction 2: Fakeddit -> LIAR
    r2 = run_cross_dataset_evaluation(
        source_df=fakeddit_df,
        target_df=liar_df,
        source_name="Fakeddit",
        target_name="LIAR",
        max_features=max_features,
        results_dir=results_dir,
    )
    results["Fakeddit→LIAR"] = r2

    return results