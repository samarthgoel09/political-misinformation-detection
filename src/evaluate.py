"""
Evaluation module.
Computes classification metrics and generates visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, label_names=None) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Optional list of label names.

    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    return metrics


def print_classification_report(y_true, y_pred, label_names=None, title=""):
    """Print a formatted classification report."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    if label_names is not None:
        # Map numeric labels to names
        unique_labels = sorted(set(y_true) | set(y_pred))
        target_names = [label_names[i] if i < len(label_names) else f"class_{i}"
                       for i in unique_labels]
        report = classification_report(
            y_true, y_pred,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
        )
    else:
        report = classification_report(y_true, y_pred, zero_division=0)

    print(report)
    return report


def plot_confusion_matrix(
    y_true,
    y_pred,
    label_names,
    title: str = "Confusion Matrix",
    save_path: str = None,
    figsize: tuple = (10, 8),
):
    """
    Plot and optionally save a confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: List of label names.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[0],
    )
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[1],
    )
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix to {save_path}")

    plt.close()


def plot_model_comparison(
    all_results: dict,
    metric_keys: list = None,
    save_path: str = None,
    figsize: tuple = (14, 8),
):
    """
    Plot a bar chart comparing models across metrics.

    Args:
        all_results: Dict mapping model_name -> metrics dict.
        metric_keys: Which metrics to plot.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    if metric_keys is None:
        metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    model_names = list(all_results.keys())
    n_metrics = len(metric_keys)

    # Clean metric names for display
    display_names = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision\n(Macro)",
        "precision_weighted": "Precision\n(Weighted)",
        "recall_macro": "Recall\n(Macro)",
        "recall_weighted": "Recall\n(Weighted)",
        "f1_macro": "F1-Score\n(Macro)",
        "f1_weighted": "F1-Score\n(Weighted)",
    }

    x = np.arange(n_metrics)
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, model_name in enumerate(model_names):
        values = [all_results[model_name].get(k, 0) for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(k, k) for k in metric_keys])
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved model comparison chart to {save_path}")

    plt.close()


def save_metrics_csv(all_results: dict, save_path: str):
    """
    Save all model metrics to a CSV file.

    Args:
        all_results: Dict mapping model_name -> metrics dict.
        save_path: Path to save CSV.
    """
    rows = []
    for model_name, metrics in all_results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nSaved metrics to {save_path}")
    print(df.to_string(index=False))


def evaluate_all_models(
    trained_models: dict,
    X_test,
    y_test,
    label_names: list,
    results_dir: str = "results",
) -> dict:
    """
    Evaluate all trained models and generate reports.

    Args:
        trained_models: Dict from train_all_models().
        X_test: Test feature matrix.
        y_test: Test labels.
        label_names: List of label names.
        results_dir: Directory to save results.

    Returns:
        Dictionary mapping model_name -> metrics.
    """
    all_metrics = {}

    for model_name, model_info in trained_models.items():
        model = model_info["model"]
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, label_names)
        metrics["train_time"] = model_info.get("train_time", 0)
        all_metrics[model_name] = metrics

        # Print report
        print_classification_report(
            y_test, y_pred,
            label_names=label_names,
            title=model_name,
        )

        # Plot confusion matrix
        safe_name = model_name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_test, y_pred,
            label_names=label_names,
            title=model_name,
            save_path=os.path.join(results_dir, f"confusion_matrix_{safe_name}.png"),
        )

    # Model comparison chart
    plot_model_comparison(
        all_metrics,
        save_path=os.path.join(results_dir, "model_comparison.png"),
    )

    # Save metrics CSV
    save_metrics_csv(
        all_metrics,
        save_path=os.path.join(results_dir, "metrics_summary.csv"),
    )

    return all_metrics
