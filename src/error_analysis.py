"""
Error analysis module.
Identifies misclassified samples, surfaces patterns, and generates
visualizations to explain where and why models fail.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support


def collect_errors(texts, y_true, y_pred, label_names: list) -> pd.DataFrame:
    """
    Build a DataFrame of misclassified samples.

    Args:
        texts: Array of raw text strings (test set).
        y_true: Ground truth label IDs.
        y_pred: Predicted label IDs.
        label_names: List mapping label_id -> label name.

    Returns:
        DataFrame with one row per misclassified sample.
    """
    rows = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        rows.append({
            "text": texts[i],
            "true_id": true,
            "pred_id": pred,
            "true_label": label_names[true] if true < len(label_names) else str(true),
            "pred_label": label_names[pred] if pred < len(label_names) else str(pred),
            "correct": true == pred,
            "text_length": len(str(texts[i])),
            "word_count": len(str(texts[i]).split()),
        })
    df = pd.DataFrame(rows)
    return df


def print_error_summary(error_df: pd.DataFrame, model_name: str = ""):
    """Print a concise summary of error patterns."""
    errors = error_df[~error_df["correct"]]
    total = len(error_df)
    n_errors = len(errors)

    header = f"Error Analysis — {model_name}" if model_name else "Error Analysis"
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"{'='*60}")
    print(f"  Total test samples : {total}")
    print(f"  Misclassified      : {n_errors} ({100 * n_errors / total:.1f}%)")
    print(f"  Correctly classified: {total - n_errors} ({100 * (total - n_errors) / total:.1f}%)")

    print(f"\n--- Per-class error rate ---")
    for true_label, grp in error_df.groupby("true_label"):
        n_class = len(grp)
        n_wrong = len(grp[~grp["correct"]])
        print(f"  {true_label:<25} {n_wrong:>3}/{n_class:<3} errors  "
              f"({100 * n_wrong / n_class:.0f}% error rate)")

    print(f"\n--- Top confusion pairs (true -> predicted) ---")
    pairs = Counter(
        zip(errors["true_label"], errors["pred_label"])
    ).most_common(10)
    for (true, pred), count in pairs:
        print(f"  {true:<25} -> {pred:<25} ({count} times)")

    print(f"\n--- Avg text length: correct vs wrong ---")
    print(f"  Correct    : {error_df[error_df['correct']]['word_count'].mean():.1f} words")
    print(f"  Misclassified: {errors['word_count'].mean():.1f} words")


def plot_error_heatmap(error_df: pd.DataFrame, label_names: list,
                       model_name: str = "", save_path: str = None):
    """
    Plot a heatmap of confusion pairs for misclassified samples only.
    Helps visualize which class pairs are most confused.
    """
    errors = error_df[~error_df["correct"]]

    pivot = pd.crosstab(
        errors["true_label"],
        errors["pred_label"],
    )

    # Reindex to ensure all classes appear
    pivot = pivot.reindex(index=label_names, columns=label_names, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="Reds",
        ax=ax,
        linewidths=0.5,
    )
    title = f"Misclassification Heatmap — {model_name}" if model_name else "Misclassification Heatmap"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved error heatmap to {save_path}")
    plt.close()


def plot_error_rate_by_class(error_df: pd.DataFrame,
                              model_name: str = "", save_path: str = None):
    """Bar chart of per-class error rates."""
    stats = error_df.groupby("true_label").apply(
        lambda g: pd.Series({
            "error_rate": (~g["correct"]).mean(),
            "n_samples": len(g),
        })
    ).reset_index()

    stats = stats.sort_values("error_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(stats["true_label"], stats["error_rate"] * 100,
                   color=sns.color_palette("Reds_r", len(stats)))

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row['error_rate']*100:.0f}%  (n={int(row['n_samples'])})",
                va="center", fontsize=9)

    title = f"Per-class Error Rate — {model_name}" if model_name else "Per-class Error Rate"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Error Rate (%)", fontsize=11)
    ax.set_xlim(0, 115)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved error rate chart to {save_path}")
    plt.close()


def plot_text_length_distribution(error_df: pd.DataFrame,
                                   model_name: str = "", save_path: str = None):
    """Compare word count distributions for correct vs misclassified samples."""
    correct = error_df[error_df["correct"]]["word_count"]
    wrong = error_df[~error_df["correct"]]["word_count"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(correct, bins=30, alpha=0.6, label=f"Correct (n={len(correct)})",
            color="steelblue", density=True)
    ax.hist(wrong, bins=30, alpha=0.6, label=f"Misclassified (n={len(wrong)})",
            color="tomato", density=True)

    title = f"Text Length: Correct vs Misclassified — {model_name}" if model_name else "Text Length Distribution"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Word Count", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved text length chart to {save_path}")
    plt.close()


def save_error_samples(error_df: pd.DataFrame, save_path: str, n_per_class: int = 5):
    """
    Save a CSV of representative misclassified samples — useful for manual inspection
    and for citing in the report.
    """
    errors = error_df[~error_df["correct"]]
    if errors.empty:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(columns=["true_label", "pred_label", "text", "word_count"]).to_csv(save_path, index=False)
        print(f"  No errors to save (0 misclassified samples)")
        return

    samples = pd.concat(
        [g.head(n_per_class) for _, g in errors.groupby(["true_label", "pred_label"])],
        ignore_index=True,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    samples[["true_label", "pred_label", "text", "word_count"]].to_csv(save_path, index=False)
    print(f"  Saved {len(samples)} error samples to {save_path}")


def run_error_analysis(
    texts,
    y_true,
    y_pred,
    label_names: list,
    model_name: str = "Model",
    results_dir: str = "results",
):
    """
    Full error analysis pipeline for one model.

    Args:
        texts: Raw text strings for the test set.
        y_true: Ground truth label IDs.
        y_pred: Predicted label IDs.
        label_names: Ordered list of class names.
        model_name: Name used in plot titles and filenames.
        results_dir: Directory to save outputs.
    """
    error_dir = os.path.join(results_dir, "error_analysis")
    safe_name = model_name.lower().replace(" ", "_")

    error_df = collect_errors(texts, y_true, y_pred, label_names)

    print_error_summary(error_df, model_name)

    plot_error_heatmap(
        error_df, label_names, model_name,
        save_path=os.path.join(error_dir, f"error_heatmap_{safe_name}.png"),
    )
    plot_error_rate_by_class(
        error_df, model_name,
        save_path=os.path.join(error_dir, f"error_rate_{safe_name}.png"),
    )
    plot_text_length_distribution(
        error_df, model_name,
        save_path=os.path.join(error_dir, f"text_length_{safe_name}.png"),
    )
    save_error_samples(
        error_df,
        save_path=os.path.join(error_dir, f"error_samples_{safe_name}.csv"),
    )

    return error_df


# ---------------------------------------------------------------------------
# Bias / Ethical Analysis
# ---------------------------------------------------------------------------

def analyze_label_distribution(y_train, y_test, label_names, save_path=None):
    """Plot side-by-side class distribution for train and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in [(axes[0], y_train, "Train"), (axes[1], y_test, "Test")]:
        unique, counts = np.unique(data, return_counts=True)
        names = [label_names[i] if i < len(label_names) else str(i) for i in unique]
        bars = ax.bar(names, counts, color="steelblue", edgecolor="black")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title} Set Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.sca(ax)
        plt.xticks(rotation=30, ha="right")

    plt.suptitle("Label Distribution Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved label distribution to {save_path}")
    plt.close()

    # Print imbalance ratio
    _, train_counts = np.unique(y_train, return_counts=True)
    ratio = train_counts.max() / train_counts.min() if train_counts.min() > 0 else float("inf")
    print(f"\n  Train imbalance ratio (max/min): {ratio:.1f}x")


def analyze_performance_disparities(y_true, y_pred, label_names, save_path=None):
    """Plot per-class precision, recall, F1 and flag underserved classes."""
    unique_labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )

    names = [label_names[i] if i < len(label_names) else str(i) for i in unique_labels]
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3")
    ax.bar(x, recall, width, label="Recall", color="#4CAF50")
    ax.bar(x + width, f1, width, label="F1-Score", color="#FF9800")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Disparities")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved performance disparities to {save_path}")
    plt.close()

    # Flag underserved classes
    for i, name in enumerate(names):
        if f1[i] < 0.5:
            print(f"  WARNING: Class '{name}' has F1={f1[i]:.3f} (below 0.5 threshold)")


def analyze_text_bias(texts, labels, label_names):
    """Analyze text length patterns per class."""
    print("\n  Text length analysis by class:")
    for label_id in sorted(set(labels)):
        mask = labels == label_id
        class_texts = texts[mask] if hasattr(texts, '__getitem__') else [texts[i] for i, m in enumerate(mask) if m]
        lengths = [len(str(t)) for t in class_texts]
        if lengths:
            name = label_names[label_id] if label_id < len(label_names) else str(label_id)
            print(f"    {name:30s}: avg={np.mean(lengths):.0f}, "
                  f"median={np.median(lengths):.0f}, "
                  f"std={np.std(lengths):.0f} chars")


def run_bias_analysis(y_train, y_test, y_pred, texts_test, label_names, results_dir):
    """
    Run full bias and ethical analysis.

    Args:
        y_train: Training labels.
        y_test: Test ground truth labels.
        y_pred: Test predictions.
        texts_test: Raw test texts.
        label_names: List mapping label_id -> label string.
        results_dir: Base results directory.
    """
    print("\n" + "=" * 60)
    print("  BIAS & ETHICAL ANALYSIS")
    print("=" * 60)

    bias_dir = os.path.join(results_dir, "bias_analysis")

    analyze_label_distribution(
        y_train, y_test, label_names,
        save_path=os.path.join(bias_dir, "label_distribution.png"),
    )
    analyze_performance_disparities(
        y_test, y_pred, label_names,
        save_path=os.path.join(bias_dir, "performance_disparities.png"),
    )
    analyze_text_bias(texts_test, y_test, label_names)

    print(f"\n  Bias analysis results saved to {bias_dir}/")
