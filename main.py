"""
Political Misinformation Detection — Main Pipeline
End-to-end runner: data loading → preprocessing → feature extraction → model training → evaluation.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sample_data import generate_sample_data
from src.preprocess import preprocess_dataframe
from src.features import extract_tfidf_features
from src.models import train_all_models
from src.evaluate import (
    evaluate_all_models,
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_model_comparison,
    save_metrics_csv,
)


def load_data(args):
    """Load dataset based on command-line arguments."""
    if args.use_sample_data:
        print("\n" + "=" * 60)
        print("  Loading SAMPLE data for development/testing")
        print("=" * 60)
        df = generate_sample_data(
            n_samples=args.n_samples,
            label_scheme=args.label_scheme,
        )
        return df

    if args.dataset in ("fakeddit", "both"):
        from data.download_fakeddit import load_fakeddit
        fakeddit_data = load_fakeddit(
            data_dir=args.fakeddit_dir,
            label_scheme=args.label_scheme,
            filter_political=True,
            max_samples=args.n_samples,
        )

        if fakeddit_data:
            # Combine train/val/test into one (we re-split later with TF-IDF)
            import pandas as pd
            dfs = list(fakeddit_data.values())
            df = pd.concat(dfs, ignore_index=True)
            print(f"\nFakeddit: {len(df)} total samples")

            if args.dataset == "fakeddit":
                return df

    if args.dataset in ("nela", "both"):
        from data.download_nela import load_nela
        nela_df = load_nela(
            data_dir=args.nela_dir,
            max_samples=args.n_samples,
        )
        print(f"\nNELA-GT: {len(nela_df)} total samples")

        if args.dataset == "nela":
            return nela_df

    if args.dataset == "both":
        import pandas as pd
        combined = pd.concat([df, nela_df], ignore_index=True)
        print(f"\nCombined: {len(combined)} total samples")
        return combined

    print("Error: No data loaded. Use --use-sample-data or provide dataset paths.")
    sys.exit(1)


def run_baseline_pipeline(df, args):
    """Run the baseline models (LR, NB, SVM) pipeline."""
    print("\n" + "=" * 60)
    print("  BASELINE MODELS PIPELINE")
    print("=" * 60)

    # Preprocess
    print("\n--- Preprocessing ---")
    df = preprocess_dataframe(df)

    # Extract features
    print("\n--- Feature Extraction ---")
    features = extract_tfidf_features(
        df,
        max_features=args.max_features,
        ngram_range=(1, 2),
    )

    # Train models
    print("\n--- Training Models ---")
    trained_models = train_all_models(features["X_train"], features["y_train"])

    # Evaluate
    print("\n--- Evaluation ---")
    label_names = features["unique_labels"]
    all_metrics = evaluate_all_models(
        trained_models,
        features["X_test"],
        features["y_test"],
        label_names=label_names,
        results_dir=args.results_dir,
    )

    return all_metrics


def run_bert_pipeline(df, args):
    """Run the DistilBERT pipeline."""
    print("\n" + "=" * 60)
    print("  DistilBERT PIPELINE")
    print("=" * 60)

    from src.bert_model import BertClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Use raw text (BERT has its own tokenizer)
    texts = df["text"].values
    labels = df["label_id"].values
    id_to_label = df.drop_duplicates("label_id").set_index("label_id")["label"].to_dict()
    label_names = [id_to_label[i] for i in range(len(id_to_label))]
    num_labels = len(label_names)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Limit samples for BERT if dataset is large
    max_bert_samples = args.bert_max_samples
    if len(X_train) > max_bert_samples:
        indices = np.random.RandomState(42).choice(
            len(X_train), max_bert_samples, replace=False
        )
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Limited BERT training to {max_bert_samples} samples")

    # Initialize and train
    bert = BertClassifier(
        num_labels=num_labels,
        batch_size=args.bert_batch_size,
        num_epochs=args.bert_epochs,
        max_length=args.bert_max_length,
    )

    history = bert.train(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    y_pred = bert.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, label_names)
    metrics["train_time"] = history.get("train_time", 0)

    print_classification_report(
        y_test, y_pred,
        label_names=label_names,
        title="DistilBERT",
    )

    plot_confusion_matrix(
        y_test, y_pred,
        label_names=label_names,
        title="DistilBERT",
        save_path=os.path.join(args.results_dir, "confusion_matrix_distilbert.png"),
    )

    # Save model
    bert.save(os.path.join(args.results_dir, "bert_model"))

    return {"DistilBERT": metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Political Misinformation Detection Pipeline"
    )

    # Data arguments
    parser.add_argument("--use-sample-data", action="store_true",
                        help="Use synthetic sample data for testing")
    parser.add_argument("--dataset", type=str, default="fakeddit",
                        choices=["fakeddit", "nela", "both"],
                        help="Which dataset to use")
    parser.add_argument("--fakeddit-dir", type=str, default="data/fakeddit",
                        help="Path to Fakeddit data directory")
    parser.add_argument("--nela-dir", type=str, default="data/nela",
                        help="Path to NELA-GT data directory")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples to use")
    parser.add_argument("--label-scheme", type=str, default="6way",
                        choices=["2way", "3way", "6way"],
                        help="Label classification scheme")

    # Feature arguments
    parser.add_argument("--max-features", type=int, default=10000,
                        help="Maximum TF-IDF vocabulary size")

    # BERT arguments
    parser.add_argument("--use-bert", action="store_true",
                        help="Also train DistilBERT model")
    parser.add_argument("--bert-epochs", type=int, default=3,
                        help="Number of BERT training epochs")
    parser.add_argument("--bert-batch-size", type=int, default=32,
                        help="BERT batch size")
    parser.add_argument("--bert-max-length", type=int, default=128,
                        help="BERT max token length")
    parser.add_argument("--bert-max-samples", type=int, default=5000,
                        help="Max samples for BERT training")

    # Output arguments
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  POLITICAL MISINFORMATION DETECTION PIPELINE")
    print("=" * 60)
    print(f"  Dataset: {'sample' if args.use_sample_data else args.dataset}")
    print(f"  Label scheme: {args.label_scheme}")
    print(f"  Samples: {args.n_samples}")
    print(f"  BERT: {'Yes' if args.use_bert else 'No'}")
    print("=" * 60)

    # Load data
    df = load_data(args)

    # Run baseline pipeline
    baseline_metrics = run_baseline_pipeline(df, args)

    # Run BERT pipeline (optional)
    bert_metrics = {}
    if args.use_bert:
        bert_metrics = run_bert_pipeline(df, args)

    # Combined comparison
    if bert_metrics:
        all_metrics = {**baseline_metrics, **bert_metrics}
        plot_model_comparison(
            all_metrics,
            save_path=os.path.join(args.results_dir, "model_comparison_all.png"),
        )
        save_metrics_csv(
            all_metrics,
            save_path=os.path.join(args.results_dir, "metrics_summary_all.csv"),
        )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {args.results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
