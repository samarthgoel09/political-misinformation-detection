"""
NELA-GT dataset loader.
Loads and processes the NELA-GT (News Landscape) dataset for credibility classification.

Supported formats:
    1. CSV file (e.g. nela_ps_newsdata.csv) placed directly in data/nela/ or data/
       Columns expected: source, title, content (labels assigned from built-in mapping)
    2. SQLite .db file + labels.csv in data/nela/
    3. articles/ JSON directory + labels.csv in data/nela/
"""

import os
import json
import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Hardcoded source credibility based on Media Bias / Fact Check & NewsGuard.
# 0 = reliable, 1 = mixed, 2 = unreliable
# ---------------------------------------------------------------------------
SOURCE_CREDIBILITY = {
    # Reliable (0)
    "apnews": 0, "ap": 0, "reuters": 0, "bbc": 0, "bbcnews": 0,
    "nytimes": 0, "newyorktimes": 0, "washingtonpost": 0, "theguardian": 0,
    "guardian": 0, "npr": 0, "pbs": 0, "pbsnews": 0, "usatoday": 0,
    "abcnews": 0, "cbsnews": 0, "nbcnews": 0, "time": 0, "theatlantic": 0,
    "economist": 0, "wsj": 0, "wallstreetjournal": 0, "forbes": 0,
    "politico": 0, "thehill": 0, "businessinsider": 0, "axios": 0,
    "vox": 0, "slate": 0, "huffpost": 0, "huffingtonpost": 0,
    "msnbc": 0, "cnn": 0, "cbcnews": 0, "cbc": 0, "aljazeera": 0,
    "derspiegel": 0, "lemonde": 0, "independent": 0, "theindependent": 0,
    "telegraph": 0, "thetelegraph": 0, "latimes": 0, "losangelestimes": 0,
    "chicagotribune": 0, "bostonglobe": 0, "sfgate": 0,
    "scientificamerican": 0, "nature": 0, "newscientist": 0,
    "albusinessdaily": 0, "metro": 0,
    # Mixed (1)
    "foxnews": 1, "fox": 1, "nypost": 1, "newyorkpost": 1,
    "dailymail": 1, "dailymailonline": 1, "usatoday": 1,
    "theblaze": 1, "nationalreview": 1, "weeklystandard": 1,
    "reason": 1, "motherjones": 1, "thenation": 1, "jacobin": 1,
    "salon": 1, "rawstory": 1, "thedailybeast": 1, "dailybeast": 1,
    "mediaite": 1, "theintercept": 1, "intercept": 1,
    "americanthinker": 1, "spectator": 1, "spectatorusa": 1,
    "washingtontimes": 1, "oann": 1, "onenewsnow": 1,
    "dailycaller": 1, "thedailycaller": 1, "townhall": 1,
    "redstate": 1, "hotair": 1, "pjmedia": 1,
    # Unreliable / far-right / conspiracy (2)
    "infowars": 2, "naturalnews": 2, "breitbart": 2,
    "thegatewaypundit": 2, "gatewaypundit": 2,
    "zerohedge": 2, "activistpost": 2, "beforeitsnews": 2,
    "prisonplanet": 2, "wnd": 2, "worldnetdaily": 2,
    "newsmax": 2, "epochtimes": 2, "theepochtimes": 2,
    "truepundit": 2, "conservativetreehouse": 2,
    "thepatriotpost": 2, "patriotpost": 2,
    "veteranstoday": 2, "globalresearch": 2,
    "21stcenturywire": 2, "dcclothesline": 2,
    "yournewswire": 2, "newspunch": 2,
}


# NELA-GT source credibility mapping
# Aggregated labels based on Media Bias/Fact Check, NewsGuard, etc.
CREDIBILITY_MAP = {
    0: "reliable",
    1: "mixed",
    2: "unreliable",
}


def load_nela_from_json(
    data_dir: str = "data/nela",
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Load NELA-GT articles from JSON files.

    Args:
        data_dir: Directory containing NELA-GT data.
        max_samples: Maximum samples to load.

    Returns:
        DataFrame with 'text', 'label', 'label_id', 'source' columns.
    """
    articles_dir = os.path.join(data_dir, "articles")
    labels_file = os.path.join(data_dir, "labels.csv")

    if not os.path.exists(labels_file):
        raise FileNotFoundError(
            f"Labels file not found at {labels_file}. "
            "Please download NELA-GT and place labels.csv in the data/nela/ directory."
        )

    # Load source-level credibility labels
    labels_df = pd.read_csv(labels_file)

    # The labels file typically has columns like:
    # source, aggregated_label (or similar)
    # Try to identify the label column
    label_col = None
    for col in labels_df.columns:
        if "label" in col.lower() or "aggregated" in col.lower() or "credibility" in col.lower():
            label_col = col
            break

    if label_col is None:
        # If can't find, try the last numeric column
        numeric_cols = labels_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            label_col = numeric_cols[-1]
        else:
            raise ValueError("Cannot identify label column in labels.csv")

    # Create source -> label mapping
    source_col = labels_df.columns[0]  # Usually the first column is the source name
    source_labels = dict(zip(
        labels_df[source_col].str.lower(),
        labels_df[label_col]
    ))

    print(f"Loaded {len(source_labels)} source credibility labels")

    # Load articles
    all_articles = []
    json_files = glob.glob(os.path.join(articles_dir, "**/*.json"), recursive=True)

    if not json_files:
        # Try loading from subdirectories named by source
        source_dirs = [d for d in os.listdir(articles_dir)
                       if os.path.isdir(os.path.join(articles_dir, d))]
        for source_name in source_dirs:
            source_path = os.path.join(articles_dir, source_name)
            json_files.extend(glob.glob(os.path.join(source_path, "*.json")))

    print(f"Found {len(json_files)} article files")

    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                articles = json.load(f)
                if isinstance(articles, dict):
                    articles = [articles]
                for article in articles:
                    source = article.get("source", os.path.basename(os.path.dirname(filepath)))
                    source_lower = source.lower()

                    if source_lower in source_labels:
                        text = article.get("content", article.get("title", ""))
                        if text and len(text.strip()) > 20:
                            label_id = int(source_labels[source_lower])
                            all_articles.append({
                                "text": text[:1000],  # Truncate very long articles
                                "source": source,
                                "label_id": label_id,
                                "label": CREDIBILITY_MAP.get(label_id, "unknown"),
                            })
        except (json.JSONDecodeError, Exception) as e:
            continue

        if max_samples and len(all_articles) >= max_samples:
            break

    df = pd.DataFrame(all_articles)

    if len(df) == 0:
        print("Warning: No articles loaded. Check data directory structure.")
        return df

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(df)} articles")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def load_nela_from_sqlite(
    db_path: str = "data/nela/nela-gt.db",
    labels_path: str = "data/nela/labels.csv",
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Load NELA-GT articles from SQLite database.

    Args:
        db_path: Path to the SQLite database.
        labels_path: Path to the source credibility labels CSV.
        max_samples: Maximum samples to load.

    Returns:
        DataFrame with 'text', 'label', 'label_id', 'source' columns.
    """
    import sqlite3

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    # Load source labels
    labels_df = pd.read_csv(labels_path)
    source_col = labels_df.columns[0]
    label_col = None
    for col in labels_df.columns:
        if "label" in col.lower() or "aggregated" in col.lower():
            label_col = col
            break
    if label_col is None:
        label_col = labels_df.select_dtypes(include=[np.number]).columns[-1]

    source_labels = dict(zip(
        labels_df[source_col].str.lower(),
        labels_df[label_col]
    ))

    # Query articles
    query = "SELECT source, title, content FROM newsdata"
    if max_samples:
        query += f" LIMIT {max_samples * 3}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Map source to credibility label
    df["source_lower"] = df["source"].str.lower()
    df = df[df["source_lower"].isin(source_labels)].copy()
    df["label_id"] = df["source_lower"].map(source_labels).astype(int)
    df["label"] = df["label_id"].map(CREDIBILITY_MAP)

    # Use title + content or just content
    df["text"] = df.apply(
        lambda row: f"{row['title']}. {row['content'][:800]}"
        if pd.notna(row.get("content")) else str(row.get("title", "")),
        axis=1
    )

    df = df[["text", "label", "label_id", "source"]].dropna()
    df = df[df["text"].str.len() > 20].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(df)} articles from SQLite")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def load_nela_from_csv(
    csv_path: str,
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Load NELA articles from a CSV file using the built-in SOURCE_CREDIBILITY mapping.

    Expected CSV columns: source, title, content (others are ignored).
    Articles whose source is not in the credibility mapping are dropped.
    """
    # Read in chunks to avoid OOM on large CSVs
    chunks = []
    unknown = 0
    chunk_size = 50_000

    reader = pd.read_csv(csv_path, low_memory=False, chunksize=chunk_size)
    for chunk in reader:
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        if "source" not in chunk.columns:
            raise ValueError(f"CSV {csv_path} has no 'source' column.")

        chunk["source_key"] = chunk["source"].str.lower().str.strip().str.replace(r"\s+", "", regex=True)
        chunk["label_id"] = chunk["source_key"].map(SOURCE_CREDIBILITY)

        unknown += chunk["label_id"].isna().sum()
        chunk = chunk.dropna(subset=["label_id"]).copy()
        if chunk.empty:
            continue

        chunk["label_id"] = chunk["label_id"].astype(int)
        chunk["label"] = chunk["label_id"].map(CREDIBILITY_MAP)

        # Build text from title + content
        if "content" in chunk.columns and "title" in chunk.columns:
            chunk["text"] = chunk["title"].fillna("") + ". " + chunk["content"].fillna("")
        elif "content" in chunk.columns:
            chunk["text"] = chunk["content"].fillna("")
        elif "title" in chunk.columns:
            chunk["text"] = chunk["title"].fillna("")
        else:
            raise ValueError("CSV must have at least a 'title' or 'content' column.")

        chunk = chunk[chunk["text"].str.len() > 20]
        chunks.append(chunk[["text", "label", "label_id", "source"]])

    if not chunks:
        raise ValueError("No articles with known source credibility found in the CSV.")

    df = pd.concat(chunks, ignore_index=True)

    print(f"Loaded {len(df)} articles (dropped {unknown} with unknown source credibility)")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    return df


def load_nela(data_dir: str = "data/nela", max_samples: int = None) -> pd.DataFrame:
    """
    Auto-detect NELA-GT format and load accordingly.

    Detection order:
      1. CSV files in data_dir (e.g. nela_ps_newsdata.csv)
      2. CSV files in the parent data/ directory
      3. SQLite .db file + labels.csv
      4. articles/ JSON directory + labels.csv
    """
    # 1. CSV in data_dir
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    # Exclude labels.csv itself
    csv_files = [f for f in csv_files if os.path.basename(f).lower() != "labels.csv"]

    # 2. CSV in parent data/ directory
    if not csv_files:
        parent_dir = os.path.dirname(data_dir)
        csv_files = glob.glob(os.path.join(parent_dir, "nela*.csv"))

    if csv_files:
        print(f"Loading NELA from CSV: {csv_files[0]}")
        return load_nela_from_csv(csv_files[0], max_samples=max_samples)

    db_files = glob.glob(os.path.join(data_dir, "*.db"))
    labels_file = os.path.join(data_dir, "labels.csv")

    if db_files:
        return load_nela_from_sqlite(
            db_path=db_files[0],
            labels_path=labels_file,
            max_samples=max_samples,
        )
    elif os.path.exists(os.path.join(data_dir, "articles")):
        return load_nela_from_json(data_dir=data_dir, max_samples=max_samples)
    else:
        raise FileNotFoundError(
            f"No NELA-GT data found in {data_dir} or its parent directory. "
            "Place nela_ps_newsdata.csv (or similar) in data/nela/ or data/."
        )


if __name__ == "__main__":
    df = load_nela(max_samples=5000)
    print(f"\nLoaded {len(df)} NELA-GT articles")
    print(df.head())
