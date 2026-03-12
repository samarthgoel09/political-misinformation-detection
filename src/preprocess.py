"""
Text preprocessing module.
Handles text cleaning, normalization, and tokenization for the pipeline.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Ensure NLTK data is downloaded
def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    for resource in ["stopwords", "wordnet", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource != "punkt_tab" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

# Initialize
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Compile regex patterns for performance
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean a single text string.

    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove @mentions
        5. Extract hashtag text
        6. Remove special characters and numbers
        7. Normalize whitespace

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove HTML tags
    text = HTML_PATTERN.sub("", text)

    # Remove @mentions
    text = MENTION_PATTERN.sub("", text)

    # Extract hashtag text (remove # but keep the word)
    text = HASHTAG_PATTERN.sub(r"\1", text)

    # Remove special characters and numbers
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)

    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from text."""
    words = text.split()
    return " ".join(w for w in words if w not in STOP_WORDS)


def lemmatize_text(text: str) -> str:
    """Apply lemmatization to text."""
    words = text.split()
    return " ".join(LEMMATIZER.lemmatize(w) for w in words)


def preprocess_text(
    text: str,
    do_clean: bool = True,
    do_remove_stopwords: bool = True,
    do_lemmatize: bool = True,
) -> str:
    """
    Full preprocessing pipeline for a single text.

    Args:
        text: Raw input text.
        do_clean: Apply text cleaning.
        do_remove_stopwords: Remove stopwords.
        do_lemmatize: Apply lemmatization.

    Returns:
        Preprocessed text string.
    """
    if do_clean:
        text = clean_text(text)
    if do_remove_stopwords:
        text = remove_stopwords(text)
    if do_lemmatize:
        text = lemmatize_text(text)
    return text


def preprocess_dataframe(df, text_col: str = "text", **kwargs):
    """
    Preprocess a DataFrame's text column.

    Args:
        df: Input DataFrame.
        text_col: Name of the text column.
        **kwargs: Arguments passed to preprocess_text.

    Returns:
        DataFrame with added 'clean_text' column.
    """
    import pandas as pd
    from tqdm import tqdm

    tqdm.pandas(desc="Preprocessing text")

    df = df.copy()
    df["clean_text"] = df[text_col].progress_apply(
        lambda x: preprocess_text(str(x), **kwargs)
    )

    # Remove empty texts after cleaning
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} empty texts after preprocessing")

    return df


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "BREAKING: The president @POTUS signed a new bill https://t.co/example #politics",
        "<p>Congress votes on <b>healthcare</b> reform TODAY!</p>",
        "Area Man Discovers Politicians Don't Actually Read Bills They Vote On 😂",
    ]

    for text in test_texts:
        print(f"Original:     {text}")
        print(f"Preprocessed: {preprocess_text(text)}")
        print()
