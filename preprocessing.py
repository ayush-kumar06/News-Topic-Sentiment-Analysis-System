
import pandas as pd
import re


def clean_text(text):
    """Clean news text before machine learning."""

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert text to lowercase
    text = str(text).lower()

    # Remove website URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep only English letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_news(df):
    """Validate and preprocess the news dataset."""

    # Check required columns
    required_columns = {"headline", "article"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain headline and article columns."
        )

    # Remove rows with missing headlines or articles
    df = df.dropna(
        subset=["headline", "article"]
    ).copy()

    # Remove extra spaces around text
    df["headline"] = df["headline"].astype(str).str.strip()
    df["article"] = df["article"].astype(str).str.strip()

    # Remove empty rows
    df = df[
        (df["headline"] != "") &
        (df["article"] != "")
    ]

    # Remove duplicate articles
    df = df.drop_duplicates(
        subset=["headline", "article"]
    )

    # Combine headline and article
    df["clean_text"] = (
        df["headline"] + " " + df["article"]
    ).apply(clean_text)

    return df.reset_index(drop=True)
