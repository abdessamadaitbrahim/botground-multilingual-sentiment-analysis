# Preprocessor — Clean and prepare text for the model
import re
import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def clean_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 0]
    print(f"Text cleaned : {len(df)} rows remaining")
    return df


def validate_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if len(text.strip()) == 0:
        return False
    return True