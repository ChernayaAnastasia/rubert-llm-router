"""Data loading helpers: splits, combined text, org text for cross-encoder."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from utils.config import (
    COL_ADDRESS,
    COL_COMBINED_TEXT,
    COL_NAME,
    COL_ORG_TEXT,
    COL_PRICELIST,
    COL_QUERY,
    COL_RELEVANCE,
    COL_REVIEWS,
    COL_RUBRIC,
    RANDOM_STATE,
)


def _filter_uncertain(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with uncertain relevance (0.1)."""
    return df[df[COL_RELEVANCE] != 0.1]


def load_dataset(
    path: str,
    drop_uncertain: bool = True,
    val_frac: float = 0.2,
    test_size: int = 570,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load jsonl and split into train / val / test.

    test_size is fixed by the course project (570 rows).
    """
    data = pd.read_json(path, lines=True)
    data.columns = data.columns.str.lower()

    if COL_RELEVANCE not in data.columns:
        raise ValueError(f"Column {COL_RELEVANCE} not found in data")

    if len(data) < test_size:
        raise ValueError(
            f"Dataset must have at least {test_size} rows for the test split"
        )

    test_data = data.iloc[:test_size].copy()
    temp_train = data.iloc[test_size:].copy()

    if drop_uncertain:
        temp_train = _filter_uncertain(temp_train)
        test_data = _filter_uncertain(test_data)

    val_data = temp_train.sample(frac=val_frac, random_state=random_state)
    train_data = temp_train.drop(val_data.index)

    return (
        train_data.reset_index(drop=True),
        val_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


def build_combined_text(row: pd.Series) -> str:
    """
    Single string for TF-IDF baseline (EDA Exp C / Stage 1).
    COL_COMBINED_TEXT is not stored in parquet — build on the fly.
    """
    reviews = str(row.get(COL_REVIEWS, "")).strip()
    pricelist = str(row.get(COL_PRICELIST, "")).strip()
    return (
        f"Query: {row[COL_QUERY]}. "
        f"Address:{row[COL_ADDRESS]}."
        f"Name: {row[COL_NAME]}. "
        f"Rubric: {row[COL_RUBRIC]}. "
        f"Reviews: {reviews} "
        f"Pricelist: {pricelist}"
    )


def attach_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add COL_COMBINED_TEXT column (copy of input frame)."""
    out = df.copy()
    out[COL_COMBINED_TEXT] = df.apply(build_combined_text, axis=1)
    return out


def make_org_text(df: pd.DataFrame) -> pd.Series:
    """
    Sequence B for cross-encoder:
    COL_NAME | COL_ADDRESS | COL_RUBRIC | COL_REVIEWS | COL_PRICELIST
    """
    name = df[COL_NAME].fillna("")
    address = df[COL_ADDRESS].fillna("")
    rubric = df[COL_RUBRIC].fillna("")
    reviews = (
        df[COL_REVIEWS].fillna("")
        if COL_REVIEWS in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    pricelist = (
        df[COL_PRICELIST].fillna("")
        if COL_PRICELIST in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )

    return name + " | " + address + " | " + rubric + " | " + reviews + " | " + pricelist


def attach_org_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add COL_ORG_TEXT for cross-encoder (Stage 2, 4, 5)."""
    out = df.copy()
    out[COL_ORG_TEXT] = make_org_text(df)
    return out
