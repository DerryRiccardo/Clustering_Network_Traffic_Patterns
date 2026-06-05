"""
Data loading and validation utilities.

Reads the fixed sample dataset and performs basic schema validation
before handing data to the rest of the pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.config import LABEL_COLUMN, SAMPLE_DATA_PATH, SELECTED_FEATURES


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the sample CSV dataset.

    Parameters
    ----------
    path : str or None
        Override path.  Defaults to ``SAMPLE_DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all columns.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If required feature columns are missing.
    """
    from pathlib import Path
    data_path = Path(path or SAMPLE_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Validate that all selected features are present
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required feature columns: {missing}"
        )

    return df


def separate_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate the label column from the feature data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe that may contain the label column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None]
        (features dataframe without label, label series or None)
    """
    if LABEL_COLUMN in df.columns:
        labels = df[LABEL_COLUMN].copy()
        features = df.drop(columns=[LABEL_COLUMN])
        return features, labels
    return df.copy(), None
