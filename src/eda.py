"""
Exploratory Data Analysis helpers.

Provides summary statistics and data‑quality information that the
EDA Streamlit page renders into tables and charts.
"""

from __future__ import annotations

import pandas as pd


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics (count, mean, std, min, quartiles, max)."""
    return df.describe().T


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    return df.select_dtypes(include="number").corr()


def get_skewness_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with skewness and a human‑readable verdict.

    Columns: ``skewness``, ``abs_skewness``, ``verdict``.
    """
    skew = df.skew(numeric_only=True)
    info = pd.DataFrame({
        "skewness": skew,
        "abs_skewness": skew.abs(),
    })

    def _verdict(val: float) -> str:
        a = abs(val)
        if a < 0.5:
            return "Approx. Symmetric"
        if a < 1.0:
            return "Moderate Skew"
        if a < 2.0:
            return "High Skew"
        return "Very High Skew"

    info["verdict"] = info["skewness"].apply(_verdict)
    return info.sort_values("abs_skewness", ascending=False)


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing / infinite values per column.

    Columns: ``missing_count``, ``missing_pct``, ``inf_count``.
    """
    import numpy as np

    missing = df.isnull().sum()
    inf_count = df.isin([np.inf, -np.inf]).sum()
    total = len(df)
    info = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": (missing / total * 100).round(2),
        "inf_count": inf_count,
    })
    return info[info["missing_count"] + info["inf_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def get_outlier_info(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers per column using the IQR method.

    Returns a DataFrame with ``q1``, ``q3``, ``iqr``, ``lower``,
    ``upper``, ``outlier_count``, and ``outlier_pct``.
    """
    numeric = df.select_dtypes(include="number")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = ((numeric < lower) | (numeric > upper)).sum()
    total = len(numeric)

    return pd.DataFrame({
        "q1": q1.round(2),
        "q3": q3.round(2),
        "iqr": iqr.round(2),
        "lower_bound": lower.round(2),
        "upper_bound": upper.round(2),
        "outlier_count": outlier_count,
        "outlier_pct": (outlier_count / total * 100).round(2),
    }).sort_values("outlier_pct", ascending=False)
