"""
Cluster interpretation and anomaly candidate flagging.

Builds per‑cluster profiles from the *original* (pre‑scaling) feature
values so that statistics remain human‑readable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Profile construction
# ──────────────────────────────────────────────
def build_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute mean, median, and count for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Original (cleaned, **not** scaled) feature data.
    labels : np.ndarray
        Cluster label per row.
    features : list[str] or None
        Subset of columns.  Defaults to all numeric columns.

    Returns
    -------
    pd.DataFrame
        Multi‑index DataFrame with ``(cluster, statistic)`` rows.
    """
    tmp = df.copy()
    if features:
        tmp = tmp[features]
    tmp = tmp.select_dtypes(include="number")
    tmp["Cluster"] = labels

    mean = tmp.groupby("Cluster").mean()
    mean.index = pd.MultiIndex.from_tuples(
        [(c, "mean") for c in mean.index], names=["Cluster", "Stat"]
    )

    median = tmp.groupby("Cluster").median()
    median.index = pd.MultiIndex.from_tuples(
        [(c, "median") for c in median.index], names=["Cluster", "Stat"]
    )

    count = tmp.groupby("Cluster").size()
    count_df = pd.DataFrame(
        {col: count for col in mean.columns},
        index=pd.MultiIndex.from_tuples(
            [(c, "count") for c in count.index], names=["Cluster", "Stat"]
        ),
    )

    return pd.concat([mean, median, count_df]).sort_index()


def get_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Simplified mean‑only profile table used in the Result page."""
    tmp = df.copy()
    if features:
        tmp = tmp[features]
    tmp = tmp.select_dtypes(include="number")
    tmp["Cluster"] = labels
    return tmp.groupby("Cluster").mean()


# ──────────────────────────────────────────────
# Dominant features
# ──────────────────────────────────────────────
def identify_dominant_features(
    summary: pd.DataFrame,
    top_n: int = 5,
) -> dict[int, list[str]]:
    """For each cluster, find the *top_n* features with the highest
    z‑score relative to the overall mean.

    Parameters
    ----------
    summary : pd.DataFrame
        Mean profile table (clusters × features).
    top_n : int
        Number of dominant features per cluster.

    Returns
    -------
    dict[int, list[str]]
        ``{cluster_id: [feature_name, ...]}``.
    """
    overall = summary.mean()
    overall_std = summary.std().replace(0, 1)  # avoid division by zero
    z = (summary - overall) / overall_std

    dominant: dict[int, list[str]] = {}
    for cluster in z.index:
        row = z.loc[cluster].abs().sort_values(ascending=False)
        dominant[int(cluster)] = row.head(top_n).index.tolist()
    return dominant


# ──────────────────────────────────────────────
# Narrative interpretation
# ──────────────────────────────────────────────
def generate_interpretations(
    summary: pd.DataFrame,
    cluster_counts: pd.Series,
) -> dict[int, str]:
    """Auto‑generate a short narrative for each cluster.

    The heuristic inspects relative feature magnitudes to assign
    a plain‑language description.
    """
    overall = summary.mean()
    interpretations: dict[int, str] = {}

    for cluster in summary.index:
        row = summary.loc[cluster]
        parts: list[str] = []
        count = int(cluster_counts.get(cluster, 0))
        parts.append(f"Cluster {cluster} ({count:,} flows)")

        # Duration
        if "Flow Duration" in row.index:
            dur_ratio = row["Flow Duration"] / max(overall["Flow Duration"], 1e-9)
            if dur_ratio < 0.3:
                parts.append("koneksi sangat singkat")
            elif dur_ratio < 0.7:
                parts.append("durasi pendek–menengah")
            elif dur_ratio > 2.0:
                parts.append("durasi sangat panjang")
            else:
                parts.append("durasi moderat")

        # Volume
        vol_cols = [c for c in ["Total Length of Fwd Packets", "Average Packet Size"] if c in row.index]
        if vol_cols:
            avg_vol = np.mean([row[c] / max(overall[c], 1e-9) for c in vol_cols])
            if avg_vol < 0.3:
                parts.append("volume paket rendah")
            elif avg_vol > 2.0:
                parts.append("volume paket sangat tinggi")
            elif avg_vol > 1.3:
                parts.append("volume paket cukup besar")

        # Packet rate
        rate_cols = [c for c in ["Flow Packets/s", "Fwd Packets/s"] if c in row.index]
        if rate_cols:
            avg_rate = np.mean([row[c] / max(overall[c], 1e-9) for c in rate_cols])
            if avg_rate > 2.0:
                parts.append("packet rate sangat tinggi (burst‑like)")
            elif avg_rate < 0.3:
                parts.append("packet rate rendah")

        # Flags
        if "FIN Flag Count" in row.index and row["FIN Flag Count"] > overall["FIN Flag Count"] * 1.5:
            parts.append("banyak koneksi FIN (connection teardown)")
        if "PSH Flag Count" in row.index and row["PSH Flag Count"] > overall["PSH Flag Count"] * 1.5:
            parts.append("frekuensi PUSH tinggi")

        interpretations[int(cluster)] = " — ".join(parts) + "."

    return interpretations


# ──────────────────────────────────────────────
# Anomaly candidate flagging
# ──────────────────────────────────────────────
def flag_anomaly_candidates(
    summary: pd.DataFrame,
    cluster_counts: pd.Series,
    size_threshold: float = 0.05,
    deviation_threshold: float = 2.0,
) -> list[dict]:
    """Flag clusters that are potential anomaly candidates.

    A cluster is flagged when **any** of these hold:
    - Its size is below *size_threshold* fraction of total flows.
    - Its mean z‑score (absolute) for any feature exceeds
      *deviation_threshold*.

    Returns
    -------
    list[dict]
        Each dict has ``cluster``, ``reasons``, ``severity``.
    """
    overall_mean = summary.mean()
    overall_std = summary.std().replace(0, 1)
    total = cluster_counts.sum()

    flags: list[dict] = []
    for cluster in summary.index:
        reasons: list[str] = []
        z = ((summary.loc[cluster] - overall_mean) / overall_std).abs()
        high_z_features = z[z > deviation_threshold].index.tolist()

        if high_z_features:
            reasons.append(
                f"Deviasi tinggi pada fitur: {', '.join(high_z_features[:5])}"
            )

        frac = cluster_counts.get(cluster, 0) / total
        if frac < size_threshold:
            reasons.append(
                f"Ukuran cluster sangat kecil ({frac:.1%} dari total flows)"
            )

        if reasons:
            severity = "HIGH" if len(reasons) > 1 else "MEDIUM"
            flags.append({
                "cluster": int(cluster),
                "reasons": reasons,
                "severity": severity,
            })

    return flags
