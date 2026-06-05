"""
Visualisation helpers.

All plotting functions return a Matplotlib ``Figure`` or a Plotly
``Figure`` so the Streamlit pages can render them with
``st.pyplot`` / ``st.plotly_chart`` without importing plotting
libraries themselves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import PCA_COMPONENTS, TSNE_LEARNING_RATE, TSNE_MAX_ITER, TSNE_PERPLEXITY


# ──────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────
PALETTE = [
    "#6366f1",  # indigo
    "#f43f5e",  # rose
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#14b8a6",  # teal
]


# ──────────────────────────────────────────────
# EDA plots
# ──────────────────────────────────────────────
def plot_histogram(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Histogram for a single feature."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[feature].dropna(), bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_title(f"Distribution of {feature}", fontsize=13, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_boxplot(df: pd.DataFrame, features: list[str]) -> plt.Figure:
    """Boxplot for selected features (max 8 for readability)."""
    features = features[:8]
    fig, ax = plt.subplots(figsize=(10, 5))
    data_to_plot = [df[f].dropna().values for f in features]
    bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=features)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Boxplot of Selected Features", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> plt.Figure:
    """Seaborn heatmap for the correlation matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Preprocessing comparison
# ──────────────────────────────────────────────
def plot_before_after_scaling(
    before: pd.DataFrame,
    after: np.ndarray,
    feature_names: list[str],
    n_features: int = 6,
    scaler_name: str = "StandardScaler"
) -> plt.Figure:
    """Side‑by‑side histograms showing distributions before and after scaling."""
    cols = feature_names[:n_features]
    after_df = pd.DataFrame(after, columns=feature_names)

    fig, axes = plt.subplots(n_features, 2, figsize=(12, 3 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(cols):
        # Before
        axes[i, 0].hist(before[col].dropna(), bins=40, color=PALETTE[0], edgecolor="white", alpha=0.8)
        axes[i, 0].set_title(f"{col} — Before Scaling", fontsize=10)
        sns.despine(ax=axes[i, 0])

        # After
        axes[i, 1].hist(after_df[col].dropna(), bins=40, color=PALETTE[1], edgecolor="white", alpha=0.8)
        axes[i, 1].set_title(f"{col} — After Scaling", fontsize=10)
        sns.despine(ax=axes[i, 1])

    fig.suptitle(f"Before vs After {scaler_name}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────
def compute_pca(
    data: np.ndarray,
    n_components: int = PCA_COMPONENTS,
) -> tuple[np.ndarray, PCA]:
    """Fit PCA and return transformed data + fitted object."""
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(data)
    return transformed, pca


def compute_tsne(
    data: np.ndarray,
    perplexity: float = TSNE_PERPLEXITY,
    learning_rate: float = TSNE_LEARNING_RATE,
    max_iter: int = TSNE_MAX_ITER,
) -> np.ndarray:
    """Fit t‑SNE and return 2‑D transformed data."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42,
    )
    return tsne.fit_transform(data)


# ──────────────────────────────────────────────
# Cluster scatter plots (Plotly — interactive)
# ──────────────────────────────────────────────
def plot_pca_scatter(
    data: np.ndarray,
    labels: np.ndarray,
    pca_obj: PCA | None = None,
) -> go.Figure:
    """Interactive 2‑D PCA scatter coloured by cluster."""
    plot_df = pd.DataFrame(data[:, :2], columns=["PC1", "PC2"])
    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        color_discrete_sequence=PALETTE,
        title="PCA — Cluster Visualization",
        opacity=0.6,
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        legend_title_text="Cluster",
    )
    if pca_obj is not None:
        ev = pca_obj.explained_variance_ratio_
        fig.update_xaxes(title=f"PC1 ({ev[0]:.1%} variance)")
        fig.update_yaxes(title=f"PC2 ({ev[1]:.1%} variance)")
    return fig


def plot_tsne_scatter(
    data: np.ndarray,
    labels: np.ndarray,
) -> go.Figure:
    """Interactive 2‑D t‑SNE scatter coloured by cluster."""
    plot_df = pd.DataFrame(data[:, :2], columns=["t-SNE 1", "t-SNE 2"])
    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter(
        plot_df,
        x="t-SNE 1",
        y="t-SNE 2",
        color="Cluster",
        color_discrete_sequence=PALETTE,
        title="t-SNE — Cluster Visualization",
        opacity=0.6,
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        legend_title_text="Cluster",
    )
    return fig


# ──────────────────────────────────────────────
# Cluster distribution
# ──────────────────────────────────────────────
def plot_cluster_distribution(labels: np.ndarray) -> go.Figure:
    """Bar chart of flow counts per cluster."""
    counts = pd.Series(labels).value_counts().sort_index()
    fig = go.Figure(
        go.Bar(
            x=[f"Cluster {c}" for c in counts.index],
            y=counts.values,
            marker_color=PALETTE[: len(counts)],
            text=counts.values,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Cluster Distribution",
        xaxis_title="Cluster",
        yaxis_title="Number of Flows",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ──────────────────────────────────────────────
# Evaluation plots
# ──────────────────────────────────────────────
def plot_elbow_curve(eval_df: pd.DataFrame) -> go.Figure:
    """Line chart of inertia (elbow) across *k*."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eval_df["k"],
        y=eval_df["inertia"],
        mode="lines+markers",
        name="Inertia",
        line=dict(color=PALETTE[0], width=3),
        marker=dict(size=10),
    ))
    fig.update_layout(
        title="Elbow Method — Inertia vs k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_metrics_comparison(eval_df: pd.DataFrame) -> go.Figure:
    """Line charts for Silhouette, Davies‑Bouldin, and Calinski‑Harabasz."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Silhouette Score ↑", "Davies-Bouldin Index ↓", "Calinski-Harabasz Index ↑"],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=eval_df["k"], y=eval_df["silhouette"],
                   mode="lines+markers", line=dict(color=PALETTE[2], width=3),
                   marker=dict(size=9), name="Silhouette"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=eval_df["k"], y=eval_df["davies_bouldin"],
                   mode="lines+markers", line=dict(color=PALETTE[1], width=3),
                   marker=dict(size=9), name="Davies-Bouldin"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=eval_df["k"], y=eval_df["calinski_harabasz"],
                   mode="lines+markers", line=dict(color=PALETTE[4], width=3),
                   marker=dict(size=9), name="Calinski-Harabasz"),
        row=1, col=3,
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="k", row=1, col=i)

    fig.update_layout(
        height=380,
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        title_text="Clustering Metrics Across k Values",
    )
    return fig


# ──────────────────────────────────────────────
# Cluster profile heatmap
# ──────────────────────────────────────────────
def plot_cluster_profiles_heatmap(summary: pd.DataFrame) -> plt.Figure:
    """Heatmap of z‑scored cluster profiles (clusters × features)."""
    # z‑score each column so features are comparable
    z = (summary - summary.mean()) / summary.std().replace(0, 1)

    fig, ax = plt.subplots(figsize=(14, max(4, len(z) * 0.8)))
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        xticklabels=[c[:20] for c in z.columns],
    )
    ax.set_title("Cluster Feature Profiles (z-scored)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_attack_type_by_cluster(
    labels: np.ndarray,
    attack_labels: pd.Series,
) -> go.Figure:
    """Stacked bar chart: Attack Type distribution within each cluster."""
    tmp = pd.DataFrame({"Cluster": labels, "Attack Type": attack_labels})
    ct = tmp.groupby(["Cluster", "Attack Type"]).size().reset_index(name="Count")

    fig = px.bar(
        ct,
        x="Cluster",
        y="Count",
        color="Attack Type",
        title="Attack Type Distribution per Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        xaxis_title="Cluster",
        yaxis_title="Number of Flows",
        barmode="stack",
    )
    return fig
