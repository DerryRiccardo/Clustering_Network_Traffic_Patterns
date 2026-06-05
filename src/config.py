"""
Centralized configuration for the Clustering Network Traffic project.

All paths, feature lists, hyperparameters, and constants used across
the pipeline are defined here so every module draws from a single
source of truth.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "cicids2017_sample_100000.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
LABEL_COLUMN = "Attack Type"

# 24 features selected based on domain knowledge, variance,
# correlation analysis, and interpretability (per AGENTS.md).
SELECTED_FEATURES: list[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Average Packet Size",
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Idle Mean",
]

# ──────────────────────────────────────────────
# Model / Training
# ──────────────────────────────────────────────
K_RANGE = range(2, 9)          # k = 2 … 8
RANDOM_STATE = 42
DEFAULT_K = 4                  # initial target for interpretation

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
SKEW_THRESHOLD = 2.0           # features with |skewness| > threshold get log‑transformed

# ──────────────────────────────────────────────
# Visualisation defaults
# ──────────────────────────────────────────────
PCA_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = 200
TSNE_MAX_ITER = 1000
