"""
 Page 5 — Evaluation
========================
Compares clustering quality across multiple k values using
Silhouette, Davies‑Bouldin, and Calinski‑Harabasz metrics.
"""

import streamlit as st

st.set_page_config(page_title="Evaluation", page_icon="", layout="wide")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_STATE
from src.data_loader import load_data, separate_labels
from src.evaluation import evaluate_multiple_k, get_best_k
from src.preprocessing import run_preprocessing_pipeline
from src.training import train_multiple_k
from src.visualization import plot_elbow_curve, plot_metrics_comparison


# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .section-header {
        background: linear-gradient(135deg, #082f49, #0369a1);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .section-header h2 { color: white; font-size: 1.3rem; margin: 0; }

    .metric-explain {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #e2e8f0;
    }
    .metric-explain h4 { color: #38bdf8; margin-top: 0; }
    .metric-explain p { color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; }

    .best-k-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 24px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load data / models
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat data …")
def _prepare():
    raw = load_data()
    features_df, labels = separate_labels(raw)
    pp = run_preprocessing_pipeline(features_df, labels=labels)
    return raw, labels, pp


try:
    raw_df, labels, pp = _prepare()
except Exception as exc:
    st.error(f" {exc}")
    st.stop()

scaled_data = pp["scaled_data"]
st.session_state["scaled_data"] = scaled_data


# ──────────────────────────────────────────────
# Ensure models are trained
# ──────────────────────────────────────────────
if "all_models" not in st.session_state:
    st.warning("Model belum dilatih. Silakan kembali ke halaman Training atau klik tombol di bawah untuk melatih model.")
    if st.button("Latih Model (k=2-8)"):
        with st.spinner("Training model untuk evaluasi (k=2–8) …"):
            models = train_multiple_k(scaled_data, range(2, 9), RANDOM_STATE)
            st.session_state["all_models"] = models
            st.session_state["k_range"] = list(range(2, 9))
            st.rerun()
    else:
        st.stop()

models = st.session_state["all_models"]


# ──────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Menghitung metrik evaluasi …")
def _evaluate_metrics(scaled_data_val, models_keys):
    return evaluate_multiple_k(scaled_data_val, st.session_state["all_models"])

eval_df = _evaluate_metrics(scaled_data, tuple(sorted(models.keys())))
best_k = get_best_k(eval_df)

# Persist
st.session_state["evaluation_results"] = eval_df
st.session_state["best_k"] = best_k


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2> Model Evaluation — Comparing k Values</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Evaluasi kualitas clustering untuk beberapa nilai **k** menggunakan "
    "tiga metrik internal. Pemilihan k akhir mempertimbangkan metrik "
    "**dan** interpretasi domain."
)


# ──────────────────────────────────────────────
# Metrics explanation
# ──────────────────────────────────────────────
st.markdown("###  Memahami Metrik Evaluasi K-Means")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-explain">
        <h4> Silhouette Score</h4>
        <p>
            <b>(Metrik Utama)</b><br>
            Mengevaluasi dua aspek sekaligus: <i>Kekohesifan</i> (kepadatan di dalam satu kelompok) dan <i>Separasi</i> (jarak dengan kelompok lain). Metrik ini sangat diandalkan karena memberikan penilaian yang paling komprehensif.<br>
            <strong>↑ Mendekati 1 = Sangat Baik</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-explain">
        <h4> Davies-Bouldin</h4>
        <p>
            Menghitung rasio penyebaran data di dalam kelompok dibandingkan dengan jarak antar kelompok. Nilai yang rendah mengindikasikan bahwa kelompok terbentuk secara padat dan terpisah secara tegas.<br>
            <strong>↓ Mendekati 0 = Sangat Baik</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-explain">
        <h4> Calinski-Harabasz</h4>
        <p>
            Dikenal juga sebagai <i>Variance Ratio Criterion</i>. Metrik ini menilai seberapa terpusat data di dalam kelompoknya masing-masing. Skor yang meroket tinggi menunjukkan pemisahan kelompok yang sangat terstruktur.<br>
            <strong>↑ Makin Tinggi = Sangat Baik</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Metrics table
# ──────────────────────────────────────────────
st.markdown("###  Tabel Metrik per Nilai k")

display_df = eval_df.copy()
display_df.columns = ["k", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑", "Inertia"]

st.dataframe(
    display_df.style.format({
        "Silhouette ↑": "{:.4f}",
        "Davies-Bouldin ↓": "{:.4f}",
        "Calinski-Harabasz ↑": "{:,.2f}",
        "Inertia": "{:,.2f}",
    }).highlight_max(subset=["Silhouette ↑", "Calinski-Harabasz ↑"], props="background-color: #1e3a5f; color: #e2e8f0;")
      .highlight_min(subset=["Davies-Bouldin ↓"], props="background-color: #1e3a5f; color: #e2e8f0;"),
    width="stretch",
    hide_index=True,
)


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────
st.markdown("###  Visualisasi Metrik")

# Metrics comparison subplot
fig_metrics = plot_metrics_comparison(eval_df)
st.plotly_chart(fig_metrics, width="stretch")

# Elbow curve
st.markdown("###  Elbow Method (Inertia)")
fig_elbow = plot_elbow_curve(eval_df)
st.plotly_chart(fig_elbow, width="stretch")

st.markdown("""
<div style="background-color: #1e293b; padding: 1.2rem; border-radius: 8px; border-left: 4px solid #facc15; font-size: 0.95rem; color: #cbd5e1; margin-top: -1rem; margin-bottom: 2rem;">
    <b> Apa itu Elbow Method?</b><br>
    <i>Elbow Method</i> adalah teknik evaluasi heuristik yang memetakan nilai <b>Inertia</b> (<i>Within-Cluster Sum of Squares / WCSS</i>) terhadap jumlah cluster (<i>k</i>). Inertia mengukur seberapa dekat titik-titik data dengan pusat kelompoknya masing-masing. Secara alami, nilai Inertia akan selalu menurun ketika kita menambah jumlah <i>k</i>.<br><br>
    <b>Panduan Membaca Kurva:</b><br>
    Tujuan dari grafik ini bukanlah mencari nilai Inertia terendah di sisi kanan bawah. Alih-alih, carilah titik <b>"Siku" (Elbow)</b>—yaitu titik kritis di mana penurunan tajam pada grafik mulai melandai secara signifikan. <br><br>
    Titik siku tersebut merepresentasikan jumlah <i>k</i> yang paling optimal (titik <i>diminishing returns</i>). Menambah jumlah kelompok (<i>k</i>) melebihi titik siku tersebut tidak lagi memberikan peningkatan kepadatan (kekohesifan) yang sebanding dengan beban komputasi modelnya.
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Best k recommendation
# ──────────────────────────────────────────────
st.markdown("###  Rekomendasi Nilai k")

st.markdown(
    f'<span class="best-k-badge">k = {best_k} (berdasarkan Silhouette Score tertinggi)</span>',
    unsafe_allow_html=True,
)

best_row = eval_df[eval_df["k"] == best_k].iloc[0]
st.markdown(f"""
| Metrik | Skor |
|---|---|
| Silhouette Score | **{best_row['silhouette']:.4f}** |
| Davies-Bouldin Index | **{best_row['davies_bouldin']:.4f}** |
| Calinski-Harabasz Index | **{best_row['calinski_harabasz']:,.2f}** |
| Inertia | **{best_row['inertia']:,.2f}** |
""")

# Let user select k for Result page
st.markdown("###  Pilih k untuk Interpretasi")

if "user_selected_k" not in st.session_state:
    st.session_state["user_selected_k"] = best_k

def _update_selected_k():
    st.session_state["user_selected_k"] = st.session_state["eval_select_k_w"]

k_opts = sorted(models.keys())
try:
    def_idx = k_opts.index(st.session_state["user_selected_k"])
except ValueError:
    def_idx = k_opts.index(best_k)

selected_k = st.selectbox(
    "Nilai k yang akan digunakan di halaman Result:",
    k_opts,
    index=def_idx,
    key="eval_select_k_w",
    on_change=_update_selected_k,
)

st.session_state["selected_k"] = selected_k
st.session_state["kmeans_model"] = models[selected_k]
st.session_state["cluster_labels"] = models[selected_k].labels_

st.info(f" Model dengan **k = {selected_k}** akan digunakan pada halaman Result & Interpretation.")