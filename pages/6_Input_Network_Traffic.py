"""
Page 7 - Input Traffic Prediction
=================================
Lets users enter one network-flow record and assigns it to a K-Means cluster.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Input Network Traffic", page_icon="", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_K, RANDOM_STATE, SELECTED_FEATURES
from src.data_loader import load_data, separate_labels
from src.interpretation import (
    flag_anomaly_candidates,
    generate_interpretations,
    get_cluster_summary,
    identify_dominant_features,
)
from src.prediction import predict_flow_cluster
from src.preprocessing import run_preprocessing_pipeline
from src.training import train_kmeans


st.markdown(
    """
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

    .result-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        color: #e2e8f0;
    }
    .result-card h3 { color: #38bdf8; margin-top: 0; }
    .result-card p, .result-card li { color: #cbd5e1; line-height: 1.6; }
    .feature-badge {
        display: inline-block;
        background: #0c4a6e;
        color: #e0f2fe;
        padding: 0.25rem 0.6rem;
        border-radius: 16px;
        font-size: 0.78rem;
        margin: 0.15rem;
    }
    .anomaly-flag {
        background: linear-gradient(145deg, #7f1d1d, #991b1b);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .anomaly-flag h3 { color: #fca5a5; margin-top: 0; }
    .anomaly-flag p, .anomaly-flag li { color: #fecaca; line-height: 1.6; }

    .anomaly-flag-medium {
        background: linear-gradient(145deg, #78350f, #92400e);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .anomaly-flag-medium h3 { color: #fde68a; margin-top: 0; }
    .anomaly-flag-medium p, .anomaly-flag-medium li { color: #fef3c7; line-height: 1.6; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Memuat dan memproses dataset ...")
def _prepare():
    raw = load_data()
    features_df, labels = separate_labels(raw)
    pp = run_preprocessing_pipeline(features_df, labels=labels)
    return raw, pp


try:
    raw_df, pp = _prepare()
except Exception as exc:
    st.error(f"Gagal memuat pipeline: {exc}")
    st.stop()

scaled_data = pp["scaled_data"]
clean_data = pp["cleaned"]
feature_names = pp["feature_names"]

selected_k = int(st.session_state.get("selected_k", DEFAULT_K))

if (
    st.session_state.get("kmeans_model") is not None
    and st.session_state.get("selected_k", selected_k) == selected_k
):
    model = st.session_state["kmeans_model"]
else:
    with st.spinner(f"Training K-Means k={selected_k} untuk prediksi input ..."):
        model = train_kmeans(scaled_data, selected_k, RANDOM_STATE)
    st.session_state["kmeans_model"] = model
    st.session_state["selected_k"] = selected_k

cluster_labels = model.labels_
summary = get_cluster_summary(clean_data, cluster_labels, feature_names)
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
interpretations = generate_interpretations(summary, cluster_counts)
dominant = identify_dominant_features(summary, top_n=5)
flags = flag_anomaly_candidates(summary, cluster_counts)
flag_by_cluster = {item["cluster"]: item for item in flags}

defaults = clean_data[feature_names].median(numeric_only=True)
mins = clean_data[feature_names].min(numeric_only=True)
maxs = clean_data[feature_names].max(numeric_only=True)


st.markdown(
    """
<div class="section-header">
    <h2>Input Network Traffic</h2>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "Masukkan satu data network flow, lalu sistem akan menentukan flow tersebut "
    "paling dekat dengan cluster K-Means yang mana. Output ini adalah hasil "
    "behavioral profiling, bukan label serangan pasti."
)

st.markdown("""
<div class="result-card" style="border-left: 4px solid #10b981; margin-bottom: 1.5rem;">
    <b> Panduan Pengujian Sistem:</b><br>
    Halaman ini mendemonstrasikan bagaimana model K-Means melakukan <i>profiling</i> terhadap satu rekaman data trafik jaringan baru secara <i>real-time</i>.<br>
    Secara default, formulir telah diisi dengan nilai Median (representasi trafik normal). Anda dapat mengubah angka-angka ini secara manual atau menggunakan tombol <b>Skenario Pengujian Cepat</b> di bawah ini untuk melihat bagaimana model K-Means merespons berbagai profil trafik jaringan yang ekstrem.
</div>
""", unsafe_allow_html=True)

st.markdown("#### Skenario Pengujian Cepat")
col_s1, col_s2, col_s3 = st.columns(3)

def load_preset(cluster_id: int):
    if cluster_id == -1:
        for f in feature_names:
            st.session_state[f"traffic_input_{f}"] = float(defaults[f])
    else:
        for f in feature_names:
            st.session_state[f"traffic_input_{f}"] = float(summary.loc[cluster_id, f])

with col_s1:
    if st.button(" Muat Profil Normal (Median Dataset)", width="stretch", key="btn_preset_normal"):
        load_preset(-1)

with col_s2:
    if st.button(" Muat Profil Rata-rata Cluster 1", width="stretch", key="btn_preset_c1"):
        load_preset(1)

with col_s3:
    last_c = summary.index[-1]
    if st.button(f" Muat Profil Rata-rata Cluster {last_c}", width="stretch", key="btn_preset_last"):
        load_preset(last_c)

st.markdown("---")
st.markdown("### Form Input Traffic")

with st.form("traffic_input_form"):
    input_values = {}
    col_a, col_b, col_c = st.columns(3)
    columns = [col_a, col_b, col_c]

    for idx, feature in enumerate(feature_names):
        key = f"traffic_input_{feature}"
        if key not in st.session_state:
            st.session_state[key] = float(defaults[feature])
        with columns[idx % 3]:
            input_values[feature] = st.number_input(
                feature,
                step=1.0,
                format="%.6f",
                key=key,
            )

    submitted = st.form_submit_button("Prediksi Cluster", type="primary", width="stretch")

if submitted:
    input_df = pd.DataFrame([input_values], columns=feature_names)

    try:
        result = predict_flow_cluster(
            input_df=input_df,
            model=model,
            scaler=pp["scaler"],
            feature_names=feature_names,
            skewed_features=pp["skewed_features"],
            log_shifts=pp["log_shifts"],
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    predicted_cluster = int(result["cluster_labels"][0])
    nearest_distance = float(result["nearest_distances"][0])
    all_distances = result["distances"][0]
    cluster_size = int(cluster_counts.get(predicted_cluster, 0))
    cluster_pct = cluster_size / len(cluster_labels) * 100

    st.markdown("### Output Sistem")
    col_result, col_context = st.columns([1, 2])

    with col_result:
        st.markdown(
            f"""
            <div class="result-card">
                <h3>Cluster {predicted_cluster}</h3>
                <p><strong>Jarak ke centroid:</strong> {nearest_distance:.4f}</p>
                <p><strong>Ukuran cluster:</strong> {cluster_size:,} flows ({cluster_pct:.2f}%)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if predicted_cluster in flag_by_cluster:
            flag = flag_by_cluster[predicted_cluster]
            css_class = "anomaly-flag" if flag["severity"] == "HIGH" else "anomaly-flag-medium"
            reasons = "".join(f"<li>{reason}</li>" for reason in flag["reasons"])
            st.markdown(
                f"""
                <div class="{css_class}">
                    <h3>Kandidat Anomali: {flag['severity']}</h3>
                    <ul>{reasons}</ul>
                    <p>Cluster ini layak dimonitor dan memerlukan investigasi lebih lanjut.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Cluster ini tidak ditandai sebagai kandidat anomali berdasarkan threshold default.")

    with col_context:
        st.markdown(
            f"""
            <div class="result-card">
                <h3>Interpretasi Cluster</h3>
                <p>{interpretations.get(predicted_cluster, "Belum ada interpretasi untuk cluster ini.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        badges = " ".join(
            f'<span class="feature-badge">{feature}</span>'
            for feature in dominant.get(predicted_cluster, [])
        )
        st.markdown(
            f"""
            <div class="result-card">
                <h3>Fitur Dominan Cluster</h3>
                <p>{badges}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Detail Jarak ke Semua Centroid")
    distance_df = pd.DataFrame(
        {
            "Cluster": list(range(len(all_distances))),
            "Distance to Centroid": all_distances,
        }
    ).sort_values("Distance to Centroid")
    st.dataframe(
        distance_df.style.format({"Distance to Centroid": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Data input yang diproses"):
        st.dataframe(input_df, width="stretch", hide_index=True)

st.markdown("---")
st.caption(
    "Catatan: prediksi ini memakai scaler, log transform, dan model K-Means dari pipeline training. "
    "Cluster bukan diagnosis serangan, melainkan profil perilaku trafik."
)
