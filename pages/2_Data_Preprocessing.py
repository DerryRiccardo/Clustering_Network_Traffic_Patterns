"""
 Page 2 — Data Preprocessing
================================
Interactive data transformation: feature selection,
cleaning, log transform, and standardization with before/after comparisons.
"""

import streamlit as st

st.set_page_config(page_title="Data Preprocessing", page_icon="", layout="wide")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SELECTED_FEATURES
from src.data_loader import load_data, separate_labels
from src.preprocessing import (
    apply_log_transform,
    clean_data,
    get_log_shifts,
    get_skewed_features,
    scale_data,
    select_features,
)
from src.visualization import plot_before_after_scaling


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

    .step-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: #e2e8f0;
    }
    .step-card h4 { color: #38bdf8; margin-top: 0; }
    .step-card p { color: #cbd5e1; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Raw Data
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat raw data …")
def _load_raw():
    raw = load_data()
    features_df, labels = separate_labels(raw)
    return raw, features_df, labels

try:
    raw_df, features_df, labels = _load_raw()
except Exception as exc:
    st.error(f" Gagal memuat data: {exc}")
    st.stop()

numeric_cols = features_df.select_dtypes(include="number").columns.tolist()

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2> Data Preprocessing</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Di halaman ini, Anda dapat bereksperimen dengan berbagai teknik pra-pemrosesan data. "
    "Setiap perubahan yang Anda buat di bawah ini akan **otomatis tersimpan** dan langsung divisualisasikan "
    "untuk melihat dampaknya."
)


# ──────────────────────────────────────────────
# Step 1: Feature Selection
# ──────────────────────────────────────────────
st.markdown("### Step 1 · Feature Selection")
st.markdown("""
<div class="step-card">
    <b> Kenapa fitur ini yang dipilih?</b><br>
    Sistem secara otomatis memilih <b>24 fitur terbaik</b> berdasarkan prinsip berikut:<br>
    1. <b>Menghapus Redundansi:</b> Fitur kembar yang berkorelasi tinggi dibuang salah satunya agar model tidak bias.<br>
    2. <b>Fokus pada Perilaku:</b> Mengutamakan metrik durasi, kecepatan (<b>Packets/s</b>), dan sinyal (<b>FIN/PSH Flag</b>) untuk membaca tingkah laku trafik.<br>
    3. <b>Konteks Layanan:</b> <b>Destination Port</b> dipertahankan untuk mengidentifikasi jenis layanan (HTTP, SSH, dll).<br>
    4. <b>Abaikan Identitas:</b> IP dan Timestamp dibuang agar model murni belajar pola serangan, bukan menghafal pelaku.
</div>
""", unsafe_allow_html=True)

with st.expander(" Ubah Pemilihan Fitur (52 Kolom Tersedia)", expanded=False):
    st.markdown("Centang fitur mana saja yang ingin Anda libatkan dalam clustering:")
    cols = st.columns(3)
    selected_features = []
    
    # Retrieve previous selection if exists, otherwise use config defaults
    if "selected_features" in st.session_state:
        current_selection = st.session_state["selected_features"]
    else:
        current_selection = SELECTED_FEATURES

    for i, col_name in enumerate(numeric_cols):
        is_selected_by_default = col_name in current_selection
        if cols[i % 3].checkbox(col_name, value=is_selected_by_default, key=f"feat_chk_{i}"):
            selected_features.append(col_name)

if len(selected_features) < 2:
    st.error(" Tolong centang minimal 2 fitur untuk bisa dilakukan clustering!")
    st.stop()

# -- Proses Step 1
selected_df = select_features(features_df, selected_features)
rows_before = len(selected_df)

st.markdown(f"""
<div class="step-card">
    <h4> Menggunakan {len(selected_features)} Fitur</h4>
    <p>Data saat ini telah disaring sehingga hanya menyisakan {len(selected_features)} kolom (fitur) yang akan diteruskan ke tahap pembersihan data.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Step 2: Data Cleaning
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 2 · Data Cleaning (Menghapus Baris Duplikat)")
st.markdown("""
Setelah membuang fitur unik seperti IP dan Waktu di Step 1, sangat mungkin muncul koneksi jaringan yang nilainya 100% kembar persis (*Duplicate Rows*).
Baris-baris duplikat ini adalah sampah data (biasanya akibat *error* perekaman alat *sniffer*) yang harus dibuang agar algoritma K-Means tidak berat sebelah (bias) saat menentukan titik tengah *cluster*.

*Sistem secara otomatis membersihkan baris duplikat ini di latar belakang.*
""")

# -- Proses Step 2
# Karena missing_strategy dihilangkan dari UI, kita set default ke "Drop Rows" (atau apapun, toh NaNs = 0)
cleaned_df, cleaned_labels = clean_data(selected_df, labels, missing_strategy="Drop Rows")
rows_after = len(cleaned_df)
removed = rows_before - rows_after

st.markdown(f"""
<div class="step-card">
    <h4> Menghapus Baris Duplikat</h4>
    <p>Baris yang isinya kembar persis 100% telah dihapus secara permanen dari tabel.</p>
</div>
""", unsafe_allow_html=True)

col_b, col_a, col_d = st.columns(3)
with col_b:
    st.metric("Total Rows Awal", f"{rows_before:,}")
with col_a:
    st.metric("Total Rows Bersih", f"{rows_after:,}")
with col_d:
    pct = (removed / rows_before * 100) if rows_before > 0 else 0
    st.metric("Rows Dihapus (Duplikat/Invalid)", f"{removed:,}", delta=f"-{pct:.2f}%", delta_color="inverse")


# ──────────────────────────────────────────────
# Step 3: Log Transform
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 3 · Skewness / Outlier Treatment")
st.markdown(
    "Seperti yang didapatkan pada halaman *Exploratory Data Analysis*, mayoritas fitur lalu lintas jaringan memiliki "
    "distribusi yang sangat miring (*Highly Skewed*) karena adanya *outlier* alami (seperti koneksi burst besar "
    "yang jarang terjadi namun angkanya raksasa)."
)

# Tentukan state log transform sebelumnya
if "preprocessing_config" in st.session_state:
    default_log = st.session_state["preprocessing_config"]["log"]
else:
    default_log = True

apply_log = st.checkbox("Terapkan Transformasi Logaritmik (Log1p) pada fitur yang sangat miring", value=default_log)

# -- Proses Step 3
skewed = get_skewed_features(cleaned_df)
log_shifts = get_log_shifts(cleaned_df, skewed)

if apply_log and skewed:
    transformed_df = apply_log_transform(cleaned_df, skewed, shifts=log_shifts)
    st.markdown(f"""
    <div class="step-card">
        <h4> Transformasi Logaritmik Aktif</h4>
        <p>Sistem mendeteksi ada <strong>{len(skewed)} fitur</strong> dengan kemiringan ekstrem. Jarak angkanya telah 
        dirapatkan secara matematis (menggunakan fungsi log1p) agar grafiknya mendekati kurva normal lonceng.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    transformed_df = cleaned_df.copy()
    st.markdown("""
    <div class="step-card" style="border-color:#f59e0b;">
        <h4 style="color:#f59e0b;"> Transformasi Logaritmik Dinonaktifkan</h4>
        <p>Anda membiarkan <i>outlier</i> tetap berwujud ekstrem. Hati-hati, model K-Means sangat gampang "tertipu" oleh 
        angka <i>outlier</i> yang terlalu raksasa, sehingga centroid-nya nanti bisa sangat berantakan (bias)!</p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Step 4: Scaling (Normalisasi)
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 4 · Scaling / Normalisasi Skala")
st.markdown("Algoritma K-Means mengelompokkan data berdasarkan **Jarak Euclidean**. Agar kolom berskala jutaan tidak menindas kolom berskala belasan, kita wajib menyamarkan skala mereka.")

# Tentukan state scaler sebelumnya
if "preprocessing_config" in st.session_state:
    default_scaler = st.session_state["preprocessing_config"]["scaler"]
else:
    default_scaler = "StandardScaler"

scaler_options = ["StandardScaler", "MinMaxScaler", "RobustScaler", "Tanpa Scaling"]
try:
    default_idx = scaler_options.index(default_scaler)
except ValueError:
    default_idx = 0

scaler_type = st.selectbox(
    "Pilih algoritma normalisasi (Scaling):",
    options=scaler_options,
    index=default_idx
)

# Kumpulan Penjelasan Dinamis Step 4
scaler_explanations = {
    "StandardScaler": "Mengubah data sehingga rata-ratanya 0 dan standar deviasinya 1. Sangat umum digunakan dan bagus jika data sudah mendekati distribusi normal (terutama jika Log Transform aktif).",
    "MinMaxScaler": "Memampatkan seluruh angka ke dalam rentang batas pasti, yaitu 0 hingga 1. Cocok jika Anda butuh perbandingan batas atas/bawah yang mutlak.",
    "RobustScaler": "Menggunakan nilai kuartil. Scaler ini adalah senjata ampuh jika Anda memiliki banyak outlier liar namun enggan menggunakan metode Log Transform.",
    "Tanpa Scaling": "Membiarkan data mentah apa adanya. Sangat tidak disarankan untuk K-Means karena fitur dengan rentang nilai raksasa akan sepenuhnya mendominasi penentuan cluster."
}

# -- Proses Step 4
scaled_data, scaler, feature_names = scale_data(transformed_df, scaler_type)

st.markdown(f"""
<div class="step-card">
    <h4> Skala Disamakan menggunakan: <code>{scaler_type}</code></h4>
    <p> <i><b>Mengapa metode ini?</b> {scaler_explanations[scaler_type]}</i></p>
</div>
""", unsafe_allow_html=True)

col_bf, col_af = st.columns(2)
with col_bf:
    st.markdown("**Statistik Sebelum Scaling (Mentah)**")
    before_stats = transformed_df.describe().loc[["mean", "std", "min", "max"]].T
    st.dataframe(before_stats.style.format("{:.2f}"), width="stretch")

with col_af:
    st.markdown("**Statistik Sesudah Scaling (Normalisasi)**")
    scaled_df = pd.DataFrame(scaled_data, columns=feature_names)
    after_stats = scaled_df.describe().loc[["mean", "std", "min", "max"]].T
    st.dataframe(after_stats.style.format("{:.4f}"), width="stretch")


# ──────────────────────────────────────────────
# Visual Comparison & Persistence
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("###  Perbandingan Visual: Before vs After")
st.markdown("Buktikan sendiri perubahannya! Geser *slider* di bawah ini untuk melihat bagaimana distribusi bentuk data berubah total setelah melewati filter *Log Transform* dan *Scaling* Anda di atas.")

if "user_pp_n_feat" not in st.session_state:
    st.session_state["user_pp_n_feat"] = min(4, len(feature_names))

def _update_n_feat():
    st.session_state["user_pp_n_feat"] = st.session_state["pp_n_feat_w"]

n_show = st.slider("Jumlah fitur yang divisualisasikan:", min_value=2, max_value=min(8, len(feature_names)), value=st.session_state["user_pp_n_feat"], key="pp_n_feat_w", on_change=_update_n_feat)
fig = plot_before_after_scaling(
    before=transformed_df,
    after=scaled_data,
    feature_names=feature_names,
    n_features=n_show,
    scaler_name=scaler_type
)
st.pyplot(fig)

st.markdown("###  Preview Final Data")
st.dataframe(scaled_df.head(20), width="stretch", height=350)

# Simpan ke Session State (Auto-save)
st.session_state["raw_data"] = raw_df
st.session_state["attack_labels"] = cleaned_labels
st.session_state["selected_features"] = feature_names
st.session_state["clean_data"] = transformed_df
st.session_state["scaled_data"] = scaled_data
st.session_state["scaler"] = scaler

# Pseudo pp_result for page 3 compatibility
st.session_state["pp_result"] = {
    "feature_names": feature_names,
    "log_transformed": transformed_df,
    "scaled_data": scaled_data,
    "scaler": scaler,
    "rows_before": rows_before,
    "rows_after": rows_after,
    "skewed_features": skewed
}

st.session_state["preprocessing_config"] = {
    "missing": "Drop Rows",
    "log": apply_log,
    "scaler": scaler_type
}

# Jika pengguna mengganti config, bersihkan model lama agar tidak bertabrakan datanya
if "all_models" in st.session_state:
    del st.session_state["all_models"]
    if "kmeans_model" in st.session_state:
        del st.session_state["kmeans_model"]


