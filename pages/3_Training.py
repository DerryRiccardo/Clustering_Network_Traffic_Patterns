"""
 Page 4 — Training
======================
K‑Means training interface with cluster count selection and progress tracking.
"""

import streamlit as st

st.set_page_config(page_title="Training", page_icon="", layout="wide")

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_K, K_RANGE, RANDOM_STATE, SELECTED_FEATURES
from src.data_loader import load_data, separate_labels
from src.preprocessing import run_preprocessing_pipeline
from src.training import train_kmeans, train_multiple_k


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

    .info-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        color: #e2e8f0;
    }
    .info-card h4 { color: #38bdf8; margin-top: 0; }
    .info-card p, .info-card li { color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; }

    .param-table {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
        color: #e2e8f0;
    }
    .param-table th { color: #38bdf8; text-align: left; padding: 0.4rem 1rem; }
    .param-table td { color: #e2e8f0; padding: 0.4rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Preprocessed Data (dari halaman sebelumnya)
# ──────────────────────────────────────────────
if "scaled_data" not in st.session_state or "pp_result" not in st.session_state:
    st.warning(" **Data Belum Diproses!**")
    st.info("Sistem mendeteksi data Anda masih kosong. Silakan buka halaman **2. Data Preprocessing** sebentar agar sistem dapat otomatis memproses data awal Anda, lalu kembali lagi ke halaman ini.")
    st.stop()

# Retrieve processed data from session state
scaled_data = st.session_state["scaled_data"]
pp = st.session_state["pp_result"]
conf = st.session_state.get("preprocessing_config", {"scaler": "Tidak diketahui", "missing": "Tidak diketahui", "log": False})


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2> Model Training</h2>
</div>
""", unsafe_allow_html=True)


st.markdown(
    "Pada tahap ini, kita melatih model K-Means clustering untuk mengelompokkan "
    "trafik berdasarkan kemiripan pola."
)

# ──────────────────────────────────────────────
# Education Expander
# ──────────────────────────────────────────────
with st.expander(" Apa itu K-Means & Cara Kerjanya?", expanded=False):
    st.markdown("""
    <div style="padding: 1rem; background-color: #1e293b; border-radius: 10px; border: 1px solid #334155;">
        <p>
            K-Means adalah algoritma <i>Unsupervised Learning</i> yang membagi data menjadi
            <strong>k</strong> kelompok (cluster) tanpa diberi tahu label/kunci jawabannya. 
        </p>
        <p><b>Cara kerjanya sangat sederhana:</b></p>
        <ol style="color:#cbd5e1; font-size:0.92rem;">
            <li><b>Lempar Jangkar:</b> Algoritma menebak acak letak titik pusat (centroid) untuk setiap kelompok.</li>
            <li><b>Tarik Data:</b> Setiap data trafik akan ditarik masuk ke kelompok yang jangkar (centroid) nya paling dekat dengannya (menggunakan jarak Euclidean).</li>
            <li><b>Geser Jangkar:</b> Setelah kelompok terbentuk, posisi jangkar akan digeser pas ke tengah-tengah kelompok tersebut.</li>
            <li><b>Ulangi:</b> Langkah 2 & 3 terus diulang sampai posisi jangkar sudah stabil tidak bergeser lagi.</li>
        </ol>
        <p style="margin-bottom:0;">
            Karena algoritma ini "buta" tidak tahu harus dibagi jadi berapa kelompok, kita menyuruhnya untuk bereksperimen menggunakan <b>Rentang k</b> (misal membagi jadi 2 sampai 8 kelompok sekaligus).
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Main Interface Layout
# ──────────────────────────────────────────────
# Persistent state for widgets
if "user_k_min" not in st.session_state:
    st.session_state["user_k_min"] = int(K_RANGE.start)
if "user_k_max" not in st.session_state:
    st.session_state["user_k_max"] = int(K_RANGE.stop - 1)

def _update_k_range():
    st.session_state["user_k_min"] = st.session_state["k_min_w"]
    st.session_state["user_k_max"] = st.session_state["k_max_w"]

col_action, col_info = st.columns([1, 1.5], gap="large")

with col_action:
    st.markdown("###  Konfigurasi Training")
    st.markdown("<p style='font-size:0.95rem; color:#cbd5e1;'>Atur rentang eksperimen model K-Means di bawah ini:</p>", unsafe_allow_html=True)
    
    k_min = st.number_input("k minimum", min_value=2, max_value=8, value=st.session_state["user_k_min"], key="k_min_w", on_change=_update_k_range)
    k_max = st.number_input("k maximum", min_value=2, max_value=8, value=st.session_state["user_k_max"], key="k_max_w", on_change=_update_k_range)
    
    if k_min > k_max:
        st.error(" k minimum harus ≤ k maximum")
        st.stop()
        
    st.markdown("<br>", unsafe_allow_html=True)
    start_training = st.button(" Mulai Training", type="primary", use_container_width=True)

with col_info:
    # 1. Reminder Data
    st.markdown(f"""
    <div class="info-card" style="margin-bottom:1rem; padding: 1.2rem;">
        <h4 style="font-size:1rem; margin-bottom:0.5rem;"> Hasil Preprocessing Data</h4>
        <ul style="margin-bottom:0;">
            <li><strong>Fitur Terpilih:</strong> {len(pp['feature_names'])} fitur</li>
            <li><strong>Log Transform:</strong> {'Aktif' if conf['log'] else 'Nonaktif'}</li>
            <li><strong>Scaling:</strong> {conf['scaler']}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 2. Parameter Table
    st.markdown(f"""
    <div class="info-card" style="padding: 1.2rem;">
        <h4 style="font-size:1rem; margin-bottom:0.5rem;"> Penjelasan Parameter Training</h4>
        <table class="param-table" style="width:100%; font-size:0.85rem; margin-top:0;">
            <tr><th style="width:20%">Parameter</th><th style="width:15%">Value</th><th>Penjelasan</th></tr>
            <tr><td><b>Init Method</b></td><td>k-means++</td><td>Menyebar titik jangkar awal saling berjauhan agar akurasi langsung tinggi.</td></tr>
            <tr><td><b>n_init</b></td><td>10</td><td>Mengulang proses acak lempar jangkar 10x, lalu memilih 1 rute yang hasilnya paling rapi.</td></tr>
            <tr><td><b>max_iter</b></td><td>300</td><td>Batas maksimal centroid boleh melangkah/bergeser mencari titik keseimbangan kelompok.</td></tr>
            <tr><td><b>random_state</b></td><td>{RANDOM_STATE}</td><td>Kunci keacakan agar hasil presentasi 100% konsisten (tidak berubah-ubah sendiri).</td></tr>
            <tr><td><b>k range</b></td><td>{k_min} – {k_max}</td><td>Sistem melatih {(k_max - k_min + 1)} jenis pengelompokan berbeda secara bersamaan.</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Training execution
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if start_training:
    k_range = range(int(k_min), int(k_max) + 1)
    total = len(k_range)
    progress = st.progress(0, text="Memulai training …")

    def _cb(current, total):
        progress.progress(current / total, text=f"Training k={k_min + current - 1} … ({current}/{total})")

    with st.spinner("Training K-Means untuk beberapa nilai k …"):
        models = train_multiple_k(scaled_data, k_range, RANDOM_STATE, progress_callback=_cb)

    progress.progress(1.0, text=" Training selesai!")

    # Persist
    st.session_state["all_models"] = models
    st.session_state["k_range"] = list(k_range)

    # Default: use best k or DEFAULT_K
    best_k = DEFAULT_K if DEFAULT_K in models else list(models.keys())[0]
    st.session_state["selected_k"] = best_k
    st.session_state["kmeans_model"] = models[best_k]
    st.session_state["cluster_labels"] = models[best_k].labels_

    st.success(f" Training selesai untuk k = {list(k_range)} — {total} model berhasil dilatih.")

    # Summary
    st.markdown("###  Ringkasan Training")
    summary_rows = []
    for k, m in sorted(models.items()):
        summary_rows.append({
            "k": k,
            "Inertia": f"{m.inertia_:,.2f}",
            "Iterations": m.n_iter_,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="info-card" style="margin-top:1rem;">
        <h4 style="font-size:1rem; margin-bottom:0.5rem;"> Cara Membaca Hasil Training:</h4>
        <ul style="margin-bottom:0; font-size:0.9rem;">
            <li><b>Inertia:</b> Mengukur seberapa "rapat" / "padat" kelompok yang dihasilkan. Semakin kecil angkanya, semakin rapat kelompoknya. <i>(Catatan: Inertia memang akan selalu turun secara alami jika jumlah K/kelompok ditambah, seperti terlihat pada tabel di atas).</i></li>
            <li><b>Iterations:</b> Berapa langkah/geseran yang dibutuhkan si jangkar (centroid) hingga akhirnya posisinya benar-benar stabil.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Show existing results if available
# ──────────────────────────────────────────────
elif "all_models" in st.session_state:
    models = st.session_state["all_models"]
    st.success(f" Model sudah dilatih untuk k = {sorted(models.keys())}")

    st.markdown("###  Ringkasan Training")
    summary_rows = []
    for k, m in sorted(models.items()):
        summary_rows.append({
            "k": k,
            "Inertia": f"{m.inertia_:,.2f}",
            "Iterations": m.n_iter_,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-card" style="margin-top:1rem;">
        <h4 style="font-size:1rem; margin-bottom:0.5rem;"> Cara Membaca Hasil Training:</h4>
        <ul style="margin-bottom:0; font-size:0.9rem;">
            <li><b>Inertia:</b> Mengukur seberapa "rapat" / "padat" kelompok yang dihasilkan. Semakin kecil angkanya, semakin rapat kelompoknya. <i>(Catatan: Inertia memang akan selalu turun secara alami jika jumlah K/kelompok ditambah, seperti terlihat pada tabel di atas).</i></li>
            <li><b>Iterations:</b> Berapa langkah/geseran yang dibutuhkan si jangkar (centroid) hingga akhirnya posisinya benar-benar stabil.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info(" Klik tombol di atas untuk memulai training K-Means.")
