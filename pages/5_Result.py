"""
 Page 6 — Result & Interpretation
=====================================
Visualises clusters with PCA/t‑SNE, shows distribution, profiles,
narrative interpretations, and anomaly candidate flags.
"""

import streamlit as st

st.set_page_config(page_title="Result & Interpretation", page_icon="", layout="wide")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_STATE, SELECTED_FEATURES
from src.data_loader import load_data, separate_labels
from src.interpretation import (
    build_cluster_profiles,
    flag_anomaly_candidates,
    generate_interpretations,
    get_cluster_summary,
    identify_dominant_features,
)
from src.preprocessing import run_preprocessing_pipeline
from src.training import train_multiple_k
from src.visualization import (
    compute_pca,
    compute_tsne,
    plot_attack_type_by_cluster,
    plot_cluster_distribution,
    plot_cluster_profiles_heatmap,
    plot_pca_scatter,
    plot_tsne_scatter,
)


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

    .cluster-card {
        background: linear-gradient(145deg, #0c4a6e, #082f49);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: #e2e8f0;
    }
    .cluster-card h4 { color: #38bdf8; margin-top: 0; }
    .cluster-card p { color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; }

    .anomaly-flag {
        background: linear-gradient(145deg, #7f1d1d, #991b1b);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .anomaly-flag h4 { color: #fca5a5; margin-top: 0; }
    .anomaly-flag p, .anomaly-flag li { color: #fecaca; font-size: 0.9rem; }

    .anomaly-flag-medium {
        background: linear-gradient(145deg, #78350f, #92400e);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .anomaly-flag-medium h4 { color: #fde68a; margin-top: 0; }
    .anomaly-flag-medium p, .anomaly-flag-medium li { color: #fef3c7; font-size: 0.9rem; }

    .dominant-badge {
        display: inline-block;
        background: #312e81;
        color: #c7d2fe;
        padding: 0.25rem 0.6rem;
        border-radius: 16px;
        font-size: 0.78rem;
        margin: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data loading / model resolution
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dan memproses data …")
def _prepare():
    raw = load_data()
    features_df, labels = separate_labels(raw)
    pp = run_preprocessing_pipeline(features_df, labels=labels)
    return raw, features_df, pp["labels"], pp


try:
    raw_df, features_df, clean_labels, pp = _prepare()
except Exception as exc:
    st.error(f" {exc}")
    st.stop()

scaled_data = pp["scaled_data"]
clean_data = pp["cleaned"]

# Ensure models exist
if "all_models" not in st.session_state:
    with st.spinner("Training model (k=2–8) …"):
        models = train_multiple_k(scaled_data, range(2, 9), RANDOM_STATE)
        st.session_state["all_models"] = models
        st.session_state["k_range"] = list(range(2, 9))

models = st.session_state["all_models"]

# Determine which k to use
selected_k = st.session_state.get("selected_k", 4)
if selected_k not in models:
    selected_k = sorted(models.keys())[0]

model = models[selected_k]
cluster_labels = model.labels_

# Persist
st.session_state["kmeans_model"] = model
st.session_state["cluster_labels"] = cluster_labels


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2> Result & Interpretation</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(f"Menampilkan hasil clustering dengan **k = {selected_k}** cluster.")


# ──────────────────────────────────────────────
# 1. PCA / t-SNE Scatter
# ──────────────────────────────────────────────
st.markdown("###  Visualisasi Cluster")
st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #3b82f6;">
    <b> Apa itu PCA dan t-SNE?</b><br>
    Dataset kita memiliki 24 kolom (fitur). Secara matematis, ini membentuk ruang berdimensi 24 yang mustahil digambar secara visual. <b>PCA (Principal Component Analysis)</b> dan <b>t-SNE (t-Distributed Stochastic Neighbor Embedding)</b> adalah teknik "kamera reduksi dimensi" yang memproyeksikan data dari ruang 24-dimensi menjadi ruang datar 2-dimensi agar bisa ditampilkan di grafik Scatter Plot.<br><br>
    • <b>PCA</b>: Fokus menjaga struktur data secara global (komputasi cepat, tapi terkadang pemisahan antar cluster kurang tajam).<br>
    • <b>t-SNE</b>: Fokus menjaga jarak titik-titik yang bertetangga dekat (komputasi sangat lambat, tapi seringkali memisahkan cluster dengan sangat tajam dan indah).<br><br>
    <i>Panduan Membaca Grafik:</i> Titik-titik yang warnanya sama menandakan mereka masuk ke cluster yang sama. Semakin mereka menumpuk rapat di satu area, berarti perilaku jaringannya semakin mirip.
</div>
""", unsafe_allow_html=True)

tab_pca, tab_tsne = st.tabs([" PCA", " t-SNE"])

with tab_pca:
    pca_data, pca_obj = compute_pca(scaled_data)
    
    # Subsample for PCA visualization to avoid browser lag
    pca_sample_size = min(20000, len(pca_data))
    rng = np.random.RandomState(42)
    pca_idx = rng.choice(len(pca_data), pca_sample_size, replace=False)
    
    fig_pca = plot_pca_scatter(pca_data[pca_idx], cluster_labels[pca_idx], pca_obj)
    st.plotly_chart(fig_pca, width="stretch")
    st.session_state["pca_result"] = pca_data

    ev = pca_obj.explained_variance_ratio_
    st.caption(
        f"PC1 menjelaskan **{ev[0]:.1%}** varian, "
        f"PC2 menjelaskan **{ev[1]:.1%}** varian "
        f"(total: **{sum(ev[:2]):.1%}**)."
    )

with tab_tsne:
    st.markdown(
        " t-SNE memerlukan waktu lebih lama. "
        "Klik tombol di bawah untuk memulai."
    )
    # Use a subsample for t-SNE performance
    tsne_sample_size = min(20000, len(scaled_data))
    if st.button(" Jalankan t-SNE", key="run_tsne"):
        with st.spinner(f"Menjalankan t-SNE pada {tsne_sample_size:,} sample …"):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(scaled_data), tsne_sample_size, replace=False)
            tsne_data = compute_tsne(scaled_data[idx])
            fig_tsne = plot_tsne_scatter(tsne_data, cluster_labels[idx])
            st.plotly_chart(fig_tsne, width="stretch")
    else:
        st.info("Klik tombol di atas untuk menjalankan t-SNE.")


# ──────────────────────────────────────────────
# 2. Cluster Distribution
# ──────────────────────────────────────────────
st.markdown("###  Distribusi Cluster")
st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #10b981;">
    <b> Apa arti tabel ini?</b><br>
    Tabel ini menunjukkan pembagian "populasi" data. Di dunia nyata, trafik jaringan normal biasanya mendominasi (mengisi 80-90% dari total populasi) dan berkumpul di satu cluster raksasa. Sebaliknya, trafik serangan atau anomali biasanya berukuran kecil namun spesifik, sehingga sering terlempar ke cluster-cluster minoritas.
</div>
""", unsafe_allow_html=True)

col_chart, col_table = st.columns([2, 1])
with col_chart:
    fig_dist = plot_cluster_distribution(cluster_labels)
    st.plotly_chart(fig_dist, width="stretch")

with col_table:
    counts = pd.Series(cluster_labels).value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Cluster": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / counts.sum() * 100).round(2),
    })
    st.dataframe(dist_df, width="stretch", hide_index=True)


# ──────────────────────────────────────────────
# 3. Cluster Profiles
# ──────────────────────────────────────────────
st.markdown("###  Profil Cluster (Rata-rata Fitur)")
st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #f59e0b;">
    <b> Apa isi tabel ini?</b><br>
    Tabel ini menampilkan <b>Titik Tengah (Rata-rata)</b> dari setiap fitur (kolom) untuk masing-masing cluster. Melalui tabel ini, kita bisa membandingkan secara numerik mentah berapa rata-rata nilai suatu fitur di Cluster 0 dibandingkan dengan rata-rata di Cluster lainnya.
</div>
""", unsafe_allow_html=True)

summary = get_cluster_summary(clean_data, cluster_labels, pp["feature_names"])
st.session_state["cluster_profiles"] = summary

st.dataframe(
    summary.style.format("{:.2f}"),
    width="stretch",
)

# Heatmap
st.markdown("###  Heatmap Profil Cluster")
st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #d97706;">
    <b> Cara Membaca Heatmap (Peta Panas):</b><br>
    Membaca puluhan angka koma di tabel kadang membuat pusing. Heatmap membantu dengan mengubah angka tersebut menjadi spektrum warna! Semakin terang/kuning warnanya pada suatu kotak, menandakan bahwa rata-rata fitur tersebut <b>sangat tinggi (menonjol)</b> di cluster tersebut dibandingkan warna gelap/ungu yang menandakan nilainya rendah.
</div>
""", unsafe_allow_html=True)
fig_hm = plot_cluster_profiles_heatmap(summary)
st.pyplot(fig_hm)


# ──────────────────────────────────────────────
# 4. Dominant Features
# ──────────────────────────────────────────────
st.markdown("###  Fitur Dominan per Cluster")
st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #8b5cf6;">
    <b> Bagaimana fitur (badge) ini dipilih?</b><br>
    Sistem membandingkan angka rata-rata di tiap cluster dengan rata-rata seluruh populasi data secara umum menggunakan rumus <b>Z-Score</b>. Sistem kemudian mengurutkannya dan secara otomatis mengambil <b>Top 5 fitur</b> dengan penyimpangan paling ekstrem (baik terlalu besar maupun terlalu kecil). Kelima fitur teratas inilah yang diangkat menjadi "Ciri Khas" (Fitur Dominan) yang mendefinisikan kelakuan kelabakan dari cluster tersebut.
</div>
""", unsafe_allow_html=True)

dominant = identify_dominant_features(summary, top_n=5)
for cluster_id, feats in sorted(dominant.items()):
    badges = " ".join(f'<span class="dominant-badge">{f}</span>' for f in feats)
    st.markdown(
        f"**Cluster {cluster_id}:** {badges}",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 5. Narrative Interpretations
# ──────────────────────────────────────────────
st.markdown("###  Interpretasi Cluster")
st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #ec4899;">
    <b> Dari mana datangnya teks narasi ini?</b><br>
    K-Means adalah algoritma matematika murni yang hanya tahu cara mengelompokkan jarak angka, ia tidak paham makna dari data tersebut. Oleh karena itu, sistem kita menerjemahkan kelakuan angka-angka ekstrem dari Fitur Dominan tersebut menjadi bahasa narasi manusia menggunakan logika heuristik (<i>Rule-of-Thumb</i>).
</div>
""", unsafe_allow_html=True)

with st.expander(" Lihat Logika Heuristik (Rule-of-Thumb) di Balik Layar"):
    st.markdown("""
    Sistem menghitung **Rasio** dengan cara membagi `Rata-rata Fitur Cluster` dibagi dengan `Rata-rata Keseluruhan Populasi Data`. 
    Berdasarkan rasio tersebut, berikut adalah *rule-of-thumb* yang digunakan di kode untuk merangkai teks narasi:
    
    *   **Durasi Koneksi (Flow Duration):**
        *   Rasio < 0.3 *(Durasi anjlok kurang dari 30% batas wajar)* ➔ `"koneksi sangat singkat"`
        *   Rasio < 0.7 ➔ `"durasi pendek–menengah"`
        *   Rasio > 2.0 *(Durasi membengkak lebih dari 200% atau 2x lipat wajar)* ➔ `"durasi sangat panjang"`
        *   Kondisi lainnya (0.7 s.d 2.0) ➔ `"durasi moderat"`
    *   **Volume Paket (Length of Fwd Packets, Avg Packet Size):**
        *   Rasio < 0.3 ➔ `"volume paket rendah"`
        *   Rasio > 2.0 ➔ `"volume paket sangat tinggi"`
        *   Rasio > 1.3 ➔ `"volume paket cukup besar"`
        *   *Kondisi lainnya (0.3 s.d 1.3) diabaikan/tidak masuk narasi*
    *   **Kecepatan Transfer (Packets/s):**
        *   Rasio < 0.3 ➔ `"packet rate rendah"`
        *   Rasio > 2.0 ➔ `"packet rate sangat tinggi (burst-like)"`
    *   **Sinyal Jaringan (TCP Flags):**
        *   FIN Flag Count > 1.5 *(Frekuensi flag pemutus koneksi naik 50% dari wajar)* ➔ `"banyak koneksi FIN (connection teardown)"`
        *   PSH Flag Count > 1.5 ➔ `"frekuensi PUSH tinggi"`
    
    *Catatan: Threshold/ambang batas ini ditetapkan secara manual (berdasarkan Domain Knowledge Cybersecurity) untuk memisahkan lalu lintas ekstrem dari lalu lintas wajar.*
    """)

cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
interpretations = generate_interpretations(summary, cluster_counts)
st.session_state["cluster_interpretations"] = interpretations

for cid, text in sorted(interpretations.items()):
    st.markdown(f"""
    <div class="cluster-card">
        <h4>Cluster {cid}</h4>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 6. Anomaly Candidates
# ──────────────────────────────────────────────
st.markdown("###  Kandidat Anomali")
st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #ef4444;">
    <b> Mengapa ditandai merah sebagai Anomali?</b><br>
    Cluster yang masuk ke perangkap kotak ini memiliki dua ciri mutlak: <b>Ukurannya sangat kecil</b> (kelompok minoritas) dan <b>perilakunya sangat ekstrem</b> menyimpang dari batas wajar. Kombinasi ini sangat dicurigai sebagai aktivitas serangan (seperti DDoS, PortScan, atau Brute Force).
</div>
""", unsafe_allow_html=True)

flags = flag_anomaly_candidates(summary, cluster_counts)

if flags:
    for f in flags:
        css_class = "anomaly-flag" if f["severity"] == "HIGH" else "anomaly-flag-medium"
        reasons_html = "".join(f"<li>{r}</li>" for r in f["reasons"])
        st.markdown(f"""
        <div class="{css_class}">
            <h4> Cluster {f['cluster']} — Severity: {f['severity']}</h4>
            <ul>{reasons_html}</ul>
            <p><em>Cluster ini berpotensi tidak lazim dan memerlukan investigasi lebih lanjut.</em></p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Tidak ada cluster yang ditandai sebagai kandidat anomali berdasarkan threshold default.")


# ──────────────────────────────────────────────
# 7. Cross-reference with Attack Type (optional)
# ──────────────────────────────────────────────
if clean_labels is not None:
    st.markdown("###  Referensi: Attack Type per Cluster")
    st.markdown("""
<div class="cluster-card" style="border-left: 4px solid #14b8a6;">
    <b> Momen Validasi (Pembuktian Kinerja Model):</b><br>
    Saat melatih algoritma K-Means, kita sengaja <b>mencabut dan menyembunyikan</b> kolom Kunci Jawaban Asli (<code>Attack Type</code>) dari dataset. Model K-Means kemudian bekerja secara "buta" murni hanya melihat pola angka matematika (durasi, jumlah paket, dll).<br><br>
    Di grafik dan tabel bawah ini, kita <b>membuka kembali Kunci Jawaban Asli</b> dan menjejerkannya dengan Cluster tebakan model. Jika sebuah cluster ternyata berhasil mengumpulkan satu jenis serangan spesifik secara dominan (misalnya: jika Anda menemukan sebuah cluster yang isinya murni serangan DDoS), ini membuktikan bahwa K-Means <b>berhasil menemukan pola kelakuan serangan tersebut secara mandiri tanpa melihat contekan label!</b>
</div>
""", unsafe_allow_html=True)

    fig_atk = plot_attack_type_by_cluster(cluster_labels, clean_labels)
    st.plotly_chart(fig_atk, width="stretch")

    # Cross-tab
    with st.expander(" Tampilkan Tabel Validasi Silang (Cross-Tabulation)"):
        st.markdown("""
        <b>Cara Membaca Tabel:</b><br>
        Tabel ini menunjukkan jenis trafik apa saja (berdasarkan label asli pada kolom) yang masuk ke dalam setiap Cluster buatan model (baris).<br>
        """, unsafe_allow_html=True)
        ct = pd.crosstab(
            pd.Series(cluster_labels, name="Cluster"),
            clean_labels.reset_index(drop=True),
        )
        st.dataframe(ct, width="stretch")


# ──────────────────────────────────────────────
# Disclaimer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
>  **Disclaimer:** Cluster yang ditandai sebagai "kandidat anomali" bukan berarti
> serangan pasti. Fokus project ini adalah **behavioral profiling**, bukan klaim deteksi intrusi final.
""")
