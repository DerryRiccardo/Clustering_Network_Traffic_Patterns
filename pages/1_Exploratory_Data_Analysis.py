"""
 Page 1 — Exploratory Data Analysis
=======================================
Displays missing values, column glossary, descriptive statistics,
outliers, correlation, skewness, and a column explorer.
"""

import streamlit as st

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="", layout="wide")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_COLUMN, SELECTED_FEATURES
from src.data_loader import load_data, separate_labels
from src.eda import (
    get_correlation_matrix,
    get_descriptive_stats,
    get_missing_info,
    get_outlier_info,
    get_skewness_info,
)
from src.visualization import plot_boxplot, plot_correlation_heatmap, plot_histogram


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
        margin-bottom: 1.5rem;
        color: #e2e8f0;
    }
    .section-header h2 { color: white; font-size: 1.3rem; margin: 0; }

    .insight-box {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-left: 4px solid #818cf8;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: #e2e8f0;
        font-size: 0.92rem;
    }

    .metric-card-sm {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: #e2e8f0;
    }
    .metric-card-sm h2 {
        color: #38bdf8;
        font-size: 1.8rem;
        margin: 0;
    }
    .metric-card-sm p {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    .stats-table-container {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat data …")
def _load():
    df = load_data()
    _, labels = separate_labels(df)
    return df, labels


try:
    raw_df, labels = _load()
except (FileNotFoundError, ValueError) as exc:
    st.error(f" {exc}")
    st.stop()

# Store in session state
st.session_state["raw_data"] = raw_df
st.session_state["attack_labels"] = labels

# Use ALL numeric columns for EDA (not just selected features)
numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
eda_df = raw_df[numeric_cols].copy()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2> Exploratory Data Analysis</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Halaman ini menampilkan insight awal data **sebelum preprocessing/training** "
    "untuk memahami karakteristik network flow."
)


# ──────────────────────────────────────────────
# Quick overview metrics
# ──────────────────────────────────────────────
st.markdown("####  Ringkasan Dataset")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card-sm">
        <h2>{raw_df.shape[0]:,}</h2>
        <p>Total Rows</p>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card-sm">
        <h2>{raw_df.shape[1]}</h2>
        <p>Total Columns</p>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card-sm">
        <h2>{len(numeric_cols)}</h2>
        <p>Numeric Columns</p>
    </div>
    """, unsafe_allow_html=True)
with m4:
    total_missing = raw_df.isnull().sum().sum() + raw_df.isin([np.inf, -np.inf]).sum().sum()
    st.markdown(f"""
    <div class="metric-card-sm">
        <h2>{int(total_missing):,}</h2>
        <p>Missing / Inf Values</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Macro Data (Inside Expanders)
# ──────────────────────────────────────────────
st.markdown("####  Macro Overview (Keseluruhan Data)")
st.markdown("Buka panel di bawah ini untuk melihat rangkuman data secara keseluruhan.")

# 1. Missing Value Table
with st.expander(" Tampilkan Missing & Infinite Values", expanded=False):
    missing_all = raw_df.isnull().sum()
    inf_all = raw_df.isin([np.inf, -np.inf]).sum()
    total = len(raw_df)

    missing_df = pd.DataFrame({
        "Column": raw_df.columns,
        "Missing Count": missing_all.values,
        "Missing %": (missing_all.values / total * 100).round(2),
        "Infinite Count": inf_all.values,
        "Total Issues": missing_all.values + inf_all.values,
    })
    missing_df = missing_df.sort_values("Total Issues", ascending=False).reset_index(drop=True)

    has_issues = missing_df[missing_df["Total Issues"] > 0]
    if has_issues.empty:
        st.success(" Tidak ada missing value atau infinity pada seluruh kolom dataset.")
    else:
        st.warning(f" Ditemukan **{len(has_issues)} kolom** dengan missing/infinite values.")
        st.dataframe(
            has_issues.style.format({
                "Missing %": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.dataframe(missing_df, use_container_width=True, hide_index=True, height=300)

    st.markdown("""
    <div class="insight-box">
         <strong>Insight:</strong> Tabel di atas menampilkan jumlah missing value dan infinite value
        untuk <strong>semua kolom</strong> dalam dataset. Kolom dengan banyak missing/infinite value
        perlu ditangani pada tahap preprocessing sebelum digunakan untuk clustering.
    </div>
    """, unsafe_allow_html=True)

# 2. Column Glossary
with st.expander(" Tampilkan Penjelasan Kolom (Glossary)", expanded=False):
    st.markdown("Berikut adalah penjelasan singkat untuk setiap kolom dalam dataset CICIDS2017:")

    COLUMN_DESCRIPTIONS = {
        "Destination Port": "Nomor port tujuan koneksi. Contoh: port 80 (HTTP), 443 (HTTPS), 22 (SSH).",
        "Flow Duration": "Durasi total koneksi dari awal sampai akhir (dalam mikrodetik).",
        "Total Fwd Packets": "Jumlah total paket yang dikirim dari sumber ke tujuan.",
        "Total Length of Fwd Packets": "Total ukuran (byte) dari semua paket yang dikirim ke tujuan.",
        "Fwd Packet Length Max": "Ukuran maksimum dari satu paket yang dikirim ke tujuan.",
        "Fwd Packet Length Min": "Ukuran minimum dari satu paket yang dikirim ke tujuan.",
        "Fwd Packet Length Mean": "Rata-rata ukuran paket yang dikirim ke tujuan.",
        "Fwd Packet Length Std": "Standar deviasi ukuran paket yang dikirim ke tujuan. Nilai tinggi berarti ukuran paket bervariasi.",
        "Bwd Packet Length Max": "Ukuran maksimum paket balasan dari tujuan ke sumber.",
        "Bwd Packet Length Min": "Ukuran minimum paket balasan dari tujuan ke sumber.",
        "Bwd Packet Length Mean": "Rata-rata ukuran paket balasan dari tujuan.",
        "Bwd Packet Length Std": "Standar deviasi ukuran paket balasan. Nilai tinggi berarti variasi besar.",
        "Flow Bytes/s": "Kecepatan transfer data dalam bytes per detik selama koneksi berlangsung.",
        "Flow Packets/s": "Kecepatan pengiriman paket per detik selama koneksi berlangsung.",
        "Flow IAT Mean": "Rata-rata waktu antar-kedatangan paket (Inter-Arrival Time) dalam koneksi.",
        "Flow IAT Std": "Standar deviasi waktu antar-kedatangan paket. Nilai tinggi berarti jeda antar paket tidak konsisten.",
        "Flow IAT Max": "Waktu terlama antar dua paket berurutan dalam koneksi.",
        "Flow IAT Min": "Waktu terpendek antar dua paket berurutan dalam koneksi.",
        "Fwd IAT Total": "Total waktu antar-kedatangan semua paket ke arah tujuan.",
        "Fwd IAT Mean": "Rata-rata waktu antar-kedatangan paket ke arah tujuan.",
        "Fwd IAT Std": "Standar deviasi waktu antar-kedatangan paket ke arah tujuan.",
        "Fwd IAT Max": "Waktu terlama antar paket ke arah tujuan.",
        "Fwd IAT Min": "Waktu terpendek antar paket ke arah tujuan.",
        "Bwd IAT Total": "Total waktu antar-kedatangan semua paket balasan.",
        "Bwd IAT Mean": "Rata-rata waktu antar-kedatangan paket balasan.",
        "Bwd IAT Std": "Standar deviasi waktu antar-kedatangan paket balasan.",
        "Bwd IAT Max": "Waktu terlama antar paket balasan.",
        "Bwd IAT Min": "Waktu terpendek antar paket balasan.",
        "Fwd Header Length": "Total ukuran header dari semua paket ke arah tujuan (byte).",
        "Bwd Header Length": "Total ukuran header dari semua paket balasan (byte).",
        "Fwd Packets/s": "Kecepatan pengiriman paket per detik ke arah tujuan.",
        "Bwd Packets/s": "Kecepatan pengiriman paket balasan per detik.",
        "Min Packet Length": "Ukuran paket terkecil dalam seluruh koneksi.",
        "Max Packet Length": "Ukuran paket terbesar dalam seluruh koneksi.",
        "Packet Length Mean": "Rata-rata ukuran semua paket (maju + mundur) dalam koneksi.",
        "Packet Length Std": "Standar deviasi ukuran semua paket dalam koneksi.",
        "Packet Length Variance": "Varian ukuran semua paket dalam koneksi (kuadrat dari standar deviasi).",
        "FIN Flag Count": "Jumlah paket dengan flag FIN (sinyal penutupan koneksi).",
        "PSH Flag Count": "Jumlah paket dengan flag PSH (sinyal untuk segera memproses data).",
        "ACK Flag Count": "Jumlah paket dengan flag ACK (sinyal konfirmasi penerimaan data).",
        "Average Packet Size": "Rata-rata ukuran paket dalam koneksi.",
        "Subflow Fwd Bytes": "Total byte yang dikirim dalam sub-flow ke arah tujuan.",
        "Init_Win_bytes_forward": "Ukuran jendela TCP awal dari sumber ke tujuan (byte). Menunjukkan kapasitas buffer pengirim.",
        "Init_Win_bytes_backward": "Ukuran jendela TCP awal dari tujuan ke sumber (byte). Menunjukkan kapasitas buffer penerima.",
        "act_data_pkt_fwd": "Jumlah paket yang benar-benar membawa data ke arah tujuan (bukan hanya header).",
        "min_seg_size_forward": "Ukuran segment minimum yang diamati ke arah tujuan.",
        "Active Mean": "Rata-rata durasi koneksi dalam keadaan aktif (sedang mengirim/menerima data).",
        "Active Max": "Durasi terlama koneksi dalam keadaan aktif.",
        "Active Min": "Durasi terpendek koneksi dalam keadaan aktif.",
        "Idle Mean": "Rata-rata durasi koneksi dalam keadaan idle (tidak ada pengiriman data).",
        "Idle Max": "Durasi terlama koneksi dalam keadaan idle.",
        "Idle Min": "Durasi terpendek koneksi dalam keadaan idle.",
        "Attack Type": "Label jenis serangan (atau 'BENIGN' untuk trafik normal). Hanya digunakan sebagai referensi, bukan untuk training.",
    }

    glossary_data = []
    for col in raw_df.columns:
        glossary_data.append({
            "Column": col,
            "Tipe Data": str(raw_df[col].dtype),
            "Penjelasan": COLUMN_DESCRIPTIONS.get(col, "—"),
        })

    glossary_df = pd.DataFrame(glossary_data)
    st.dataframe(glossary_df, use_container_width=True, hide_index=True, height=400)

# 3. Descriptive Statistics
with st.expander(" Tampilkan Statistik Deskriptif Keseluruhan", expanded=False):
    st.markdown(
        "Statistik deskriptif berikut menampilkan ringkasan distribusi setiap fitur numerik, "
        "termasuk count, mean, std, min, quartiles, dan max."
    )

    stats = get_descriptive_stats(eda_df)
    st.dataframe(stats.style.format("{:.2f}"), use_container_width=True, height=400)

    st.markdown("""
    <div class="insight-box">
         <strong>Insight:</strong> Perhatikan perbedaan skala antar fitur —
        beberapa fitur seperti <em>Flow Bytes/s</em> memiliki rentang sangat besar
        dibanding <em>FIN Flag Count</em>. Ini menunjukkan bahwa
        <strong>scaling wajib</strong> sebelum K-Means clustering.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 4. Data Visualization Tabs
# ──────────────────────────────────────────────
st.markdown("####  Data Visualization & Analysis")

tab_explorer, tab_box, tab_corr, tab_skew = st.tabs([
    " Column Explorer",
    " Boxplot Comparison",
    " Correlation",
    " Skewness",
])


# ── Tab 1: Column Explorer ──────────────────
with tab_explorer:
    st.markdown(
        "Pilih salah satu kolom untuk melihat detail informasi, "
        "statistik yang lebih ringkas, dan visualisasinya secara mendalam."
    )

    if "user_explorer_col" not in st.session_state:
        st.session_state["user_explorer_col"] = raw_df.columns.tolist()[0]
    
    def _update_explorer_col():
        st.session_state["user_explorer_col"] = st.session_state["column_explorer_select_w"]

    try:
        def_idx = raw_df.columns.tolist().index(st.session_state["user_explorer_col"])
    except ValueError:
        def_idx = 0

    explorer_col = st.selectbox(
        "Pilih kolom untuk dieksplorasi:",
        raw_df.columns.tolist(),
        index=def_idx,
        key="column_explorer_select_w",
        on_change=_update_explorer_col,
    )

    col_data = raw_df[explorer_col]

    # Column info explanation
    desc = COLUMN_DESCRIPTIONS.get(explorer_col, "Tidak ada deskripsi tersedia.")
    st.markdown(f"""
    <div class="insight-box" style="margin-top:0;">
         <strong>{explorer_col}:</strong> {desc}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Layout: Stats on the left (compact), Charts on the right
    col_left, col_right = st.columns([1.2, 2.5])

    with col_left:
        st.markdown("#####  Basic Info")
        basic_info = pd.DataFrame({
            "Metric": ["Tipe Data", "Non-Null Count", "Null Count", "Unique Values"],
            "Value": [
                str(col_data.dtype),
                f"{col_data.notna().sum():,}",
                f"{col_data.isna().sum():,}",
                f"{col_data.nunique():,}"
            ]
        })
        st.dataframe(basic_info, hide_index=True, use_container_width=True)
        
        if pd.api.types.is_numeric_dtype(col_data):
            st.markdown("#####  Statistics")
            
            skew_val = col_data.skew()
            skew_abs = abs(skew_val)
            if skew_abs < 0.5:
                skew_verdict = "Approx. Symmetric"
            elif skew_abs < 1.0:
                skew_verdict = "Moderate Skew"
            elif skew_abs < 2.0:
                skew_verdict = "High Skew"
            else:
                skew_verdict = "Very High Skew"
                
            stats_data = pd.DataFrame({
                "Metric": ["Mean", "Median (Q2)", "Std Dev", "Min", "Max", "Q1 (25%)", "Q3 (75%)", "Skewness", "Skew Verdict"],
                "Value": [
                    f"{col_data.mean():,.4f}",
                    f"{col_data.median():,.4f}",
                    f"{col_data.std():,.4f}",
                    f"{col_data.min():,.4f}",
                    f"{col_data.max():,.4f}",
                    f"{col_data.quantile(0.25):,.4f}",
                    f"{col_data.quantile(0.75):,.4f}",
                    f"{skew_val:,.4f}",
                    skew_verdict
                ]
            })
            st.dataframe(stats_data, hide_index=True, use_container_width=True)
            
        st.markdown("#####  Sample Data")
        st.dataframe(raw_df[[explorer_col]].head(10), use_container_width=True)

    with col_right:
        if pd.api.types.is_numeric_dtype(col_data):
            st.markdown("#####  Distribusi")
            fig_explorer = plot_histogram(eda_df if explorer_col in eda_df.columns else raw_df[[explorer_col]], explorer_col)
            st.pyplot(fig_explorer)

            st.markdown("#####  Boxplot (Sebaran & Outlier)")
            fig_box_explorer = plot_boxplot(
                eda_df if explorer_col in eda_df.columns else raw_df[[explorer_col]],
                [explorer_col],
            )
            st.pyplot(fig_box_explorer)
        else:
            # Non-numeric column (e.g., Attack Type)
            st.markdown("#####  Distribusi Kategori")
            vc = col_data.value_counts().reset_index()
            vc.columns = ["Value", "Count"]
            vc["Percentage"] = (vc["Count"] / len(col_data) * 100).round(2)
            
            import plotly.express as px
            fig_vc = px.bar(
                vc,
                x="Value",
                y="Count",
                color="Value",
                title=f"Distribusi {explorer_col}",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_vc.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis_tickangle=-45,
                height=500,
            )
            st.plotly_chart(fig_vc, use_container_width=True)
            
            st.markdown("#####  Rincian Value Counts")
            st.dataframe(vc, use_container_width=True, hide_index=True)


# ── Tab 2: Boxplot Comparison ────────────────
with tab_box:
    st.markdown("####  Feature Comparison (Boxplot)")
    st.markdown("""
    **Tujuan:** Membandingkan rentang nilai dan sebaran (*distribution*) dari beberapa fitur secara bersamaan.
    Grafik Boxplot sangat berguna untuk mendeteksi **outlier** (titik-titik ekstrem yang berada di luar batas kotak utama) 
    dan melihat apakah skala antar-kolom seragam atau jomplang.
    """)

    if "user_box_features" not in st.session_state:
        st.session_state["user_box_features"] = numeric_cols[:6]

    def _update_box_features():
        st.session_state["user_box_features"] = st.session_state["eda_boxplot_features_w"]

    box_features = st.multiselect(
        "Pilih fitur (maks 8):",
        numeric_cols,
        default=st.session_state["user_box_features"],
        max_selections=8,
        key="eda_boxplot_features_w",
        on_change=_update_box_features,
    )
    if box_features:
        fig_box = plot_boxplot(eda_df, box_features)
        st.pyplot(fig_box)

    st.markdown("####  Outlier Summary (IQR Method)")
    st.markdown(
        "Tabel ini merangkum persentase *outlier* (data ekstrem) pada masing-masing fitur. "
        "Nilai *outlier* dihitung menggunakan metode matematis IQR (Interquartile Range)."
    )
    outlier_info = get_outlier_info(eda_df)
    st.dataframe(outlier_info.style.format("{:.2f}"), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
         <strong>Insight & Tindakan Lanjut:</strong>
        <ul>
            <li><strong>Skala Jomplang:</strong> Lewat Boxplot, kita bisa melihat bahwa rentang nilai antar-kolom sangat berbeda (ada yang ratusan ribu, ada yang puluhan). Ini menegaskan mengapa kita <strong>Wajib Melakukan Scaling (Normalisasi)</strong>. K-Means tidak bisa membandingkan apel dan gajah secara adil jika skalanya tidak disamakan.</li>
            <li><strong>Dominasi Outlier:</strong> Banyak fitur jaringan yang didominasi oleh nilai ekstrem (seperti saat terjadi <i>traffic burst</i> / lonjakan koneksi). K-Means sangat sensitif menarik titik tengah (centroid) ke arah <i>outlier</i>. Oleh karena itu, kita akan menerapkan <strong>Log Transformation</strong> di tahap preprocessing untuk menjinakkan nilai-nilai liar ini.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 3: Correlation ───────────────────────
with tab_corr:
    st.markdown("####  Correlation Analysis")
    st.markdown("""
    **Tujuan:** Mengukur seberapa kuat hubungan antara satu fitur dengan fitur lainnya. 
    Nilai korelasi berkisar dari **-1 hingga 1**:
    - **Mendekati 1 (Merah Pekat):** Dua fitur bergerak searah (contoh: jika jumlah paket naik, total ukuran paket pasti ikut naik).
    - **Mendekati -1 (Biru Pekat):** Dua fitur memiliki hubungan yang berlawanan arah.
    - **Mendekati 0 (Warna Netral):** Tidak ada hubungan sama sekali.
    """)
    corr = get_correlation_matrix(eda_df)
    fig_corr = plot_correlation_heatmap(corr)
    st.pyplot(fig_corr)

    # Top correlations
    st.markdown("####  Top Korelasi Terkuat")
    st.markdown("Tabel pasangan fitur yang memiliki hubungan paling kuat secara matematis.")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_pairs = corr.where(mask).stack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
    top_corr = corr_pairs.sort_values("Abs Correlation", ascending=False).head(10)
    st.dataframe(
        top_corr[["Feature 1", "Feature 2", "Correlation"]].style.format({"Correlation": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("#####  Contoh Keterkaitan Logis di Dataset Ini:")
    st.markdown("""
    Berdasarkan data korelasi di atas, kita bisa melihat beberapa fitur yang saling berkaitan secara alami:
    - **`Total Length of Fwd Packets` & `Subflow Fwd Bytes` (Korelasi > 0.99):** Keduanya mengukur volume byte yang dikirim ke depan. Hal ini sangat wajar karena subflow seringkali merupakan aliran utama itu sendiri.
    - **`Fwd Packets/s` & `Flow Packets/s` (Korelasi > 0.95):** Kecepatan paket ke arah depan sangat mendominasi kecepatan paket keseluruhan. Jika *forward* ngebut, maka *flow* keseluruhan juga ikut ngebut.
    - **`Idle Mean` & `Idle Max` (Korelasi > 0.98):** Rata-rata waktu istirahat (*idle*) koneksi sangat berkaitan erat dengan waktu istirahat terlamanya.
    - **`Flow Duration` & `Fwd IAT Total`:** Durasi total koneksi pasti berbanding lurus dengan total waktu antar-paket (*Inter-Arrival Time*).
    """)

    st.markdown("""
    <div class="insight-box">
         <strong>Insight & Pemahaman:</strong>
        <ul>
            <li>Fitur yang memiliki korelasi hampir sempurna (seperti 0.99 atau 1.0) membawa informasi yang sama persis (<strong>redundan</strong>). Menyimpan keduanya ibarat memasukkan kolom "Tinggi Badan (cm)" dan "Tinggi Badan (inch)" sekaligus.</li>
            <li>Dalam beberapa model klasifikasi biasa, fitur redundan ini akan langsung dihapus agar tidak membingungkan model. Namun, karena model kita (K-Means) berfokus mencari 'kemiripan jarak' pada ruang data, kita dapat menoleransi korelasi moderat selama fitur tersebut memiliki karakteristik unik untuk mendeskripsikan <i>profil jaringan</i>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 4: Skewness ──────────────────────────
with tab_skew:
    st.markdown("####  Feature Skewness Analysis")
    st.markdown("""
    **Tujuan:** Mengukur kemiringan distribusi data (*Skewness*). 
    - Data normal ideal berbentuk seperti lonceng simetris (*skewness* mendekati 0).
    - Jika *skewness* tinggi (positif), artinya mayoritas data bernilai kecil, tetapi ada sedikit data pencilan yang nilainya buas memanjang ke arah angka raksasa. 
    - Model K-Means menyukai data yang simetris (bulat). Data yang terlalu miring akan membuat model membentuk *cluster* yang berat sebelah.
    """)
    skew_info = get_skewness_info(eda_df)
    st.dataframe(
        skew_info.style.format({"skewness": "{:.3f}", "abs_skewness": "{:.3f}"}),
        use_container_width=True,
    )

    very_skewed = skew_info[skew_info["abs_skewness"] > 2.0]
    if not very_skewed.empty:
        st.markdown(f"**{len(very_skewed)} fitur** memiliki skewness sangat tinggi (|skew| > 2):")
        for feat in very_skewed.index:
            st.markdown(f"- `{feat}` — skew = {very_skewed.loc[feat, 'skewness']:.2f}")

    st.markdown("""
    <div class="insight-box">
         <strong>Insight & Tindakan Lanjut:</strong>
        <ul>
            <li>Mayoritas fitur dalam trafik jaringan memiliki status <strong>Very High Skew</strong>. Hal ini sangat wajar! Di dunia nyata, 90% lalu lintas jaringan hanya mengirim paket-paket kecil, dan sisanya (10%) digunakan untuk *download* file raksasa yang menyumbang angka ekstrem.</li>
            <li>Untuk "menjinakkan" data yang sangat miring ini, pada tahap Preprocessing selanjutnya kita akan mengubah datanya dengan <strong>Transformasi Logaritmik (Log1p)</strong>. Efeknya, rentang jarak antara data kecil dan data raksasa akan dirapatkan sehingga bentuk grafiknya menjadi lebih lonceng (Normal) dan siap disuapkan ke K-Means.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
