"""
 Home — Clustering Network Traffic Patterns
================================================
Main entry point for the Streamlit multipage application.
"""

import streamlit as st

# ──────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Home",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #082f49 0%, #0369a1 50%, #0284c7 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(14,165,233,0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-container h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-container p {
        font-size: 1.1rem;
        opacity: 0.9;
        line-height: 1.6;
    }

    /* Info cards */
    .info-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: #e2e8f0;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(14,165,233,0.15);
    }
    .info-card h3 {
        color: #38bdf8;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .info-card p, .info-card li {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Pipeline steps */
    .pipeline-step {
        display: inline-block;
        background: linear-gradient(135deg, #0369a1, #0284c7);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.3rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .pipeline-arrow {
        display: inline-block;
        color: #38bdf8;
        font-size: 1.2rem;
        margin: 0 0.2rem;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Group 7")
    st.markdown("""
    - **2802403555** - Akira Agha Nugroho
    - **2802409533** - Darrell Richie Wibawa
    - **2802399513** - Derry Riccardo
    """)
    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Final Project — Machine Learning<br>"
        "BINUS University • Semester 4</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1> Clustering Network Traffic Patterns</h1>
    <p>
        Mengidentifikasi pola trafik jaringan server yang berpotensi tidak lazim
        menggunakan <strong>unsupervised learning</strong> (K-Means Clustering)
        pada dataset <strong>CICIDS2017</strong>.
    </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Problem Statement & Objective
# ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3> Problem Statement</h3>
        <p>
            Server modern menangani trafik yang semakin kompleks — request web,
            komunikasi antar layanan, sinkronisasi cloud, dan potensi trafik
            tidak normal. Pendekatan berbasis aturan statis sulit mengikuti pola
            jaringan yang dinamis dan terus berubah.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3> Objective</h3>
        <p>
            Membangun sistem <strong>behavioral profiling</strong> untuk mengelompokkan
            pola trafik jaringan server dan menandai cluster yang berpotensi tidak lazim.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────
st.markdown("###  Machine Learning Pipeline")
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0;">
    <span class="pipeline-step"> Data Preparation</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step"> Exploratory Data Analysis</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step"> Data Preprocessing</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step"> Training</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step"> Evaluation</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step"> Result</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step"> Input Network Traffic</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model & Dataset summary
# ──────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="info-card">
        <h3> Model</h3>
        <p><strong>K-Means Clustering</strong></p>
        <ul>
            <li>Unsupervised learning berbasis centroid</li>
            <li>Evaluasi beberapa nilai k (2–8)</li>
            <li>Pemilihan k berdasarkan metrik &amp; interpretasi domain</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="info-card">
        <h3> Dataset</h3>
        <p><strong>CICIDS2017</strong></p>
        <ul>
            <li>Benchmark intrusion detection dataset</li>
            <li>Sample: 100.000 rows (stratified)</li>
            <li>Fitur terpilih: 24 fitur numerik network flow</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="info-card">
        <h3> Evaluation Metrics</h3>
        <p><strong>Internal Clustering Metrics</strong></p>
        <ul>
            <li>Silhouette Score (↑ better)</li>
            <li>Davies-Bouldin Index (↓ better)</li>
            <li>Calinski-Harabasz Index (↑ better)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Input / Output
# ──────────────────────────────────────────────
col_in, col_out = st.columns(2)

with col_in:
    st.markdown("""
    <div class="info-card">
        <h3> Input</h3>
        <ul>
            <li>Dataset CICIDS2017 (sample 100.000 rows)</li>
            <li>24 fitur numerik terbaik dari network flow</li>
            <li>Label <code>Attack Type</code> hanya digunakan untuk validasi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_out:
    st.markdown("""
    <div class="info-card">
        <h3> Output</h3>
        <ul>
            <li>Label cluster per network flow</li>
            <li>Visualisasi PCA/t-SNE scatter plot</li>
            <li>Profil &amp; interpretasi cluster</li>
            <li>Kandidat anomali yang layak dimonitor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.85rem;'>"
    " Gunakan menu sidebar di sebelah kiri untuk mulai mengeksplorasi aplikasi."
    "</p>",
    unsafe_allow_html=True,
)
