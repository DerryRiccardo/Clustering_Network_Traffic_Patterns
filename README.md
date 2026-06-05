# Clustering Network Traffic Patterns

Membangun sistem *Machine Learning* berbasis **Unsupervised Learning (K-Means Clustering)** untuk melakukan *behavioral profiling* pada pola trafik jaringan server. Sistem ini bertujuan untuk mengelompokkan karakteristik lalu lintas jaringan dan menandai *cluster* yang berpotensi tidak lazim (kandidat anomali).

---

## 🎯 Problem Statement & Objective

Server modern menangani trafik yang sangat dinamis dan kompleks (request web, komunikasi antar layanan, cloud sync, dll). Pendekatan berbasis aturan (*rule-based*) statis seringkali kesulitan mengikuti pola jaringan yang terus berubah. Oleh karena itu, *project* ini mengimplementasikan algoritma **K-Means Clustering** untuk menemukan struktur alami di dalam data lalu lintas jaringan tanpa bergantung penuh pada label (Unsupervised). 

Output utama sistem ini adalah pemahaman visual dan analitis mengenai profil lalu lintas jaringan, serta memisahkan trafik yang bersifat anomali untuk diinvestigasi lebih lanjut.

## 📦 Dataset

Dataset utama yang digunakan adalah **CICIDS2017**, sebuah *benchmark intrusion detection dataset* berformat *network flow*.
- **Ukuran Sample:** 100.000 rows (Stratified Random Sampling)
- **Fitur Terpilih:** 24 fitur numerik terbaik yang mendeskripsikan perilaku trafik (*behavioral profiling*), seperti `Flow Duration`, metrik *Packets/s*, ukuran *Header/Bytes*, dan statistik jeda kedatangan paket (*Inter-Arrival Time*).
- **Catatan:** Label jenis serangan (`Attack Type`) dari dataset asli tidak digunakan dalam fase *training*, melainkan hanya disimpan sebagai referensi untuk validasi silang pada tahap akhir interpretasi.

## 🛠️ Tech Stack

- **Bahasa Pemrograman:** Python 3
- **Framework Web/UI:** Streamlit
- **Data Preprocessing & Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (K-Means, PCA, StandardScaler, Metrics)
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Model Serialization:** Joblib

## 🚀 Fitur Aplikasi (Streamlit Multipage)

Aplikasi dibangun dengan format presentasi bertahap untuk menjelaskan alur *Machine Learning Pipeline* dari awal hingga akhir:

1. **🏠 Home (`app.py`)** — Ringkasan *project*, tujuan, daftar arsitektur, dan penjelasan parameter.
2. **📊 Exploratory Data Analysis (`pages/1_...`)** — Menampilkan ringkasan distribusi data (KDE), deteksi kemiringan data (*skewness*), *outlier* (Boxplot), dan hubungan antar fitur (Korelasi).
3. **⚙️ Data Preprocessing (`pages/2_...`)** — Proses pembersihan nilai *missing*/*infinity*, *Feature Selection*, *Log Transformation*, dan *Scaling* (*Standardization*).
4. **🤖 Training (`pages/3_...`)** — Melatih algoritma **K-Means** dengan kebebasan mencoba berbagai nilai parameter jumlah cluster (*k*).
5. **📏 Evaluation (`pages/4_...`)** — Membandingkan kualitas pemisahan *cluster* menggunakan *Silhouette Score*, *Davies-Bouldin Index*, dan *Calinski-Harabasz Index*.
6. **💡 Result & Interpretation (`pages/5_...`)** — Menampilkan hasil persebaran data dalam visualisasi **PCA 2D**, serta tabel karakteristik dominan (*profiling*) untuk setiap *cluster*.
7. **🔮 Input Network Traffic (`pages/6_...`)** — Halaman simulasi *real-time* di mana pengguna bisa meng-input suatu data aliran jaringan baru, dan melihat model menempatkannya ke dalam *cluster* yang paling sesuai.

## 💻 Panduan Instalasi & Menjalankan Aplikasi

1. **Clone repositori ini** (atau ekstrak *file* `.zip` proyek ke komputer Anda):
   ```bash
   git clone https://github.com/username/clustering-network-traffic.git
   cd clustering-network-traffic
   ```

2. **Buat Virtual Environment (Opsional namun direkomendasikan):**
   ```bash
   python -m venv venv
   # Pengguna Windows:
   venv\Scripts\activate
   # Pengguna Mac/Linux:
   source venv/bin/activate
   ```

3. **Install *Dependencies*:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka pada *browser* Anda (biasanya di `http://localhost:8501`).