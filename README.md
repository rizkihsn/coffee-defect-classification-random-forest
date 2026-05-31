# ☕ CoffeeDefect AI - Coffee Bean Defect Classifier

Sistem klasifikasi cacat biji kopi hijau (*green coffee beans*) berbasis Artificial Intelligence (AI) kelas industri dengan antarmuka web modern premium (*futuristic glassmorphism dark mode*). Aplikasi ini dirancang menggunakan algoritma **Random Forest Classifier** yang diintegrasikan dengan pemrosesan gambar tingkat tinggi (**Computer Vision** menggunakan OpenCV & Scikit-Image) untuk mengklasifikasikan **17 jenis cacat fisik biji kopi**.

Proyek ini dibangun sebagai tugas besar/tugas kuliah praktis mata kuliah **Artificial Intelligence** dengan fokus pada performa model yang andal (*fully working*), arsitektur modular (*production-ready*), serta desain UI/UX kelas dunia yang sangat cocok untuk presentasi dosen maupun demo YouTube.

---

## 📸 Fitur Unggulan

1. **Dashboard UI Premium (Dark Mode & Glassmorphism)**
   * Desain visual modern menggunakan harmoni warna Navy & Gold dengan aksen pendaran halus (*glow effect*).
   * Animasi latar belakang interaktif *Gold Particle Network* yang dinamis.
   * Efek transisi halaman yang halus (*smooth page transition*).

2. **Deteksi Gambar Real-time (Drag & Drop)**
   * Unggah gambar instan menggunakan fitur seret-dan-lepas (*drag & drop*) atau klik dengan validasi file.
   * Pratinjau gambar (*live preview*) sebelum prediksi dikirimkan ke server.
   * Animasi loading bertema cangkir kopi berputar saat model sedang memproses gambar.

3. **Pipelines Ekstraksi Fitur Tingkat Tinggi (451 Dimensi)**
   * **Shape & Texture**: HOG (*Histogram of Oriented Gradients*) sebanyak 324 fitur.
   * **Surface Texture**: LBP (*Local Binary Pattern*) sebanyak 59 fitur.
   * **Edge Density**: Canny Edge Detector sebanyak 33 fitur.
   * **Color Distribution**: Color Histogram sebanyak 32 fitur intensitas warna.
   * **Morphology**: Pengukuran geometris Biji Kopi (Area, Perimeter, Sirkularitas) sebanyak 3 fitur.

4. **Model Machine Learning Teroptimasi**
   * **Random Forest Classifier** dengan 200 Decision Trees.
   * Penanganan ketidakseimbangan kelas (*class imbalance*) menggunakan bobot seimbang (*balanced class weights*).
   * Kecepatan prediksi sangat tinggi (< 100ms per gambar).

5. **Analytics & Performance Dashboard Lengkap**
   * Metrik performa interaktif: Akurasi (Test & Train), Presisi, Recall, dan F1-Score.
   * **Confusion Matrix Heatmap** dinamis dalam bentuk sel gradien intensitas warna.
   * Grafik **Feature Importance** interaktif (Top 20 fitur paling berpengaruh) & kontribusi grup fitur menggunakan **Chart.js**.
   * Detail laporan klasifikasi (*classification report*) per-kelas yang presisi.

6. **Riwayat Prediksi & Unduh Hasil**
   * Menyimpan riwayat prediksi terakhir secara lokal dengan pratinjau mini.
   * Fitur mengunduh ringkasan hasil deteksi dalam file teks `.txt` resmi.

---

## 🛠️ Arsitektur Proyek (Modular & Bersih)

Struktur direktori proyek dirapikan agar sesuai dengan standar pengembangan perangkat lunak modern:

```
coffe/
├── models/                     # Tempat penyimpanan model dan metadata latih
│   ├── coffee_defect_model.pkl # Bobot model Random Forest (Pickle)
│   ├── model_metadata.json    # Metadata model, metrik, & nama kelas
│   └── visualizations/        # Visualisasi grafik performa hasil latih (.png)
├── static/                     # Aset front-end statis
│   ├── css/
│   │   └── style.css          # Desain sistem premium dark mode & glassmorphism
│   └── js/
│       ├── main.js            # Logika UI global, status server, & notifikasi toast
│       └── particles.js       # Efek animasi latar belakang partikel emas
├── templates/                  # Template HTML Jinja2 Flask
│   ├── base.html              # Layout dasar aplikasi global (Navbar, Footer, Partikel)
│   ├── index.html             # Landing page interaktif & informasi proyek
│   ├── dashboard.html         # Halaman deteksi utama (Classifier)
│   ├── analytics.html         # Panel performa model (Heatmap & Chart.js)
│   └── about.html             # Latar belakang ilmiah & 17 daftar cacat
├── uploads/                    # Folder penyimpanan gambar yang diunggah sementara
├── utils/                      # Package utilitas pemrosesan gambar & model
│   ├── __init__.py            # Inisialisasi python package
│   └── preprocessing.py       # Kelas FeatureExtractor (HOG, LBP, Color, Edge, Morph)
├── .gitignore                  # Berkas pengecualian Git
├── app.py                      # Flask Application Server (Entrypoint utama)
├── train_model.py              # Script training & evaluasi model mandiri
├── predict.py                  # Standalone CLI untuk testing prediksi gambar tunggal
├── requirements.txt            # Daftar dependensi modul python 3.12+
├── runtime.txt                 # Spesifikasi versi Python untuk cloud deployment
└── vercel.json                 # Konfigurasi deploy Vercel
```

---

## 📂 17 Jenis Cacat Biji Kopi yang Didukung

Sistem mampu mengklasifikasikan jenis biji kopi normal atau memiliki salah satu cacat fisik berikut sesuai dataset SCA (Specialty Coffee Association):
1. **Broken** (Pecah)
2. **Cut** (Terpotong)
3. **Dry Cherry** (Beri Kering)
4. **Fade** (Luntur)
5. **Floater** (Mengapung)
6. **Full Black** (Hitam Total)
7. **Full Sour** (Asam Total)
8. **Fungus Damage** (Serangan Jamur)
9. **Husk** (Kulit Ari)
10. **Immature** (Muda)
11. **Parchment** (Kulit Tanduk)
12. **Partial Black** (Hitam Sebagian)
13. **Partial Sour** (Asam Sebagian)
14. **Severe Insect Damage** (Lubang Serangga Parah)
15. **Shell** (Cangkang/Kuping)
16. **Slight Insect Damage** (Lubang Serangga Ringan)
17. **Withered** (Keriput)

---

## 🚀 Cara Menjalankan Secara Lokal (Windows / macOS / Linux)

### 1. Kloning atau Unduh Repositori Ini
Ekstrak folder proyek ke direktori kerja Anda (misalnya `C:\Users\Username\Downloads\coffe`).

### 2. Aktifkan Virtual Environment (Opsional namun Sangat Direkomendasikan)
Di terminal / PowerShell Anda, jalankan:
```bash
# Windows (PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependensi yang Dibutuhkan
Jalankan perintah berikut untuk menginstal semua modul python (kompatibel penuh dengan Python 3.12+):
```bash
pip install -r requirements.txt
```

Untuk melatih ulang model (butuh matplotlib, pandas, dll.):
```bash
pip install -r requirements-train.txt
```

### 4. Melatih Model (Training Model)
Jika Anda ingin melatih ulang model menggunakan dataset yang ada di folder `./archive`:
```bash
python train_model.py
```
*Catatan: Proses ini akan mengekstrak fitur gambar secara otomatis, melatih RandomForest Classifier, menyimpan model baru di folder `models/`, serta membuat grafik evaluasi.*

### 5. Menjalankan Server Web Flask
Mulai server Flask lokal Anda dengan menjalankan:
```bash
python app.py
```

Buka browser Anda dan akses halaman web di alamat:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🖥️ Uji Coba Model Via CLI (Command Line Interface)

Anda juga bisa melakukan prediksi gambar tunggal secara instan langsung dari terminal tanpa membuka web browser menggunakan berkas `predict.py`:
```bash
python predict.py --image path/ke/gambar_biji_kopi.jpg
```
Output terminal akan memunculkan:
* Hasil klasifikasi teratas (terprediksi).
* Nilai confidence score (persentase keyakinan).
* Daftar 3 kandidat cacat terkuat beserta probabilitasnya.

---

## ☁️ Deployment ke Platform Cloud (Render / Heroku)

Proyek ini telah dikonfigurasi untuk langsung di-deploy ke **Render** atau **Heroku** dengan langkah mudah:

1. **Unggah Proyek ke GitHub**: Buat repositori baru di GitHub dan lakukan commit serta push repositori lokal Anda.
2. **Koneksikan ke Render**:
   * Masuk ke dashboard Render, lalu pilih **New Web Service**.
   * Hubungkan repositori GitHub proyek ini.
   * Konfigurasikan pengaturan berikut:
     * **Environment**: `Python`
     * **Build Command**: `pip install -r requirements.txt`
     * **Start Command**: `gunicorn app:app`
     * **Plan**: Pilih Free (atau berbayar jika butuh memori lebih besar).
3. **Selesai!** Render akan otomatis membangun aplikasi Anda sesuai instruksi di berkas `Procfile` dan `runtime.txt`.

---

## 🎓 Kontributor & Tujuan Akademik
* **Tujuan**: Tugas Praktik Mata Kuliah Artificial Intelligence
* **Topik**: Penerapan Algoritma Random Forest untuk Klasifikasi Gambar Berbasis Web
* **Studi Kasus**: Deteksi Mutu & Cacat Fisik Biji Kopi Hijau (*Coffee Green Bean Defects*)
* **Teknologi**: Random Forest + OpenCV Feature Extractors (HOG + LBP + Color + Edge + Morphology) + Web App Flask (Bootstrap 5 & Premium Glassmorphism)

---
*Kopi yang nikmat berawal dari biji kopi yang bersih dan berkualitas tinggi. Let's make coffee selection smarter with AI!* ☕🚀
