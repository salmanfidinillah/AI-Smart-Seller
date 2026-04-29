# 🚀 AI Smart Seller – Market Intelligence System for UMKM Indonesia

AI Smart Seller adalah sistem berbasis Artificial Intelligence yang membantu pelaku UMKM di Indonesia untuk menganalisis potensi penjualan produk berdasarkan harga, diskon, ongkir, dan kategori pasar.

Sistem ini tidak hanya menggunakan Machine Learning, tetapi juga dikombinasikan dengan **Business Logic (Rule-Based System)** agar hasil tetap realistis dan applicable di dunia nyata.

---

## 🔥 Fitur Utama

✅ Prediksi potensi produk (Laris / Tidak Laris)  
✅ Analisis harga vs pasar kompetitor  
✅ Rekomendasi strategi harga & diskon  
✅ Dashboard visualisasi data (Chart.js)  
✅ API berbasis FastAPI  
✅ Sistem hybrid: AI + Business Logic  

---

## 🧠 Teknologi yang Digunakan

- Python (Machine Learning)
- FastAPI (Backend API)
- Random Forest Classifier
- Pandas & Scikit-learn
- HTML, CSS, JavaScript (Frontend)
- Chart.js (Dashboard)
- Imbalanced Learning (Oversampling)

---

## 📊 Cara Kerja Sistem

1. Data transaksi dianalisis
2. Dibuat fitur:
   - Harga
   - Ongkir
   - Diskon
   - Kategori
   - Price Ratio (dibanding pasar)
3. Model ML dilatih
4. Sistem menambahkan **Business Rules**:
   - Diskon tidak boleh tidak realistis (>50%)
   - Harga dibandingkan dengan pasar
   - Deteksi terlalu mahal / terlalu murah
5. Output:
   - Prediksi
   - Confidence score
   - Rekomendasi bisnis

---

## ⚙️ Cara Menjalankan Project

### 1. Install dependency
```bash
pip install -r requirements.txt
