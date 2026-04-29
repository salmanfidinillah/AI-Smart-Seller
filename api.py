from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import os

app = FastAPI()

# Izinkan Frontend untuk akses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD DATA & TRAINING SAAT SERVER NYALA
# ==========================================
if not os.path.exists('data/indo.csv'):
    os.makedirs('data', exist_ok=True)
    with open('data/indo.csv', 'w') as f:
        f.write("order_id;Harga;Ongkir;Diskon;Kategori;Status Pesanan;total_qty\nORD01;150000;15000;10;Pakaian;Selesai;2\nORD02;200000;20000;0;Elektronik;Batal;1\nORD03;50000;10000;5;Makanan;Selesai;5\nORD04;300000;25000;20;Elektronik;Selesai;1\nORD05;75000;15000;0;Pakaian;Batal;2")

df = pd.read_csv('data/indo.csv', sep=';')

# Buat Target
df['laris'] = df['Status Pesanan'].apply(lambda x: 1 if str(x).strip().lower() == 'selesai' else 0)

# Encode Kategori
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['Kategori'])

# Cari Konteks Pasar (Median Harga & Ongkir)
category_medians = df.groupby('category_encoded').agg(
    med_price=('Harga', 'median'),
    med_freight=('Ongkir', 'median')
).reset_index()

df = df.merge(category_medians, on='category_encoded', how='left')
df['price_ratio'] = df['Harga'] / df['med_price']

# Fitur dan Model
fitur = ['Harga', 'Ongkir', 'Diskon', 'category_encoded', 'price_ratio']
X = df[fitur]
y = df['laris']

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_res, y_res)

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def home():
    return {"message": "AI Smart Seller API v5 (Real Business Logic) is running 🚀"}

# Endpoint untuk Prediksi
@app.get("/predict")
def predict(price: float, freight: float, discount: float, category: str):
    # 1. Cari Data Pasar Kompetitor
    if category in le.classes_:
        cat_encoded = le.transform([category])[0]
    else:
        cat_encoded = 0 
        
    med_data = category_medians[category_medians['category_encoded'] == cat_encoded]
    
    if not med_data.empty:
        optimal_price = med_data['med_price'].values[0]
        optimal_freight = med_data['med_freight'].values[0]
    else:
        optimal_price = category_medians['med_price'].median()
        optimal_freight = category_medians['med_freight'].median()

    # 2. Perhitungan Logika Bisnis (Akal Sehat)
    # Harga setelah diskon ditambah ongkir
    user_final_price = (price - (price * (discount / 100))) + freight
    # Harga pasar total (kita asumsikan kompetitor jarang pakai diskon, pakai harga normal + ongkir)
    competitor_final_price = optimal_price + optimal_freight

    price_ratio = price / optimal_price if optimal_price > 0 else 1.0

    # Prediksi dasar dari ML
    input_data = [[price, freight, discount, cat_encoded, price_ratio]]
    proba = model.predict_proba(input_data)[0][1]
    
    # 🔥 3. RULE-BASED GUARDRAIL (BATASAN LOGIKA SESUAI PERMINTAANMU) 🔥
    
    # Cek diskon ngaco dulu
    if discount > 50:
        hasil = "⚠️ DISKON TIDAK REALISTIS"
        proba = 0.10 # Peluang sukses rendah karena berisiko bangkrut
        rekomendasi = "Diskon di atas 50% merusak harga pasar dan profitmu. Kurangi diskon maksimal ke batas wajar 20-30%."
        
    # Cek apakah lebih mahal dari kompetitor
    elif user_final_price > competitor_final_price:
        hasil = "❌ SULIT BERSAING"
        proba = min(proba, 0.40) # Maksimal peluang cuma 40%
        selisih = user_final_price - competitor_final_price
        rekomendasi = f"Total belanjamu (Rp {user_final_price:,.0f}) lebih mahal Rp {selisih:,.0f} dari pasar. Naikkan diskon wajar (max 20%) atau cari ongkir lebih murah."
        
    # Jika lebih murah, cek kewajarannya (jangan sampai beda 50% ke bawah)
    else:
        # Batas wajar adalah setengah harga dari kompetitor
        batas_wajar_bawah = competitor_final_price * 0.5 
        
        if user_final_price < batas_wajar_bawah:
            hasil = "⚠️ TERLALU MURAH (TIDAK SEHAT)"
            proba = 0.50 # Bisa jadi orang curiga barang palsu
            rekomendasi = f"Total hargamu terlalu murah, jauh di bawah pasar (Rp {competitor_final_price:,.0f}). Awas rugi atau pembeli curiga kualitas. Naikkan harga sedikit."
        else:
            hasil = "🔥 POTENSI LARIS"
            proba = max(proba, 0.80) # Pasti laku karena lebih murah tapi masih wajar
            rekomendasi = f"Strategi sempurna! Total hargamu (Rp {user_final_price:,.0f}) bersaing dengan pasar (Rp {competitor_final_price:,.0f}) dalam batas aman."

    return {
        "prediction": hasil,
        "confidence": float(proba),
        "optimal_price": float(optimal_price), # Harga barang pasar aja buat info
        "recommendation": rekomendasi
    }

# Endpoint untuk Dashboard
@app.get("/stats")
def get_stats():
    cat_counts = df['Kategori'].value_counts().to_dict()
    status_counts = {"Laris (Selesai)": int(sum(df['laris'] == 1)), "Tidak Laris (Batal)": int(sum(df['laris'] == 0))}
    
    return {
        "categories": list(cat_counts.keys()),
        "category_totals": list(cat_counts.values()),
        "status_labels": list(status_counts.keys()),
        "status_totals": list(status_counts.values())
    }