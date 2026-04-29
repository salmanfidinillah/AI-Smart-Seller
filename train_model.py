import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import warnings
import os

warnings.filterwarnings('ignore')

print("🔥 MEMULAI PROSES MACHINE LEARNING (LEVEL LOMBA) 🔥\n")

# ==========================================
# STEP 1: LOAD DATA INDONESIA
# ==========================================
print("1️⃣ Loading Data Indonesia...")

# MOCK DATA: Ini fungsi untuk membuat file dummy jika file aslinya belum ada
# NANTI HAPUS BAGIAN INI KALAU FILE indo.csv KAMU SUDAH SIAP
if not os.path.exists('data/indo.csv'):
    os.makedirs('data', exist_ok=True)
    dummy_data = """order_id;Harga;Ongkir;Diskon;Kategori;Status Pesanan;total_qty
    ORD001;150000;15000;10;Pakaian;Selesai;2
    ORD002;200000;20000;0;Elektronik;Batal;1
    ORD003;50000;10000;5;Makanan;Selesai;5
    ORD004;300000;25000;20;Elektronik;Selesai;1
    ORD005;75000;15000;0;Pakaian;Batal;2
    ORD006;120000;10000;15;Pakaian;Selesai;3
    """
    with open('data/indo.csv', 'w') as f:
        f.write(dummy_data)

# Baca data asli menggunakan separator ';' sesuai dataset kamu
df = pd.read_csv('data/indo.csv', sep=';')

# ==========================================
# STEP 2: BIKIN TARGET & FEATURE ENGINEERING
# ==========================================
print("2️⃣ Feature Engineering & Target...")

# Bikin target realistis: Kalau statusnya 'Selesai' berarti Laris/Berhasil (1), selain itu (0)
df['laris'] = df['Status Pesanan'].apply(lambda x: 1 if x.strip().lower() == 'selesai' else 0)

# Encode Kategori agar bisa dibaca oleh Machine Learning
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['Kategori'])

# Kita cari tau harga wajar (median) per kategori untuk jadi "Konteks Pasar"
category_medians = df.groupby('category_encoded').agg(
    med_price=('Harga', 'median'),
    med_freight=('Ongkir', 'median')
).reset_index()

# Gabungkan info pasar ke data utama
df = df.merge(category_medians, on='category_encoded', how='left')

# Bikin rasio untuk bantu model mikir: "Ini kemahalan nggak ya dibanding pasar?"
df['price_ratio'] = df['Harga'] / df['med_price']

print("\n📊 DISTRIBUSI TARGET SEBELUM DIBALANCE:")
print(df['laris'].value_counts())

# ==========================================
# STEP 3: SPLIT & BALANCE DATA
# ==========================================
print("\n3️⃣ Splitting & Balancing Data...")

# Fitur kita sekarang lebih canggih: Harga, Ongkir, Diskon, Kategori, dan Rasio Harga
fitur = ['Harga', 'Ongkir', 'Diskon', 'category_encoded', 'price_ratio']
X = df[fitur]
y = df['laris']

# Bagi data untuk belajar (train) dan ujian (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# OVERSAMPLING: Biar jumlah data laris dan tidak laris seimbang
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# ==========================================
# STEP 4: MODELING (RANDOM FOREST)
# ==========================================
print("\n4️⃣ Training Model...\n")
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
print("📊 HASIL EVALUASI MODEL (Kinerja Nyata):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

print("\n✅ Proses ML selesai! Sekarang kamu bisa jalankan api.py")