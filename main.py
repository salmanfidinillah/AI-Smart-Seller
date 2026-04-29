import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("🔥 MEMULAI PROSES MACHINE LEARNING 🔥\n")

# ==========================================
# STEP 1: LOAD & MERGE DATA
# ==========================================
print("1️⃣ Loading & Merging Data...")
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
orders = pd.read_csv('data/olist_orders_dataset.csv')

df = order_items.merge(products, on='product_id', how='left')
df = df.merge(orders, on='order_id', how='left')

# Ambil kolom penting & hapus data kosong
df = df[['product_id', 'price', 'freight_value', 'product_category_name', 'order_purchase_timestamp']].dropna()

# ==========================================
# STEP 2: FEATURE ENGINEERING
# ==========================================
print("2️⃣ Feature Engineering...")

product_stats = df.groupby('product_id').agg(
    avg_price=('price', 'mean'),
    avg_freight=('freight_value', 'mean'),
    total_sold=('product_id', 'count'),
    category=('product_category_name', 'first')
).reset_index()

median_sales = product_stats['total_sold'].median()
product_stats['laris'] = (product_stats['total_sold'] > median_sales).astype(int)

le = LabelEncoder()
product_stats['category_encoded'] = le.fit_transform(product_stats['category'])

# ==========================================
# STEP 3: SPLIT DATA
# ==========================================
print("3️⃣ Splitting Data...")

X = product_stats[['avg_price', 'avg_freight', 'category_encoded']]
y = product_stats['laris']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: MODELING (RANDOM FOREST)
# ==========================================
print("4️⃣ Training Model...\n")
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("📊 HASIL EVALUASI MODEL:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ==========================================
# STEP 5: EDA & FEATURE IMPORTANCE
# ==========================================
print("\n5️⃣ Analisis Model & Visualisasi...")

importances = model.feature_importances_
features = X.columns

print("\n🧠 INTERPRETASI FEATURE IMPORTANCE:")
for f, imp in zip(features, importances):
    print(f"👉 {f} berpengaruh sebesar {imp*100:.1f}% terhadap kelarisan produk.")

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title("Fitur Apa yang Paling Mempengaruhi Kelarisan? 🔥")
plt.xlabel("Tingkat Kepentingan")
plt.ylabel("Fitur")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 6: INSIGHTS BISNIS 
# ==========================================
print("\n💡 INSIGHT BISNIS:")

# Rata-rata harga
avg_price_laris = product_stats[product_stats['laris'] == 1]['avg_price'].mean()
avg_price_tidak = product_stats[product_stats['laris'] == 0]['avg_price'].mean()
print(f"💰 Rata-rata harga produk LARIS       : Rp {avg_price_laris:,.0f}")
print(f"💰 Rata-rata harga produk TIDAK LARIS : Rp {avg_price_tidak:,.0f}")

# Kategori Paling Laris
top_category = product_stats.groupby('category')['total_sold'].sum().sort_values(ascending=False).head(3)
print("\n🏆 TOP 3 KATEGORI PALING LARIS (Total Penjualan):")
print(top_category.to_string())

# ==========================================
# STEP 7: SIMULASI PRODUK BARU 
# ==========================================
print("\n🚀 SIMULASI PRODUK BARU:")

kategori_dict = dict(zip(le.classes_, le.transform(le.classes_)))
kategori_target = 'beleza_saude' 

if kategori_target in kategori_dict:
    contoh_kategori = kategori_dict[kategori_target]
    contoh_harga = 150000
    contoh_ongkir = 20000
    
    print(f"\nSimulasi jualan produk di kategori: '{kategori_target}'")
    print(f"Harga: Rp {contoh_harga} | Ongkir: Rp {contoh_ongkir}")
    
    prediksi_baru = model.predict([[contoh_harga, contoh_ongkir, contoh_kategori]])

    if prediksi_baru[0] == 1:
        print("🔥 HASIL: Produk ini punya potensi besar untuk LARIS!")
    else:
        print("❌ HASIL: Produk ini kemungkinan susah bersaing (TIDAK LARIS).")

# ==========================================
# STEP 8: PENUTUP (THE KILLER CLOSING 🎯)
# ==========================================
print("\n📊 STRATEGI PRAKTIS:")
print("👉 Fokus pada kategori dengan volume penjualan tinggi")
print("👉 Gunakan harga kompetitif dibanding rata-rata kategori")
print("👉 Optimalkan ongkir untuk meningkatkan daya saing")

print("\n🎯 KESIMPULAN:")
print("Model ini dapat membantu UMKM menentukan strategi harga dan kategori produk secara data-driven untuk meningkatkan peluang penjualan.")