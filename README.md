#  AI Smart Seller Pro
##  Live Prediction Demo (Simulation inside main.py)

AI Smart Seller Pro adalah sistem berbasis Machine Learning yang membantu UMKM dalam memprediksi apakah suatu produk akan laris di pasaran berdasarkan data historis e-commerce.

---

##  Problem

Banyak UMKM mengalami kesulitan dalam:
- Menentukan harga optimal
- Mengelola strategi penjualan
- Memahami perilaku pasar

Akibatnya:
- Produk tidak laku
- Margin keuntungan rendah
- Kalah bersaing di marketplace

---

##  Solution

AI Smart Seller Pro menawarkan solusi berbasis data:

- Prediksi produk laris / tidak  
- Analisis harga dan ongkir  
- Insight bisnis berbasis data  
- Simulasi produk baru  

---

##  Machine Learning Approach

Model yang digunakan:
- Random Forest Classifier

Feature:
- Average Price
- Average Freight Cost
- Product Category (Encoded)

Target:
- Produk "laris" (berdasarkan median penjualan)

---

##  Dataset

Menggunakan Olist E-commerce Dataset:

- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_orders_dataset.csv`

---

##  How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python main.py
