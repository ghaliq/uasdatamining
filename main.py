import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATASET_PATH = "Regression.csv"

st.markdown("""
      <style>
        /* Gaya font untuk seluruh aplikasi */
         body {
            background-color: #F4F6F9;
            font-family: 'Roboto', sans-serif;
        }

        /* Gaya font untuk judul halaman */
       .css-1d391kg {
            font-family: 'Poppins', sans-serif;
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }
            
        /* Gaya font untuk elemen sidebar */
        .sidebar .sidebar-content {
            background: #f4f7fa;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            font-family: 'Courier New', monospace;
        }

        /* Gaya font untuk teks di dalam sidebar */
        .sidebar .sidebar-content h3 {
            font-size: 18px;
            font-weight: bold;
            color: #333333;
        }

        /* Gaya font untuk subheader di Streamlit */
        .css-1v3fvcr {
            font-family: 'Georgia', serif;
            font-weight: normal;
            font-size: 24px;
            color: #3F51B5;
        }
             
        .css-1v3fvcr {
            font-family: 'Georgia', serif;
            font-weight: normal;
            font-size: 24px;
            color: #3F51B5;
        }

     
        .css-1emrehk {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }

        /* Gaya untuk tabel */
        .stDataFrame {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        /* Styling untuk grafik */
        .stGraph {
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
            
          .stButton>button {
            background-color: #4CAF50;  /* Warna hijau */
            color: white;  /* Teks berwarna putih */
            font-size: 16px;
            font-weight: bold;  /* Teks tebal */
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;  /* Warna hijau sedikit lebih gelap saat hover */
            color: white;  /* Pastikan teks tetap putih saat hover */
        }
        .stButton>button:active {
            background-color: #2196F3;  /* Warna biru saat tombol diklik */
            color: white;  /* Pastikan teks tetap putih saat tombol diklik */
        }
    </style>
""", unsafe_allow_html=True)

st.title("Selamat datang di Aplikasi Prediksi Biaya Asuransi")

st.write("""
## 🎯 **Tentang Aplikasi Prediksi Biaya Asuransi**

Aplikasi ini dirancang untuk memprediksi biaya asuransi kesehatan berdasarkan beberapa faktor penting. 
Berikut adalah fitur utama dari aplikasi ini:

### 📋 **Fitur Utama:**
1. **Prediksi Biaya Asuransi**:
   - Faktor yang dipertimbangkan:
     - ✅ **Umur**: Usia pengguna.
     - ✅ **BMI (Indeks Massa Tubuh)**: Indikator berat badan pengguna.
     - ✅ **Jumlah Anak**: Jumlah anak tanggungan pengguna.
     - ✅ **Kebiasaan Merokok**: Status perokok pengguna.
     - ✅ **Wilayah Tempat Tinggal**: Area geografis pengguna.
2. **Dua Model Machine Learning**:
   - 📈 **Polynomial Regression**: Memberikan prediksi berbasis hubungan non-linear.
   - 🌲 **Random Forest**: Model berbasis pohon keputusan yang sangat akurat.
   - 🚀 **Gradient Boosting**: Model berbasis ensemble yang menggabungkan beberapa model pohon keputusan untuk meningkatkan akurasi prediksi.
3. **Konversi ke Mata Uang Lokal**:
   - 🔄 Konversi prediksi biaya asuransi dari **USD ke IDR** berdasarkan nilai tukar yang dapat diatur pengguna.
3. **Visualisasi Data Interaktif**:
   - 📊 Grafik dan plot untuk membantu memahami performa model dan pola dalam data.

### 🔍 **Cara Menggunakan**:
1. Masukkan data Anda di sidebar aplikasi. 🖊️
2. Klik tombol **Prediksi** untuk melihat hasil prediksi biaya asuransi. 🚀
3. Periksa hasil prediksi dan analisis visualisasi untuk mendapatkan wawasan lebih lanjut. 📉

### 🎨 **Mengapa Aplikasi Ini Penting?**
Aplikasi ini membantu pengguna memahami estimasi biaya asuransi berdasarkan data pribadi mereka dan memberikan wawasan yang lebih mendalam tentang pola dalam dataset terkait asuransi kesehatan. Dengan visualisasi interaktif dan hasil prediksi yang akurat, pengguna dapat membuat keputusan yang lebih baik.

---
""")

st.image("https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif", width=600, caption="Prediksi Biaya Asuransi")



try:
    data = pd.read_csv(DATASET_PATH)
    st.write("Tabel Dataset")
    
    st.dataframe(data, use_container_width=True, height=400)  
    
except FileNotFoundError:
    st.error(f"File dataset '{DATASET_PATH}' tidak ditemukan. Pastikan file berada di folder yang sama dengan script ini. :x:")
    st.stop()




encoded_data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
X = encoded_data.drop('charges', axis=1)
y = encoded_data['charges']


gender_data = {
    "Jenis Kelamin": ["Laki-laki", "Perempuan"],
    "Jumlah": [120, 80],
}


smoker_data = {
    "Status Merokok": ["Perokok", "Bukan Perokok"],
    "Jumlah": [90, 110],
}


gender_df = pd.DataFrame(gender_data)
smoker_df = pd.DataFrame(smoker_data)


st.subheader("📊 Diagram Pie: Jenis Kelamin")
fig_gender, ax_gender = plt.subplots()


explode = (0.08, 0)  

ax_gender.pie(
    gender_df["Jumlah"],
    labels=gender_df["Jenis Kelamin"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["#4CAF50", "#FFC107"],
    explode=explode, 
)

ax_gender.set_title("Distribusi Jenis Kelamin")


st.pyplot(fig_gender)

st.subheader("📊 Diagram Pie: Distribusi Perokok")
fig_smoker, ax_smoker = plt.subplots()


explode_smoker = (0.08, 0) 

ax_smoker.pie(
    smoker_df["Jumlah"],
    labels=smoker_df["Status Merokok"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["#2196F3", "#FF5722"],
    explode=explode_smoker,  
)

ax_smoker.axis("equal")  
st.pyplot(fig_smoker)

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Membangun dan melatih model-model
poly = PolynomialFeatures(degree=3) 
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Polynomial Regression Model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Random Forest Model
random_forest_model = RandomForestRegressor(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# Gradient Boosting Model
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gradient_boosting_model.fit(X_train, y_train)

# Prediksi dari setiap model
poly_pred = poly_model.predict(X_test_poly)
rf_pred = random_forest_model.predict(X_test)
gb_pred = gradient_boosting_model.predict(X_test)

# Prediksi gabungan
combined_pred = (poly_pred + rf_pred) / 2

# Menghitung metrik untuk setiap model
mse_poly = mean_squared_error(y_test, poly_pred)
mse_rf = mean_squared_error(y_test, rf_pred)
mse_gb = mean_squared_error(y_test, gb_pred)
mse_combined = mean_squared_error(y_test, combined_pred)

mae_poly = mean_absolute_error(y_test, poly_pred)
mae_rf = mean_absolute_error(y_test, rf_pred)
mae_gb = mean_absolute_error(y_test, gb_pred)
mae_combined = mean_absolute_error(y_test, combined_pred)

rmse_poly = np.sqrt(mse_poly)
rmse_rf = np.sqrt(mse_rf)
rmse_gb = np.sqrt(mse_gb)
rmse_combined = np.sqrt(mse_combined)

r2_poly = r2_score(y_test, poly_pred)
r2_rf = r2_score(y_test, rf_pred)
r2_gb = r2_score(y_test, gb_pred)
r2_combined = r2_score(y_test, combined_pred)

# Tabel hasil evaluasi model
data_evaluasi = {
    "Model": ["Polynomial Regression", "Random Forest", "Gradient Boosting", "Model Gabungan"],
    "MAE": [mae_poly, mae_rf, mae_gb, mae_combined],
    "MSE": [mse_poly, mse_rf, mse_gb, mse_combined],
    "R²": [r2_poly, r2_rf, r2_gb, r2_combined],
}

st.subheader("Hasil Evaluasi Model :bar_chart:")
df_evaluasi = pd.DataFrame(data_evaluasi)

# Menampilkan tabel dengan format dan pewarnaan
st.dataframe(
    df_evaluasi.style.format({
        "MSE": "{:.2f}",
        "R²": "{:.2f}",
    }).background_gradient(cmap="Blues", subset=["MSE", "R²"])
)

# Tabel hasil akurasi dengan RMSE untuk model tertentu
results = {
    "Model": ["Polynomial Regression", "Random Forest", "Gradient Boosting", "Model Gabungan"],
    "MAE": [mae_poly, mae_rf, mae_gb, mae_combined],
    "RMSE": [rmse_poly, rmse_rf, rmse_gb, rmse_combined]
}

results_df = pd.DataFrame(results)
st.write("### Hasil Akurasi Model", results_df)


def regression_classification_report(y_true, y_pred):
    residual = y_true - y_pred
    threshold = residual.std()  # Menentukan threshold berdasarkan standar deviasi residual
    residual_class = (abs(residual) < threshold).astype(int)  # Menentukan apakah prediksi dalam kategori 'Inlier' atau 'Outlier'
    y_class_true = np.ones_like(residual_class)  # Nilai target untuk 'Inlier' semua bernilai 1
    report = classification_report(y_class_true, residual_class, target_names=['Outlier', 'Inlier'], output_dict=True)
    return report

# Mendapatkan laporan klasifikasi untuk masing-masing model
poly_report = regression_classification_report(y_test, poly_model.predict(X_test_poly))
random_report = regression_classification_report(y_test, random_forest_model.predict(X_test))
gb_report = regression_classification_report(y_test, gradient_boosting_model.predict(X_test))

# Menyusun data untuk tabel Precision, Recall, dan F1-Score
data_precision_recall = {
    "Model": ["Polynomial Regression", "Random Forest", "Gradient Boosting"],
    "Precision": [poly_report["Inlier"]["precision"], random_report["Inlier"]["precision"], gb_report["Inlier"]["precision"]],
    "Recall": [poly_report["Inlier"]["recall"], random_report["Inlier"]["recall"], gb_report["Inlier"]["recall"]],
    "F1-Score": [poly_report["Inlier"]["f1-score"], random_report["Inlier"]["f1-score"], gb_report["Inlier"]["f1-score"]],
}

# Menampilkan tabel Precision, Recall, dan F1-Score
st.subheader("Precision, Recall, F1-Score :star:")
df_precision_recall = pd.DataFrame(data_precision_recall)
st.dataframe(
    df_precision_recall.style.format({
        "Precision": "{:.2f}",
        "Recall": "{:.2f}",
        "F1-Score": "{:.2f}",
    }).background_gradient(cmap="Greens", subset=["Precision", "Recall", "F1-Score"])
)

st.subheader("Tambahan Evaluasi dengan Print :clipboard:")

# Prediksi untuk masing-masing model
y_predPoly = poly_model.predict(X_test_poly)
y_predRandom = random_forest_model.predict(X_test)
y_predGB = gradient_boosting_model.predict(X_test)

# Evaluasi untuk Polynomial Regression
st.text("Polynomial Regression")
st.text(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_predPoly):.2f}")
st.text(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_predPoly):.2f}")
st.text(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_predPoly)):.2f}")

# Evaluasi untuk Random Forest Regression
st.text("Random Forest Regression")
st.text(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_predRandom):.2f}")
st.text(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_predRandom):.2f}")
st.text(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_predRandom)):.2f}")

# Evaluasi untuk Gradient Boosting Regression
st.text("Gradient Boosting Regression")
st.text(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_predGB):.2f}")
st.text(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_predGB):.2f}")
st.text(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_predGB)):.2f}")



st.subheader("Analisis Rinci Data :mag:")

st.write("Distribusi Data (Histogram + KDE Plot) ")
fig, ax = plt.subplots()
sns.histplot(y, kde=True, ax=ax, color="blue", bins=30, stat="density", edgecolor="black", alpha=0.6)
sns.kdeplot(y, fill=True, ax=ax, color="DarkTurquoise", alpha=0.4)
ax.set_title("Distribusi Biaya Asuransi")
ax.set_xlabel("Biaya Asuransi")
ax.set_ylabel("Density")
st.pyplot(fig)

st.write("Heatmap Korelasi ")
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = encoded_data.corr()


sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Spectral", 
    ax=ax,
    linewidths=0.5,  
    linecolor='gray',  
    cbar_kws={'shrink': 0.8}  
)

ax.set_title("Korelasi Antar Fitur", fontsize=16)
st.pyplot(fig)

st.write("Visualisasi Catplot :bar_chart:")
fig = sns.catplot(
    data=data, 
    x="smoker", 
    y="charges", 
    hue="sex", 
    kind="box", 
    aspect=2,
    palette="Set2" 
)

fig.set_axis_labels("Perokok", "Biaya Asuransi")
fig.fig.suptitle("Biaya Asuransi Berdasarkan Kebiasaan Merokok dan Jenis Kelamin", fontsize=16)
st.pyplot(fig)


gb_pred = gradient_boosting_model.predict(X_test)
st.write("Garis Regresi Sempurna (Ideal) ")
fig, ax = plt.subplots()

ax.scatter(y_test, poly_pred, color='dodgerblue', alpha=0.7, label='Polynomial Regression')
ax.scatter(y_test, rf_pred, color='MediumVioletRed', alpha=0.7, label='Random Forest')
ax.scatter(y_test, gb_pred, color='gold', alpha=0.7, label='Gradient Boosting')


ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='darkorange', lw=2, label='Perfect Prediction')


ax.set_xlabel("Biaya Asli", fontsize=12)
ax.set_ylabel("Biaya Prediksi", fontsize=12)
ax.set_title("Perbandingan Prediksi vs Realita", fontsize=14)

ax.legend()

st.pyplot(fig)

st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background: #f4f7fa;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content .stTextInput, 
        .sidebar .sidebar-content .stSlider, 
        .sidebar .sidebar-content .stSelectbox {
            margin-bottom: 10px;
        }
        .sidebar .sidebar-content h3 {
            margin-top: 0;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns([1, 5]) 

with col1:
    st.image("logo.jpg", width=50)

with col2:

    st.markdown("<h2 style='text-align: left;'>Sidebar</h2>", unsafe_allow_html=True)


def user_input_features():
    age = st.sidebar.slider("Umur", 18, 100, 30)
    sex = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    bmi = st.sidebar.slider("Indeks Massa Tubuh (BMI)", 10.0, 50.0, 25.0)
    children = st.sidebar.slider("Jumlah Anak", 0, 10, 1)
    smoker = st.sidebar.selectbox("Perokok", ["Ya", "Tidak"])
    region = st.sidebar.selectbox("Wilayah", ["Northeast", "Northwest", "Southeast", "Southwest"])

    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "Laki-laki" else 0,
        "smoker_yes": 1 if smoker == "Ya" else 0,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0,
    }

    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0

    return pd.DataFrame(input_data, index=[0])


input_df = user_input_features()


st.sidebar.header("Tolong Input mata uang untuk di konversasi")


exchange_rate_str = st.sidebar.text_input("Masukkan Nilai Tukar 1 USD ke IDR (Rp)", value="Rp 16201.924")


if exchange_rate_str.startswith("Rp"):
    try:
        exchange_rate = float(exchange_rate_str[2:].replace(",", "").strip())
    except ValueError:
        exchange_rate = 16201.924 
else:
    exchange_rate = 16201.924


st.sidebar.write(f"Nilai tukar: 1 USD = {exchange_rate} IDR")


predict_button = st.sidebar.button("Lakukan Prediksi ")
  
st.subheader("Data yang Dipilih Pengguna")
input_data_display = {
    "Umur": input_df["age"][0],
    "Jenis Kelamin": "Laki-laki" if input_df["sex_male"][0] == 1 else "Perempuan",
    "BMI": input_df["bmi"][0],
    "Jumlah Anak": input_df["children"][0],
    "Perokok": "Ya" if input_df["smoker_yes"][0] == 1 else "Tidak",
    "Wilayah": input_df[["region_northwest", "region_southeast", "region_southwest"]].idxmax(axis=1)[0].replace("region_", "").capitalize()
}


st.table(pd.DataFrame(list(input_data_display.items()), columns=["Feature", "Value"]))

if predict_button:
    
    # Prediksi untuk masing-masing model
    input_poly = poly.transform(input_df)
    poly_pred = poly_model.predict(input_poly)[0]
    rf_pred = random_forest_model.predict(input_df)[0]
    gb_pred = gradient_boosting_model.predict(input_df)[0]
    combined_pred = (poly_pred + rf_pred + gb_pred) / 3

    # Menampilkan hasil prediksi dalam USD
    st.subheader("Hasil Prediksi Biaya (USD) :dollar:")
    st.write(f"Prediksi Polynomial Regression: ${poly_pred:.2f}")
    st.write(f"Prediksi Random Forest: ${rf_pred:.2f}")
    st.write(f"Prediksi Gradient Boosting: ${gb_pred:.2f}")
    st.write(f"Prediksi Gabungan: ${combined_pred:.2f}")

    # Mengkonversi hasil prediksi ke IDR
    poly_pred_idr = poly_pred * exchange_rate
    rf_pred_idr = rf_pred * exchange_rate
    gb_pred_idr = gb_pred * exchange_rate
    combined_pred_idr = combined_pred * exchange_rate

    # Menampilkan hasil prediksi dalam IDR
    st.subheader("Hasil Prediksi Biaya (IDR) :money_with_wings:")
    st.write(f"Prediksi Polynomial Regression: Rp{poly_pred_idr:,.2f}")
    st.write(f"Prediksi Random Forest: Rp{rf_pred_idr:,.2f}")
    st.write(f"Prediksi Gradient Boosting: Rp{gb_pred_idr:,.2f}")
    st.write(f"Prediksi Gabungan: Rp{combined_pred_idr:,.2f}")

    # Menampilkan Tabel Hasil Prediksi
    st.subheader("Tabel Hasil Prediksi")
    prediction_data = {
        "Metode": ["Polynomial Regression", "Random Forest", "Gradient Boosting", "Gabungan"],
        "Prediksi (IDR)": [
            f"Rp{poly_pred_idr:,.2f}",
            f"Rp{rf_pred_idr:,.2f}",
            f"Rp{gb_pred_idr:,.2f}",
            f"Rp{combined_pred_idr:,.2f}",
        ]
    }
    
    prediction_df = pd.DataFrame(prediction_data)
    st.table(prediction_df)

    # Menampilkan grafik perbandingan prediksi dalam IDR
    fig, ax = plt.subplots()
    methods = ["Polynomial Regression", "Random Forest", "Gradient Boosting", "Gabungan"]
    predictions = [poly_pred_idr, rf_pred_idr, gb_pred_idr, combined_pred_idr]

    ax.bar(methods, predictions, color=['cyan', 'CornflowerBlue', 'MediumOrchid', 'DarkRed'])

    ax.tick_params(axis='y', labelrotation=90) 

    for i, pred in enumerate(predictions):
        ax.text(i, pred + 200000, f'Rp{pred:,.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Prediksi Biaya (IDR)")
    ax.set_title("Perbandingan Prediksi (Dalam Rupiah)")

    st.pyplot(fig)
    
    # Perbandingan dan Menentukan Metode Terbaik
    st.subheader("Perbandingan dan Metode Terbaik")
    
    differences = {
        "Polynomial Regression": abs(poly_pred_idr - combined_pred_idr),
        "Random Forest": abs(rf_pred_idr - combined_pred_idr),
        "Gradient Boosting": abs(gb_pred_idr - combined_pred_idr)
    }
    
    best_method = min(differences, key=differences.get)
    
    st.write(f"Metode terbaik: **{best_method}**")
    
    st.write(f"Selisih antara Gabungan dan Polynomial Regression: Rp{differences['Polynomial Regression']:,.2f}")
    st.write(f"Selisih antara Gabungan dan Random Forest: Rp{differences['Random Forest']:,.2f}")
    st.write(f"Selisih antara Gabungan dan Gradient Boosting: Rp{differences['Gradient Boosting']:,.2f}")

