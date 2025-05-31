import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt

# === LOAD MODEL ===
rf_clf = joblib.load("rf_classifier.pkl")
lr = joblib.load("linear_regression.pkl")
le = joblib.load("label_encoder.pkl")

# === PAGE SETUP ===
st.set_page_config(page_title="Simulasi Gempa Indonesia", layout="centered")
st.title("Simulasi Prediksi Gempa Bumi Indonesia")
st.subheader("Oleh : Addien Munadiya Yunadi (2201848)")
st.markdown("Masukkan parameter sumber gempa untuk memprediksi **kategori** dan **estimasi magnitudo** gempa bumi.")

# === INPUT: LOKASI ===
with st.expander("📍 Masukkan Lokasi"):
    lokasi_opsi = st.radio("Metode Input Lokasi:", ["Ketik Nama Daerah", "Manual Koordinat (Lat/Lon)"])

    if lokasi_opsi == "Ketik Nama Daerah":
        nama_daerah = st.text_input("Nama Daerah", value="Jakarta")
        lat, lon = None, None
        if nama_daerah:
            try:
                geolocator = Nominatim(user_agent="gempa-predictor")
                location = geolocator.geocode(nama_daerah)
                if location:
                    lat = location.latitude
                    lon = location.longitude
                    st.success(f"Koordinat ditemukan: ({lat:.4f}, {lon:.4f})")
                else:
                    st.warning("Lokasi tidak ditemukan.")
            except Exception as e:
                st.error(f"Gagal mendapatkan koordinat: {e}")
    else:
        lat = st.number_input("Latitude", format="%.4f", value=-6.2)
        lon = st.number_input("Longitude", format="%.4f", value=106.8)

# === PENJELASAN PARAMETER ===
with st.expander("❓ Apa itu Parameter Mekanisme Gempa?"):
    st.markdown("""
    - **Depth (Kedalaman)**: Seberapa dalam gempa terjadi dari permukaan (0–700 km).
    - **Strike**: Sudut orientasi garis patahan terhadap utara (0–360°).
    - **Dip**: Sudut kemiringan bidang patahan ke bawah (0–90°).
    - **Rake**: Arah gerakan relatif sepanjang bidang patahan (-180° hingga 180°).
    """)

# === INPUT PARAMETER ===
st.markdown("### ⚙️ Parameter Mekanisme Sumber Gempa")
depth = st.slider("Kedalaman (km)", 0, 700, 50)
strike = st.slider("Strike (°)", 0, 360, 180)
dip = st.slider("Dip (°)", 0, 90, 30)
rake = st.slider("Rake (°)", -180, 180, 90)

# === TOMBOL PREDIKSI ===
if st.button("🔮 Prediksi Gempa"):
    if lat is not None and lon is not None:
        input_df = pd.DataFrame({
            'lat': [lat],
            'lon': [lon],
            'depth': [depth],
            'strike': [strike],
            'dip': [dip],
            'rake': [rake]
        })

        input_rf = input_df[['depth', 'strike', 'dip', 'rake']]

        # Prediksi kategori dan magnitudo
        pred_kategori_enc = rf_clf.predict(input_rf)[0]
        pred_kategori = le.inverse_transform([pred_kategori_enc])[0]
        pred_magnitudo = lr.predict(input_rf)[0]

        # ✅ INI HARUS ADA DI DALAM BLOK if, DENGAN INDENTASI YANG BENAR (4 spasi)
        st.session_state.pred = {
            "kategori": pred_kategori,
            "magnitudo": float(pred_magnitudo),
            "lat": lat,
            "lon": lon
        }

    else:
        st.error("Koordinat tidak tersedia. Masukkan lokasi dengan benar.")

# === OUTPUT & VISUALISASI ===
if "pred" in st.session_state:
    pred = st.session_state.pred
    st.markdown("### Hasil Prediksi")
    st.write(f"**Kategori Gempa:** `{pred['kategori']}`")
    st.write(f"**Estimasi Magnitudo:** `{pred['magnitudo']:.2f}`")

    # Chart kecil magnitudo
    st.markdown("### 📊 Skala Magnitudo ")
    fig, ax = plt.subplots()
    ax.barh(["Magnitudo"], [pred['magnitudo']], color="salmon")
    ax.set_xlim([0, 10])
    ax.set_xlabel("Magnitudo")
    st.pyplot(fig)

    # Penjelasan kategori & mitigasi
    with st.expander("🧾 Penjelasan Kategori dan Mitigasi"):
        if pred['kategori'] == "Light":
            st.markdown("""
            🟡 **Light** (Magnitudo < 4.0)  
            - Dampak: Umumnya tidak merusak, hanya getaran ringan.
            - Mitigasi:
                - Tetap tenang, hindari panik.
                - Evaluasi struktur bangunan jika sering terjadi.
                - Edukasi masyarakat tentang evakuasi ringan.
            """)
        elif pred['kategori'] == "Moderate":
            st.markdown("""
            🟠 **Moderate** (4.0 ≤ Magnitudo < 6.0)  
            - Dampak: Dapat menyebabkan kerusakan ringan pada bangunan tua.
            - Mitigasi:
                - Periksa struktur bangunan, terutama bangunan lama.
                - Siapkan jalur evakuasi dan tas darurat.
                - Simulasi evakuasi rutin di sekolah/kantor.
            """)
        elif pred['kategori'] == "Strong":
            st.markdown("""
            🔴 **Strong** (Magnitudo ≥ 6.0)  
            - Dampak: Berpotensi menyebabkan kerusakan serius dan korban jiwa.
            - Mitigasi:
                - Waspadai potensi gempa susulan.
                - Perkuat bangunan sesuai standar tahan gempa.
                - Siapkan shelter, logistik, dan koordinasi dengan BNPB/BPBD.
            """)

    # Peta Lokasi Gempa (OpenStreetMap)
    st.markdown("### 🗺️ Lokasi Prediksi Gempa")
    m = folium.Map(location=[pred['lat'], pred['lon']], zoom_start=7, tiles='OpenStreetMap')
    folium.CircleMarker(
        location=[pred['lat'], pred['lon']],
        radius=10,
        popup=folium.Popup(
            f"<b>Gempa</b><br>Kategori: {pred['kategori']}<br>Magnitudo: {pred['magnitudo']:.2f}",
            max_width=200
        ),
        color='red' if pred['kategori'] == 'Strong' else 'orange' if pred['kategori'] == 'Moderate' else 'yellow',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)
    st_folium(m, width=700, height=450)
