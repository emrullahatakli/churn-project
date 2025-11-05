import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler

#Sayfa yapılandırması
st.set_page_config(layout="wide")
st.title("Canlı Churn Risk Analiz Dashboard")

#Veri, Model ve Scaler Yükleme Fonksiyonları
@st.cache_data
def load_data(csv_path):
    try:
        df= pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error(f"HATA: '{csv_path}' dosyası bulunamadı.")
        st.stop()

@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"HATA: '{model_path}' model dosyası bulunamadı.")
        st.error("Modeli eğitirken kullandığınız 'scaler.joblib dosyasını bu klasöre eklemelisiniz.")
        st.stop()
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        st.stop()

@st.cache_resource
def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except FileNotFoundError:
        st.error(f"HATA '{scaler_path}' scaler dosyası bulunamadı.")
        st.stop()
    except Exception as e:
        st.error(f"Scaler yüklenirken bir hata oluştu: {e}")
        st.stop()

#Veri ön işleme
COLS_TO_SCALE = ['tenure', 'MonthlyCharges', 'TotalCharges']

def preprocess_data(df_input, scaler, model_features_list):
    df_processed = df_input.copy()

    if "TotalCharges" in df_processed.columns:
        df_processed["TotalCharges"] = pd.to_numeric(df_processed["TotalCharges"], errors="coerce")
        df_processed["TotalCharges"] = df_processed["TotalCharges"].fillna(0)
    
    if "SeniorCitizen" in df_processed.columns:
        df_processed["SeniorCitizen"] = df_processed["SeniorCitizen"].map({0: "No", 1: "Yes"})

    if "customerID" in df_processed.columns:
        df_processed.drop("customerID", axis=1, inplace=True)

    try:
        df_processed["NEW_HighRisk_Customer"] = (
            (df_processed["tenure"] < 12)
            & (df_processed["Contract"] == "Month-to-month")
            & (df_processed["Partner"] == "No")).astype(int)
    except Exception as e:
        st.warning(f"Sorun oluştu.{e}")


    cols_to_scale_existing = [col for col in COLS_TO_SCALE if col in df_processed.columns]
    if cols_to_scale_existing:
        df_processed[cols_to_scale_existing] = scaler.transform(df_processed[cols_to_scale_existing])

    if "Churn" in df_processed.columns:
        df_processed = df_processed.drop("Churn", axis = 1)

    df_processed = pd.get_dummies(df_processed, dtype=int)

    df_processed = df_processed.reindex(columns = model_features_list, fill_value=0)

    return df_processed

#Ana Uygulama Akışı
CSV_FILE = "telco_churn.csv"
MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.joblib"

MODEL_FEATURES_LIST = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "NEW_HighRisk_Customer",
    "gender_Male",
    "SeniorCitizen_Yes",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]

df_raw = load_data(CSV_FILE)
model = load_model(MODEL_FILE)
scaler = load_scaler(SCALER_FILE)

df_raw_features = df_raw.drop("Churn", axis = 1, errors = 'ignore')

st.info("Toplu veri işleniyor ve tahminler yapılıyor...")

df_processed_all = preprocess_data(df_raw_features, scaler, MODEL_FEATURES_LIST)
probabilities_all = model.predict_proba(df_processed_all)
df_raw["Churn_Probability"] = probabilities_all[:,1]
st.success("Tüm müşteriler için tahminler oluşturuldu.")

#Dashboard Sekmeleri

tab1, tab2, tab3, tab4 = st.tabs([
    "Genel Dağılım",
    "Segment Analizi",
    "Aksiyon Listesi",
    "Bireysel Risk Simülatörü"
])

with tab1:
    st.header("Genel Churn Olasılık Dağılımı")
    fig_hist = px.histogram(
        df_raw,
        x="Churn_Probability",
        nbins=50,
        title="Müşterilerin Churn Olasılıklarının Dağılımı",
    )
    st.plotly_chart(fig_hist, width="stretch", key="hist_chart")

with tab2:
    st.header("Segmentlere Göre Ortalama Risk")

    segment_options = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "Contract",
        "PaymentMethod",
    ]

    valid_options = [opt for opt in segment_options if opt in df_raw.columns]

    if valid_options:
        selected_segment = st.selectbox("Analiz Edilecek Kırılımı Seçin:", valid_options)

        analysis_df = df_raw.groupby(selected_segment, as_index=False)[ "Churn_Probability"].mean()
        chart_title = f"{selected_segment} Bazlı Ortalama Churn Riski"

        fig_bar = px.bar(
            analysis_df,
            x=selected_segment,
            y="Churn_Probability",
            title=chart_title,
            color=selected_segment,
        )
        fig_bar.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_bar, width="stretch", key="bar_chart")

    else:
        st.warning("Analiz için 'segment_options' listesindeki sütunlar veride bulunamadı.")

with tab3:
    st.header("Aksiyon Gereken Müşteri Listesi")

    risk_threshold = st.slider("Risk Eşiği Belirleyin", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key= "tab3_slider")

    high_risk_customers = df_raw[df_raw["Churn_Probability"] >= risk_threshold].sort_values(by="Churn_Probability", ascending=False)

    st.subheader(f"Olasılığı {risk_threshold} Üzerinde Olan {len(high_risk_customers)} Müşteri Bulundu")

    columns_to_show = ['Partner', 'InternetService', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Probability']

    valid_cols_to_show = [col for col in columns_to_show if col in df_raw.columns]
    st.dataframe(high_risk_customers[valid_cols_to_show], width="stretch")

with tab4:
    st.header("Bireysel Müşteri Risk Simülatörü")
    st.info("SHAP analizine göre en önemli özelliklere seçerek bir müşterinin churn riskini hesaplayın.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Müşteri Özellikleri")

        tenure_input = st.slider("" \
        "Tenure (Müşteri Kıdemi - Ay)",
        min_value=0, max_value=72, value=1, step = 1)

        monthly_input = st.number_input(
            "Monthly Charges (Aylık Ücret)",
            min_value = 0.0, value=70.0, step = 0.1
        )

        senior_input = st.radio("Senior Citizen (Yaşlı Müşteri)", [0,1], format_func= lambda x: "Evet" if x == 1 else "Hayır")
        partner_input = st.radio("Partner (Evli/Partneri var mı?)", ["Yes", "No"])

        high_risk_input = st.checkbox("NEW_high_risk customer", value=True)

        with col2:
            st.subheader("Hizmet Bilgileri")
        
            internet_input = st.selectbox("Internet Service Tipi:", ["No", "DSL", "Fiber optic"])
            tech_support_input = st.selectbox("Tech Support:", ["No", "Yes", "No internet service"])
            online_backup_input = st.selectbox("Online Backup:", ["No", "Yes", "No internet service"])
            online_security_input = st.selectbox("Online Security:", ["No", "Yes", "No internet service"])
            streaming_tv_input = st.selectbox("Streaming TV:", ["No", "Yes", "No internet service"])

    if st.button('Riski Hesapla', type='primary'):
        customer_data = {
            "tenure": tenure_input,
            "MonthlyCharges": monthly_input,
            "SeniorCitizen": senior_input,
            "Partner": partner_input,
            "InternetService": internet_input,
            "TechSupport": tech_support_input,
            "OnlineBackup": online_backup_input,
            "OnlineSecurity": online_security_input,
            "StreamingTV": streaming_tv_input,
            #"NEW_high_risk customer": 1 if high_risk_input else 0,
            "gender": "Male",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "DeviceProtection": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "TotalCharges": 1394.55,
        }

        df_single = pd.DataFrame([customer_data])

        st.info("Bireysel müşteri verisi işleniyor...")
        df_single_processed = preprocess_data(df_single, scaler, MODEL_FEATURES_LIST)

        st.info("Tahmin Yapılıyor...")
        prediction_proba = model.predict_proba(df_single_processed)
        churn_risk = prediction_proba[0][1]

        st.success("Tahmin Tamamlandı.")

        delta_color = "inverse" if churn_risk < 0.5 else "normal"
        st.metric(
            label="Bu Müşterinin Churn Riski",
            value=f"{churn_risk:.1%}",
            delta=f"Risk: {'Düşük' if churn_risk < 0.5 else 'Yüksek'}",
            delta_color=delta_color,
        )
    