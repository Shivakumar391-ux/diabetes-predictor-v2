
import streamlit as st
st.set_page_config(page_title="Diabetes Readmission Predictor", page_icon="🏥", layout="wide")
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

@st.cache_data
def load_and_train():
    dataset = fetch_ucirepo(id=296)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)

    df.drop(columns=['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code'], inplace=True)
    df['race'] = df['race'].fillna(df['race'].mode()[0])
    df['diag_1'] = df['diag_1'].fillna('Unknown')
    df['diag_2'] = df['diag_2'].fillna('Unknown')
    df['diag_3'] = df['diag_3'].fillna('Unknown')
    df['readmit_risk'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df.drop(columns=['readmitted', 'diag_1', 'diag_2', 'diag_3'], inplace=True)

    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Classification
    X_cls = df.drop(columns=['readmit_risk'])
    y_cls = df['readmit_risk']
    scaler_cls = StandardScaler()
    X_cls_scaled = scaler_cls.fit_transform(X_cls)
    X_train, X_test, y_train, y_test = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    scale = 72326 / 9086
    xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    # Regression
    X_reg = df.drop(columns=['readmit_risk', 'time_in_hospital'])
    y_reg = df['time_in_hospital']
    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg)
    X_train_r, _, y_train_r, _ = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_r, y_train_r)

    # Clustering
    cluster_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                        'num_medications', 'number_diagnoses', 'number_inpatient']
    scaler_cl = StandardScaler()
    X_cl = scaler_cl.fit_transform(df[cluster_features])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_cl)

    return xgb, scaler_cls, X_cls.columns.tolist(), rf_reg, scaler_reg, X_reg.columns.tolist(), km, scaler_cl, cluster_features

xgb, scaler_cls, cls_cols, rf_reg, scaler_reg, reg_cols, km, scaler_cl, cluster_features = load_and_train()

st.title("🏥 Diabetes Patient Readmission Risk Predictor")
st.markdown("---")

page = st.sidebar.selectbox("Select Module", ["Readmission Risk Predictor", "Hospital Stay Predictor", "Patient Cluster Finder"])

if page == "Readmission Risk Predictor":
    st.header("🚨 Readmission Risk Predictor")
    st.write("Predict if a diabetic patient will be readmitted within 30 days.")

    col1, col2 = st.columns(2)
    with col1:
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
        num_medications = st.slider("Number of Medications", 1, 81, 15)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
    with col2:
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        number_inpatient = st.slider("Prior Inpatient Visits", 0, 21, 0)
        number_emergency = st.slider("Prior Emergency Visits", 0, 76, 0)
        number_outpatient = st.slider("Prior Outpatient Visits", 0, 42, 0)

    if st.button("Predict Readmission Risk"):
        input_data = pd.DataFrame(np.zeros((1, len(cls_cols))), columns=cls_cols)
        input_data['time_in_hospital'] = time_in_hospital
        input_data['num_lab_procedures'] = num_lab_procedures
        input_data['num_medications'] = num_medications
        input_data['number_diagnoses'] = number_diagnoses
        input_data['num_procedures'] = num_procedures
        input_data['number_inpatient'] = number_inpatient
        input_data['number_emergency'] = number_emergency
        input_data['number_outpatient'] = number_outpatient
        scaled = scaler_cls.transform(input_data)
        pred = xgb.predict(scaled)[0]
        prob = xgb.predict_proba(scaled)[0][1]
        if pred == 1:
            st.error(f"⚠️ HIGH READMISSION RISK — Probability: {prob:.1%}")
        else:
            st.success(f"✅ LOW READMISSION RISK — Probability: {prob:.1%}")

elif page == "Hospital Stay Predictor":
    st.header("🛏️ Hospital Stay Predictor")
    st.write("Predict expected number of days a patient will stay.")

    col1, col2 = st.columns(2)
    with col1:
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
        num_medications = st.slider("Number of Medications", 1, 81, 15)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
    with col2:
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        number_inpatient = st.slider("Prior Inpatient Visits", 0, 21, 0)
        number_emergency = st.slider("Prior Emergency Visits", 0, 76, 0)

    if st.button("Predict Hospital Stay"):
        input_data = pd.DataFrame(np.zeros((1, len(reg_cols))), columns=reg_cols)
        input_data['num_lab_procedures'] = num_lab_procedures
        input_data['num_medications'] = num_medications
        input_data['number_diagnoses'] = number_diagnoses
        input_data['num_procedures'] = num_procedures
        input_data['number_inpatient'] = number_inpatient
        input_data['number_emergency'] = number_emergency
        scaled = scaler_reg.transform(input_data)
        days = rf_reg.predict(scaled)[0]
        st.info(f"🛏️ Predicted Hospital Stay: **{days:.1f} days**")

elif page == "Patient Cluster Finder":
    st.header("👥 Patient Cluster Finder")
    profiles = {
        0: "💚 Mild Case — Short stay, few medications, low complexity",
        1: "🔴 Complex Case — Long stay, many medications, multiple diagnoses",
        2: "🟡 Moderate Case — Average stay, moderate procedures",
        3: "⚠️ High Risk — Frequent hospital visitor, high readmission risk"
    }

    col1, col2 = st.columns(2)
    with col1:
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    with col2:
        num_medications = st.slider("Number of Medications", 1, 81, 15)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
        number_inpatient = st.slider("Prior Inpatient Visits", 0, 21, 0)

    if st.button("Find Patient Cluster"):
        input_cl = np.array([[time_in_hospital, num_lab_procedures, num_procedures,
                              num_medications, number_diagnoses, number_inpatient]])
        scaled_cl = scaler_cl.transform(input_cl)
        cluster = km.predict(scaled_cl)[0]
        st.success(f"Cluster {cluster}: {profiles[cluster]}")
