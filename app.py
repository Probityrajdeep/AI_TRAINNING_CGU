import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD & TRAIN MODEL (LOCAL)
# =========================

@st.cache_resource
def train_model():
    df = pd.read_csv("E:\\ai_project\\placementdata.csv")

    df.drop("StudentID", axis=1, inplace=True)

    df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"Yes": 1, "No": 0})
    df["PlacementTraining"] = df["PlacementTraining"].map({"Yes": 1, "No": 0})
    df["PlacementStatus"] = df["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})

    X = df.drop("PlacementStatus", axis=1)
    y = df["PlacementStatus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model, scaler


model, scaler = train_model()

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Placement Prediction", layout="centered")

st.title("🎓 Student Placement Prediction System")
st.markdown("---")

with st.form("placement_form"):

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
        ssc = st.number_input("SSC Marks", 0, 100, 70)
        hsc = st.number_input("HSC Marks", 0, 100, 75)
        internships = st.slider("Internships", 0, 5, 1)
        projects = st.slider("Projects", 0, 5, 2)

    with col2:
        workshops = st.slider("Workshops/Certifications", 0, 5, 1)
        aptitude = st.number_input("Aptitude Score", 0, 100, 70)
        soft = st.number_input("Soft Skills Rating", 0.0, 5.0, 3.5)
        extra = st.radio("Extracurricular Activities", ["Yes", "No"])
        training = st.radio("Placement Training", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

    if submit:

        if cgpa < 6.0:
            st.error("❌ NOT PLACED (CGPA below cutoff)")
        else:
            extra_val = 1 if extra == "Yes" else 0
            training_val = 1 if training == "Yes" else 0

            input_data = [[
                cgpa,
                internships,
                projects,
                workshops,
                aptitude,
                soft,
                extra_val,
                training_val,
                ssc,
                hsc
            ]]

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            if prediction[0] == 1:
                st.success("🎉 Student is likely to be PLACED")
            else:
                st.error("❌ Student is likely to be NOT PLACED")

st.markdown("---")

st.subheader("📊 Feature Importance")

feature_names = [
    'CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
    'AptitudeTestScore', 'SoftSkillsRating',
    'ExtracurricularActivities', 'PlacementTraining',
    'SSC_Marks', 'HSC_Marks'
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)