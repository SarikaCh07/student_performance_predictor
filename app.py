# app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will pass or fail based on some key inputs.")

# Input sliders
study_hours = st.slider("📘 Study Hours per Day", 0, 12, 6)
attendance = st.slider("📅 Attendance (%)", 0, 100, 75)
past_grade = st.slider("📊 Previous Grade (%)", 0, 100, 65)
participation = st.slider("🙋 Participation Level (1-5)", 1, 5, 3)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[study_hours, attendance, past_grade, participation]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ The student is likely to PASS!")
    else:
        st.error("❌ The student is likely to FAIL.")

