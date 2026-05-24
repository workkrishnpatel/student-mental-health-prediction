import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")

st.title("Mental Health Risk Prediction")

st.write("Student Depression Prediction System")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    15,
    40,
    20
)

academic_pressure = st.slider(
    "Academic Pressure",
    0,
    5,
    2
)

cgpa = st.slider(
    "CGPA",
    0.0,
    10.0,
    7.0
)

study_satisfaction = st.slider(
    "Study Satisfaction",
    0,
    5,
    3
)

sleep_duration = st.selectbox(
    "Sleep Duration",
    [
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours"
    ]
)

dietary_habits = st.selectbox(
    "Dietary Habits",
    [
        "Healthy",
        "Moderate",
        "Unhealthy"
    ]
)

suicidal_thoughts = st.selectbox(
    "Suicidal Thoughts",
    [
        "Yes",
        "No"
    ]
)

work_study_hours = st.slider(
    "Work Study Hours",
    0,
    16,
    6
)

financial_stress = st.slider(
    "Financial Stress",
    0,
    5,
    2
)

family_history = st.selectbox(
    "Family History of Mental Illness",
    [
        "Yes",
        "No"
    ]
)

if st.button("Predict"):

    gender = 1 if gender == "Male" else 0

    suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0

    family_history = 1 if family_history == "Yes" else 0

    sleep_map = {
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3
    }

    diet_map = {
        "Healthy": 0,
        "Moderate": 1,
        "Unhealthy": 2
    }

    sleep_duration = sleep_map[sleep_duration]

    dietary_habits = diet_map[dietary_habits]

    input_data = pd.DataFrame({

        "gender": [gender],

        "age": [age],

        "academic_pressure": [academic_pressure],

        "cgpa": [cgpa],

        "study_satisfaction": [study_satisfaction],

        "sleep_duration": [sleep_duration],

        "dietary_habits": [dietary_habits],

        "suicidal_thoughts": [suicidal_thoughts],

        "work_study_hours": [work_study_hours],

        "financial_stress": [financial_stress],

        "family_history": [family_history]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:

        st.error(
            f"High Depression Risk Detected\n\nConfidence: {probability:.2%}"
        )

    else:

        st.success(
            f"Low Depression Risk\n\nConfidence: {(1 - probability):.2%}")