import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved Random Forest model
with open('best_bayesian_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define all expected features based on the trained model
# Ensure this list matches the features the model was trained on
expected_features = [
    'schoolsup_yes', 'Medu', 'Fedu', 'Mjob_health', 'sex_M',
    'reason_reputation', 'famsize_LE3', 'internet_yes', 'studytime',
    'Mjob_services', 'Fjob_other', 'reason_home', 'famsup_yes',
    'Mjob_other', 'Dalc', 'guardian_Mother', 'guardian_Father',
    'higher_yes', 'address_U', 'Fjob_teacher', 'Walc', 'traveltime',
    'absences', 'goout', 'age', 'failures'
]

# Define mappings
binary_mappings = {
    "sex": {"Female": 0, "Male": 1},
    "schoolsup": {"No": 0, "Yes": 1},
    "famsize": {"LE3 (≤ 3 members)": 1, "GT3 (> 3 members)": 0},
    "internet": {"No": 0, "Yes": 1},
    "famsup": {"No": 0, "Yes": 1},
    "higher": {"No": 0, "Yes": 1},
    "address": {"Rural": 0, "Urban": 1},
}

dummy_mappings = {
    "guardian": {"Mother": [1, 0], "Father": [0, 1], "Other": [0, 0]},
    "reason": {
        "Home": [1, 0, 0, 0],
        "Reputation": [0, 1, 0, 0],
        "Course": [0, 0, 1, 0],
        "Other": [0, 0, 0, 1],
    },
    "Mjob": {
        "Teacher": [1, 0, 0, 0, 0],
        "Health": [0, 1, 0, 0, 0],
        "Services": [0, 0, 1, 0, 0],
        "At Home": [0, 0, 0, 1, 0],
        "Other": [0, 0, 0, 0, 1],
    },
    "Fjob": {"Teacher": [1, 0], "Other": [0, 1]},
}

# Streamlit UI
st.title("Student Final Grade Predictor")
st.write("Input student details below to predict their final grade (G3).")

# Collect user inputs
age = st.number_input("Age (15-22 years)", min_value=15, max_value=22, value=18)
Medu = int(st.selectbox("Mother's Education", ["0: None", "1: Primary", "2: 5th-9th grade", "3: Secondary", "4: Higher"]).split(":")[0])
Fedu = int(st.selectbox("Father's Education", ["0: None", "1: Primary", "2: 5th-9th grade", "3: Secondary", "4: Higher"]).split(":")[0])
studytime = int(st.selectbox("Weekly Study Time", ["1: <2 hours", "2: 2-5 hours", "3: 5-10 hours", "4: >10 hours"]).split(":")[0])
traveltime = int(st.selectbox("Travel Time to School", ["1: <15 min", "2: 15-30 min", "3: 30 min-1 hour", "4: >1 hour"]).split(":")[0])
failures = st.number_input("Number of Past Failures (0-4)", min_value=0, max_value=4, value=0)
goout = st.slider("Frequency of Going Out (1-5)", min_value=1, max_value=5, value=3)
Walc = st.slider("Weekend Alcohol Consumption (1-5)", min_value=1, max_value=5, value=2)
Dalc = st.slider("Workday Alcohol Consumption (1-5)", min_value=1, max_value=5, value=1)
absences = st.number_input("Number of School Absences (0-93)", min_value=0, max_value=93, value=0)
sex = st.selectbox("Sex", ["Female", "Male"])
schoolsup = st.selectbox("Extra Educational Support", ["No", "Yes"])
famsize = st.selectbox("Family Size", ["LE3 (≤ 3 members)", "GT3 (> 3 members)"])
internet = st.selectbox("Internet Access at Home", ["No", "Yes"])
famsup = st.selectbox("Family Educational Support", ["No", "Yes"])
higher = st.selectbox("Wants Higher Education", ["No", "Yes"])
address = st.selectbox("Home Address", ["Rural", "Urban"])
guardian = st.selectbox("Guardian", ["Mother", "Father", "Other"])
reason = st.selectbox("Reason for School Choice", ["Home", "Reputation", "Course", "Other"])
Mjob = st.selectbox("Mother's Job", ["Teacher", "Health", "Services", "At Home", "Other"])
Fjob = st.selectbox("Father's Job", ["Teacher", "Other"])

# Process inputs
inputs = {
    "age": age,
    "Medu": Medu,
    "Fedu": Fedu,
    "studytime": studytime,
    "traveltime": traveltime,
    "failures": failures,
    "goout": goout,
    "Walc": Walc,
    "Dalc": Dalc,
    "absences": absences,
    "sex_M": binary_mappings["sex"][sex],
    "schoolsup_yes": binary_mappings["schoolsup"][schoolsup],
    "famsize_LE3": binary_mappings["famsize"][famsize],
    "internet_yes": binary_mappings["internet"][internet],
    "famsup_yes": binary_mappings["famsup"][famsup],
    "higher_yes": binary_mappings["higher"][higher],
    "address_U": binary_mappings["address"][address],
    **dict(zip(["guardian_Mother", "guardian_Father"], dummy_mappings["guardian"][guardian])),
    **dict(zip(["reason_home", "reason_reputation", "reason_course", "reason_other"], dummy_mappings["reason"][reason])),
    **dict(zip(["Mjob_teacher", "Mjob_health", "Mjob_services", "Mjob_at_home", "Mjob_other"], dummy_mappings["Mjob"][Mjob])),
    **dict(zip(["Fjob_teacher", "Fjob_other"], dummy_mappings["Fjob"][Fjob])),
}

# Align inputs with expected features
input_df = pd.DataFrame([inputs]).reindex(columns=expected_features, fill_value=0)
input_array = input_df.to_numpy()

# Ensure the input matches the model's feature count
input_array = input_array[:, :model.n_features_in_]

# Predict grade
try:
    predicted_grade = model.predict(input_array)[0]
    st.subheader("Prediction Results")
    st.write(f"Predicted Grade: {predicted_grade:.2f}")
    st.write(f"Status: {'Pass' if predicted_grade >= 12 else 'Fail'}")
except Exception as e:
    st.error(f"Error: {e}")
