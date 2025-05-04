import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("career_path_recommender_model.pkl")

# Load the scaler for numerical features
scaler = StandardScaler()

# Streamlit title and description
st.title("AI-Powered Career Path Recommender")
st.write("Fill in your profile details to get the best career path recommendation.")

# Input fields
interest_ai = st.selectbox("Are you interested in AI?", ["Yes", "No"]) == "Yes"
interest_webdev = st.selectbox("Are you interested in Web Development?", ["Yes", "No"]) == "Yes"
interest_cybersec = st.selectbox("Are you interested in Cybersecurity?", ["Yes", "No"]) == "Yes"
interest_uiux = st.selectbox("Are you interested in UI/UX Design?", ["Yes", "No"]) == "Yes"

skill_python = st.slider("How skilled are you in Python?", 1, 3, 2)  # 1 = Beginner, 3 = Advanced
skill_html_css = st.slider("How skilled are you in HTML/CSS?", 1, 3, 2)
skill_sql = st.slider("How skilled are you in SQL?", 1, 3, 2)
skill_cloud = st.slider("How skilled are you in Cloud Technologies?", 1, 3, 2)

work_style = st.selectbox("What is your preferred work style?", ["Team", "Solo", "Remote"])
learning_pref = st.selectbox("What is your preferred learning style?", ["Visual", "Reading", "Hands-on"])
strength = st.selectbox("What is your main strength?", ["Logic", "Creative", "Communication"])

# Encode categorical features
work_style_encoded = {"Team": 0, "Solo": 1, "Remote": 2}[work_style]
learning_pref_encoded = {"Visual": 0, "Reading": 1, "Hands-on": 2}[learning_pref]
strength_encoded = {"Logic": 0, "Creative": 1, "Communication": 2}[strength]

# Prepare the input data
input_data = np.array([
    interest_ai, interest_webdev, interest_cybersec, interest_uiux,
    skill_python, skill_html_css, skill_sql, skill_cloud,
    work_style_encoded, learning_pref_encoded, strength_encoded
]).reshape(1, -1)

# Scale numerical features
input_data[0, 4:8] = scaler.fit_transform(input_data[0, 4:8].reshape(1, -1))[0]

# Make prediction
prediction = model.predict(input_data)

# Career path labels (make sure these correspond to the order in your model)
career_paths = [
    "Machine Learning Engineer", "Web Developer", "Cybersecurity Analyst",
    "UI/UX Designer", "Data Analyst", "Cloud Engineer", "DevOps Engineer", "Business Analyst"
]

# Display the result
st.write(f"Recommended Career Path: {career_paths[prediction[0]]}")

