#!/usr/bin/env python
# coding: utf-8

# CareerWise: AI-Powered Career Path Recommender
# 
# ##Problem Statement
# Many students struggle to choose the right tech career path (e.g., Web Dev, ML, Cybersecurity). This tool helps recommend an ideal career based on user inputs like interests, skills, and preferences.
# 
# ##Objective
# Build a machine learning model that recommends a suitable tech career path based on user profile data.
# 
# ##Tools & Libraries
# - Python
# - Jupyter Notebook
# - Pandas, NumPy, Scikit-learn
# 

# In[1]:


# Target labels for classification
career_paths = [
    "Machine Learning Engineer",
    "Web Developer",
    "Cybersecurity Analyst",
    "UI/UX Designer",
    "Data Analyst",
    "Cloud Engineer",
    "DevOps Engineer",
    "Business Analyst"
]

print("Target Career Paths:")
for i, path in enumerate(career_paths, start=1):
    print(f"{i}. {path}")


# In[2]:


# Input features used in the model
features = [
    "Interest_AI",
    "Interest_WebDev",
    "Interest_Cybersecurity",
    "Interest_UIUX",
    "Skill_Python",
    "Skill_HTML_CSS",
    "Skill_SQL",
    "Skill_Cloud",
    "Work_Style",
    "Learning_Preference",
    "Strength"
]

print("Selected Input Features:")
for f in features:
    print("-", f)


# ###Feature Descriptions
# 
# | Feature | Description | Type |
# |--------|-------------|------|
# | Interest_AI | Interest in AI/ML | Binary (0 = No, 1 = Yes) |
# | Interest_WebDev | Interest in Web Development | Binary (0 or 1) |
# | Interest_Cybersecurity | Interest in Cybersecurity | Binary (0 or 1) |
# | Interest_UIUX | Interest in UI/UX Design | Binary (0 or 1) |
# | Skill_Python | Python skill level (1 = Beginner, 2 = Intermediate, 3 = Advanced) | Ordinal |
# | Skill_HTML_CSS | Frontend (HTML/CSS) skill level (1–3) | Ordinal |
# | Skill_SQL | SQL/DB skill level (1–3) | Ordinal |
# | Skill_Cloud | Cloud computing skill level (1–3) | Ordinal |
# | Work_Style | Preferred work style: 'Team', 'Solo', 'Remote' | Categorical |
# | Learning_Preference | How user prefers to learn: 'Visual', 'Reading', 'Hands-on' | Categorical |
# | Strength | User's strength: 'Logic', 'Creative', 'Communication' | Categorical |
# 

# In[3]:


import pandas as pd
import numpy as np
import random


# In[4]:


def random_interest():
    return random.randint(0, 1)

def random_skill():
    return random.randint(1, 3)  # 1 = Beginner, 2 = Intermediate, 3 = Advanced

def random_work_style():
    return random.choice(["Team", "Solo", "Remote"])

def random_learning_preference():
    return random.choice(["Visual", "Reading", "Hands-on"])

def random_strength():
    return random.choice(["Logic", "Creative", "Communication"])


# In[5]:


def assign_career(profile):
    if profile['Interest_AI'] and profile['Skill_Python'] >= 2:
        return "Machine Learning Engineer"
    elif profile['Interest_WebDev'] and profile['Skill_HTML_CSS'] >= 2:
        return "Web Developer"
    elif profile['Interest_Cybersecurity'] and profile['Skill_Cloud'] >= 2:
        return "Cybersecurity Analyst"
    elif profile['Interest_UIUX'] and profile['Strength'] == "Creative":
        return "UI/UX Designer"
    elif profile['Skill_SQL'] >= 2 and profile['Strength'] == "Logic":
        return "Data Analyst"
    elif profile['Skill_Cloud'] == 3 and profile['Work_Style'] == "Remote":
        return "Cloud Engineer"
    elif profile['Skill_Cloud'] >= 2 and profile['Skill_Python'] >= 2:
        return "DevOps Engineer"
    elif profile['Strength'] == "Communication":
        return "Business Analyst"
    else:
        return random.choice([
            "Machine Learning Engineer", "Web Developer", "Cybersecurity Analyst",
            "UI/UX Designer", "Data Analyst", "Cloud Engineer", "DevOps Engineer", "Business Analyst"
        ])


# In[6]:


data = []

for _ in range(300):
    profile = {
        "Interest_AI": random_interest(),
        "Interest_WebDev": random_interest(),
        "Interest_Cybersecurity": random_interest(),
        "Interest_UIUX": random_interest(),
        "Skill_Python": random_skill(),
        "Skill_HTML_CSS": random_skill(),
        "Skill_SQL": random_skill(),
        "Skill_Cloud": random_skill(),
        "Work_Style": random_work_style(),
        "Learning_Preference": random_learning_preference(),
        "Strength": random_strength(),
    }
    profile["Career_Path"] = assign_career(profile)
    data.append(profile)

df = pd.DataFrame(data)
df.head()


# In[7]:


df.to_csv("synthetic_career_data.csv", index=False)


# In[8]:


# Load the synthetic dataset
df = pd.read_csv("synthetic_career_data.csv")

# Show the first few rows of the dataset
df.head()


# In[9]:


# Check for missing values
df.isnull().sum()

# Drop rows with missing values (if any)
df = df.dropna()

# Alternatively, you could fill missing values if needed
# df['column_name'] = df['column_name'].fillna(df['column_name'].mean())


# In[10]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Encode categorical features
df['Work_Style'] = le.fit_transform(df['Work_Style'])
df['Learning_Preference'] = le.fit_transform(df['Learning_Preference'])
df['Strength'] = le.fit_transform(df['Strength'])

# Check the encoded values
df[['Work_Style', 'Learning_Preference', 'Strength']].head()


# In[11]:


# Encode the Career Path (target variable)
df['Career_Path'] = le.fit_transform(df['Career_Path'])

# Check the encoded target variable
df['Career_Path'].value_counts()


# In[12]:


# Features (X) and target (y)
X = df.drop(columns=['Career_Path'])
y = df['Career_Path']


# In[13]:


from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Scale numerical features (all except categorical ones)
numerical_features = ['Skill_Python', 'Skill_HTML_CSS', 'Skill_SQL', 'Skill_Cloud']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Check the scaled features
X.head()


# In[14]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the train-test split
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[16]:


# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)


# In[17]:


# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=career_paths, yticklabels=career_paths)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[19]:


from sklearn.model_selection import GridSearchCV

# Set hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best model
best_rf_model = grid_search.best_estimator_


# In[20]:


import joblib

# Save the model to a file
joblib.dump(rf_model, "career_path_recommender_model.pkl")

# To load the model in the future:
# model = joblib.load("career_path_recommender_model.pkl")


# In[21]:


pip install streamlit


# In[22]:


import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[23]:


# Load the trained model
model = joblib.load("career_path_recommender_model.pkl")

# Load the scaler for numerical features
scaler = StandardScaler()


# In[24]:


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


# In[25]:


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


# In[26]:


# Make prediction
prediction = model.predict(input_data)

# Career path labels (make sure these correspond to the order in your model)
career_paths = [
    "Machine Learning Engineer", "Web Developer", "Cybersecurity Analyst",
    "UI/UX Designer", "Data Analyst", "Cloud Engineer", "DevOps Engineer", "Business Analyst"
]

# Display the result
st.write(f"Recommended Career Path: {career_paths[prediction[0]]}")


# In[ ]:




