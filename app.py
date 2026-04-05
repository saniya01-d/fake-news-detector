import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv("train.csv", encoding="latin1")

# Remove useless column (Improvement 4)
data = data.drop(columns=["Unnamed: 6"], errors='ignore')

# -------------------------------
# CLEAN TEXT FUNCTION
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

# -------------------------------
# FIX LABELS
# -------------------------------
data["class"] = data["class"].str.strip().str.lower()
data["class"] = data["class"].map({"fake": 0, "real": 1})

# Remove invalid rows
data = data.dropna(subset=["class"])

# -------------------------------
# IMPROVEMENT: USE TITLE + TEXT
# -------------------------------
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["class"]

# -------------------------------
# VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer(max_features=7000)
X = vectorizer.fit_transform(X)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_news(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    
    prediction = model.predict(vector)
    probability = model.predict_proba(vector)

    confidence = max(probability[0]) * 100

    if prediction[0] == 0:
        return "Fake News ❌", confidence
    else:
        return "Real News ✅", confidence


# -------------------------------
# 🌐 STREAMLIT UI
# -------------------------------
st.title("🧠 Fake News Detection System")
st.write("Check whether a news article is Fake or Real")

user_input = st.text_area("Enter News Text:")

if st.button("Check News"):
    if user_input.strip() != "":
        result, confidence = predict_news(user_input)
        st.subheader(result)
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text")