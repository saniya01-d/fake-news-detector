import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# LOAD DATA FROM GITHUB (FIXED)
# -------------------------------
data = pd.read_csv(
    "https://raw.githubusercontent.com/YOUR-USERNAME/fake-news-detector/main/train.csv",
    encoding="latin1"
)

# Remove useless column
data = data.drop(columns=["Unnamed: 6"], errors='ignore')

# -------------------------------
# CLEAN TEXT FUNCTION
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data["text"] = data["text"].apply(clean_text)

# -------------------------------
# FIX LABELS
# -------------------------------
data["class"] = data["class"].str.strip().str.lower()
data["class"] = data["class"].map({"fake": 0, "real": 1})
data = data.dropna(subset=["class"])

# -------------------------------
# USE TITLE + TEXT
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
# MODEL
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------------
# PREDICTION
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
# UI
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="🧠")

st.title("🧠 Fake News Detection System")
st.subheader("By Saniya Dhawade")

user_input = st.text_area("Enter News Text:")

if st.button("Check News"):
    if user_input.strip() != "":
        result, confidence = predict_news(user_input)
        st.subheader(result)
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text")

st.markdown("---")
st.markdown("Created by Saniya Dhawade 🚀")
