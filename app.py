import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# LOAD DATA (PUBLIC DATASET)
# -------------------------------
data = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/fake-news/master/data/train.csv",
    encoding="latin1"
)

# -------------------------------
# CLEAN TEXT FUNCTION
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# -------------------------------
# HANDLE DIFFERENT COLUMN NAMES
# -------------------------------
# Some datasets use different column names
if "text" in data.columns:
    text_col = "text"
elif "content" in data.columns:
    text_col = "content"
else:
    text_col = data.columns[0]  # fallback

if "label" in data.columns:
    label_col = "label"
elif "class" in data.columns:
    label_col = "class"
else:
    label_col = data.columns[-1]  # fallback

# Clean text
data[text_col] = data[text_col].apply(clean_text)

# Fix labels
data[label_col] = data[label_col].astype(str).str.strip().str.lower()
data[label_col] = data[label_col].map({
    "fake": 0,
    "real": 1,
    "0": 0,
    "1": 1
})

# Remove invalid rows
data = data.dropna(subset=[label_col])

# -------------------------------
# INPUT & OUTPUT
# -------------------------------
X = data[text_col]
y = data[label_col]

# -------------------------------
# VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# -------------------------------
# MODEL
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
# 🌐 UI
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="🧠")

st.title("🧠 Fake News Detection System")
st.subheader("By Saniya Dhawade")

st.write("Enter news text to check if it's Fake or Real")

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
