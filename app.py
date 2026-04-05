import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# SMALL BUILT-IN DATASET
# -------------------------------
data = pd.DataFrame({
    "text": [
        "Government announces new policy for education",
        "Aliens landed in Mumbai yesterday shocking everyone",
        "Stock market reaches all time high today",
        "Scientists discovered new species in ocean",
        "Celebrity caught in fake scandal news spreading online"
    ],
    "label": [1, 0, 1, 1, 0]  # 1 = Real, 0 = Fake
})

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data["text"] = data["text"].apply(clean_text)

# -------------------------------
# MODEL TRAINING
# -------------------------------
X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

model = LogisticRegression()
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


