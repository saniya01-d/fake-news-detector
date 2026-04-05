import pandas as pd
import re

# STEP 1: Load dataset
data = pd.read_csv("train.csv", encoding="latin1")

# STEP 2: Check columns
print(data.columns)

# STEP 3: Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data["text"] = data["text"].apply(clean_text)

# STEP 4: Fix labels (VERY IMPORTANT)
data["class"] = data["class"].str.strip().str.lower()
data["class"] = data["class"].map({"fake": 0, "real": 1})

# Remove rows with invalid labels
data = data.dropna(subset=["class"])

# STEP 5: Input & Output
X = data["text"]
y = data["class"]

# STEP 6: Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# STEP 7: Train model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# STEP 8: Test model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# STEP 9: Prediction function
def predict_news(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    
    prediction = model.predict(vector)
    probability = model.predict_proba(vector)

    confidence = max(probability[0]) * 100

    if prediction[0] == 0:
        return f"Fake News ❌ ({confidence:.2f}%)"
    else:
        return f"Real News ✅ ({confidence:.2f}%)"

# STEP 10: User input
user_input = input("Enter news text: ")
print(predict_news(user_input))