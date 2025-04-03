import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords if not already available
nltk.download("stopwords")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
try:
    model = load_model("lstm_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None if loading fails

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None  # Set to None if loading fails

# Text preprocessing function
def clean_text(text):
    """Cleans text by lowercasing, removing stopwords, and applying stemming."""
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    cleaned_text = " ".join(words)
    if not cleaned_text:
        print("Error: Cleaned text is empty!")
    return cleaned_text

# Prediction function
def predict_sentiment(text):
    """Predicts sentiment for a given text."""
    if model is None or tokenizer is None:
        print("Error: Model or tokenizer is not loaded!")
        return "Error", 0.0  # Return default values if model or tokenizer is missing

    text = clean_text(text)
    if not text:
        print("Error: Input text is empty after preprocessing!")
        return "Error", 0.0  # Handle empty text input

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"

    print(f"Debug - Prediction Output: Sentiment={sentiment}, Confidence={prediction}")
    return sentiment, float(prediction)

# Home and prediction route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text")
        if text:
            result, confidence = predict_sentiment(text)

    if confidence is None:
        confidence = 0.0  # Default value to prevent error

    return render_template("index.html", result=result, confidence=round(confidence, 2))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
