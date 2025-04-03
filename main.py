import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords
nltk.download("stopwords")

# Load the trained model and tokenizer
@st.cache_resource
def load_resources():
    try:
        model = load_model("lstm_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {e}")
        return None, None

model, tokenizer = load_resources()

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
    return cleaned_text if cleaned_text else None

# Prediction function
def predict_sentiment(text):
    """Predicts sentiment for a given text."""
    if model is None or tokenizer is None:
        return "Error", 0.0  # Model or tokenizer not loaded

    text = clean_text(text)
    if not text:
        return "Error", 0.0  # Empty input after preprocessing

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, float(prediction)

# Streamlit UI
st.title("Sentiment Analysis using LSTM")

user_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {round(confidence, 2)}")
    else:
        st.warning("Please enter some text!")
