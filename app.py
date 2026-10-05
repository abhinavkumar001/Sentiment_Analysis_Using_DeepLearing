import streamlit as st
import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = load_model(MODEL_DIR / "sentiment_model.keras")

with open(MODEL_DIR / "tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open(MODEL_DIR / "label_encode.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🐦 X Sentiment Analysis")

st.write(
    "Enter a tweet and the model will predict its sentiment."
)

tweet = st.text_area("Enter your tweet:")


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Sentiment"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:

        # Clean tweet
        cleaned_tweet = clean_text(tweet)

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences(
            [cleaned_tweet]
        )

        # Padding
        padded_sequence = pad_sequences(
            sequence,
            maxlen=70,
            padding="post",
            truncating="post"
        )

        # Prediction
        prediction = model.predict(
            padded_sequence,
            verbose=0
        )

        # Get predicted class
        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]

        # Convert class number to sentiment
        sentiment = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        # Display result
        st.success(
            f"Predicted Sentiment: {sentiment}"
        )



# streamlit run app.py