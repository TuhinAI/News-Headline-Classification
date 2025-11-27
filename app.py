import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and objects
lr_model = joblib.load("lr_model.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")
tokenizer = joblib.load("tokenizer.pkl")
lstm_model = load_model("lstm_model.h5")

# Parameters
MAX_SEQ_LEN = 30

# Streamlit UI
st.title("News Headline Classifier")
st.write("Classify news headlines into categories like Politics, Sports, Business, Tech, Entertainment, Lifestyle")

headline = st.text_input("Enter a news headline:")

model_choice = st.radio(
    "Choose model:",
    ("Logistic Regression (TF-IDF)", "LSTM (Word Embeddings)")
)

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        if model_choice == "Logistic Regression (TF-IDF)":
            X_tfidf = tfidf.transform([headline])
            pred = lr_model.predict(X_tfidf)[0]
        
        else:  # LSTM
            seq = tokenizer.texts_to_sequences([headline])
            pad_seq = pad_sequences(seq, maxlen=MAX_SEQ_LEN)
            pred_probs = lstm_model.predict(pad_seq)
            pred = np.argmax(pred_probs, axis=1)[0]
        
        category = label_encoder.inverse_transform([pred])[0]
        st.success(f"Predicted Category: **{category}**")
