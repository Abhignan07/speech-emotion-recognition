import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import extract_features
import os

# Load model
MODEL_PATH = "saved_model/audio_emotion_model.h5"
model = load_model(MODEL_PATH)

# Emotion labels (must match training)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, -1)  # match CNN input
    prediction = model.predict(features)
    return emotions[np.argmax(prediction)]
