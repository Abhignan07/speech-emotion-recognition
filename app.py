import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import joblib

# -------------------------------
# Load Model and Label Encoder
# -------------------------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("saved_model/emotion_model.h5")
    label_encoder = joblib.load("saved_model/label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(y, sr, n_mfcc=40, max_pad_len=174):
    """
    Extract MFCC features and pad/truncate to fixed length
    Shape: (n_mfcc, max_pad_len, 1)
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate along time axis
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_pad_len]

    # Add channel dimension → (n_mfcc, max_pad_len, 1)
    mfccs = np.expand_dims(mfccs, axis=-1)

    return mfccs

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio file to predict the emotion.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    try:
        # Load audio safely
        y, sr = sf.read(uploaded_file, dtype="float32")

        # Stereo → Mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Ensure float32
        y = np.array(y, dtype=np.float32)

        # Extract features
        features = extract_features(y, sr)

        # Reshape for batch dimension → (1, n_mfcc, max_pad_len, 1)
        features = np.expand_dims(features, axis=0)

        st.write(f"✅ Feature shape for model: {features.shape}")

        # Prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        emotion = label_encoder.inverse_transform(predicted_class)[0]

        st.success(f"### 😃 Predicted Emotion: **{emotion}**")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
