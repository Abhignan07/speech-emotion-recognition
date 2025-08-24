# src/preprocess.py

import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------------------
# Feature extraction function
# ---------------------------
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Pad or truncate MFCCs to fixed length
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# ---------------------------
# Load dataset
# ---------------------------
def load_data(dataset_path="data/TESS"):
    X, y = [], []

    # Loop over emotion folders
    for emotion in os.listdir(dataset_path):
        emotion_folder = os.path.join(dataset_path, emotion)
        if not os.path.isdir(emotion_folder):
            continue

        print(f"📂 Processing {emotion} folder...")
        for file in os.listdir(emotion_folder):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_folder, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Total samples: {len(X)}")
    print(f"✅ Emotions: {set(y)}")
    return X, y

# ---------------------------
# Encode labels
# ---------------------------
def encode_labels(y, encoder_path="saved_model/label_encoder.pkl"):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save encoder for inference
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    joblib.dump(le, encoder_path)
    print(f"✅ Label encoder saved at {encoder_path}")

    return y_encoded
# For backward compatibility with train_model.py
def extract_features_from_directory(dataset_path="data/TESS"):
    return load_data(dataset_path)
