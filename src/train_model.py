import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.preprocess import extract_features_from_directory  # ✅ feature extraction

# ============================
# Paths
# ============================
DATASET_PATH = r"data/TESS"
SAVED_MODEL_PATH = "saved_model/emotion_model.h5"
ENCODER_PATH = "saved_model/label_encoder.pkl"

# ============================
# Load dataset
# ============================
print("🔍 Extracting features...")
X, y = extract_features_from_directory(DATASET_PATH)

print(f"✅ Features extracted: {X.shape}, Labels: {len(y)}")

# ============================
# Encode labels
# ============================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save encoder
os.makedirs("saved_model", exist_ok=True)
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"✅ Label encoder saved at {ENCODER_PATH}")

# ============================
# Train-test split
# ============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Reshape for Conv2D: (samples, height, width, channels)
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)

print(f"📊 Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# ============================
# Build Conv2D Model
# ============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================
# Callbacks
# ============================
checkpoint = ModelCheckpoint(SAVED_MODEL_PATH, monitor='val_accuracy',
                             save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ============================
# Train
# ============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# ============================
# Plot Training History
# ============================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training History")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

print(f"🎉 Model training completed. Best model saved at {SAVED_MODEL_PATH}")
