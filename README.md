# 🎭 MultiModal Emotion Recognition
A deep learning project for **speech-based emotion recognition**. The system preprocesses audio data, trains a model, and provides predictions through an interactive **Streamlit web application**.

---

## 🚀 Features
- 🎙️ Extracts MFCC features from audio signals  
- 🧠 Trains a neural network for emotion classification  
- 📊 Visualizes training performance  
- 🌐 Deployable via **Streamlit** web app  
- 📁 Modular code structure for easy extension  

---

## 📂 Folder Structure
```bash
MultiModal-Emotion-Recognition/
│── app.py                # Streamlit web application
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│
├── src/                  # Source code
│   ├── preprocess.py     # Data preprocessing & feature extraction
│   ├── train_model.py    # Model training script
│   ├── predict.py        # Inference script
│
└── saved_model/          # Saved models
    ├── emotion_model.h5  # Trained model
    └── label_encoder.pkl # Label encoder
```

---

## 🛠️ Installation
1. Clone the repository
```bash
git clone https://github.com/Abhignan07/speech-emotion-recognition.git
cd MultiModal-Emotion-Recognition
```
2. Create & activate virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Download Dataset

This project uses the Toronto Emotional Speech Set (TESS).
👉 [Download here](https://tspace.library.utoronto.ca/handle/1807/24487)

 Extract the dataset and place it in a folder named `data/` inside the project.

---

## 📊 Training the Model

To train the model on your dataset:
```bash
python src/train_model.py
```

Model will be saved in `saved_model/emotion_model.h5`

Encoder will be saved in `saved_model/label_encoder.pkl`

---

## 🔮 Running Predictions

You can test predictions on an audio file using:
```bash
python src/predict.py --file path_to_audio.wav
```
## 🌐 Running the Streamlit Web App

Launch the app:
```bash
streamlit run app.py
```

Then open the browser at the URL shown in the terminal (usually `http://localhost:8501`).

## 📌 Notes

Dataset is ignored in Git (.gitignore) for size/privacy reasons.

Large model files (.h5, .pkl) may also be excluded from GitHub if they exceed limits.

Use tools like Git LFS for large file versioning if needed.

---

## Author

**Abhignan07**  
GitHub: [@Abhignan07](https://github.com/Abhignan07)  
Feel free to contribute or reach out with questions!

---
