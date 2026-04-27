# 🔬 SkinScan AI — Skin Disease Detection Application
**B.Tech CSE Final Year Project**

A deep learning-based web application that detects skin diseases from uploaded images using Transfer Learning (MobileNetV2) and presents results with a modern Streamlit UI.

---

## 📁 Project Structure

```
skin_disease_detector/
│
├── app.py                    # Main Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── model/
│   ├── train_model.py        # Model training script
│   ├── skin_disease_model.h5 # Saved model (generated after training)
│   └── class_names.txt       # Class labels (generated after training)
│
├── utils/
│   ├── preprocess.py         # Image preprocessing helpers
│   └── disease_info.py       # Disease descriptions and remedies
│
└── dataset/                  # Your training dataset (you download this)
    ├── Acne/
    ├── Eczema/
    ├── Psoriasis/
    └── ...
```

---

## 🛠️ Step-by-Step Setup Guide

### Step 1 — Install Python
Make sure Python 3.9, 3.10, or 3.11 is installed.
```bash
python --version
```

### Step 2 — Create a Virtual Environment
```bash
# Navigate to project folder
cd skin_disease_detector

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```
This installs: Streamlit, TensorFlow, NumPy, Pillow, Matplotlib, scikit-learn.

---

## 📦 Step 4 — Download Dataset

### Option A — DermNet Dataset (Recommended)
1. Go to: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
2. Download and unzip into a `dataset/` folder in the project root
3. Ensure each disease has its own subfolder with images

### Option B — Skin Disease Dataset
1. Go to: https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset
2. Download and organize similarly

### Dataset Folder Structure (Required)
```
dataset/
├── Acne/
│   ├── acne_001.jpg
│   ├── acne_002.jpg
│   └── ...
├── Eczema/
│   ├── eczema_001.jpg
│   └── ...
├── Psoriasis/
│   └── ...
└── (other disease folders...)
```
> At least 100–200 images per class is recommended for decent accuracy.

---

## 🧠 Step 5 — Train the Model

```bash
python model/train_model.py
```

This will:
- Load and augment your dataset automatically
- Train a MobileNetV2-based transfer learning model in 2 phases
- Save `model/skin_disease_model.h5` and `model/class_names.txt`
- Print training accuracy and validation metrics

**Training time:** ~10–30 minutes depending on dataset size and GPU/CPU.

> 💡 **Tip:** Use Google Colab with GPU for faster training. Upload your dataset, run train_model.py, then download the `.h5` file.

---

## 🚀 Step 6 — Run the Application

```bash
streamlit run app.py
```

- The app opens at: **http://localhost:8501**
- Upload a skin image → Click "Analyze Image" → Get results!

---

## 🌐 Step 7 — Deploy Online (Optional)

### Deploy on Streamlit Cloud (Free)
1. Push project to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo and set `app.py` as the entry point
4. Add model files via Git LFS or download them in the app startup

---

## 🖥️ Running Without a Trained Model (Demo Mode)
If you haven't trained the model yet, the app will run in **Demo Mode** with simulated (random) predictions. All UI features, disease info, and encyclopedia are fully functional.

---

## 📊 Model Details

| Property        | Value                     |
|----------------|---------------------------|
| Base Model     | MobileNetV2 (ImageNet)    |
| Input Size     | 224 × 224 px              |
| Augmentation   | Rotation, flip, zoom, etc |
| Optimizer      | Adam                      |
| Loss Function  | Categorical Crossentropy  |
| Output         | Softmax (multi-class)     |

---

## 🩺 Detectable Diseases
- Athlete-foot
- Cellulitis
- Chickenpox
- Cutaneous-larva-migrans
- Impetigo
- Nail-fungus
- Ringworm
- Shingles

---

## ⚠️ Disclaimer
This application is built for **educational and academic purposes only**. It is NOT intended for real medical diagnosis. Always consult a certified dermatologist for any skin condition.

---

## 👨‍💻 Tech Stack
- **Frontend/UI:** Streamlit
- **Deep Learning:** TensorFlow / Keras
- **Model:** MobileNetV2 (Transfer Learning)
- **Image Processing:** Pillow, NumPy
- **Language:** Python 3.10+
