# Breast Cancer Classifier Web App

A simple web application using Streamlit and a neural network to classify breast cancer as **Malignant** or **Benign** based on input features.

## 🔧 How to Run

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python train_model.py
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

## 🧠 Model

- Neural Network: `MLPClassifier` from `scikit-learn`
- Dataset: Breast Cancer Wisconsin dataset (built-in from `sklearn`)
