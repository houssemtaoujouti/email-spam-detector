# 📧 Email Spam Detector

A Machine Learning web app that detects whether an email is spam or not using XGBoost.

## 🚀 Live Demo
[Click here to try the app](https://email-spam-detector-ht.streamlit.app/)

## 📌 Features
- ✉️ **Email Checker** — Paste an email and get an instant spam prediction with confidence score
- 📂 **Batch Prediction** — Upload a CSV dataset (must contain only the 3000 word columns with their counts, no extra columns) to predict multiple emails at once and compare with true labels
- 📊 **Dataset Statistics** — Explore the training dataset with visualizations
- ℹ️ **About** — General information, model details and downloads

## 🤖 Model
- **XGBoost Classifier** tuned with RandomizedSearchCV (200 iterations, 4 folds)
- **97% Accuracy** on test set
- **96% F1-Score**

## 📊 Dataset
- 5172 emails
- 3000 most common words as features
- Labels: 1 = Spam, 0 = Not Spam

## 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Author
Houssem Taoujouti
