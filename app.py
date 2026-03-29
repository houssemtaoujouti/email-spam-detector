import streamlit as st
import numpy as np
import joblib
import pandas as pd
import re
from collections import Counter
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
#c'est une fonction pour compter le nombres des mots du mail existant dans notre dataset
def preprocess_email(text):
    # 1. Minuscules
    text = text.lower()
    # 2. Garder uniquement les lettres
    words = re.findall(r'[a-z]+', text)
    # 3. Compter les mots
    word_count = Counter(words)
    # 4. Créer le vecteur avec les 3000 features
    vector = {word: word_count.get(word, 0) for word in feature_names}
    return pd.DataFrame([vector])
# Charger le modèle et les features
model = joblib.load('spam_model1.pkl')
feature_names = joblib.load('feature_names1.pkl')
# Titre de l'app
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Detector")
# Sidebar
st.sidebar.title("📋 Menu")
page = st.sidebar.radio("Navigation", [
    "📂 Predict a dataset",
    "✉️ Email Checker",
    "📊 Dataset Statistics",
    "ℹ️ About"
])
if page == "✉️ Email Checker":
    st.header("✉️ Email Checker")
    email_input = st.text_area("Paste your email here", height=200)
    if st.button("🔍 Check"):
        if email_input.strip() == "":
            st.warning("⚠️ Please enter an email.")
        else:
            X_input = preprocess_email(email_input)
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]
            if prediction == 1:
                st.error("🚨 This email is SPAM!")
            else:
                st.success("✅ This email is NOT spam.")
            col1, col2 = st.columns(2)
            col1.metric("✅ Non-Spam Probability", f"{proba[0]*100:.1f}%")
            col2.metric("🚨 Spam Probability", f"{proba[1]*100:.1f}%")
elif page == "📂 Predict a dataset":
    st.header("📂 Predict a dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"✅ {len(df)} emails loaded")
        y_pred=model.predict(df)
        nb_non_spam=np.sum(y_pred == 0)
        nb_spam=np.sum(y_pred == 1)
        col1, col2 = st.columns(2)
        col1.success(f"{nb_non_spam} ✅ Not Spam")
        col2.error(f"{nb_spam} 🚨 Spam !")
        #comparaison avec les vrais résultats 
        uploaded_file2 = st.file_uploader("Upload true labels", type=["csv"])
        if uploaded_file2 is not None:
            y_test=pd.read_csv(uploaded_file2)
            #classification_report
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            col1, col2, col3, col4 = st.columns(4)
            if col1.checkbox("Accuracy"):
                st.metric("Accuracy", f"{accuracy*100:.1f}%")
            if col2.checkbox("Precision"):
                st.metric("Precision", f"{precision*100:.1f}%")
            if col3.checkbox("Recall"):
                st.metric("Recall", f"{recall*100:.1f}%")
            if col4.checkbox("F1-Score"):
                st.metric("F1-Score", f"{f1*100:.1f}%")
            cm = confusion_matrix(y_test, y_pred)
            if st.checkbox("Confusiion_matrix"):
                st.write("Confusion_matrix")
                st.write(cm)
            if st.button("Classification Results"):
                tn, fp, fn, tp = cm.ravel()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(['True Neg', 'False Pos', 'False Neg', 'True Pos'], 
                   [tn, fp, fn, tp],
                   color=['green', 'red', 'orange', 'blue'])
                ax.set_title("Classification Results")
                ax.set_ylabel('Count')
                st.pyplot(fig)
elif page == "ℹ️ About":
    st.header("ℹ️ About this App")
    st.markdown("---")

    st.subheader("📌 Description")
    st.write("This app uses a Machine Learning model to detect whether an email is **spam or not**.")
    st.write("It also allows you to upload a dataset (CSV file) to predict multiple emails at once and compare results with true labels.")

    st.markdown("---")

    st.subheader("📖 Pages")
    st.write("**✉️ Email Checker** — Paste an email and get an instant spam prediction with confidence score.")
    st.write("**📂 Batch Prediction** — Upload a CSV dataset to predict multiple emails at once. You can also upload the true labels to compare with the model predictions.")
    st.write("**📊 Dataset Statistics** — Explore the training dataset with visualizations : spam distribution and top words in spam and not spam emails.")
    st.write("**ℹ️ About** — You are here ! General information about the app, the dataset, the model and downloads.")

    st.markdown("---")

    st.subheader("📊 Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Emails", "5172")
    col2.metric("Features", "3000 words")
    col3.metric("Classes", "Spam / Not Spam")
    st.warning("⚠️ The uploaded CSV must contain only the 3000 word columns with their counts. No extra columns like 'Email No.' or labels.")

    st.markdown("---")

    st.subheader("🤖 Model")
    st.write("**XGBoost Classifier** tuned with **RandomizedSearchCV** (200 iterations, 4 folds)")
    st.code("learning_rate: 0.15 | n_estimators: 100 | max_depth: 9 | subsample: 0.8 | colsample_bytree: 0.55")

    st.markdown("---")

    st.subheader("📈 Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "97%")
    col2.metric("F1-Score", "96%")
    col3.metric("Precision", "94%")
    col4.metric("Recall", "97%")

    st.markdown("---")

    st.subheader("📥 Downloads")
    col1, col2, col3,col4 = st.columns(4)
    with col1:
        with open("emails.csv", "rb") as f:
            st.download_button("⬇️ Training Dataset", data=f, file_name="emails.csv", mime="text/csv")
    with col2:
        with open("X_test.csv", "rb") as f:
            st.download_button("⬇️ Test Dataset", data=f, file_name="X_test.csv", mime="text/csv")
    with col3:
        with open("y_test.csv", "rb") as f:
            st.download_button("⬇️ True Labels", data=f, file_name="y_test.csv", mime="text/csv")
    with col4:
        with open("spam_detector.ipynb", "rb") as f:
            st.download_button("⬇️ Notebook", data=f, file_name="spam_detector.ipynb", mime="application/octet-stream")
    st.info("💡 Download the **Test Dataset** to try the batch prediction, then download the **True Labels** to compare with the model predictions! Download the **Notebook** to see the full training code.")
    st.markdown("---")

    st.subheader("👨‍💻 Author")
    st.write("**Houssem Taoujouti**")
elif page == "📊 Dataset Statistics":
    st.header("📊 Dataset Statistics")
    df = pd.read_csv("emails.csv")

    # Ligne 1 : Pie chart centré
    st.subheader("📊 Spam vs Not Spam Distribution")
    spam_counts = df["Prediction"].value_counts()
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(spam_counts, labels=["Not Spam", "Spam"], autopct="%1.1f%%", colors=["green", "red"])
    ax.set_title("Spam Distribution")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)

    st.markdown("---")

    # Ligne 2 : deux bar charts côte à côte
    st.subheader("🔝 Top 10 Words")
    col1, col2 = st.columns(2)

    spam_df = df[df["Prediction"] == 1].drop(columns=["Email No.", "Prediction"])
    top_spam_words = spam_df.sum()
    top_spam_words = top_spam_words[top_spam_words.index.str.len() > 2]
    top_spam_words = top_spam_words.sort_values(ascending=False).head(10)
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.bar(top_spam_words.index, top_spam_words.values, color="red")
    ax1.set_title("Top 10 Spam Words")
    ax1.set_ylabel("Count")
    plt.sca(ax1)
    plt.xticks(rotation=45)
    with col1:
        st.pyplot(fig1)

    ham_df = df[df["Prediction"] == 0].drop(columns=["Email No.", "Prediction"])
    top_ham_words = ham_df.sum()
    top_ham_words = top_ham_words[top_ham_words.index.str.len() > 2]
    top_ham_words = top_ham_words.sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(top_ham_words.index, top_ham_words.values, color="green")
    ax2.set_title("Top 10 Not Spam Words")
    ax2.set_ylabel("Count")
    plt.sca(ax2)
    plt.xticks(rotation=45)
    with col2:
        st.pyplot(fig2)
    