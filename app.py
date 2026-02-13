import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.title("Machine Learning Classification App")

uploaded_file = st.file_uploader("Upload CSV file (Adult Dataset)", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file is not None:

    # Adult dataset column names
    columns = ["age","workclass","fnlwgt","education","education-num","marital-status",
               "occupation","relationship","race","sex","capital-gain","capital-loss",
               "hours-per-week","native-country","income"]

    # Load dataset
    df = pd.read_csv(uploaded_file, names=columns, sep=",", skipinitialspace=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop("income", axis=1)
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 🔥 FIX FOR KNN MEMORY ISSUE
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    # Model dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN":  KNeighborsClassifier(algorithm="brute"),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    model = models[model_choice]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("AUC:", roc_auc_score(y_test, y_prob))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("MCC:", matthews_corrcoef(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
