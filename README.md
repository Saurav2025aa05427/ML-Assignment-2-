# Machine Learning Assignment 2
## Income Classification using Multiple ML Models

---

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual's income exceeds $50K per year.

This project demonstrates an end-to-end machine learning workflow including:

- Data preprocessing
- Model training
- Model evaluation
- Web application development using Streamlit
- Cloud deployment

---

## 2. Dataset Description

This project uses the **Adult Income Dataset** from the UCI Machine Learning Repository.

- Total Instances: ~32,000
- Total Features: 14
- Target Variable: Income (>50K or <=50K)
- Problem Type: Binary Classification

The dataset contains demographic and employment-related attributes such as age, education, occupation, hours-per-week, marital status, etc.

---

## 3. Models Implemented

The following 6 classification models were implemented and evaluated:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Naive Bayes (GaussianNB)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

---

## 4. Evaluation Metrics

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

---

## 5. Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|------|----------|--------|----------|------|
| Logistic Regression | 0.846 | 0.898 | 0.744 | 0.600 | 0.664 | 0.572 |
| Decision Tree | 0.813 | 0.752 | 0.628 | 0.647 | 0.637 | 0.512 |
| KNN | 0.760 | 0.669 | 0.550 | 0.312 | 0.398 | 0.279 |
| Naive Bayes | 0.791 | 0.831 | 0.689 | 0.320 | 0.438 | 0.366 |
| Random Forest | 0.849 | 0.904 | 0.733 | 0.639 | 0.683 | 0.588 |
| XGBoost | 0.870 | 0.927 | 0.781 | 0.678 | 0.726 | 0.644 |

---

## 6. Observations

- Logistic Regression performs well as a strong linear baseline model.
- Decision Tree shows reasonable performance but may overfit.
- KNN performs lower due to high dimensionality after encoding.
- Naive Bayes is limited due to independence assumption.
- Random Forest improves stability and accuracy.
- XGBoost achieved the best overall performance across most metrics.

---

## 7. Streamlit Web Application Features

The Streamlit application includes:

- CSV Dataset Upload
- Model Selection Dropdown
- Evaluation Metrics Display
- Confusion Matrix Display

---

## 8. Project Structure
ML_Assignment_2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   └── training.py


---

## 9. Installation & Running the Project

### Step 1: Clone the Repository

git clone https://github.com/Saurav2025aa05427/ML-Assignment-2-



### Step 2: Install Dependencies

pip install -r requirements.txt


### Step 3: Run the Streamlit App
streamlit run app.py

---

## 10. Deployment

The application is deployed using Streamlit Community Cloud.
