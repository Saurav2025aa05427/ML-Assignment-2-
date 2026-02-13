Machine Learning Assignment 2

Income Classification using Multiple ML Models

1\. Problem Statement



The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual's income exceeds $50K per year. The project demonstrates end-to-end machine learning workflow including data preprocessing, model training, evaluation, web application development using Streamlit, and cloud deployment.



2\. Dataset Description



This project uses the Adult Income Dataset from the UCI Machine Learning Repository.



Total Instances: ~32,000



Total Features: 14



Target Variable: Income (>50K or <=50K)



Type: Binary Classification



The dataset contains demographic and employment-related attributes such as age, education, occupation, hours-per-week, marital status, etc.



3\. Models Implemented



The following 6 classification models were implemented and evaluated on the same dataset:



Logistic Regression



Decision Tree Classifier



K-Nearest Neighbors (KNN)



Naive Bayes (GaussianNB)



Random Forest (Ensemble)



XGBoost (Ensemble)



4\. Evaluation Metrics



Each model was evaluated using the following metrics:



Accuracy



AUC Score



Precision



Recall



F1 Score



Matthews Correlation Coefficient (MCC)



5\. Model Comparison Table

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |

| ------------------- | -------- | ----- | --------- | ------ | -------- | ----- |

| Logistic Regression | 0.846    | 0.898 | 0.744     | 0.600  | 0.664    | 0.572 |

| Decision Tree       | 0.811    | 0.752 | 0.626     | 0.630  | 0.628    | 0.501 |

| KNN                 | 0.760    | 0.669 | 0.550     | 0.312  | 0.398    | 0.279 |

| Naive Bayes         | 0.791    | 0.831 | 0.689     | 0.320  | 0.438    | 0.366 |

| Random Forest       | 0.850    | 0.902 | 0.737     | 0.641  | 0.685    | 0.591 |

| XGBoost             | 0.870    | 0.927 | 0.781     | 0.678  | 0.726    | 0.644 |





6\. Observations on Model Performance

ML Model	Observation

Logistic Regression	Performs well as a strong linear baseline model with good AUC score.

Decision Tree	Shows reasonable performance but prone to overfitting.

KNN	Lower performance due to high-dimensional feature space after encoding.

Naive Bayes	Performance limited due to feature independence assumption.

Random Forest	Strong ensemble model with improved stability and accuracy.

XGBoost	Best performing model with highest Accuracy, AUC, F1-score, and MCC.

7\. Streamlit Web Application Features



The deployed Streamlit application includes:



CSV Dataset Upload



Model Selection Dropdown



Evaluation Metrics Display



Confusion Matrix Display



8\. Project Structure

ML\_Assignment\_2/

│-- app.py

│-- requirements.txt

│-- README.md

│-- model/



9\. Deployment

The application is deployed using Streamlit Community Cloud.

