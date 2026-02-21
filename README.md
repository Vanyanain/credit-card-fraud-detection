Credit Card Fraud Detection | End-to-End Machine Learning Project


Overview:

Our objective is to create the classifier for credit card fraud detection. To do it, we'll compare classification models from different methods :

Logistic regression

Support Vector Machine

Bagging (Random Forest)

Boosting (XGBoost)

Neural Network (tensorflow/keras)

Dataset:

Credit Card Fraud Detection

The datasets contains transactions made by credit cards in September 2020 by Indian cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. I decided to proceed to an undersampling strategy to re-balance the class.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.
Data Exploration:

Only 492 (or 0.172%) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable Class.

The dataset is highly imbalanced ! It's a big problem because classifiers will always predict the most common class without performing any analysis of the features and it will have a high accuracy rate, obviously not the correct one. To change that, I will proceed to random undersampling.

The simplest undersampling technique involves randomly selecting examples from the majority class and deleting them from the training dataset. This is referred to as random undersampling.

Although simple and effective, a limitation of this technique is that examples are removed without any concern for how useful or important they might be in determining the decision boundary between the classes. This means it is possible, or even likely, that useful information will be deleted.

For undersampling, we can use the package imblearn with RandomUnderSampler function.

import imblearn
from imblearn.under_sampling import RandomUnderSampler 
undersample = RandomUnderSampler(sampling_strategy=0.5)

This project demonstrates:
Strong ML fundamentals
Data preprocessing expertise
Imbalanced data handling
Model comparison & evaluation
Production-ready model saving


🎯 Business Problem
Credit card fraud causes significant financial losses.
The challenge is:
Fraud transactions represent only ~0.17% of total data
Traditional accuracy metrics are misleading
False negatives are costly
The objective was to build models that effectively identify fraud transactions while handling extreme class imbalance.

🛠️ Tech Stack
Python
Scikit-learn
XGBoost
TensorFlow / Keras
Imbalanced-learn
Pandas & NumPy
Matplotlib & Seaborn
Joblib (Model Persistence)
🔍 Key Technical Contributions

✅ Data Engineering
Removed non-informative features
Standardized transaction amount using StandardScaler
Created engineered feature: std_Amount
Visualized class imbalance
Applied Random UnderSampling to balance dataset

✅ Model Development
Implemented and compared multiple algorithms:
Logistic Regression
Support Vector Machine (SVM)
Random Forest
XGBoost
MLP Classifier
Artificial Neural Network (Keras)

✅ Model Evaluation Strategy
Since dataset is highly imbalanced, prioritized:
Recall (Fraud Detection Rate)
Precision
F1 Score
ROC-AUC Score
Precision-Recall Curve
Confusion Matrix Analysis

📊 Performance Highlights

Improved fraud detection using balanced dataset
Logistic Regression performed efficiently after scaling
Tree-based models handled non-linearity effectively
ROC-AUC used as primary performance metric
Best-performing model saved for deployment readiness

📂 Project Pipeline

Data Loading → Cleaning → Scaling → Handling Imbalance →
Train-Test Split → Model Training → Evaluation →
ROC & PR Curve Analysis → Model Saving

💡 Why This Project Stands Out
✔ Demonstrates understanding of real-world financial risk problems
✔ Handles extreme class imbalance properly
✔ Uses multiple ML algorithms for comparison
✔ Applies correct evaluation metrics beyond accuracy
✔ Shows deployment readiness via model persistence

📈 Core Learning Outcomes

Practical experience with imbalanced datasets
Importance of Recall in fraud detection
Comparative model analysis
Feature scaling impact on linear models
Model interpretability via confusion matrix & ROC curves

👩‍💻 Author
Vanya Nain
Machine Learning & Data Science Enthusiast


