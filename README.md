# 💳 Credit Card Fraud Detection  
## End-to-End Machine Learning + Transformer-Based System

---

## 📌 Overview

This project builds a complete end-to-end fraud detection system to classify credit card transactions as **fraudulent** or **legitimate**.

To solve this highly imbalanced real-world financial problem, multiple approaches were implemented and compared:

- Traditional Machine Learning models  
- Gradient Boosting (LightGBM)  
- Transformer-based Deep Learning architecture  

The objective was to maximize fraud detection performance while minimizing false negatives.

---

## 📊 Dataset

- **Total transactions:** 284,807  
- **Fraudulent transactions:** 492  
- **Fraud rate:** 0.172%  
- **Highly imbalanced dataset**

The dataset contains only numerical features obtained through PCA transformation due to confidentiality constraints.

**Target variable:**
- `0` → Legitimate transaction  
- `1` → Fraudulent transaction  

---

## 🔎 Problem Challenge

Because fraud accounts for only 0.17% of transactions:

- Accuracy becomes misleading  
- Models tend to predict the majority class  
- False negatives are extremely costly  

To address this, **Random Undersampling** was applied to rebalance the dataset.
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy=0.5)
## 🧠 Algorithms Implemented

This project compares classical ML, boosting techniques, and deep learning architectures.

###🔹 Traditional Machine Learning Models
• Logistic Regression
Baseline linear classifier
Improved performance after feature scaling

• Support Vector Machine (SVM)
Effective in high-dimensional feature space
Captures complex decision boundaries

• Decision Tree
Handles non-linear relationships
Interpretable model structure

• Random Forest (Bagging)
Ensemble learning method
Reduces overfitting
Strong performance on structured tabular data

###🔹 Gradient Boosting Model
• LightGBM (Light Gradient Boosting Machine)
Fast training speed
Handles large datasets efficiently
Excellent performance on imbalanced data
Leaf-wise tree growth strategy
Memory efficient and highly optimized
LightGBM significantly improved predictive performance compared to basic tree models.

###🔹 Advanced Deep Learning Model
• Transformer-Based Neural Network
Implemented a Transformer architecture adapted for tabular fraud detection.
Key Components:
Multi-Head Self-Attention
Positional Encoding
Feed-Forward Layers
Layer Normalization
Dropout Regularization
The Transformer captures complex feature interactions and enhances fraud pattern recognition through attention mechanisms.

##⚙️ Data Engineering & Preprocessing
✔ Removed non-informative features
✔ Standardized transaction amount using StandardScaler
✔ Engineered scaled feature (std_Amount)
✔ Visualized fraud distribution
✔ Applied Random Undersampling
✔ Train-Test split (80/20)

##📊 Model Evaluation Strategy
Due to extreme class imbalance, the following metrics were prioritized:
Recall (Primary metric – Fraud Detection Rate)
Precision
F1 Score
ROC-AUC Score
Precision-Recall Curve
Confusion Matrix
Accuracy alone was not used as a primary evaluation metric.

##📂 End-to-End Pipeline
Data Loading
→ Data Cleaning
→ Feature Scaling
→ Handling Class Imbalance
→ Train-Test Split
→ Model Training (ML + LightGBM + Transformer)
→ Model Evaluation
→ ROC & PR Curve Analysis
→ Model Persistence

##📈 Why This Project Stands Out
✔ Solves a real-world financial risk problem
✔ Handles extreme class imbalance properly
✔ Compares multiple ML paradigms
✔ Implements boosting + attention-based deep learning
✔ Uses correct evaluation metrics beyond accuracy
✔ Deployment-ready model saving

##🛠️ Tech Stack
Python
Pandas & NumPy
Scikit-learn
LightGBM
XGBoost
TensorFlow / Keras
Imbalanced-learn
Matplotlib & Seaborn
Joblib

##📈 Core Learning Outcomes
Handling highly imbalanced datasets
Importance of Recall in fraud detection
Comparative model performance analysis
Boosting vs Bagging techniques
Attention mechanisms in tabular data
Model interpretability & evaluation

## Author
Vanya Nain
Machine Learning | Deep Learning | AI Enthusiast
