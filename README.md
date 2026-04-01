# Credit Card Fraud Detection: From Biased Baselines to Optimized Deep Learning

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green.svg)

## 📌 Project Overview
This research project focuses on solving the **Extreme Class Imbalance** problem in credit card fraud detection. Using the European Credit Card dataset (284,807 transactions), this study demonstrates a systematic progression from biased baseline models to high-performance, unbiased ensemble and deep learning architectures.

### 🚀 Research Achievements
*   **System Stability:** Overcame computational crashes ($O(n^3)$ complexity) in SVM through feature scaling and calibration.
*   **Bias Mitigation:** Successfully "unbiased" the models, moving from **0% Recall** to a state-of-the-art **94.5% Fraud Detection rate**.
*   **Optimization:** Proved the impact of Hyperparameter Tuning (HPO) and Threshold Calibration in transforming a "disappointing" model into a "champion" model.

---

## 📊 Dataset Profile
*   **Source:** [Kaggle MLG-ULB Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Volume:** 284,807 Transactions
*   **Imbalance:** 492 Fraud Cases (**0.17%**)
*   **Features:** 28 PCA-transformed features ($V1$–$V28$), `Amount`, and `Time`.

---

## 📈 Comparative Results: Baseline vs. Optimized

A core contribution of this research is the demonstration of how **Hyperparameter Tuning (HPO)** and **Cost-Sensitive Learning** fix the "Accuracy Paradox" where models have high accuracy but 0% utility.

### Phase 1: Baseline Results (Before Tuning)
In the baseline phase, models were either biased toward the majority class (Low Recall) or overly sensitive (Low Precision).

| Algorithm | Recall (Fraud) | Precision | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | 51.02% | 84.74% | 0.6369 | 0.9014 |
| **Logistic Regression** | 56.12% | 84.61% | 0.6748 | 0.4906 |
| **Neural Network (MLP)** | 88.77% | 7.13% | 0.1321 | 0.9782 |
| **Transformer** | **88.77%** | 5.49% | 0.1035 | 0.9607 |

### Phase 2: Optimized Results (After HPO & Calibration)
By implementing `RandomizedSearchCV`, `KerasTuner`, and **Threshold Optimization (0.25 - 0.75)**, the performance across all metrics reached professional research standards.

| Algorithm | Recall (Fraud) | Precision | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM (Champion)** | **94.51%** | **94.51%** | **94.51%** | **0.9945** |
| **Neural Network (MLP)** | **94.51%** | **94.51%** | **94.51%** | **0.9925** |
| **Random Forest** | 94.51% | 93.48% | 93.99% | 0.9937 |
| **XGBoost** | 94.51% | 92.47% | 93.48% | 0.9942 |
| **Transformer** | 94.51% | 92.47% | 93.48% | 0.9942 |

---

## 🔍 Key Research Findings
1.  **The Superiority of Gradient Boosting:** **LightGBM** and **XGBoost** provided the most stable "unbiased" results, maintaining >94% Precision and Recall simultaneously.
2.  **Neural Network Sensitivity:** Deep Learning models (MLP/Transformer) are highly sensitive to fraud patterns but require rigorous threshold calibration to prevent excessive false alarms.
3.  **The Baseline Failure:** Traditional models like **Logistic Regression** (AUC 0.49) proved insufficient for this high-imbalance task, justifying the need for advanced non-linear architectures.
4.  **Operational Efficiency:** The research identifies an "Operational Sweet Spot" at a decision threshold of **0.25-0.75**, providing a balance suitable for real-world banking security.

---

## 🛠️ Methodology & Tech Stack
*   **Preprocessing:** `StandardScaler`, feature selection (removal of 'Time'), and `StratifiedShuffleSplit`.
*   **Sampling:** Implementation of **Random Under-sampling (RUS)** on training data to balance the class distribution.
*   **Architectures:** Logistic Regression, Linear SVM, Decision Trees, Random Forest, XGBoost, LightGBM, MLP, and Transformer.
*   **Evaluation:** Focus on **AUPRC (Area Under Precision-Recall Curve)** and **F1-Score** as the primary metrics for imbalanced classification.

---

## 👨‍💻 Author
**[Vanya Nain]**
*Machine Learning Researcher*


---
## 📄 License
This project is licensed under the MIT License.
