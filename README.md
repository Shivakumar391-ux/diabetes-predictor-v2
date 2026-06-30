
# 🏥 Diabetes Patient Readmission Risk Predictor

A machine learning project that predicts hospital readmission risk for diabetic patients, estimates length of hospital stay, and segments patients into risk profiles.

## 🚀 Problem Statement
Hospitals struggle to predict which discharged diabetic patients will be readmitted within 30 days, leading to poor resource planning and patient outcomes.

## 📊 Dataset
- Source: UCI ML Repository — Diabetes 130-US Hospitals (ID: 296)
- 101,766 patients, 48 features
- Target: Readmitted (<30 days / >30 days / No)

## 🤖 ML Models Used
| Task | Algorithm | Performance |
|------|-----------|-------------|
| Readmission Classification | XGBoost | 68% Accuracy, 0.56 Recall |
| Hospital Stay Prediction | Random Forest Regressor | R2 = 0.36 |
| Patient Segmentation | K-Means Clustering | 4 Patient Profiles |

## 👥 Patient Cluster Profiles
- Cluster 0: Mild Cases — Short stay, few medications
- Cluster 1: Complex Cases — Long stay, many medications
- Cluster 2: Moderate Cases — Average stay, moderate procedures
- Cluster 3: High Risk — Frequent hospital visitors

## 🛠️ Tech Stack
Python, XGBoost, Scikit-learn, Pandas, NumPy, Streamlit, Imbalanced-learn

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run healthcare_app.py
```
