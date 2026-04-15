# Heart Disease Diagnostic Analysis

## Overview
This project implements a complete machine learning pipeline to predict the presence of heart disease using clinical data.

It includes data preprocessing, exploratory data analysis (EDA), multiple model training, hyperparameter tuning, and detailed performance evaluation using advanced visualizations.

---

## Objectives
- Predict heart disease using clinical features
- Compare multiple machine learning models
- Optimize models using GridSearchCV
- Evaluate models using multiple metrics
- Interpret model behavior using feature importance

---

## Dataset
- Dataset: Heart Disease Dataset (CSV format)
- Features include:
  - Age, Sex, Chest Pain Type
  - Cholesterol, Blood Pressure
  - Max Heart Rate (thalach)
  - Exercise-induced angina
  - Target (0 = No Disease, 1 = Disease)

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## Project Workflow

### 1. Data Preprocessing
- Train-test split (80/20)
- Feature scaling using StandardScaler
- Stratified sampling

---

### 2. Exploratory Data Analysis (EDA)
- Target distribution
- Age distribution by disease
- Correlation heatmap
- Feature relationships
- Multiple visual dashboards

 Output:
- `eda_dashboard.png`

---

### 3. Machine Learning Models

The following models are implemented:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Voting Classifier (Ensemble)

---

### 4. Hyperparameter Tuning
- Performed using GridSearchCV
- 5-fold Stratified Cross Validation
- Optimization metric: ROC-AUC

---

### 5. Model Evaluation

Metrics used:
- Accuracy
- ROC-AUC Score
- Precision-Recall Curve
- Confusion Matrix
- Cross-validation scores

 Outputs:
- `model_comparison.png`
- `cv_boxplot.png`

---

### 6. Model Comparison Dashboard
- Accuracy vs AUC comparison
- ROC curves
- Precision-recall curves
- Confusion matrices for top models

---

### 7. Feature Importance
- Tree-based importance (RF/GB)
- Logistic regression coefficients

 Output:
- `feature_importance.png`

---

## Results
- Ensemble model achieved highest performance
- Strong predictive features identified:
  - Chest pain type
  - Max heart rate (thalach)
  - Cholesterol levels
- Balanced performance across evaluation metrics

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/heart-disease-analysis.git
cd heart-disease-analysis
