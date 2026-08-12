# ❤️ Heart Disease Prediction System

A machine-learning-based web application that predicts heart disease risk from clinical patient data. The system uses a tuned Random Forest classifier and provides both single-patient and bulk CSV prediction capabilities through an interactive Streamlit interface.

The final model was selected after comparing Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine models, followed by hyperparameter tuning and threshold optimization.

---

## 🚀 Live Application

🔗 **Streamlit App:**  
Add your deployed Streamlit URL here

---

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Early identification of potential risk can support further medical evaluation and decision-making.

This project develops a machine learning classification system that analyzes clinical attributes and predicts whether a patient is likely to have heart disease.

The application provides:

- Single-patient prediction
- Bulk prediction using CSV files
- Optimized decision threshold
- Model performance comparison
- Prediction probability/score
- Downloadable prediction results
- Dockerized deployment

> ⚠️ **Medical Disclaimer:** This application is an educational machine learning project and is not intended to provide medical diagnosis or replace professional medical advice.

---

## 🎯 Objectives

- Build a complete machine learning classification pipeline.
- Preprocess numerical and categorical clinical features.
- Compare multiple classification algorithms.
- Tune model hyperparameters using cross-validation.
- Optimize the classification threshold.
- Select the best-performing model.
- Deploy the final model through Streamlit.
- Containerize the application using Docker.

---

## 📊 Dataset

The project uses clinical heart disease data containing **11 input features**.

### Features

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | Patient sex |
| ChestPainType | Type of chest pain |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar indicator |
| RestingECG | Resting electrocardiogram result |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina |
| Oldpeak | ST depression |
| ST_Slope | Slope of peak exercise ST segment |

---

## 🔄 Machine Learning Pipeline

The project follows the following workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Train-Test Split
     ↓
Numerical & Categorical Preprocessing
     ↓
Model Training
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Threshold Optimization
     ↓
Final Random Forest Model
     ↓
Streamlit Deployment
     ↓
Docker Containerization