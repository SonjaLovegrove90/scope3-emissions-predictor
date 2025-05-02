# Scope 3 Emissions Prediction Platform

This repository contains the full implementation of a machine learning platform developed for my MSc Dissertation: **Predicting Scope 3 Emissions Using Machine Learning on ERP-style Data**.

The project includes data preprocessing, model training, SHAP-based explainability, and a user-facing interface built with Voila.

---

## Project Overview

Scope 3 emissions are notoriously hard to measure, relying heavily on upstream and downstream supplier data. This platform demonstrates how machine learning, using ERP-style inputs, can provide fast, explainable estimates of product-level emissions.

**Core components:**
- Structured preprocessing pipeline with encoding, transformation, and feature engineering
- XGBoost model for emissions prediction (with cross-validation)
- SHAP integration for interpretability
- Interactive UI built using Voila (Jupyter Widgets)

---

## Project Structure

```
scope3-emissions-predictor/
├── notebooks/
│   └── scope3_voila_ui_dropdowns.ipynb        # Main Voila interface
├── scripts/
│   ├── data_preprocessing.py                 # Cleans and transforms raw data
│   ├── train_model.py                        # Trains XGBoost model
│   ├── shap_explainer.py                     # Generates SHAP plots
│   └── utils.py                              # Helper functions
├── models/
│   └── xgb_model.joblib                      # Trained model (optional)
├── requirements.txt                          # All dependencies
├── README.md                                 # This file
```

---

## How to Run

### 1. Install Dependencies
Create a virtual environment and run:
```bash
pip install -r requirements.txt
```

Make sure you're using `ipywidgets==7.7.2` for Voila compatibility.

### 2. Train the Model
```bash
python scripts/data_preprocessing.py
python scripts/train_model.py
```

### 3. Generate SHAP Explanations (optional)
```bash
python scripts/shap_explainer.py
```

### 4. Launch the UI
```bash
voila notebooks/scope3_voila_ui_dropdowns.ipynb
```

---

## Example Input
```
Commodity: Cement
Production Value: 42000
Unit: tonnes
Parent Entity: GlobalCem
Reporting Entity: Plant_A
Year: 2022
```

Output: `Predicted Emissions: 0.3742 MtCO2`

Top SHAP Contributors:
```
log_production_value: 0.19
parent_entity: 0.15
commodity: 0.12
```

---

## Dissertation Details
This platform supports the research findings in:

**Title**: _Predicting Scope 3 Emissions Using Machine Learning on ERP-style Data_  
**Author**: [Sonja Lovegrove]  
**Degree**: MSc Data Science & Applied AI  
**University**: [Solent University]  
**Year**: 2025
