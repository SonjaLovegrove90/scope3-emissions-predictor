#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# ## Import Libraries and Dataset

# In[76]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import joblib

# Global constants
DATA_PATH = "emissions_high_granularity (2).csv"
MODEL_DATA_OUTPUT = "preprocessed_data.pkl"
LABEL_ENCODERS_PATH = "label_encoders.pkl"

def load_data(filepath=DATA_PATH):
    return pd.read_csv(filepath)


# ## Missing Data and Outliers

# In[79]:


def handle_missing_and_outliers(df):
    numerical_cols = [
        'total_operational_emissions_MtCO2e',
        'product_emissions_MtCO2',
        'production_value'
    ]

    # Replace infinities
    df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)

    # Treat zero values as missing for key columns
    for col in numerical_cols:
        df[col] = df[col].replace(0, np.nan)

    # IQR capping
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    return df


# ## Label Encoding

# In[82]:


def encode_categoricals(df):
    label_encoders = {}
    cat_cols = ['commodity', 'parent_entity', 'parent_type', 'reporting_entity', 'production_unit']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


# ## Feature Engineering

# In[85]:


def engineer_features(df):
    df['emissions_intensity'] = df['product_emissions_MtCO2'] / (df['production_value'] + 1e-6)
    df['log_production_value'] = np.log1p(df['production_value'])
    df['log_product_emissions'] = np.log1p(df['product_emissions_MtCO2'])
    return df


# ## Preprocessing

# In[88]:


def preprocess():
    df = load_data()
    df = handle_missing_and_outliers(df)
    df, encoders = encode_categoricals(df)
    df = engineer_features(df)

    features = [
        'commodity', 'parent_entity', 'parent_type', 'reporting_entity',
        'production_unit', 'year', 'log_production_value',
        'total_operational_emissions_MtCO2e', 'emissions_intensity'
    ]
    target = 'product_emissions_MtCO2'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save outputs
    joblib.dump((X_train, X_test, y_train, y_test), MODEL_DATA_OUTPUT)
    joblib.dump(encoders, LABEL_ENCODERS_PATH)
    print("Preprocessing complete. Data saved.")


# In[90]:


if __name__ == "__main__":
    preprocess()


# In[ ]:





# In[ ]:





# In[ ]:




