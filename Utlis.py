#!/usr/bin/env python
# coding: utf-8

# Utlis

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


def cap_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))
    return df


def encode_labels(df, columns):
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def apply_label_encoders(df, encoders):
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    return df


def save_plot(fig, filename, folder="charts"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"✅ Plot saved to {path}")
    return path


# In[ ]:




