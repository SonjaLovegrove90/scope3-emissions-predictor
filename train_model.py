#!/usr/bin/env python
# coding: utf-8

# # Train Model

# ## Import Libraries and Data

# In[41]:


import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

# Constants
MODEL_DATA_PATH = "preprocessed_data.pkl"
MODEL_OUTPUT_PATH = "xgb_model.joblib"

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load(MODEL_DATA_PATH)


# ## Train XGBoost Model

# In[44]:


# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# ## Metrics

# In[47]:


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

# Save the model
joblib.dump(model, MODEL_OUTPUT_PATH)
print("\n✅ Model trained and saved successfully.")


# In[ ]:




