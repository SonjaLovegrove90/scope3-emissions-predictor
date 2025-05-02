#!/usr/bin/env python
# coding: utf-8

# # SHAP Explainer

# ## Import Libraries and Dataset

# In[7]:


import shap
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# Paths
MODEL_PATH = "xgb_model.joblib"
DATA_PATH = "preprocessed_data.pkl"
OUTPUT_DIR = "shap_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model and data
model = joblib.load(MODEL_PATH)
X_train, X_test, y_train, y_test = joblib.load(DATA_PATH)


# ## SHAP Explainer

# In[10]:


# Create SHAP explainer and values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Save SHAP summary plot (global)
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"), bbox_inches='tight', dpi=300)
plt.close()

# Optional: Save one instance's SHAP force plot (as HTML)
shap_html = shap.plots.force(explainer.expected_value, shap_values[0].values, X_test.iloc[0], matplotlib=False)
with open(os.path.join(OUTPUT_DIR, "shap_force_plot.html"), "w") as f:
    f.write(shap_html.data)

print("SHAP explainability plots generated and saved.")


# In[ ]:




