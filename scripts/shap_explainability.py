# 04_shap_explainability.py
# Author: Azmatulla Mohammad
# Purpose: Generate SHAP summary plot using saved logistic model with shape check

import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle

# Load data
df = pd.read_csv("data/cleaned_hr_data.csv")
X = df.drop("Attrition", axis=1)

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Use KernelExplainer with 100-row background
background = X.sample(n=100, random_state=42)
explainer = shap.KernelExplainer(model.predict_proba, background)

# Explain predictions on 100 rows
X_sample = X.iloc[:100, :]
shap_values = explainer.shap_values(X_sample)

# Check shape alignment
print(f"SHAP output type: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"SHAP values[1] shape: {shap_values[1].shape}")
    print(f"X_sample shape: {X_sample.shape}")
    shap.summary_plot(shap_values[1], X_sample, show=False)
else:
    print(f"SHAP shape: {shap_values.shape}")
    shap.summary_plot(shap_values, X_sample, show=False)

# Save plot
plt.tight_layout()
plt.savefig("app/shap_summary_plot.png", dpi=300)
print("✅ SHAP summary plot saved to app/shap_summary_plot.png")
