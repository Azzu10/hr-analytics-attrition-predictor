# 01_data_cleaning.py
# Author: Azmatulla Mohammad
# Purpose: Clean HR dataset for model training

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 📥 Load original dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 🧹 Drop irrelevant columns
df.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"], inplace=True)

# 🔁 Encode binary categorical variables
le = LabelEncoder()
df["Attrition"] = le.fit_transform(df["Attrition"])       # Yes=1, No=0
df["Gender"] = le.fit_transform(df["Gender"])             # Male=1, Female=0
df["OverTime"] = le.fit_transform(df["OverTime"])         # Yes=1, No=0

# 🔄 One-hot encode remaining categorical columns
df = pd.get_dummies(df, drop_first=True)

# 💾 Save cleaned dataset
df.to_csv("data/cleaned_hr_data.csv", index=False)
print("✅ Cleaned data saved to /data/cleaned_hr_data.csv")
