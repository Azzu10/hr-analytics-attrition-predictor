# 03_model_building.py
# Author: Azmatulla Mohammad
# Purpose: Train and evaluate classification models

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Load cleaned data
df = pd.read_csv("data/cleaned_hr_data.csv")

# Split features & target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Model 1: Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("\n🔹 Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Model 2: Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("\n🔹 Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Save the best model (logreg)
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(logreg, f)

print("\n✅ Best model (Logistic Regression) saved to models/model.pkl")
