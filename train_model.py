import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — LO1: Data Preparation using Pandas
# ══════════════════════════════════════════════════════════════════

data = pd.read_csv("finshield_dataset_v3_final.csv")

print("Dataset shape:", data.shape)
print("\nNull values:\n", data.isnull().sum())
print("\nLabel distribution:\n", data['label'].value_counts())
print(f"Fraud rate: {data['label'].mean()*100:.2f}%")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — LO2: Visualise Data
# ══════════════════════════════════════════════════════════════════

# Fraud distribution
plt.figure(figsize=(5, 3))
data['label'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title("Fraud vs Normal Distribution")
plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
plt.tight_layout()
plt.savefig("fraud_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — LO2: Data Modeling using Statistics
# ══════════════════════════════════════════════════════════════════

print("\n=== Experiment 3: Statistical Summary by Class ===")
fraud  = data[data['label'] == 1]
normal = data[data['label'] == 0]
stat_cols = ['amount', 'avg_txn_amount', 'amount_deviation', 'failed_attempts', 'txn_count_24h', 'account_age_days']
for col in stat_cols:
    print(f"  {col:<22} Fraud mean: {fraud[col].mean():.2f}   Normal mean: {normal[col].mean():.2f}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — LO4: Classification using Random Forest
# Feature list must EXACTLY match ml_input keys in app.py
# ══════════════════════════════════════════════════════════════════

FEATURES = [
    "amount", "is_mal_ip", "is_new_device", "odd_time",
    "txn_count_24h", "account_age_days", "failed_attempts",
    "location_change", "avg_txn_amount", "is_international"
]

X = data[FEATURES]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"   # handles 7.28% fraud imbalance
)
model.fit(X_train, y_train)
print("\nModel training complete.")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 12 — LO6: Mini Project — Save model
# This .pkl is loaded by app.py for real-time /payment inference
# ══════════════════════════════════════════════════════════════════

joblib.dump(model, "fraud_model.pkl")
print("Model saved: fraud_model.pkl")

# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy:       {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Training Score: {model.score(X_train, y_train)*100:.2f}%")
print(f"Testing Score:  {model.score(X_test, y_test)*100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal","Fraud"]))

# Confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.show()

# Feature importance
importance = model.feature_importances_
sorted_idx = importance.argsort()
feat_series = pd.Series(importance, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importance:\n", feat_series.round(4))
plt.figure(figsize=(8, 5))
plt.barh([FEATURES[i] for i in sorted_idx], [importance[i] for i in sorted_idx], color="steelblue")
plt.title("Feature Importance"); plt.xlabel("Importance Score")
plt.tight_layout(); plt.savefig("feature_importance.png"); plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\nROC AUC Score: {roc_auc:.4f}  (target >0.85, IEEE-CIS benchmark 0.92)")
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],"r--", label="Random (AUC=0.50)")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("roc_curve.png"); plt.show()

print("\nAll experiments complete.")