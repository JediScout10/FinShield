import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score

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
# FIX 1: Compute amount_deviation before any training step.
#
# WHY: The model needs to judge whether an amount is unusual FOR
# THIS USER, not whether it is large in absolute terms.
# amount=25 for a user whose avg is Rs.5,665 = deviation 0.004 (safe)
# amount=200 for a user whose avg is Rs.46   = deviation 4.35  (risky)
# Without this feature, the model uses raw amount as a proxy and
# systematically gets small-amount fraud wrong.
#
# app.py computes this same formula at inference time:
#   amount_deviation = round(payment.amount / avg_txn_amount, 4)
# Both must use the same formula or predictions will be inconsistent.
# ══════════════════════════════════════════════════════════════════

data['amount_deviation'] = (data['amount'] / data['avg_txn_amount']).round(4)
data['avg_txn_amount']=data['avg_txn_amount'].replace(0,1)
data['txn_count_1h'] = data['txn_count_1h'].fillna(0)

data['time_since_last_txn'] = data['time_since_last_txn'].fillna(9999)

data['is_proxy_ip'] = data['is_proxy_ip'].fillna(0)
data=data.replace([np.inf,-np.inf],0)
print("\namount_deviation stats:")
print(f"  Fraud mean:  {data[data['label']==1]['amount_deviation'].mean():.3f}")
print(f"  Normal mean: {data[data['label']==0]['amount_deviation'].mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — LO2: Visualise Data
# ══════════════════════════════════════════════════════════════════

plt.figure(figsize=(5, 3))
data['label'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title("Fraud vs Normal Distribution")
plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
plt.tight_layout()
plt.savefig("fraud_distribution.png")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — LO2: Statistics
# ══════════════════════════════════════════════════════════════════

print("\n=== Experiment 3: Statistical Summary by Class ===")
fraud  = data[data['label'] == 1]
normal = data[data['label'] == 0]
stat_cols = ['amount', 'amount_deviation', 'avg_txn_amount',
             'failed_attempts', 'txn_count_24h', 'account_age_days']
for col in stat_cols:
    print(f"  {col:<22} Fraud mean: {fraud[col].mean():.2f}   Normal mean: {normal[col].mean():.2f}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — LO4: Classification using Random Forest
#
# FEATURE LIST RULES — READ CAREFULLY:
# 1. Every feature listed here MUST also be a key in ml_input in app.py
# 2. The dict key name in app.py MUST match exactly (case-sensitive)
# 3. amount_deviation MUST be computed the same way in both files:
#    train: data['amount'] / data['avg_txn_amount']
#    app:   payment.amount / avg_txn_amount
# ══════════════════════════════════════════════════════════════════

# FIX: amount_deviation is now included
FEATURES = [

"amount",

"avg_txn_amount",

"amount_deviation",

"txn_count_24h",

"txn_count_1h",

"time_since_last_txn",

"account_age_days",

"failed_attempts",

"location_change",

"is_international",

"is_mal_ip",

"is_proxy_ip",

"is_new_device",

"odd_time"

]

X = data[FEATURES]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining: {X_train.shape[0]} rows | Testing: {X_test.shape[0]} rows")
print(f"Fraud in train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# FIX 2: class_weight="balanced"
#
# WHY THIS FIX IS MANDATORY:
# Without it, the 12.7:1 normal:fraud ratio makes the model learn
# that account_age_days < 7 is an overwhelmingly strong fraud signal.
# This causes a Rs.25 domestic trusted transaction with a 5-day
# account to score 90% fraud — which is wrong.
#
# class_weight="balanced" makes sklearn automatically set each
# class weight = total_samples / (n_classes * class_count)
# This means fraud samples each carry 12.7x more weight than normal,
# forcing the model to learn fraud patterns from ALL features together
# rather than over-relying on the single strongest correlator.
# ══════════════════════════════════════════════════════════════════

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"   # FIX: was missing — caused 90% on safe transactions
)

model.fit(X_train, y_train)
print("\nModel training complete.")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 12 — LO6: Save model
# ══════════════════════════════════════════════════════════════════

joblib.dump(model, "fraud_model.pkl")
print("Model saved: fraud_model.pkl")

# Verify features the saved model knows — these must match app.py ml_input exactly
print(f"Model feature names: {list(model.feature_names_in_)}")

# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy:       {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Training Score: {model.score(X_train, y_train)*100:.2f}%")
print(f"Testing Score:  {model.score(X_test, y_test)*100:.2f}%")
print(f"AUC:            {roc_auc_score(y_test, y_prob):.4f}")
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
print(f"\nROC AUC: {roc_auc:.4f}")
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],"r--", label="Random (AUC=0.50)")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("roc_curve.png"); plt.show()

print("\nAll experiments complete.")