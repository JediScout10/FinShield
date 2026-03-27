import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

print("="*50)
print("FINSHIELD ML WORKFLOW STARTING")
print("="*50)

# ==========================================
# EXPERIMENT 1: Data Preprocessing
# ==========================================
print("\n[Experiment 1] Data Preprocessing")
# Use the correct dataset
data = pd.read_csv("finshield_dataset_v3_final.csv")

print("\n--- Dataset Summary ---")
print(data.info())
print("\n--- Dataset Statistics ---")
print(data.describe())
print("\n--- Null Checks ---")
print(data.isnull().sum())

# ==========================================
# EXPERIMENT 2: Visualization
# ==========================================
print("\n[Experiment 2] Visualization - Saving plots...")

# Fraud Distribution plot
plt.figure(figsize=(6,4))
sns.countplot(data=data, x='label')
plt.title("Distribution of Normal (0) vs Fraud (1) Transactions")
plt.savefig("fraud_distribution.png")
plt.close()

# Use correct exact 10 features as requested
features = [
    "amount", 
    "is_mal_ip", 
    "is_new_device", 
    "odd_time", 
    "txn_count_24h", 
    "account_age_days", 
    "failed_attempts", 
    "location_change", 
    "avg_txn_amount", 
    "is_international"
]

# Correlation Heatmap
corr_data = data[features + ['label']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# ==========================================
# EXPERIMENT 3: Statistical Analysis
# ==========================================
print("\n[Experiment 3] Statistical Analysis grouped by Fraud vs Normal")
stats = data.groupby('label')[features].agg(['mean', 'std'])
print("\n--- Mean & Std by Fraud Label ---")
print(stats)

# ==========================================
# MODEL TRAINING PIPELINE
# ==========================================
print("\n[Pipeline] Preparing Model Training...")

# Select the Features
# Why RandomForest? It creates an ensemble of decision trees, handles non-linear relationships well, 
# naturally outputs feature importance, and provides probability scores necessary for our risk explanation engine.
X = data[features]
y = data["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n[Pipeline] Training Random Forest Ensemble...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# MODEL EVALUATION
# ==========================================
print("\n[Pipeline] Evaluating Model Performance...")
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Training Score:", model.score(X_train, y_train))
print("Testing Score:", model.score(X_test, y_test))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

# Feature Importance Visualization
importance = model.feature_importances_
plt.figure(figsize=(8,5))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.savefig("feature_importance.png")
plt.close()

# ROC Curve Evaluation
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# ==========================================
# MODEL SAVING
# ==========================================
print("\n[Pipeline] Saving Updated Model...")
joblib.dump(model, "fraud_model.pkl")
print("Model saved successfully as 'fraud_model.pkl'.")

print("="*50)
print("FINSHIELD ML WORKFLOW COMPLETED")
print("="*50)