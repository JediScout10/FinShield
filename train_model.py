import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

<<<<<<< HEAD
print("="*50)
print("FINSHIELD ML WORKFLOW STARTING")
print("="*50)
=======
# ============================================================
# EXPERIMENT 1 — LO1
# Data preparation using NumPy and Pandas
# We load the dataset using Pandas (pd.read_csv) and select
# only the relevant feature columns required by our ML model.
# This step covers data loading, inspection, and feature selection.
# ============================================================
>>>>>>> 6c437711b3953028c85aaea874a0e4a194300f8f

# ==========================================
# EXPERIMENT 1: Data Preprocessing
# ==========================================
print("\n[Experiment 1] Data Preprocessing")
# Use the correct dataset
data = pd.read_csv("finshield_dataset_v3_final.csv")

<<<<<<< HEAD
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
=======
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nNull values:")
print(data.isnull().sum())


# ============================================================
# EXPERIMENT 1 (continued) — LO1
# Feature selection: selecting the exact 10 features that
# our fraud detection model expects as input.
# 'label' column is the target variable (0 = Normal, 1 = Fraud)
# ============================================================

X = data[[
    "amount",
    "is_mal_ip",
    "is_mal_device",
    "odd_time",
    "txn_count_24h",
    "user_age_days",
    "failed_attempts",
    "location_change",
    "avg_txn_amount",
    "is_international"
]]
>>>>>>> 6c437711b3953028c85aaea874a0e4a194300f8f

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

<<<<<<< HEAD
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n[Pipeline] Training Random Forest Ensemble...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
=======
print("\nFeatures shape:", X.shape)
print("Label distribution:\n", y.value_counts())


# ============================================================
# EXPERIMENT 6 — LO4
# Classification using Random Forest algorithm
# We split data into 80% training and 20% testing sets.
# random_state=42 ensures reproducibility — same split every run.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# EXPERIMENT 6 (continued) — LO4
# Training the Random Forest Classifier
# n_estimators=100 means 100 decision trees are built.
# The final prediction is a majority vote across all 100 trees.
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training complete.")


# ============================================================
# EXPERIMENT 12 — LO6 (Mini Project)
# Saving the trained model to disk using joblib.
# This saved .pkl file is loaded by the FastAPI backend (app.py)
# to make real-time fraud predictions on new transactions.
# ============================================================

joblib.dump(model, "fraud_model.pkl")

print("Model saved as fraud_model.pkl")


# ============================================================
# EXPERIMENT 6 (continued) — LO4
# Making predictions on the test set
# predict()      → returns class labels (0 or 1)
# predict_proba()→ returns probability scores (used for ROC curve)
# ============================================================
>>>>>>> 6c437711b3953028c85aaea874a0e4a194300f8f

# ==========================================
# MODEL EVALUATION
# ==========================================
print("\n[Pipeline] Evaluating Model Performance...")
y_pred = model.predict(X_test)

<<<<<<< HEAD
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
=======

# ============================================================
# EXPERIMENT 3 — LO2
# Data modeling using basics of statistics
# Accuracy, precision, recall, and F1-score are all statistical
# metrics used to evaluate classification model performance.
# Training score vs Testing score comparison reveals overfitting.
# ============================================================

print("\n--- Model Evaluation (Experiment 3 & 6) ---")

print("\nAccuracy         :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Training Score   :", round(model.score(X_train, y_train) * 100, 2), "%")
print("Testing Score    :", round(model.score(X_test, y_test) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Confusion matrix explanation:
# [ TN  FP ]   TN = correctly predicted Normal
# [ FN  TP ]   TP = correctly predicted Fraud
#              FP = Normal predicted as Fraud (false alarm)
#              FN = Fraud predicted as Normal (missed fraud — dangerous)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))


# ============================================================
# EXPERIMENT 2 — LO2
# Visualise data using Matplotlib and Seaborn
# Seaborn heatmap: visually shows how many predictions were
# correct (diagonal) vs incorrect (off-diagonal).
# ============================================================

plt.figure(figsize=(5, 4))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title("Confusion Matrix — Experiment 2 (Seaborn Visualization)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()


# ============================================================
# EXPERIMENT 2 (continued) — LO2
# Feature Importance Bar Chart using Matplotlib
# Shows which of the 10 features contributed most to the
# Random Forest model's decision-making.
# Higher importance = more influential in predicting fraud.
# ============================================================

importance = model.feature_importances_
features = X.columns

# Sort by importance for cleaner visualization
sorted_idx = importance.argsort()

plt.figure(figsize=(8, 5))
plt.barh(features[sorted_idx], importance[sorted_idx], color="steelblue")
plt.title("Feature Importance — Experiment 2 (Matplotlib Visualization)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ============================================================
# EXPERIMENT 2 (continued) — LO2
# ROC Curve (Receiver Operating Characteristic)
# Plots True Positive Rate vs False Positive Rate at all thresholds.
# AUC (Area Under Curve): 1.0 = perfect, 0.5 = random guessing.
# A good fraud model should have AUC > 0.90.
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]   # probability of fraud (class=1)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", label="ROC Curve (AUC = %0.4f)" % roc_auc)
plt.plot([0, 1], [0, 1], "r--", label="Random Classifier (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Experiment 2 (Matplotlib Visualization)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print("\nROC AUC Score:", round(roc_auc, 4))
print("\nAll experiments complete.")
>>>>>>> 6c437711b3953028c85aaea874a0e4a194300f8f
