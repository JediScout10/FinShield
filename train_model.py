import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import seaborn as sns


# 1 Load dataset (Experiment 1 – Data preprocessing)

data = pd.read_csv("fraud_transactions_10col.csv")


# 2 Select EXACT features

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

y = data["label"]


# 3 Train test split

X_train,X_test,y_train,y_test = train_test_split(

X,
y,
test_size=0.2,
random_state=42

)


# 4 Train Random Forest (Experiment 6)

model = RandomForestClassifier(

n_estimators=100,
random_state=42

)

model.fit(X_train,y_train)


# 5 Save model

joblib.dump(model,"fraud_model.pkl")

print("Model saved")


# 6 Predictions

y_pred = model.predict(X_test)


# 7 Evaluation metrics

print("\nAccuracy:",accuracy_score(y_test,y_pred))

print("\nTraining Score:",model.score(X_train,y_train))

print("Testing Score:",model.score(X_test,y_test))


print("\nConfusion Matrix:")

print(confusion_matrix(y_test,y_pred))


print("\nClassification Report:")

print(classification_report(y_test,y_pred))


# 8 Confusion matrix visualization (Experiment 2)

plt.figure(figsize=(5,4))

sns.heatmap(

confusion_matrix(y_test,y_pred),

annot=True,

fmt="d",

cmap="Blues"

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



# 9 Feature importance visualization

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

plt.barh(features,importance)

plt.title("Feature Importance")

plt.xlabel("Importance Score")

plt.show()



# 10 ROC Curve (extra improvement)

y_prob = model.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test,y_prob)

roc_auc = auc(fpr,tpr)


plt.figure(figsize=(6,5))

plt.plot(fpr,tpr,label="ROC curve (area = %0.2f)" % roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.show()