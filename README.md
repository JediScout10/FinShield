🚨 FinShield — Real-Time Payment Fraud Detection

A real-time fraud detection system that scores payments as they happen,
using a Random Forest model behind a FastAPI backend, with Firebase
Authentication and Firestore for per-user trust profiles.

🔹 Model Overview

Algorithm: Random Forest Classifier (100 estimators, class_weight="balanced")

Dataset Size: 29,347 transactions (finshield_dataset_v3_final.csv)

Fraud Ratio: ~12.75 : 1 (7.28% fraud) — corrected for class imbalance during training

Performance: AUC 0.90

Output: Normal / Review / Fraud, based on calibrated probability thresholds
(< 0.25 → Normal, 0.25–0.60 → Review, ≥ 0.60 → Fraud)

The model never sees raw IPs, device IDs, or user IDs directly. All inputs
are risk-based features computed by the backend from account history.

🔹 Input Features (Model Contract — 14 features)

The backend builds this exact feature set before calling the model:

{
  "amount": 8200,
  "avg_txn_amount": 3100,
  "amount_deviation": 2.65,
  "txn_count_24h": 5,
  "txn_count_1h": 2,
  "time_since_last_txn": 340,
  "account_age_days": 42,
  "failed_attempts": 0,
  "location_change": 0,
  "is_international": 0,
  "is_mal_ip": 1,
  "is_proxy_ip": 0,
  "is_new_device": 0,
  "odd_time": 0
}

Feature Meaning:

amount — transaction amount

avg_txn_amount — user's historical average transaction amount

amount_deviation — amount / avg_txn_amount (top-ranked feature, importance 0.2483)

txn_count_24h / txn_count_1h — transaction frequency in the last 24h / 1h

time_since_last_txn — seconds since the user's previous transaction

account_age_days — account age

failed_attempts — prior failed login attempts

location_change — 1 if the transaction's country isn't among the account's known countries

is_international — 1 if the detected country isn't India

is_mal_ip — 1 if the transaction's ISP/network isn't among the account's known networks

is_proxy_ip — 1 if is_mal_ip + is_international + location_change all fire together

is_new_device — 1 if the device fingerprint isn't among the account's known devices

odd_time — 1 if the transaction hour is before 5 AM

🔹 How Real Data Is Used

Raw signals (IP address, device fingerprint, timestamp) are never passed to
the model directly. The backend first converts them into risk features by
comparing against each account's trust profile stored in Firestore.

Trust is frequency-based, not single-last-value: each account keeps a map
of every ISP/network, device, and country it has ever transacted from on an
Approved transaction, with a seen-count for each. A network/device/country
is "known" if it appears anywhere in that history — not only if it happens
to be the single most recent one used. This avoids false positives for
users who legitimately switch between a small number of regular networks
or devices (e.g. wifi ↔ mobile data), while still flagging a genuinely new
one.

Only Approved (Normal) transactions grow this trust history. A
Review/Blocked transaction updates transaction counters only — it cannot
poison the account's trust profile even if it later turns out to be
legitimate-looking fraud.

🔹 How to Run the Backend

1. Install dependencies

pip install -r requirements.txt

2. Add Firebase credentials

Set up firebase_config.py with your Firebase service account credentials
(not committed to this repo).

3. Start the API server

python -m uvicorn app:app --reload

Server runs at: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs

🔹 API Endpoints

POST /payment
Scores a transaction in real time. Request body:

{
  "user_id": "user_123",
  "amount": 2000,
  "device_fingerprint": "abc123xyz",
  "txn_type": "PAYMENT"
}

amount must be > 0 and ≤ ₹1,00,00,000; user_id and device_fingerprint
must be non-empty; txn_type must be one of PAYMENT, TRANSFER, CASH_OUT.
Invalid input is rejected with a 422 before it reaches the model.

Response:

{
  "prediction": "Normal",
  "probability": 0.0421,
  "risk_factors": [...],
  "safe_factors": [...]
}

POST /register-user — creates a Firestore user profile after Firebase Auth signup

POST /login-success / POST /login-failed — tracks failed login attempts per account

GET / , /app , /learn , /login — serves the frontend pages

🔹 Model Training

To retrain the model:

python train_model.py

This regenerates fraud_model.pkl from finshield_dataset_v3_final.csv.
Feature list and order are defined once (FEATURES in train_model.py) and
must match the ml_input dict built in app.py exactly — see the comments
in train_model.py for this constraint.

🔹 Known Limitations (honest, not hidden)

- is_new_device is exact-match on the device fingerprint string. A device
  whose fingerprint shifts (browser/OS updates) will look "new" even
  though it's the same physical device. Proper fuzzy fingerprint
  similarity scoring is a natural next step, not yet implemented.
- Geolocation/ISP lookup uses the free ip-api.com tier — fine for a demo,
  not rate-limit or SLA suitable for production.
- No cross-account/network-graph analysis — each account's trust profile
  is evaluated independently, unlike production fraud systems that link
  signals across many accounts (e.g. shared device used by many different
  users).
- This is a demo/prototype system trained on a synthetic dataset. It
  demonstrates the shape of the fraud detection problem — imbalanced
  classification, explainability, trust-profile design — rather than
  being production-hardened.

🔹 Why This Design

- Risk features are stable and explainable; raw identifiers are
  high-cardinality and unstable — this mirrors how real fraud pipelines
  separate risk feature extraction from classification.
- Every prediction ships with human-readable risk/safe factors instead of
  a black-box score, so a decision can always be explained.
- Trust-building logic (frequency-based, Approved-only) is deliberately
  designed to resist the specific "one clean transaction whitewashes the
  account" poisoning pattern found and fixed during development.
