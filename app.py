from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

from datetime import datetime, timedelta
from pydantic import BaseModel

import firebase_config
import os
import urllib.request
import json


app = FastAPI()

MODEL_PATH = os.path.join(os.getcwd(), "fraud_model.pkl")
model = joblib.load(MODEL_PATH)
print(f"Fraud model loaded. Features: {list(model.feature_names_in_)}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

config = firebase_config.FIREBASE_CONFIG
cred = credentials.Certificate(config)
firebase_admin.initialize_app(cred)
db = firestore.client()
print("Firebase ready")


class PaymentRequest(BaseModel):
    user_id: str
    amount: float
    device_fingerprint: str
    txn_type: str = "PAYMENT"


class RegisterUser(BaseModel):
    uid: str
    email: str


class LoginUser(BaseModel):
    uid: str
    email: str


class LoginFailed(BaseModel):
    email: str


def get_real_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


def get_location_from_ip(ip: str) -> str:
    if ip in ("127.0.0.1", "localhost", "0.0.0.0"):
        return "Demo Mode"
    try:
        req = urllib.request.Request(
            f"http://ip-api.com/json/{ip}",
            headers={"User-Agent": "FinShield/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                return data.get("country", "Demo Mode")
    except Exception:
        pass
    return "Demo Mode"


def generate_explanations(
    amount: float,
    avg_txn_amount: float,
    amount_deviation: float,
    is_mal_ip: int,
    is_new_device: int,
    location_change: int,
    is_international: int,
    odd_time: int,
    txn_count_24h: int,
    account_age_days: int,
    failed_attempts: int,
    detected_country: str,
    current_ip: str,
    prob: float,
    txn_type: str = "PAYMENT"
) -> tuple[list, list]:

    risk = []
    safe = []

    # ── 1. Amount Deviation (top feature: importance 0.2483) ──
    if amount_deviation >= 3.0:
        risk.append(
            f"Extreme spending spike: ₹{amount:,.0f} is {amount_deviation:.1f}× "
            f"your historical average of ₹{avg_txn_amount:,.0f}"
        )
    elif amount_deviation >= 2.0:
        risk.append(
            f"Amount significantly above your normal pattern: "
            f"₹{amount:,.0f} is {amount_deviation:.1f}× your average (₹{avg_txn_amount:,.0f})"
        )
    elif amount_deviation >= 1.5:
        risk.append(
            f"Amount moderately elevated: ₹{amount:,.0f} is "
            f"{amount_deviation:.1f}× above your average of ₹{avg_txn_amount:,.0f}"
        )
    else:
        safe.append(
            f"Amount consistent with spending history "
            f"(₹{amount:,.0f} — {amount_deviation:.2f}× avg ₹{avg_txn_amount:,.0f})"
        )

    # ── 2. Raw Amount ──
    if amount >= 50000:
        risk.append(f"Very high transaction: ₹{amount:,.0f} exceeds Rs.50,000 alert threshold")
    elif amount >= 20000:
        risk.append(f"High transaction amount: ₹{amount:,.0f}")
    elif amount < 500:
        safe.append(f"Low-value transaction (₹{amount:,.0f}) — minimal financial exposure")
    else:
        safe.append(f"Transaction amount within normal range: ₹{amount:,.0f}")

    # ── 3. Failed Attempts ──
    if failed_attempts >= 4:
        risk.append(
            f"Critical: {failed_attempts} failed login attempts — "
            f"possible brute-force or account takeover"
        )
    elif failed_attempts >= 2:
        risk.append(
            f"Multiple authentication failures: {failed_attempts} failed attempts"
        )
    elif failed_attempts == 1:
        risk.append("One prior failed login attempt on this account")
    else:
        safe.append("No failed authentication attempts — clean login history")

    # ── 4. Malicious IP ──
    if is_mal_ip:
        risk.append(
            f"IP address changed from last known location — "
            f"current IP ({current_ip}) does not match account history"
        )
    else:
        safe.append(f"Transaction from trusted IP address ({current_ip})")

    # ── 5. Account Age ──
    if account_age_days < 7:
        risk.append(
            f"Very new account: {account_age_days} day(s) old — "
            f"highest risk window for account fraud"
        )
    elif account_age_days < 30:
        risk.append(f"New account: {account_age_days} days old — under 30-day elevated risk period")
    elif account_age_days >= 180:
        safe.append(f"Well-established account: {account_age_days} days old")
    else:
        safe.append(f"Account age {account_age_days} days — past the new-account risk window")

    # ── 6. Device Change ──
    if is_new_device:
        risk.append("Unrecognised device: not previously associated with this account")
    else:
        safe.append("Transaction from a previously trusted device")

    # ── 7. Location Change ──
    if location_change:
        risk.append(
            f"Geographic anomaly: transaction country changed to {detected_country} — "
            f"differs from this account's established location"
        )
    else:
        safe.append(
            "Transaction country consistent with account history"
            + (f" ({detected_country})" if detected_country not in ("Demo Mode", "") else "")
        )

    # ── 8. International ──
    if is_international:
        risk.append(
            f"International transaction: {detected_country} — "
            f"cross-border payments have 4.8× higher fraud rate"
        )
    else:
        safe.append("Domestic transaction — lower cross-border fraud risk")

    # ── 9. Transaction Frequency ──
    if txn_count_24h > 15:
        risk.append(f"Very high frequency: {txn_count_24h} transactions in 24 hours")
    elif txn_count_24h > 10:
        risk.append(f"Elevated frequency: {txn_count_24h} transactions in 24 hours")
    else:
        safe.append(f"Normal transaction frequency: {txn_count_24h} transaction(s) in 24h")

    # ── 10. Transaction Type ──
    if txn_type == "TRANSFER":
        risk.append("High-risk transaction type: TRANSFER has 2.7× higher fraud rate than payments")
    elif txn_type == "CASH_OUT":
        risk.append("High-risk transaction type: CASH_OUT has elevated fraud rate")
    else:
        safe.append(f"Standard transaction type: {txn_type}")

    # ── 11. Odd Time ──
    if odd_time:
        risk.append("Transaction before 5 AM — late-night transactions have elevated fraud correlation")
    else:
        safe.append("Transaction during normal business hours")

    # ── Black Box Guarantee ──
    # If model scores > 50% but no single rule fired, surface the combination
    if prob > 0.50 and len(risk) == 0:
        risk.append(
            f"Combined risk profile: {prob*100:.0f}% fraud probability from "
            f"multiple sub-threshold signals aligned together"
        )
        if amount_deviation >= 1.2:
            risk.append(
                f"Contributing: amount deviation of {amount_deviation:.2f}× "
                f"combined with account profile elevated risk score"
            )
        elif failed_attempts >= 1:
            risk.append(
                f"Contributing: {failed_attempts} prior login failure(s) "
                f"in combination with transaction context"
            )

    return risk, safe


@app.post("/payment")
async def process_payment(payment: PaymentRequest, request: Request):

    current_ip     = get_real_ip(request)
    current_device = payment.device_fingerprint
    current_time   = datetime.now()
    odd_time = 1 if current_time.hour < 5 else 0

    user_ref = db.collection("users").document(payment.user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        demo_country = get_location_from_ip(current_ip)
        demo_created = (datetime.now() - timedelta(days=90)).isoformat()
        user_data = {
            "email":           f"{payment.user_id}@demo.com",
            "last_ip":         current_ip,
            "last_device":     current_device,
            "last_country":    demo_country,
            "last_txn_time":   (datetime.now() - timedelta(hours=2)).isoformat(),
            "txn_count_24h":   1,
            "created_at":      demo_created,
            "failed_attempts": 0,
            "avg_txn_amount":  payment.amount,
            "total_txn_count": 5
        }
        user_ref.set(user_data)
    else:
        user_data = user_doc.to_dict()

    last_ip      = user_data.get("last_ip")
    last_device  = user_data.get("last_device")
    last_country = user_data.get("last_country")

    detected_country = get_location_from_ip(current_ip)

    is_mal_ip     = 1 if (last_ip     and current_ip     != last_ip)     else 0
    is_new_device = 1 if (last_device and current_device != last_device) else 0
    location_change = (
        1 if (last_country
              and detected_country not in ("Demo Mode",)
              and detected_country != last_country)
        else 0
    )

    last_txn_time = user_data.get("last_txn_time")
    if last_txn_time:
        last_txn = datetime.fromisoformat(last_txn_time)
        txn_count_24h = (
            user_data.get("txn_count_24h", 0) + 1
            if current_time - last_txn < timedelta(hours=24) else 1
        )
    else:
        txn_count_24h = 1

    created = user_data.get("created_at")
    account_age_days = (
        (datetime.now() - datetime.fromisoformat(created)).days if created else 0
    )

    failed_attempts = user_data.get("failed_attempts", 0)
    avg_txn_amount  = user_data.get("avg_txn_amount", payment.amount) or payment.amount

    # FIX: Compute amount_deviation — must use identical formula as train_model.py
    # train: data['amount'] / data['avg_txn_amount']
    # here:  payment.amount / avg_txn_amount
    amount_deviation = round(payment.amount / avg_txn_amount, 4)

    is_international = 1 if detected_country not in ("Demo Mode", "India") else 0

    # ══════════════════════════════════════════════════════════════
    # FIX: ml_input now includes amount_deviation to match FEATURES
    # in train_model.py exactly. Key names and order must match.
    # The model uses feature_names_in_ to validate — any mismatch
    # causes a hard sklearn ValueError at prediction time.
    # ══════════════════════════════════════════════════════════════
    ml_input = {
        "amount":           float(payment.amount),
        "amount_deviation": float(amount_deviation),  # FIX: was missing
        "is_mal_ip":        int(is_mal_ip),
        "is_new_device":    int(is_new_device),
        "odd_time":         int(odd_time),
        "txn_count_24h":    int(txn_count_24h),
        "account_age_days": int(account_age_days),
        "failed_attempts":  int(failed_attempts),
        "location_change":  int(location_change),
        "avg_txn_amount":   float(avg_txn_amount),
        "is_international": int(is_international)
    }

    df   = pd.DataFrame([ml_input])
    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[0][1])
    probability_formatted = f"{prob * 100:.2f}"

    # Sanity guard: suppress model noise for clearly minimal-risk transactions
    if (payment.amount < 1000
            and failed_attempts == 0
            and location_change == 0
            and txn_count_24h <= 5
            and is_new_device == 0
            and is_mal_ip == 0
            and amount_deviation < 1.5):
        if 0.70 < prob < 0.85:
            prob = 0.69

    if prob < 0.40:
        prediction = "Normal"
        decision   = "Approved"
        risk_level = "Low"
    elif prob <= 0.70:
        prediction = "Review Required"
        decision   = "Held for Review"
        risk_level = "Medium"
    else:
        prediction = "Fraud"
        decision   = "Blocked"
        risk_level = "High"

    risk, safe = generate_explanations(
        amount=payment.amount,
        avg_txn_amount=avg_txn_amount,
        amount_deviation=amount_deviation,
        is_mal_ip=is_mal_ip,
        is_new_device=is_new_device,
        location_change=location_change,
        is_international=is_international,
        odd_time=odd_time,
        txn_count_24h=txn_count_24h,
        account_age_days=account_age_days,
        failed_attempts=failed_attempts,
        detected_country=detected_country,
        current_ip=current_ip,
        prob=prob,
        txn_type=payment.txn_type
    )

    print(f"\n--- FINSHIELD INFERENCE LOG ---")
    print(f"Features:     {ml_input}")
    print(f"Probability:  {prob:.4f}  Decision: {decision}")
    print(f"Risk: {len(risk)}  Safe: {len(safe)}")
    print(f"-------------------------------\n")

    prev_avg   = user_data.get("avg_txn_amount", 0) or 0
    prev_count = user_data.get("total_txn_count", 0) or 0
    new_count  = prev_count + 1
    new_avg    = ((prev_avg * prev_count) + payment.amount) / new_count

    user_ref.update({
        "last_ip":         current_ip,
        "last_device":     current_device,
        "last_country":    detected_country,
        "last_txn_time":   current_time.isoformat(),
        "txn_count_24h":   txn_count_24h,
        "avg_txn_amount":  round(new_avg, 2),
        "total_txn_count": new_count
    })

    return {
        "prediction":  prediction,
        "probability": probability_formatted,
        "decision":    decision,
        "risk_level":  risk_level,
        "risk_factors": risk,
        "safe_factors": safe,
        "detected_ip":  current_ip,
        "feature_values": {
            "IP Address":         current_ip,
            "Country":            detected_country,
            "Device":             current_device,
            "Time":               current_time.strftime("%H:%M:%S"),
            "Amount":             f"₹{payment.amount:.2f}",
            "Amount Deviation":   f"{amount_deviation:.2f}×",
            "Txn count 24h":      str(txn_count_24h),
            "Account age (days)": str(account_age_days),
            "Failed attempts":    str(failed_attempts)
        }
    }


@app.post("/register-user")
async def register_user(data: RegisterUser):
    user_ref = db.collection("users").document(data.uid)
    user_ref.set({
        "email":           data.email,
        "last_ip":         None,
        "last_device":     None,
        "last_country":    None,
        "last_txn_time":   None,
        "txn_count_24h":   0,
        "created_at":      datetime.now().isoformat(),
        "failed_attempts": 0,
        "avg_txn_amount":  0,
        "total_txn_count": 0
    })
    return {"success": True}


@app.post("/login-failed")
async def login_failed(data: LoginFailed):
    users_ref = db.collection("users").where("email", "==", data.email).limit(1).stream()
    for user_doc in users_ref:
        user_doc.reference.update({
            "failed_attempts": user_doc.to_dict().get("failed_attempts", 0) + 1
        })
        break
    return {"updated": True}


@app.post("/login-success")
async def login_success(data: LoginUser):
    user_ref = db.collection("users").document(data.uid)
    user = user_ref.get()
    if user.exists:
        user_ref.update({"failed_attempts": 0})
    return {"success": True}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def landing():
    return FileResponse(os.path.join(BASE_DIR, "static", "landing.html"))

@app.get("/app")
async def app_page():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.get("/learn")
async def learn():
    return FileResponse(os.path.join(BASE_DIR, "static", "learn-more.html"))

@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(BASE_DIR, "static", "login.html"))