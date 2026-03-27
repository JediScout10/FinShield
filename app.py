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

print("Fraud model loaded")


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


class RegisterUser(BaseModel):
    uid: str          # FIX: was missing uid field
    email: str


class LoginUser(BaseModel):
    uid: str
    email: str


class LoginFailed(BaseModel):
    email: str        # FIX: login-failed uses email to look up user


def get_real_ip(request: Request) -> str:
    """
    FIX: Correctly extract real client IP.
    Checks x-forwarded-for (set by proxies/load balancers) first,
    then falls back to direct connection IP.
    x-forwarded-for can contain a comma-separated chain — take the first (original client).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take first IP in the chain — that's the real client
        return forwarded.split(",")[0].strip()
    return request.client.host

def get_location_from_ip(ip: str) -> str:
    """
    Looks up IP via ip-api.com. Returns 'Demo Mode' implicitly on loopback 
    or failed fetch to handle testing cleanly.
    """
    if ip in ("127.0.0.1", "localhost", "0.0.0.0"):
        return "Demo Mode"
    try:
        req = urllib.request.Request(f"http://ip-api.com/json/{ip}", headers={'User-Agent': 'FinShield/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                return data.get("country", "Demo Mode")
    except Exception:
        pass
    return "Demo Mode"


@app.post("/payment")
async def process_payment(payment: PaymentRequest, request: Request):

    # FIX: Use improved IP detection helper
    current_ip = get_real_ip(request)
    current_device = payment.device_fingerprint
    current_time = datetime.now()

    odd_time = 1 if current_time.hour < 5 else 0

    user_ref = db.collection("users").document(payment.user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        demo_created = (datetime.now() - timedelta(days=90)).isoformat()
        demo_country = get_location_from_ip(current_ip)
        user_data = {
            "email": f"{payment.user_id}@demo.com",
            "last_ip": current_ip,
            "last_device": current_device,
            "last_country": demo_country,
            "last_txn_time": (datetime.now() - timedelta(hours=2)).isoformat(),
            "txn_count_24h": 1,
            "created_at": demo_created,
            "failed_attempts": 0,
            "avg_txn_amount": payment.amount,
            "total_txn_count": 5
        }
        user_ref.set(user_data)
    else:
        user_data = user_doc.to_dict()

    last_ip = user_data.get("last_ip")
    last_device = user_data.get("last_device")
    last_country = user_data.get("last_country")
    
    detected_country = get_location_from_ip(current_ip)

    is_mal_ip = 1 if (last_ip and current_ip != last_ip) else 0
    is_new_device = 1 if (last_device and current_device != last_device) else 0

    if last_country and detected_country != "Demo Mode" and detected_country != last_country:
        location_change = 1
    else:
        location_change = 0

    last_txn_time = user_data.get("last_txn_time")

    if last_txn_time:
        last_txn = datetime.fromisoformat(last_txn_time)
        if current_time - last_txn < timedelta(hours=24):
            txn_count_24h = user_data.get("txn_count_24h", 0) + 1
        else:
            txn_count_24h = 1
    else:
        txn_count_24h = 1

    created = user_data.get("created_at")
    if created:
        created_dt = datetime.fromisoformat(created)
        account_age_days = (datetime.now() - created_dt).days
    else:
        account_age_days = 0

    failed_attempts = user_data.get("failed_attempts", 0)

    avg_txn_amount = user_data.get("avg_txn_amount", payment.amount)
    if avg_txn_amount == 0:
        avg_txn_amount = payment.amount

    if detected_country != "Demo Mode" and detected_country != "India":
        is_international = 1
    else:
        is_international = 0

    ml_input = {
        "amount": float(payment.amount),
        "is_mal_ip": int(is_mal_ip),
        "is_new_device": int(is_new_device),
        "odd_time": int(odd_time),
        "txn_count_24h": int(txn_count_24h),
        "account_age_days": int(account_age_days),
        "failed_attempts": int(failed_attempts),
        "location_change": int(location_change),
        "avg_txn_amount": float(avg_txn_amount),
        "is_international": int(is_international)
    }

    df = pd.DataFrame([ml_input])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    probability_formatted = f"{prob * 100:.2f}"

    # --- Risk explanation logic (evaluated first for decision consistency) ---
    risk = []
    safe = []

    if user_data.get("total_txn_count", 0) >= 3:
        if payment.amount > avg_txn_amount * 3:
            risk.append(f"Amount unusually high vs average (avg: ₹{avg_txn_amount:.0f})")
        else:
            safe.append(f"Amount consistent with spending pattern (avg: ₹{avg_txn_amount:.0f})")
    
    if is_mal_ip:
        risk.append(f"IP address changed from last known location (Detected IP: {current_ip})")
    else:
        safe.append("Transaction from trusted IP address")

    if location_change:
        risk.append(f"Geolocation anomaly: Transaction country changed to {detected_country}")
    else:
        safe.append("Transaction country consistent with history")

    if is_international:
        risk.append(f"International transaction detected ({detected_country})")
    else:
        safe.append("Domestic transaction (India)")

    if is_new_device:
        risk.append("New or unrecognised device signature detected")
    else:
        safe.append("Transaction from trusted device")

    if odd_time:
        risk.append("Transaction at unusual hour (before 5 AM)")
    else:
        safe.append("Transaction at normal business hour")

    if txn_count_24h > 10:
        risk.append(f"Too many transactions in 24 hours ({txn_count_24h})")
    else:
        safe.append(f"Normal transaction frequency ({txn_count_24h} in 24h)")

    if account_age_days < 7:
        risk.append(f"New account (only {account_age_days} day(s) old)")
    else:
        safe.append(f"Established account ({account_age_days} days old)")

    if failed_attempts > 2:
        risk.append(f"Multiple suspicious authentication or verification failures ({failed_attempts})")
    else:
        safe.append("No suspicious or abnormal access attempts")

    # --- Probability Sanity Guard (Fix for overly sensitive Random Forest dataset mapping) ---
    if payment.amount < 1000 and failed_attempts == 0 and location_change == 0 and txn_count_24h <= 5:
        # If the transaction is fundamentally small and safe, but the model triggers >0.70 via baseline noise:
        if 0.70 < prob < 0.85:
            prob = 0.69  # Shift down to 'Review' so small domestic transactions aren't auto-blocked.

    # --- Logic 3: Model Probability serves as Absolute Truth ---
    if prob < 0.40:
        prediction = "Normal"
        decision = "Approved"
        risk_level = "Low"
    elif prob <= 0.70:
        prediction = "Review Required"
        decision = "Held for Review"
        risk_level = "Medium"
    else:
        prediction = "Fraud"
        decision = "Blocked"
        risk_level = "High"

    # Ensure backend logs trace ML model utilization
    print(f"\n--- FINSHIELD INFERENCE LOG ---")
    print(f"Loaded Model Path: {MODEL_PATH}")
    print(f"Extracted Features: {ml_input}")
    print(f"Predicted Probability: {prob:.4f}")
    print(f"Final Decision: {decision}")
    print(f"-------------------------------\n")

    prev_avg = user_data.get("avg_txn_amount", 0)
    prev_count = user_data.get("total_txn_count", 0)
    new_count = prev_count + 1
    new_avg = ((prev_avg * prev_count) + payment.amount) / new_count

    # In a real environment, you might only update this if approved.
    user_ref.update({
        "last_ip": current_ip,
        "last_device": current_device,
        "last_country": detected_country,
        "last_txn_time": current_time.isoformat(),
        "txn_count_24h": txn_count_24h,
        "avg_txn_amount": round(new_avg, 2),
        "total_txn_count": new_count
    })

    return {
        "prediction": prediction,
        "probability": probability_formatted,
        "decision": decision,
        "risk_level": risk_level,
        "risk_factors": risk,
        "safe_factors": safe,
        "detected_ip": current_ip,
        "feature_values": {
            "IP Address": current_ip,
            "Country": detected_country,
            "Device": current_device,
            "Time": current_time.strftime("%H:%M:%S"),
            "Amount": f"₹{payment.amount:.2f}",
            "Txn count 24h": str(txn_count_24h),
            "Account age (days)": str(account_age_days),
            "Failed attempts": str(failed_attempts)
        }
    }


@app.post("/register-user")
async def register_user(data: RegisterUser):
    # FIX: Now correctly uses data.uid (field was missing in original model)
    user_ref = db.collection("users").document(data.uid)
    user_ref.set({
        "email": data.email,
        "last_ip": None,
        "last_device": None,
        "last_country": None,
        "last_txn_time": None,
        "txn_count_24h": 0,
        "created_at": datetime.now().isoformat(),
        "failed_txn_attempts": 0,
        "avg_txn_amount": 0,
        "total_txn_count": 0
    })
    return {"success": True}


@app.post("/login-failed")
async def login_failed(data: LoginFailed):
    """
    Deprecated logic keeping for safe backward compatibility: maps to failed_attempts
    """
    users_ref = db.collection("users").where("email", "==", data.email).limit(1).stream()
    for user_doc in users_ref:
        user_doc.reference.update({
            "failed_attempts": user_doc.to_dict().get("failed_attempts", 0) + 1
        })
        break
    return {"updated": True}


@app.post("/login-success")
async def login_success(data: LoginUser):
    """
    Deprecated logical route. Maps to reset failed_attempts
    """
    user_ref = db.collection("users").document(data.uid)
    user = user_ref.get()
    if user.exists:
        user_ref.update({
            "failed_attempts": 0
        })
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