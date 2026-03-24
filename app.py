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
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    last_ip = user_data.get("last_ip")
    last_device = user_data.get("last_device")

    # FIX: IP change is a risk signal but only suspicious if there WAS a known IP before
    is_mal_ip = 1 if (last_ip and current_ip != last_ip) else 0
    is_mal_device = 1 if (last_device and current_device != last_device) else 0

    location_change = is_mal_ip

    last_txn_time = user_data.get("last_txn_time")

    if last_txn_time:
        last_txn = datetime.fromisoformat(last_txn_time)
        if current_time - last_txn < timedelta(hours=24):
            txn_count_24h = user_data.get("txn_count_24h", 0) + 1
        else:
            txn_count_24h = 1
    else:
        txn_count_24h = 1

    # FIX: Real account age in days from created_at timestamp
    created = user_data.get("created_at")
    if created:
        created_dt = datetime.fromisoformat(created)
        user_age_days = (datetime.now() - created_dt).days
    else:
        user_age_days = 0

    failed_attempts = user_data.get("failed_attempts", 0)

    avg_txn_amount = user_data.get("avg_txn_amount", payment.amount)
    if avg_txn_amount == 0:
        avg_txn_amount = payment.amount

    is_international = 0

    ml_input = {
        "amount": payment.amount,
        "is_mal_ip": is_mal_ip,
        "is_mal_device": is_mal_device,
        "odd_time": odd_time,
        "txn_count_24h": txn_count_24h,
        "user_age_days": user_age_days,
        "failed_attempts": failed_attempts,
        "location_change": location_change,
        "avg_txn_amount": avg_txn_amount,
        "is_international": is_international
    }

    df = pd.DataFrame([ml_input])
    pred = model.predict(df)[0]
    prediction = "Fraud" if pred == 1 else "Normal"

    # --- Risk explanation logic ---
    risk = []
    safe = []

    if payment.amount > 8000:
        risk.append("High transaction amount (above ₹8,000)")
    else:
        safe.append("Normal transaction amount")

    if is_mal_ip:
        risk.append("IP address changed from last known location")
    else:
        safe.append("Transaction from trusted IP address")

    if is_mal_device:
        risk.append("New or unrecognised device used")
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

    if user_age_days < 7:
        risk.append(f"New account (only {user_age_days} day(s) old)")
    else:
        safe.append(f"Established account ({user_age_days} days old)")

    if failed_attempts > 2:
        risk.append(f"Multiple failed login attempts ({failed_attempts})")
    else:
        safe.append("No suspicious login history")

    if payment.amount > avg_txn_amount * 3:
        risk.append(f"Amount unusually high vs average (avg: ₹{avg_txn_amount:.0f})")
    else:
        safe.append(f"Amount consistent with spending pattern (avg: ₹{avg_txn_amount:.0f})")

    risk_score = len(risk) * 10

    # FIX: Update avg_txn_amount as running average, not just last value
    prev_avg = user_data.get("avg_txn_amount", 0)
    prev_count = user_data.get("total_txn_count", 0)
    new_count = prev_count + 1
    new_avg = ((prev_avg * prev_count) + payment.amount) / new_count

    user_ref.update({
        "last_ip": current_ip,
        "last_device": current_device,
        "last_txn_time": current_time.isoformat(),
        "txn_count_24h": txn_count_24h,
        "avg_txn_amount": round(new_avg, 2),
        "total_txn_count": new_count
    })

    return {
        "prediction": prediction,
        "allowed": prediction == "Normal",
        "risk_score": risk_score,
        "risk_factors": risk,
        "safe_factors": safe,
        "detected_data": {
            "IP Address": current_ip,
            "Device": current_device,
            "Time": current_time.strftime("%H:%M:%S"),
            "Txn count 24h": txn_count_24h,
            "Account age (days)": user_age_days,
            "Failed logins": failed_attempts
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
        "last_txn_time": None,
        "txn_count_24h": 0,
        "created_at": datetime.now().isoformat(),
        "failed_attempts": 0,
        "avg_txn_amount": 0,
        "total_txn_count": 0
    })
    return {"success": True}


@app.post("/login-failed")
async def login_failed(data: LoginFailed):
    """
    FIX: Original code used LoginUser model (missing) and looked up by uid.
    Now correctly looks up by email since uid is not known at login-failed time.
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
    FIX: Reset failed_attempts on successful login to prevent stale risk signals.
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