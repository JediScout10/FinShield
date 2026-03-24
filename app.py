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
import requests


app = FastAPI()


MODEL_PATH=os.path.join(os.getcwd(),"fraud_model.pkl")

model=joblib.load(MODEL_PATH)

print("Fraud model loaded")


app.add_middleware(

CORSMiddleware,

allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"]

)


config=firebase_config.FIREBASE_CONFIG

cred=credentials.Certificate(config)

firebase_admin.initialize_app(cred)

db=firestore.client()

print("Firebase ready")



class PaymentRequest(BaseModel):

    user_id:str

    amount:float

    device_fingerprint:str



class RegisterUser(BaseModel):

    uid:str

    email:str



@app.post("/payment")

async def process_payment(payment:PaymentRequest,request:Request):


    try:

        current_ip=requests.get(
            "https://api64.ipify.org?format=json",
            timeout=2
        ).json()["ip"]

    except:

        current_ip=request.client.host


    current_device=payment.device_fingerprint

    current_time=datetime.now()

    odd_time=1 if current_time.hour<5 else 0


    user_ref=db.collection("users").document(payment.user_id)

    user_doc=user_ref.get()

    if not user_doc.exists:

        raise HTTPException(404,"User not found")


    user_data=user_doc.to_dict()


    last_ip=user_data.get("last_ip")

    last_device=user_data.get("last_device")


    is_mal_ip=1 if current_ip!=last_ip and last_ip else 0

    is_mal_device=1 if current_device!=last_device and last_device else 0

    location_change=1 if current_ip!=last_ip and last_ip else 0


    last_txn_time=user_data.get("last_txn_time")

    if last_txn_time:

        last_txn=datetime.fromisoformat(last_txn_time)

        if current_time-last_txn < timedelta(hours=24):

            txn_count_24h=user_data.get("txn_count_24h",0)+1

        else:

            txn_count_24h=1

    else:

        txn_count_24h=1


    user_age_days=user_data.get("account_age_days",30)

    failed_attempts=user_data.get("failed_attempts",0)

    avg_txn_amount=user_data.get("avg_txn_amount",payment.amount)

    is_international=0


    ml_input={

        "amount":payment.amount,

        "is_mal_ip":is_mal_ip,

        "is_mal_device":is_mal_device,

        "odd_time":odd_time,

        "txn_count_24h":txn_count_24h,

        "user_age_days":user_age_days,

        "failed_attempts":failed_attempts,

        "location_change":location_change,

        "avg_txn_amount":avg_txn_amount,

        "is_international":is_international

    }


    df=pd.DataFrame([ml_input])

    pred=model.predict(df)[0]

    prediction="Fraud" if pred==1 else "Normal"


    risk=[]
    safe=[]


    if payment.amount>8000:
        risk.append("High transaction amount")
    else:
        safe.append("Normal transaction amount")


    if is_mal_ip:
        risk.append("IP changed")
    else:
        safe.append("Trusted IP")


    if is_mal_device:
        risk.append("New device")
    else:
        safe.append("Trusted device")


    if odd_time:
        risk.append("Unusual time")
    else:
        safe.append("Normal time")


    if txn_count_24h>10:
        risk.append("Too many transactions")
    else:
        safe.append("Normal frequency")


    if user_age_days<7:
        risk.append("New account")
    else:
        safe.append("Established account")


    if failed_attempts>2:
        risk.append("Failed logins")
    else:
        safe.append("Clean login history")


    if payment.amount>avg_txn_amount*3:
        risk.append("Unusual amount")
    else:
        safe.append("Normal pattern")


    risk_score=len(risk)*10


    user_ref.update({

        "last_ip":current_ip,

        "last_device":current_device,

        "last_txn_time":current_time.isoformat(),

        "txn_count_24h":txn_count_24h,

        "avg_txn_amount":payment.amount

    })


    return{

        "prediction":prediction,

        "allowed":prediction=="Normal",

        "risk_score":risk_score,

        "risk_factors":risk,

        "safe_factors":safe,

        "detected_data":{

            "IP Address":current_ip,

            "Device":current_device,

            "Time":current_time.strftime("%H:%M:%S"),

            "Txn count 24h":txn_count_24h,

            "Account age":user_age_days,

            "Failed logins":failed_attempts

        }

    }



@app.post("/register-user")

async def register_user(data:RegisterUser):

    user_ref=db.collection("users").document(data.uid)

    user_ref.set({

        "email":data.email,

        "last_ip":None,

        "last_device":None,

        "last_txn_time":None,

        "txn_count_24h":0,

        "account_age_days":30,

        "failed_attempts":0,

        "avg_txn_amount":0

    })

    return{"success":True}



app.mount("/static",StaticFiles(directory="static"),name="static")


@app.get("/")

async def home():

    return FileResponse("static/landing.html")


BASE_DIR=os.path.dirname(os.path.abspath(__file__))

app.mount("/static",StaticFiles(directory="static"),name="static")


@app.get("/")
async def landing():

    return FileResponse(
        os.path.join(BASE_DIR,"static","landing.html")
    )


@app.get("/app")
async def app_page():

    return FileResponse(
        os.path.join(BASE_DIR,"static","index.html")
    )


@app.get("/learn")
async def learn():

    return FileResponse(
        os.path.join(BASE_DIR,"static","learn-more.html")
    )