import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_model.pkl")


def predict_transaction(transaction):

    df=pd.DataFrame([transaction])

    pred=model.predict(df)[0]

    return "Fraud" if pred==1 else "Normal"



if __name__=="__main__":

    txn={

        "amount":2000,

        "is_mal_ip":0,

        "is_mal_device":1,

        "odd_time":1,

        "txn_count_24h":4,

        "user_age_days":120,

        "failed_attempts":1,

        "location_change":0,

        "avg_txn_amount":1800,

        "is_international":0
    }

    print("Prediction:",predict_transaction(txn))