from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import sqlite3
import os
import uvicorn
# Importing from our other file in the same folder
from preprocessor import calculate_haversine_distance, encode_time_features

app = FastAPI(title="FraudGuard AI - Enterprise Suite")

# Define where the models and data are located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
DB_PATH = os.path.join(BASE_DIR, '..', 'data', 'fraud_guard.db')

# Load the Brains (Models)
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))

class TransactionRequest(BaseModel):
    user_id: int
    amount: float
    txn_lat: float
    txn_lon: float
    hour: int

def db_query(query, params=(), fetch=False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchone() if fetch else None
    conn.commit()
    conn.close()
    return result

@app.post("/v1/predict/fraud-score")
async def predict_fraud(data: TransactionRequest):
    # 1. Get User Data
    user_data = db_query("SELECT home_lat, home_lon FROM users WHERE user_id = ?", (data.user_id,), fetch=True)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    home_lat, home_lon = user_data

    # 2. Check Velocity (Transaction History)
    velocity_data = db_query("SELECT COUNT(*) FROM transactions WHERE user_id = ?", (data.user_id,), fetch=True)
    velocity = velocity_data[0] if velocity_data else 0

    # 3. Feature Engineering
    dist = calculate_haversine_distance(home_lat, home_lon, data.txn_lat, data.txn_lon)
    hr_sin, hr_cos = encode_time_features(data.hour)
    
    # 4. Model Prediction
    features = np.array([[data.amount, dist, hr_sin, hr_cos]])
    p1 = xgb_model.predict_proba(features)[0][1] # XGBoost probability
    p2 = rf_model.predict_proba(features)[0][1]  # Random Forest probability
    final_score = (p1 * 0.6) + (p2 * 0.4)        # Weighted Ensemble
    
    # 5. Log Transaction
    db_query("INSERT INTO transactions (user_id, amount, lat, lon) VALUES (?, ?, ?, ?)", 
             (data.user_id, data.amount, data.txn_lat, data.txn_lon))

    # 6. Final Decision
    decision = "BLOCK" if final_score > 0.7 or velocity > 5 else "APPROVE"

    return {
        "status": "Success",
        "fraud_score": round(float(final_score), 4),
        "decision": decision,
        "details": {
            "distance_km": round(dist, 2),
            "velocity_count": velocity
        }
    }

if __name__ == "__main__":
    print("FraudGuard AI API is starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)