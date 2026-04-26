# FraudGuard AI: Enterprise Fraud Detection System
**MSc Data Science Major Project**

## Overview
FraudGuard AI is an ensemble-based machine learning system designed to detect fraudulent financial transactions in real-time. It uses a combination of XGBoost and Random Forest models, supported by spatial intelligence (Haversine distance) and cyclical time features.

## Key Features
- **Hybrid Ensemble:** Combines XGBoost (60%) and Random Forest (40%).
- **Real-time API:** Built with FastAPI for high-performance inference.
- **Explainable AI (XAI):** Integrated with SHAP for model transparency.
- **Spatial Intelligence:** Calculates geographical distance between transaction and user home.

## Tech Stack
- **Language:** Python 3.x
- **Frameworks:** FastAPI, Uvicorn
- **ML Libraries:** Scikit-learn, XGBoost, SHAP, Joblib
- **Database:** SQLite