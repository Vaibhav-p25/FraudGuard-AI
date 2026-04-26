import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

def train_ensemble_model():
    # Synthetic data matching your project features
    data = pd.DataFrame({
        'amount': [100, 50000, 200, 85000, 150, 92000, 300, 45000, 120, 77000],
        'dist': [1.5, 600.0, 0.5, 1100.0, 2.1, 950.0, 3.2, 550.0, 0.8, 880.0],
        'hr_sin': [0.5, -0.7, 0.2, -0.9, 0.4, -0.8, 0.6, -0.5, 0.1, -0.9],
        'hr_cos': [0.8, 0.3, 0.9, -0.1, 0.8, 0.2, 0.7, 0.4, 0.9, 0.1],
        'is_fraud': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    # 1. Train XGBoost
    model_xgb = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    model_xgb.fit(X, y)
    
    # 2. Train Random Forest
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X, y)

    # Save models into the 'models' folder (going up one level from src)
    model_dir = os.path.join('..', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(model_xgb, os.path.join(model_dir, 'xgb_model.pkl'))
    joblib.dump(model_rf, os.path.join(model_dir, 'rf_model.pkl'))
    
    print("Models trained and saved in /models/ folder!")

    # --- SHAP SECTION ---
    print("Generating SHAP graph...")
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    # Saving the plot to the main project folder
    plt.savefig('../shap_summary_plot.png', bbox_inches='tight')
    plt.close()
    print("SHAP graph saved as 'shap_summary_plot.png' in the main folder.")

if __name__ == "__main__":
    train_ensemble_model()