import xgboost
import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_macro_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")


try:
    xgb_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Production Binary XGBoost Engine & Feature Scaler loaded successfully.")
except FileNotFoundError:
    xgb_model = None
    scaler = None
    print("Warning: Model artifacts missing. Falling back to default thresholds.")

def normalize_labels(df):
    value_mapping = {
        'Stock price went up': 'up',
        'Stock price went down': 'down',
        'Stock price remained unchanged': 'unchanged',
    }
    df['Movement'] = df['Movement'].replace(value_mapping)

    if xgb_model is not None and scaler is not None:
        feature_columns = [
            "Aggregate_Score", "EPS", "Revenue_Growth", "Free_Cash_Flow", 
            "Net_Profit_Margin", "ROE", "PE_Ratio", "PEG_Ratio", "Debt_to_Equity"
        ]
        
        X_raw = df[feature_columns]
        X_scaled = scaler.transform(X_raw)
        
        binary_predictions = xgb_model.predict(X_scaled)
        
        
        predictions = []
        for pred in binary_predictions:
            if pred == 1:
                predictions.append('up')
            else:
                predictions.append('down')

        df['Predictions'] = predictions

    else:
        predictions = []
        for score in df['Aggregate_Score']:
            if score > 0.3: 
                predictions.append('up')
            elif score < -0.3: 
                predictions.append('down')
            else: 
                predictions.append('unchanged')
        df['Predictions'] = predictions

    return df
