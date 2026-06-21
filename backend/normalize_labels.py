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
            "Aggregate_Score", "Positive_Ratio", "Negative_Ratio", "Sentiment_Spread", 
            "Max_Confidence", "Article_Count", "EPS", "Revenue_Growth", "Free_Cash_Flow", 
            "Net_Profit_Margin", "ROE", "PE_Ratio", "PEG_Ratio", "Debt_to_Equity"
        ]

        X_raw = df[feature_columns].fillna(0)
        X_scaled = scaler.transform(X_raw)
        
        # Predict 3-class encoded labels: 0=down, 1=unchanged, 2=up
        predictions_encoded = xgb_model.predict(X_scaled)
        class_mapping = {0: 'down', 1: 'unchanged', 2: 'up'}
        model_predictions = np.array([class_mapping[p] for p in predictions_encoded])

        collapse_ratio = max(
            (model_predictions == 'up').mean(),
            (model_predictions == 'down').mean(),
            (model_predictions == 'unchanged').mean(),
        )
        if collapse_ratio >= 0.95:
            fallback_predictions = []
            for score in df['Aggregate_Score']:
                if score > 0.3:
                    fallback_predictions.append('up')
                elif score < -0.3:
                    fallback_predictions.append('down')
                else:
                    fallback_predictions.append('unchanged')
            df['Predictions'] = fallback_predictions
        else:
            df['Predictions'] = model_predictions

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
