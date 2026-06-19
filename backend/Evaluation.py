import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay


def evaluate_predictions(df: pd.DataFrame) -> dict:
  
    y_true = df["Movement"]
    y_pred = df["Predictions"]

    labels = ["up", "down", "unchanged"]

    # Core confusion matrix (side effect: saves plot)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix for Stock Movement Prediction")

    # Accuracy
    correct_mask = y_true == y_pred
    accuracy = correct_mask.mean() * 100
    total_predictions = len(df)
    correct_predictions = int(correct_mask.sum())
    incorrect_predictions = total_predictions - correct_predictions

    # Per-class precision/recall/f1 via classification_report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class_metrics = {}
    for label in labels:
        if label in report:
            per_class_metrics[label] = {
                "precision": round(report[label]["precision"] * 100, 1),
                "recall": round(report[label]["recall"] * 100, 1),
                "f1": round(report[label]["f1-score"] * 100, 1),
                "support": int(report[label]["support"]),
            }

    # Daily accuracy (how accurate was each day's prediction)
    daily_accuracy = []
    if "date" in df.columns:
        for d, grp in df.groupby("date"):
            day_correct = (grp["Movement"] == grp["Predictions"]).sum()
            day_total = len(grp)
            daily_accuracy.append({
                "date": str(d),
                "accuracy": round(day_correct / max(day_total, 1) * 100, 1),
                "total": day_total,
                "correct": int(day_correct),
            })

    return {
        "accuracy": round(accuracy, 2),
        "total_articles": int(total_predictions),
        "correct_predictions": int(correct_predictions),
        "incorrect_predictions": int(incorrect_predictions),
        "movement_distribution": {
            "up": int((y_true == "up").sum()),
            "down": int((y_true == "down").sum()),
            "unchanged": int((y_true == "unchanged").sum()),
        },
        "prediction_distribution": {
            "up": int((y_pred == "up").sum()),
            "down": int((y_pred == "down").sum()),
            "unchanged": int((y_pred == "unchanged").sum()),
        },
        "latest_market_data": {
            "open": float(df["Open"].iloc[-1]),
            "high": float(df["High"].iloc[-1]),
            "low": float(df["Low"].iloc[-1]),
            "close": float(df["Close"].iloc[-1]),
            "volume": int(df["Volume"].iloc[-1]),
        },
        "per_class_metrics": per_class_metrics,
        "daily_accuracy": daily_accuracy,

        "corporate_financials": {
        "eps": round(float(df["EPS"].iloc[-1]), 2),
        "revenue_growth_pct": round(float(df["Revenue_Growth"].iloc[-1]) * 100, 2), 
        "net_margin_pct": round(float(df["Net_Profit_Margin"].iloc[-1]) * 100, 2),
        "roe_pct": round(float(df["ROE"].iloc[-1]) * 100, 2)
    }
    }