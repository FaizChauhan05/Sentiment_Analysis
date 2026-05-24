import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def evaluate_predictions(df):

    y_true = df['Movement']
    y_pred = df['Predictions']

    cm = confusion_matrix(y_true, y_pred, labels = ['up', 'down', 'unchanged'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['up', 'down', 'unchanged'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Stock Movement Prediction')
    plt.savefig('confusion_matrix.png')
    plt.close()
    accuracy = (
        (y_true == y_pred).mean()
    ) * 100

    total_predictions = len(df)
    correct_predictions = int((y_true == y_pred).sum())
    incorrect_predictions = int(total_predictions - correct_predictions)

    return {
    "accuracy": round(accuracy, 2),

    "total_articles": int(total_predictions),

    "correct_predictions": int(correct_predictions),

    "incorrect_predictions": int(incorrect_predictions),

    "movement_distribution": {
        "up": int((y_true == 'up').sum()),
        "down": int((y_true == 'down').sum()),
        "unchanged": int((y_true == 'unchanged').sum())
    },

    "prediction_distribution": {
    "up": int((df['Predictions'] == 'up').sum()),
    "down": int((df['Predictions'] == 'down').sum()),
    "unchanged": int((df['Predictions'] == 'unchanged').sum())
   },

    "latest_market_data": {
        "open": float(df['Open'].iloc[-1]),
        "high": float(df['High'].iloc[-1]),
        "low": float(df['Low'].iloc[-1]),
        "close": float(df['Close'].iloc[-1]),
        "volume": int(df['Volume'].iloc[-1])
    }
}