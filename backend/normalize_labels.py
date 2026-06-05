import pandas as pd

def normalize_labels(df):
    
    value_mapping = {
        'Stock price went up': 'up',
        'Stock price went down': 'down',
        'Stock price remained unchanged': 'unchanged',
    }

    df['Movement'] = df['Movement'].replace(value_mapping)

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