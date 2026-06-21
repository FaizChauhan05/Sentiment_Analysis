import pandas as pd

def aggregate_data(df):
    if df.empty:
        return df

    sentiment_score = []
    weighted_score = []
    for index, row in df.iterrows():
        if row['Sentiment'] == 'positive':
            score = 1
        elif row['Sentiment'] == 'negative':
            score = -1
        else:
            score = 0
        sentiment_score.append(score)
        weighted_score.append(score * float(row['Confidence']))
    
    df['Sentiment_Score'] = sentiment_score
    df['Aggregate_Score'] = weighted_score

    df['date'] = pd.to_datetime(
        df['published_at'],
        utc=True
    ).dt.normalize()
    
    # Vectorized flags for ratios
    df['is_pos'] = (df['Sentiment'] == 'positive').astype(int)
    df['is_neg'] = (df['Sentiment'] == 'negative').astype(int)

    # Perform daily aggregation
    aggregated_df = df.groupby(['ticker', 'date']).agg(
        Aggregate_Score=('Aggregate_Score', 'mean'),
        Article_Count=('headline', 'count'),
        Max_Confidence=('Confidence', 'max'),
        pos_count=('is_pos', 'sum'),
        neg_count=('is_neg', 'sum')
    ).reset_index()

    # Calculate ratios and spreads
    aggregated_df['Positive_Ratio'] = aggregated_df['pos_count'] / aggregated_df['Article_Count']
    aggregated_df['Negative_Ratio'] = aggregated_df['neg_count'] / aggregated_df['Article_Count']
    aggregated_df['Sentiment_Spread'] = aggregated_df['Positive_Ratio'] - aggregated_df['Negative_Ratio']

    # Clean up temp columns
    aggregated_df = aggregated_df.drop(columns=['pos_count', 'neg_count'])

    return aggregated_df