import pandas as pd

def aggregate_data(df):

    sentiment_score = []
    weighted_score = []
    for index, row in df.iterrows():

        if row['Sentiment'] == 'positive':
            score = (1)
        elif row['Sentiment'] == 'negative':
            score = (-1)
        else :
            score = (0)
        sentiment_score.append(score)
        weighted_score.append(score * float(row['Confidence']))
    

    df['Sentiment_Score'] = sentiment_score
    df['Aggregate_Score'] = weighted_score

    df['date'] = pd.to_datetime(
    df['published_at'],
    utc=True
)
    aggregated_df = df.groupby(['ticker', 'date'])[['Aggregate_Score']].mean().reset_index()

    return aggregated_df