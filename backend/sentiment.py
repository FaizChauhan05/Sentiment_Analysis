import torch
import pandas as pd
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification



def sentiment_analysis(df):
    model_name = 'ProsusAI/finbert'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        dtype="auto"
    )

    df = df.drop(columns=['description', 'url'], errors = 'ignore')

    Sentiment = []
    Confidence = []

    for index, row in df.iterrows():

        headline = row['headline']

        inputs = tokenizer(
            headline,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            logit = model(**inputs)

            probabilities = F.softmax(logit.logits, dim=1)

            predicted_label = torch.argmax(probabilities).item()

            Sentiment.append(
                model.config.id2label[predicted_label]
            )

            Confidence.append(
                f"{probabilities[0][predicted_label].item():.3f}"
            )

    df['Sentiment'] = Sentiment
    df['Confidence'] = Confidence

    return df