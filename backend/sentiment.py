import os
import time
import logging

# Suppress noisy transformers warnings before importing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds between retries on rate-limit

@st.cache_resource
def load_finbert_model():
    model_name = 'ProsusAI/finbert'

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
            model.eval()
            return tokenizer, model
        except Exception as exc:
            if "429" in str(exc) or "rate" in str(exc).lower() or "Too Many Requests" in str(exc):
                if attempt < MAX_RETRIES:
                    st.warning(
                        f"HuggingFace rate-limited (attempt {attempt}/{MAX_RETRIES}). "
                        f"Retrying in {RETRY_DELAY}s..."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(
                        "HuggingFace rate limit exceeded after multiple retries. "
                        "Please wait a few minutes and try again."
                    ) from exc
            else:
                raise

def sentiment_analysis(df):
    if df.empty:
        return df
        
    tokenizer, model = load_finbert_model()
    df = df.drop(columns=['description', 'url'], errors='ignore')
    
    Sentiment = []
    Confidence = []

    for index, row in df.iterrows():
        headline = row['headline']
        inputs = tokenizer(headline, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            logit = model(**inputs)
            probabilities = F.softmax(logit.logits, dim=1)
            predicted_label = int(torch.argmax(probabilities).item())
            
            Sentiment.append(model.config.id2label[predicted_label])
            Confidence.append(f"{probabilities[0][predicted_label].item():.3f}")

    df['Sentiment'] = Sentiment
    df['Confidence'] = Confidence
    
    return df