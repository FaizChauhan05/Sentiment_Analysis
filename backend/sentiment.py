import os
import time
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_finbert_api(payload, retries=3):
    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        elif "estimated_time" in response.text:
            # The API is waking the model up from sleep; wait and retry
            wait_time = response.json().get("estimated_time", 10)
            print(f"[API] Waking up FinBERT. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        else:
            time.sleep(2)
            
    raise Exception(f"API Error after retries: {response.text}")

def sentiment_analysis(df):
    if df.empty or 'headline' not in df.columns:
        return df
        
    df = df.dropna(subset=['headline']).copy()
    headlines = df['headline'].astype(str).tolist()
    
    all_sentiments = []
    all_confidences = []

    # Send data to the API in chunks of 100 to prevent payload timeouts
    BATCH_SIZE = 100
    print(f"[Sentiment] Routing {len(headlines)} articles to Hugging Face API...")
    
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i : i + BATCH_SIZE]
        try:
            api_results = query_finbert_api({"inputs": batch})
            
            
            for result in api_results:
                top_prediction = result[0] 
                all_sentiments.append(top_prediction['label'])
                all_confidences.append(top_prediction['score'])
                
        except Exception as e:
            print(f"[Sentiment] Batch failed: {e}")
            
            all_sentiments.extend(['neutral'] * len(batch))
            all_confidences.extend([0.0] * len(batch))
            
    df['Sentiment'] = all_sentiments
    df['Confidence'] = all_confidences

    return df