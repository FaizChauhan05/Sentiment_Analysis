# %% [markdown]
# # SentiCore XGBoost Retraining Pipeline
# 
# This notebook is designed to train a new 3-class XGBoost model to replace the hardcoded sentiment thresholds in the SentiCore financial sentiment dashboard.
# 
# ### Requirements Met:
# 1. **Real Historical Data**: Fetches Google News RSS, Yahoo Finance, and Bing News RSS historical articles (no GDELT/synthetic data).
# 2. **GPU Acceleration**: Utilizes Colab's GPU for both FinBERT batch inference (batch size = 64) and XGBoost training.
# 3. **ThreadPoolExecutor Parallelization**: Downloads stock history and financial metrics in parallel across multiple tickers.
# 4. **3-Class Model**: Predicts stock movement direction (`up`, `down`, or `unchanged`) using a `+/-0.5%` daily close return threshold.
# 5. **No Look-Ahead Bias**: Uses a chronological, time-based split (80% train, 20% test) and fits the StandardScaler *only* on the training split.
# 6. **14 Engineered Features**: Combines 6 aggregated daily sentiment features and 8 corporate financials.
# 7. **Progress Caching**: Automatically saves progress every 5 chunks to Parquet and caches duplicate headlines.
# 8. **scikit-learn 1.6.1 Compatibility**: Saves StandardScaler and XGBoost model via `joblib`/`pickle` to match the SentiCore dashboard environment.

# %%
# CELL 1: Install Dependencies
# Installs packages needed for yfinance, local FinBERT inference, and XGBoost.
# !pip install -q yfinance xgboost==3.2.0 scikit-learn==1.6.1 transformers torch tqdm pandas pyarrow requests

# %% [markdown]
# ### GPU Detection & Hugging Face Authentication
# We check for a CUDA-enabled GPU and set up Hugging Face authentication to fetch models or log in safely.

# %%
# CELL 2: GPU Detection & Hugging Face Token Handling
import os
import sys
import locale
import torch

# Force UTF-8 preferred encoding in Colab environment to prevent UnicodeEncodeErrors
locale.getpreferredencoding = lambda: "UTF-8"
os.environ["PYTHONIOENCODING"] = "utf-8"

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not detected. Running on CPU (FinBERT and XGBoost will be slower).")

# Retrieve Hugging Face Token using Colab secrets (userdata) or getpass
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    print("Successfully retrieved HF_TOKEN from Colab Secrets.")
except Exception:
    import getpass
    HF_TOKEN = os.environ.get("HF_TOKEN") or getpass.getpass("Enter your Hugging Face Token (optional, press Enter to skip): ")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    print("Hugging Face token configured in environment.")

# %% [markdown]
# ### Constants & Configuration
# Here we define the training parameters, date ranges, and feature groups.

# %%
# CELL 3: Constants & Configuration
from datetime import datetime

# Tickers supported in the SentiCore application
TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "AMD", "JPM"]

# Training configuration
START_DATE = "2025-10-01"
END_DATE = "2026-06-01"
MOVEMENT_THRESHOLD = 0.005  # +/-0.5% close return defines "unchanged"

# Features to build (14 total)
SENTIMENT_FEATURES = [
    "Aggregate_Score",
    "Positive_Ratio",
    "Negative_Ratio",
    "Sentiment_Spread",
    "Max_Confidence",
    "Article_Count"
]

FINANCIAL_FEATURES = [
    "EPS",
    "Revenue_Growth",
    "Free_Cash_Flow",
    "Net_Profit_Margin",
    "ROE",
    "PE_Ratio",
    "PEG_Ratio",
    "Debt_to_Equity"
]

FEATURE_COLUMNS = SENTIMENT_FEATURES + FINANCIAL_FEATURES

# %% [markdown]
# ### Multi-Source News Fetcher (RSS Feeds Only)
# Porting the SentiCore news fetching engine to collect articles from Google News RSS, Yahoo Finance RSS, and Bing News RSS feeds.

# %%
# CELL 4: News Fetcher Code
import re
import time
import random
import requests
import pandas as pd
from datetime import timedelta
from xml.etree import ElementTree

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GOOGLE_NEWS_TIMEOUT = 15
GOOGLE_NEWS_CHUNK_DAYS = 30

YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
YAHOO_TIMEOUT = 15

BING_NEWS_RSS = "https://www.bing.com/news/search"
BING_NEWS_TIMEOUT = 15

COMPANY_MAPPING = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "META": "Meta",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "JPM": "JPMorgan",
}

_FINANCIAL_QUERY_SUFFIXES = ("stock", "earnings", "shares", "revenue")
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

def _clamp_window(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("start_date and end_date must be valid date strings")
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")
    # Clamp to 14 days for chunked news pipeline efficiency
    if (end_dt - start_dt).days > 14:
        start_dt = end_dt - timedelta(days=14)
    return start_dt.normalize(), end_dt.normalize()

def _fetch_google_news_query(query_str: str) -> list[dict]:
    params = {"q": query_str, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(GOOGLE_NEWS_RSS, params=params, headers=headers, timeout=GOOGLE_NEWS_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError:
        return []

    articles = []
    for item in root.iter("item"):
        title_raw = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source_elem = item.findtext("source", "").strip()

        if not title_raw:
            continue

        title = title_raw
        source = source_elem or "Google News"
        if " - " in title_raw:
            parts = title_raw.rsplit(" - ", 1)
            title = parts[0].strip()
            if not source_elem:
                source = parts[1].strip()

        articles.append({
            "title": title,
            "url": link,
            "seendate": pub_date,
            "domain": source,
        })
    return articles

def _date_chunks(start_dt: pd.Timestamp, end_dt: pd.Timestamp, chunk_days: int = GOOGLE_NEWS_CHUNK_DAYS):
    chunks = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks

def _fetch_from_google_news(ticker: str, company: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> list[dict]:
    # Optimization: Combine terms into one combined OR search query to save 6x requests
    if company.upper() != ticker.upper():
        query_str = f'("{ticker}" OR "{company}" OR "{company} stock" OR "{company} earnings" OR "{company} shares" OR "{company} revenue")'
    else:
        query_str = f'"{ticker}"'

    chunks = _date_chunks(start_dt, end_dt)
    all_articles = []
    for chunk_start, chunk_end in chunks:
        date_filter = f"after:{chunk_start.strftime('%Y-%m-%d')} before:{chunk_end.strftime('%Y-%m-%d')}"
        arts = _fetch_google_news_query(f"{query_str} {date_filter}")
        all_articles.extend(arts)
        time.sleep(0.5)
    return all_articles

def _fetch_from_yahoo_rss(ticker: str) -> list[dict]:
    params = {"s": ticker, "region": "US", "lang": "en-US"}
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(YAHOO_RSS_URL, params=params, headers=headers, timeout=YAHOO_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError:
        return []

    articles = []
    for item in root.iter("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source = item.findtext("source", "Yahoo Finance").strip()

        if not title:
            continue

        articles.append({
            "title": title,
            "url": link,
            "seendate": pub_date,
            "domain": source,
        })
    return articles

def _fetch_from_bing_news(ticker: str, company: str) -> list[dict]:
    search_terms = [f"{ticker} stock"]
    if company.upper() != ticker.upper():
        search_terms.extend([
            f"{company} stock",
            f"{company} earnings",
            f"{company} shares",
        ])

    headers = {"User-Agent": _USER_AGENT}
    all_articles = []
    for term in search_terms:
        params = {"q": term, "format": "rss"}
        try:
            resp = requests.get(BING_NEWS_RSS, params=params, headers=headers, timeout=BING_NEWS_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException:
            continue

        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError:
            continue

        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            pub_date = item.findtext("pubDate", "").strip()

            if not title:
                continue

            source = "Bing News"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            all_articles.append({
                "title": title,
                "url": link,
                "seendate": pub_date,
                "domain": source,
            })
        time.sleep(0.3)
    return all_articles

def _normalise_headline(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def _normalise_articles(raw_articles: list[dict], ticker: str, company: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for art in raw_articles:
        headline = art.get("title", "").strip()
        published = art.get("seendate", "")
        if not headline or not published:
            continue

        rows.append({
            "ticker": ticker,
            "company": company,
            "headline": headline,
            "source": art.get("domain", ""),
            "published_at": published,
            "url": art.get("url", ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])

    start_utc = pd.Timestamp(start_dt, tz="UTC")
    end_utc = pd.Timestamp(end_dt, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[df["published_at"].between(start_utc, end_utc)]

    df["_norm"] = df["headline"].apply(_normalise_headline)
    df = df.drop_duplicates(subset=["_norm"])
    df = df.drop(columns=["_norm"])

    df = df.sort_values("published_at", ascending=False)
    df["published_at"] = df["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df.reset_index(drop=True)

def fetch_rss_news(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    company = COMPANY_MAPPING.get(ticker, ticker)
    start_dt, end_dt = _clamp_window(start_date, end_date)

    all_articles = []
    # Google News RSS, Yahoo RSS, and Bing News RSS fetching (replaces GDELT)
    all_articles.extend(_fetch_from_google_news(ticker, company, start_dt, end_dt))
    all_articles.extend(_fetch_from_yahoo_rss(ticker))
    all_articles.extend(_fetch_from_bing_news(ticker, company))

    return _normalise_articles(all_articles, ticker, company, start_dt, end_dt)

# %% [markdown]
# ### GPU-Accelerated FinBERT Sentiment Pipeline
# Set up model inference on GPU using standard Hugging Face pipelines, using a local JSON cache to save time and avoid duplicate work.

# %%
# CELL 5: GPU Sentiment Pipeline
import json
import torch
from pathlib import Path
from transformers import pipeline

CACHE_PATH = Path("sentiment_cache.json")

def load_sentiment_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_sentiment_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache))
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

# Load the model globally on GPU if available
device_idx = 0 if torch.cuda.is_available() else -1
print(f"Loading FinBERT on device: {device_idx}...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    device=device_idx,
    batch_size=64
)

def _normalize_label(raw_label: str) -> str:
    label = str(raw_label).strip().lower()
    if label in {"positive", "neutral", "negative"}:
        return label
    if label in {"label_0", "0"}:
        return "negative"
    if label in {"label_1", "1"}:
        return "neutral"
    if label in {"label_2", "2"}:
        return "positive"
    return label

def run_sentiment_analysis(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    if df.empty or "headline" not in df.columns:
        return df

    df = df.copy()
    headlines = df["headline"].astype(str).tolist()
    normalized = [" ".join(h.lower().split()) for h in headlines]

    seen_uncached = set()
    uncached = []
    for headline, norm in zip(headlines, normalized):
        if norm not in cache and norm not in seen_uncached:
            uncached.append(headline)
            seen_uncached.add(norm)

    if uncached:
        print(f"[Sentiment] Inference on {len(uncached)} uncached headlines via GPU...")
        results = sentiment_pipeline(uncached)
        for headline, res in zip(uncached, results):
            norm = " ".join(headline.lower().split())
            cache[norm] = {
                "label": _normalize_label(res["label"]),
                "score": float(res["score"])
            }
        save_sentiment_cache(cache)

    sentiments = []
    confidences = []
    for headline in headlines:
        norm = " ".join(headline.lower().split())
        entry = cache.get(norm)
        sentiments.append(entry["label"])
        confidences.append(entry["score"])

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    return df

# %% [markdown]
# ### Parallelized Market Data & Financial Fetcher
# Fetches yfinance financials and historical price data in parallel across tickers using `ThreadPoolExecutor`.

# %%
# CELL 6: Market Data & Financials Fetcher
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

def _first_number(info: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        try:
            if pd.notna(value):
                return float(value)
        except (TypeError, ValueError):
            continue
    return default

def fetch_ticker_financials(ticker_symbol: str) -> dict:
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}
    except Exception as e:
        print(f"[yfinance] Error fetching financials for {ticker_symbol}: {e}")
        info = {}

    return {
        "EPS": _first_number(info, "trailingEps"),
        "Revenue_Growth": _first_number(info, "revenueGrowth"),
        "Free_Cash_Flow": _first_number(info, "freeCashflow", "freeCashFlow"),
        "Net_Profit_Margin": _first_number(info, "profitMargins"),
        "ROE": _first_number(info, "returnOnEquity"),
        "PE_Ratio": _first_number(info, "trailingPE", "forwardPE"),
        "PEG_Ratio": _first_number(info, "pegRatio", "trailingPegRatio"),
        "Debt_to_Equity": _first_number(info, "debtToEquity")
    }

def fetch_single_ticker_data(ticker: str, start_date: str, end_date: str) -> tuple[str, dict, pd.DataFrame]:
    financials = fetch_ticker_financials(ticker)
    try:
        ticker_obj = yf.Ticker(ticker)
        # Pad dates to resolve business days offsets accurately
        start_pad = (pd.to_datetime(start_date) - BDay(5)).strftime("%Y-%m-%d")
        end_pad = (pd.to_datetime(end_date) + BDay(5)).strftime("%Y-%m-%d")
        history = ticker_obj.history(start=start_pad, end=end_pad, interval="1d")
        history.index = history.index.tz_localize(None)  # Make naive
    except Exception as e:
        print(f"[yfinance] Error downloading history for {ticker}: {e}")
        history = pd.DataFrame()

    return ticker, financials, history

def build_market_cache(tickers: list[str], start_date: str, end_date: str) -> dict:
    print(f"Downloading market data for {len(tickers)} tickers in parallel (max_workers=4)...")
    cache = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_single_ticker_data, t, start_date, end_date) for t in tickers]
        for f in futures:
            ticker, financials, history = f.result()
            cache[ticker] = {"financials": financials, "history": history}
    return cache

# %% [markdown]
# ### Feature Engineering & Data Aggregation
# Daily grouping and calculating the 6 daily sentiment features.

# %%
# CELL 7: Feature Engineering & Aggregation
def aggregate_sentiment_features(articles_df: pd.DataFrame) -> pd.DataFrame:
    if articles_df.empty:
        return pd.DataFrame()

    df = articles_df.copy()
    scores = []
    weighted_scores = []
    for _, row in df.iterrows():
        sentiment = str(row['Sentiment']).lower()
        confidence = float(row['Confidence'])
        if sentiment == 'positive':
            score = 1.0
        elif sentiment == 'negative':
            score = -1.0
        else:
            score = 0.0
        scores.append(score)
        weighted_scores.append(score * confidence)

    df['score'] = scores
    df['weighted_score'] = weighted_scores
    df['date'] = pd.to_datetime(df['published_at'], utc=True).dt.normalize()

    grouped = df.groupby(['ticker', 'date'])

    aggregated = grouped.agg(
        Aggregate_Score=('weighted_score', 'mean'),
        Article_Count=('headline', 'count'),
        Max_Confidence=('Confidence', 'max')
    ).reset_index()

    pos_counts = grouped.apply(lambda x: (x['Sentiment'] == 'positive').sum(), include_groups=False)
    neg_counts = grouped.apply(lambda x: (x['Sentiment'] == 'negative').sum(), include_groups=False)

    pos_neg = pd.DataFrame({
        'pos_count': pos_counts,
        'neg_count': neg_counts
    }).reset_index()

    aggregated = pd.merge(aggregated, pos_neg, on=['ticker', 'date'])
    aggregated['Positive_Ratio'] = aggregated['pos_count'] / aggregated['Article_Count']
    aggregated['Negative_Ratio'] = aggregated['neg_count'] / aggregated['Article_Count']
    aggregated['Sentiment_Spread'] = aggregated['Positive_Ratio'] - aggregated['Negative_Ratio']

    return aggregated.drop(columns=['pos_count', 'neg_count'])

# %% [markdown]
# ### Alignment & Labeling Stock Price Movement (Target Variable)
# Maps article dates to New York trading days and labels stock movement based on the 3-class system (threshold `+/-0.5%`).

# %%
# CELL 8: Alignment & Labeling
import numpy as np
from pandas.tseries.offsets import BDay

def get_target_date(utc_date) -> pd.Timestamp:
    utc_dt = pd.to_datetime(utc_date, utc=True)
    est_dt = utc_dt.tz_convert('America/New_York')
    # Hour >= 16 represents after-hours news; maps to the next business trading day
    if est_dt.hour >= 16:
        target_date = est_dt + BDay(1)
    else:
        target_date = est_dt
    return target_date.normalize()

def get_movement_and_ohlcv(history_df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    target_dt_naive = pd.to_datetime(target_date).tz_localize(None)
    sub_df = history_df[history_df.index <= target_dt_naive]
    if len(sub_df) < 2:
        return {
            "Open": np.nan, "High": np.nan, "Low": np.nan,
            "Close": np.nan, "Volume": np.nan, "Pct_Change": np.nan, "Movement": "No data"
        }

    last_row = sub_df.iloc[-1]
    prev_row = sub_df.iloc[-2]

    current_close = float(last_row['Close'])
    previous_close = float(prev_row['Close'])
    pct_change = (current_close - previous_close) / previous_close

    # Apply 3-class mapping with +/-0.5% threshold
    if pct_change > MOVEMENT_THRESHOLD:
        movement = 'up'
    elif pct_change < -MOVEMENT_THRESHOLD:
        movement = 'down'
    else:
        movement = 'unchanged'

    return {
        "Open": float(last_row['Open']),
        "High": float(last_row['High']),
        "Low": float(last_row['Low']),
        "Close": current_close,
        "Volume": int(last_row['Volume']),
        "Pct_Change": pct_change,
        "Movement": movement
    }

# %% [markdown]
# ### Chunk-by-Chunk Dataset Construction & Saving
# Running the data pipeline chunk by chunk to fetch historical data, run inference, aggregate features, and save checkpoints every 5 chunks to Parquet.

# %%
# CELL 9: Historical Loop
import os
from tqdm.notebook import tqdm

def build_date_chunks(start_date: str, end_date: str, chunk_size_days: int = 14) -> list[tuple[str, str]]:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    chunks = []
    curr = start_dt
    while curr < end_dt:
        chunk_end = min(curr + pd.Timedelta(days=chunk_size_days), end_dt)
        chunks.append((curr.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        curr = chunk_end
    return chunks

# 1. Initialize market data cache and sentiment cache
market_cache = build_market_cache(TICKERS, START_DATE, END_DATE)
sentiment_cache = load_sentiment_cache()

# 2. Get date chunks (14-day intervals to optimize execution speed)
date_chunks = build_date_chunks(START_DATE, END_DATE, chunk_size_days=14)
print(f"Divided the date range into {len(date_chunks)} chunks.")

# 3. Master chunk loop
all_processed_chunks = []
checkpoint_file = "historical_dataset_checkpoint.parquet"

# Define news fetching worker inside thread pool to process 4 tickers in parallel
def fetch_news_worker(args):
    ticker, start, end = args
    try:
        return fetch_rss_news(ticker, start, end)
    except Exception as e:
        print(f"Error fetching news for {ticker} from {start} to {end}: {e}")
        return pd.DataFrame()

for idx, (chunk_start, chunk_end) in enumerate(tqdm(date_chunks, desc="Processing Chunks")):
    print(f"\n--- Chunk {idx+1}/{len(date_chunks)}: {chunk_start} to {chunk_end} ---")
    
    # Fetch news in parallel for tickers
    chunk_articles_list = []
    worker_args = [(t, chunk_start, chunk_end) for t in TICKERS]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(fetch_news_worker, worker_args)
        for df_t in results:
            if not df_t.empty:
                chunk_articles_list.append(df_t)
                
    if not chunk_articles_list:
        print("No articles found in this chunk.")
        continue
        
    chunk_news_df = pd.concat(chunk_articles_list, ignore_index=True)
    
    # GPU Sentiment Inference
    chunk_scored_df = run_sentiment_analysis(chunk_news_df, sentiment_cache)
    
    # Aggregate Features
    chunk_agg_df = aggregate_sentiment_features(chunk_scored_df)
    if chunk_agg_df.empty:
        continue
        
    # Align Sentiment with Market Movements and Financials
    chunk_aligned_rows = []
    for _, row in chunk_agg_df.iterrows():
        t = row["ticker"]
        utc_date = row["date"]
        
        target_dt = get_target_date(utc_date)
        cache_data = market_cache.get(t, {})
        history_df = cache_data.get("history", pd.DataFrame())
        financials = cache_data.get("financials", {})
        
        movement_data = get_movement_and_ohlcv(history_df, target_dt)
        if movement_data["Movement"] == "No data":
            continue
            
        combined_row = {
            "ticker": t,
            "date": utc_date.strftime("%Y-%m-%d"),
            **row.to_dict(),
            **financials,
            **movement_data
        }
        chunk_aligned_rows.append(combined_row)
        
    if chunk_aligned_rows:
        chunk_aligned_df = pd.DataFrame(chunk_aligned_rows)
        all_processed_chunks.append(chunk_aligned_df)
        print(f"Aligned {len(chunk_aligned_df)} rows in chunk.")
        
    # Save checkpoint every 5 chunks
    if (idx + 1) % 5 == 0 and all_processed_chunks:
        temp_complete_df = pd.concat(all_processed_chunks, ignore_index=True)
        temp_complete_df.to_parquet(checkpoint_file, index=False)
        print(f"[Checkpoint] Progress saved to {checkpoint_file}. Total rows: {len(temp_complete_df)}")

# Concatenate all chunks to make the complete dataset
if all_processed_chunks:
    df_complete = pd.concat(all_processed_chunks, ignore_index=True)
    df_complete.to_parquet("historical_dataset_complete.parquet", index=False)
    print(f"\nMaster Run Completed! Saved to 'historical_dataset_complete.parquet'. Total rows: {len(df_complete)}")
else:
    print("\nFatal Error: No data collected in the master run.")
    df_complete = pd.DataFrame()

# %% [markdown]
# ### Time-Based Train/Test Split
# We sort the dataset chronologically and partition it into 80% train and 20% test splits. Features are standardized using `StandardScaler` fitted only on training data.

# %%
# CELL 10: Train/Test Split & Scaling
from sklearn.preprocessing import StandardScaler

if df_complete.empty:
    raise ValueError("Dataset is empty. Cannot perform split.")

# Sort chronologically by date
df_complete = df_complete.sort_values(by="date").reset_index(drop=True)

# Split index (80% train, 20% test)
split_idx = int(len(df_complete) * 0.8)
train_df = df_complete.iloc[:split_idx]
test_df = df_complete.iloc[split_idx:]

print(f"Dataset split index: {split_idx}")
print(f"Train set date range: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} rows)")
print(f"Test set date range: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} rows)")

# Map target label class to integer index
label_map = {"down": 0, "unchanged": 1, "up": 2}
y_train = train_df["Movement"].map(label_map)
y_test = test_df["Movement"].map(label_map)

# Extract and fill feature NaNs with 0
X_train_raw = train_df[FEATURE_COLUMNS].fillna(0)
X_test_raw = test_df[FEATURE_COLUMNS].fillna(0)

# Normalize features (scaler fitted on training set only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print("Feature scaling completed.")

# %% [markdown]
# ### GPU-Accelerated 3-Class XGBoost Model Training
# Trains XGBoost using `multi:softprob` objective and enables GPU support dynamically (handling old/new versions of XGBoost).

# %%
# CELL 11: XGBoost Training
import xgboost as xgb

xgb_params = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "objective": "multi:softprob",
    "num_class": 3,
    "random_state": 42
}

# Apply GPU parameter overrides if CUDA is active
if torch.cuda.is_available():
    try:
        from pkg_resources import parse_version
        if parse_version(xgb.__version__) >= parse_version("2.0.0"):
            xgb_params["device"] = "cuda"
        else:
            xgb_params["tree_method"] = "gpu_hist"
        print(f"Enabling GPU acceleration inside XGBoost: {xgb_params.get('device') or xgb_params.get('tree_method')}")
    except Exception as e:
        print(f"Could not apply GPU parameter overrides: {e}")

# Train the model
model = xgb.XGBClassifier(**xgb_params)
print("\nTraining XGBoost classifier...")
model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=50
)
print("XGBoost training completed successfully.")

# %% [markdown]
# ### Evaluation & Analysis
# Generates classification performance reports, plots feature importances, and confusion matrix.

# %%
# CELL 12: Evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Predict on test split
y_pred = model.predict(X_test_scaled)
target_names = ["down", "unchanged", "up"]

# Generate Classification Report
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Plot and save Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_names, cmap="Blues", ax=ax)
plt.title("Confusion Matrix (SentiCore 3-Class XGBoost)")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Plot and save Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (SentiCore 14 Features)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [FEATURE_COLUMNS[i] for i in indices], rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()

test_accuracy = report["accuracy"] * 100
print(f"Overall Model Accuracy on Test Set: {test_accuracy:.2f}%")

# %% [markdown]
# ### Export and Download Artifacts
# Pickles the XGBoost classifier and StandardScaler (joblib formats), saves metadata, and downloads everything locally.

# %%
# CELL 13: Export & Download
import joblib
import pickle

# Save XGBoost Model & Scaler (compatible with joblib.load)
joblib.dump(model, "xgboost_macro_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
print("Saved models: xgboost_macro_model.pkl and feature_scaler.pkl")

# Compile and save training metadata
metadata = {
    "features": FEATURE_COLUMNS,
    "accuracy_pct": round(test_accuracy, 2),
    "feature_importances": {FEATURE_COLUMNS[i]: float(importances[i]) for i in range(len(FEATURE_COLUMNS))},
    "class_mapping": label_map,
    "data_summary": {
        "total_rows": len(df_complete),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "start_date": df_complete["date"].min(),
        "end_date": df_complete["date"].max(),
        "movement_distribution": {
            "down": int((df_complete["Movement"] == "down").sum()),
            "unchanged": int((df_complete["Movement"] == "unchanged").sum()),
            "up": int((df_complete["Movement"] == "up").sum()),
        }
    },
    "training_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("Saved metadata to model_metadata.json")

# Download files to local drive if running in Google Colab environment
try:
    from google.colab import files
    print("\nTriggering files downloads...")
    files.download("xgboost_macro_model.pkl")
    files.download("feature_scaler.pkl")
    files.download("model_metadata.json")
    files.download("historical_dataset_complete.parquet")
    files.download("confusion_matrix.png")
except ImportError:
    print("\nSkipping files.download since we are not in Google Colab.")
    print("Produced files:")
    print("- xgboost_macro_model.pkl")
    print("- feature_scaler.pkl")
    print("- model_metadata.json")
    print("- historical_dataset_complete.parquet")
    print("- confusion_matrix.png")
