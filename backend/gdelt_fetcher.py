"""
Multi-source financial news fetcher for SentiCore.

Sources (tried in parallel, results merged):
    1. GDELT DOC 2.0 API   – up to 250 articles, 90-day window, rate-limited
    2. Google News RSS      – ~100 articles per query, date-filterable, very reliable
    3. Yahoo Finance RSS    – ~20 recent articles, always-on fallback

The module exposes a single public function:

    fetch_gdelt_news(ticker, start_date, end_date) -> pd.DataFrame

Columns: ticker, company, headline, source, published_at, url
"""

import random
import re
import time
from datetime import datetime, timedelta
from xml.etree import ElementTree

import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_INTERVAL = 6          # seconds between GDELT requests
GDELT_MAX_RECORDS = 250         # artlist mode cap
GDELT_MAX_RETRIES = 2           # keep low so fallback kicks in fast
GDELT_TIMEOUT = 30

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GOOGLE_NEWS_TIMEOUT = 15

YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
YAHOO_TIMEOUT = 15

COMPANY_MAPPING: dict[str, str] = {
    "AAPL":  "Apple",
    "TSLA":  "Tesla",
    "NVDA":  "Nvidia",
    "MSFT":  "Microsoft",
    "AMZN":  "Amazon",
    "GOOGL": "Google",
    "GOOG":  "Google",
    "META":  "Meta",
    "NFLX":  "Netflix",
    "AMD":   "AMD",
    "JPM":   "JPMorgan",
    "INTC":  "Intel",
    "BA":    "Boeing",
    "DIS":   "Disney",
    "V":     "Visa",
}

# Financial search terms appended to company name for Google News queries
_FINANCIAL_QUERY_SUFFIXES = ("stock", "earnings", "shares", "revenue")

_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_last_gdelt_ts: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp_window(
    start_date: str, end_date: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse and clamp dates to a max 90-day window."""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("start_date and end_date must be valid date strings")
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")
    if (end_dt - start_dt).days > 90:
        start_dt = end_dt - timedelta(days=90)

    return start_dt.normalize(), end_dt.normalize()


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: GDELT  (primary — high volume when available)
# ═══════════════════════════════════════════════════════════════════════════════

def _gdelt_rate_wait() -> None:
    global _last_gdelt_ts
    elapsed = time.monotonic() - _last_gdelt_ts
    if elapsed < GDELT_MIN_INTERVAL:
        time.sleep(GDELT_MIN_INTERVAL - elapsed)
    _last_gdelt_ts = time.monotonic()


def _fetch_from_gdelt(
    ticker: str,
    company: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> list[dict]:
    """Hit GDELT DOC 2.0 with a simple query + exponential backoff."""

    parts = [f'"{company}"'] if company.upper() != ticker.upper() else []
    parts.append(f'"{ticker}"')
    query = f"({' OR '.join(parts)}) sourcelang:english"

    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    GDELT_MAX_RECORDS,
        "format":        "json",
        "sort":          "datedesc",
        "STARTDATETIME": start_dt.strftime("%Y%m%d000000"),
        "ENDDATETIME":   end_dt.strftime("%Y%m%d235959"),
    }
    headers = {"User-Agent": _USER_AGENT}

    for attempt in range(GDELT_MAX_RETRIES):
        _gdelt_rate_wait()
        try:
            resp = requests.get(
                GDELT_DOC_URL, params=params, headers=headers, timeout=GDELT_TIMEOUT
            )
        except requests.RequestException as exc:
            print(f"[GDELT] Request error (attempt {attempt + 1}): {exc}")
            time.sleep(GDELT_MIN_INTERVAL)
            continue

        if resp.status_code == 429:
            wait = GDELT_MIN_INTERVAL * (2 ** attempt) + random.uniform(1, 3)
            print(f"[GDELT] 429 rate-limited, backing off {wait:.0f}s "
                  f"(attempt {attempt + 1}/{GDELT_MAX_RETRIES})")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"[GDELT] HTTP {resp.status_code}: {resp.text[:200]}")
            return []

        try:
            data = resp.json()
        except ValueError:
            print("[GDELT] Non-JSON response")
            return []

        articles = data.get("articles", [])
        print(f"[GDELT] Returned {len(articles)} articles")
        return articles

    print("[GDELT] Exhausted retries — will rely on other sources")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: GOOGLE NEWS RSS  (high volume, very reliable)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_google_news_query(query_str: str) -> list[dict]:
    """Fetch a single Google News RSS query and return normalised dicts."""
    params = {
        "q":    query_str,
        "hl":   "en-US",
        "gl":   "US",
        "ceid": "US:en",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(
            GOOGLE_NEWS_RSS, params=params, headers=headers, timeout=GOOGLE_NEWS_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Google News] Error for '{query_str}': {exc}")
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        print(f"[Google News] XML parse error: {exc}")
        return []

    articles: list[dict] = []
    for item in root.iter("item"):
        title_raw = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source_elem = item.findtext("source", "").strip()

        if not title_raw:
            continue

        # Google News appends " - SourceName" to titles; split it off
        title = title_raw
        source = source_elem or "Google News"
        if " - " in title_raw:
            parts = title_raw.rsplit(" - ", 1)
            title = parts[0].strip()
            if not source_elem:
                source = parts[1].strip()

        articles.append({
            "title":    title,
            "url":      link,
            "seendate": pub_date,
            "domain":   source,
        })

    return articles


def _fetch_from_google_news(
    ticker: str,
    company: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> list[dict]:
    """
    Run multiple Google News RSS queries with date filtering.
    Returns raw article dicts in GDELT-compatible shape.
    """
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    date_filter = f"after:{start_str} before:{end_str}"

    # Build diverse queries for maximum coverage
    queries: list[str] = []

    # Ticker-based queries
    queries.append(f"{ticker} {date_filter}")

    # Company + financial terms
    if company.upper() != ticker.upper():
        for suffix in _FINANCIAL_QUERY_SUFFIXES:
            queries.append(f"{company} {suffix} {date_filter}")

    all_articles: list[dict] = []
    for q in queries:
        arts = _fetch_google_news_query(q)
        all_articles.extend(arts)
        # Small delay to be polite
        time.sleep(0.5)

    print(f"[Google News] {len(all_articles)} articles from {len(queries)} queries "
          f"(before dedup)")
    return all_articles


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: YAHOO FINANCE RSS  (always-on fallback, ~20 articles)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_from_yahoo_rss(ticker: str) -> list[dict]:
    """Scrape Yahoo Finance RSS feed for a ticker."""
    params = {"s": ticker, "region": "US", "lang": "en-US"}
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(
            YAHOO_RSS_URL, params=params, headers=headers, timeout=YAHOO_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Yahoo RSS] Error: {exc}")
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        print(f"[Yahoo RSS] XML parse error: {exc}")
        return []

    articles: list[dict] = []
    for item in root.iter("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source = item.findtext("source", "Yahoo Finance").strip()

        if not title:
            continue

        articles.append({
            "title":    title,
            "url":      link,
            "seendate": pub_date,
            "domain":   source,
        })

    print(f"[Yahoo RSS] {len(articles)} articles")
    return articles


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISATION & DEDUPLICATION  (shared across all sources)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_headline(text: str) -> str:
    """Lowercase, strip punctuation — used for fuzzy dedup."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _normalise_articles(
    raw_articles: list[dict],
    ticker: str,
    company: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    """Turn raw article dicts into a clean, de-duped, date-filtered DataFrame."""

    rows: list[dict] = []
    for art in raw_articles:
        headline = art.get("title", "").strip()
        published = art.get("seendate", "")

        if not headline or not published:
            continue

        rows.append({
            "ticker":       ticker,
            "company":      company,
            "headline":     headline,
            "source":       art.get("domain", ""),
            "published_at": published,
            "url":          art.get("url", ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])

    # Date-window filter
    start_utc = pd.Timestamp(start_dt, tz="UTC")
    end_utc = (
        pd.Timestamp(end_dt, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
    df = df[df["published_at"].between(start_utc, end_utc)]

    # Fuzzy dedup: normalise headlines and drop near-duplicates
    df["_norm"] = df["headline"].apply(_normalise_headline)
    df = df.drop_duplicates(subset=["_norm"])
    df = df.drop(columns=["_norm"])

    df = df.sort_values("published_at", ascending=False)
    df["published_at"] = df["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_gdelt_news(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch financial news headlines for *ticker* between *start_date* and
    *end_date* by aggregating results from multiple free sources.

    Returns a DataFrame with columns:
        ticker, company, headline, source, published_at, url
    """
    ticker = ticker.upper().strip()
    company = COMPANY_MAPPING.get(ticker, ticker)
    start_dt, end_dt = _clamp_window(start_date, end_date)

    all_articles: list[dict] = []

    # ── 1. GDELT (best volume when available) ──
    print(f"[Fetcher] Fetching from GDELT for {ticker}…")
    gdelt_articles = _fetch_from_gdelt(ticker, company, start_dt, end_dt)
    all_articles.extend(gdelt_articles)

    # ── 2. Google News RSS (reliable, high volume) ──
    print(f"[Fetcher] Fetching from Google News for {ticker}…")
    google_articles = _fetch_from_google_news(ticker, company, start_dt, end_dt)
    all_articles.extend(google_articles)

    # ── 3. Yahoo Finance RSS (always-on, small batch) ──
    print(f"[Fetcher] Fetching from Yahoo Finance RSS for {ticker}…")
    yahoo_articles = _fetch_from_yahoo_rss(ticker)
    all_articles.extend(yahoo_articles)

    # ── Merge, deduplicate, filter ──
    print(f"[Fetcher] Total raw articles (all sources): {len(all_articles)}")
    df = _normalise_articles(all_articles, ticker, company, start_dt, end_dt)
    print(f"[Fetcher] After dedup & date filter: {len(df)} unique articles for {ticker}")

    return df
