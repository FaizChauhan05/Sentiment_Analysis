import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import numpy as np
from datetime import date, timedelta

from backend.gdelt_fetcher import fetch_gdelt_news
from backend.sentiment import sentiment_analysis
from backend.market_data import market_data
from backend.normalize_labels import normalize_labels
from backend.Evaluation import evaluate_predictions
from backend.aggregation import aggregate_data
from backend.keywords import extract_keywords, extract_bigrams

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiCore | Analytical Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Aura Sentiment System
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "primary":              "#4648d4", # Original SentiCore Indigo
    "primary_container":    "#6063ee",
    "primary_fixed":        "#e1e0ff",
    "secondary":            "#006c49", # Original SentiCore Green
    "secondary_container":  "#6cf8bb",
    "tertiary":             "#825100",
    "tertiary_container":   "#a36700",
    "error":                "#ba1a1a",
    "error_container":      "#ffdad6",
    "background":           "#faf8ff", # Original light background
    "surface":              "#ffffff", # Original white card surface
    "surface_container":    "#eaedff",
    "surface_container_lo": "#f2f3ff",
    "surface_container_hi": "#e2e7ff",
    "surface_container_highest": "#dae2fd",
    "on_surface":           "#131b2e", # Dark charcoal text
    "on_surface_variant":   "#464554", # Dark muted text
    "outline":              "#767586",
    "outline_variant":      "#c7c4d7", # Border color
    "inverse_surface":      "#283044",
    # Sentiment tokens (light mode)
    "pos":    "#10b981",
    "pos_bg": "#ecfdf5",
    "neu":    "#f59e0b",
    "neu_bg": "#fef3c7",
    "neg":    "#f43f5e",
    "neg_bg": "#fff1f2",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* ─── Material Symbols ─── */
.material-symbols-outlined {{
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    font-size: 24px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}}
.material-symbols-outlined.filled {{
    font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}}

/* ─── Base ─── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}}
.stApp {{
    background-color: {C['background']};
}}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {{
    background: {C['background']} !important;
    border-right: 1px solid {C['outline_variant']} !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label {{
    color: {C['on_surface']};
    font-weight: 600;
    font-size: 0.82rem;
}}

/* ─── Metric Cards ─── */
div[data-testid="metric-container"] {{
    background: {C['surface']};
    border: 1px solid {C['outline_variant']};
    border-radius: 8px;
    padding: 20px 24px;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
}}
div[data-testid="metric-container"]:hover {{
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
    transform: translateY(-2px);
}}
div[data-testid="metric-container"] label {{
    color: {C['on_surface_variant']} !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C['on_surface']} !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.02em !important;
}}

/* ─── Buttons ─── */
.stButton > button {{
    background: {C['primary']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: -0.01em;
    transition: all 0.2s ease;
    width: 100%;
    font-family: 'Inter', sans-serif;
}}
.stButton > button:hover {{
    background: {C['primary_container']};
    box-shadow: 0 4px 12px rgba(70,72,212,0.25);
    transform: translateY(-1px);
}}
.stButton > button:active {{
    transform: scale(0.97);
}}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['surface_container']};
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border-bottom: none;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {C['on_surface_variant']};
    font-weight: 600;
    font-size: 0.82rem;
    padding: 8px 16px;
    font-family: 'Inter', sans-serif;
    border-bottom: none;
}}
.stTabs [aria-selected="true"] {{
    background: {C['primary']} !important;
    color: white !important;
    border-bottom: none !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    display: none;
}}

/* ─── DataFrames ─── */
.stDataFrame {{
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid {C['outline_variant']};
}}

/* ─── HTML Tables ─── */
table {{
    width: 100% !important;
    border-collapse: collapse !important;
    background-color: {C['surface']} !important;
    color: {C['on_surface']} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid {C['outline_variant']} !important;
    margin: 16px 0 !important;
}}
th {{
    background-color: {C['surface_container_lo']} !important;
    color: {C['on_surface_variant']} !important;
    font-weight: 700 !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid {C['outline_variant']} !important;
}}
td {{
    background-color: {C['surface']} !important;
    color: {C['on_surface']} !important;
    padding: 14px 16px !important;
    border-bottom: 1px solid {C['outline_variant']} !important;
}}
tr:hover td {{
    background-color: {C['surface_container']} !important;
}}

/* ─── Headers ─── */
h1, h2, h3, h4 {{
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {C['on_surface']};
}}

/* ─── Card ─── */
.sc-card {{
    background: {C['surface']};
    border: 1px solid {C['outline_variant']};
    border-radius: 12px;
    padding: 24px;
    transition: box-shadow 0.25s ease;
}}
.sc-card:hover {{
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
}}

/* ─── Badge (pill) ─── */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 10px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.badge-pos {{ background: {C['pos_bg']}; color: {C['pos']}; }}
.badge-neu {{ background: {C['neu_bg']}; color: {C['neu']}; }}
.badge-neg {{ background: {C['neg_bg']}; color: {C['neg']}; }}

/* ─── Typography tokens ─── */
.label-caps {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: {C['outline']};
}}
.metric-lg {{
    font-family: 'Inter', sans-serif;
    font-size: 28px; font-weight: 700;
    line-height: 34px; letter-spacing: -0.02em;
    color: {C['on_surface']};
    margin: 0;
}}
.headline-md {{
    font-family: 'Inter', sans-serif;
    font-size: 20px; font-weight: 600;
    line-height: 28px; letter-spacing: -0.01em;
    color: {C['on_surface']};
    margin: 0;
}}
.body-bold {{
    font-family: 'Inter', sans-serif;
    font-size: 14px; font-weight: 600;
    line-height: 22px; color: {C['on_surface']};
}}

/* ─── Page Header ─── */
.page-hdr .tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: {C['primary']}; margin-bottom: 4px;
}}
.page-hdr .title {{
    font-family: 'Inter', sans-serif;
    font-size: 32px; font-weight: 700;
    line-height: 40px; letter-spacing: -0.02em;
    color: {C['on_surface']}; margin: 0;
}}
.page-hdr .sub {{
    font-size: 14px; color: {C['on_surface_variant']};
    margin: 4px 0 0 0;
}}

/* ─── Progress ─── */
.prog {{ height: 4px; width: 100%; background: {C['surface_container']}; border-radius: 9999px; overflow: hidden; margin-top: 12px; }}
.prog-fill {{ height: 100%; border-radius: 9999px; transition: width 0.6s ease; }}

/* ─── Icon Box ─── */
.icon-box {{
    width: 40px; height: 40px; border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 20px; transition: all 0.2s ease;
    background: {C['surface_container']}; color: {C['primary']};
}}

/* ─── Brand ─── */
.brand {{
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; padding-bottom: 16px;
    border-bottom: 1px solid {C['outline_variant']};
}}
.brand-icon {{
    width: 40px; height: 40px; background: {C['primary']};
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; color: white;
}}
.brand h2 {{ margin: 0; font-size: 1.1rem; font-weight: 700; color: {C['primary']}; letter-spacing: -0.02em; line-height: 1.2; }}
.brand p {{ margin: 0; font-size: 0.68rem; color: {C['outline']}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; line-height: 1.2; }}

/* ─── Info Box ─── */
.info-box {{
    background: {C['surface_container']}; border: 1px solid {C['outline_variant']};
    border-radius: 12px; padding: 20px; margin-top: 16px;
}}
.info-box .info-title {{ font-size: 13px; font-weight: 700; color: {C['primary']}; margin-bottom: 10px; }}
.info-box .info-text {{ font-size: 12px; color: {C['on_surface_variant']}; line-height: 1.6; }}

/* ─── Accent Metric ─── */
.accent-metric {{
    background: {C['surface']}; border: 1px solid {C['outline_variant']};
    border-radius: 12px; padding: 24px; position: relative; overflow: hidden;
}}
.accent-metric .left-bar {{
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
}}

/* ─── Hide Streamlit chrome (Deploy, MainMenu) but keep Sidebar toggle ─── */
[data-testid="stDeployButton"] {{ display: none !important; }}
#MainMenu {{ visibility: hidden !important; }}
footer {{ visibility: hidden !important; }}
header, [data-testid="stHeader"] {{
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
    pointer-events: none !important;
    height: 3.5rem !important;
}}
[data-testid="collapsedControl"] {{
    pointer-events: auto !important;
    background-color: {C['surface']} !important;
    border-radius: 8px !important;
    border: 1px solid {C['outline_variant']} !important;
    margin-left: 16px !important;
    margin-top: 8px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}}
hr {{ border-color: {C['outline_variant']}; opacity: 0.5; }}

/* ─── Selectbox / Radio ─── */
.stSelectbox > div > div {{ border-color: {C['outline_variant']}; border-radius: 8px; }}
section[data-testid="stSidebar"] .stRadio label {{
    color: {C['on_surface']} !important;
    font-weight: 600;
    font-size: 0.86rem;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _badge(sentiment: str) -> str:
    s = sentiment.lower().strip()
    cls = "badge-pos" if s == "positive" else ("badge-neg" if s == "negative" else "badge-neu")
    return f'<span class="badge {cls}">{s.upper()}</span>'


def _page_header(tag: str, title: str, sub: str) -> str:
    return f'<div class="page-hdr" style="margin-bottom:28px;"><p class="tag">{tag}</p><h2 class="title">{title}</h2><p class="sub">{sub}</p></div>'


def _icon(name: str, size: int = 20, color: str = "") -> str:
    style = f"font-size:{size}px;"
    if color:
        style += f"color:{color};"
    return f'<span class="material-symbols-outlined" style="{style}">{name}</span>'


def _metric_card(icon_name: str, label: str, value: str,
                 change: str = "", up: bool = True, pct: int = 0) -> str:
    clr = C["pos"] if up else C["neg"]
    arrow = "arrow_upward" if up else "arrow_downward"
    ch = f'<div style="display:flex;align-items:center;gap:3px;color:{clr};font-size:12px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{_icon(arrow, 14, clr)} {change}</div>' if change else ""
    bar = f'<div class="prog"><div class="prog-fill" style="width:{pct}%;background:{C["primary"]};"></div></div>' if pct > 0 else ""
    return f'<div class="sc-card"><div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;"><div class="icon-box">{_icon(icon_name, 22, C["primary"])}</div>{ch}</div><p style="color:{C["on_surface_variant"]};font-weight:600;font-size:13px;margin:0 0 4px 0;">{label}</p><p class="metric-lg">{value}</p>{bar}</div>'


def _plotly_base(title: str = "", height: int = 350) -> dict:
    return dict(
        title=title,
        plot_bgcolor=C["surface"], paper_bgcolor=C["surface"],
        font=dict(family="Inter, sans-serif", color=C["on_surface"]),
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        height=height,
        xaxis=dict(showgrid=True, gridcolor=C["outline_variant"], griddash="dot",
                   linecolor=C["outline_variant"],
                   tickfont=dict(family="JetBrains Mono, monospace", size=10, color=C["outline"])),
        yaxis=dict(showgrid=True, gridcolor=C["outline_variant"], griddash="dot",
                   linecolor=C["outline_variant"],
                   tickfont=dict(family="JetBrains Mono, monospace", size=10, color=C["outline"])),
        hoverlabel=dict(bgcolor=C["surface_container_highest"], font_color=C["on_surface"],
                        font_family="Inter, sans-serif", font_size=12,
                        bordercolor=C["outline"]),
        legend=dict(font=dict(family="JetBrains Mono, monospace", size=10),
                    bgcolor="rgba(0,0,0,0)"),
    )


def _table_header(*cols, align_last_right=False):
    ths = ""
    for i, col in enumerate(cols):
        align = "right" if (align_last_right and i == len(cols) - 1) else "left"
        ths += f'<th style="text-align:{align};padding:12px 16px;" class="label-caps">{col}</th>'
    return f"""<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:'Inter',sans-serif;">
    <thead><tr style="background:{C['surface_container_lo']};border-bottom:1px solid {C['outline_variant']};">{ths}</tr></thead><tbody>"""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="brand">
        <div class="brand-icon">{_icon('analytics', 22, 'white')}</div>
        <div><h2>SentiCore</h2><p>Analytical Intelligence</p></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Analysis", "Mentions", "Reports"],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown("---")
    st.markdown(f'<p class="label-caps" style="margin-bottom:12px;">ANALYSIS SETTINGS</p>',
                unsafe_allow_html=True)

    TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "AMD", "JPM"]
    TICKER_LABELS = {
        "AAPL": "Apple  (AAPL)",       "TSLA": "Tesla  (TSLA)",
        "NVDA": "NVIDIA  (NVDA)",      "MSFT": "Microsoft  (MSFT)",
        "AMZN": "Amazon  (AMZN)",      "GOOGL": "Google  (GOOGL)",
        "META": "Meta  (META)",        "NFLX": "Netflix  (NFLX)",
        "AMD":  "AMD  (AMD)",          "JPM":  "JPMorgan  (JPM)",
    }

    ticker = st.selectbox("Stock Ticker", TICKERS,
                          format_func=lambda t: TICKER_LABELS[t], index=0)

    default_end   = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=7)
    start_date = st.date_input("Start Date", value=default_start)
    end_date   = st.date_input("End Date",   value=default_end)

    # NewsAPI free tier and model size disclaimer card
    st.markdown(f"""
    <div style="background:{C['surface_container_lo']}; border: 1px solid {C['outline_variant']};
                border-radius: 8px; padding: 12px; margin-top: 10px; margin-bottom: 6px;">
        <p style="font-size: 11px; font-weight: 700; color: {C['tertiary']}; margin: 0 0 6px 0;
                  display: flex; align-items: center; gap: 6px; font-family: 'Inter', sans-serif;">
            {_icon('info', 16, C['tertiary'])} System Limitations
        </p>
        <ul style="font-size: 11.5px; color: {C['on_surface_variant']}; margin: 0; padding-left: 16px; line-height: 1.4; font-family: 'Inter', sans-serif;">
            <li style="margin-bottom: 6px;"><strong>API Constraints:</strong> NewsAPI Free Tier limits searches to articles published within <strong>30 days of today</strong> (e.g. from 1 month ago to today/yesterday).</li>
            <li><strong>Model Performance:</strong> Due to hosting costs and local model size constraints, FinBERT prediction latency and accuracy will vary depending on the volume of headlines collected and processed.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_btn = st.button("Run Analysis", key="run_analysis", use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
        <p class="info-title">How it works</p>
        <p class="info-text">
            1. News headlines fetched via NewsAPI<br>
            2. FinBERT classifies each headline<br>
            3. Scores aggregated per trading day<br>
            4. Market data from Yahoo Finance<br>
            5. Predictions vs. actual movements
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:8px;">
        <div style="display:flex;align-items:center;gap:8px;color:{C['on_surface_variant']};font-size:13px;font-weight:600;">
            {_icon('contact_support', 18, C['on_surface_variant'])} Support
        </div>
        <div style="display:flex;align-items:center;gap:8px;color:{C['on_surface_variant']};font-size:13px;font-weight:600;">
            {_icon('logout', 18, C['on_surface_variant'])} Log out
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (session-state cached)
# ══════════════════════════════════════════════════════════════════════════════
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if run_btn:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    bar = st.progress(0, text="Initializing...")
    try:
        bar.progress(10, text="Fetching news articles...")
        news_df = fetch_gdelt_news(ticker, str(start_date), str(end_date))
        if news_df.empty:
            st.error(f"No articles found for {ticker} between {start_date} and {end_date}. "
                     "Try a wider date range or a different ticker.")
            bar.empty(); st.stop()

        bar.progress(30, text="Running FinBERT sentiment analysis...")
        sentiment_df = sentiment_analysis(news_df)

        bar.progress(55, text="Aggregating sentiment scores...")
        aggregated_df = aggregate_data(sentiment_df)

        bar.progress(70, text="Fetching market data...")
        market_df = market_data(aggregated_df)
        if market_df.empty:
            st.warning("Market data could not be retrieved for the selected period.")
            bar.empty(); st.stop()

        bar.progress(88, text="Evaluating predictions...")
        normalized_df = normalize_labels(market_df)
        results = evaluate_predictions(normalized_df)

        bar.progress(100, text="Done")
        bar.empty()

        # Persist
        st.session_state.update(
            analysis_done=True, ticker=ticker,
            start_date=start_date, end_date=end_date,
            sentiment_df=sentiment_df, aggregated_df=aggregated_df,
            normalized_df=normalized_df, results=results,
        )
    except Exception as exc:
        bar.empty()
        st.error(f"An error occurred during analysis: {exc}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
def _empty_overview():
    st.markdown(f"""
    <div style="text-align:center;padding:80px 20px;">
        <div style="width:64px;height:64px;background:{C['surface_container']};border-radius:16px;
                    display:inline-flex;align-items:center;justify-content:center;margin-bottom:20px;">
            {_icon('analytics', 32, C['primary'])}
        </div>
        <h3 style="font-size:22px;font-weight:700;margin-bottom:8px;color:{C['on_surface']};">Welcome to SentiCore</h3>
        <p style="color:{C['on_surface_variant']};font-size:14px;max-width:460px;margin:0 auto 28px auto;line-height:1.6;">
            Configure the stock ticker and date range in the sidebar, then click
            <strong>Run Analysis</strong> to generate insights.
        </p>
        <div style="display:inline-flex;gap:20px;flex-wrap:wrap;justify-content:center;">
            <div style="background:{C['surface_container']};border-radius:8px;padding:16px 20px;text-align:left;width:180px;">
                <div style="margin-bottom:8px;">{_icon('target', 24, C['primary'])}</div>
                <p style="font-weight:600;font-size:13px;margin:0 0 4px 0;color:{C['on_surface']};">Accuracy</p>
                <p style="color:{C['outline']};font-size:12px;margin:0;">FinBERT vs market</p>
            </div>
            <div style="background:{C['pos_bg']};border-radius:8px;padding:16px 20px;text-align:left;width:180px;">
                <div style="margin-bottom:8px;">{_icon('trending_up', 24, C['pos'])}</div>
                <p style="font-weight:600;font-size:13px;margin:0 0 4px 0;color:{C['pos']};">Trends</p>
                <p style="color:{C['outline']};font-size:12px;margin:0;">Sentiment over time</p>
            </div>
            <div style="background:{C['neu_bg']};border-radius:8px;padding:16px 20px;text-align:left;width:180px;">
                <div style="margin-bottom:8px;">{_icon('newspaper', 24, C['neu'])}</div>
                <p style="font-weight:600;font-size:13px;margin:0 0 4px 0;color:{C['neu']};">Mentions</p>
                <p style="color:{C['outline']};font-size:12px;margin:0;">Article-level detail</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _empty_analysis():
    st.markdown(f"""
    <div style="text-align:center;padding:80px 20px;">
        <div style="width:64px;height:64px;background:{C['surface_container']};border-radius:16px;
                    display:inline-flex;align-items:center;justify-content:center;margin-bottom:20px;">
            {_icon('psychology', 32, C['primary'])}
        </div>
        <h3 style="font-size:22px;font-weight:700;margin-bottom:8px;color:{C['on_surface']};">Sentiment Analysis Detail</h3>
        <p style="color:{C['on_surface_variant']};font-size:14px;max-width:460px;margin:0 auto 28px auto;line-height:1.6;">
            Waiting for data analysis. Please select a ticker and run the analysis in the sidebar to populate these metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)


def _empty_mentions():
    st.markdown(f"""
    <div style="text-align:center;padding:80px 20px;">
        <div style="width:64px;height:64px;background:{C['surface_container']};border-radius:16px;
                    display:inline-flex;align-items:center;justify-content:center;margin-bottom:20px;">
            {_icon('forum', 32, C['primary'])}
        </div>
        <h3 style="font-size:22px;font-weight:700;margin-bottom:8px;color:{C['on_surface']};">Mentions Feed</h3>
        <p style="color:{C['on_surface_variant']};font-size:14px;max-width:460px;margin:0 auto 28px auto;line-height:1.6;">
            Waiting for news fetch. Please run the analysis in the sidebar to view the article feed.
        </p>
    </div>
    """, unsafe_allow_html=True)


def _empty_reports():
    st.markdown(f"""
    <div style="text-align:center;padding:80px 20px;">
        <div style="width:64px;height:64px;background:{C['surface_container']};border-radius:16px;
                    display:inline-flex;align-items:center;justify-content:center;margin-bottom:20px;">
            {_icon('assessment', 32, C['primary'])}
        </div>
        <h3 style="font-size:22px;font-weight:700;margin-bottom:8px;color:{C['on_surface']};">Reports & Analytics</h3>
        <p style="color:{C['on_surface_variant']};font-size:14px;max-width:460px;margin:0 auto 28px auto;line-height:1.6;">
            Waiting for evaluation metrics. Run the analysis to generate reports and confusion matrices.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def pg_overview():
    st.markdown(_page_header("Dashboard", "Analytics Overview",
                             "Real-time sentiment monitoring and market correlation metrics."),
                unsafe_allow_html=True)
    if not st.session_state.analysis_done:
        _empty_overview(); return

    R = st.session_state.results
    sdf = st.session_state.sentiment_df
    adf = st.session_state.aggregated_df

    total = R["total_articles"]
    acc   = R["accuracy"]
    close = R["latest_market_data"]["close"]

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_metric_card("forum", "Total Articles", f"{total:,}",
                                 pct=min(100, total * 2)), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_card("sentiment_satisfied", "Sentiment Score",
                                 f"{acc:.1f}%", f"{acc:.1f}%", acc > 50,
                                 int(acc)), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_card("check_circle", "Correct",
                                 str(R["correct_predictions"]),
                                 pct=int(R["correct_predictions"] / max(total, 1) * 100)),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(_metric_card("payments", "Last Close",
                                 f"${close:.2f}"), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Charts Row ──
    ch_col, dn_col = st.columns([2, 1])

    with ch_col:
        st.markdown('<div class="sc-card">', unsafe_allow_html=True)
        st.markdown('<p class="headline-md">Sentiment Trends</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:13px;color:{C["on_surface_variant"]};margin-top:-4px;">Score variation over the analysis period</p>', unsafe_allow_html=True)

        ap = adf.copy()
        ap["date"] = pd.to_datetime(ap["date"]).dt.date

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ap["date"], y=ap["Aggregate_Score"], mode="lines+markers",
            line=dict(color=C["primary"], width=3, shape="spline"),
            marker=dict(size=6, color=C["primary"], line=dict(width=2, color="white")),
            fill="tozeroy", fillcolor="rgba(70,72,212,0.08)", name="Score",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>"))
        fig.add_hline(y=0.3, line_dash="dash", line_color=C["pos"], opacity=0.5,
                      annotation_text="Bullish", annotation_font_color=C["pos"])
        fig.add_hline(y=-0.3, line_dash="dash", line_color=C["neg"], opacity=0.5,
                      annotation_text="Bearish", annotation_font_color=C["neg"])
        fig.add_hline(y=0, line_dash="dot", line_color=C["outline"], opacity=0.3)
        ly = _plotly_base(height=320); ly["showlegend"] = False
        fig.update_layout(**ly)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with dn_col:
        st.markdown('<div class="sc-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<p class="headline-md">Sentiment Mix</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:13px;color:{C["on_surface_variant"]};margin-top:-4px;">Distribution breakdown</p>', unsafe_allow_html=True)

        sc = sdf["Sentiment"].value_counts()
        labs = sc.index.tolist(); vals = sc.values.tolist()
        cmap = {"positive": C["pos"], "negative": C["neg"], "neutral": C["neu"]}
        cols = [cmap.get(l, C["outline"]) for l in labs]
        dominant_pct = int(vals[0] / sum(vals) * 100) if vals else 0

        fd = go.Figure(data=[go.Pie(
            labels=[l.capitalize() for l in labs], values=vals, hole=0.65,
            marker=dict(colors=cols, line=dict(color="white", width=3)),
            textinfo="percent",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>")])
        fd.update_layout(
            plot_bgcolor=C["surface"], paper_bgcolor=C["surface"],
            margin=dict(l=10, r=10, t=10, b=10), height=230, showlegend=False,
            annotations=[dict(
                text=f"<b>{dominant_pct}%</b><br><span style='font-size:9px;color:{C['outline']}'>Dominant</span>",
                x=0.5, y=0.5, font_size=22, font_family="Inter", font_color=C["on_surface"], showarrow=False)])
        st.plotly_chart(fd, use_container_width=True, config={"displayModeBar": False})

        for lbl in labs:
            clr = cmap.get(lbl, C["outline"]); pct = int(sc[lbl] / sum(vals) * 100)
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="width:10px;height:10px;border-radius:50%;background:{clr};"></div>
                    <span class="body-bold">{lbl.capitalize()}</span>
                </div><span class="label-caps">{pct}%</span></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Recent Mentions Table ──
    st.markdown('<div class="sc-card">', unsafe_allow_html=True)
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown('<p class="headline-md">Recent Mentions</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:13px;color:{C["on_surface_variant"]};">Real-time engagement feed</p>', unsafe_allow_html=True)
    with hc2:
        st.markdown(f'<p style="text-align:right;"><a style="color:{C["primary"]};font-weight:600;font-size:13px;text-decoration:none;" href="#">View All Activity</a></p>', unsafe_allow_html=True)

    ddf = sdf[["headline", "source", "published_at", "Sentiment", "Confidence"]].head(10).copy()
    ddf["published_at"] = pd.to_datetime(ddf["published_at"]).dt.strftime("%Y-%m-%d %H:%M")
    ddf["Confidence"] = ddf["Confidence"].apply(lambda x: f"{float(x)*100:.1f}%")

    html = _table_header("Headline", "Source", "Sentiment", "Confidence", align_last_right=True)
    for _, r in ddf.iterrows():
        html += f"""<tr style="border-bottom:1px solid {C['outline_variant']};">
            <td style="padding:14px 16px;max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:{C['on_surface']};">{r['headline']}</td>
            <td style="padding:14px 16px;color:{C['on_surface_variant']};">{r['source']}</td>
            <td style="padding:14px 16px;">{_badge(r['Sentiment'])}</td>
            <td style="padding:14px 16px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;color:{C['on_surface']};">{r['Confidence']}</td></tr>"""
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def pg_analysis():
    st.markdown(_page_header("Analysis", "Sentiment Analysis",
                             "Cross-platform perception intelligence across the analysis period."),
                unsafe_allow_html=True)
    if not st.session_state.analysis_done:
        _empty_analysis(); return

    R   = st.session_state.results
    sdf = st.session_state.sentiment_df
    tk  = st.session_state.ticker

    sc = sdf["Sentiment"].value_counts()
    tot = len(sdf)
    pos_p = round(sc.get("positive", 0) / max(tot, 1) * 100, 1)
    neu_p = round(sc.get("neutral",  0) / max(tot, 1) * 100, 1)
    neg_p = round(sc.get("negative", 0) / max(tot, 1) * 100, 1)

    # ── Breakdown Cards ──
    c1, c2, c3 = st.columns(3)
    for col, (lbl, pct, clr, bg, ico) in zip(
        [c1, c2, c3],
        [("Positive", pos_p, C["pos"], C["pos_bg"], "mood"),
         ("Neutral",  neu_p, C["neu"], C["neu_bg"], "sentiment_neutral"),
         ("Negative", neg_p, C["neg"], C["neg_bg"], "mood_bad")]):
        with col:
            st.markdown(f"""
            <div class="sc-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                    <div style="width:40px;height:40px;border-radius:8px;background:{bg};
                                display:flex;align-items:center;justify-content:center;">
                        {_icon(ico, 22, clr)}
                    </div>
                    <span class="badge" style="background:{bg};color:{clr};">{lbl.upper()}</span>
                </div>
                <p class="metric-lg">{pct}%</p>
                <p style="color:{C['on_surface_variant']};font-size:13px;margin-top:4px;">
                    {sc.get(lbl.lower(), 0)} articles out of {tot}
                </p>
                <div class="prog" style="margin-top:16px;">
                    <div class="prog-fill" style="width:{pct}%;background:{clr};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Keyword Cloud + AI Summary ──
    kw_col, ai_col = st.columns([3, 2])

    with kw_col:
        st.markdown('<div class="sc-card">', unsafe_allow_html=True)
        kws = extract_keywords(sdf["headline"], top_n=12)
        vol = sum(c for _, c in kws) if kws else 0
        st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <p class="headline-md" style="margin:0;">Trending Keywords</p>
            <span class="label-caps">Volume: {vol:,}</span></div>""", unsafe_allow_html=True)

        cloud = f'<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:12px;padding:24px;background:{C["surface_container_lo"]};border-radius:8px;min-height:260px;">'
        sizes = [28, 24, 22, 20, 18, 16, 15, 14, 13, 13, 12, 12]
        clrs  = [C["primary"], C["on_surface_variant"], C["pos"], C["tertiary"],
                 C["primary_container"], C["neg"], C["outline"], C["on_surface"],
                 C["primary_fixed"].replace("#e1e0ff", C["primary"]),
                 C["on_surface_variant"], C["outline"], C["tertiary_container"]]
        for i, (word, cnt) in enumerate(kws):
            sz = sizes[i] if i < len(sizes) else 12
            cl = clrs[i % len(clrs)]
            fw = "700" if i < 4 else "600"
            cloud += f'<span style="font-size:{sz}px;font-weight:{fw};color:{cl};cursor:default;transition:all 0.2s;font-family:Inter,sans-serif;">{word.capitalize()}</span>'
        cloud += '</div>'
        st.markdown(cloud, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ai_col:
        dominant = "positive" if pos_p > neg_p else "negative"
        dom_pct  = max(pos_p, neg_p)
        st.markdown(f"""
        <div style="background:{C['surface_container']};border:1px solid {C['outline_variant']};
                    border-radius:12px;padding:24px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                {_icon('auto_awesome', 20, C['primary'])}
                <span class="body-bold">AI Summary</span>
            </div>
            <p style="color:{C['on_surface_variant']};font-size:13px;line-height:1.7;">
                Analysis of <strong>{tot}</strong> articles for <strong>{tk}</strong> shows an overall
                <span style="color:{C['pos'] if dominant == 'positive' else C['neg']};font-weight:700;">
                    {dominant}
                </span>
                sentiment bias at <strong>{dom_pct:.1f}%</strong>.
                The model achieved <strong>{R['accuracy']}%</strong> accuracy in predicting market movements.
            </p>
            <div style="margin-top:20px;padding-top:16px;border-top:1px solid {C['outline_variant']};">
                <p class="label-caps" style="margin-bottom:8px;">Model Confidence</p>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="flex:1;height:6px;background:{C['surface_container_hi']};border-radius:9999px;overflow:hidden;">
                        <div style="height:100%;width:{R['accuracy']}%;background:{C['primary']};border-radius:9999px;"></div>
                    </div>
                    <span class="body-bold">{R['accuracy']:.0f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Per-class metrics from enhanced Evaluation
        pcm = R.get("per_class_metrics", {})
        if pcm:
            st.markdown(f"""
            <div class="sc-card">
                <p class="body-bold" style="margin-bottom:16px;">Per-Class Performance</p>
            """, unsafe_allow_html=True)
            for cls_name, m in pcm.items():
                clr = C["pos"] if cls_name == "up" else (C["neg"] if cls_name == "down" else C["neu"])
                st.markdown(f"""
                <div style="display:flex;gap:12px;padding:6px 0;border-bottom:1px solid {C['outline_variant']};">
                    <div style="flex:1;">
                        <span style="font-weight:600;color:{clr};text-transform:capitalize;">{cls_name}</span>
                    </div>
                    <div style="text-align:right;">
                        <span class="label-caps" style="font-size:10px;">P {m['precision']}%  R {m['recall']}%  F1 {m['f1']}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Source Breakdown Chart ──
    st.markdown('<div class="sc-card">', unsafe_allow_html=True)
    st.markdown('<p class="headline-md">Sentiment by Source</p>', unsafe_allow_html=True)

    src_sent = sdf.groupby(["source", "Sentiment"]).size().unstack(fill_value=0)
    top_src = src_sent.sum(axis=1).nlargest(8).index
    src_data = src_sent.loc[top_src]

    fb = go.Figure()
    for s, cl in [("positive", C["pos"]), ("neutral", C["neu"]), ("negative", C["neg"])]:
        if s in src_data.columns:
            fb.add_trace(go.Bar(y=src_data.index, x=src_data[s], name=s.capitalize(),
                                orientation="h", marker=dict(color=cl, cornerradius=4)))
    ly = _plotly_base(height=320)
    ly["barmode"] = "stack"
    ly["yaxis"]["tickfont"] = dict(family="Inter", size=11, color=C["on_surface"])
    ly["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(family="JetBrains Mono, monospace", size=10))
    fb.update_layout(**ly)
    st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── High Impact Mentions ──
    st.markdown('<div class="sc-card">', unsafe_allow_html=True)
    st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <p class="headline-md" style="margin:0;">High Impact Mentions</p>
        <a style="color:{C['primary']};font-weight:600;font-size:13px;text-decoration:none;" href="#">View All Source Data</a>
    </div>""", unsafe_allow_html=True)

    hi = sdf[["headline", "source", "published_at", "Sentiment", "Confidence"]].copy()
    hi["_conf"] = hi["Confidence"].apply(float)
    hi = hi.nlargest(10, "_conf")
    hi["published_at"] = pd.to_datetime(hi["published_at"]).dt.strftime("%Y-%m-%d %H:%M")
    hi["Confidence"] = hi["_conf"].apply(lambda x: f"{x*100:.1f}%")

    html = _table_header("Source", "Content", "Sentiment", "Confidence", "Time", align_last_right=True)
    for _, r in hi.iterrows():
        html += f"""<tr style="border-bottom:1px solid {C['outline_variant']};">
            <td style="padding:14px 16px;font-weight:600;white-space:nowrap;color:{C['on_surface']};">{r['source']}</td>
            <td style="padding:14px 16px;max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:{C['on_surface']};">{r['headline']}</td>
            <td style="padding:14px 16px;">{_badge(r['Sentiment'])}</td>
            <td style="padding:14px 16px;font-family:'JetBrains Mono',monospace;font-size:12px;color:{C['on_surface']};">{r['Confidence']}</td>
            <td style="padding:14px 16px;text-align:right;color:{C['outline']};font-size:12px;">{r['published_at']}</td></tr>"""
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MENTIONS FEED
# ══════════════════════════════════════════════════════════════════════════════
def pg_mentions():
    st.markdown(_page_header("Feed", "Recent Mentions Feed",
                             "Real-time analysis of article-level sentiment data."),
                unsafe_allow_html=True)
    if not st.session_state.analysis_done:
        _empty_mentions(); return

    sdf = st.session_state.sentiment_df

    # ── Filters ──
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        q = st.text_input("Search", placeholder="Filter by keyword...", label_visibility="collapsed")
    with fc2:
        sf = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"], label_visibility="collapsed")
    with fc3:
        so = st.selectbox("Sort", ["Latest First", "Oldest First", "Highest Confidence"], label_visibility="collapsed")

    f = sdf.copy()
    if q:
        f = f[f["headline"].str.contains(q, case=False, na=False)]
    if sf != "All":
        f = f[f["Sentiment"] == sf.lower()]

    f["_dt"] = pd.to_datetime(f["published_at"])
    f["_cf"] = f["Confidence"].apply(float)
    if so == "Latest First":
        f = f.sort_values("_dt", ascending=False)
    elif so == "Oldest First":
        f = f.sort_values("_dt", ascending=True)
    else:
        f = f.sort_values("_cf", ascending=False)

    total_f = len(f)
    page_sz = 25
    showing = min(page_sz, total_f)

    st.markdown(f"""<p style="color:{C['on_surface_variant']};font-size:13px;margin-bottom:16px;">
        Showing <strong style="color:{C['on_surface']};">{showing}</strong>
        of <strong style="color:{C['on_surface']};">{total_f:,}</strong> mentions</p>""", unsafe_allow_html=True)

    # ── Table ──
    st.markdown('<div class="sc-card" style="padding:0;overflow:hidden;">', unsafe_allow_html=True)
    html = f"""<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:'Inter',sans-serif;">
        <thead><tr style="background:{C['surface_container']};border-bottom:1px solid {C['outline_variant']};">
            <th style="text-align:left;padding:14px 16px;" class="label-caps">Source</th>
            <th style="text-align:left;padding:14px 16px;" class="label-caps">Mention Content</th>
            <th style="text-align:left;padding:14px 16px;" class="label-caps">Sentiment</th>
            <th style="text-align:right;padding:14px 16px;" class="label-caps">Time</th>
        </tr></thead><tbody>"""

    for _, r in f.head(page_sz).iterrows():
        ts = r["_dt"].strftime("%Y-%m-%d %H:%M")
        html += f"""<tr style="border-bottom:1px solid {C['outline_variant']};">
            <td style="padding:14px 16px;"><p style="font-weight:600;margin:0;">{r['source']}</p>
                <p style="font-size:11px;color:{C['outline']};margin:2px 0 0 0;">News</p></td>
            <td style="padding:14px 16px;max-width:500px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r['headline']}</td>
            <td style="padding:14px 16px;">{_badge(r['Sentiment'])}</td>
            <td style="padding:14px 16px;text-align:right;color:{C['on_surface_variant']};font-size:12px;">{ts}</td></tr>"""
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Bottom Widgets ──
    w1, w2, w3 = st.columns(3)

    sc = sdf["Sentiment"].value_counts()
    tot = len(sdf)
    pos_p = round(sc.get("positive", 0) / max(tot, 1) * 100, 1)

    with w1:
        tr_clr = C["pos"] if pos_p > 50 else C["neg"]
        tr_ico = "trending_up" if pos_p > 50 else "trending_down"
        st.markdown(f"""<div class="sc-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span class="label-caps">SENTIMENT TREND</span>
                <span style="color:{tr_clr};font-weight:700;font-size:13px;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:4px;">
                    {_icon(tr_ico, 16, tr_clr)} {pos_p}%</span>
            </div>
            <p style="color:{C['on_surface_variant']};font-size:13px;margin-top:8px;">
                {'Overall mood is positive.' if pos_p > 50 else 'Overall mood needs monitoring.'}</p>
        </div>""", unsafe_allow_html=True)

    with w2:
        kws = extract_keywords(sdf["headline"], top_n=6)
        tags = f'<div style="display:flex;flex-wrap:wrap;gap:6px;">'
        for w, _ in kws:
            tags += f'<span style="background:{C["surface_container"]};padding:4px 10px;border-radius:4px;font-weight:600;font-size:12px;">{w.capitalize()}</span>'
        tags += '</div>'
        st.markdown(f"""<div class="sc-card">
            <p class="label-caps" style="margin-bottom:12px;">HOT TOPICS</p>
            {tags}</div>""", unsafe_allow_html=True)

    with w3:
        dom = int(pos_p)
        st.markdown(f"""<div class="sc-card">
            <p class="label-caps" style="margin-bottom:12px;">SHARE OF VOICE</p>
            <div style="display:flex;align-items:center;gap:16px;">
                <div style="width:48px;height:48px;">
                    <svg viewBox="0 0 36 36" style="transform:rotate(-90deg);width:100%;height:100%;">
                        <circle cx="18" cy="18" r="16" fill="none" stroke="{C['surface_container']}" stroke-width="4"></circle>
                        <circle cx="18" cy="18" r="16" fill="none" stroke="{C['primary']}" stroke-width="4"
                                stroke-dasharray="{dom} 100"></circle>
                    </svg>
                </div>
                <div><p class="metric-lg" style="font-size:22px;">{dom}%</p>
                    <p style="color:{C['on_surface_variant']};font-size:12px;">Dominant Sentiment</p></div>
            </div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REPORTS & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def pg_reports():
    st.markdown(_page_header("Reports", "Analytics Reports",
                             "Comprehensive sentiment and prediction performance dashboard."),
                unsafe_allow_html=True)
    if not st.session_state.analysis_done:
        _empty_reports(); return

    R   = st.session_state.results
    sdf = st.session_state.sentiment_df
    adf = st.session_state.aggregated_df
    ndf = st.session_state.normalized_df

    sc  = sdf["Sentiment"].value_counts()
    tot = len(sdf)
    pos_p = round(sc.get("positive", 0) / max(tot, 1) * 100, 1)
    neg_p = round(sc.get("negative", 0) / max(tot, 1) * 100, 1)

    # ── Accent Metric Cards ──
    m1, m2, m3 = st.columns(3)
    for col, (label, val, delta, clr) in zip(
        [m1, m2, m3],
        [("Total Articles", f"{tot:,}", "Analyzed", C["pos"]),
         ("Positive Share", f"{pos_p}%", f"+{pos_p:.1f}%", C["primary"]),
         ("Negative Volatility", f"{neg_p}%", f"-{neg_p:.1f}%", C["neg"])]):
        with col:
            st.markdown(f"""
            <div class="accent-metric">
                <div class="left-bar" style="background:{clr};"></div>
                <p class="label-caps" style="margin-bottom:4px;">{label}</p>
                <div style="display:flex;align-items:flex-end;gap:8px;">
                    <span class="metric-lg">{val}</span>
                    <span style="font-size:12px;font-weight:700;color:{clr};margin-bottom:6px;">{delta}</span>
                </div>
                <div class="prog" style="margin-top:12px;">
                    <div class="prog-fill" style="width:{pos_p if 'Positive' in label else (neg_p if 'Negative' in label else 74)}%;background:{clr};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Tabs ──
    tabs = st.tabs(["Sentiment Over Time", "Confusion Matrix", "Distribution"])

    with tabs[0]:
        st.markdown('<div class="sc-card">', unsafe_allow_html=True)
        tmp = sdf.copy()
        tmp["date"] = pd.to_datetime(tmp["published_at"], utc=True).dt.normalize()
        daily = tmp.groupby(["date", "Sentiment"]).size().unstack(fill_value=0).reset_index()
        daily["date"] = daily["date"].dt.date

        fa = go.Figure()
        for s, cl, fl in [("positive", C["pos"], C["pos_bg"]),
                           ("neutral", C["neu"], C["neu_bg"]),
                           ("negative", C["neg"], C["neg_bg"])]:
            if s in daily.columns:
                fa.add_trace(go.Scatter(
                    x=daily["date"], y=daily[s], mode="lines",
                    name=s.capitalize(), line=dict(color=cl, width=2, shape="spline"),
                    stackgroup="one", fillcolor=fl,
                    hovertemplate=f"<b>{s.capitalize()}</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>"))
        ly = _plotly_base(height=380)
        ly["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(family="JetBrains Mono, monospace", size=10))
        fa.update_layout(**ly)
        st.plotly_chart(fa, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="sc-card">', unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix
        labels = ["up", "down", "unchanged"]
        cm = confusion_matrix(ndf["Movement"], ndf["Predictions"], labels=labels)
        fc = px.imshow(
            cm, labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Up", "Down", "Unchanged"], y=["Up", "Down", "Unchanged"],
            color_continuous_scale=[[0.0, "#e1e0ff"], [0.5, "#6063ee"], [1.0, "#4648d4"]],
            text_auto=True)
        fc.update_layout(plot_bgcolor=C["surface"], paper_bgcolor=C["surface"],
                         font=dict(family="Inter", color=C["on_surface"]),
                         margin=dict(l=20, r=20, t=40, b=20), height=380,
                         xaxis=dict(tickfont=dict(family="JetBrains Mono", size=11)),
                         yaxis=dict(tickfont=dict(family="JetBrains Mono", size=11)))
        fc.update_traces(textfont=dict(size=18, color="white", family="Inter"))
        st.plotly_chart(fc, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="sc-card">', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        for col, (title, dist) in zip([d1, d2],
            [("Actual Movement", R["movement_distribution"]),
             ("Predicted Movement", R["prediction_distribution"])]):
            with col:
                st.markdown(f'<p class="headline-md">{title}</p>', unsafe_allow_html=True)
                fg = go.Figure(data=[go.Bar(
                    x=list(dist.keys()), y=list(dist.values()),
                    marker=dict(color=[C["pos"], C["neg"], C["neu"]], cornerradius=6),
                    text=list(dist.values()), textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=12, color=C["on_surface"]))])
                ly = _plotly_base(height=300); ly["showlegend"] = False
                ly["xaxis"]["tickfont"] = dict(family="JetBrains Mono", size=11, color=C["on_surface"])
                fg.update_layout(**ly)
                st.plotly_chart(fg, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Market Data ──
    st.markdown('<div class="sc-card">', unsafe_allow_html=True)
    st.markdown('<p class="headline-md">Latest Market Data</p>', unsafe_allow_html=True)
    lmd = R["latest_market_data"]
    items = [("Open", f"${lmd['open']:.2f}"), ("High", f"${lmd['high']:.2f}"),
             ("Low", f"${lmd['low']:.2f}"), ("Close", f"${lmd['close']:.2f}"),
             ("Volume", f"{lmd['volume']:,}")]
    mh = '<div style="display:flex;gap:16px;flex-wrap:wrap;">'
    for lab, val in items:
        mh += f"""<div style="flex:1;min-width:140px;background:{C['surface_container_lo']};
                    border-radius:8px;padding:16px;">
            <p class="label-caps" style="margin-bottom:4px;">{lab}</p>
            <p style="font-size:20px;font-weight:700;margin:0;">{val}</p></div>"""
    mh += '</div>'
    st.markdown(mh, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
{"Overview": pg_overview, "Analysis": pg_analysis,
 "Mentions": pg_mentions, "Reports": pg_reports}[page]()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<p style="text-align:center;color:{C['outline']};font-size:11px;
          font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
    Powered by FinBERT  |  NewsAPI  |  Yahoo Finance  |  Streamlit
</p>""", unsafe_allow_html=True)
