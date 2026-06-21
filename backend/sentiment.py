import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
MODEL_ID = "ProsusAI/finbert"
MAX_RETRIES = 4
RETRY_DELAY = 5
BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "32"))
CACHE_PATH = Path(__file__).with_name(".sentiment_cache.json")
CACHE_VERSION = 3


@lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    return InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=120)


def _is_poisoned_cache(entries: dict[str, dict[str, float | str]]) -> bool:
    if len(entries) < 10:
        return False

    labels: list[str] = []
    scores: list[float] = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            return True
        labels.append(str(entry.get("label", "")).lower())
        try:
            scores.append(float(entry.get("score", 0.0)))
        except (TypeError, ValueError):
            return True

    return bool(labels) and set(labels) <= {"neutral"} and max(scores, default=0.0) <= 0.0


def _load_cache() -> dict[str, dict[str, float | str]]:
    try:
        if CACHE_PATH.exists():
            raw = json.loads(CACHE_PATH.read_text())
            entries = raw.get("entries", {})
            if (
                raw.get("__meta__", {}).get("version") == CACHE_VERSION
                and isinstance(entries, dict)
            ):
                if _is_poisoned_cache(entries):
                    logging.warning("Ignoring poisoned all-neutral sentiment cache.")
                    return {}
                return entries
    except Exception:
        pass
    return {}


def _save_cache(cache: dict[str, dict[str, float | str]]) -> None:
    try:
        CACHE_PATH.write_text(
            json.dumps({"__meta__": {"version": CACHE_VERSION}, "entries": cache})
        )
    except Exception:
        logging.warning("Could not persist sentiment cache.")


def _normalize_headline(headline: str) -> str:
    return " ".join(headline.lower().split())


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


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
    raise ValueError(f"Unknown sentiment label: {raw_label}")




def _query_finbert_batch(headlines: list[str]) -> list[tuple[str, float]]:
    client = _get_client()
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            results = client.text_classification(headlines)
            scored: list[tuple[str, float]] = []
            for r in results:
                scored.append((_normalize_label(r.label), float(r.score)))
            return scored
        except Exception as exc:
            last_error = str(exc)
            wait_time = RETRY_DELAY * attempt
            logging.warning(
                "Hugging Face sentiment request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {last_error}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Hugging Face sentiment endpoint failed after {MAX_RETRIES} retries: {last_error}"
    )


def sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "headline" not in df.columns:
        return df

    df = df.dropna(subset=["headline"]).copy()
    df["headline"] = df["headline"].astype(str)

    cache = _load_cache()
    ordered_headlines = df["headline"].tolist()
    normalized = [_normalize_headline(h) for h in ordered_headlines]

    seen_uncached: set[str] = set()
    uncached: list[str] = []
    for headline, key in zip(ordered_headlines, normalized):
        if key not in cache and key not in seen_uncached:
            uncached.append(headline)
            seen_uncached.add(key)

    if uncached:
        print(f"[Sentiment] Scoring {len(uncached)} uncached headlines via Hugging Face API...")
        for batch in _chunk(uncached, BATCH_SIZE):
            try:
                scored = _query_finbert_batch(batch)
                for headline, (label, score) in zip(batch, scored):
                    cache[_normalize_headline(headline)] = {
                        "label": label,
                        "score": score,
                    }
            except Exception as exc:
                _save_cache(cache)
                raise RuntimeError(
                    "Sentiment scoring failed before every headline was processed. "
                    "Completed batch results were cached; rerun after the API limit clears."
                ) from exc

        _save_cache(cache)

    sentiments: list[str] = []
    confidences: list[float] = []
    for headline in ordered_headlines:
        entry = cache.get(_normalize_headline(headline))
        if entry is None:
            raise RuntimeError("Sentiment scoring incomplete for at least one headline.")
        sentiments.append(str(entry["label"]).lower())
        confidences.append(float(entry["score"]))

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences

    return df
