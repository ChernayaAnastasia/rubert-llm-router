"""Tavily search with on-disk cache for Stage 4 agent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

try:
    from tavily import TavilyClient
except Exception as e:  # pragma: no cover
    TavilyClient = None  # type: ignore
    _tavily_import_error = e

from utils.config import SEARCH_CACHE_DIR

logger = logging.getLogger(__name__)

_client: Optional["TavilyClient"] = None

try:
    os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
except OSError as exc:
    logger.error("Cannot create search cache dir %s: %s", SEARCH_CACHE_DIR, exc)


def get_tavily_client() -> "TavilyClient":
    if TavilyClient is None:  # pragma: no cover
        raise RuntimeError(
            "Python package 'tavily-python' is required for Stage 4 search. "
            "Install dependencies: pip install -r requirements.txt"
        ) from _tavily_import_error
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to .env or your environment."
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def search(query: str, *, max_results: int = 3, use_cache: bool = True) -> str:
    """
    Perform web search and return concatenated snippets (may be empty).
    Results are cached by query hash under agent/search_cache/.
    """
    if not query or not query.strip():
        return ""

    q = query.strip()
    cache_key = hashlib.md5(q.encode("utf-8")).hexdigest()
    cache_path = os.path.join(SEARCH_CACHE_DIR, f"{cache_key}.json")

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f).get("results", "")
        except OSError as exc:
            logger.warning("Search cache read failed: %s", exc)

    try:
        results = get_tavily_client().search(q, max_results=max_results)
        snippets = [r.get("content", "") for r in results.get("results", [])]
        snippets = [s.strip() for s in snippets if isinstance(s, str) and s.strip()]
        text = "\n".join(snippets)
    except Exception as exc:
        logger.warning("Tavily search failed for %r: %s", q, exc)
        return ""

    if use_cache:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"query": q, "results": text}, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Search cache write failed: %s", exc)

    return text
