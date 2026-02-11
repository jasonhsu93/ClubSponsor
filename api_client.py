"""
api_client.py - Parallel AI API wrapper

Provides a reusable client for Parallel AI's Search, Extract, and Chat APIs
with built-in rate limiting, retry logic, and logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("PARALLEL_API_KEY", "")
BASE_URL = "https://api.parallel.ai"
BETA_HEADER = "search-extract-2025-10-10"

# Rate limiting: stay well under 300/min (Chat) and 600/min (Search/Extract)
DEFAULT_SLEEP = 0.5  # seconds between calls

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # exponential: 2s, 4s, 8s

# Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("parallel_api")
logger.setLevel(logging.DEBUG)

# File handler - detailed logs
_fh = logging.FileHandler(
    os.path.join(LOG_DIR, f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

# Console handler - info level
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Parallel AI Client
# ---------------------------------------------------------------------------

class ParallelClient:
    """Wrapper for Parallel AI's Search, Extract, and Chat APIs."""

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL, sleep: float = DEFAULT_SLEEP):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.sleep = sleep
        self._call_count = 0
        self._last_call_time = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers_search_extract(self) -> dict:
        """Headers for Search and Extract endpoints."""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "parallel-beta": BETA_HEADER,
        }

    def _headers_chat(self) -> dict:
        """Headers for Chat endpoint."""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def _rate_limit(self):
        """Enforce minimum time between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.sleep:
            time.sleep(self.sleep - elapsed)
        self._last_call_time = time.time()

    def _request(self, method: str, url: str, headers: dict, payload: dict) -> dict:
        """Make an HTTP request with retry and exponential backoff."""
        self._rate_limit()
        self._call_count += 1

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"[Call #{self._call_count}] {method} {url} (attempt {attempt})\n"
                    f"  Payload: {json.dumps(payload, indent=2)[:500]}"
                )

                resp = requests.request(method, url, headers=headers, json=payload, timeout=300)

                logger.debug(f"  Status: {resp.status_code}")
                logger.debug(f"  Response: {resp.text[:1000]}")

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"  Rate limited (429). Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"  Server error ({resp.status_code}). Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue

                # Client error (4xx, not 429) — don't retry
                logger.error(f"  Client error ({resp.status_code}): {resp.text[:500]}")
                resp.raise_for_status()

            except requests.exceptions.Timeout:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"  Timeout. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            except requests.exceptions.ConnectionError:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"  Connection error. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {method} {url}")

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def search(
        self,
        objective: str,
        search_queries: Optional[list[str]] = None,
        max_results: int = 10,
        mode: str = "one-shot",
        max_chars_per_result: int = 10000,
    ) -> dict:
        """
        Search the web via Parallel's Search API.

        Args:
            objective: Natural language description of what to find.
            search_queries: Optional keyword queries to guide search.
            max_results: Max number of results (default 10).
            mode: 'one-shot' | 'agentic' | 'fast'
            max_chars_per_result: Max chars per excerpt.

        Returns:
            Full API response dict with 'results' array.
        """
        url = f"{self.base_url}/v1beta/search"
        payload = {
            "mode": mode,
            "objective": objective,
            "max_results": max_results,
            "excerpts": {
                "max_chars_per_result": max_chars_per_result,
            },
        }
        if search_queries:
            payload["search_queries"] = search_queries

        logger.info(f"SEARCH: {objective[:80]}...")
        result = self._request("POST", url, self._headers_search_extract(), payload)
        n_results = len(result.get("results", []))
        logger.info(f"  → {n_results} results returned")
        return result

    # ------------------------------------------------------------------
    # Extract API
    # ------------------------------------------------------------------

    def extract(
        self,
        urls: list[str],
        objective: Optional[str] = None,
        excerpts: bool = True,
        full_content: bool = False,
    ) -> dict:
        """
        Extract content from specific URLs via Parallel's Extract API.

        Args:
            urls: List of URLs to extract from.
            objective: What to extract (guides excerpt selection).
            excerpts: Whether to return excerpts.
            full_content: Whether to return full page markdown.

        Returns:
            Full API response dict with 'results' array.
        """
        url = f"{self.base_url}/v1beta/extract"
        payload = {
            "urls": urls,
            "excerpts": excerpts,
            "full_content": full_content,
        }
        if objective:
            payload["objective"] = objective

        logger.info(f"EXTRACT: {len(urls)} URL(s) — {urls[0][:60]}...")
        result = self._request("POST", url, self._headers_search_extract(), payload)
        n_results = len(result.get("results", []))
        n_errors = len(result.get("errors", []))
        logger.info(f"  → {n_results} extracted, {n_errors} errors")
        return result

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: str = "base",
        response_schema: Optional[dict] = None,
    ) -> dict:
        """
        Chat completions via Parallel's Chat API.
        Models 'lite', 'base', 'core' do automatic web research.
        Model 'speed' is fast but no web research.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: 'speed' | 'lite' | 'base' | 'core'
            response_schema: Optional JSON schema for structured output.

        Returns:
            Full API response dict (OpenAI-compatible ChatCompletion + basis).
        """
        url = f"{self.base_url}/v1beta/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }

        # Log first user message for context
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        logger.info(f"CHAT [{model}]: {user_msg[:80]}...")

        result = self._request("POST", url, self._headers_chat(), payload)

        # Log response summary
        choices = result.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            logger.info(f"  → Response length: {len(content)} chars")
        basis = result.get("basis", [])
        if basis:
            logger.info(f"  → {len(basis)} basis citations")

        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def chat_json(
        self,
        prompt: str,
        schema: dict,
        schema_name: str = "response",
        model: str = "base",
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        Chat with structured JSON output, returning the parsed JSON directly.

        Args:
            prompt: User prompt.
            schema: JSON Schema for the expected output.
            schema_name: Name for the schema.
            model: Chat model to use.
            system_prompt: Optional system prompt.

        Returns:
            Tuple of (parsed_json, basis_citations).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response_schema = {
            "name": schema_name,
            "schema": schema,
        }

        result = self.chat(messages, model=model, response_schema=response_schema)

        # Parse response content
        content = result["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content[:500]}")
            parsed = {"raw_content": content}

        basis = result.get("basis", [])
        return parsed, basis

    @property
    def call_count(self) -> int:
        """Total number of API calls made."""
        return self._call_count
