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
from typing import Any, Optional, Union

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
DEFAULT_SLEEP = 0.2  # seconds between calls (5 req/s, well under limits)

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

    @staticmethod
    def extract_source_urls(basis: list) -> list[str]:
        """Extract source URLs from basis citations.

        The Chat API returns basis as:
            [{"field": "...", "citations": [{"url": "...", ...}], ...}]
        The Task API returns output.basis in the same nested format.
        """
        urls = []
        for entry in basis:
            if not isinstance(entry, dict):
                continue
            # Nested citations (standard format)
            for cit in entry.get("citations", []):
                if isinstance(cit, dict) and cit.get("url"):
                    urls.append(cit["url"])
            # Flat fallback (some older responses)
            for key in ("url", "source"):
                if key in entry and entry[key]:
                    urls.append(entry[key])
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                deduped.append(u)
        return deduped

    # ------------------------------------------------------------------
    # Task Group API
    # ------------------------------------------------------------------

    def _headers_task(self) -> dict:
        """Headers for Task / Task Group endpoints."""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def create_task_group(self, metadata: Optional[dict] = None) -> str:
        """Create a Task Group and return its taskgroup_id."""
        url = f"{self.base_url}/v1beta/tasks/groups"
        payload: dict[str, Any] = {}
        if metadata:
            payload["metadata"] = metadata

        logger.info("TASK GROUP: Creating...")
        result = self._request("POST", url, self._headers_task(), payload)
        tg_id = result["taskgroup_id"]
        logger.info(f"  → Task Group created: {tg_id}")
        return tg_id

    def add_task_runs(
        self,
        taskgroup_id: str,
        task_spec: dict,
        inputs: list[dict],
        processor: str = "base-fast",
    ) -> list[str]:
        """Add task runs to a Task Group.

        Args:
            taskgroup_id: The Task Group ID.
            task_spec: Task spec with input_schema and output_schema.
            inputs: List of input dicts matching input_schema.
            processor: Processor tier ('lite', 'base-fast', 'core', etc.).

        Returns:
            List of run_id strings.
        """
        url = f"{self.base_url}/v1beta/tasks/groups/{taskgroup_id}/runs"

        # Build run inputs
        run_inputs = [
            {"input": inp, "processor": processor}
            for inp in inputs
        ]

        payload = {
            "default_task_spec": task_spec,
            "inputs": run_inputs,
        }

        logger.info(f"TASK GROUP ADD: {len(run_inputs)} runs (processor={processor})")
        result = self._request("POST", url, self._headers_task(), payload)
        run_ids = result.get("run_ids", [])
        status = result.get("status", {})
        logger.info(
            f"  → {len(run_ids)} runs queued. "
            f"Total in group: {status.get('num_task_runs', '?')}"
        )
        return run_ids

    def poll_task_group(
        self,
        taskgroup_id: str,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> dict:
        """Poll a Task Group until all runs complete.

        Returns:
            Final status dict with num_task_runs, task_run_status_counts, etc.
            If the timeout is reached, the dict includes ``"timed_out": True``
            so callers can handle partial results gracefully.
        """
        url = f"{self.base_url}/v1beta/tasks/groups/{taskgroup_id}"
        headers = self._headers_task()
        start = time.time()
        status: dict = {}  # initialise in case timeout fires on first iteration

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                print()  # newline after \r
                logger.warning(
                    f"  Task Group {taskgroup_id} timed out after {timeout}s "
                    f"— returning partial results"
                )
                status["timed_out"] = True
                return status

            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", {})

            counts = status.get("task_run_status_counts", {})
            total = status.get("num_task_runs", 0)
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            running = counts.get("running", 0)
            queued = counts.get("queued", 0)

            logger.info(
                f"  Poll [{elapsed:.0f}s]: "
                f"{completed}/{total} completed, "
                f"{running} running, {queued} queued, {failed} failed"
            )
            print(
                f"  ⏳ {completed}/{total} done "
                f"({running} running, {queued} queued, {failed} failed) "
                f"[{elapsed:.0f}s]",
                end="\r",
            )

            if not status.get("is_active", True):
                print()  # newline after \r
                logger.info(
                    f"  Task Group complete: {completed} completed, {failed} failed"
                )
                return status

            time.sleep(poll_interval)

    def get_task_results(
        self,
        taskgroup_id: str,
    ) -> list[dict]:
        """Fetch all task run results from a completed Task Group via SSE stream.

        Returns:
            List of dicts with keys: run_id, status, input, output, basis.
        """
        url = (
            f"{self.base_url}/v1beta/tasks/groups/{taskgroup_id}/runs"
            f"?include_input=true&include_output=true"
        )
        headers = self._headers_task()
        headers["Accept"] = "text/event-stream"

        logger.info(f"TASK GROUP RESULTS: Streaming from {taskgroup_id}")

        resp = requests.get(url, headers=headers, timeout=300, stream=True)
        resp.raise_for_status()

        results = []
        buffer = ""

        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            buffer += chunk
            # SSE events are separated by double newlines
            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)
                # Parse SSE fields
                data_lines = []
                for line in event_text.strip().split("\n"):
                    if line.startswith("data: "):
                        data_lines.append(line[6:])
                    elif line.startswith("data:"):
                        data_lines.append(line[5:])
                if not data_lines:
                    continue
                raw = "\n".join(data_lines)
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"  Skipping unparseable SSE data: {raw[:200]}")
                    continue

                if event.get("type") == "task_run.state":
                    run_data = {
                        "run_id": event.get("run_id", ""),
                        "status": event.get("status", ""),
                        "input": event.get("input", {}).get("input", {})
                            if event.get("input") else {},
                        "output": None,
                        "basis": [],
                    }
                    output = event.get("output")
                    if output and isinstance(output, dict):
                        run_data["output"] = output.get("content", {})
                        run_data["basis"] = output.get("basis", [])
                    results.append(run_data)

        logger.info(f"  → {len(results)} results fetched")
        return results

    def get_run_result(self, run_id: str) -> dict:
        """Get the result of a single task run (blocks until complete)."""
        url = f"{self.base_url}/v1/tasks/runs/{run_id}/result"
        headers = self._headers_task()

        logger.info(f"TASK RUN RESULT: {run_id}")
        resp = requests.get(url, headers=headers, timeout=600)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # FindAll API
    # ------------------------------------------------------------------

    FINDALL_BETA_HEADER = "findall-2025-09-15"

    def _headers_findall(self) -> dict:
        """Headers for FindAll endpoints."""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "parallel-beta": self.FINDALL_BETA_HEADER,
        }

    def findall_ingest(self, objective: str) -> dict:
        """Convert a natural-language objective into a structured FindAll schema.

        Returns:
            Dict with keys: objective, entity_type, match_conditions.
        """
        url = f"{self.base_url}/v1beta/findall/ingest"
        payload = {"objective": objective}

        logger.info(f"FINDALL INGEST: {objective[:80]}...")
        result = self._request("POST", url, self._headers_findall(), payload)
        logger.info(
            f"  → entity_type={result.get('entity_type')}, "
            f"{len(result.get('match_conditions', []))} match conditions"
        )
        return result

    def findall_create_run(
        self,
        objective: str,
        entity_type: str,
        match_conditions: list[dict],
        generator: str = "base",
        match_limit: int = 100,
        exclude_list: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Start a FindAll run and return the findall_id.

        Args:
            objective: Natural-language description.
            entity_type: e.g. "student clubs".
            match_conditions: List of {name, description} dicts.
            generator: 'preview', 'base', 'core', or 'pro'.
            match_limit: Max matched candidates (5–1000).
            exclude_list: Optional [{name, url}, ...] to skip.
            metadata: Optional metadata dict.

        Returns:
            findall_id string.
        """
        url = f"{self.base_url}/v1beta/findall/runs"
        payload: dict[str, Any] = {
            "objective": objective,
            "entity_type": entity_type,
            "match_conditions": match_conditions,
            "generator": generator,
            "match_limit": max(5, min(match_limit, 1000)),
        }
        if exclude_list:
            payload["exclude_list"] = exclude_list
        if metadata:
            payload["metadata"] = metadata

        logger.info(
            f"FINDALL RUN: generator={generator}, limit={match_limit} "
            f"— {objective[:60]}..."
        )
        result = self._request("POST", url, self._headers_findall(), payload)
        fid = result["findall_id"]
        logger.info(f"  → FindAll run created: {fid}")
        return fid

    def poll_findall_run(
        self,
        findall_id: str,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> dict:
        """Poll a FindAll run until it completes.

        Returns:
            Run status dict.  Includes ``"timed_out": True`` if the timeout
            is reached before the run finishes.
        """
        url = f"{self.base_url}/v1beta/findall/runs/{findall_id}"
        headers = self._headers_findall()
        start = time.time()
        status: dict = {}

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                print()
                logger.warning(
                    f"  FindAll {findall_id} timed out after {timeout}s"
                )
                status["timed_out"] = True
                return status

            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", {})
            metrics = status.get("metrics", {})
            generated = metrics.get("generated_candidates_count", 0)
            matched = metrics.get("matched_candidates_count", 0)

            logger.info(
                f"  Poll [{elapsed:.0f}s]: "
                f"{generated} generated, {matched} matched"
            )
            print(
                f"  ⏳ {matched} matched ({generated} generated) [{elapsed:.0f}s]",
                end="\r",
            )

            if not status.get("is_active", True):
                print()
                logger.info(
                    f"  FindAll complete: {generated} generated, {matched} matched"
                )
                return status

            time.sleep(poll_interval)

    def get_findall_results(self, findall_id: str) -> list[dict]:
        """Fetch matched candidates from a completed FindAll run.

        Returns:
            List of candidate dicts (name, url, description, output, basis, ...).
        """
        url = f"{self.base_url}/v1beta/findall/runs/{findall_id}/result"
        headers = self._headers_findall()

        logger.info(f"FINDALL RESULTS: {findall_id}")

        self._rate_limit()
        self._call_count += 1
        resp = requests.get(url, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        matched = [c for c in candidates if c.get("match_status") == "matched"]
        logger.info(
            f"  → {len(matched)} matched candidates "
            f"(of {len(candidates)} total)"
        )
        return matched

    def findall_add_enrichment(
        self,
        findall_id: str,
        output_schema: dict,
        processor: str = "base",
    ) -> dict:
        """Add an enrichment to a FindAll run.

        Args:
            findall_id: The FindAll run ID.
            output_schema: JSON schema for enrichment output fields.
            processor: Task API processor for enrichment ('lite', 'base', etc.).

        Returns:
            API response dict.
        """
        url = f"{self.base_url}/v1beta/findall/runs/{findall_id}/enrich"
        payload = {
            "generator": processor,
            "output_schema": {
                "type": "json",
                "json_schema": output_schema,
            },
        }

        logger.info(f"FINDALL ENRICH: {findall_id} (processor={processor})")
        result = self._request("POST", url, self._headers_findall(), payload)
        logger.info(f"  → Enrichment added")
        return result

    @property
    def call_count(self) -> int:
        """Total number of API calls made."""
        return self._call_count
