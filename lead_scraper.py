#!/usr/bin/env python3
"""
lead_scraper.py - Canadian Club Sponsorship Lead List Builder (v3)

Uses Parallel AI's Search, Extract, Chat, Task Group, and FindAll APIs to:
1. Discover top N Canadian universities by QS ranking (FindAll)
2. Enumerate clubs at EACH university (sitemap for UBC, generic for others)
3. Find sponsorship contacts for ALL clubs (single Task Group + Extract)
4. Validate and export to CSV

Usage:
    python lead_scraper.py --test                                 # Smoke test
    python lead_scraper.py --all                                  # Top 10 unis
    python lead_scraper.py --all --top-n 5                        # Top 5 unis
    python lead_scraper.py --all --university "McGill University"  # Single uni
    python lead_scraper.py --stage 1                              # Run stage 1
    python lead_scraper.py --stage 2                              # Run stage 2
    python lead_scraper.py --stage 3                              # Run stage 3
    python lead_scraper.py --stage 4                              # Run stage 4
    python lead_scraper.py --stage 3 --resume                     # Resume
    python lead_scraper.py --max-clubs 30 --all                   # Cap clubs/uni
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

import dns.resolver
import pandas as pd
import phonenumbers
import requests
from thefuzz import fuzz

from api_client import ParallelClient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STAGE1_FILE = os.path.join(CHECKPOINT_DIR, "stage1_universities.json")
STAGE2_FILE = os.path.join(CHECKPOINT_DIR, "stage2_clubs.json")
STAGE3_FILE = os.path.join(CHECKPOINT_DIR, "stage3_contacts.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "sponsorship_leads.csv")

logger = logging.getLogger("lead_scraper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Well-known club directory URLs (saves an API call when available)
# ---------------------------------------------------------------------------
KNOWN_DIRECTORIES: dict[str, str] = {
    "University of British Columbia": "https://amsclubs.ca/all-clubs/",
    "University of Toronto": "https://sop.utoronto.ca/groups/",
    "McGill University": "https://involvement.mcgill.ca/club-search",
    "University of Alberta": "https://campusconnect.ualberta.ca/organizations",
    "McMaster University": "https://msumcmaster.ca/clubs/",
    "Western University": "https://westernu.campuslabs.ca/engage/organizations",
    "Queen's University": "https://queensu.campuslabs.ca/engage/organizations",
    "University of Waterloo": "https://wusa.ca/clubs/",
    "University of Calgary": "https://www.ucalgary.ca/student-services/student-groups",
    "University of Ottawa": "https://www.uottawa.ca/campus-life/clubs-associations",
    "York University": "https://yorku.campuslabs.ca/engage/organizations",
    "Simon Fraser University": "https://go.sfss.ca/clubs/list",
    "Dalhousie University": "https://dsu.ca/clubs-societies",
    "University of Manitoba": "https://umsu.ca/clubs/",
}

# Hardcoded QS-ranked Canadian universities (offline fallback)
# Source: QS World University Rankings 2025 — top 15 Canadian universities
QS_TOP_CANADIAN: list[dict] = [
    {"university": "University of Toronto", "province": "Ontario"},
    {"university": "McGill University", "province": "Quebec"},
    {"university": "University of British Columbia", "province": "British Columbia"},
    {"university": "University of Alberta", "province": "Alberta"},
    {"university": "University of Waterloo", "province": "Ontario"},
    {"university": "Western University", "province": "Ontario"},
    {"university": "University of Montreal", "province": "Quebec"},
    {"university": "McMaster University", "province": "Ontario"},
    {"university": "University of Ottawa", "province": "Ontario"},
    {"university": "Queen's University", "province": "Ontario"},
    {"university": "University of Calgary", "province": "Alberta"},
    {"university": "Simon Fraser University", "province": "British Columbia"},
    {"university": "Dalhousie University", "province": "Nova Scotia"},
    {"university": "York University", "province": "Ontario"},
    {"university": "University of Manitoba", "province": "Manitoba"},
]

# amsclubs.ca-specific constants
AMS_CLUBS_BASE = "https://amsclubs.ca"

# Map amsclubs.ca categories → our categories
AMSCLUBS_CATEGORY_MAP: dict[str, str] = {
    "academic": "Academic",
    "athletic or recreation": "Sports",
    "cultural or identity": "Cultural",
    "grassroots or political": "Political",
    "leisure or hobby": "Social",
    "media or performance": "Arts",
    "other": "Other",
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(data, filepath):
    """Save data to JSON checkpoint file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath):
    """Load data from JSON checkpoint file."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            f"Checkpoint loaded: {filepath} "
            f"({len(data) if isinstance(data, list) else 'dict'})"
        )
        return data
    return None


# =========================================================================
# PHASE 0: Smoke Test
# =========================================================================

def run_smoke_test(client: ParallelClient):
    """Test API connectivity with one Search call and one Chat call."""
    print("\n" + "=" * 60)
    print("PHASE 0: Smoke Test")
    print("=" * 60)

    # --- Test 1: Search API ---
    print("\n--- Test 1: Search API ---")
    try:
        result = client.search(
            objective="Find the largest Canadian universities with the most student clubs",
            search_queries=["Canadian universities most student clubs directory"],
            max_results=3,
        )
        print(f"✅ Search API works! Got {len(result.get('results', []))} results.")
        for r in result.get("results", [])[:3]:
            print(f"   - {r.get('title', 'N/A')[:60]} | {r.get('url', 'N/A')[:60]}")
        print(f"   Usage: {result.get('usage', 'N/A')}")
    except Exception as e:
        print(f"❌ Search API failed: {e}")
        return False

    # --- Test 2: Chat API with JSON schema ---
    print("\n--- Test 2: Chat API (base model, json_schema) ---")
    try:
        schema = {
            "type": "object",
            "properties": {
                "universities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "province": {"type": "string"},
                            "estimated_clubs": {"type": "integer"},
                        },
                    },
                }
            },
        }
        parsed, basis = client.chat_json(
            prompt=(
                "List 3 Canadian universities known for having the most "
                "student clubs. Include estimated club count."
            ),
            schema=schema,
            schema_name="test_universities",
            model="base",
        )
        source_urls = ParallelClient.extract_source_urls(basis)
        print("✅ Chat API works! Parsed JSON response:")
        print(f"   {json.dumps(parsed, indent=2)[:500]}")
        print(f"   Basis citations: {len(basis)}")
        print(f"   Extracted source URLs: {source_urls[:3]}")
    except Exception as e:
        print(f"❌ Chat API failed: {e}")
        return False

    print(f"\n✅ All tests passed! Total API calls: {client.call_count}")
    print("=" * 60)
    return True


# =========================================================================
# STAGE 1: Discover Top N Canadian Universities (FindAll or single-uni)
# =========================================================================

def run_stage1(
    client: ParallelClient,
    university: str | None = None,
    top_n: int = 10,
) -> list[dict]:
    """Discover target universities and their clubs directory URLs.

    Two modes:
      • **Single-uni** (``--university "McGill University"``): Resolves one
        university via KNOWN_DIRECTORIES or Search+Chat.  Fast, 0-2 API calls.
      • **Top-N** (default): Uses FindAll to discover the top *N* Canadian
        universities by QS ranking, then resolves directory URLs for each.

    Returns:
        List of uni dicts with keys: university, province, estimated_club_count,
        clubs_directory_url, source_urls.
    """
    print("\n" + "=" * 60)
    if university:
        print(f"STAGE 1: Resolve Directory URL for {university}")
    else:
        print(f"STAGE 1: Discover Top {top_n} Canadian Universities")
    print("=" * 60)

    # ── Single-university fast path ──────────────────────────
    if university:
        uni_data = _resolve_single_university(client, university)
        result = [uni_data]
        save_checkpoint(result, STAGE1_FILE)
        print(f"\nTotal API calls so far: {client.call_count}")
        return result

    # ── FindAll top-N path (with Chat fallback) ────────────────
    universities: list[dict] = []
    findall_ok = False

    try:
        print(f"\n  🔍 Using FindAll to discover top {top_n} Canadian universities...")
        objective = (
            f"Find the top {top_n} Canadian universities ranked by the "
            f"QS World University Rankings.  For each university, provide "
            f"its name, province, and a URL for its student clubs directory."
        )

        # Step A: Ingest objective into FindAll schema
        schema = client.findall_ingest(objective)
        entity_type = schema.get("entity_type", "universities")
        match_conditions = schema.get("match_conditions", [])

        # Ensure we always filter for Canadian + ranking
        mc_texts = " ".join(mc.get("description", "") for mc in match_conditions).lower()
        if "canad" not in mc_texts:
            match_conditions.append({
                "name": "canadian_university",
                "description": "Must be a university located in Canada.",
            })
        if "rank" not in mc_texts and "qs" not in mc_texts:
            match_conditions.append({
                "name": "qs_ranked",
                "description": (
                    "Must be a top university appearing in the QS World "
                    "University Rankings."
                ),
            })

        print(f"  Entity type: {entity_type}")
        print(f"  Match conditions ({len(match_conditions)}):")
        for mc in match_conditions:
            print(f"    • {mc['name']}: {mc['description'][:80]}")

        # Step B: Create and poll FindAll run
        findall_id = client.findall_create_run(
            objective=objective,
            entity_type=entity_type,
            match_conditions=match_conditions,
            generator="base",
            match_limit=max(5, top_n + 5),  # small buffer
            metadata={"stage": "1_universities"},
        )

        print("\n  Polling for completion...")
        status = client.poll_findall_run(
            findall_id=findall_id,
            poll_interval=5.0,
            timeout=180.0,
        )

        timed_out = status.get("timed_out", False)
        metrics = status.get("metrics", {})
        matched = metrics.get("matched_candidates_count", 0)
        if timed_out:
            print(f"\n  ⚠ Timed out — got {matched} matches")
        else:
            print(f"\n  ✅ Complete: {matched} matches")

        # Step C: Fetch and convert results
        candidates = client.get_findall_results(findall_id)

        for cand in candidates[:top_n]:
            name = cand.get("name", "")
            dir_url = KNOWN_DIRECTORIES.get(name, cand.get("url", ""))
            source_urls = []
            for b in cand.get("basis", []):
                for cit in b.get("citations", []):
                    u = cit.get("url", "")
                    if u and u not in source_urls:
                        source_urls.append(u)

            universities.append({
                "university": name,
                "province": "",
                "estimated_club_count": 0,
                "clubs_directory_url": dir_url,
                "source_urls": source_urls,
            })
            print(f"  {len(universities):>2}. {name:40s} → {dir_url[:60]}")

        findall_ok = True

    except Exception as e:
        logger.warning(f"FindAll failed: {e}")
        print(f"\n  ⚠ FindAll unavailable ({e})")
        print(f"  Falling back to Chat-based university discovery...")

    # ── Chat fallback if FindAll failed or returned nothing ───
    if not findall_ok or not universities:
        try:
            universities = _discover_universities_chat(client, top_n)
        except Exception as e:
            logger.warning(f"Chat fallback also failed: {e}")
            print(f"\n  ⚠ Chat also unavailable ({e})")
            print(f"  Using hardcoded QS rankings as final fallback...")

    # ── Hardcoded fallback if both FindAll and Chat failed ───
    if not universities:
        for qs in QS_TOP_CANADIAN[:top_n]:
            name = qs["university"]
            dir_url = KNOWN_DIRECTORIES.get(name, "")
            universities.append({
                "university": name,
                "province": qs["province"],
                "estimated_club_count": 0,
                "clubs_directory_url": dir_url,
                "source_urls": [],
            })
            marker = "" if dir_url else " (no known directory)"
            print(f"  {len(universities):>2}. {name:40s} {qs['province']}{marker}")

    # If fewer than top_n, pad with KNOWN_DIRECTORIES
    found_names = {u["university"] for u in universities}
    for kn_name, kn_url in KNOWN_DIRECTORIES.items():
        if len(universities) >= top_n:
            break
        if kn_name not in found_names:
            universities.append({
                "university": kn_name,
                "province": "",
                "estimated_club_count": 0,
                "clubs_directory_url": kn_url,
                "source_urls": [],
            })
            found_names.add(kn_name)
            print(f"  {len(universities):>2}. {kn_name:40s} → {kn_url[:60]} (fallback)")

    save_checkpoint(universities, STAGE1_FILE)
    print(f"\n  → {len(universities)} universities resolved")
    print(f"Total API calls so far: {client.call_count}")
    return universities


def _resolve_single_university(client: ParallelClient, university: str) -> dict:
    """Resolve a single university's directory URL (fast path)."""
    known_url = KNOWN_DIRECTORIES.get(university, "")

    if known_url:
        print(f"  ✅ Known directory URL: {known_url}")
        return {
            "university": university,
            "province": "",
            "estimated_club_count": 0,
            "clubs_directory_url": known_url,
            "source_urls": [],
        }

    # Search + Chat fallback
    print("  🔍 Searching for clubs directory...")
    search_result = client.search(
        objective=(
            f"Find the official student clubs directory or list of "
            f"student organizations at {university} in Canada"
        ),
        search_queries=[
            f"{university} student clubs list directory",
            f"{university} student organizations directory",
            f"{university} student union clubs",
        ],
        max_results=5,
    )

    search_urls = [
        r.get("url", "") for r in search_result.get("results", [])
        if r.get("url")
    ]

    schema = {
        "type": "object",
        "properties": {
            "university": {"type": "string"},
            "province": {"type": "string"},
            "estimated_club_count": {
                "type": "integer",
                "description": "Estimated number of student clubs",
            },
            "clubs_directory_url": {
                "type": "string",
                "description": "The URL of the student clubs directory page",
            },
        },
        "required": [
            "university", "province",
            "estimated_club_count", "clubs_directory_url",
        ],
    }

    url_list = "\n".join(f"  - {u}" for u in search_urls[:5])
    parsed, basis = client.chat_json(
        prompt=(
            f"For {university} (Canada), find:\n"
            f"1. The province it's in\n"
            f"2. Estimated number of student clubs\n"
            f"3. The best URL for their student clubs directory\n\n"
            f"Candidate URLs from search:\n{url_list}\n\n"
            f"Pick the URL that leads to the actual clubs listing page."
        ),
        schema=schema,
        schema_name="university_info",
        model="base",
    )

    source_urls = ParallelClient.extract_source_urls(basis)
    uni_data = {
        "university": parsed.get("university", university),
        "province": parsed.get("province", ""),
        "estimated_club_count": parsed.get("estimated_club_count", 0),
        "clubs_directory_url": parsed.get("clubs_directory_url", ""),
        "source_urls": source_urls,
    }
    print(f"  → Directory URL: {uni_data['clubs_directory_url']}")
    print(f"  → Province: {uni_data['province']}")
    print(f"  → Estimated clubs: {uni_data['estimated_club_count']}")
    return uni_data


def _discover_universities_chat(
    client: ParallelClient,
    top_n: int = 10,
) -> list[dict]:
    """Discover top N Canadian universities via Chat API (FindAll fallback).

    Uses the Chat (base) model to research QS-ranked universities, then
    overlays KNOWN_DIRECTORIES for clubs directory URLs.
    """
    print(f"\n  🤖 Using Chat to discover top {top_n} Canadian universities...")

    schema = {
        "type": "object",
        "properties": {
            "universities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "province": {"type": "string"},
                    },
                    "required": ["name", "province"],
                },
            }
        },
        "required": ["universities"],
    }

    parsed, basis = client.chat_json(
        prompt=(
            f"List the top {top_n} Canadian universities by the latest "
            f"QS World University Rankings. For each, provide the official "
            f"university name and the Canadian province it is in."
        ),
        schema=schema,
        schema_name="top_universities",
        model="base",
        system_prompt=(
            "You are a higher-education researcher. Return only universities "
            "in Canada. Use official names (e.g., 'University of British "
            "Columbia' not 'UBC')."
        ),
    )

    source_urls = ParallelClient.extract_source_urls(basis)
    universities: list[dict] = []

    for u in parsed.get("universities", [])[:top_n]:
        name = u.get("name", "")
        if not name:
            continue
        dir_url = KNOWN_DIRECTORIES.get(name, "")
        universities.append({
            "university": name,
            "province": u.get("province", ""),
            "estimated_club_count": 0,
            "clubs_directory_url": dir_url,
            "source_urls": source_urls,
        })
        marker = "" if dir_url else " (no known directory)"
        print(f"  {len(universities):>2}. {name:40s} {u.get('province', '')}{marker}")

    return universities


# =========================================================================
# STAGE 2: Enumerate Clubs at the University
# =========================================================================

STAGE2_SCHEMA = {
    "type": "object",
    "properties": {
        "clubs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "club_name": {
                        "type": "string",
                        "description": "Official name of the student club",
                    },
                    "club_description": {
                        "type": "string",
                        "description": "Brief description of the club's purpose",
                    },
                    "club_website": {
                        "type": "string",
                        "description": "Club's website or social media URL",
                    },
                    "club_category": {
                        "type": "string",
                        "description": (
                            "Category: Academic, Cultural, Sports, Professional, "
                            "Social, Political, Religious, Arts, Technology, "
                            "Community Service, or Other"
                        ),
                    },
                },
                "required": ["club_name"],
            },
        }
    },
    "required": ["clubs"],
}


AMS_CLUBS_SITEMAP = "https://amsclubs.ca/clubs_sitemap.xml"

# Slugs that appear in the sitemap but are not actual clubs
_AMS_SKIP_SLUGS = frozenset({
    "all-clubs", "all-events", "login", "contact-us", "wp-content",
    "wp-admin", "wp-login", "wp-json", "feed", "xmlrpc",
})


def _slug_to_name(slug: str) -> str:
    """Convert a URL slug like 'ai-club' → 'AI Club'.

    Handles common abbreviations (UBC, AI, STEM, etc.)."""
    UPPER = {
        "ubc", "ai", "bc", "stem", "hk", "msf", "it", "ieee",
        "irc", "grsj", "ires", "capsi", "csss", "usu", "bmm",
        "cvc", "nomas", "ux", "nwplus", "aiesec", "ecegsa",
        "citr", "sisu", "brasa", "devec", "mexsa", "canfar",
    }
    words = slug.split("-")
    titled: list[str] = []
    for w in words:
        if w.lower() in UPPER:
            titled.append(w.upper())
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def _extract_amsclubs_directory(
    client: ParallelClient,          # kept for API-compat; not used
    max_clubs: int = 500,
) -> list[dict]:
    """Fetch all club URLs from the Yoast SEO clubs sitemap.

    The amsclubs.ca directory pages are JS-rendered and cannot be read by
    the Extract API.  The Yoast sitemap at /clubs_sitemap.xml lists every
    club URL in plain XML, so we fetch it directly with *requests* — zero
    API calls needed.

    Returns:
        List of club dicts with club_name, club_description, club_website,
        club_category.
    """
    import xml.etree.ElementTree as ET

    print(f"  Fetching clubs sitemap from {AMS_CLUBS_SITEMAP}...")

    try:
        resp = requests.get(AMS_CLUBS_SITEMAP, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠ Failed to fetch sitemap: {e}")
        return []

    # Parse the XML — namespace-aware
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  ⚠ Failed to parse sitemap XML: {e}")
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    loc_elements = root.findall(".//sm:url/sm:loc", ns)

    clubs: list[dict] = []
    seen_slugs: set[str] = set()

    for loc_el in loc_elements:
        url = (loc_el.text or "").strip().rstrip("/")
        if not url or "amsclubs.ca" not in url:
            continue
        # Extract slug from URL: https://amsclubs.ca/<slug>/
        slug = url.rsplit("/", 1)[-1]
        if not slug or slug in _AMS_SKIP_SLUGS or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        clubs.append({
            "club_name": _slug_to_name(slug),
            "club_description": "",            # filled in Stage 3
            "club_website": url + "/",
            "club_category": "",               # filled in Stage 3
        })

    print(f"  ✅ Parsed {len(clubs)} clubs from amsclubs.ca sitemap (0 API calls)")
    return clubs[:max_clubs]


def run_stage2(
    client: ParallelClient,
    max_clubs: int = 100,
    resume: bool = False,
    timeout: float = 45.0,
) -> list[dict]:
    """Enumerate clubs at EVERY university from Stage 1.

    For UBC (amsclubs.ca), directly parses the Yoast sitemap (0 API calls).
    For other universities, uses the Search → Extract → Chat pipeline.

    Args:
        max_clubs: Maximum clubs to find *per university*.
        timeout: Wallclock limit in seconds per university (generic path only).
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Enumerate Clubs")
    print("=" * 60)

    # Load Stage 1 checkpoint
    universities = load_checkpoint(STAGE1_FILE)
    if not universities:
        print("❌ Stage 1 checkpoint not found. Run --stage 1 first.")
        sys.exit(1)

    # Resume support
    all_clubs: list[dict] = []
    if resume:
        existing = load_checkpoint(STAGE2_FILE)
        if existing:
            all_clubs = existing
            done_unis = {c["university"] for c in existing}
            universities = [u for u in universities if u["university"] not in done_unis]
            print(f"Resuming: {len(existing)} clubs already found, "
                  f"{len(universities)} universities remaining")
            if not universities:
                return all_clubs

    print(f"\nUniversities to process: {len(universities)}")

    for idx, uni in enumerate(universities, 1):
        uni_name = uni["university"]
        dir_url = uni.get("clubs_directory_url", "")

        print(f"\n{'─' * 50}")
        print(f"[{idx}/{len(universities)}] {uni_name}")

        # ── amsclubs.ca fast path (UBC) ────────────────────────
        if "amsclubs.ca" in dir_url:
            print(f"  🏫 amsclubs.ca sitemap — 0 API calls")
            clubs = _extract_amsclubs_directory(client, max_clubs=max_clubs)
        else:
            # ── Generic path ───────────────────────────────────
            clubs = _enumerate_clubs_generic(
                client, uni, max_clubs=max_clubs, timeout=timeout,
            )

        # Annotate each club
        for club in clubs:
            club["university"] = uni_name
            club["province"] = uni.get("province", "")
            if not club.get("source_urls"):
                club["source_urls"] = [club.get("club_website", "")]

        all_clubs.extend(clubs)
        print(f"  → {len(clubs)} clubs at {uni_name}  (running total: {len(all_clubs)})")
        save_checkpoint(all_clubs, STAGE2_FILE)

    print(f"\n{'=' * 50}")
    print(f"Stage 2 complete: {len(all_clubs)} total clubs across "
          f"{len({c['university'] for c in all_clubs})} universities")
    print(f"Total API calls so far: {client.call_count}")
    return all_clubs


def _enumerate_clubs_generic(
    client: ParallelClient,
    uni: dict,
    max_clubs: int = 100,
    timeout: float = 45.0,
) -> list[dict]:
    """Generic Search → Extract → Chat pipeline for one university."""
    uni_name = uni["university"]
    stage2_start = time.time()

    def _remaining() -> float:
        return max(0.0, timeout - (time.time() - stage2_start))

    # --- Step A: Search for clubs directory ---
    search_result = client.search(
        objective=(
            f"Find the official student clubs directory or list of "
            f"student organizations at {uni_name}"
        ),
        search_queries=[
            f"{uni_name} student clubs list",
            f"{uni_name} student organizations directory",
            f"{uni_name} student union clubs",
        ],
        max_results=3,
        mode="fast",
    )

    # Collect URLs to extract
    urls_to_extract: list[str] = []
    if uni.get("clubs_directory_url"):
        urls_to_extract.append(uni["clubs_directory_url"])
    for r in search_result.get("results", [])[:3]:
        url = r.get("url", "")
        if url and url not in urls_to_extract:
            urls_to_extract.append(url)

    # --- Step B: Extract content from directory pages ---
    extracted_content = ""
    if _remaining() <= 0:
        print(f"  ⚠ Timeout ({timeout}s) reached after Search — skipping Extract")
    elif urls_to_extract:
        print(f"  Extracting from {len(urls_to_extract)} URL(s)...")
        try:
            extract_result = client.extract(
                urls=urls_to_extract[:3],
                objective=f"List of student clubs and organizations at {uni_name}",
                excerpts=True,
                full_content=True,
            )

            for r in extract_result.get("results", []):
                content = r.get("full_content") or ""
                excerpts = r.get("excerpts", [])
                if content:
                    extracted_content += (
                        f"\n\n--- From {r.get('url', 'unknown')} ---\n{content}"
                    )
                elif excerpts:
                    extracted_content += (
                        f"\n\n--- From {r.get('url', 'unknown')} ---\n"
                        + "\n".join(excerpts)
                    )
        except Exception as e:
            logger.warning(f"  Extract failed: {e}")
            print(f"  ⚠ Extract failed ({e}), falling back to Chat-only")
    else:
        print("  ⚠ No directory URLs found, using Chat research only")

    # --- Step C: Parse clubs with Chat ---
    if extracted_content:
        prompt = (
            f"Based on the following extracted content from {uni_name}'s "
            f"student clubs directory, list up to {max_clubs} student "
            f"clubs/organizations.\n\n"
            f"For each club, provide: club name, brief description, website "
            f"URL (if available), and category (Academic, Cultural, Sports, "
            f"Professional, Social, Political, Religious, Arts, Technology, "
            f"Community Service, or Other).\n\n"
            f"Extracted content:\n{extracted_content[:15000]}"
        )
    else:
        prompt = (
            f"Research and list up to {max_clubs} student "
            f"clubs/organizations at {uni_name} in Canada. "
            f"For each club, provide: club name, brief description, "
            f"website URL (if available), and category.\n\n"
            f"Focus on active clubs that are likely to have sponsorship needs."
        )

    # Use 'speed' model when we have content (just parsing), 'base' when
    # the model needs to do its own web research.
    chat_model = "speed" if extracted_content else "base"

    parsed, basis = client.chat_json(
        prompt=prompt,
        schema=STAGE2_SCHEMA,
        schema_name="clubs_list",
        model=chat_model,
        system_prompt=(
            "You are a research assistant cataloging student clubs at a "
            "Canadian university. Only include clubs you have evidence "
            "actually exist. Do not fabricate club names."
        ),
    )

    elapsed = time.time() - stage2_start
    print(f"  Generic pipeline completed in {elapsed:.1f}s (limit: {timeout}s)")

    clubs = parsed.get("clubs", [])[:max_clubs]

    source_urls = ParallelClient.extract_source_urls(basis)
    for club in clubs:
        club["source_urls"] = source_urls

    return clubs


# =========================================================================
# STAGE 3: Find Sponsorship Contacts
# =========================================================================

# Task spec for contact lookup — flat object, all fields required,
# nullable via union types for the Task API.
TASK_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "club_name": {
            "type": "string",
            "description": "Name of the student club",
        },
        "club_website": {
            "type": "string",
            "description": (
                "Club website or social media URL. For UBC clubs this is "
                "usually https://amsclubs.ca/<slug>/ which has a Contact "
                "section with an email and an Our Team section with names "
                "and roles. Check this URL first."
            ),
        },
        "university": {
            "type": "string",
            "description": "University the club belongs to",
        },
    },
    "required": ["club_name", "club_website", "university"],
    "additionalProperties": False,
}

TASK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "contact_name": {
            "type": ["string", "null"],
            "description": (
                "Full name of the person best suited to handle sponsorship. "
                "Priority: Sponsorship Coordinator > VP Sponsorship > "
                "VP Finance/Treasurer > VP External > President. "
                "Null if not found."
            ),
        },
        "contact_role": {
            "type": ["string", "null"],
            "description": "Role/title of the contact person. Null if not found.",
        },
        "contact_email": {
            "type": ["string", "null"],
            "description": "Email address. Null if not found.",
        },
        "contact_phone": {
            "type": ["string", "null"],
            "description": "Phone number. Null if not found.",
        },
        "is_fallback_contact": {
            "type": "boolean",
            "description": (
                "True if the contact is NOT in a sponsorship-specific role "
                "(e.g., just the President)."
            ),
        },
    },
    "required": [
        "contact_name", "contact_role", "contact_email",
        "contact_phone", "is_fallback_contact",
    ],
    "additionalProperties": False,
}

TASK_SPEC = {
    "input_schema": {"json_schema": TASK_INPUT_SCHEMA},
    "output_schema": {"json_schema": TASK_OUTPUT_SCHEMA},
}

# Regex helpers for parsing amsclubs.ca individual club pages
_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
_TEAM_ROLE_RE = re.compile(
    r'^\s*(.+?)\s*[:\u2013\u2014-]\s*(.+)$',  # "Name : Role" or "Name – Role"
)


def _extract_amsclubs_contacts(
    client: ParallelClient,
    clubs: list[dict],
) -> list[dict]:
    """Extract contacts directly from amsclubs.ca club pages via Extract API.

    Each amsclubs.ca club page has:
      - A **Contact** section with a general email (usually @gmail.com)
      - An **Our Team** section listing members with Name, Role, and
        sometimes personal email links.

    We batch-Extract 5 URLs at a time, then parse the returned content
    for emails and team members.
    """
    contacts: list[dict] = []
    batch_size = 5
    total = len(clubs)

    for i in range(0, total, batch_size):
        batch = clubs[i : i + batch_size]
        urls = [c["club_website"] for c in batch if c.get("club_website")]
        if not urls:
            for c in batch:
                contacts.append(_empty_contact(c))
            continue

        print(f"    Extracting contacts {i + 1}\u2013{min(i + batch_size, total)}/{total}...",
              end="\r")

        try:
            result = client.extract(
                urls=urls,
                objective=(
                    "Find the club contact email address and team member "
                    "names with their roles (especially President, VP "
                    "Sponsorship, VP Finance, Treasurer, VP External)"
                ),
                excerpts=True,
                full_content=True,
            )
        except Exception as e:
            logger.warning(f"  Extract batch failed: {e}")
            for c in batch:
                contacts.append(_empty_contact(c))
            continue

        # Map URL → extracted content
        url_to_content: dict[str, str] = {}
        for r in result.get("results", []):
            page_url = r.get("url", "")
            content = r.get("full_content") or ""
            if not content:
                excerpts_list = r.get("excerpts", [])
                content = "\n".join(excerpts_list)
            url_to_content[page_url] = content

        # Parse each club
        for c in batch:
            cw = c.get("club_website", "")
            page_text = url_to_content.get(cw, "")
            if not page_text:
                contacts.append(_empty_contact(c))
                continue

            contact = _parse_amsclub_page(c, page_text)
            contacts.append(contact)

    print()  # newline after \r
    return contacts


def _parse_amsclub_page(club: dict, page_text: str) -> dict:
    """Parse a single amsclubs.ca club page for contact info."""
    emails: list[str] = _EMAIL_RE.findall(page_text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_emails: list[str] = []
    for e in emails:
        e_lower = e.lower()
        if e_lower not in seen:
            seen.add(e_lower)
            unique_emails.append(e)

    # The first email is usually the club's general contact email
    club_email = unique_emails[0] if unique_emails else ""

    # Parse team members from "Our Team" section
    team_members: list[dict] = []
    in_team = False
    for line in page_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Detect team section
        if "our team" in stripped.lower() or "executive" in stripped.lower():
            in_team = True
            continue
        # Detect section breaks
        if stripped.startswith("#") and in_team:
            if "team" not in stripped.lower() and "exec" not in stripped.lower():
                in_team = False
                continue
        if not in_team:
            continue

        # Try "Name : Role" pattern
        m = _TEAM_ROLE_RE.match(stripped)
        if m:
            name, role = m.group(1).strip(), m.group(2).strip()
            if len(name) > 1 and len(role) > 1:
                team_members.append({"name": name, "role": role})
                continue

        # Standalone lines: first line = name, next = role
        # Common pattern on amsclubs.ca:
        #   Lia Tulchinsky
        #   President
        if team_members and not team_members[-1].get("role"):
            team_members[-1]["role"] = stripped
        elif stripped and not stripped.startswith("[") and not stripped.startswith("!"):
            # Check if this looks like a role keyword
            role_keywords = [
                "president", "vice", "treasurer", "director",
                "coordinator", "secretary", "logistics",
                "developer", "manager", "lead", "chair",
            ]
            if not any(kw in stripped.lower() for kw in role_keywords):
                team_members.append({"name": stripped, "role": ""})
            elif team_members:
                if not team_members[-1].get("role"):
                    team_members[-1]["role"] = stripped
            else:
                team_members.append({"name": stripped, "role": ""})

    # Pick the best contact: prioritise sponsorship-related roles
    PRIORITY_ROLES = [
        "sponsorship", "sponsor", "finance", "treasurer",
        "external", "vp finance", "vp external", "president",
    ]
    best_member: dict = {}
    is_fallback = True

    for member in team_members:
        role_lower = member.get("role", "").lower()
        for idx, kw in enumerate(PRIORITY_ROLES):
            if kw in role_lower:
                if not best_member or idx < best_member.get("_priority", 999):
                    best_member = {**member, "_priority": idx}
                    if idx < len(PRIORITY_ROLES) - 1:  # not "president"
                        is_fallback = False
                    else:
                        is_fallback = True
                break

    if not best_member and team_members:
        best_member = team_members[0]
        is_fallback = True

    contact_name = best_member.get("name", "") if best_member else ""
    contact_role = best_member.get("role", "") if best_member else ""

    return {
        "club_name": club.get("club_name", ""),
        "university": club.get("university", ""),
        "province": club.get("province", ""),
        "contact_name": contact_name,
        "contact_role": contact_role,
        "contact_email": club_email,
        "contact_phone": "",
        "is_fallback_contact": is_fallback,
        "source_urls": [club.get("club_website", "")],
    }


def _empty_contact(club: dict) -> dict:
    """Return an empty contact dict for a club."""
    return {
        "club_name": club.get("club_name", ""),
        "university": club.get("university", ""),
        "province": club.get("province", ""),
        "contact_name": "",
        "contact_role": "",
        "contact_email": "",
        "contact_phone": "",
        "is_fallback_contact": True,
        "source_urls": [],
    }


def run_stage3(
    client: ParallelClient,
    resume: bool = False,
    processor: str = "base-fast",
) -> list[dict]:
    """Find sponsorship contacts for each club.

    For clubs with amsclubs.ca URLs, directly extracts contact info from
    each club's page via the Extract API (cheap, fast, reliable).
    For all other clubs, falls back to the Task Group API.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Find Sponsorship Contacts")
    print("=" * 60)

    # Load Stage 2 checkpoint
    clubs = load_checkpoint(STAGE2_FILE)
    if not clubs:
        print("❌ Stage 2 checkpoint not found. Run --stage 2 first.")
        sys.exit(1)

    # Load existing Stage 3 progress if resuming
    if resume:
        existing = load_checkpoint(STAGE3_FILE)
        if existing:
            completed_names = {c["club_name"] for c in existing}
            remaining = [c for c in clubs if c["club_name"] not in completed_names]
            if not remaining:
                print(f"All {len(existing)} contacts already completed.")
                return existing
            print(f"Resuming: {len(existing)} done, {len(remaining)} remaining")
            clubs = remaining
        else:
            existing = []
    else:
        existing = []

    # Split clubs: amsclubs.ca vs others
    amsclubs_list = [
        c for c in clubs
        if "amsclubs.ca" in (c.get("club_website") or "")
    ]
    other_clubs = [
        c for c in clubs
        if "amsclubs.ca" not in (c.get("club_website") or "")
    ]

    print(f"Clubs to process: {len(clubs)}")
    print(f"  amsclubs.ca (direct Extract): {len(amsclubs_list)}")
    print(f"  Other (Task Group): {len(other_clubs)}")

    contacts: list[dict] = list(existing)

    # ── Path A: Direct Extract for amsclubs.ca clubs ──────────
    if amsclubs_list:
        print(f"\n  📧 Extracting contacts from {len(amsclubs_list)} amsclubs.ca pages...")
        ams_contacts = _extract_amsclubs_contacts(client, amsclubs_list)
        contacts.extend(ams_contacts)

        with_email = sum(1 for c in ams_contacts if c.get("contact_email"))
        with_name = sum(1 for c in ams_contacts if c.get("contact_name"))
        print(f"  amsclubs.ca results: {with_email}/{len(ams_contacts)} emails, "
              f"{with_name}/{len(ams_contacts)} names")

    # ── Path B: Task Group for other clubs ────────────────────
    if other_clubs:
        print(f"\n  🤖 Using Task Group for {len(other_clubs)} non-amsclubs clubs...")
        print(f"  Processor: {processor}")

        university = other_clubs[0].get("university", "Unknown")
        taskgroup_id = client.create_task_group(
            metadata={"stage": "3_contacts", "university": university}
        )

        inputs = []
        for club in other_clubs:
            inputs.append({
                "club_name": club.get("club_name", ""),
                "club_website": club.get("club_website", "") or "",
                "university": club.get("university", ""),
            })

        all_run_ids: list[str] = []
        tg_batch_size = 1000
        for i in range(0, len(inputs), tg_batch_size):
            batch = inputs[i : i + tg_batch_size]
            run_ids = client.add_task_runs(
                taskgroup_id=taskgroup_id,
                task_spec=TASK_SPEC,
                inputs=batch,
                processor=processor,
            )
            all_run_ids.extend(run_ids)

        print(f"  Submitted {len(all_run_ids)} task runs")
        print(f"  Estimated cost: ${len(all_run_ids) * 0.01:.2f}")

        print("\n  Polling for completion...")
        status = client.poll_task_group(
            taskgroup_id=taskgroup_id,
            poll_interval=3.0,
            timeout=600.0,
        )

        timed_out = status.get("timed_out", False)
        counts = status.get("task_run_status_counts", {})
        completed_count = counts.get("completed", 0)
        failed_count = counts.get("failed", 0)

        if timed_out:
            print(f"\n  ⚠ Timed out — collecting {completed_count} partial results")
        else:
            print(f"\n  Final: {completed_count} completed, {failed_count} failed")

        print("\n  Fetching results...")
        results = client.get_task_results(taskgroup_id)

        for res in results:
            inp = res.get("input", {})
            output = res.get("output") or {}
            basis_data = res.get("basis", [])

            contact = {
                "club_name": inp.get("club_name", ""),
                "university": inp.get("university", ""),
                "province": "",
                "contact_name": output.get("contact_name") or "",
                "contact_role": output.get("contact_role") or "",
                "contact_email": output.get("contact_email") or "",
                "contact_phone": output.get("contact_phone") or "",
                "is_fallback_contact": output.get("is_fallback_contact", True),
                "source_urls": ParallelClient.extract_source_urls(basis_data),
            }

            for club in other_clubs:
                if club.get("club_name") == contact["club_name"]:
                    contact["province"] = club.get("province", "")
                    break

            contacts.append(contact)

    save_checkpoint(contacts, STAGE3_FILE)

    # Stats
    with_contact = sum(1 for c in contacts if c.get("contact_name"))
    with_email = sum(1 for c in contacts if c.get("contact_email"))
    print(f"\n{'=' * 40}")
    print(f"Total contacts: {len(contacts)}")
    print(f"  With name:  {with_contact}")
    print(f"  With email: {with_email}")
    print(f"Total API calls: {client.call_count}")
    return contacts


# =========================================================================
# STAGE 4: Validate, Deduplicate & Export CSV
# =========================================================================

def validate_email(email: str) -> bool:
    """Check if email has valid format using regex."""
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def check_mx_record(domain: str) -> bool:
    """Check if email domain has valid MX records."""
    try:
        dns.resolver.resolve(domain, "MX", lifetime=5)
        return True
    except Exception:
        return False


def normalize_phone(phone: str, region: str = "CA") -> str:
    """Normalize phone number to E.164 format."""
    if not phone or not isinstance(phone, str):
        return ""
    try:
        parsed = phonenumbers.parse(phone, region)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
    except Exception:
        pass
    return phone.strip()


def deduplicate_clubs(clubs: list[dict], threshold: int = 85) -> list[dict]:
    """Remove duplicate clubs within each university using fuzzy matching."""
    deduped: list[dict] = []
    seen: dict[str, list[str]] = {}

    for club in clubs:
        uni = club.get("university", "")
        name = club.get("club_name", "")

        if uni not in seen:
            seen[uni] = []

        is_dup = False
        for existing_name in seen[uni]:
            if fuzz.token_sort_ratio(name.lower(), existing_name.lower()) >= threshold:
                is_dup = True
                logger.info(f"Dedup: '{name}' ≈ '{existing_name}' at {uni}")
                break

        if not is_dup:
            seen[uni].append(name)
            deduped.append(club)

    removed = len(clubs) - len(deduped)
    if removed:
        print(f"  Removed {removed} duplicate clubs")
    return deduped


def run_stage4():
    """Validate, deduplicate, and export to CSV."""
    print("\n" + "=" * 60)
    print("STAGE 4: Validate & Export CSV")
    print("=" * 60)

    # Load checkpoints
    clubs = load_checkpoint(STAGE2_FILE) or []
    contacts = load_checkpoint(STAGE3_FILE) or []

    if not clubs:
        print("❌ Stage 2 checkpoint not found. Run stages 1-3 first.")
        sys.exit(1)

    # --- Step A: Deduplicate clubs ---
    print("\nStep A: Deduplicating clubs...")
    clubs = deduplicate_clubs(clubs)
    print(f"  {len(clubs)} unique clubs after deduplication")

    # --- Step B: Build merged DataFrame ---
    print("\nStep B: Merging clubs with contacts...")

    clubs_df = pd.DataFrame(clubs)
    clubs_cols = [
        "university", "province", "club_name", "club_description",
        "club_website", "club_category", "source_urls",
    ]
    for col in clubs_cols:
        if col not in clubs_df.columns:
            clubs_df[col] = ""

    if contacts:
        contacts_df = pd.DataFrame(contacts)
        contacts_cols = [
            "club_name", "university", "contact_name", "contact_role",
            "contact_email", "contact_phone", "is_fallback_contact",
            "source_urls",
        ]
        for col in contacts_cols:
            if col not in contacts_df.columns:
                contacts_df[col] = ""

        # Merge on club_name + university
        merged = clubs_df.merge(
            contacts_df[[
                "club_name", "university", "contact_name", "contact_role",
                "contact_email", "contact_phone", "is_fallback_contact",
            ]],
            on=["club_name", "university"],
            how="left",
            suffixes=("", "_contact"),
        )
    else:
        merged = clubs_df.copy()
        for col in [
            "contact_name", "contact_role", "contact_email",
            "contact_phone", "is_fallback_contact",
        ]:
            merged[col] = ""

    # --- Step C: Validate emails ---
    print("\nStep C: Validating emails...")
    merged["email_format_valid"] = merged["contact_email"].apply(validate_email)

    print("  Checking MX records for email domains...")
    mx_cache: dict[str, bool] = {}
    domain_valid_list: list[bool] = []
    for email in merged["contact_email"]:
        if not validate_email(email):
            domain_valid_list.append(False)
            continue
        domain = email.strip().split("@")[1].lower()
        if domain not in mx_cache:
            mx_cache[domain] = check_mx_record(domain)
        domain_valid_list.append(mx_cache[domain])

    merged["email_domain_valid"] = domain_valid_list
    valid_emails = int(merged["email_format_valid"].sum())
    valid_domains = int(merged["email_domain_valid"].sum())
    print(f"  {valid_emails} valid-format emails, {valid_domains} with valid MX records")

    # --- Step D: Normalize phone numbers ---
    print("\nStep D: Normalizing phone numbers...")
    merged["contact_phone"] = merged["contact_phone"].apply(normalize_phone)

    # --- Step E: Data quality scoring ---
    print("\nStep E: Scoring data quality...")

    def score_quality(row):
        if row.get("email_domain_valid") and row.get("contact_name"):
            if not row.get("is_fallback_contact"):
                return "high"
            return "medium"
        if row.get("contact_name") and row.get("email_format_valid"):
            return "medium"
        if row.get("contact_name"):
            return "low"
        return "no_contact"

    merged["data_quality"] = merged.apply(score_quality, axis=1)

    quality_counts = merged["data_quality"].value_counts()
    print("  Data quality distribution:")
    for q, c in quality_counts.items():
        print(f"    {q}: {c}")

    # --- Step F: Format source URLs ---
    if "source_urls" in merged.columns:
        merged["source_url"] = merged["source_urls"].apply(
            lambda x: x[0] if isinstance(x, list) and x else ""
        )
    else:
        merged["source_url"] = ""

    # --- Step G: Select and order final columns ---
    final_columns = [
        "university", "province", "club_name", "club_description",
        "club_website", "club_category", "contact_name", "contact_role",
        "contact_email", "contact_phone", "is_fallback_contact",
        "email_format_valid", "email_domain_valid", "data_quality",
        "source_url",
    ]
    final_columns = [c for c in final_columns if c in merged.columns]
    output_df = merged[final_columns].copy()
    output_df = output_df.fillna("")

    # --- Step H: Export ---
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Exported {len(output_df)} rows to {OUTPUT_CSV}")

    # Summary
    print(f"\n{'=' * 40}")
    print("SUMMARY")
    print(f"{'=' * 40}")
    print(f"Universities: {output_df['university'].nunique()}")
    print(f"Total clubs: {len(output_df)}")
    print(f"Clubs with contacts: {(output_df['contact_name'] != '').sum()}")
    print(f"Clubs with valid emails: {valid_emails}")
    print("Quality breakdown:")
    for q, c in quality_counts.items():
        print(f"  {q}: {c}")

    return output_df


# =========================================================================
# CLI
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Canadian Club Sponsorship Lead List Builder (v3)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run smoke test (Phase 0)",
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3, 4],
        help="Run a specific stage",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all stages 1-4",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--university", type=str, default=None,
        help="Single university mode (skip FindAll discovery)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top universities to discover (default 10)",
    )
    parser.add_argument(
        "--max-clubs", type=int, default=50,
        help="Max clubs per university (default 50)",
    )
    parser.add_argument(
        "--stage2-timeout", type=float, default=45.0,
        help="Wallclock limit in seconds per university in Stage 2 (default 45)",
    )
    parser.add_argument(
        "--processor", type=str, default="base-fast",
        help="Task processor for Stage 3 (default: base-fast)",
    )
    args = parser.parse_args()

    if not any([args.test, args.stage, args.all]):
        parser.print_help()
        sys.exit(0)

    client = ParallelClient()

    if args.test:
        success = run_smoke_test(client)
        sys.exit(0 if success else 1)

    stages_to_run: list[int] = []
    if args.all:
        stages_to_run = [1, 2, 3, 4]
    elif args.stage:
        stages_to_run = [args.stage]

    for stage in stages_to_run:
        if stage == 1:
            run_stage1(
                client,
                university=args.university,
                top_n=args.top_n,
            )
        elif stage == 2:
            run_stage2(
                client,
                max_clubs=args.max_clubs,
                resume=args.resume,
                timeout=args.stage2_timeout,
            )
        elif stage == 3:
            run_stage3(client, resume=args.resume, processor=args.processor)
        elif stage == 4:
            run_stage4()

    print(f"\n🏁 Done! Total API calls: {client.call_count}")


if __name__ == "__main__":
    main()
