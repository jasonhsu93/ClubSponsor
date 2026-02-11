#!/usr/bin/env python3
"""
lead_scraper.py - Canadian Club Sponsorship Lead List Builder (v2)

Uses Parallel AI's Search, Extract, Chat, and Task Group APIs to:
1. Resolve the target university's clubs directory URL
2. Enumerate up to 100 clubs at that university
3. Find sponsorship contacts for each club via Task Group (base-fast)
4. Validate and export to CSV

Usage:
    python lead_scraper.py --test                                 # Smoke test API key
    python lead_scraper.py --stage 1                              # Run stage 1
    python lead_scraper.py --stage 2                              # Run stage 2
    python lead_scraper.py --stage 3                              # Run stage 3
    python lead_scraper.py --stage 4                              # Run stage 4
    python lead_scraper.py --all                                  # Run all stages 1-4
    python lead_scraper.py --all --university "McGill University"  # Different uni
    python lead_scraper.py --stage 3 --resume                     # Resume from checkpoint
    python lead_scraper.py --max-clubs 50 --all                   # Cap clubs
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
# STAGE 1: Resolve University Directory URL
# =========================================================================

def run_stage1(client: ParallelClient, university: str) -> list[dict]:
    """Resolve the target university's clubs directory URL.

    If the university is in KNOWN_DIRECTORIES, we use the cached URL.
    Otherwise, we use Search + Chat to find it.
    """
    print("\n" + "=" * 60)
    print(f"STAGE 1: Resolve Directory URL for {university}")
    print("=" * 60)

    # Check known directories first
    known_url = KNOWN_DIRECTORIES.get(university, "")

    if known_url:
        print(f"  ✅ Known directory URL: {known_url}")
        uni_data = {
            "university": university,
            "province": "",
            "estimated_club_count": 0,
            "clubs_directory_url": known_url,
            "source_urls": [],
        }
    else:
        # Search for the directory URL
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

        # Use Chat to pick the best URL
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

    result = [uni_data]
    save_checkpoint(result, STAGE1_FILE)
    print(f"\nTotal API calls so far: {client.call_count}")
    return result


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


def run_stage2(
    client: ParallelClient,
    max_clubs: int = 100,
    resume: bool = False,
    timeout: float = 45.0,
) -> list[dict]:
    """Enumerate up to max_clubs at the target university.

    Args:
        timeout: Wallclock limit in seconds for the Search → Extract → Chat
                 sequence.  If exceeded between steps and at least 2 clubs
                 have already been found, remaining steps are skipped.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Enumerate Clubs")
    print("=" * 60)

    # Load Stage 1 checkpoint
    universities = load_checkpoint(STAGE1_FILE)
    if not universities:
        print("❌ Stage 1 checkpoint not found. Run --stage 1 first.")
        sys.exit(1)

    uni = universities[0]  # Single university in v2
    uni_name = uni["university"]

    # Check for existing Stage 2 data
    if resume:
        existing = load_checkpoint(STAGE2_FILE)
        if existing:
            print(f"Resuming: {len(existing)} clubs already found")
            return existing

    print(f"\nProcessing: {uni_name}")
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
        max_results=5,
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

    parsed, basis = client.chat_json(
        prompt=prompt,
        schema=STAGE2_SCHEMA,
        schema_name="clubs_list",
        model="base",
        system_prompt=(
            "You are a research assistant cataloging student clubs at a "
            "Canadian university. Only include clubs you have evidence "
            "actually exist. Do not fabricate club names."
        ),
    )

    elapsed = time.time() - stage2_start
    print(f"  Stage 2 completed in {elapsed:.1f}s (limit: {timeout}s)")

    clubs = parsed.get("clubs", [])[:max_clubs]

    # Extract source URLs from basis (fixed extraction)
    source_urls = ParallelClient.extract_source_urls(basis)

    # Annotate each club with university info
    for club in clubs:
        club["university"] = uni_name
        club["province"] = uni.get("province", "")
        club["source_urls"] = source_urls

    print(f"  → Found {len(clubs)} clubs at {uni_name}")

    save_checkpoint(clubs, STAGE2_FILE)
    print(f"\nTotal API calls so far: {client.call_count}")
    return clubs


# =========================================================================
# STAGE 2 (FindAll): Discover Clubs via FindAll API
# =========================================================================

def run_stage2_findall(
    client: ParallelClient,
    max_clubs: int = 100,
    resume: bool = False,
) -> list[dict]:
    """Discover clubs at the target university using the FindAll API.

    FindAll handles candidate generation, validation, and (optionally)
    enrichment in a single async run.  More expensive than the
    Search→Extract→Chat pipeline ($0.25 + $0.03/match) but tends to
    find more clubs with built-in citation backing.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Enumerate Clubs (FindAll API)")
    print("=" * 60)

    # Load Stage 1 checkpoint
    universities = load_checkpoint(STAGE1_FILE)
    if not universities:
        print("❌ Stage 1 checkpoint not found. Run --stage 1 first.")
        sys.exit(1)

    uni = universities[0]
    uni_name = uni["university"]

    if resume:
        existing = load_checkpoint(STAGE2_FILE)
        if existing:
            print(f"Resuming: {len(existing)} clubs already found")
            return existing

    print(f"\nProcessing: {uni_name}")

    # --- Step A: Ingest – let the API decompose our objective ---
    objective = (
        f"FindAll active student clubs and organizations at "
        f"{uni_name} in Canada"
    )
    print(f"  Ingesting objective...")
    schema = client.findall_ingest(objective)

    entity_type = schema.get("entity_type", "student clubs")
    match_conditions = schema.get("match_conditions", [])

    # Ensure we always have a university-scoped condition
    uni_condition_present = any(
        uni_name.lower() in mc.get("description", "").lower()
        for mc in match_conditions
    )
    if not uni_condition_present:
        match_conditions.append({
            "name": "belongs_to_university_check",
            "description": (
                f"The club/organization must be officially affiliated with "
                f"or registered at {uni_name}."
            ),
        })

    print(f"  Entity type: {entity_type}")
    print(f"  Match conditions ({len(match_conditions)}):")
    for mc in match_conditions:
        print(f"    • {mc['name']}: {mc['description'][:80]}")

    # --- Step B: Create FindAll run ---
    match_limit = max(5, min(max_clubs, 1000))
    estimated_cost = 0.25 + 0.03 * match_limit
    print(f"\n  Creating FindAll run (base, limit={match_limit})...")
    print(f"  Estimated max cost: ${estimated_cost:.2f}")

    findall_id = client.findall_create_run(
        objective=objective,
        entity_type=entity_type,
        match_conditions=match_conditions,
        generator="base",
        match_limit=match_limit,
        metadata={"stage": "2_clubs_findall", "university": uni_name},
    )

    # --- Step C: Poll until complete ---
    print("\n  Polling for completion...")
    status = client.poll_findall_run(
        findall_id=findall_id,
        poll_interval=5.0,
        timeout=300.0,
    )

    timed_out = status.get("timed_out", False)
    metrics = status.get("metrics", {})
    generated = metrics.get("generated_candidates_count", 0)
    matched = metrics.get("matched_candidates_count", 0)

    if timed_out:
        print(f"\n  ⚠ Timed out — collecting {matched} partial matches "
              f"({generated} generated)")
    else:
        print(f"\n  Complete: {matched} matched ({generated} generated)")

    # --- Step D: Fetch matched candidates ---
    print("\n  Fetching results...")
    candidates = client.get_findall_results(findall_id)

    # --- Step E: Convert to our club dict format ---
    clubs: list[dict] = []
    for cand in candidates[:max_clubs]:
        # Extract source URLs from basis
        source_urls = []
        for b in cand.get("basis", []):
            for cit in b.get("citations", []):
                url = cit.get("url", "")
                if url and url not in source_urls:
                    source_urls.append(url)

        # Try to infer category from description
        club = {
            "club_name": cand.get("name", ""),
            "club_description": cand.get("description", ""),
            "club_website": cand.get("url", ""),
            "club_category": "",  # may be enriched later
            "university": uni_name,
            "province": uni.get("province", ""),
            "source_urls": source_urls,
        }
        clubs.append(club)

    print(f"  → Found {len(clubs)} clubs at {uni_name}")

    save_checkpoint(clubs, STAGE2_FILE)
    print(f"\nTotal API calls so far: {client.call_count}")
    return clubs


# =========================================================================
# STAGE 3: Find Sponsorship Contacts via Task Group API
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
            "description": "Club website or social media URL (may be empty)",
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


def run_stage3(
    client: ParallelClient,
    resume: bool = False,
    processor: str = "base-fast",
) -> list[dict]:
    """Find sponsorship contacts using Parallel AI Task Group API.

    Creates a Task Group, submits one run per club, polls until complete,
    then streams all results.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Find Sponsorship Contacts (Task Group API)")
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

    print(f"Clubs to process: {len(clubs)}")
    print(f"Processor: {processor}")

    # --- Step A: Create Task Group ---
    university = clubs[0].get("university", "Unknown")
    taskgroup_id = client.create_task_group(
        metadata={"stage": "3_contacts", "university": university}
    )

    # --- Step B: Build inputs and submit runs ---
    inputs = []
    for club in clubs:
        inputs.append({
            "club_name": club.get("club_name", ""),
            "club_website": club.get("club_website", "") or "",
            "university": club.get("university", ""),
        })

    # Task Group accepts up to 1,000 runs per call
    all_run_ids: list[str] = []
    batch_size = 1000
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        run_ids = client.add_task_runs(
            taskgroup_id=taskgroup_id,
            task_spec=TASK_SPEC,
            inputs=batch,
            processor=processor,
        )
        all_run_ids.extend(run_ids)

    print(f"\n  Submitted {len(all_run_ids)} task runs to group {taskgroup_id}")
    print(f"  Estimated cost: ${len(all_run_ids) * 0.01:.2f} (at $10/1K runs)")

    # --- Step C: Poll until complete ---
    print("\n  Polling for completion...")
    status = client.poll_task_group(
        taskgroup_id=taskgroup_id,
        poll_interval=5.0,
        timeout=600.0,
    )

    timed_out = status.get("timed_out", False)
    counts = status.get("task_run_status_counts", {})
    completed_count = counts.get("completed", 0)
    failed_count = counts.get("failed", 0)

    if timed_out:
        print(f"\n  ⚠ Timed out — collecting {completed_count} partial results "
              f"({failed_count} failed)")
    else:
        print(f"\n  Final: {completed_count} completed, {failed_count} failed")

    # --- Step D: Stream results (works for both complete & partial) ---
    print("\n  Fetching results...")
    results = client.get_task_results(taskgroup_id)

    # --- Step E: Build contacts list ---
    contacts: list[dict] = list(existing)  # start with any resumed data

    for res in results:
        inp = res.get("input", {})
        output = res.get("output") or {}
        basis_data = res.get("basis", [])

        contact = {
            "club_name": inp.get("club_name", ""),
            "university": inp.get("university", ""),
            "province": "",  # filled from clubs data below
            "contact_name": output.get("contact_name") or "",
            "contact_role": output.get("contact_role") or "",
            "contact_email": output.get("contact_email") or "",
            "contact_phone": output.get("contact_phone") or "",
            "is_fallback_contact": output.get("is_fallback_contact", True),
            "source_urls": ParallelClient.extract_source_urls(basis_data),
        }

        # Fill province from clubs data
        for club in clubs:
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

DEFAULT_UNIVERSITY = "University of British Columbia"


def main():
    parser = argparse.ArgumentParser(
        description="Canadian Club Sponsorship Lead List Builder (v2)"
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
        "--university", type=str, default=DEFAULT_UNIVERSITY,
        help=f"Target university (default: {DEFAULT_UNIVERSITY})",
    )
    parser.add_argument(
        "--max-clubs", type=int, default=100,
        help="Max clubs to enumerate (default 100)",
    )
    parser.add_argument(
        "--stage2-timeout", type=float, default=45.0,
        help="Wallclock limit in seconds for Stage 2 (default 45)",
    )
    parser.add_argument(
        "--findall", action="store_true",
        help="Use FindAll API for Stage 2 club discovery",
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
            run_stage1(client, university=args.university)
        elif stage == 2:
            if args.findall:
                run_stage2_findall(
                    client,
                    max_clubs=args.max_clubs,
                    resume=args.resume,
                )
            else:
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
