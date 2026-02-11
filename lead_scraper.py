#!/usr/bin/env python3
"""
lead_scraper.py - Canadian Club Sponsorship Lead List Builder

Uses Parallel AI's Search, Extract, and Chat APIs to:
1. Identify top 10 Canadian universities with 25+ student clubs
2. Enumerate up to 100 clubs per university
3. Find sponsorship contacts for each club
4. Validate and export to CSV

Usage:
    python lead_scraper.py --test                  # Smoke test API key
    python lead_scraper.py --stage 1               # Run stage 1
    python lead_scraper.py --stage 2               # Run stage 2
    python lead_scraper.py --stage 3               # Run stage 3
    python lead_scraper.py --stage 4               # Run stage 4
    python lead_scraper.py --all                   # Run all stages 1-4
    python lead_scraper.py --stage 3 --resume      # Resume from checkpoint
    python lead_scraper.py --max-clubs 50 --all    # Cap clubs per school
"""

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
        logger.info(f"Checkpoint loaded: {filepath} ({len(data) if isinstance(data, list) else 'dict'})")
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
            prompt="List 3 Canadian universities known for having the most student clubs. Include estimated club count.",
            schema=schema,
            schema_name="test_universities",
            model="base",
        )
        print(f"✅ Chat API works! Parsed JSON response:")
        print(f"   {json.dumps(parsed, indent=2)[:500]}")
        print(f"   Basis citations: {len(basis)}")
        if basis:
            for b in basis[:2]:
                print(f"     - {json.dumps(b)[:120]}")
    except Exception as e:
        print(f"❌ Chat API failed: {e}")
        return False

    print(f"\n✅ All tests passed! Total API calls: {client.call_count}")
    print("=" * 60)
    return True


# =========================================================================
# STAGE 1: Identify Top 10 Universities
# =========================================================================

STAGE1_SCHEMA = {
    "type": "object",
    "properties": {
        "universities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "university": {"type": "string", "description": "Full official name of the university"},
                    "province": {"type": "string", "description": "Canadian province"},
                    "estimated_club_count": {"type": "integer", "description": "Estimated number of student clubs"},
                    "clubs_directory_url": {"type": "string", "description": "URL of the official student clubs directory page"},
                },
                "required": ["university", "province", "estimated_club_count"],
            },
        }
    },
    "required": ["universities"],
}


def run_stage1(client: ParallelClient) -> list[dict]:
    """Identify top 10 Canadian universities with 25+ student clubs."""
    print("\n" + "=" * 60)
    print("STAGE 1: Identify Top 10 Canadian Universities")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Strategy: We KNOW the large Canadian universities. The model struggles
    # to return accurate club counts for all of them in a single query.
    # Instead, we'll:
    #   1. Start with a known list of 15 large Canadian universities
    #   2. Use Search + Chat to find each university's clubs directory URL
    #      and verify they have 25+ clubs
    #   3. Take the top 10
    # -------------------------------------------------------------------

    candidate_universities = [
        {"university": "University of Toronto", "province": "Ontario"},
        {"university": "University of British Columbia", "province": "British Columbia"},
        {"university": "McGill University", "province": "Quebec"},
        {"university": "University of Alberta", "province": "Alberta"},
        {"university": "McMaster University", "province": "Ontario"},
        {"university": "Western University", "province": "Ontario"},
        {"university": "Queen's University", "province": "Ontario"},
        {"university": "University of Waterloo", "province": "Ontario"},
        {"university": "University of Calgary", "province": "Alberta"},
        {"university": "University of Ottawa", "province": "Ontario"},
        {"university": "York University", "province": "Ontario"},
        {"university": "Toronto Metropolitan University", "province": "Ontario"},
        {"university": "Simon Fraser University", "province": "British Columbia"},
        {"university": "Dalhousie University", "province": "Nova Scotia"},
        {"university": "University of Manitoba", "province": "Manitoba"},
    ]

    # Use a single Chat call to look up club directory URLs and counts for all
    uni_names = "\n".join(f"- {u['university']} ({u['province']})" for u in candidate_universities)

    schema = {
        "type": "object",
        "properties": {
            "universities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "university": {"type": "string"},
                        "province": {"type": "string"},
                        "estimated_club_count": {"type": "integer", "description": "Estimated number of student clubs. Use 0 only if you truly cannot find any information."},
                        "clubs_directory_url": {"type": "string", "description": "The URL of the student clubs directory or listing page"},
                    },
                    "required": ["university", "province", "estimated_club_count", "clubs_directory_url"],
                },
            }
        },
        "required": ["universities"],
    }

    prompt = (
        "For each of the following Canadian universities, find:\n"
        "1. The number of student clubs/organizations (search their student union/student life website)\n"
        "2. The URL of their official student clubs directory or clubs listing page\n\n"
        "Universities:\n"
        f"{uni_names}\n\n"
        "Important: Most large Canadian universities have between 100 and 500+ student clubs. "
        "If you cannot find an exact number, provide your best estimate based on the university's "
        "size and what you find on their website. Do NOT return 0 unless the university truly has "
        "no clubs directory."
    )

    parsed, basis = client.chat_json(
        prompt=prompt,
        schema=schema,
        schema_name="universities",
        model="core",  # Use core for this critical first step — need thorough research
        system_prompt=(
            "You are a research assistant. For each university, search for their "
            "student clubs directory page and count how many clubs are listed. "
            "Return accurate URLs and club counts. Every large Canadian university "
            "has at least 50-100 student clubs."
        ),
    )

    universities = parsed.get("universities", [])

    # Extract source URLs from basis
    source_urls = []
    for b in basis:
        if isinstance(b, dict):
            if "citations" in b:
                for cit in b["citations"]:
                    if isinstance(cit, dict) and "url" in cit:
                        source_urls.append(cit["url"])
            for key in ("url", "source"):
                if key in b:
                    source_urls.append(b[key])

    for u in universities:
        u["source_urls"] = source_urls

    # Filter to 25+ clubs, take top 10
    qualified = [u for u in universities if u.get("estimated_club_count", 0) >= 25]
    qualified.sort(key=lambda x: x.get("estimated_club_count", 0), reverse=True)

    # If filtering removed too many (model returned 0s), fall back to taking all
    # with a non-zero count, or even just the first 10 candidates
    if len(qualified) < 10:
        print(f"  ⚠ Only {len(qualified)} universities had 25+ clubs reported.")
        # Add back universities with 0 count but valid directory URLs
        for u in universities:
            if u not in qualified and u.get("clubs_directory_url"):
                u["estimated_club_count"] = 50  # reasonable default for large uni
                qualified.append(u)
        # If still not enough, add remaining candidates
        if len(qualified) < 10:
            existing_names = {u["university"] for u in qualified}
            for c in candidate_universities:
                if c["university"] not in existing_names and len(qualified) < 10:
                    c["estimated_club_count"] = 50
                    c["clubs_directory_url"] = ""
                    c["source_urls"] = []
                    qualified.append(c)

    qualified = qualified[:10]

    print(f"\nFound {len(qualified)} qualifying universities:")
    for i, u in enumerate(qualified, 1):
        print(f"  {i}. {u['university']} ({u['province']}) — ~{u.get('estimated_club_count', '?')} clubs")
        if u.get("clubs_directory_url"):
            print(f"     Directory: {u['clubs_directory_url']}")

    save_checkpoint(qualified, STAGE1_FILE)
    print(f"\nTotal API calls so far: {client.call_count}")
    return qualified


# =========================================================================
# STAGE 2: Enumerate Clubs per University
# =========================================================================

STAGE2_SCHEMA = {
    "type": "object",
    "properties": {
        "clubs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "club_name": {"type": "string", "description": "Official name of the student club"},
                    "club_description": {"type": "string", "description": "Brief description of the club's purpose"},
                    "club_website": {"type": "string", "description": "Club's website or social media URL"},
                    "club_category": {"type": "string", "description": "Category (e.g., Academic, Cultural, Sports, Professional, Social)"},
                },
                "required": ["club_name"],
            },
        }
    },
    "required": ["clubs"],
}


def run_stage2(client: ParallelClient, max_clubs: int = 100, resume: bool = False) -> list[dict]:
    """Enumerate up to max_clubs per university using Search + Extract + Chat."""
    print("\n" + "=" * 60)
    print("STAGE 2: Enumerate Clubs per University")
    print("=" * 60)

    # Load Stage 1 checkpoint
    universities = load_checkpoint(STAGE1_FILE)
    if not universities:
        print("❌ Stage 1 checkpoint not found. Run --stage 1 first.")
        sys.exit(1)

    # Load existing Stage 2 progress if resuming
    all_clubs = []
    completed_unis = set()
    if resume:
        existing = load_checkpoint(STAGE2_FILE)
        if existing:
            all_clubs = existing
            completed_unis = {c["university"] for c in existing}
            print(f"Resuming: {len(completed_unis)} universities already processed")

    for i, uni in enumerate(universities, 1):
        uni_name = uni["university"]
        if uni_name in completed_unis:
            print(f"\n[{i}/{len(universities)}] SKIP (already done): {uni_name}")
            continue

        print(f"\n[{i}/{len(universities)}] Processing: {uni_name}")

        # --- Step A: Search for clubs directory ---
        search_result = client.search(
            objective=f"Find the official student clubs directory or list of student organizations at {uni_name}",
            search_queries=[
                f"{uni_name} student clubs list",
                f"{uni_name} student organizations directory",
                f"{uni_name} student union clubs",
            ],
            max_results=5,
        )

        # Collect URLs to extract from
        urls_to_extract = []
        # Prefer the directory URL from Stage 1 if available
        if uni.get("clubs_directory_url"):
            urls_to_extract.append(uni["clubs_directory_url"])
        # Add search result URLs
        for r in search_result.get("results", [])[:3]:
            url = r.get("url", "")
            if url and url not in urls_to_extract:
                urls_to_extract.append(url)

        if not urls_to_extract:
            print(f"  ⚠ No directory URLs found for {uni_name}, using Chat research only")
            # Fall back to Chat with web research
            extracted_content = ""
        else:
            # --- Step B: Extract content from directory pages ---
            print(f"  Extracting from {len(urls_to_extract)} URL(s)...")
            extract_result = client.extract(
                urls=urls_to_extract[:3],  # Max 3 URLs
                objective=f"List of student clubs and organizations at {uni_name}",
                excerpts=True,
                full_content=True,
            )

            # Combine extracted content
            extracted_content = ""
            for r in extract_result.get("results", []):
                content = r.get("full_content") or ""
                excerpts = r.get("excerpts", [])
                if content:
                    extracted_content += f"\n\n--- From {r.get('url', 'unknown')} ---\n{content}"
                elif excerpts:
                    extracted_content += f"\n\n--- From {r.get('url', 'unknown')} ---\n" + "\n".join(excerpts)

        # --- Step C: Parse clubs with Chat ---
        if extracted_content:
            prompt = (
                f"Based on the following extracted content from {uni_name}'s student clubs directory, "
                f"list up to {max_clubs} student clubs/organizations.\n\n"
                f"For each club, provide: club name, brief description, website URL (if available), "
                f"and category (Academic, Cultural, Sports, Professional, Social, Political, Religious, "
                f"Arts, Technology, Community Service, or Other).\n\n"
                f"Extracted content:\n{extracted_content[:15000]}"
            )
        else:
            prompt = (
                f"Research and list up to {max_clubs} student clubs/organizations at {uni_name} in Canada. "
                f"For each club, provide: club name, brief description, website URL (if available), "
                f"and category (Academic, Cultural, Sports, Professional, Social, Political, Religious, "
                f"Arts, Technology, Community Service, or Other).\n\n"
                f"Focus on active clubs that are likely to have sponsorship needs."
            )

        parsed, basis = client.chat_json(
            prompt=prompt,
            schema=STAGE2_SCHEMA,
            schema_name="clubs_list",
            model="base",
            system_prompt=(
                "You are a research assistant cataloging student clubs at Canadian universities. "
                "Only include clubs you have evidence actually exist. Do not fabricate club names."
            ),
        )

        clubs = parsed.get("clubs", [])[:max_clubs]

        # Extract source URLs from basis
        source_urls = []
        for b in basis:
            if isinstance(b, dict):
                for key in ("url", "source", "citation"):
                    if key in b:
                        source_urls.append(b[key])

        # Annotate each club with university info
        for club in clubs:
            club["university"] = uni_name
            club["province"] = uni.get("province", "")
            club["source_urls"] = source_urls

        all_clubs.extend(clubs)
        print(f"  → Found {len(clubs)} clubs at {uni_name}")

        # Save progress after each university
        save_checkpoint(all_clubs, STAGE2_FILE)

    print(f"\n{'=' * 40}")
    print(f"Total clubs found: {len(all_clubs)}")
    print(f"Total API calls: {client.call_count}")
    return all_clubs


# =========================================================================
# STAGE 3: Find Sponsorship Contacts (Batched)
# =========================================================================

STAGE3_SCHEMA = {
    "type": "object",
    "properties": {
        "contacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "club_name": {"type": "string", "description": "Name of the club (must match input exactly)"},
                    "contact_name": {"type": "string", "description": "Full name of the contact person"},
                    "contact_role": {"type": "string", "description": "Their role/title (e.g., Sponsorship Coordinator, VP Finance, President)"},
                    "contact_email": {"type": "string", "description": "Email address"},
                    "contact_phone": {"type": "string", "description": "Phone number if available"},
                    "is_fallback_contact": {"type": "boolean", "description": "True if this is not a sponsorship-specific role"},
                },
                "required": ["club_name"],
            },
        }
    },
    "required": ["contacts"],
}


def run_stage3(client: ParallelClient, batch_size: int = 10, resume: bool = False) -> list[dict]:
    """Find sponsorship contacts for each club using batched Chat lite calls."""
    print("\n" + "=" * 60)
    print("STAGE 3: Find Sponsorship Contacts")
    print("=" * 60)

    # Load Stage 2 checkpoint
    clubs = load_checkpoint(STAGE2_FILE)
    if not clubs:
        print("❌ Stage 2 checkpoint not found. Run --stage 2 first.")
        sys.exit(1)

    # Load existing Stage 3 progress if resuming
    all_contacts = []
    completed_clubs = set()
    if resume:
        existing = load_checkpoint(STAGE3_FILE)
        if existing:
            all_contacts = existing
            completed_clubs = {c["club_name"] for c in existing}
            print(f"Resuming: {len(completed_clubs)} clubs already processed")

    # Filter out already-completed clubs
    remaining_clubs = [c for c in clubs if c["club_name"] not in completed_clubs]
    print(f"Clubs to process: {len(remaining_clubs)} (of {len(clubs)} total)")

    # Group clubs by university for better batching
    uni_groups = {}
    for club in remaining_clubs:
        uni = club["university"]
        if uni not in uni_groups:
            uni_groups[uni] = []
        uni_groups[uni].append(club)

    total_batches = 0
    for uni_name, uni_clubs in uni_groups.items():
        # Batch within each university
        for batch_start in range(0, len(uni_clubs), batch_size):
            batch = uni_clubs[batch_start:batch_start + batch_size]
            total_batches += 1

            club_list = "\n".join(
                f"  {j}. {c['club_name']}" + (f" (website: {c.get('club_website', 'N/A')})" if c.get('club_website') else "")
                for j, c in enumerate(batch, 1)
            )

            prompt = (
                f"For each of the following {len(batch)} student clubs at {uni_name} (Canada), "
                f"find the person best suited to handle sponsorship inquiries.\n\n"
                f"Search priority for contact role:\n"
                f"1. Sponsorship Coordinator / Director of Sponsorship\n"
                f"2. VP Sponsorship\n"
                f"3. VP Finance / Treasurer\n"
                f"4. VP External / External Relations\n"
                f"5. President\n\n"
                f"For each person found, provide: their full name, role/title, email, and phone number (if available). "
                f"Set is_fallback_contact to true if the person is NOT in a sponsorship-specific role.\n\n"
                f"Clubs:\n{club_list}\n\n"
                f"If you cannot find any contact information for a club, still include it with empty fields."
            )

            logger.info(f"Stage 3 batch {total_batches}: {uni_name} — {len(batch)} clubs")

            parsed, basis = client.chat_json(
                prompt=prompt,
                schema=STAGE3_SCHEMA,
                schema_name="contacts",
                model="lite",
                system_prompt=(
                    "You are a research assistant finding contact information for student club "
                    "sponsorship outreach. Only provide contact info you actually find — do not "
                    "fabricate names, emails, or phone numbers. If you cannot find contact info, "
                    "leave the fields empty."
                ),
            )

            contacts = parsed.get("contacts", [])

            # Extract source URLs from basis
            source_urls = []
            for b in basis:
                if isinstance(b, dict):
                    for key in ("url", "source", "citation"):
                        if key in b:
                            source_urls.append(b[key])

            # Annotate contacts with university info
            for contact in contacts:
                contact["university"] = uni_name
                contact["province"] = uni_clubs[0].get("province", "")
                contact["source_urls"] = source_urls

            all_contacts.extend(contacts)
            print(f"  Batch {total_batches}: {len(contacts)} contacts from {uni_name}")

            # Save progress after each batch
            save_checkpoint(all_contacts, STAGE3_FILE)

    print(f"\n{'=' * 40}")
    print(f"Total contacts found: {len(all_contacts)}")
    print(f"Total API calls: {client.call_count}")
    return all_contacts


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
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass
    return phone.strip()


def deduplicate_clubs(clubs: list[dict], threshold: int = 85) -> list[dict]:
    """Remove duplicate clubs within each university using fuzzy matching."""
    deduped = []
    seen = {}  # university -> list of club names

    for club in clubs:
        uni = club.get("university", "")
        name = club.get("club_name", "")

        if uni not in seen:
            seen[uni] = []

        # Check against existing names for this university
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
    universities = load_checkpoint(STAGE1_FILE) or []
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

    # Create clubs DataFrame
    clubs_df = pd.DataFrame(clubs)
    clubs_cols = ["university", "province", "club_name", "club_description",
                  "club_website", "club_category", "source_urls"]
    for col in clubs_cols:
        if col not in clubs_df.columns:
            clubs_df[col] = ""

    # Create contacts DataFrame
    if contacts:
        contacts_df = pd.DataFrame(contacts)
        contacts_cols = ["club_name", "university", "contact_name", "contact_role",
                         "contact_email", "contact_phone", "is_fallback_contact", "source_urls"]
        for col in contacts_cols:
            if col not in contacts_df.columns:
                contacts_df[col] = ""

        # Merge on club_name + university
        merged = clubs_df.merge(
            contacts_df[["club_name", "university", "contact_name", "contact_role",
                          "contact_email", "contact_phone", "is_fallback_contact"]],
            on=["club_name", "university"],
            how="left",
            suffixes=("", "_contact"),
        )
    else:
        merged = clubs_df.copy()
        for col in ["contact_name", "contact_role", "contact_email", "contact_phone", "is_fallback_contact"]:
            merged[col] = ""

    # --- Step C: Validate emails ---
    print("\nStep C: Validating emails...")
    merged["email_format_valid"] = merged["contact_email"].apply(validate_email)

    # MX record check (only for valid-format emails)
    print("  Checking MX records for email domains...")
    mx_cache = {}
    domain_valid_list = []
    for email in merged["contact_email"]:
        if not validate_email(email):
            domain_valid_list.append(False)
            continue
        domain = email.strip().split("@")[1].lower()
        if domain not in mx_cache:
            mx_cache[domain] = check_mx_record(domain)
        domain_valid_list.append(mx_cache[domain])

    merged["email_domain_valid"] = domain_valid_list
    valid_emails = merged["email_format_valid"].sum()
    valid_domains = merged["email_domain_valid"].sum()
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
    print(f"  Data quality distribution:")
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
        "university", "province", "club_name", "club_description", "club_website",
        "club_category", "contact_name", "contact_role", "contact_email",
        "contact_phone", "is_fallback_contact", "email_format_valid",
        "email_domain_valid", "data_quality", "source_url",
    ]
    # Only include columns that exist
    final_columns = [c for c in final_columns if c in merged.columns]
    output_df = merged[final_columns].copy()

    # Fill NaN
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
    print(f"Quality breakdown:")
    for q, c in quality_counts.items():
        print(f"  {q}: {c}")

    return output_df


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Canadian Club Sponsorship Lead List Builder")
    parser.add_argument("--test", action="store_true", help="Run smoke test (Phase 0)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="Run a specific stage")
    parser.add_argument("--all", action="store_true", help="Run all stages 1-4")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-clubs", type=int, default=100, help="Max clubs per university (default 100)")
    parser.add_argument("--batch-size", type=int, default=10, help="Clubs per Chat batch in Stage 3 (default 10)")
    args = parser.parse_args()

    if not any([args.test, args.stage, args.all]):
        parser.print_help()
        sys.exit(0)

    client = ParallelClient()

    if args.test:
        success = run_smoke_test(client)
        sys.exit(0 if success else 1)

    stages_to_run = []
    if args.all:
        stages_to_run = [1, 2, 3, 4]
    elif args.stage:
        stages_to_run = [args.stage]

    for stage in stages_to_run:
        if stage == 1:
            run_stage1(client)
        elif stage == 2:
            run_stage2(client, max_clubs=args.max_clubs, resume=args.resume)
        elif stage == 3:
            run_stage3(client, batch_size=args.batch_size, resume=args.resume)
        elif stage == 4:
            run_stage4()

    print(f"\n🏁 Done! Total API calls: {client.call_count}")


if __name__ == "__main__":
    main()
