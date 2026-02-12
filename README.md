# ClubSponsor

Build a CSV lead list of sponsorship contacts for student clubs at top Canadian universities using [Parallel AI](https://parallel.ai).

## First-Time Setup

```bash
# 1. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Parallel AI API key
cp .env.example .env
# Edit .env and replace your_api_key_here with your real key
```

## Running the Pipeline

Each stage checkpoints its output so you can stop/resume at any point.

```bash
# Smoke test — verify API key works (1 Search + 1 Chat call)
python lead_scraper.py --test

# Run all 4 stages for the top 10 QS-ranked Canadian universities
python lead_scraper.py --all

# Limit to top 5 universities
python lead_scraper.py --all --top-n 5

# Target a single university (skips discovery)
python lead_scraper.py --all --university "McGill University"

# Run stages individually
python lead_scraper.py --stage 1              # Discover universities
python lead_scraper.py --stage 2              # Enumerate clubs
python lead_scraper.py --stage 3              # Find contacts
python lead_scraper.py --stage 4              # Validate & export CSV

# Resume Stage 3 if interrupted
python lead_scraper.py --stage 3 --resume

# Tune performance
python lead_scraper.py --all --max-clubs 30          # Cap clubs per university
python lead_scraper.py --all --processor lite-fast   # Cheaper Task processor
python lead_scraper.py --all --stage2-timeout 60     # More time per university
```

## Pipeline Stages

| Stage | What it does | Key technique |
|-------|-------------|---------------|
| **1. Discover** | Find top N Canadian universities by QS ranking | FindAll → Chat → hardcoded fallback |
| **2. Enumerate** | List clubs at each university | Sitemap (UBC) or Search→Extract→Chat |
| **3. Contacts** | Find sponsorship contacts per club | Extract (amsclubs.ca) + Task Group (others) |
| **4. Export** | Validate emails, deduplicate, score quality | MX checks, fuzzy dedup, CSV export |

## Output

`sponsorship_leads.csv` — one row per club with columns:

| Column | Description |
|--------|-------------|
| `university` | University name |
| `province` | Province |
| `club_name` | Club name |
| `club_description` | Brief description |
| `club_website` | Club URL |
| `club_category` | Category (Academic, Cultural, Sports, etc.) |
| `contact_name` | Sponsorship contact's full name |
| `contact_role` | Their title/role |
| `contact_email` | Email address |
| `contact_phone` | Phone (E.164 format) |
| `is_fallback_contact` | `True` if not a sponsorship-specific role |
| `email_format_valid` | Regex validation result |
| `email_domain_valid` | DNS MX record check result |
| `data_quality` | `high` / `medium` / `low` / `no_contact` |
| `source_url` | Citation URL from Parallel AI |

## Project Structure

```
lead_scraper.py     # 4-stage pipeline + CLI (v3)
api_client.py       # Parallel AI API wrapper (Search, Extract, Chat, Task Group, FindAll)
requirements.txt    # Python dependencies
.env.example        # Template for API key
PLAN.md             # Architecture & API reference
checkpoints/        # Stage output JSONs (gitignored)
logs/               # API call logs (gitignored)
```