# ClubSponsor

Build a CSV lead list of sponsorship contacts for student clubs at a Canadian university using [Parallel AI](https://parallel.ai).

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

# Run all 4 stages end-to-end (defaults to UBC)
python lead_scraper.py --all

# Target a different university
python lead_scraper.py --all --university "McGill University"

# Or run stages individually
python lead_scraper.py --stage 1              # Resolve directory URL
python lead_scraper.py --stage 2              # Enumerate clubs (≤100)
python lead_scraper.py --stage 3              # Find contacts (Task Group API)
python lead_scraper.py --stage 4              # Validate, deduplicate, export CSV

# Resume Stage 3 if interrupted
python lead_scraper.py --stage 3 --resume

# Cap clubs or use a different Task processor
python lead_scraper.py --all --max-clubs 50
python lead_scraper.py --all --processor lite-fast
```

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
api_client.py       # Parallel AI API wrapper (Search, Extract, Chat, Task Group)
lead_scraper.py     # 4-stage pipeline + CLI (v2)
requirements.txt    # Python dependencies
PLAN.md             # Architecture & API reference
.env.example        # Template for API key
checkpoints/        # Stage output JSONs (gitignored)
logs/               # API call logs (gitignored)
```