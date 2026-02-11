# ClubSponsor

Build a CSV lead list of sponsorship contacts for student clubs at the top 10 Canadian universities

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

# Run all 4 stages end-to-end
python lead_scraper.py --all

# Or run stages individually
python lead_scraper.py --stage 1              # Identify top 10 universities
python lead_scraper.py --stage 2 --max-clubs 100   # Enumerate clubs (≤100/school)
python lead_scraper.py --stage 3 --batch-size 10    # Look up sponsorship contacts
python lead_scraper.py --stage 4              # Validate, deduplicate, export CSV

# Resume a stage that was interrupted
python lead_scraper.py --stage 3 --resume
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
api_client.py       # Parallel AI API wrapper (Search, Extract, Chat)
lead_scraper.py     # 4-stage pipeline + CLI
requirements.txt    # Python dependencies
PLAN.md             # Architecture & API reference
.env.example        # Template for API key
checkpoints/        # Stage output JSONs (gitignored)
logs/               # API call logs (gitignored)
```