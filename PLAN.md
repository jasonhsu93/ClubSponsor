# Plan: Canadian Club Sponsorship Lead List via Parallel AI

## Objective
Build a Python pipeline to identify the top 10 Canadian universities with 25+ student clubs, enumerate up to 100 clubs per school, and find sponsorship contacts — exporting to CSV.

## API Configuration
- **API Key**: Set via `PARALLEL_API_KEY` env var (see `.env.example`)
- **Base URL**: `https://api.parallel.ai`
- **Auth Header**: `x-api-key`
- **Beta Header**: `parallel-beta: search-extract-2025-10-10` (for Search/Extract)

## Endpoints Used
| Endpoint | Method | Use |
|---|---|---|
| `/v1beta/search` | POST | Find university club directories, club pages |
| `/v1beta/extract` | POST | Pull content from specific URLs |
| `/v1beta/chat/completions` | POST | Structure data with `json_schema` response format |

## Models
| Model | Web Research? | Used In | Cost/call |
|---|---|---|---|
| `base` | Yes | Stage 1 (universities), Stage 2 (club parsing) | $0.01 |
| `lite` | Yes | Stage 3 (contact lookups, batched) | $0.005 |

## Pipeline Stages

### Phase 0: Smoke Test (`--test`)
- 1 Search call + 1 Chat call to verify API key & response formats
- Must pass before proceeding

### Stage 1: Identify Universities (~3 Chat `base` calls)
- Ask Chat `base` for top 15 Canadian universities by club count
- Filter to top 10 with 25+ clubs
- Output: `checkpoints/stage1_universities.json`
- Schema: `{university, province, est_club_count, clubs_directory_url, source_urls}`

### Stage 2: Enumerate Clubs (~30-40 calls)
- Per university: Search → Extract → Chat `base`
- Parse up to 100 clubs per school from directory content
- Output: `checkpoints/stage2_clubs.json`
- Schema: `{university, province, club_name, club_description, club_website, club_category}`

### Stage 3: Find Contacts (~100 Chat `lite` calls)
- Batch 10 clubs per Chat `lite` call
- Priority: Sponsorship Coordinator → VP Sponsorship → VP Finance → VP External → President
- Output: `checkpoints/stage3_contacts.json`
- Schema: `{club_name, contact_name, contact_role, contact_email, contact_phone, is_fallback_contact}`

### Stage 4: Validate & Export CSV (pure Python/pandas)
- Fuzzy dedup via `thefuzz` (threshold 85)
- Email regex validation + DNS MX lookup
- Phone normalization via `phonenumbers`
- Data quality scoring (high/medium/low)
- Source URL propagation from `basis` citations
- Output: `sponsorship_leads.csv`

## Estimated Budget: ~140 calls, ~$0.73

## CLI Usage
```bash
python lead_scraper.py --test          # Phase 0: smoke test
python lead_scraper.py --stage 1       # Run stage 1 only
python lead_scraper.py --stage 2       # Run stage 2 only
python lead_scraper.py --stage 3       # Run stage 3 only
python lead_scraper.py --stage 4       # Run stage 4 only
python lead_scraper.py --all           # Run stages 1-4
python lead_scraper.py --stage 2 --resume  # Resume stage 2 from checkpoint
python lead_scraper.py --max-clubs 50  # Cap clubs per school
```
