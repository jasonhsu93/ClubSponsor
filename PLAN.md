# Plan: Canadian Club Sponsorship Lead List via Parallel AI (v2)

## Objective
Build a Python pipeline to enumerate student clubs at a target Canadian university and find sponsorship contacts — exporting to CSV. Default target: **University of British Columbia**.

## v2 Changes (from v1)
- **Narrowed scope**: Single university at a time (via `--university` flag) instead of 10
- **Task Group API**: Stage 3 now uses Parallel AI's Task Group API for concurrent contact lookups (up to 2,000 runs/min) instead of sequential Chat `lite` batches
- **Better model**: `base-fast` processor ($10/1K runs, ~5 output fields) replaces `lite` ($5/1K) for higher quality
- **Fixed basis extraction**: Source URLs now correctly walk `basis[].citations[].url` nested structure
- **1 run per club**: Eliminates generic fallback email contamination from batching

## API Configuration
- **API Key**: Set via `PARALLEL_API_KEY` env var (see `.env.example`)
- **Base URL**: `https://api.parallel.ai`
- **Auth Header**: `x-api-key`
- **Beta Header**: `parallel-beta: search-extract-2025-10-10` (for Search/Extract)

## Endpoints Used
| Endpoint | Method | Use |
|---|---|---|
| `/v1beta/search` | POST | Find university club directories |
| `/v1beta/extract` | POST | Pull content from directory URLs |
| `/v1beta/chat/completions` | POST | Parse club lists (Stage 2) |
| `/v1beta/tasks/groups` | POST | Create Task Group (Stage 3) |
| `/v1beta/tasks/groups/{id}/runs` | POST | Add task runs |
| `/v1beta/tasks/groups/{id}` | GET | Poll group status |
| `/v1beta/tasks/groups/{id}/runs?include_input=true&include_output=true` | GET (SSE) | Stream results |

## Models / Processors
| Model/Processor | Web Research? | Used In | Cost |
|---|---|---|---|
| `base` (Chat) | Yes | Stage 1 (directory), Stage 2 (club parsing) | $10/1K calls |
| `base-fast` (Task) | Yes | Stage 3 (contact lookups, concurrent) | $10/1K runs |

## Pipeline Stages

### Phase 0: Smoke Test (`--test`)
- 1 Search call + 1 Chat call to verify API key & response formats

### Stage 1: Resolve Directory URL (~0-2 calls)
- If university is in `KNOWN_DIRECTORIES` dict → 0 API calls
- Otherwise: Search + Chat `base` to find the clubs directory URL
- Output: `checkpoints/stage1_universities.json`

### Stage 2: Enumerate Clubs (~3-4 calls)
- Search → Extract → Chat `base`
- Parse up to 100 clubs from directory content
- Output: `checkpoints/stage2_clubs.json`

### Stage 3: Find Contacts via Task Group (~100 task runs)
- Create Task Group → submit 1 run per club (`base-fast`) → poll → stream results
- Each run independently researches one club's sponsorship contact
- Priority: Sponsorship Coordinator → VP Sponsorship → VP Finance → VP External → President
- Output: `checkpoints/stage3_contacts.json`
- Task spec uses flat object schema with `["string", "null"]` union types, `additionalProperties: false`

### Stage 4: Validate & Export CSV (pure Python/pandas, no API calls)
- Fuzzy dedup via `thefuzz` (threshold 85)
- Email regex + DNS MX validation
- Phone normalization via `phonenumbers`
- Data quality scoring (high/medium/low/no_contact)
- Source URL propagation from `basis[].citations[].url`
- Output: `sponsorship_leads.csv`

## Estimated Cost (100 clubs at UBC)
| Stage | Calls/Runs | Cost |
|---|---|---|
| Stage 1 | 0 (known URL) | $0.00 |
| Stage 2 | ~4 (Search + Extract + Chat) | ~$0.03 |
| Stage 3 | 100 task runs | $1.00 |
| Stage 4 | 0 | $0.00 |
| **Total** | **~104** | **~$1.03** |

## CLI Usage
```bash
python lead_scraper.py --test                                   # Smoke test
python lead_scraper.py --all                                    # All stages (UBC default)
python lead_scraper.py --all --university "McGill University"    # Different university
python lead_scraper.py --stage 3 --resume                       # Resume Stage 3
python lead_scraper.py --max-clubs 50 --all                     # Cap at 50 clubs
python lead_scraper.py --all --processor lite-fast              # Use cheaper processor
```
