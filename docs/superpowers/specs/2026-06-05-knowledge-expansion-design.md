# Knowledge Expansion: Web & PDF Ingestion Pipeline

**Date:** 2026-06-05
**Status:** Approved
**Goal:** Broaden the chatbot's knowledge base from 129 hand-written docs to hundreds of authoritative documents pulled from live government websites and Purdue University's ISS office, with full provenance (source URL + fetch date) surfaced in every answer.

---

## Architecture

The existing pipeline stays intact. A new **fetch → parse → normalize** front-end feeds into it:

```
sources.yaml  ─┐
(URLs + PDFs)  │
               ▼
        ┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
        │ fetch layer  │ ──► │ parse layer  │ ──► │ web_normalizer     │
        │ html / pdf   │     │ clean text   │     │ → normalized docs  │
        └─────────────┘     └──────────────┘     │   + provenance     │
                                                  └─────────┬──────────┘
                                                            ▼
   hand-written JSON (existing) ──────────────►  normalized_dataset (merged)
                                                            ▼
                                          EXISTING: dedup → chunk → Chroma
                                                            ▼
                                          EXISTING: hybrid retriever → LLM
                                                            ▼
                                   answer + [Source: URL — fetched DATE]
```

**Design principle:** Every source (HTML, PDF, hand-written JSON) converges on the same normalized doc schema. Everything downstream — dedup, chunking, retrieval, LLM — is source-agnostic.

---

## New Components

### 1. `sources.yaml` — source registry

A manually maintained list of URLs and local PDF paths. Each entry specifies:

```yaml
- name: "USCIS — Optional Practical Training"
  type: html
  url: "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students"
  category: "OPT"

- name: "Purdue ISS — F-1 Employment"
  type: html
  url: "https://www.purdue.edu/iss/employment/"
  category: "EMPLOYMENT"

- name: "IRS Publication 519"
  type: pdf
  path: "datasets/pdfs/irs_p519.pdf"
  category: "TAXES"
```

Fields:
- `name` (str, required): human-readable source label, surfaced in citations
- `type` (str, required): `"html"` or `"pdf"`
- `url` (str, required for html): full URL to fetch
- `path` (str, required for pdf): local path relative to project root
- `category` (str, required): maps to existing category taxonomy

### 2. `app/fetch_html.py` — HTML fetcher

Single function:

```python
def fetch(url: str, timeout: int = 30) -> str | None
```

- Uses `requests.get()` with a browser-like User-Agent
- Returns raw HTML on success, `None` on failure (timeout, 404, network error)
- Logs warnings on failure; never raises — the orchestrator skips failed sources

### 3. `app/fetch_pdf.py` — PDF text extractor

Single function:

```python
def extract(path: str) -> str | None
```

- Uses `pypdf.PdfReader` to concatenate all page text
- Returns full text on success, `None` if file missing or unreadable
- Same error contract as `fetch_html`: return None, log, never raise

### 4. `app/parse_html.py` — HTML cleaner

Single function:

```python
def extract_content(raw_html: str) -> str | None
```

- Uses `trafilatura.extract()` to strip boilerplate (nav, footer, ads, cookie banners)
- Returns clean article text, `None` if extraction finds no meaningful content
- This is the most critical quality gate: garbage HTML → garbage retrieval

### 5. `app/web_normalizer.py` — normalized doc emitter

```python
def normalize_web_source(
    text: str,
    source_entry: dict,  # one entry from sources.yaml
) -> list[dict]
```

- Splits long text into logical sections (by headings or paragraph boundaries, targeting ~800 chars to match CHUNK_SIZE)
- Each doc follows the existing normalized schema:
  ```json
  {
    "id": "{source_key}__{topic_slug}__{index}",
    "text": "...",
    "category": "OPT",
    "topic": "Optional Practical Training (OPT)",
    "type": "paragraph",
    "source": "uscis",
    "metadata": {
      "source_url": "https://www.uscis.gov/...",
      "source_name": "USCIS — Optional Practical Training",
      "fetched_at": "2026-06-05"
    }
  }
  ```
- The `source` field uses a short key derived from the source name (e.g., "uscis", "purdue_iss", "irs")
- `fetched_at` is stamped at run time (ISO date string)

### 6. `app/build_kb.py` — orchestrator

Entry point: `python3 -m app.build_kb [--reset]`

Flow:
1. Load `sources.yaml`
2. For each entry: fetch (html or pdf) → parse → normalize → collect docs
3. Load existing `normalized_dataset.json` (hand-written docs)
4. Merge web docs + hand-written docs into one list
5. Write merged output to `datasets/normalized_all.json`
6. Call existing `app.ingest` pipeline (dedup → chunk → Chroma)
7. Print summary: sources processed, sources skipped, total docs, total chunks

The `--reset` flag passes through to `ingest` to wipe and rebuild Chroma.

---

## Changes to Existing Modules

### `app/config.py`
- Add `SOURCES_YAML = BASE_DIR / "sources.yaml"`
- Add `PDFS_DIR = BASE_DIR / "datasets" / "pdfs"`
- Add `NORMALIZED_ALL = BASE_DIR / "datasets" / "normalized_all.json"` (merged output)
- Update `NORMALIZED_DATASET` usage: `ingest.py` reads `NORMALIZED_ALL` when it exists, falls back to `NORMALIZED_DATASET`

### `app/ingest.py`
- `_to_documents()`: pass through `source_url`, `source_name`, `fetched_at` as metadata fields (already flattens metadata to scalars, so this works automatically)
- No other changes needed — the schema is compatible

### `app/chain.py` — `format_context()`
- Extend context block to include provenance when available:
  ```
  [OPT | USCIS — uscis.gov/... | 2026-06-05] F-1 students may apply...
  ```

### `app/prompts.py` — system prompt
- Add rule 6: "When source URLs are available in the context, end your answer with a 'Sources' section listing each unique URL and its retrieval date."

### `app/retriever.py` — `KNOWN_TERMS`
- Extend the known-terms list in `config.py` with any new immigration terms that appear in the expanded corpus (e.g., FICA, 1040-NR, W-4, Glacier, ITIN)

---

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `trafilatura` | `>=1.8.0,<2` | HTML boilerplate removal |
| `pypdf` | `>=4.0.0,<5` | PDF text extraction |
| `pyyaml` | `>=6.0,<7` | Parse `sources.yaml` |

All free, pure-Python, no external services.

---

## Initial Source List

### Government (universal)
1. USCIS — OPT for F-1 Students
2. USCIS — STEM OPT Extension
3. USCIS — Students and Exchange Visitors (overview)
4. Study in the States (DHS/SEVP) — Working in the U.S.
5. Study in the States — SEVIS Overview
6. Study in the States — Travel
7. IRS — Taxation of Nonresident Aliens (PDF or web)
8. SSA — Social Security Numbers for Noncitizens
9. State Dept — Student Visa (F and M)

### Purdue-specific
10. Purdue ISS — F-1 Employment
11. Purdue ISS — CPT
12. Purdue ISS — OPT
13. Purdue ISS — Travel & Re-entry
14. Purdue ISS — Maintaining Status
15. Purdue ISS — New Student Checklist

This is a starting set. Adding more is just adding lines to `sources.yaml`.

---

## Citation Format (student-facing)

```
Assistant: F-1 students are eligible for 12 months of OPT after
completing their degree. STEM-designated programs may apply for
a 24-month extension...

Sources:
• USCIS — uscis.gov/opt-for-f1-students (retrieved 2026-06-05)
• Purdue ISS — purdue.edu/iss/employment (retrieved 2026-06-05)

This is general guidance, not legal advice. Always confirm
with your DSO or an immigration attorney.
```

Hand-written docs without a URL cite by topic name only (existing behavior).

---

## Testing Strategy

- `test_fetch_html.py`: mock `requests.get`, verify success/failure paths, User-Agent header
- `test_fetch_pdf.py`: test with a small real PDF fixture, verify text extraction
- `test_parse_html.py`: test with raw HTML fixture, verify boilerplate stripped
- `test_web_normalizer.py`: verify output schema matches normalized format, provenance fields present, long text splits correctly
- `test_build_kb.py`: integration test with mocked fetchers, verify merge + handoff to ingest
- Existing tests: all 26 must keep passing (ingest, retriever, chain, etc.)

---

## What This Does NOT Do (deliberate scope cuts)

- **No auto-refresh / cron**: you run `build_kb` manually when you want to update
- **No content caching / change detection**: every run re-fetches everything (phase 2 optimization)
- **No site-wide crawling**: you list specific URLs, not "crawl all of uscis.gov"
- **No multi-university support**: Purdue only; no per-school routing
- **No LLM-generated content**: all docs come from real sources with real URLs
