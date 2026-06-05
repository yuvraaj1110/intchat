# Implementation Plan: Knowledge Expansion Pipeline

**Spec:** `docs/superpowers/specs/2026-06-05-knowledge-expansion-design.md`
**Branch:** `knowledge-expansion`

---

## Task 1: Dependencies & scaffolding
- Add `trafilatura`, `pypdf`, `pyyaml` to `requirements.txt`
- `pip install` them
- Create `datasets/pdfs/` directory with `.gitkeep`
- Create starter `sources.yaml` with 2-3 entries for testing
- **Verify:** `python3 -c "import trafilatura, pypdf, yaml; print('OK')"`

## Task 2: Config updates
- Add to `config.py`: `SOURCES_YAML`, `PDFS_DIR`, `NORMALIZED_ALL`
- Add new known terms to `KNOWN_TERMS`: FICA, 1040-NR, W-4, Glacier, ITIN, W-8BEN, I-94
- **Tests:** `test_config.py` — verify new constants exist
- **Verify:** existing 26 tests still pass

## Task 3: HTML fetcher (`app/fetch_html.py`)
- `fetch(url, timeout=30) -> str | None`
- Browser-like User-Agent header
- Returns None + logs warning on failure
- **Tests:** `test_fetch_html.py` — mock requests.get, test success, 404, timeout, network error

## Task 4: PDF extractor (`app/fetch_pdf.py`)
- `extract(path) -> str | None`
- Uses `pypdf.PdfReader`
- Returns None + logs warning if file missing/unreadable
- **Tests:** `test_fetch_pdf.py` — create tiny PDF fixture, test extraction + missing file

## Task 5: HTML cleaner (`app/parse_html.py`)
- `extract_content(raw_html) -> str | None`
- Uses `trafilatura.extract()`
- Returns None if no meaningful content found
- **Tests:** `test_parse_html.py` — HTML with nav/footer boilerplate, verify only article text survives

## Task 6: Web normalizer (`app/web_normalizer.py`)
- `normalize_web_source(text, source_entry) -> list[dict]`
- Splits by headings/paragraphs targeting ~800 chars
- Stamps each doc with `source_url`, `source_name`, `fetched_at`
- IDs: `{source_key}__{slug}__{index}`
- **Tests:** `test_web_normalizer.py` — verify schema match, provenance fields, long text splits, short text passthrough

## Task 7: Orchestrator (`app/build_kb.py`)
- Entry point: `python3 -m app.build_kb [--reset]`
- Load sources.yaml → fetch → parse → normalize → merge with hand-written JSON → write normalized_all.json → call ingest pipeline
- Print summary report
- **Tests:** `test_build_kb.py` — mock fetchers, verify merge logic and ingest handoff

## Task 8: Provenance in retrieval & answers
- `chain.py` `format_context()`: include `source_url` + `fetched_at` when present
- `prompts.py`: add rule 6 for source citation
- `config.py` `KNOWN_TERMS`: confirm expanded list works with retriever
- **Tests:** update `test_chain.py` with provenance-bearing docs, update `test_prompts.py` to verify new rule

## Task 9: Populate sources.yaml & build
- Fill `sources.yaml` with full initial source list (9 gov + 6 Purdue)
- Run `python3 -m app.build_kb --reset`
- Verify chunk count, spot-check answers
- **Verify:** all tests pass, chatbot answers with citations

## Task 10: Final verification & push
- Run full test suite
- Manual smoke test: 3-4 questions covering old + new sources
- Commit, push, update README with new `build_kb` instructions
