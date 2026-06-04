"""
Normalizer for the international student chatbot knowledge base.

Reads 3 source datasets (each with a different schema) and produces a single
flat list of documents ready for embedding and vector-store ingestion.

Source schemas handled:
  A  – categories[].topics[].content_blocks[]   (general student life)
  B  – topics[] with eligibility/rules/steps     (F-1 employment & compliance)
  QA – categories[].entries[].{question, answer}  (mentor-style Q&A)

Output schema (one JSON array):
  [
    {
      "id":       "dataset_A__sevis_overview__0",
      "text":     "...",           # the content to embed
      "category": "SEVIS System Overview",
      "topic":    "SEVIS Purpose and Management",
      "type":     "paragraph|list|note|rule|step|qa|...",
      "source":   "dataset_A",
      "metadata": { ... }         # extra fields preserved per-format
    },
    ...
  ]
"""

import json
import os
from pathlib import Path
from datetime import date

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_FILE = DATASETS_DIR / "normalized_dataset.json"

# Only the canonical (deduplicated) sources:
#   archived.json == dataset_B == jsonraw/f1_employment  → keep dataset_B only
#   nextregen.json == mentorstyle.json                   → keep mentorstyle only
SOURCES = {
    "dataset_A": DATASETS_DIR / "dataset_A_phase2_general_intl_student_life.json",
    "dataset_B": DATASETS_DIR / "dataset_B_phase3_f1_employment_and_compliance.json",
    "mentorstyle": DATASETS_DIR / "mentorstyle.json",
}


# ── Format A: categories → topics → content_blocks ─────────────────────────
def normalize_format_A(data: dict, source: str) -> list[dict]:
    """Dataset A: nested categories/topics/content_blocks."""
    docs = []
    for cat in data.get("categories", []):
        cat_title = cat.get("title", "")
        for topic in cat.get("topics", []):
            topic_id = topic.get("id", "unknown")
            topic_title = topic.get("title", "")
            meta = {
                "related_forms": topic.get("related_forms", []),
                "related_visa_types": topic.get("related_visa_types", []),
                "audience": topic.get("audience", []),
                "tags": topic.get("tags", []),
                "source_urls": topic.get("source_urls", []),
            }

            for idx, block in enumerate(topic.get("content_blocks", [])):
                block_type = block.get("type", "paragraph")
                text = block.get("text")

                # Lists get joined into a single doc (bullet points)
                if isinstance(text, list):
                    text = "\n• ".join([""] + text).strip()
                if not text:
                    continue

                docs.append({
                    "id": f"{source}__{topic_id}__{idx}",
                    "text": text,
                    "category": cat_title,
                    "topic": topic_title,
                    "type": block_type,
                    "source": source,
                    "metadata": meta,
                })
    return docs


# ── Format B: flat topics with structured fields ───────────────────────────
def normalize_format_B(data: dict, source: str) -> list[dict]:
    """Dataset B: flat topics[] with eligibility, rules, steps, etc."""
    docs = []
    for topic in data.get("topics", []):
        topic_id = topic.get("topic_id", "unknown")
        topic_title = topic.get("title", "")
        category = topic.get("category", "")
        meta = {
            "audience": topic.get("audience", []),
            "source_urls": topic.get("source_urls", []),
            "required_forms": topic.get("required_forms", []),
        }

        # Summary → one doc
        if topic.get("summary"):
            docs.append({
                "id": f"{source}__{topic_id}__summary",
                "text": topic["summary"],
                "category": category,
                "topic": topic_title,
                "type": "summary",
                "source": source,
                "metadata": meta,
            })

        # Eligibility → one doc (bulleted)
        if topic.get("eligibility"):
            text = f"Eligibility for {topic_title}:\n• " + "\n• ".join(topic["eligibility"])
            docs.append({
                "id": f"{source}__{topic_id}__eligibility",
                "text": text,
                "category": category,
                "topic": topic_title,
                "type": "eligibility",
                "source": source,
                "metadata": meta,
            })

        # Process steps → one doc (numbered)
        if topic.get("process_steps"):
            steps = [f"{i+1}. {s}" for i, s in enumerate(topic["process_steps"])]
            text = f"Steps for {topic_title}:\n" + "\n".join(steps)
            docs.append({
                "id": f"{source}__{topic_id}__steps",
                "text": text,
                "category": category,
                "topic": topic_title,
                "type": "steps",
                "source": source,
                "metadata": meta,
            })

        # Key rules → one doc (bulleted)
        if topic.get("key_rules"):
            text = f"Key rules for {topic_title}:\n• " + "\n• ".join(topic["key_rules"])
            docs.append({
                "id": f"{source}__{topic_id}__rules",
                "text": text,
                "category": category,
                "topic": topic_title,
                "type": "rules",
                "source": source,
                "metadata": meta,
            })

        # Timelines → one doc
        if topic.get("timelines"):
            tl = topic["timelines"]
            if isinstance(tl, dict):
                parts = [f"{k}: {v}" for k, v in tl.items()]
                text = f"Timelines for {topic_title}:\n• " + "\n• ".join(parts)
            else:
                text = f"Timelines for {topic_title}: {tl}"
            docs.append({
                "id": f"{source}__{topic_id}__timelines",
                "text": text,
                "category": category,
                "topic": topic_title,
                "type": "timelines",
                "source": source,
                "metadata": meta,
            })

        # Common mistakes → one doc
        if topic.get("common_mistakes"):
            text = f"Common mistakes with {topic_title}:\n• " + "\n• ".join(topic["common_mistakes"])
            docs.append({
                "id": f"{source}__{topic_id}__mistakes",
                "text": text,
                "category": category,
                "topic": topic_title,
                "type": "mistakes",
                "source": source,
                "metadata": meta,
            })

        # Consequences → one doc
        if topic.get("consequences_of_violation"):
            text = f"Consequences of violating {topic_title} rules:\n• " + "\n• ".join(topic["consequences_of_violation"])
            docs.append({
                "id": f"{source}__{topic_id}__consequences",
                "text": text,
                "category": category,
                "topic": topic_title,
                "type": "consequences",
                "source": source,
                "metadata": meta,
            })

    return docs


# ── Format QA: categories → entries with question/answer ───────────────────
def normalize_format_QA(data: dict, source: str) -> list[dict]:
    """Mentor-style Q&A: categories[].entries[].{question, answer}."""
    docs = []
    phase = data.get("phase", "")
    tone = data.get("tone", "")

    for cat in data.get("categories", []):
        cat_title = cat.get("category", "")
        for entry in cat.get("entries", []):
            entry_id = entry.get("id", "unknown")
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            if not answer:
                continue

            # Combine Q+A so the embedding captures both the question intent
            # and the answer content
            text = f"Q: {question}\nA: {answer}"
            docs.append({
                "id": f"{source}__{entry_id}",
                "text": text,
                "category": cat_title,
                "topic": question,
                "type": "qa",
                "source": source,
                "metadata": {
                    "phase": phase,
                    "tone": tone,
                    "question": question,
                    "answer": answer,
                },
            })
    return docs


# ── Auto-detect format and dispatch ────────────────────────────────────────
def detect_and_normalize(data: dict, source: str) -> list[dict]:
    """Pick the right normalizer based on the shape of the data."""
    # QA format: has 'categories' with 'entries' that have 'question'/'answer'
    cats = data.get("categories", [])
    if cats and "entries" in cats[0] and "question" in cats[0]["entries"][0]:
        return normalize_format_QA(data, source)

    # Format A: has 'categories' with 'topics' that have 'content_blocks'
    if cats and "topics" in cats[0] and "content_blocks" in cats[0]["topics"][0]:
        return normalize_format_A(data, source)

    # Format B: has top-level 'topics' with 'eligibility'/'process_steps'
    topics = data.get("topics", [])
    if topics and ("eligibility" in topics[0] or "process_steps" in topics[0]):
        return normalize_format_B(data, source)

    print(f"  ⚠ Unknown format in {source}, skipping")
    return []


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    all_docs = []

    for source_name, file_path in SOURCES.items():
        if not file_path.exists():
            print(f"  ✗ Not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = detect_and_normalize(data, source_name)
        print(f"  ✓ {source_name}: {len(docs)} documents")
        all_docs.extend(docs)

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Total documents: {len(all_docs)}")
    print(f"Output: {OUTPUT_FILE}")

    # Summary by source
    from collections import Counter
    by_source = Counter(d["source"] for d in all_docs)
    by_type = Counter(d["type"] for d in all_docs)
    print(f"\nBy source: {dict(by_source)}")
    print(f"By type:   {dict(by_type)}")


if __name__ == "__main__":
    main()
