"""
ingest_benchmark.py — builds/rebuilds the ChromaDB collection for a CIS benchmark.

Creates the CIS benchmark embeddings used by Agent 2 (agent_a2_llm.py).
Chunking is structure-aware with a 1024-token threshold.

Each Ansible project is audited against a specific CIS Benchmark version
(a different operating system = a different document = a different
collection), so the collection name is a parameter.

Usage:
  python ingest_benchmark.py --analysis=analysis_results.json --collection=cis_flat
  python ingest_benchmark.py --analysis=analysis_results_rocky9.json --collection=cis_flat_rocky9

By default an existing collection is overwritten (use --no-reset to avoid
this, in which case duplicate ids must be prevented manually).
"""

import re
import sys
import json
import argparse

import chromadb
import torch
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "BAAI/bge-m3"

SUBSECTION_RE = re.compile(
    r"(Description:|Rationale:|Impact:|Audit:|Remediation:|References:|CIS Controls:)",
    re.MULTILINE,
)


def build_chunks(rules):
    """Structure-aware chunking: a rule <= 1024 tokens -> 1 _full chunk, a
    longer one -> up to 3 chunks (_context/_audit/_remediation) split on CIS
    section headings. Every chunk stores the rule's full title in its
    metadata."""
    chunks = []

    for rule in rules:
        rule_id = rule["rule_id"]
        title   = rule["title"]
        text    = rule.get("full_text", "")
        tokens  = rule["token_count"]
        prefix  = f"{rule_id} {title[:120]}\n\n"

        if tokens <= 1024:
            chunks.append({
                "chunk_id":   f"{rule_id}_full",
                "rule_id":    rule_id,
                "title":      title,
                "chunk_type": "full",
                "section_id": rule_id.split(".")[0],
                "text":       prefix + text,
            })
            continue

        matches = list(SUBSECTION_RE.finditer(text))
        if not matches:
            chunks.append({
                "chunk_id":   f"{rule_id}_full",
                "rule_id":    rule_id,
                "title":      title,
                "chunk_type": "full",
                "section_id": rule_id.split(".")[0],
                "text":       prefix + text,
            })
            continue

        subs = {}
        for i, m in enumerate(matches):
            header = m.group(1).replace(":", "").strip()
            start  = m.start()
            end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            subs[header] = text[start:end].strip()

        for chunk_type, headers in [
            ("context",     ["Description", "Rationale", "Impact"]),
            ("audit",       ["Audit"]),
            ("remediation", ["Remediation"]),
        ]:
            content = "\n\n".join(subs[h] for h in headers if h in subs)
            if content:
                chunks.append({
                    "chunk_id":   f"{rule_id}_{chunk_type}",
                    "rule_id":    rule_id,
                    "title":      title,
                    "chunk_type": chunk_type,
                    "section_id": rule_id.split(".")[0],
                    "text":       prefix + content,
                })

    return chunks


def ingest(analysis_path: str, collection_name: str, reset: bool = True):
    print(f"[ingest] Loading {analysis_path}...")
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    rules = data["rules"] if isinstance(data, dict) and "rules" in data else data
    print(f"[ingest] {len(rules)} rules in the benchmark")

    chunks = build_chunks(rules)
    print(f"[ingest] Built {len(chunks)} chunks")

    # Uses Apple Silicon MPS when available, with a CPU fallback.
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[ingest] Loading model {EMBED_MODEL} (device={_device})...")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=_device)

    texts = [c["text"] for c in chunks]
    print(f"[ingest] Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True,
                              normalize_embeddings=True)

    client = chromadb.PersistentClient(CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[ingest] Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    col = client.get_or_create_collection(collection_name)
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{
            "rule_id":    c["rule_id"],
            "title":      c["title"],
            "chunk_type": c["chunk_type"],
            "section_id": c["section_id"],
        } for c in chunks],
    )

    print(f"[ingest] Done: collection '{collection_name}' contains {col.count()} vectors "
          f"(path: {CHROMA_PATH})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True, help="JSON file from the benchmark PDF analysis (analyze_pdf_docling.py)")
    p.add_argument("--collection", required=True, help="name of the target ChromaDB collection")
    p.add_argument("--no-reset", action="store_true", help="do not delete an existing collection before writing")
    args = p.parse_args()

    ingest(args.analysis, args.collection, reset=not args.no_reset)
