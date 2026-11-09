"""
analyze_pdf_docling.py — structure-aware CIS Benchmark parsing via docling.

docling uses a layout-aware model to convert the PDF into Markdown, with
headings for rule titles and subsections, real tables and shell commands in
code blocks. Rules and their subsections are then identified by headings,
without guessing rule boundaries from flat text.

Usage:
  python3 analyze_pdf_docling.py --md cis_rhel10_docling.md --output analysis_results.json
  (the .md file is produced separately by docling's DocumentConverter;
  converting a 1000+ page PDF takes from tens of minutes to a few hours,
  so it is run once, outside this script.)

Output schema: summary/sections/rules, each rule with
rule_id/section_id/title/token_count/char_count/subsections/full_text,
used by ingest_benchmark.py, agent_a1_onboarding.py and the judge scripts.
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from collections import defaultdict

import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# docling renders each rule's own title as a markdown heading whose text
# starts with the rule's dotted number, e.g. "## 2.1.12 Ensure rsync
# services are not in use (Automated)" — 1 to 4 dots covers every CIS
# benchmark numbering depth observed (x.x through x.x.x.x.x).
RULE_HEADING_RE = re.compile(
    r"^#{1,3}\s+(\d+(?:\.\d+){1,4})\s+(.+?)\s*$",
    re.MULTILINE,
)

# Subsections are matched as markdown headings by name (e.g.
# "## Remediation:"), not as a substring anywhere in the rule text.
SUBSECTION_NAMES = [
    "Description", "Rationale", "Impact", "Audit",
    "Remediation", "References", "CIS Controls"
]
SUBSECTION_HEADING_RE = re.compile(
    r"^#{1,3}\s+(" + "|".join(re.escape(s) for s in SUBSECTION_NAMES) + r")\s*:?\s*$",
    re.MULTILINE,
)


@dataclass
class SectionStats:
    section_id: str
    rule_count: int = 0
    token_count: int = 0
    avg_tokens: int = 0
    min_tokens: int = 0
    max_tokens: int = 0


def split_into_rules(md_text: str) -> list:
    """Same overall strategy as analyze_pdf.split_into_rules() (find every
    rule-numbered heading, slice the text between consecutive matches) —
    but matching a proper Markdown heading instead of a free-text regex
    guess, so a rule number that merely APPEARS inside a paragraph
    (references, cross-mentions) can no longer be mistaken for a rule
    boundary the way it could in the old flat-text approach."""
    matches = list(RULE_HEADING_RE.finditer(md_text))
    print(f"Found {len(matches)} rule-heading candidates")

    candidates = []
    for i, m in enumerate(matches):
        rule_id = m.group(1)
        title = f"{rule_id} {m.group(2)}"
        section_id = rule_id.split(".")[0]
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        text = md_text[start:end].strip()
        candidates.append((rule_id, title, section_id, text))
    return candidates


def select_best_per_rule(candidates: list) -> list:
    """Mirrors analyze_pdf.select_best_per_rule()'s selection logic
    (prefer the shortest >100-token entry that has a real Remediation
    subsection — filters out short table-of-contents/appendix duplicates
    of the same rule number) but detects subsections via
    SUBSECTION_HEADING_RE (exact heading match) instead of a substring
    check against the whole text."""
    grouped = defaultdict(list)
    for rule_id, title, section_id, text in candidates:
        token_count = len(TOKENIZER.encode(text))
        subsections = list(dict.fromkeys(SUBSECTION_HEADING_RE.findall(text)))
        grouped[rule_id].append({
            "rule_id": rule_id, "title": title, "section_id": section_id,
            "text": text, "token_count": token_count, "subsections": subsections,
        })

    best = []
    for rule_id, versions in grouped.items():
        with_content = [v for v in versions if len(v["subsections"]) > 0]
        if not with_content:
            continue
        with_rem = [v for v in with_content if "Remediation" in v["subsections"]]
        if with_rem:
            substantial = [v for v in with_rem if v["token_count"] > 100]
            best.append(min(substantial or with_rem, key=lambda x: x["token_count"]))
            continue
        best.append(max(with_content, key=lambda x: len(x["subsections"])))

    best.sort(key=lambda x: [int(p) for p in x["rule_id"].split(".")])
    print(f"After selection: {len(best)} unique rules")
    return best


def compute_section_stats(rules: list) -> list:
    grouped = defaultdict(list)
    for r in rules:
        grouped[r["section_id"]].append(r["token_count"])
    stats = []
    for section_id, tokens in sorted(grouped.items(), key=lambda x: int(x[0])):
        stats.append(SectionStats(
            section_id=section_id, rule_count=len(tokens), token_count=sum(tokens),
            avg_tokens=sum(tokens) // len(tokens), min_tokens=min(tokens), max_tokens=max(tokens),
        ))
    return stats


def save_results(rules: list, section_stats: list, output_path: str):
    tokens = [r["token_count"] for r in rules]
    results = {
        "summary": {
            "total_rules": len(rules),
            "avg_tokens_per_rule": sum(tokens) // len(tokens),
            "median_tokens_per_rule": sorted(tokens)[len(tokens) // 2],
            "pct_under_512": round(sum(1 for t in tokens if t <= 512) / len(tokens) * 100, 1),
            "pct_under_768": round(sum(1 for t in tokens if t <= 768) / len(tokens) * 100, 1),
            "pct_over_1024": round(sum(1 for t in tokens if t > 1024) / len(tokens) * 100, 1),
            "complete_rules": sum(1 for r in rules
                                  if "Description" in r["subsections"] and "Remediation" in r["subsections"]),
            "parser": "docling",
        },
        "sections": [
            {"section_id": s.section_id, "rule_count": s.rule_count, "token_count": s.token_count,
             "avg_tokens": s.avg_tokens, "min_tokens": s.min_tokens, "max_tokens": s.max_tokens}
            for s in section_stats
        ],
        "rules": [
            {"rule_id": r["rule_id"], "section_id": r["section_id"], "title": r["title"][:120],
             "token_count": r["token_count"], "char_count": len(r["text"]),
             "subsections": r["subsections"], "full_text": r["text"]}
            for r in rules
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse a docling-exported CIS Benchmark Markdown file")
    parser.add_argument("--md", required=True, help="Path to the docling markdown export")
    parser.add_argument("--output", default="analysis_results.json", help="Output JSON path")
    args = parser.parse_args()

    with open(args.md, encoding="utf-8") as f:
        md_text = f.read()

    candidates = split_into_rules(md_text)
    rules = select_best_per_rule(candidates)
    if not rules:
        print("ERROR: no CIS rules found.")
        return
    section_stats = compute_section_stats(rules)

    has_desc = sum(1 for r in rules if "Description" in r["subsections"])
    has_rem = sum(1 for r in rules if "Remediation" in r["subsections"])
    complete = sum(1 for r in rules if "Description" in r["subsections"] and "Remediation" in r["subsections"])
    print(f"\nHas Description: {has_desc}/{len(rules)}  Has Remediation: {has_rem}/{len(rules)}  "
          f"Complete (both): {complete}/{len(rules)}")

    save_results(rules, section_stats, args.output)


if __name__ == "__main__":
    main()
