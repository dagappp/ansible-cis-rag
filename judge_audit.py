"""
judge_audit.py — independent re-verification of Agent 2's own COMPLIANT
audit verdicts (judge_combined.py covers only the generated fixes).

Checking only the tasks a classifier rejects says nothing about the tasks
it accepts, so the COMPLIANT verdicts are re-checked as well, which is
needed for a full confusion matrix and balanced accuracy (Collot et al.
2025, arXiv:2512.08121).

Data source: the debug-dump JSONL (DEBUG_DUMP_PATH), which stores every
Agent 2 'A2_audit' call with the task's raw_yaml and Agent 2's explanation
(final_report.json keeps only non-compliant findings).

Reuses grounded_judge_one() from model_comparison_grounded_judge.py
unchanged, applied to Agent 2's compliant verdicts.

Usage:
  python3 judge_audit.py --dump path/to/dump.jsonl --since "2026-08-31 15:00:00" \
      --project-path red-hat-linux-family-10_test [--judge=MODEL] [--out=CSV] [--limit=N]
"""

import os
import re
import sys
import csv
import json
import argparse
import statistics as stats
from datetime import datetime
from collections import defaultdict

import agent_a2_llm
from model_comparison_grounded_judge import grounded_judge_one
from judge_audit_helpers import extract_raw_yaml, extract_semantic_description  # small, testable helpers

JUDGE_MODEL = "qwen3.8-27b"


def load_compliant_audits(dump_path: str, since_ts: float = None, until_ts: float = None) -> list:
    """Reads the debug-dump JSONL and returns every 'A2_audit' entry whose
    Agent 2 verdict was COMPLIANT, at or after since_ts and (if given) at or
    before until_ts. The dump file is append-only and shared between runs,
    so --until should be given when a later run may already be writing."""
    entries = []
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("stage") != "A2_audit":
                continue
            ts = e.get("ts", 0)
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            parsed = e.get("parsed") or {}
            if parsed.get("status") != "COMPLIANT":
                continue
            entries.append(e)
    return entries


def load_all_audits_for_siblings(dump_path: str, since_ts: float = None, until_ts: float = None,
                                 task_tags: dict = None) -> dict:
    """rule_id -> [raw_yaml, ...] across every audited task in this run (any
    status), so that a task covering only part of a rule is judged together
    with its siblings. Entries are indexed under both Agent 2's own rule IDs
    and the project's declared tag for the task (task_tags: task_name ->
    [rule_id, ...]), so mistagged siblings are not lost."""
    by_rule = defaultdict(list)
    task_tags = task_tags or {}
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("stage") != "A2_audit":
                continue
            ts = e.get("ts", 0)
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            parsed = e.get("parsed") or {}
            yaml_text = extract_raw_yaml(e["messages"][1]["content"])
            if not yaml_text:
                continue
            rids = set(parsed.get("rule_ids") or [])
            rids |= set(task_tags.get(e.get("identifier", ""), []))
            for rid in rids:
                by_rule[rid].append(yaml_text)
    return by_rule


def run_audit_verification(dump_path: str, project_path: str, since: str = None, until: str = None,
                           judge_model: str = JUDGE_MODEL, out_csv: str = None,
                           limit: int = None, collection: str = "cis_flat",
                           analysis: str = "analysis_results.json") -> str:
    # grounded_judge_one() -> agent_a3_llm.get_remediation_chunk() reads
    # agent_a2_llm.COLLECTION. Must match the benchmark the audited project
    # was run against, or every Agent-2 verdict is re-checked against the
    # wrong CIS document (default 'cis_flat' = RHEL 10). Same fix as
    # judge_combined.py. Default unchanged so RHEL runs are unaffected.
    agent_a2_llm.COLLECTION = collection
    print(f"[judge-audit] grounded judge will retrieve CIS text from collection: '{collection}'")
    since_ts = None
    if since:
        since_ts = datetime.strptime(since, "%Y-%m-%d %H:%M:%S").timestamp()
    until_ts = None
    if until:
        until_ts = datetime.strptime(until, "%Y-%m-%d %H:%M:%S").timestamp()

    compliant_entries = load_compliant_audits(dump_path, since_ts, until_ts)

    # When Agent 2's own response has no rule ID, the project's declared
    # rule tag (task.cis_rule_ids from Parser A0) is used instead, matched
    # by task name. Without this fallback such entries would be silently
    # dropped, which affects runs without RAG far more often. Entries with
    # neither are skipped and their number is printed.
    _task_tags = {}
    try:
        from parser_a0_final import parse_ansible_project
        for _t in parse_ansible_project(project_path):
            if getattr(_t, "cis_rule_ids", None):
                _task_tags[_t.task_name] = list(_t.cis_rule_ids)
    except Exception as _e:
        print(f"[judge-audit] Warning: could not build task_name->cis_rule_ids fallback map "
              f"({_e}) — entries with empty rule_ids will be skipped as before")

    n_fallback_applied = 0
    n_still_empty = 0
    for e in compliant_entries:
        if not (e["parsed"].get("rule_ids") or []):
            fallback = _task_tags.get(e["identifier"])
            if fallback:
                e["parsed"]["rule_ids"] = fallback
                n_fallback_applied += 1
            else:
                n_still_empty += 1
    if n_fallback_applied or n_still_empty:
        print(f"[judge-audit] rule_id fallback: {n_fallback_applied} entries recovered via project tag, "
              f"{n_still_empty} still have no resolvable rule_id (will be skipped, not silently — "
              f"see 'still no rule_id' count below)")

    # A rule ID given by Agent 2 that does not exist in the benchmark is
    # replaced by the project's declared tag, in the same way as a missing
    # rule ID.
    try:
        with open(analysis, encoding="utf-8") as _f:
            _analysis_data = json.load(_f)
        _valid_rule_ids = {r["rule_id"] for r in _analysis_data.get("rules", [])}
    except Exception:
        _valid_rule_ids = set()

    n_invalid_fallback_applied = 0
    n_invalid_unresolvable = 0
    if _valid_rule_ids:
        for e in compliant_entries:
            rule_ids = e["parsed"].get("rule_ids") or []
            if rule_ids and rule_ids[0] not in _valid_rule_ids:
                fallback = _task_tags.get(e["identifier"])
                if fallback:
                    e["parsed"]["rule_ids"] = fallback
                    n_invalid_fallback_applied += 1
                else:
                    n_invalid_unresolvable += 1
        if n_invalid_fallback_applied or n_invalid_unresolvable:
            print(f"[judge-audit] invalid rule_id fallback: {n_invalid_fallback_applied} entries "
                  f"named a rule_id the benchmark doesn't contain, recovered via project tag "
                  f"({n_invalid_unresolvable} left as-is, no project tag available either)")

    # Rules marked "(Manual)" by CIS are excluded, in the same way as in
    # Agent 2's own compliance rate.
    try:
        from parser_a0_final import get_manual_rule_ids
        manual_ids = get_manual_rule_ids(analysis)   # must match the audited project's benchmark
    except Exception:
        manual_ids = set()
    n_before_manual_filter = len(compliant_entries)
    compliant_entries = [e for e in compliant_entries
                         if not (set(e["parsed"].get("rule_ids") or []) & manual_ids)]
    n_manual_excluded = n_before_manual_filter - len(compliant_entries)

    print(f"[judge-audit] {len(compliant_entries)} tasks Agent 2 called COMPLIANT "
          f"({n_manual_excluded} Manual-rule task(s) excluded, same as Agent 2's own scoring) "
          f"(source: {dump_path}, since={since or 'start of file'}, until={until or 'end of file'})")
    if limit:
        compliant_entries = compliant_entries[:limit]

    yaml_by_rule = load_all_audits_for_siblings(dump_path, since_ts, until_ts, task_tags=_task_tags)
    print(f"[judge-audit] Sibling context built from {sum(len(v) for v in yaml_by_rule.values())} "
          f"audited task(s) across {len(yaml_by_rule)} rule(s)")

    from judge_combined import _script_content_note, _handler_content_note, _rendered_template_note

    out_csv = out_csv or "results/judge_audit__per_task.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["task_name", "rule_id", "agent2_explanation",
                  "grounded_verdict", "grounded_score", "grounded_reason", "error"]

    rows = []
    n_skipped_no_rule_id = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, e in enumerate(compliant_entries, 1):
            parsed = e["parsed"]
            task_name = e["identifier"]
            rule_ids = parsed.get("rule_ids") or []
            rule_id = rule_ids[0] if rule_ids else ""
            if not rule_id:
                n_skipped_no_rule_id += 1
                continue
            yaml_text = extract_raw_yaml(e["messages"][1]["content"])
            siblings = [y for y in yaml_by_rule.get(rule_id, []) if y != yaml_text]
            # Template, script and handler content notes are added for each
            # sibling as well, not only for the primary task.
            siblings = [
                y + _script_content_note(y, project_path)
                  + _handler_content_note(y, project_path)
                  + _rendered_template_note(y, project_path)
                for y in siblings
            ]
            semantic_desc = extract_semantic_description(e["messages"][1]["content"])
            semantic_note = (
                f"\nSEMANTIC DESCRIPTION (Agent 2's own parser-built summary of this task, which "
                f"may include Jinja-RESOLVED values — e.g. a template's actual rendered content — "
                f"that the raw YAML above does not show since it is deliberately unresolved project "
                f"source):\n{semantic_desc}\n"
                if semantic_desc else ""
            )
            script_note = (_script_content_note(yaml_text, project_path)
                          + _handler_content_note(yaml_text, project_path)
                          + _rendered_template_note(yaml_text, project_path)
                          + semantic_note)

            result = grounded_judge_one(rule_id, task_name, yaml_text, judge_model,
                                        sibling_yamls=siblings, script_content_note=script_note)
            row = {
                "task_name": task_name, "rule_id": rule_id,
                "agent2_explanation": parsed.get("explanation", ""),
                "grounded_verdict": result.get("verdict", ""),
                "grounded_score": result.get("score"),
                "grounded_reason": result.get("reason", ""),
                "error": result.get("error", ""),
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)

            if i % 10 == 0 or i == len(compliant_entries):
                print(f"  [{i}/{len(compliant_entries)}] {rule_id}: "
                      f"Agent2=COMPLIANT -> judge={row['grounded_verdict'] or 'ERR'} "
                      f"({row['grounded_score']})")

    n = len(rows)
    confirmed = sum(1 for r in rows if r["grounded_verdict"] == "COMPLIANT")
    disputed = [r for r in rows if r["grounded_verdict"] in ("PARTIAL", "NON_COMPLIANT")]
    print(f"\n[judge-audit] DONE. {confirmed}/{n} of Agent 2's COMPLIANT calls confirmed "
          f"({round(100*confirmed/n, 1) if n else 0}%). "
          f"{len(disputed)} disputed by the independent judge — see {out_csv}.")
    print(f"[judge-audit] coverage: {n_skipped_no_rule_id} of {n + n_skipped_no_rule_id} total "
          f"COMPLIANT-task candidates had NO resolvable rule_id (LLM gave none AND no project tag "
          f"fallback either) — excluded, never silently: {n}/{n + n_skipped_no_rule_id} "
          f"= {round(100*n/(n + n_skipped_no_rule_id), 1) if (n + n_skipped_no_rule_id) else 0}% "
          f"of candidates were actually checkable.")
    for r in disputed:
        print(f"  DISPUTED: {r['rule_id']} ({r['task_name'][:50]}) -> {r['grounded_verdict']} "
              f"({r['grounded_score']}): {r['grounded_reason'][:150]}")
    return out_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True)
    p.add_argument("--project-path", required=True)
    p.add_argument("--since", default=None, help="'%%Y-%%m-%%d %%H:%%M:%%S', filters dump to this run only")
    p.add_argument("--until", default=None, help="'%%Y-%%m-%%d %%H:%%M:%%S', upper bound — pass this when a LATER run may already be writing to the same shared dump file while this one reads it")
    p.add_argument("--judge", default=JUDGE_MODEL)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--collection", default="cis_flat", help="ChromaDB collection matching the project's benchmark (cis_flat_debian12 etc.)")
    p.add_argument("--analysis", default="analysis_results.json", help="analysis JSON matching the project's benchmark (for Manual-rule exclusion)")
    args = p.parse_args()
    run_audit_verification(args.dump, args.project_path, args.since, args.until, args.judge, args.out, args.limit, args.collection, args.analysis)
