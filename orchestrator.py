"""
Orchestrator
============
Ties the whole pipeline into a single call: Agent 1 (onboarding) -> Parser
A0 (parses the final state) -> Agent 2 (RAG+LLM audit) -> Agent 3 (fix
generation) -> Agent 4 (validation + write) -> Agent 5 (report).

Agent 1 runs first, because it can modify defaults/main.yml, and Parser A0
must see the state after those changes.

Usage:
  python orchestrator.py <project_path> [options]

Options:
  --analysis=FILE        CIS benchmark analysis (default: analysis_results.json)
  --collection=NAME      ChromaDB collection for Agent 2 (default: cis_flat).
                         Must match the benchmark version of --analysis
                         (see ingest_benchmark.py).
  --mock                 don't call the real LLM (Agent 2/3 run on the mock)
  --model=NAME           overrides the LLM model for the whole pipeline; the
                         report gets the model name as a suffix
                         (results/final_report__NAME.json/.md)
  --rule-prefix=PREFIX   restrict the whole pipeline to CIS rules starting
                         with this prefix (e.g. "1." for section 1)
  --max-tasks=N          limit Agent 2 to N tasks (test mode)
  --skip-tasks=N         skip the first N tasks before taking --max-tasks
  --no-apply             Agent 4 only validates, never asks about writing to files
  --no-knowledge-base    Agent 4 skips the knowledge-base semantic check
                         (lint-only validation)
  --min-severity=LEVEL   severity threshold for automatic fix generation
                         (LOW/MEDIUM/HIGH/CRITICAL, default LOW, i.e. a fix
                         for every NON_COMPLIANT/PARTIAL finding)
"""

import os
import sys

import chromadb

from parser_a0_final import parse_ansible_project_full, get_manual_rule_ids
from agent_a1_onboarding import run_onboarding
from agent_a2_llm import audit_tasks, audit_and_remediate_missing_rules, CHROMA_PATH
from agent_a3_llm import generate_remediations, detect_and_create_missing_steps
from agent_a4 import run_validation_and_apply
from agent_a5_report import build_report, save_report


def _check_collection_exists(collection: str):
    """Fail fast: better to abort immediately than discover after 400
    tasks and dozens of minutes of LLM calls that Agent 2 was querying
    the wrong benchmark version (exactly what happened on the first
    ansible-lockdown run)."""
    client = chromadb.PersistentClient(CHROMA_PATH)
    existing = {c.name for c in client.list_collections()}
    if collection not in existing:
        raise SystemExit(
            f"[Orchestrator] ChromaDB collection '{collection}' does not exist "
            f"(available: {sorted(existing) or '(none)'}). "
            f"Build it with: python ingest_benchmark.py --analysis=<file> --collection={collection}"
        )


def run_pipeline(project_path: str, analysis_json_path: str = "analysis_results.json",
                  use_llm: bool = True, max_tasks: int = None,
                  apply_fixes: bool = True, min_severity: str = "LOW",
                  collection: str = "cis_flat", skip_tasks: int = 0,
                  model: str = None, use_knowledge_base: bool = True,
                  rule_prefix: str = None) -> dict:
    """Runs the full pipeline and returns the final report.

    model: overrides agent_a2_llm.API_MODEL for the whole pipeline; the
    output file gets the model name as a suffix
    (results/final_report__<model>.json/.md).

    use_knowledge_base: whether Agent 4 additionally checks each fix that
    passes ansible-lint against the CIS knowledge base
    (see agent_a4.verify_with_knowledge_base).

    rule_prefix: restrict the whole pipeline (audit, fixes, validation,
    coverage and report) to CIS rules whose ID starts with this prefix
    (e.g. "5.1." for the SSH sub-section), for section-by-section review."""

    print("\n" + "=" * 70)
    print("ORCHESTRATOR — Agent 1 -> Parser A0 -> Agent 2 -> Agent 3 -> Agent 4 -> Agent 5")
    print("=" * 70)

    _check_collection_exists(collection)
    import agent_a2_llm
    agent_a2_llm.COLLECTION = collection
    print(f"[Orchestrator] Agent 2 will use ChromaDB collection: '{collection}'")

    if model:
        agent_a2_llm.API_MODEL = model
        print(f"[Orchestrator] LLM model overridden to: '{model}'")

    # ── STEP 1: Agent 1 (onboarding) — MUST run first, see module docstring ──
    print("\n>>> STEP 1/5: Agent 1 — onboarding")
    onboarding_result = run_onboarding(project_path, analysis_json_path)

    # ── STEP 2: Parser A0 — parse the state AFTER onboarding ────────────────
    print("\n>>> STEP 2/5: Parser A0 — parsing and rule coverage")
    full = parse_ansible_project_full(project_path, analysis_json_path)
    tasks = full["tasks"]
    coverage = full["coverage"]
    print(f"[Orchestrator] {len(tasks)} tasks, {len(coverage)} rules in the benchmark")

    if rule_prefix:
        tasks = [t for t in tasks if any(r.startswith(rule_prefix) for r in t.cis_rule_ids)]
        coverage = [c for c in coverage if c.rule_id.startswith(rule_prefix)]
        print(f"[Orchestrator] rule_prefix={rule_prefix!r}: limiting to {len(tasks)} tasks, "
              f"{len(coverage)} rules")

    if max_tasks:
        # skip_tasks skips the first N tasks (e.g. in ansible-lockdown
        # this is usually audit infrastructure at the start of a role,
        # with no mapped CIS rule — without this option a test on a
        # small sample would always land only on that).
        tasks = tasks[skip_tasks:skip_tasks + max_tasks]
        print(f"[Orchestrator] Test mode: limiting Agent 2 to {len(tasks)} tasks "
              f"(skipping the first {skip_tasks})")

    # ── STEP 3: Agent 2 — audit existing tasks ───────────────────────────────
    print(f"\n>>> STEP 3/5: Agent 2 — compliance audit ({len(tasks)} tasks)")
    import agent_a2_llm
    agent_a2_llm.USE_LLM = use_llm
    manual_rule_ids = get_manual_rule_ids(analysis_json_path)
    print(f"[Orchestrator] {len(manual_rule_ids)} CIS rules marked \"(Manual)\" in the benchmark — "
          f"still audited, but kept out of the scored compliance % (informational only)")
    compliance_results = audit_tasks(tasks, disabled_rules=onboarding_result["disabled_rules"],
                                     manual_rule_ids=manual_rule_ids, project_path=project_path)

    # ── STEP 4: Agent 3 — fix generation ─────────────────────────────────────
    print(f"\n>>> STEP 4a/5: Agent 3 — fixes for NON_COMPLIANT/PARTIAL (FIX, severity threshold: {min_severity or 'none'})")
    fix_remediations = generate_remediations(tasks, compliance_results, use_llm=use_llm,
                                             min_severity=min_severity, project_path=project_path)

    print("\n>>> STEP 4b/5: Agent 2 -> Agent 3 — missing tasks (CREATE)")
    coverage_gaps = [
        {"rule_id": c.rule_id, "title": c.title, "status": c.status}
        for c in coverage if c.status == "MISSING_TASK"
    ]
    if max_tasks and len(coverage_gaps) > max_tasks:
        # Without this --max-tasks only limited STEP 3 (FIX) — CREATE
        # would always run over the full list of project gaps regardless,
        # so a "cheap 20-task test" could silently generate 88 new tasks
        # via the LLM. This limit stays consistent with what the user
        # asked for via --max-tasks.
        print(f"[Orchestrator] Test mode: limiting CREATE to {max_tasks} of {len(coverage_gaps)} gap(s)")
        coverage_gaps = coverage_gaps[:max_tasks]
    create_remediations = []
    if coverage_gaps:
        _decisions, create_remediations = audit_and_remediate_missing_rules(coverage_gaps, use_llm=use_llm)
    else:
        print("[Orchestrator] No MISSING_TASK rules — skipping the CREATE step")

    # Rules that have tasks but miss a required step entirely (no task to
    # audit) — see detect_and_create_missing_steps().
    print("\n>>> STEP 4c/5: Agent 3 — missing STEPS within partially-implemented rules (CREATE)")
    step_gap_remediations = detect_and_create_missing_steps(tasks, coverage, use_llm=use_llm)

    all_remediations = fix_remediations + create_remediations + step_gap_remediations
    print(f"[Orchestrator] {len(all_remediations)} fix(es) total "
          f"({len(fix_remediations)} FIX + {len(create_remediations)} CREATE)")

    # ── STEP 5: Agent 4 — validation + (optionally) write ────────────────────
    a4_result = {"results": [], "conflicts": [], "applied": []}
    if all_remediations:
        print(f"\n>>> STEP 5/5: Agent 4 — validation{' + write' if apply_fixes else ' (no write)'}"
              f"{' + knowledge base' if use_knowledge_base else ''}")
        if apply_fixes:
            a4_result = run_validation_and_apply(all_remediations, project_path,
                                                 use_knowledge_base=use_knowledge_base)
        else:
            from agent_a4 import validate_all
            results, conflicts = validate_all(all_remediations, use_knowledge_base=use_knowledge_base)
            a4_result = {"results": results, "conflicts": conflicts, "applied": []}
    else:
        print("\n>>> STEP 5/5: Agent 4 — skipping, no fixes to validate")

    # ── Agent 5 — final report ────────────────────────────────────────────────
    print("\n>>> Agent 5 — final report")
    report = build_report(
        project_path=project_path,
        onboarding_result=onboarding_result,
        coverage=coverage,
        compliance_results=compliance_results,
        remediations=all_remediations,
        a4_result=a4_result,
        use_llm=use_llm,
    )
    if model or rule_prefix:
        # model-name / rule_prefix suffix — see run_pipeline() docstring.
        # Replace characters unsafe for a filename (e.g. '/' in some model
        # names or in a rule_prefix like "6.2.").
        def _safe(s): return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
        suffix = "__".join(filter(None, [_safe(model) if model else None,
                                         f"section_{_safe(rule_prefix)}" if rule_prefix else None]))
        results_dir = os.path.join(project_path, "results")
        save_report(
            report,
            json_path=os.path.join(results_dir, f"final_report__{suffix}.json"),
            md_path=os.path.join(results_dir, f"final_report__{suffix}.md"),
        )
    else:
        save_report(report)

    print("\n" + "=" * 70)
    print("ORCHESTRATOR — done")
    print("=" * 70)
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    _project_path = sys.argv[1]
    _analysis_json = next((a.split("=")[1] for a in sys.argv if a.startswith("--analysis=")),
                          "analysis_results.json")
    _use_llm = "--mock" not in sys.argv
    _max_tasks = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--max-tasks=")), None)
    _apply_fixes = "--no-apply" not in sys.argv
    _use_knowledge_base = "--no-knowledge-base" not in sys.argv
    _min_severity = next((a.split("=")[1] for a in sys.argv if a.startswith("--min-severity=")), "LOW")
    _collection = next((a.split("=")[1] for a in sys.argv if a.startswith("--collection=")), "cis_flat")
    _skip_tasks = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--skip-tasks=")), 0)
    _model = next((a.split("=")[1] for a in sys.argv if a.startswith("--model=")), None)
    _rule_prefix = next((a.split("=")[1] for a in sys.argv if a.startswith("--rule-prefix=")), None)

    run_pipeline(_project_path, _analysis_json, use_llm=_use_llm,
                max_tasks=_max_tasks, apply_fixes=_apply_fixes, min_severity=_min_severity,
                model=_model, use_knowledge_base=_use_knowledge_base,
                collection=_collection, skip_tasks=_skip_tasks, rule_prefix=_rule_prefix)
