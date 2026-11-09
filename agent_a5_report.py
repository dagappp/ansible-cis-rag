"""
Agent A5 — Reporter
=====================
The last step of the pipeline. Two separate parts:

  1. AGGREGATION (deterministic, no LLM) — collects the results of
     Parser A0 and Agents 1-4, computes statistics and sorts findings.

  2. EXECUTIVE SYNTHESIS (LLM, no RAG) — generate_executive_summary()
     writes an evaluative, prioritising comment based only on the already
     computed data. RAG is not needed here, because all the required
     information is given directly in the prompt.

Inputs (all optional — a missing stage simply has no section):
  - onboarding_result : {"decisions": [...], "disabled_rules": [...]}   from Agent 1
  - coverage          : list of CoverageReport                          from Parser A0
  - compliance_results: list of ComplianceResult (has .severity)        from Agent 2
  - remediations      : list of Remediation                             from Agent 3
  - a4_result         : {"results": [...], "conflicts": [...], "applied": [...], ...} from Agent 4

Main output: results/final_report.json (machine-readable) +
results/final_report.md (human-readable, sorted by severity, with an
"Assessment and recommendations" section written by the LLM).
"""

import os
import re
import json
import time
import urllib.request
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass

import agent_a2_llm  # dynamic access (agent_a2_llm.API_MODEL), not "from X import Y" —
# see the rationale in agent_a3_llm.py: import-by-value wouldn't see a
# later model override from orchestrator.py
from debug_dump import dump_llm_call

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "": 4}


def _as_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return obj


def _count_by(items, key_fn):
    counts = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# Report construction
# ═══════════════════════════════════════════════════════════════════════════

def build_report(project_path: str,
                  onboarding_result: dict = None,
                  coverage: list = None,
                  compliance_results: list = None,
                  remediations: list = None,
                  a4_result: dict = None,
                  use_llm: bool = True) -> dict:
    """Collect the results of every stage into a single dict. Each
    section is optional — if a given stage wasn't run (e.g.
    apply_fixes=False in orchestrator.py), it's simply omitted instead of
    faking data that doesn't exist.

    use_llm controls ONLY the 'executive_summary' section (the
    synthesis) — the rest of the report is, and always was, deterministic
    and LLM-independent."""
    onboarding_result = onboarding_result or {}
    coverage = coverage or []
    compliance_results = compliance_results or []
    remediations = remediations or []
    a4_result = a4_result or {}

    report = {
        "project_path": project_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Agent 1: onboarding ──────────────────────────────────────────────
    decisions = onboarding_result.get("decisions", [])
    if decisions:
        decisions_d = [_as_dict(d) for d in decisions]
        report["onboarding"] = {
            "total_decisions": len(decisions_d),
            "by_action": _count_by(decisions_d, lambda d: d["action"]),
            "risk_accepted_rules": [
                {"rule_id": d["rule_id"], "title": d["title"], "reason": d["reason"], "decided_at": d["decided_at"]}
                for d in decisions_d if d["action"] == "RISK_ACCEPTED"
            ],
            "disabled_rules_for_agent2": onboarding_result.get("disabled_rules", []),
        }

    # ── Parser A0: coverage ────────────────────────────────────────────────
    if coverage:
        coverage_d = [_as_dict(c) for c in coverage]
        report["coverage"] = {
            "total_rules": len(coverage_d),
            "by_status": _count_by(coverage_d, lambda c: c["status"]),
        }

    # ── Agent 2: compliance audit ───────────────────────────────────────────
    # Results for rules marked "(Manual)" by CIS are reported separately in
    # "manual_controls": they are audited, but per CIS's definition they need
    # human validation, so they do not count toward the compliance rate and
    # do not produce scored findings.
    findings = []
    manual_review_notes = []
    if compliance_results:
        results_d = [_as_dict(r) for r in compliance_results]
        automated_d = [r for r in results_d if not r.get("manual_control")]
        manual_d = [r for r in results_d if r.get("manual_control")]

        by_status_d = _count_by(automated_d, lambda r: r["status"])
        # Only NOT_MAPPED tasks (no CIS rule tag, e.g. helper tasks that
        # prepare the system) are excluded from the compliance rate.
        # LLM_UNCERTAIN and API_ERROR stay in the denominator, because the
        # audit could not determine whether an in-scope task is compliant.
        not_mapped_count = sum(
            1 for r in automated_d
            if r["status"] == "UNKNOWN" and r.get("unknown_reason") == "NOT_MAPPED"
        )
        scored_total = len(automated_d) - not_mapped_count
        compliant_count = by_status_d.get("COMPLIANT", 0)
        report["compliance"] = {
            "total_audited": len(automated_d),
            "scored_total": scored_total,
            "compliance_rate": round(compliant_count / scored_total * 100, 1) if scored_total else None,
            "by_status": by_status_d,
            "by_severity": _count_by(
                [r for r in automated_d if r.get("severity")], lambda r: r["severity"]
            ),
            # UNKNOWN covers three qualitatively different situations (no
            # rule mapping — irrelevant; LLM uncertainty — a methodological
            # caveat; API failure — the audit did NOT actually happen,
            # needs a retry). Without this breakdown, a single "UNKNOWN: 12"
            # number conflates all three.
            "unknown_by_reason": _count_by(
                [r for r in automated_d if r["status"] == "UNKNOWN" and r.get("unknown_reason")],
                lambda r: r["unknown_reason"]
            ),
        }
        # NOT_MAPPED tasks are listed by name, so that the report shows
        # which tasks were excluded and why.
        not_mapped_tasks = [
            {"task_name": r["task_name"]}
            for r in automated_d
            if r["status"] == "UNKNOWN" and r.get("unknown_reason") == "NOT_MAPPED"
        ]
        if not_mapped_tasks:
            report["compliance"]["not_mapped_tasks"] = not_mapped_tasks
        if manual_d:
            report["manual_controls"] = {
                "total_audited": len(manual_d),
                "by_status": _count_by(manual_d, lambda r: r["status"]),
                "note": "CIS marks these rules \"(Manual)\" — assessment cannot be fully "
                        "automated per CIS's own definition. The status below is the LLM's "
                        "informational read, kept OUT of the compliance score and findings; "
                        "a human should validate these directly.",
            }
            for r in manual_d:
                manual_review_notes.append({
                    "task_name": r["task_name"],
                    "rule_ids": r.get("rule_ids", []),
                    "status": r["status"],
                    "explanation": r.get("explanation", ""),
                })
        for r in automated_d:
            if r["status"] in ("NON_COMPLIANT", "PARTIAL"):
                findings.append({
                    "source": "compliance_audit",
                    "task_name": r["task_name"],
                    "rule_ids": r.get("rule_ids", []),
                    "status": r["status"],
                    "severity": r.get("severity", ""),
                    "cvss_vector": r.get("cvss_vector", ""),
                    "cvss_score": r.get("cvss_score"),
                    "explanation": r.get("explanation", ""),
                })
            elif r["status"] == "UNKNOWN" and r.get("unknown_reason") == "API_ERROR":
                # The only UNKNOWN subcategory that means "we didn't
                # actually manage to audit this" — actionable, so it
                # lands among the findings (for re-auditing), not just in
                # the statistics.
                findings.append({
                    "source": "compliance_audit",
                    "task_name": r["task_name"],
                    "rule_ids": r.get("rule_ids", []),
                    "status": "UNKNOWN (API_ERROR)",
                    "severity": "",
                    "explanation": "The LLM audit failed (API connection error after exhausting "
                                  "retry attempts) — this task was NOT actually verified, "
                                  "and needs to be re-audited.",
                })

    # ── Agent 3: generated fixes ─────────────────────────────────────────────
    if remediations:
        rem_d = [_as_dict(r) for r in remediations]
        report["remediation"] = {
            "total_generated": len(rem_d),
            "by_action_type": _count_by(rem_d, lambda r: r["action_type"]),
        }

    # ── Agent 4: validation + write ───────────────────────────────────────
    a4_results = a4_result.get("results", [])
    a4_conflicts = a4_result.get("conflicts", [])
    a4_applied = a4_result.get("applied", [])

    # The full code of each fix (for manual review): rem.yaml_fix (Agent 3,
    # before the repair loop) and result.final_yaml (Agent 4, after repair),
    # matched by position, together with Agent 4's semantic verdict
    # (semantic_valid is None when no knowledge-base text was found).
    if remediations:
        details = []
        for i, rem in enumerate(remediations):
            r4 = a4_results[i] if i < len(a4_results) else None
            r4_d = _as_dict(r4) if r4 is not None else {}
            details.append({
                "rule_id": rem.rule_id,
                "action_type": rem.action_type,
                "original_task": rem.original_task,
                "severity": getattr(rem, "severity", ""),
                "yaml_fix": rem.yaml_fix,
                "final_yaml": r4_d.get("final_yaml", rem.yaml_fix),
                "valid": r4_d.get("valid"),
                "lint_valid": r4_d.get("lint_valid"),
                "give_up_reason": r4_d.get("give_up_reason", ""),
                "semantic_valid": r4_d.get("semantic_valid"),
                "semantic_verdict": r4_d.get("semantic_verdict", ""),
                "semantic_score": r4_d.get("semantic_score"),
                "semantic_reason": r4_d.get("semantic_reason", ""),
                "knowledge_base_found": r4_d.get("knowledge_base_found", False),
            })
        report["remediation_details"] = details

    if a4_results or a4_conflicts or a4_applied or "cancelled" in a4_result:
        results_d = [_as_dict(r) for r in a4_results]
        report["validation"] = {
            "total_validated": len(results_d),
            "passed": sum(1 for r in results_d if r.get("valid")),
            "failed": [
                {"rule_id": r["rule_id"], "task": r["original_task"], "reason": r.get("give_up_reason", "")}
                for r in results_d if not r.get("valid")
            ],
            "conflicts": [_as_dict(c) for c in a4_conflicts],
            "applied": a4_applied,
            "applied_count": len(a4_applied),
            "cancelled_by_user": a4_result.get("cancelled", False),
            # Knowledge-base semantic check summary — separate from the
            # ansible-lint pass/fail count above. semantically_checked
            # counts only fixes where a knowledge-base chunk was actually
            # found and a verdict returned (excludes those where the check
            # was skipped for lack of a knowledge-base match).
            "semantically_checked": sum(1 for r in results_d if r.get("semantic_valid") is not None),
            "semantically_passed": sum(1 for r in results_d if r.get("semantic_valid") is True),
        }
        for c in a4_conflicts:
            c_d = _as_dict(c)
            findings.append({
                "source": "conflict", "task_name": c_d["rule_a"],
                "rule_ids": [], "status": "CONFLICT", "severity": "HIGH",
                "explanation": f"Conflict with '{c_d['rule_b']}': {c_d['reason']}",
            })

    # Each finding is marked with whether a fix was generated and validated,
    # so that the executive summary does not report already fixed problems
    # as open. Lookup is by the exact original task name first (several
    # tasks can share one rule ID), with rule_id only as a fallback.
    _fix_status_by_task = {}
    _fix_status_by_rule = {}
    for d in report.get("remediation_details", []):
        status = ("FIXED (validated)" if d.get("valid")
                   else "fix generated but FAILED validation — still open")
        task_name = d.get("original_task")
        if task_name:
            _fix_status_by_task[task_name] = status
        rid = d.get("rule_id")
        if rid and rid not in _fix_status_by_rule:
            _fix_status_by_rule[rid] = status
    for finding in findings:
        task_name = finding.get("task_name")
        if task_name in _fix_status_by_task:
            finding["fix_status"] = _fix_status_by_task[task_name]
        else:
            matched = [_fix_status_by_rule[rid] for rid in finding.get("rule_ids", []) if rid in _fix_status_by_rule]
            finding["fix_status"] = matched[0] if matched else "no fix generated — still open"

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.get("severity", ""), 4))
    report["findings"] = findings
    if manual_review_notes:
        report["manual_review_notes"] = manual_review_notes
    report["executive_summary"] = generate_executive_summary(report, use_llm=use_llm)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Executive synthesis (LLM, NO RAG — see rationale in the module docstring)
# ═══════════════════════════════════════════════════════════════════════════

SUMMARY_SYSTEM_PROMPT = (
    "You are a security compliance analyst writing an executive summary for a "
    "CIS Benchmark compliance report. You will receive ALREADY-COMPUTED "
    "statistics and findings — you must NOT invent, recalculate, or alter any "
    "numbers. Your job is ONLY to interpret and prioritize what is given: "
    "highlight the most concerning patterns, explain why certain findings "
    "matter more than raw percentages suggest, and give 2-3 concrete, "
    "actionable recommendations. Write in English, concise (max 200 words), "
    "plain paragraphs, no markdown headers. "
    "IMPORTANT: each item in top_findings carries a 'fix_status' field — "
    "'FIXED (validated)' means a fix was ALREADY generated and independently "
    "verified to correctly resolve that finding (it is history, not a "
    "current gap); 'fix generated but FAILED validation — still open' or "
    "'no fix generated — still open' mean it genuinely remains unresolved. "
    "NEVER describe a 'FIXED (validated)' finding as an ongoing, current, or "
    "unresolved problem — if you mention it at all, frame it as something "
    "the audit caught and the pipeline already corrected, not as something "
    "requiring further action."
)


def _call_llm_summary(prompt_data: dict) -> str:
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Here is the exact, already-computed data from the CIS compliance "
            "audit (do NOT change these numbers, only interpret them):\n\n"
            + json.dumps(prompt_data, ensure_ascii=False, indent=2)
        )},
    ]
    payload = json.dumps({
        "model": agent_a2_llm.API_MODEL,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        agent_a2_llm.API_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {agent_a2_llm.API_TOKEN}",
                 "User-Agent": "curl/7.88.1"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read())
        agent_a2_llm._record_request(time.time() - t0, success=True)
    except Exception:
        agent_a2_llm._record_request(time.time() - t0, success=False)
        raise
    text = data["choices"][0]["message"]["content"].strip()
    dump_llm_call("A5_executive_summary", "executive_summary", messages, text, {"summary": text})
    return text


def _deterministic_summary(report: dict) -> str:
    """Fallback with no LLM (mock mode or connection error) — a simple,
    computed text so the section is never empty without an explanation
    why."""
    c = report.get("compliance", {})
    by_status = c.get("by_status", {})
    by_sev = c.get("by_severity", {})
    compliant = by_status.get("COMPLIANT", 0)
    # Uses the already computed compliance_rate/scored_total (NOT_MAPPED
    # excluded from the denominator, see above).
    pct = c.get("compliance_rate")
    scored_total = c.get("scored_total", c.get("total_audited", 0))
    if pct is None:
        pct = round(compliant / scored_total * 100, 1) if scored_total else 0
    critical = by_sev.get("CRITICAL", 0)
    high = by_sev.get("HIGH", 0)
    return (
        f"[Deterministic summary — LLM unavailable/disabled] "
        f"Compliance: {pct}% ({compliant}/{scored_total} in-scope audited tasks). "
        f"CRITICAL findings: {critical}, HIGH: {high}. Full data in the sections below."
    )


def generate_executive_summary(report: dict, use_llm: bool = True) -> str:
    """Write a synthesizing assessment from the ALREADY-COMPUTED report
    data (coverage/onboarding/compliance/remediation/validation) — no raw
    tasks and no CIS Benchmark text, so no RAG is needed (rationale in
    the module docstring: this is a synthesis OVER DATA, not an
    assessment against external knowledge that would first need to be
    retrieved)."""
    if not use_llm:
        return _deterministic_summary(report)

    # Only the already-aggregated numbers — not the whole report (e.g.
    # no full 'applied'/'conflicts' list or raw onboarding decisions), to
    # keep the prompt small and avoid tempting the model to cite details
    # it doesn't need for the synthesis.
    prompt_data = {k: v for k, v in {
        "coverage": report.get("coverage"),
        "onboarding": {k2: v2 for k2, v2 in report.get("onboarding", {}).items()
                       if k2 != "risk_accepted_rules"} if "onboarding" in report else None,
        "compliance": report.get("compliance"),
        "remediation": report.get("remediation"),
        "validation": {k2: v2 for k2, v2 in report.get("validation", {}).items()
                       if k2 not in ("applied", "conflicts")} if "validation" in report else None,
        "top_findings": report.get("findings", [])[:10],
    }.items() if v is not None}

    try:
        return _call_llm_summary(prompt_data)
    except Exception as e:
        print(f"[A5] LLM error while generating the synthesis: {e} — using the deterministic fallback")
        return _deterministic_summary(report)


# ═══════════════════════════════════════════════════════════════════════════
# Markdown rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_markdown(report: dict) -> str:
    lines = []
    lines.append(f"# CIS compliance pipeline report")
    lines.append(f"\nProject: `{report['project_path']}`  ")
    lines.append(f"Generated: {report['generated_at']}\n")

    if report.get("executive_summary"):
        lines.append("## Assessment and recommendations")
        lines.append(report["executive_summary"])
        lines.append("")

    if "coverage" in report:
        c = report["coverage"]
        lines.append("## CIS rule coverage (Parser A0)")
        lines.append(f"Total rules: **{c['total_rules']}**\n")
        for status, count in sorted(c["by_status"].items()):
            lines.append(f"- `{status}`: {count}")
        lines.append("")

    if "onboarding" in report:
        o = report["onboarding"]
        lines.append("## Onboarding (Agent 1)")
        lines.append(f"Decisions reviewed: **{o['total_decisions']}**\n")
        for action, count in sorted(o["by_action"].items()):
            lines.append(f"- `{action}`: {count}")
        if o["risk_accepted_rules"]:
            lines.append("\n### Knowingly accepted risk")
            for r in o["risk_accepted_rules"]:
                lines.append(f"- **{r['rule_id']}** — {r['title']}")
                lines.append(f"  > {r['reason']} _(decided: {r['decided_at'][:10]})_")
        lines.append("")

    if "compliance" in report:
        c = report["compliance"]
        lines.append("## Compliance audit (Agent 2)")
        lines.append(f"Audited tasks: **{c['total_audited']}**\n")
        if c.get("compliance_rate") is not None:
            lines.append(
                f"**Compliance rate: {c['compliance_rate']}%** "
                f"(COMPLIANT / {c.get('scored_total', c['total_audited'])} in-scope tasks — "
                f"excludes NOT_MAPPED tasks with no CIS rule to compare against; "
                f"see the NOT_MAPPED list below for exactly which ones)\n"
            )
        lines.append("Status:")
        for status, count in sorted(c["by_status"].items()):
            lines.append(f"- `{status}`: {count}")
        if c["by_severity"]:
            lines.append("\nSeverity (NON_COMPLIANT/PARTIAL only):")
            # "NONE" is included too (CVSS score 0.0, no C/I/A impact).
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"):
                if sev in c["by_severity"]:
                    lines.append(f"- `{sev}`: {c['by_severity'][sev]}")
        if c.get("unknown_by_reason"):
            lines.append("\nUNKNOWN by reason:")
            _reason_labels = {
                "NOT_MAPPED":     "no mapped CIS rule (expected)",
                "LOW_CONFIDENCE": "low match score (heuristic)",
                "LLM_UNCERTAIN":  "the LLM judged the data insufficient",
                "API_ERROR":      "LLM connection failure — NOT audited, needs a retry",
            }
            for reason, count in sorted(c["unknown_by_reason"].items()):
                lines.append(f"- `{reason}` ({_reason_labels.get(reason, reason)}): {count}")
        if c.get("not_mapped_tasks"):
            lines.append("\n### NOT_MAPPED tasks (excluded from the compliance rate above)")
            lines.append("No CIS rule tag exists for these — confirmed infra/bootstrap tooling, "
                          "not a compliance gap:")
            for t in c["not_mapped_tasks"]:
                lines.append(f"- {t['task_name']}")
        lines.append("")

    if "manual_controls" in report:
        m = report["manual_controls"]
        lines.append("## Manual controls (excluded from the compliance score)")
        lines.append(f"_{m['note']}_\n")
        lines.append(f"Audited (informational only): **{m['total_audited']}**\n")
        for status, count in sorted(m["by_status"].items()):
            lines.append(f"- `{status}`: {count}")
        if report.get("manual_review_notes"):
            lines.append("\n### Notes for manual review")
            for n in report["manual_review_notes"]:
                rules = ", ".join(n.get("rule_ids", [])) or "-"
                lines.append(f"- **{rules}** — `{n['status']}` ({n['task_name'][:60]})")
                lines.append(f"  > {n['explanation']}")
        lines.append("")

    if "remediation" in report:
        r = report["remediation"]
        lines.append("## Generated fixes (Agent 3)")
        lines.append(f"Total: **{r['total_generated']}**\n")
        for atype, count in sorted(r["by_action_type"].items()):
            lines.append(f"- `{atype}`: {count}")
        lines.append("")

    if "validation" in report:
        v = report["validation"]
        lines.append("## Validation and write (Agent 4)")
        lines.append(f"Validated: **{v['total_validated']}**, passed: **{v['passed']}**\n")
        if v.get("semantically_checked"):
            lines.append(f"Knowledge-base semantic check: **{v['semantically_passed']}/{v['semantically_checked']}** "
                         f"passed (only counts fixes where a knowledge-base match was found)\n")
        if v["cancelled_by_user"]:
            lines.append("_Writing to files was CANCELLED by the user — nothing was changed._\n")
        else:
            lines.append(f"Written to the project: **{v['applied_count']}**")
            for a in v["applied"]:
                lines.append(f"- `{a['action']}` {a['rule_id']} -> `{a['file']}`")
        if v["failed"]:
            lines.append("\n### Failed validation")
            for f in v["failed"]:
                lines.append(f"- **{f['rule_id']}** ({f['task']}): {f['reason']}")
        if v["conflicts"]:
            lines.append("\n### Conflicts (require a manual decision)")
            for c in v["conflicts"]:
                lines.append(f"- '{c['rule_a']}' vs '{c['rule_b']}' — {c['reason']}")
        lines.append("")

    if report.get("remediation_details"):
        lines.append("## Generated fix details (full code, for manual review)")
        for d in report["remediation_details"]:
            status = "✓ passed validation" if d["valid"] else "✗ did NOT pass validation"
            lines.append(f"\n### {d['rule_id']} — {d['action_type']} ({status})")
            lines.append(f"Original task: `{d['original_task']}`  ")
            if d.get("severity"):
                lines.append(f"Severity: `{d['severity']}`  ")
            if d.get("semantic_verdict"):
                sem = "✓" if d.get("semantic_valid") else "✗"
                lines.append(f"Knowledge-base check: {sem} `{d['semantic_verdict']}`"
                             + (f" ({d['semantic_score']}/5)" if d.get("semantic_score") is not None else "")
                             + "  ")
                if d.get("semantic_reason"):
                    lines.append(f"  > {d['semantic_reason']}  ")
            if not d["valid"] and d.get("give_up_reason"):
                lines.append(f"Failure reason: {d['give_up_reason']}  ")
            shown_yaml = d["final_yaml"] or d["yaml_fix"]
            lines.append(f"\n```yaml\n{shown_yaml.strip()}\n```")
        lines.append("")

    if report["findings"]:
        lines.append("## All findings, by severity")
        lines.append("| Severity | CVSS | Source | Task/rules | Description |")
        lines.append("|---|---|---|---|---|")
        for f in report["findings"]:
            rules = ", ".join(f.get("rule_ids", [])) or "-"
            cvss = f"{f['cvss_score']} (`{f['cvss_vector']}`)" if f.get("cvss_score") is not None else "-"
            lines.append(f"| {f.get('severity') or '-'} | {cvss} | {f['source']} | {f['task_name'][:40]} ({rules}) | {f['explanation'][:80]} |")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Saving
# ═══════════════════════════════════════════════════════════════════════════

def save_report(report: dict, json_path: str = None, md_path: str = None) -> None:
    """Default paths used to be relative to CWD, not the project — when
    the orchestrator was run from a different working directory, the
    report would land e.g. in magisterka_rag/results/ instead of
    <project_path>/results/. Now they default to being relative to
    report['project_path'], the same project-relative convention used
    throughout the pipeline for anything project-scoped."""
    if json_path is None or md_path is None:
        project_results = os.path.join(report.get("project_path", "."), "results")
        json_path = json_path or os.path.join(project_results, "final_report.json")
        md_path = md_path or os.path.join(project_results, "final_report.md")

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    print(f"\n[A5] Report saved: {json_path}, {md_path}")
