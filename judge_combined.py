"""
judge_combined.py — combines two LLM-as-judge assessments of each fix into
one per-model summary, computed by code (not self-reported by the judge).

The two questions are kept separate rather than merged into one weighted
score, since weights would have no methodological basis:

  1. COMPLIANCE — does the fix satisfy the CIS rule? grounded_judge_one()
     from model_comparison_grounded_judge.py gives the judge the real CIS
     text from the knowledge base and asks it to verify against that text,
     not from parametric memory.
  2. CODE QUALITY — is the fix well-written Infrastructure-as-Code?
     quality_judge_one() from quality_judge.py, 6 dimensions adapted from a
     published IaC quality framework (arXiv:2502.03127) and an Ansible
     security-smell taxonomy (arXiv:1907.07159).

Usage:
  python3 judge_combined.py <final_report1.json> [<final_report2.json> ...] [--judge=MODEL] [--out-dir=DIR] [--limit=N]

Writes, per invocation:
  <out-dir>/combined_judge__per_fix.csv       one row per fix, both axes' raw output
  <out-dir>/combined_judge__summary.csv       one row per model, code-computed means/stdev
"""

import os
import re
import sys
import csv
import json
import statistics as stats

import agent_a2_llm
from model_comparison_grounded_judge import grounded_judge_one
from quality_judge import quality_judge_one, QUALITY_DIMENSIONS
from parser_a0_final import parse_ansible_project, load_variables

JUDGE_MODEL = "qwen3.8-27b"

_SCRIPT_CMD_RE = re.compile(r"(?:script|cmd):\s*\{?\s*(?:cmd:\s*)?['\"]?([\w./-]+\.sh)")


def _script_content_note(generated_yaml: str, project_path: str) -> str:
    """When a fix uses the 'script:'/'ansible.builtin.script' module (the
    logic lives in an external .sh file), finds that file under the project
    and returns its content, so that the judge sees what the script does."""
    m = _SCRIPT_CMD_RE.search(generated_yaml)
    if not m or not project_path:
        return ""
    fname = m.group(1)
    for root, _dirs, files in os.walk(project_path):
        if fname in files:
            try:
                with open(os.path.join(root, fname), encoding="utf-8") as f:
                    content = f.read()
                return (f"\nACTUAL CONTENT OF THE REFERENCED SCRIPT '{fname}' (the YAML above "
                        f"only calls this script — its own enforcement logic, which you must "
                        f"judge, is here):\n{content[:2000]}\n")
            except Exception:
                return ""
    return ""


_TEMPLATE_SRC_RE = re.compile(
    r"(?:ansible\.builtin\.)?template:\s*\n\s+src:\s*[\"']?([\w./-]+?\.j2)[\"']?\s*(?:\n|$)"
)
_RENDERED_TEMPLATE_CACHE: dict = {}


def _render_template(project_path: str, src: str) -> str:
    """Renders a role's own .j2 template with its own defaults/main.yml +
    vars/main.yml (load_variables()), caching per (project_path, src) since
    the same template is re-rendered across many judge calls. Returns ""
    on any failure (missing jinja2, missing file, a render-time error) —
    never raises, same fail-soft contract as _script_content_note()."""
    key = (project_path, src)
    if key in _RENDERED_TEMPLATE_CACHE:
        return _RENDERED_TEMPLATE_CACHE[key]
    result = ""
    try:
        import jinja2
        templates_dir = os.path.join(project_path, "templates")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            undefined=jinja2.Undefined,  # lenient: undefined var -> "", never raises
        )
        template = env.get_template(src)
        rendered = template.render(**load_variables(project_path))
        lines = [ln for ln in rendered.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        result = "\n".join(lines)[:1500]
    except Exception:
        result = ""
    _RENDERED_TEMPLATE_CACHE[key] = result
    return result


def _rendered_template_note(generated_yaml: str, project_path: str) -> str:
    """When a fix (or an original task, for judge_audit's re-verification)
    deploys a '.j2' template via 'ansible.builtin.template: src: ...',
    renders the template with the role's own defaults/main.yml and returns
    the result, so the judge sees the configuration lines the template
    actually produces. Handles both 'template' and the fully qualified
    module name."""
    if not project_path:
        return ""
    notes = []
    seen_src = set()
    for m in _TEMPLATE_SRC_RE.finditer(generated_yaml):
        src = m.group(1)
        if src in seen_src:
            continue
        seen_src.add(src)
        rendered = _render_template(project_path, src)
        if rendered:
            notes.append(
                f"\nRENDERED TEMPLATE CONTENT of '{src}' (rendered with this role's OWN "
                f"defaults/main.yml + vars/main.yml — the task above only shows 'src:', not "
                f"the template's actual config-line output):\n{rendered}\n"
            )
    return "".join(notes)


# Single-line form: notify: "name"  /  notify: name
_NOTIFY_SINGLE_RE = re.compile(r"notify:\s*\[?\s*\"?'?([^\"'\[\]\n]+?)\"?'?\s*\]?\s*(?:\n|$)")
# Multi-line YAML-list form: notify:\n  - "name1"\n  - name2\n...
_NOTIFY_LIST_RE = re.compile(r"notify:\s*\n((?:[ \t]*-[ \t]*.+\n?)+)")
_NOTIFY_LIST_ITEM_RE = re.compile(r"^[ \t]*-[ \t]*\"?'?([^\"'\n]+?)\"?'?[ \t]*$", re.MULTILINE)

# A task's own 'vars:' block ("key: value" lines indented under it) and a
# bare '{{ key }}' Jinja reference (no filters) — just enough to resolve a
# notify target like 'notify: "Remount {{ mount_point }}"' against the SAME
# task's 'vars: {mount_point: "/dev/shm"}', see _resolve_notify_targets()'s
# docstring for why this is needed.
_VARS_BLOCK_RE = re.compile(r"\n[ \t]*vars:[ \t]*\n((?:[ \t]+\S.*\n?)+)")
_VAR_LINE_RE = re.compile(r'^[ \t]+([A-Za-z_][A-Za-z0-9_]*):[ \t]*(.+?)[ \t]*$', re.MULTILINE)
_JINJA_BARE_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _extract_vars_map(yaml_text: str) -> dict:
    """Best-effort key->value map from a task's own 'vars:' block (quotes
    stripped). Not a real Jinja/YAML evaluator — only handles the simple
    'key: "literal"' shape actually used for notify-target variables in
    this project (e.g. mount_point/required_mount); anything else is left
    as whatever text follows the colon, which is harmless since
    _resolve_notify_targets() only substitutes bare '{{ var }}' refs."""
    m = _VARS_BLOCK_RE.search(yaml_text)
    if not m:
        return {}
    out = {}
    for vm in _VAR_LINE_RE.finditer(m.group(1)):
        key, val = vm.group(1), vm.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def _resolve_notify_targets(targets: list, yaml_text: str) -> list:
    """Substitutes bare '{{ var }}' references in each notify target using
    the task's own 'vars:' block, so a target like 'Remount {{ mount_point
    }}' resolves to 'Remount /dev/shm' and can be matched against a
    handler's 'listen:' topic. Targets with no '{{ }}' or an unresolvable
    variable are returned unchanged."""
    vars_map = _extract_vars_map(yaml_text)
    if not vars_map:
        return list(targets)
    return [_JINJA_BARE_VAR_RE.sub(lambda m: vars_map.get(m.group(1), m.group(0)), t)
            for t in targets]


def _extract_notify_targets(generated_yaml: str) -> list:
    """Returns every handler name a task's 'notify:' targets, for all three
    YAML forms: a bare string ('notify: X'), an inline list
    ('notify: [X, Y]') and a multi-line block list ('notify:\n  - X\n  - Y')."""
    m = _NOTIFY_LIST_RE.search(generated_yaml)
    if m:
        return [i.strip() for i in _NOTIFY_LIST_ITEM_RE.findall(m.group(1)) if i.strip()]
    m = _NOTIFY_SINGLE_RE.search(generated_yaml)
    if m:
        name = m.group(1).strip()
        return [name] if name else []
    return []


def _handler_content_note(generated_yaml: str, project_path: str) -> str:
    """When a fix uses 'notify: <handler name(s)>', shows what each notified
    handler actually runs (e.g. 'authselect apply-changes'), so the judge
    does not have to guess from the handler name. Handlers are matched both
    by 'name:' and by 'listen:' topic, after resolving simple
    '{{ var }}' references from the task's own 'vars:'."""
    targets = _extract_notify_targets(generated_yaml)
    if not targets or not project_path:
        return ""
    try:
        from agent_a3_llm import discover_handlers
        handlers = discover_handlers(project_path)
    except Exception:
        return ""
    by_name = {h["name"]: h["action"] for h in handlers}
    by_listen = {}
    for h in handlers:
        for topic in h.get("listen", []):
            by_listen.setdefault(topic, []).append(h["action"])
    resolved_targets = _resolve_notify_targets(targets, generated_yaml)
    lines = []
    for name, resolved in zip(targets, resolved_targets):
        if resolved in by_name:
            lines.append(f"  '{name}' runs: {by_name[resolved]}")
        elif resolved in by_listen:
            for action in by_listen[resolved]:
                lines.append(f"  '{name}' (listen topic '{resolved}') runs: {action}")
    if not lines:
        return ""
    return (f"\nWHAT THE NOTIFIED HANDLER(S) ACTUALLY RUN (Ansible executes each of these "
            f"automatically, once, at the end of the play — this IS the activation/reload step, "
            f"even though no command for it appears directly in the task above):\n"
            + "\n".join(lines) + "\n")


def _model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    return base.replace("final_report__", "").rsplit(".json", 1)[0]


def run_combined(report_paths: list, judge_model: str = JUDGE_MODEL,
                 out_dir: str = "results", limit: int = None,
                 collection: str = "cis_flat") -> tuple:
    # The judge must use the same CIS collection as the audited run
    # (e.g. cis_flat_debian12 for Debian), otherwise fixes would be judged
    # against another distribution's benchmark. Default: 'cis_flat' (RHEL 10).
    agent_a2_llm.COLLECTION = collection
    print(f"[combined-judge] grounded judge will retrieve CIS text from collection: '{collection}'")
    os.makedirs(out_dir, exist_ok=True)
    per_fix_path = os.path.join(out_dir, "combined_judge__per_fix.csv")
    summary_path = os.path.join(out_dir, "combined_judge__summary.csv")

    per_fix_fields = (
        ["model", "rule_id", "action_type", "pipeline_valid", "pipeline_lint_valid",
         "pipeline_semantic_valid"]
        + ["grounded_verdict", "grounded_score", "grounded_reason", "grounded_kb_found", "grounded_error"]
        + [f"quality_{d}_score" for d in QUALITY_DIMENSIONS]
        + [f"quality_{d}_note" for d in QUALITY_DIMENSIONS]
        + ["quality_overall_score", "quality_overall_reason", "quality_error"]
    )

    all_rows = []  # kept in memory too, for the summary pass below

    with open(per_fix_path, "w", newline="", encoding="utf-8") as pf:
        writer = csv.DictWriter(pf, fieldnames=per_fix_fields)
        writer.writeheader()

        for path in report_paths:
            model_name = _model_name_from_path(path)
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            details = report.get("remediation_details") or []
            if limit:
                details = details[:limit]
            print(f"[combined-judge] {model_name}: {len(details)} fix(es) — "
                  f"grounded (compliance) + quality (code) judge, judge model: {judge_model}")

            # rule_id -> [raw_yaml, ...] of every other fix for the same
            # rule, so the judge assesses combined sibling coverage instead
            # of one task in isolation.
            yaml_by_rule: dict = {}
            for d in details:
                rid = d.get("rule_id", "")
                y = d.get("final_yaml") or d.get("yaml_fix") or ""
                if rid and y:
                    yaml_by_rule.setdefault(rid, []).append(y)

            # Existing project tasks of the same rule (re-parsed from
            # project_path) are added as siblings as well, because a fix may
            # rely on a task that was already compliant and never needed
            # fixing.
            project_path = report.get("project_path", "")
            if project_path:
                try:
                    all_tasks = parse_ansible_project(project_path)
                    # The original tasks replaced by the fixes are excluded,
                    # so that a superseded task is not shown to the judge as
                    # still-active sibling coverage.
                    replaced_task_names = {d.get("original_task") for d in details
                                           if d.get("original_task")}
                    for t in all_tasks:
                        if t.task_name in replaced_task_names:
                            continue
                        for rid in getattr(t, "cis_rule_ids", []) or []:
                            y = getattr(t, "raw_yaml", "") or ""
                            if y:
                                yaml_by_rule.setdefault(rid, []).append(y)
                    print(f"  [combined-judge] Re-parsed {project_path} — "
                          f"{len(all_tasks)} project tasks available as sibling context "
                          f"({len(replaced_task_names)} pre-fix original task(s) excluded, "
                          f"not just generated fixes)")
                except Exception as e:
                    print(f"  [combined-judge] Warning: could not re-parse {project_path} "
                          f"for full sibling context ({e}) — using generated-fix siblings only")

            for i, d in enumerate(details, 1):
                rule_id = d.get("rule_id", "")
                rule_title = d.get("rule_title", rule_id)
                original_task = d.get("original_task", "")
                yaml_fix = d.get("final_yaml") or d.get("yaml_fix") or ""
                siblings = [y for y in yaml_by_rule.get(rule_id, []) if y != yaml_fix]
                # Template, script and handler content is rendered for each
                # sibling fix as well, not only for the primary one.
                siblings = [
                    y + _script_content_note(y, project_path)
                      + _handler_content_note(y, project_path)
                      + _rendered_template_note(y, project_path)
                    for y in siblings
                ]
                script_note = (_script_content_note(yaml_fix, project_path)
                              + _handler_content_note(yaml_fix, project_path)
                              + _rendered_template_note(yaml_fix, project_path))

                grounded = grounded_judge_one(rule_id, original_task, yaml_fix, judge_model,
                                              sibling_yamls=siblings,
                                              script_content_note=script_note)
                quality = quality_judge_one(rule_id, rule_title, yaml_fix, judge_model)

                row = {
                    "model": model_name,
                    "rule_id": rule_id,
                    "action_type": d.get("action_type", ""),
                    "pipeline_valid": d.get("valid"),
                    "pipeline_lint_valid": d.get("lint_valid"),
                    "pipeline_semantic_valid": d.get("semantic_valid"),
                    "grounded_verdict": grounded.get("verdict", ""),
                    "grounded_score": grounded.get("score"),
                    "grounded_reason": grounded.get("reason", ""),
                    "grounded_kb_found": grounded.get("knowledge_base_found"),
                    "grounded_error": grounded.get("error", ""),
                    "quality_error": quality.get("error", ""),
                }
                if not quality.get("error"):
                    for dim in QUALITY_DIMENSIONS:
                        dim_result = quality.get(dim, {}) or {}
                        row[f"quality_{dim}_score"] = dim_result.get("score")
                        row[f"quality_{dim}_note"] = dim_result.get("note", "")
                    row["quality_overall_score"] = quality.get("overall_score")
                    row["quality_overall_reason"] = quality.get("overall_reason", "")

                writer.writerow(row)
                pf.flush()
                all_rows.append(row)

                if i % 10 == 0 or i == len(details):
                    g = f"{row['grounded_verdict'] or 'ERR'} ({row['grounded_score']}/5)"
                    q = f"{row['quality_overall_score']}/5" if row.get('quality_overall_score') is not None else "ERR"
                    print(f"  [{i}/{len(details)}] {rule_id}: compliance={g}  quality={q}")

    print(f"[combined-judge] Per-fix results saved: {per_fix_path}")

    # ── Summary — every number below is a plain arithmetic mean/stdev
    # computed by THIS code from the per-fix rows above, never taken from
    # any judge's own self-reported aggregate. Grouped by model.
    by_model: dict = {}
    for row in all_rows:
        by_model.setdefault(row["model"], []).append(row)

    summary_fields = (
        ["model", "n_fixes",
         "n_compliant", "n_partial", "n_non_compliant", "n_grounded_error",
         "pct_compliant", "grounded_score_mean", "grounded_score_stdev"]
        + [f"quality_{d}_mean" for d in QUALITY_DIMENSIONS]
        + ["quality_overall_mean", "quality_overall_stdev", "n_quality_error"]
    )

    with open(summary_path, "w", newline="", encoding="utf-8") as sf:
        writer = csv.DictWriter(sf, fieldnames=summary_fields)
        writer.writeheader()

        for model_name, rows in by_model.items():
            n = len(rows)
            verdicts = [r["grounded_verdict"] for r in rows if r["grounded_verdict"]]
            n_compliant = verdicts.count("COMPLIANT")
            n_partial = verdicts.count("PARTIAL")
            n_non_compliant = verdicts.count("NON_COMPLIANT")
            n_grounded_error = sum(1 for r in rows if r.get("grounded_error"))

            g_scores = [r["grounded_score"] for r in rows if isinstance(r.get("grounded_score"), (int, float))]

            summary_row = {
                "model": model_name,
                "n_fixes": n,
                "n_compliant": n_compliant,
                "n_partial": n_partial,
                "n_non_compliant": n_non_compliant,
                "n_grounded_error": n_grounded_error,
                "pct_compliant": round(100 * n_compliant / len(verdicts), 1) if verdicts else None,
                "grounded_score_mean": round(stats.mean(g_scores), 2) if g_scores else None,
                "grounded_score_stdev": round(stats.stdev(g_scores), 2) if len(g_scores) > 1 else None,
            }

            n_quality_error = sum(1 for r in rows if r.get("quality_error"))
            for dim in QUALITY_DIMENSIONS:
                vals = [r[f"quality_{dim}_score"] for r in rows
                        if isinstance(r.get(f"quality_{dim}_score"), (int, float))]
                summary_row[f"quality_{dim}_mean"] = round(stats.mean(vals), 2) if vals else None

            overall_vals = [r["quality_overall_score"] for r in rows
                            if isinstance(r.get("quality_overall_score"), (int, float))]
            summary_row["quality_overall_mean"] = round(stats.mean(overall_vals), 2) if overall_vals else None
            summary_row["quality_overall_stdev"] = round(stats.stdev(overall_vals), 2) if len(overall_vals) > 1 else None
            summary_row["n_quality_error"] = n_quality_error

            writer.writerow(summary_row)
            print(f"[combined-judge] {model_name}: compliance {summary_row['pct_compliant']}% COMPLIANT "
                  f"(mean {summary_row['grounded_score_mean']}/5)  |  "
                  f"quality mean {summary_row['quality_overall_mean']}/5")

    print(f"[combined-judge] Summary saved: {summary_path}")
    return per_fix_path, summary_path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    _judge = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--judge=")), JUDGE_MODEL)
    _out_dir = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out-dir=")), "results")
    _limit = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
    # MUST match the benchmark version the report was produced against
    # (same value as the orchestrator's --collection). Default 'cis_flat'
    # = RHEL 10, kept for backward compatibility with existing RHEL runs.
    _collection = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--collection=")), "cis_flat")
    run_combined(args, judge_model=_judge, out_dir=_out_dir, limit=_limit, collection=_collection)
