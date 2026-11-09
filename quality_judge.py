"""
quality_judge.py — LLM-as-judge CODE QUALITY evaluation (not compliance).

Complements model_comparison_grounded_judge.py (compliance with the CIS
remediation text) with a different question: is the fix well-written
Infrastructure-as-Code?

Literature grounding:
  - "A Framework for Measuring the Quality of Infrastructure-as-Code
    Scripts" (arXiv:2502.03127) — 9 IaC quality dimensions, adapted here to
    the 6 that apply to a single task: Function and Purpose, Automation,
    Error Handling, Code Maintainability, Code Security, Code Structure.
  - "Security Smells in Ansible and Chef Scripts: A Replication Study"
    (arXiv:1907.07159) — Ansible security smell taxonomy used for the Code
    Security dimension.

Judge model: qwen3.8-27b (JUDGE_MODEL below), using agent_a2_llm's API
configuration. The prompt does not suggest any expected score.

Usage:
  python3 quality_judge.py <final_report1.json> [<final_report2.json> ...] [--judge=MODEL] [--out=CSV] [--limit=N]
"""

import os
import csv
import sys
import json
import time
import urllib.request

import agent_a2_llm
from debug_dump import dump_llm_call, extract_json_object

JUDGE_MODEL = "qwen3.8-27b"

QUALITY_DIMENSIONS = [
    "function_and_purpose",
    "automation",
    "error_handling",
    "code_maintainability",
    "code_security",
    "code_structure",
]


def _quality_prompt(rule_id: str, rule_title: str, generated_yaml: str) -> list:
    return [
        {"role": "system", "content": (
            "You are an experienced Ansible/Infrastructure-as-Code reviewer "
            "assessing CODE QUALITY — not whether the task is compliant with a "
            "specific requirement (that has already been checked separately), "
            "but whether it is WELL-WRITTEN Ansible code by professional "
            "engineering standards. Score each of the 6 dimensions below "
            "1-5 (5 = excellent, 1 = seriously deficient), then an overall "
            "1-5 score. Definitions, adapted from a published IaC quality "
            "framework (Infrastructure-as-Code quality dimensions) and an "
            "Ansible-specific security-smell taxonomy:\n"
            "1. function_and_purpose: does the task's structure clearly express "
            "a single, well-defined intent (clear task name, no dead/unreachable "
            "logic, no unrelated side effects)?\n"
            "2. automation: does it use a dedicated, purpose-built Ansible "
            "module appropriate for the job, rather than shell/command where "
            "a proper module exists (dedicated modules are more portable, "
            "idempotent, and reliably report changed-state)?\n"
            "3. error_handling: does it handle failure/edge cases sensibly — "
            "appropriate use of register/changed_when/failed_when/ignore_errors, "
            "no silent failure masking, no unguarded assumptions about system "
            "state?\n"
            "4. code_maintainability: is it readable and easy to modify safely — "
            "clear naming, no needless duplication, no fragile/obscure "
            "constructs (e.g. hand-rolled regex where a built-in filter exists)?\n"
            "5. code_security: does it avoid recognized security smells — no "
            "hardcoded secrets/credentials, no unnecessarily broad file "
            "permissions, no disabling of a security control instead of fixing "
            "it, no use of an insecure protocol/verb where a safer one exists?\n"
            "6. code_structure: is the complexity appropriate for the task — "
            "neither needlessly convoluted (over-engineered) nor missing "
            "necessary structure (e.g. a loop where one is clearly needed)?\n"
            "Respond with ONLY valid JSON, no markdown, no explanation outside "
            "the JSON structure."
        )},
        {"role": "user", "content": f"""CIS rule: {rule_id} — {rule_title}

GENERATED ANSIBLE TASK (already verified to correctly implement the CIS
remediation — you are assessing its CODE QUALITY, not its compliance):
{generated_yaml[:2500]}

Respond ONLY with this JSON structure:
{{
  "function_and_purpose": {{"score": integer 1-5, "note": "1 sentence"}},
  "automation": {{"score": integer 1-5, "note": "1 sentence"}},
  "error_handling": {{"score": integer 1-5, "note": "1 sentence"}},
  "code_maintainability": {{"score": integer 1-5, "note": "1 sentence"}},
  "code_security": {{"score": integer 1-5, "note": "1 sentence"}},
  "code_structure": {{"score": integer 1-5, "note": "1 sentence"}},
  "overall_score": integer 1-5,
  "overall_reason": "2-3 sentences summarizing the assessment"
}}"""},
    ]


def quality_judge_one(rule_id: str, rule_title: str, generated_yaml: str,
                      judge_model: str = JUDGE_MODEL) -> dict:
    """One quality assessment of ONE generated fix. Returns a dict with the
    6 dimension scores + overall_score/overall_reason, or an 'error' field.
    Never raises."""
    messages = _quality_prompt(rule_id, rule_title, generated_yaml)
    payload = json.dumps({
        "model": judge_model,
        # enable_thinking:False — same rationale as
        # model_comparison_grounded_judge.py: qwen3.8-27b reasons by
        # default and can burn the whole max_tokens budget on internal
        # chain-of-thought before ever emitting the JSON, with N calls
        # here (one per fix) that adds up fast.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0,
    }).encode()

    for attempt in range(1, 3):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                agent_a2_llm.API_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {agent_a2_llm.API_TOKEN}",
                         "User-Agent": "curl/7.88.1"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            # extract_json_object() takes the first complete JSON object,
            # even if the model adds text after it.
            json_str = extract_json_object(text)
            if not json_str:
                agent_a2_llm._record_request(time.time() - t0, success=False)
                dump_llm_call("Quality_judge", rule_id, messages, text,
                              {"error": "no valid JSON in response"})
                if attempt < 2:
                    continue
                return {"error": "judge did not return valid JSON"}
            # Same lenient-JSON repair as agent_a4._lenient_json_loads() —
            # a raw backslash in a "note"/"overall_reason" string (e.g.
            # quoting a regex or path) produces invalid JSON otherwise;
            # confirmed (rule 1.6.1) that this exact failure mode
            # silently discards a real result if not handled.
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                import re as _re
                result = json.loads(_re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str))
            agent_a2_llm._record_request(time.time() - t0, success=True)
            # overall_score is computed by the code as the mean of the 6
            # dimension scores. The model's own overall figure is kept
            # separately as llm_self_reported_overall_score.
            dim_scores = [
                (result.get(d) or {}).get("score")
                for d in QUALITY_DIMENSIONS
            ]
            dim_scores = [s for s in dim_scores if isinstance(s, (int, float))]
            if dim_scores:
                result["llm_self_reported_overall_score"] = result.get("overall_score")
                result["overall_score"] = round(sum(dim_scores) / len(dim_scores), 2)
            dump_llm_call("Quality_judge", rule_id, messages, text, result)
            return result
        except Exception as e:
            agent_a2_llm._record_request(time.time() - t0, success=False)
            dump_llm_call("Quality_judge", rule_id, messages, "", {"error": str(e)})
            if attempt < 2:
                continue
            return {"error": str(e)}


def run_quality_judge(report_json_paths: list, judge_model: str = JUDGE_MODEL,
                      out_csv: str = None, limit: int = None):
    all_details = []
    for path in report_json_paths:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        details = report.get("remediation_details") or []
        base = os.path.basename(path)
        section_label = base.replace("final_report__gemma4-26b-a4b__section_", "").rsplit(".json", 1)[0]
        for d in details:
            d = dict(d)
            d["_section"] = section_label
            all_details.append(d)
        print(f"[quality-judge] {path}: {len(details)} fix(es)")

    if not all_details:
        print("[quality-judge] No remediation_details found in any given report — nothing to score.")
        sys.exit(1)

    if limit:
        all_details = all_details[:limit]

    safe_judge = "".join(c if c.isalnum() or c in "-_." else "_" for c in judge_model)
    out_csv = out_csv or f"results/quality_judge__all_sections__judge_{safe_judge}.csv"

    print(f"[quality-judge] Judge: {judge_model} | {len(all_details)} fix(es) total | writing to: {out_csv}")

    fieldnames = ["section", "rule_id", "action_type"] + [f"{d}_score" for d in QUALITY_DIMENSIONS] + \
                 [f"{d}_note" for d in QUALITY_DIMENSIONS] + \
                 ["overall_score", "llm_self_reported_overall_score", "overall_reason", "error"]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(all_details, 1):
            rule_id = d.get("rule_id", "")
            rule_title = d.get("rule_title", "")
            generated_yaml = d.get("final_yaml") or d.get("yaml_fix") or ""
            section = d.get("_section", "")

            print(f"  [{i}/{len(all_details)}] ({section}) Scoring {rule_id} ({d.get('action_type')})...")
            result = quality_judge_one(rule_id, rule_title, generated_yaml, judge_model)

            row = {"section": section, "rule_id": rule_id, "action_type": d.get("action_type", "")}
            if result.get("error"):
                row["error"] = result["error"]
                print(f"      error: {result['error']}")
            else:
                for dim in QUALITY_DIMENSIONS:
                    dim_result = result.get(dim, {}) or {}
                    row[f"{dim}_score"] = dim_result.get("score")
                    row[f"{dim}_note"] = dim_result.get("note", "")
                row["overall_score"] = result.get("overall_score")
                row["llm_self_reported_overall_score"] = result.get("llm_self_reported_overall_score")
                row["overall_reason"] = result.get("overall_reason", "")
                row["error"] = ""
                print(f"      overall {result.get('overall_score')}/5")
            writer.writerow(row)
            f.flush()

    print(f"[quality-judge] DONE. Saved: {out_csv}")
    return out_csv


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    _judge = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--judge=")), JUDGE_MODEL)
    _out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    _limit = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
    run_quality_judge(args, judge_model=_judge, out_csv=_out, limit=_limit)
