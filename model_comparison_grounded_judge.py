"""
model_comparison_grounded_judge.py — per-fix judge grounded in the CIS text.

For every fix in a final report, the judge receives the CIS Benchmark text
for the rule (Audit and Remediation sections, retrieved from ChromaDB as in
Agents 3 and 4) and assesses the fix against that text, not from the
model's parametric memory. The CIS text is treated as the best available
reference, not as an infallible ground truth.

Requires final_report__<model>.json with a "remediation_details" field.

Usage:
  python3 model_comparison_grounded_judge.py <final_report.json> [--judge=MODEL] [--out=CSV] [--limit=N]

Writes the CSV incrementally (flushed after every row), so an interrupted
run keeps the rows already written.
"""

import os
import csv
import sys
import json
import time
import urllib.request

import agent_a2_llm
from agent_a3_llm import get_remediation_chunk, get_audit_and_remediation_chunks
from debug_dump import extract_json_object

JUDGE_MODEL = "qwen3.8-27b"


def _sibling_block(sibling_yamls: list) -> str:
    """Renders the other tasks/fixes of the same CIS rule for the judge
    (the same idea as the sibling context in Agents 2-4). A rule is often
    split across several tasks, each implementing one part, so assessing
    each task in isolation would penalise it for what its siblings cover."""
    if not sibling_yamls:
        return ""
    # Maximum number of siblings shown (20) and maximum length of each
    # sibling (3000 characters), large enough for the largest rule blocks
    # and for the rendered template/handler/script notes appended to them.
    sib_list = "\n".join(f"--- sibling fix {i+1} ---\n{y[:3000]}" for i, y in enumerate(sibling_yamls[:20]))
    return f"""
OTHER FIXES FOR THE SAME CIS RULE (already generated, in the same project — a
single CIS rule is frequently implemented as SEVERAL separate tasks, one per
directive):
{sib_list}
Judge the COMBINED coverage of this fix PLUS the siblings above, not this one
fix in isolation — do NOT mark PARTIAL/NON_COMPLIANT solely because this fix
alone omits a directive that a sibling fix above already implements."""


def _remediation_priority_slice(text: str, limit: int) -> str:
    """Truncates a CIS rule chunk to the limit without dropping the
    Remediation section: if Remediation does not fit after the earlier
    sections, keeps the rule's title line and prioritises the Remediation
    section over the earlier ones."""
    if not text or len(text) <= limit:
        return text
    # The text may start with an "AUDIT — PASS CRITERIA" section followed by
    # "REMEDIATION"; the Audit header is checked first, so that truncation
    # keeps the pass criteria intact.
    candidates = [h for h in ("AUDIT — PASS CRITERIA", "REMEDIATION — one concrete", "Remediation:") if h in text]
    idx = min((text.find(h) for h in candidates), default=-1)
    if idx == -1:
        # No recognized heading at all — behavior unchanged.
        return text[:limit]
    # When truncation is unavoidable, the Remediation section is kept in
    # preference to the earlier sections (Profile/Description/Rationale/
    # Impact/Audit).
    title = text.split("\n", 1)[0]
    budget_for_rem = limit - len(title) - 10
    rem = text[idx:idx + budget_for_rem]
    return f"{title}\n...\n{rem}"


def _knowledge_base_prompt(rule_id: str, original_task: str, cis_remediation_text: str,
                           generated_yaml: str, sibling_yamls: list = None,
                           script_content_note: str = "") -> list:
    return [
        {"role": "system", "content": (
            "You are an impartial reviewer checking whether a SPECIFIC "
            "generated Ansible fix actually implements the official CIS "
            "Benchmark Remediation instruction. Judge AGAINST THE GIVEN "
            "KNOWLEDGE-BASE REFERENCE TEXT, not from memory. "
            "CONFIRMED TECHNICAL FACT (verified against official Ansible module "
            "documentation — do not contradict this): the 'mount' module's "
            "'state: mounted' (ansible.builtin.mount / ansible.posix.mount) WRITES "
            "the entry into /etc/fstab AND mounts the filesystem immediately, in "
            "ONE task — it does NOT affect only the live/runtime state. This is a "
            "DIFFERENT, stronger guarantee than 'state: present' (which writes "
            "fstab only, taking effect after reboot) and 'state: remounted' "
            "(which affects only the live state, without persisting). A fix using "
            "'mount:' with 'state: mounted' to add/change an option (e.g. nodev/"
            "nosuid/noexec) is therefore a COMPLETE fix satisfying a CIS remediation "
            "that requires BOTH editing /etc/fstab AND applying the change now — do "
            "NOT flag it PARTIAL/NON_COMPLIANT for 'only affecting the runtime "
            "state' or 'not persisting to /etc/fstab'; that claim is factually "
            "incorrect for this specific state value. Confirmed live: this exact "
            "false claim was independently verified against the official module "
            "documentation and found wrong, after it was raised identically across "
            "16 separate mount-option fixes in one evaluation run. "
            "CONFIRMED TECHNICAL FACT — when the CIS Remediation text ITSELF uses "
            "language like 'investigate', 'review to determine', 'may be "
            "appropriate', 'if appropriate', 'establish unique ... based on the "
            "results of the audit script', 'perform the appropriate action', or "
            "explicitly says the correct value/location 'will vary depending on "
            "your environment' — "
            "the rule requires HUMAN JUDGMENT or environment-specific information "
            "that CANNOT be safely auto-computed generically (e.g. which of "
            "several files sharing a duplicate UID should keep it and what the "
            "new UID should be; whether an unconfined SELinux process is actually "
            "needed; where to physically create a new partition/volume for /tmp, "
            "/home, /var, etc.). For such a rule, a task that DETECTS the "
            "condition and FAILS LOUDLY with a clear, descriptive message (an "
            "'ansible.builtin.fail' with msg, or a shell/command with "
            "'failed_when' on the detected condition) IS a complete, correct, and "
            "HONEST implementation — do NOT mark it PARTIAL/NON_COMPLIANT for "
            "'only auditing/not performing the remediation action'; a fabricated "
            "guess at the environment-specific action would be WORSE than an "
            "honest detect-and-fail, not better. Reserve PARTIAL/NON_COMPLIANT "
            "for 'audit-only, fails loudly' ONLY when the CIS Remediation text "
            "gives an EXACT, deterministic, environment-independent command (e.g. "
            "'firewall-cmd --remove-service=<service>', 'systemctl stop X && dnf "
            "remove Y') that the fix could safely have executed using the "
            "condition it already detected, but didn't — that IS a genuine gap. "
            "CONFIRMED TECHNICAL FACT — for auditd rules (auditctl lines / /etc/audit/rules.d/*.rules "
            "entries), do NOT assume a '-F arch=b32' (32-bit) counterpart rule is required just "
            "because a task's own NAME contains '32 bit'/'32bit'/'64 bit'/'64bit' — these labels are "
            "frequently MISLEADING in this project (task names in general are sometimes garbled/"
            "non-descriptive — trust the retrieved CIS text over a task's own name for what it "
            "actually requires). Check ONLY whether the retrieved CIS Remediation/Example-output text "
            "itself explicitly shows an 'arch=b32' line anywhere — if the CIS text's own example "
            "output shows ONLY 'arch=b64' (or no arch field at all) and never mentions b32/32-bit "
            "coverage, do NOT flag a fix as PARTIAL/NON_COMPLIANT for 'missing the 32-bit rule' — "
            "that would be inventing a requirement the benchmark itself does not state for this rule. "
            "Confirmed live: rules 6.3.3.1/6.3.3.5/6.3.3.12/6.3.3.16/6.3.3.31's own project tasks are "
            "named '...32bit'/'...64bit' as arbitrary internal labels for two DIFFERENT parts of the "
            "SAME remediation (e.g. a path-based rule vs a syscall-based rule), not architecture "
            "variants — and the retrieved CIS text for every one of them shows only a single "
            "'arch=b64' example, never a b32 counterpart. "
            "CONFIRMED TECHNICAL FACT — a CIS remediation's 'EITHER/OR' framing can describe "
            "alternative REMEDIATION ACTIONS, not only alternative config VALUES (e.g. 'remove the "
            "package -OR- stop and mask the service' are both valid ways to satisfy 'service not in "
            "use'; 'the file does not exist -OR- contains X' are both valid ways to satisfy an audit). "
            "Do not require one specific action/value when the reference text offers it as one of "
            "several alternatives and the fix correctly implements a different, equally valid "
            "alternative. Confirmed live: a fix that stopped+masked a service was rejected for 'not "
            "also removing the package', when the CIS text offers package removal OR stop+mask as "
            "equally valid options. "
            "CONFIRMED TECHNICAL FACT — this pipeline's own judges have TWICE made an incorrect "
            "permission-math claim (rejecting a correct symbolic chmod as not reaching a target octal "
            "value; rejecting a correct octal mode for not textually matching a symbolic form used "
            "elsewhere). Before asserting a mode (symbolic OR octal) does or does not satisfy a "
            "required permission level, explicitly compute the resulting permission bits (owner/group/"
            "other read/write/execute) for BOTH forms and compare the actual bits, never the surface "
            "text/format. "
            "CONFIRMED TECHNICAL FACT — the 'ansible.builtin.file' module's 'state:' parameter's valid "
            "values are: absent, directory, file, hard, link, touch. 'state: touch' IS a valid, "
            "correct value (creates an empty file / updates its mtime) — do not reject it as invalid. "
            "CONFIRMED TECHNICAL FACT — in an authselect custom profile file (under /etc/authselect/"
            "custom/<name>/), the syntax '{include if \"<feature-name>\"}' is authselect's OWN native "
            "conditional-include directive (not Jinja2/Ansible templating) and is valid, standard "
            "syntax — do not flag it as broken or invented Jinja2. "
            "Before finalizing a NON_COMPLIANT or PARTIAL verdict, re-read your OWN stated reasoning: "
            "if your own explanation actually concludes the CIS requirement IS satisfied, the verdict "
            "must be COMPLIANT — never emit a verdict that contradicts your own stated reasoning. "
            "CONFIRMED against the official CIS Benchmark PDF itself (not just the Remediation prose "
            "excerpt given above, which may be truncated before the Audit section): for the "
            "kernel-module-disable family (1.1.1.1 cramfs, 1.1.1.2 freevxfs, 1.1.1.3 hfs, 1.1.1.4 "
            "hfsplus, 1.1.1.5 jffs2, 1.1.1.6 squashfs, 1.1.1.7 udf, 1.1.1.8 usb-storage, and the "
            "network-protocol equivalents 3.2.1 dccp, 3.2.2 tipc, 3.2.3 rds, 3.2.4 sctp), the "
            "Benchmark's own AUDIT script explicitly checks "
            "'grep -Pq ...install MODULE /bin/(true|false)...' and its prose states 'An entry "
            "including /bin/true or /bin/false exists' — '/bin/true' and '/bin/false' are BOTH "
            "explicitly, officially compliant values for the 'install MODULE ...' line; the "
            "Remediation section's example merely shows /bin/false as one illustrative choice, it "
            "does not exclude /bin/true. A fix using 'install MODULE /bin/true' is NOT NON_COMPLIANT "
            "for that reason — do not reject it. (The upstream ansible-lockdown RHEL9-CIS project "
            "itself uses /bin/true for exactly this family, e.g. tasks/section_1/cis_1.1.1.x.yml and "
            "tasks/section_3/cis_3.2.x.yml.) "
            "CONFIRMED — in the ansible-lockdown RHEL9-CIS role, a per-rule task that only does "
            "'set_fact: rhel9cis_sysctl_update: true' (rules 1.5.1, 1.5.2, 3.3.1-3.3.11) or "
            "'set_fact: update_audit_template: true' (6.3.3.1-6.3.3.19), optionally followed by a "
            "'debug' message, is a DEFERRED-ENFORCEMENT flag, not a no-op — tasks/main.yml always "
            "imports tasks/post.yml, whose 'POST | Update sysctl' task (when: "
            "rhel9cis_sysctl_update) templates /etc/sysctl.d/60-kernel_sysctl.conf, "
            "60-netipv4_sysctl.conf and 60-netipv6_sysctl.conf (each value gated by its own rule "
            "flag) and notifies 'Reload sysctl' (runs sysctl --system); tasks/main.yml also imports "
            "tasks/auditd.yml (when: update_audit_template), which templates "
            "/etc/audit/rules.d/99_auditd.rules and notifies 'augenrules --load'. Any 'ACTUAL "
            "ENFORCEMENT CONTENT' block shown alongside such a task IS its real remediation — do NOT "
            "mark it NON_COMPLIANT for 'only sets a fact / only prints a debug message / does not "
            "write /etc/sysctl.d or /etc/audit/rules.d' when that block is present and contains the "
            "required directive. "
            "CONFIRMED — ansible-lockdown's mount-option tasks (1.1.2.1.2 through 1.1.2.7.4) that "
            "'set_fact: prelim_mount_point_fs_and_options' to append nodev/nosuid/noexec, with "
            "'notify: \"Remount {{ mount_point }}\"', are complete remediations: handlers/main.yml "
            "defines, under 'listen: \"Remount <mount_point>\"', an ansible.posix.mount task with "
            "'state: present' (writes /etc/fstab) followed by one with 'state: remounted' (applies "
            "it live) — together equivalent to 'state: mounted'. Do NOT flag these as 'only updates "
            "an in-memory fact / never writes /etc/fstab / never remounts' when a 'WHAT THE NOTIFIED "
            "HANDLER(S) ACTUALLY RUN' block confirms this. "
            "CONFIRMED against the official CIS PDF (2.4.2.1 Audit section): /etc/at.allow (and "
            "/etc/at.deny if present) is compliant when owned by root and 'group owned by the group "
            "daemon or group root' — the Audit text itself shows 'Group: (daemon)' -OR- 'Group: "
            "(root)' as equally valid. A fix setting 'group: root' is NOT PARTIAL or NON_COMPLIANT "
            "for not using 'daemon' instead. "
            "CONFIRMED against the official CIS PDF (5.4.3.2 Audit script): the audit regex accepts "
            "'readonly TMOUT' and 'export TMOUT' as separate lines, in addition to the single-line "
            "'typeset -xr TMOUT=N' form. The 3-line form 'TMOUT=900' / 'readonly TMOUT' / 'export "
            "TMOUT' therefore PASSES the benchmark's own audit; the Remediation section's 'deprecated "
            "methods' label is a style preference, not a compliance failure — do not mark it "
            "NON_COMPLIANT for using the 3-line form when the value, readonly, and export are all "
            "otherwise correct. "
            "CONFIRMED against the official CIS PDF (5.4.2.4): the remediation is 'passwd root' "
            "(an operator-chosen secret, environment-specific) -OR- 'usermod -L root' (whose own "
            "Impact section warns it can break root-dependent automation) — a site-policy choice, "
            "not a single deterministic command. A task that runs the Audit check itself ('passwd "
            "-S root', accepting either 'Password set' or 'Password locked' as compliant) and FAILS "
            "LOUDLY (assert/failed_when) when neither holds is a complete, honest implementation — "
            "do not mark it NON_COMPLIANT for not itself choosing and applying one of the two "
            "site-policy options."
        )},
        {"role": "user", "content": f"""CIS rule: {rule_id}
Original (non-compliant) task: {original_task}

KNOWLEDGE-BASE REFERENCE (official CIS Benchmark Audit + Remediation text — source of reference.
The AUDIT section is the actual PASS CRITERIA: if the fix's resulting configuration already
satisfies it, the fix IS compliant even if it differs from the Remediation section's own example):
{_remediation_priority_slice(cis_remediation_text, 3000) or "[NOT FOUND — no remediation chunk indexed for this rule]"}

GENERATED FIX (Ansible YAML):
{generated_yaml[:2000]}
{script_content_note}
{_sibling_block(sibling_yamls or [])}
Does this fix — COMBINED with any sibling coverage shown above — CORRECTLY implement the Remediation instruction above?
Respond with ONLY valid JSON, no markdown:
{{"verdict": "COMPLIANT" or "PARTIAL" or "NON_COMPLIANT",
"score": integer 1-5 (5 = fully implements the remediation, 1 = does not implement it at all or does something unrelated),
"reason": "1-2 sentences, specifically what is correct/missing"}}"""},
    ]


def grounded_judge_one(rule_id: str, original_task: str, generated_yaml: str,
                       judge_model: str = JUDGE_MODEL, sibling_yamls: list = None,
                       script_content_note: str = "") -> dict:
    """One knowledge-base-grounded score of one fix. Returns a dict with
    verdict/score/reason/error. Never raises: a failure is returned in the
    'error' field, so the calling loop can continue.

    sibling_yamls: final YAML of other generated fixes for the same rule_id
    (see _sibling_block).

    script_content_note: the actual content of a referenced shell script
    (module 'script:'/'cmd:'), when the fix's logic lives in an external
    file rather than in the YAML."""
    # Both the Audit section (pass criteria, often with accepted
    # alternatives) and the Remediation section are given to the judge.
    cis_text = get_audit_and_remediation_chunks([rule_id]) or ""
    messages = _knowledge_base_prompt(rule_id, original_task, cis_text, generated_yaml,
                                      sibling_yamls, script_content_note)

    def _build_payload(max_tokens: int) -> bytes:
        return json.dumps({
            "model": judge_model,
            # enable_thinking:False — a short compliance check; with
            # reasoning enabled the judge model can use up max_tokens before
            # returning the JSON.
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()

    # If the previous attempt was truncated (finish_reason='length'),
    # max_tokens is raised on retry instead of repeating the same request.
    max_tokens = 500
    prev_truncated = False

    for attempt in range(1, 3):
        if prev_truncated:
            max_tokens = 1500
        payload = _build_payload(max_tokens)
        t0 = time.time()
        try:
            req = urllib.request.Request(
                agent_a2_llm.API_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {agent_a2_llm.API_TOKEN}",
                         "User-Agent": "curl/7.88.1"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=150) as resp:
                data = json.loads(resp.read())
            choice = data["choices"][0]
            # Recorded ONCE, after a usable result is actually parsed —
            # not right after urlopen() succeeds, otherwise a downstream
            # parsing failure double-counts this same HTTP call (one
            # success + one spurious failure in the except block below).
            text = choice["message"]["content"].strip()
            json_str = extract_json_object(text)
            if not json_str:
                agent_a2_llm._record_request(time.time() - t0, success=False)
                if choice.get("finish_reason") == "length":
                    prev_truncated = True
                if attempt < 2:
                    continue
                return {"verdict": "", "score": None, "reason": "",
                        "error": "judge did not return valid JSON",
                        "knowledge_base_found": bool(cis_text)}
            result = json.loads(json_str)
            agent_a2_llm._record_request(time.time() - t0, success=True)
            return {
                "verdict": result.get("verdict", ""),
                "score": result.get("score"),
                "reason": result.get("reason", ""),
                "error": "",
                "knowledge_base_found": bool(cis_text),
            }
        except Exception as e:
            agent_a2_llm._record_request(time.time() - t0, success=False)
            if attempt < 2:
                continue
            return {"verdict": "", "score": None, "reason": "",
                    "error": str(e), "knowledge_base_found": bool(cis_text)}


def run_grounded_judge(report_json_path: str, judge_model: str = JUDGE_MODEL,
                       out_csv: str = None, limit: int = None,
                       collection: str = "cis_flat"):
    # grounded_judge_one() -> agent_a3_llm.get_remediation_chunk() reads
    # agent_a2_llm.COLLECTION. Must match the benchmark the report was
    # produced against or every fix is judged against the wrong CIS
    # document (default 'cis_flat' = RHEL 10). Same fix as judge_combined.py
    # / judge_audit.py. Default unchanged so RHEL runs are unaffected.
    agent_a2_llm.COLLECTION = collection
    print(f"[grounded-judge] retrieving CIS text from collection: '{collection}'")
    with open(report_json_path, encoding="utf-8") as f:
        report = json.load(f)

    details = report.get("remediation_details")
    if not details:
        print(f"[grounded-judge] MISSING 'remediation_details' in {report_json_path} — "
              f"this report was saved before 2026-08-23 (when this field was added), "
              f"or the model generated no fixes. Nothing to score.")
        sys.exit(1)

    base = os.path.basename(report_json_path)
    model_name = base.replace("final_report__", "").rsplit(".json", 1)[0]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    safe_judge = "".join(c if c.isalnum() or c in "-_." else "_" for c in judge_model)
    out_csv = out_csv or f"results/grounded_judge__{safe}__judge_{safe_judge}.csv"

    if limit:
        details = details[:limit]

    print(f"[grounded-judge] Model: {model_name} | Judge: {judge_model} | "
          f"{len(details)} fix(es) to score | writing to: {out_csv}")

    # ansible_lint_valid (lint only) and agent4_semantic_valid are read from
    # remediation_details as separate columns. grounded_verdict/_score/
    # _reason are this script's own, independent assessment.
    fieldnames = ["model", "rule_id", "action_type", "ansible_lint_valid",
                  "agent4_semantic_valid", "knowledge_base_found",
                  "grounded_verdict", "grounded_score", "grounded_reason", "error"]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(details, 1):
            rule_id = d.get("rule_id", "")
            original_task = d.get("original_task", "")
            generated_yaml = d.get("final_yaml") or d.get("yaml_fix") or ""

            print(f"  [{i}/{len(details)}] Scoring {rule_id} ({d.get('action_type')})...")
            result = grounded_judge_one(rule_id, original_task, generated_yaml, judge_model)

            writer.writerow({
                "model": model_name,
                "rule_id": rule_id,
                "action_type": d.get("action_type", ""),
                "ansible_lint_valid": d.get("lint_valid"),
                "agent4_semantic_valid": d.get("semantic_valid"),
                "knowledge_base_found": result.get("knowledge_base_found"),
                "grounded_verdict": result.get("verdict", ""),
                "grounded_score": result.get("score"),
                "grounded_reason": result.get("reason", ""),
                "error": result.get("error", ""),
            })
            f.flush()

            if result.get("error"):
                print(f"      error: {result['error']}")
            else:
                print(f"      {result.get('verdict')} ({result.get('score')}/5)"
                      + ("" if result.get("knowledge_base_found") else " [WARNING: no CIS text found in the knowledge base]"))

    print(f"[grounded-judge] DONE. Saved: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print(__doc__)
        sys.exit(1)
    _report_path = sys.argv[1]
    _judge = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--judge=")), JUDGE_MODEL)
    _out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    _limit = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
    _collection = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--collection=")), "cis_flat")
    run_grounded_judge(_report_path, judge_model=_judge, out_csv=_out, limit=_limit, collection=_collection)
