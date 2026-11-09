"""
Agent A3 — YAML Generator and Fixer (Remediation & Creation Agent)
=====================================================================
Task 1: For every NON_COMPLIANT/PARTIAL task, generate a fix.
Task 2: For every missing rule (MISSING), generate a brand-new task from scratch.
Uses the university's LLM (OpenAI-compatible API).
"""

import os
import json
import yaml
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import agent_a2_llm
from agent_a2_llm import ComplianceResult, TaskDescription, assess_missing_rule_cvss, _resolved_params_note, _template_content_note
from debug_dump import dump_llm_call

# API_URL/API_TOKEN/API_MODEL are read from agent_a2_llm at call time (not
# imported by value), so a model set by orchestrator.py applies everywhere.
USE_LLM   = True


# ── Output structure ──────────────────────────────────────────────────────────
@dataclass
class Remediation:
    original_task: str
    rule_id:       str
    rule_title:    str
    yaml_fix:      str
    explanation:   str
    source:        str = "CIS_REMEDIATION"
    action_type:   str = "FIX" # "FIX" for fixes, "CREATE" for new rules
    source_file:   str = ""    # path to the tasks/ file containing the original task
                                # (FIX only — needed by Agent A4 to locate and
                                # replace the task after saving; empty for CREATE,
                                # since there is no target file yet in that case)
    severity:      str = ""    # LOW/MEDIUM/HIGH/CRITICAL, derived from cvss_score.
                                # Used by Agent A4 to scale the number of repair attempts.
    cvss_vector:   str = ""    # e.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N"
    cvss_score:    Optional[float] = None  # 0.0-10.0, per the official CVSS v3.1 formula
    rule_ids:      List[str] = field(default_factory=list)  # ALL CIS rule IDs this ONE task must
                                # satisfy (one task can cover several rules, e.g. one mount
                                # task setting nodev+nosuid+noexec).
    sibling_context: str = ""  # Pre-rendered "OTHER TASKS FOR THE SAME CIS RULE... do NOT
                                # duplicate" block (see render_sibling_block()), also passed
                                # to Agent 4's repair prompts.
    original_path: str = ""    # The original task's target path/resource (from
                                # _extract_target_path(task.params)), re-checked by Agent
                                # A4's path guardrail.


# ── Remediation section extractor from a CIS chunk ────────────────────────────
def get_remediation_chunk(rule_id: str) -> str:
    """
    Fetches from ChromaDB the chunk with chunk_type=remediation for the given
    rule. For rules stored as a single chunk (chunk_type=full, within the
    1024-token threshold) it returns the whole chunk, and the Remediation
    section is then extracted by extract_remediation_text().
    """
    try:
        from agent_a2_llm import get_collection
        col = get_collection()
        res = col.get(where={"rule_id": rule_id}, include=["documents", "metadatas"])
        if not res["documents"]:
            return ""

        # Prefer chunk_type=remediation if it exists
        for doc, meta in zip(res["documents"], res["metadatas"]):
            if meta.get("chunk_type") == "remediation":
                return doc

        # Fallback: rule was not split (full) - return the whole text,
        # extraction via regex will happen in extract_remediation_text
        for doc, meta in zip(res["documents"], res["metadatas"]):
            if meta.get("chunk_type") in ("full", "full_fallback"):
                return doc

        # Last-resort fallback: first available chunk
        return res["documents"][0]
    except Exception as e:
        print(f"  [A3] Error fetching remediation chunk for {rule_id}: {e}")
        return ""


def get_remediation_chunks_multi(rule_ids: List[str]) -> str:
    """Like get_remediation_chunk(), but for a single Ansible task that maps to
    several CIS rules at once (e.g. one 'ansible.posix.mount' task setting
    nodev+nosuid+noexec). Concatenates each rule's Remediation section under
    its own '=== CIS Rule X ===' header, so Agent 3 and Agent 4's semantic
    check see all applicable rules.
    """
    if not rule_ids:
        return ""
    parts = []
    for rid in rule_ids:
        chunk = get_remediation_chunk(rid)
        if not chunk:
            continue
        parts.append(f"=== CIS Rule {rid} ===\n{extract_remediation_text(chunk)}")
    return "\n\n".join(parts)


def extract_remediation_text(chunk_text: str) -> str:
    """Extracts only the remediation instructions section from the CIS documentation.

    The section ends at a known section header (optionally '#'-prefixed).
    'Example:' headers are not treated as section ends, because they appear
    inside CIS remediation steps."""
    _REM_END_HEADINGS = (
        r"References:|CIS Controls:|Additional Information:|Default Value:|"
        r"Notes?:|Impact:|Audit:"
    )
    patterns = [
        rf"Remediation:\s*\n(.*?)(?=\n#{{0,3}}\s*(?:{_REM_END_HEADINGS})|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, chunk_text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            if len(text) > 20:
                return text

    lines = chunk_text.strip().split("\n")
    remediation_lines = []
    in_rem = False
    for line in lines:
        if re.match(r"^remediation:", line, re.I):
            in_rem = True
            continue
        if in_rem:
            if re.match(r"^(references:|cis controls:|audit:)", line, re.I):
                break
            remediation_lines.append(line)

    if remediation_lines:
        return "\n".join(remediation_lines).strip()
    return chunk_text[:500] # Fallback: return the start of the text if there is no section


def extract_audit_text(chunk_text: str) -> str:
    """Extracts only the Audit section (the CIS Benchmark's own PASS CRITERIA —
    what configuration already counts as compliant) from a 'full' (unsplit)
    chunk. Mirrors extract_remediation_text() above — see its docstring for
    the chunking background (rules >1024 tokens get a separate chunk_type==
    'audit' chunk fetched directly by get_audit_chunk() below; shorter rules
    are a single 'full' chunk and need this regex extraction instead)."""
    _AUDIT_END_HEADINGS = (
        r"References:|CIS Controls:|Additional Information:|Default Value:|"
        r"Notes?:|Impact:|Remediation:"
    )
    pat = rf"Audit:\s*\n(.*?)(?=\n#{{0,3}}\s*(?:{_AUDIT_END_HEADINGS})|\Z)"
    m = re.search(pat, chunk_text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
        if len(text) > 20:
            return text
    return ""


def get_audit_chunk(rule_id: str) -> str:
    """Fetches the chunk_type=='audit' chunk from ChromaDB for the given rule:
    the CIS Benchmark's pass criteria, i.e. what configuration already counts
    as compliant. For rules stored as a single chunk, the Audit section is
    extracted by regex.

    The Audit section often accepts alternatives that the Remediation example
    does not show (e.g. rule 5.1.7 accepts DenyUsers, DenyGroups, AllowUsers
    or AllowGroups). See get_audit_and_remediation_chunks()."""
    try:
        from agent_a2_llm import get_collection
        col = get_collection()
        res = col.get(where={"rule_id": rule_id}, include=["documents", "metadatas"])
        if not res["documents"]:
            return ""
        for doc, meta in zip(res["documents"], res["metadatas"]):
            if meta.get("chunk_type") == "audit":
                return doc
        for doc, meta in zip(res["documents"], res["metadatas"]):
            if meta.get("chunk_type") in ("full", "full_fallback"):
                return extract_audit_text(doc)
        return ""
    except Exception as e:
        print(f"  [A3] Error fetching audit chunk for {rule_id}: {e}")
        return ""


def get_audit_and_remediation_chunks(rule_ids: List[str]) -> str:
    """Like get_remediation_chunks_multi(), but prepends each rule's Audit
    section (the actual PASS CRITERIA) ahead of its Remediation section —
    see get_audit_chunk()'s docstring for why this matters. Use this instead
    of get_remediation_chunk(s)_multi wherever a model must JUDGE whether an
    existing/generated configuration is already compliant (the judge, Agent
    4's semantic check, Agent 3's fix generation) — NOT needed for callers
    that only display reference text for humans."""
    if not rule_ids:
        return ""
    parts = []
    for rid in rule_ids:
        rem_raw  = get_remediation_chunk(rid)
        rem_text = extract_remediation_text(rem_raw) if rem_raw else ""
        audit_text = get_audit_chunk(rid)
        if not rem_text and not audit_text:
            continue
        block = f"=== CIS Rule {rid} ==="
        if audit_text:
            block += ("\nAUDIT — PASS CRITERIA (if the current configuration already satisfies "
                      "this, it IS compliant — do not require the Remediation section's exact "
                      "wording/value when the Audit section itself accepts an alternative):\n"
                      + audit_text.strip())
        if rem_text:
            block += "\nREMEDIATION — one concrete way to become compliant (not the only way):\n" + rem_text
        parts.append(block)
    return "\n\n".join(parts)


# ── Handler discovery — so a fix can 'notify:' an existing activation step ────
def discover_handlers(project_path: str) -> List[dict]:
    """Scans roles/*/handlers/*.yml and the project's root-level
    handlers/*.yml (a project can itself be a single role) and returns a flat
    list of {"name": ..., "listen": [...], "action": ...}: the handler's name,
    its 'listen:' topics (both can be targeted by 'notify:') and a one-line
    summary of what it runs (e.g. 'command: augenrules --load').

    Lets a fix trigger the project's existing activation step through
    'notify:'. Never raises: a missing or unparseable file gives an empty
    list."""
    import glob
    handlers = []
    patterns = [
        os.path.join(project_path, "roles", "*", "handlers", "*.yml"),
        os.path.join(project_path, "handlers", "*.yml"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for h in data:
                if not isinstance(h, dict) or "name" not in h:
                    continue
                # First module key that isn't a bare Ansible keyword — same
                # "module: params" shape task_dict already assumes elsewhere
                # in this file (see _mock_fix).
                skip = {"name", "become", "become_user", "when", "listen",
                        "ignore_errors", "changed_when", "failed_when", "tags",
                        "vars", "register", "notify", "loop", "with_items"}
                action = ""
                for k, v in h.items():
                    if k not in skip:
                        action = f"{k}: {v}"
                        break
                listen = h.get("listen")
                if isinstance(listen, str):
                    listen = [listen]
                elif not isinstance(listen, list):
                    listen = []
                handlers.append({"name": h["name"], "listen": listen, "action": action})
    return handlers


def render_handlers_block(handlers: List[dict]) -> str:
    """Renders the 'EXISTING PROJECT HANDLERS' block shown to _llm_fix()
    (and, via Remediation, to Agent 4's repair prompts) — see
    discover_handlers()'s docstring for the concrete gap this fixes."""
    if not handlers:
        return ""
    h_list = "\n".join(f"  - name: \"{h['name']}\"  (runs: {h['action']})" for h in handlers[:30])
    return f"""
EXISTING PROJECT HANDLERS (already defined in this project's handlers/ files —
Ansible runs a notified handler ONCE, automatically, at the end of the play,
even if several tasks notify it):
{h_list}
ONLY use 'notify: "<exact handler name>"' when the CIS remediation text ITSELF
explicitly requires an activation/reload/restart step (e.g. 'run systemctl
reload-or-restart X', 'run sysctl --system') AND a handler above's own action
is a SEMANTIC, SPECIFIC match for exactly that named service/subsystem — not
just any handler that happens to exist. Confirmed live: a fix for a modprobe
kernel-module rule (install/blacklist directives, which take effect on the
NEXT module load attempt with NO 'apply now' command needed at all) wrongly
added 'notify: "load rules"' (a handler that runs 'augenrules --load' — for
AUDIT rules, entirely unrelated) merely because a handler existed with a
plausible-sounding name. Most rules need NO activation step at all — do not
add a notify (or invent a new command) just because the handler list is
non-empty; only do so when the retrieved remediation text names that specific
step AND a listed handler's own action genuinely performs it.
"""


# ── Mode 1: Fixing an existing task (LLM) ─────────────────────────────────────
def render_sibling_block(sibling_tasks: List[TaskDescription]) -> str:
    """Renders the 'OTHER TASKS FOR THE SAME CIS RULE... do NOT duplicate' block
    used by _llm_fix() and, via Remediation.sibling_context, by Agent 4's
    repair prompts, so that repaired fixes also avoid duplicating sibling
    tasks.
    """
    if not sibling_tasks:
        return ""
    # Siblings are shown as full raw YAML, so that register/when/failed_when
    # keys are visible to the model.
    sib_list = "\n".join(
        f"  - {s.task_name}:\n"
        + "\n".join(f"      {line}" for line in (s.raw_yaml or f"(module: {s.module}, params: {json.dumps(s.params)})").split("\n"))
        for s in sibling_tasks[:10]
    )
    return f"""
OTHER TASKS FOR THE SAME CIS RULE (already in the project, do NOT duplicate their work —
shown as their exact raw YAML, including register/when/failed_when — a task below may
register: a variable that OTHER tasks for this same rule legitimately depend on):
{sib_list}
Fix ONLY what is actually wrong in THIS task's own logic/parameters. Do NOT add a
directive, module call, or step that one of the sibling tasks above already
implements — that would create a redundant/duplicate task. If the non-compliance
finding turns out to be purely about missing coverage that a sibling ALREADY
provides, make the minimal correction to this task alone and do not re-implement
the sibling's part here.
IF you decide this task's underlying approach is wrong and switch to a DIFFERENT
module/target entirely (not just adjusting parameters of the same approach): check
your new approach against EVERY sibling listed above, not only the one a reviewer's
finding happened to mention — a pivot to a new approach can accidentally duplicate
a DIFFERENT sibling than the one you were originally told about.
"""


def _llm_fix(task: TaskDescription, result: ComplianceResult, rem_text: str, extra_note: str = "",
             sibling_tasks: List[TaskDescription] = None, handlers: List[dict] = None) -> str:
    rule_id    = result.rule_ids[0]    if result.rule_ids    else "?"
    rule_title = result.rule_titles[0] if result.rule_titles else ""

    # When one task must satisfy several CIS rules, the header lists all of
    # them, so the model knows they all apply to this task.
    rule_ids_header = (
        f"CIS Rules (this ONE task must satisfy ALL of them): {', '.join(result.rule_ids)}"
        if len(result.rule_ids) > 1
        else f"CIS Rule: {rule_id} - {rule_title}"
    )

    # Sibling tasks of the same rule are shown, so that the fix does not
    # duplicate what another task already does.
    sibling_block = render_sibling_block(sibling_tasks)
    handlers_block = render_handlers_block(handlers or [])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior Ansible hardening engineer specializing in CIS Benchmark compliance, "
                "fixing a non-compliant task STRICTLY GROUNDED in the provided CIS text. "
                "ANTI-HALLUCINATION RULES (critical): "
                "A. Use ONLY concrete values (module names, file paths, parameters) EXPLICITLY written in "
                "the provided CIS text below. NEVER invent, extend, or complete a list beyond what is "
                "literally stated in the source text, even if it seems incomplete. "
                "B. If the CIS text describes a DYNAMIC discovery procedure (e.g. run this script to "
                "generate a list of currently loaded or mounted items) rather than a FIXED list, your fix "
                "MUST implement that dynamic discovery using shell or command module running the audit "
                "logic and registering output, NOT a hardcoded static list you invented. "
                "C. If the CIS text gives only a FEW illustrative examples rather than an exhaustive list, "
                "do NOT treat those examples as complete - implement the discovery logic shown in the "
                "Audit section instead, or use a loop variable placeholder the engineer must populate. "
                "D. If unsure whether a list is exhaustive or illustrative, prefer the SAFER, SHORTER "
                "interpretation and add a YAML comment noting the list should be reviewed by the engineer. "
                "E. Fix ONLY the specific non-compliance described - do not add unrelated changes. "
                "GENERAL ANSIBLE BEST PRACTICES (apply these on every fix, not just the CIS-specific "
                "cases below): (1) Prefer a dedicated Ansible module (mount, lineinfile, systemd, "
                "sysctl, ...) over shell/command whenever one exists — dedicated modules are idempotent "
                "and safer. (2) NEVER guess a module's parameter name or a fact's field name by "
                "analogy or because it 'sounds right' — a fact's field name and a module's parameter "
                "for the same underlying data are often NOT spelled the same. (3) Every task must be "
                "idempotent (safe to re-run without side effects or duplicated entries) — use "
                "unique/select filters and precise 'when' guards rather than blindly re-applying. "
                "(4) Prefer built-in Ansible facts (ansible_facts, e.g. ansible_mounts, "
                "ansible_distribution) over parsing shell command output when the same information is "
                "already available as a fact. (5) Prefer safe, well-known Jinja2 filters/tests (split, "
                "join, unique, selectattr, regex_search) over a custom regex_replace() for string "
                "manipulation — regex is powerful but easy to get subtly wrong. "
                "F. To add a value to an EXISTING comma-separated string (e.g. appending a mount option "
                "to ansible_mounts' 'options' field) use safe list operations: "
                "\"{{ (item.options.split(',') + ['newvalue']) | unique | join(',') }}\" — NEVER a custom "
                "regex_replace() for this. A zero-width regex like '(?<!^)(?<!,)' matches at EVERY "
                "character boundary, not just where you intend — confirmed live to turn "
                "'rw,relatime,seclabel' into 'r,w,,r,e,l,a,t,i,m,e,,s,e,c,l,a,b,e,l,' (a comma inserted "
                "between every letter), silently corrupting the fstab entry. That 'unique' filter "
                "already makes the append idempotent by itself — do NOT also add an extra 'when' "
                "condition (e.g. a regex_search checking whether the value is already present) to "
                "'guard' against re-adding it; such conditions are unnecessary and easy to get "
                "subtly wrong (confirmed live: a regex like '(?<!^)(?<!,)(?<!,nodev)(?<!nodev,)nodev' "
                "searching a comma-separated options string for 'nodev' NEVER matches ANYTHING, "
                "because 'nodev' in such a string is always either at the very start or right after "
                "a comma, and both are excluded by the lookbehinds — making the 'when' condition "
                "silently, permanently True, a dead check that looks like a safeguard but does "
                "nothing). "
                "G. When looping over 'ansible_mounts | selectattr(...)' (the standard fact for mount "
                "info), 'item' has ONLY these fields: mount, device, fstype, options, uuid, size_total, "
                "size_available — there is NO 'src' field. Use 'item.device' for the mount module's "
                "'src' PARAMETER, never 'item.src' (confirmed live: a fix used 'item.src', which does "
                "not exist on ansible_mounts and silently evaluates to Undefined at runtime). "
                "H. When using the mount module to CHANGE OPTIONS on an ALREADY-MOUNTED filesystem "
                "(e.g. adding nodev/nosuid/noexec), use state: mounted, NEVER state: present. "
                "'present' only writes /etc/fstab — the change takes effect only after a reboot. CIS "
                "Audit checks (e.g. 'findmnt | grep -v nodev') read the LIVE mount table, not fstab, so "
                "a 'present'-only fix leaves the system NON_COMPLIANT until reboot even though fstab is "
                "correct — the CIS Remediation text itself confirms this by requiring BOTH steps ('edit "
                "/etc/fstab' AND 'run mount -o remount'); 'state: mounted' does both in one task. "
                "The mount MODULE's parameter for the options string is 'opts' — NEVER 'options' "
                "(that is the ansible_mounts FACT's key name for the same data, an easy mix-up; "
                "confirmed live: 'options' fails ansible-lint with 'Unsupported parameters'). "
                "I. NEVER use a Jinja lookup/query() (e.g. query('shell', ...), lookup('shell', ...), "
                "query('pipe', ...)) inside a 'when:' condition to check REMOTE host state — lookups "
                "execute on the ANSIBLE CONTROL NODE, not the managed host, so the result would reflect "
                "the WRONG machine entirely, regardless of which lookup plugin name is used. There is "
                "also no built-in 'shell' lookup plugin at all (confirmed: the real ones are 'pipe', "
                "'file', 'env', 'password', 'lines', 'ini' — 'shell' does not exist and would fail at "
                "runtime with a lookup-plugin-not-found error). To check remote command output before "
                "deciding whether to run something, use a SEPARATE preceding task with 'register:', then "
                "reference that registered variable's .stdout/.rc in this task's own 'when:' — never an "
                "inline lookup/query(). "
                "J. GNOME dconf LOCK files (/etc/dconf/db/*.d/locks/*, used to prevent a setting from "
                "being overridden) use a DIFFERENT, SIMPLER format than dconf VALUE files: each line is "
                "JUST the full GSettings key path starting with '/', one per line — e.g. "
                "'/org/gnome/desktop/session/idle-delay' — NO '[section]' header and NO 'lock' keyword "
                "(confirmed via official GNOME documentation: help.gnome.org/admin/system-admin-guide, "
                "'Lock down specific settings'). The '[section]' + 'key = value' format is ONLY for dconf "
                "VALUE files (e.g. /etc/dconf/db/*.d/*, setting the actual default) — do NOT mix the two "
                "formats or invent a 'lock <key>' directive, which does not exist. Unless the CIS "
                "remediation text EXPLICITLY names a different database (e.g. 'gdm.d' specifically "
                "for GDM login-screen-only settings), the standard, machine-wide dconf database is "
                "'local.d' — write to /etc/dconf/db/local.d/ and /etc/dconf/db/local.d/locks/, NOT "
                "/etc/dconf/db/gdm.d/, even for a rule whose title mentions 'GDM' (many GDM-related "
                "CIS rules, e.g. screen-lock and idle-delay settings, are enforced via the general "
                "'local.d' database, not a GDM-specific one) — verify against the exact path the "
                "retrieved CIS remediation text itself gives, never assume 'gdm.d' from the rule's "
                "title alone. "
                "K. Ansible file-module 'mode:' accepts SYMBOLIC notation (u/g/o/a + +/-/= + rwx), "
                "e.g. 'go-rwx' means 'remove read/write/execute for Group and Other' — this TIGHTENS "
                "permissions, it is NOT the octal value 0777 and does not grant broad access. Never "
                "read a symbolic mode string as if its letters were octal digits, and never 'fix' a "
                "correctly-tightened symbolic mode by rewriting the task's actual target/content. This "
                "ALSO applies to multi-clause mode strings, e.g. 'o-x,g-wx,o-rwx' — every '-' clause "
                "still REMOVES no matter how many clauses there are or whether one re-targets the same "
                "letter as another; a '-' clause never adds a permission back. "
                "L. The 'ansible.builtin.replace'/'lineinfile'/'blockinfile' modules' 'path:' "
                "parameter takes EXACTLY ONE file path — it does NOT accept a glob pattern like "
                "'/etc/rsyslog.d/*.conf'. To apply the same edit across a directory of files, use "
                "a preceding task with 'ansible.builtin.find' (or 'with_fileglob') to enumerate the "
                "real files first, then loop the edit task over that discovered list. A loop "
                "variable only has dict-style attributes ('item.somekey') when its SOURCE is a "
                "list of dicts — a loop built from 'stdout_lines' or a 'regex_replace' filter is a "
                "plain string list with NO such attributes; referencing 'item.something' on it is "
                "Undefined and fails at runtime. "
                "N. For an AUDIT/CHECK-style task (verifying state, not a remediation command): "
                "'ignore_errors: yes' and NO 'register:' (or one never checked by a "
                "'failed_when:'/'assert:') does not actually verify anything — never generate a "
                "'compliance check' this way; always pair a check with a 'register:' + 'failed_when:' "
                "(or 'assert:') so its result is actually evaluated. This does NOT apply to a "
                "state-changing remediation command (dnf/package, a mutating firewall-cmd/systemctl "
                "call) merely swallowing its own failure — that is a separate, minor robustness point. "
                "This concern applies ONLY when failure-suppression is ACTUALLY PRESENT (ignore_errors: "
                "yes / failed_when: false) — a plain task with normal failure semantics needs no "
                "separate verification step. "
                "M. The 'ansible.posix.sysctl' module's 'state:' parameter accepts ONLY 'present' "
                "or 'absent' — there is NO 'immediate'/'applied'/'reloaded' value. The module "
                "ALREADY applies the value immediately via 'sysctl -w' by default — 'state: "
                "present' (or omitting 'state') is already the correct, complete fix; never invent "
                "a state value to express 'apply now'. "
                "O. A CIS rule about DUPLICATE identifiers (duplicate UIDs, GIDs, user names, or "
                "group names) requires ESTABLISHING A NEW, UNIQUE value for the duplicate — NEVER "
                "'state: absent' / deleting the account or group (destructive, confirmed live rule "
                "7.2.6: remediation text says 'establish unique user names... file ownerships will "
                "automatically reflect the change', a rename, not a deletion). BUT ALSO: the CIS "
                "text itself typically phrases this as a MANUAL review step ('based on the results "
                "of the audit script, establish unique...') — there is USUALLY no safe, generic way "
                "to auto-compute what the new unique value should be without knowing which specific "
                "GID/UID a set of files is actually supposed to belong to. Do NOT invent a variable "
                "(e.g. a 'duplicate_x_mapping' fact) that no other task in the project actually "
                "defines or populates — a task referencing an undefined/always-empty variable is a "
                "silent no-op, not a fix. When you cannot safely auto-compute the correct new value, "
                "the CORRECT implementation is a task that DETECTS the duplicate and FAILS LOUDLY "
                "with a clear message directing to manual remediation (same pattern as an "
                "'ansible.builtin.fail' with a descriptive msg) — an honest, visible non-fix is "
                "better than a fabricated one that silently does nothing or breaks the system. "
                "P. Never use 'authselect select <profile> ...' to 'fix' a rule about a SPECIFIC PAM "
                "module's presence (e.g. 'ensure pam_unix/pam_pwquality/pam_faillock module is "
                "enabled') — 'authselect select' REPLACES THE ENTIRE ACTIVE PROFILE with a named "
                "one, which can silently drop unrelated, already-correct settings and features the "
                "system currently has enabled; it is not a targeted fix for one module's presence, "
                "and guessing which profile/features to select is fabricating configuration state "
                "with no basis in the task's own context. 'authselect select <profile> --with-X' "
                "additionally only accepts REAL, existing authselect feature flags (e.g. "
                "'--with-faillock', '--with-sudo') — there is no '--with-pam' flag; pam_unix is part "
                "of every baseline profile already, not an optional feature. A CIS remediation "
                "phrased as a verification script (e.g. grep for 'pam_unix.so' inside the profile's "
                "own template files under /etc/authselect/ or /usr/share/authselect/default/) should "
                "be implemented as EXACTLY that check — a shell/command task running the equivalent "
                "grep and failing if not found — never as a profile-replacing 'authselect select'. "
                "Q. The 'CIS Rule: <id> - <title>' header above is the AUTHORITATIVE description of "
                "what is required — trust it over the original (possibly typo'd, garbled, or "
                "inverted) task name shown in the project's own source. Confirmed live: a project "
                "task named 'Ensure nodexec option set on /var/log partition' (a typo — 'nodexec', "
                "not 'noexec') produced a fix that added 'exec' to the mount options, the OPPOSITE "
                "of what CIS rule 1.1.2.6.4 ('Ensure noexec option set...') actually requires — the "
                "task's own garbled name was mistakenly treated as authoritative over the correct "
                "rule title provided above it. Before writing a fix whose effect is the OPPOSITE of "
                "a positive/prohibitive framing in the rule title (set vs. not set, enable vs. "
                "disable, present vs. absent), re-read the CIS Rule title and Remediation text above "
                "and resolve any conflict in ITS favor, not the source project's own (possibly "
                "mistagged) task name. "
                "R. The CIS Remediation text sometimes includes an 'Example Script' — a dense bash "
                "one-liner CIS itself provides as ONE way to implement the step (e.g. finding and "
                "commenting out conflicting sysctl entries across multiple candidate files). Do NOT "
                "copy that script verbatim into a 'shell:' task — a code-quality reviewer running "
                "the SAME analysis on this pipeline's own output confirmed live (24 sysctl rules, "
                "3.3.1.x/3.3.2.x family) that doing so scores 1-2/5 on readability/maintainability "
                "(obscure regex, dense variable manipulation) AND 1-2/5 on error handling (no real "
                "failure detection — 'changed_when: result.rc == 0' does NOT stop the play on "
                "failure the way 'failed_when' does, it only affects the changed/ok label). Extract "
                "the INTENT from the example script (e.g. 'find files matching X pattern, comment "
                "out lines matching Y') and implement it with Ansible-native constructs where "
                "practical (a loop over a small, explicit file list with 'lineinfile'/'replace', or "
                "a SHORTER shell command with an explicit 'failed_when' guarding real failure "
                "conditions) rather than reproducing the CIS script's own control flow line-for-line "
                "— the goal is the SAME end state, not a byte-identical script. CONCRETE TEMPLATE for "
                "the extremely common 'find and comment out conflicting sysctl entries across "
                "candidate files' pattern (sysctl parameter families) — use THIS shape instead of the "
                "bash one-liner. DELIBERATELY A SINGLE TASK (not find+register+loop) — confirmed live "
                "that the 2-task version was frequently only PARTIALLY generated (the find task alone, "
                "with the actual cleanup replace task missing) across many sysctl rules in one "
                "evaluation run; one task is less surface area to omit half of:\n"
                "  - name: Comment out conflicting <param> entries across sysctl config files\n"
                "    ansible.builtin.replace:\n"
                "      path: \"{{ item }}\"\n"
                "      regexp: '^\\s*(<param>\\s*=\\s*(?!<compliant_value>\\b).*)$'\n"
                "      replace: '# \\1'\n"
                "    with_fileglob:\n"
                "      - /etc/sysctl.d/*.conf\n"
                "      - /run/sysctl.d/*.conf\n"
                "      - /usr/lib/sysctl.d/*.conf\n"
                "      - /usr/local/lib/sysctl.d/*.conf\n"
                "    failed_when: false\n"
                "This ONE task IS the complete cleanup step — do not split it into a separate find + "
                "loop; 'with_fileglob' already enumerates matching files directly, no 'register'/"
                "'loop' over a discovery task needed. Follow it with your OWN separate task that sets "
                "the compliant value in the authoritative drop-in file — that is TWO tasks total for "
                "the whole rule (cleanup + set), not three. "
                "(replace <param>/<compliant_value> with the actual sysctl key/value for this rule; "
                "'replace' module's 'regexp'/negative-lookahead already skips lines that are ALREADY "
                "correct, so this is naturally idempotent) — this achieves the SAME effect as CIS's "
                "example script, is idempotent, uses dedicated modules (better 'automation' score), "
                "and is readable (better 'maintainability' score), which the raw bash one-liner is not. "
                "CRITICAL — the cleanup/find-and-comment-out step is MANDATORY whenever the CIS text "
                "requires it, REGARDLESS of which form you use for it: confirmed live that switching "
                "to the Ansible-native template sometimes caused the step to be DROPPED ENTIRELY "
                "instead of rewritten (rule 3.3.1.6's fix skipped it completely) — that is WORSE than "
                "keeping the original bash script, since a missing step is a genuine compliance gap, "
                "not just a style issue. If you are not fully confident the Ansible-native template "
                "correctly covers this specific rule's cleanup requirement, INCLUDE IT via the "
                "shorter shell-command form instead (with explicit failed_when) rather than omitting "
                "it — completeness always outranks style; never trade away a required step for "
                "cleaner code. "
                "S. When the CIS Remediation text explicitly NAMES SEVERAL target files that must "
                "all be kept consistent/updated together (e.g. '/etc/pam.d/password-auth AND /etc/"
                "pam.d/system-auth', or 'both system-auth and password-auth'), the fix (or its "
                "sibling fixes together) must cover EVERY named file, not just one — a fix that only "
                "targets one of two explicitly-named files is incomplete even if it correctly handles "
                "the one it does target. Confirmed live: rule 5.3.2.4.3's fix updated only "
                "password-auth, leaving system-auth untouched, despite the CIS text explicitly naming "
                "both. "
                "T. When the CIS Remediation text says a setting is acceptable as 'X OR Y' (e.g. "
                "'sha512 or yescrypt' are BOTH acceptable strong values), a regex/fix meant to correct "
                "a WEAK/wrong value must NOT also match and overwrite the OTHER already-acceptable "
                "alternative — a pattern like matching '(sha512|yescrypt|md5|...)' to replace "
                "everything with 'sha512' would incorrectly downgrade/rewrite an already-compliant "
                "'yescrypt' setting. Build the match pattern from ONLY the values that are actually "
                "non-compliant (exclude every value the CIS text itself lists as already acceptable). "
                "Confirmed live: rule 5.3.2.4.3's fix regex included 'yescrypt' among the values to "
                "replace with 'sha512', which would silently overwrite a compliant setting. "
                "U. When the CIS Remediation text gives MULTIPLE explicit steps — e.g. 'add "
                "<directive>' AND SEPARATELY 'remove any occurrence of <negated-directive>' (a "
                "negated form like '!use_pty' that would otherwise CANCEL the positive setting) — "
                "implement BOTH steps, not just the ADD step. Adding the positive directive while "
                "leaving a negated form of the SAME directive in place can leave the setting "
                "ineffective (later occurrences frequently override earlier ones in config files "
                "like sudoers). Confirmed live: rule 5.2.2's fix added 'Defaults use_pty' but never "
                "searched for/removed '!use_pty', despite the CIS text explicitly listing removal of "
                "the negated form as its own separate step. "
                "V. When a fix must assign a NEW UNIQUE value (UID/GID) to EACH of potentially "
                "SEVERAL accounts/groups sharing a duplicate identifier, computing ONE new value "
                "(e.g. 'max existing UID + 1') ONCE and then applying that SAME value to every "
                "detected account in a loop is WRONG — if more than one account shares the "
                "duplicate, they would all receive the SAME new value, creating a fresh duplicate "
                "instead of fixing one. Each account in the loop needs its OWN unique value — e.g. "
                "increment a counter per iteration, or recompute the 'next free UID' after each "
                "assignment (so the next iteration sees the previous one's change). Confirmed live: "
                "rule 5.4.2.1's fix computed a single 'next_uid' before the loop and reused it for "
                "every non-root UID-0 account found. "
                "W. Do NOT add 'ignore_errors: yes' to a REMEDIATION task (one that changes state — "
                "installs/removes a package, stops/masks a service, edits a file) just as a generic "
                "safety net — a reviewer running the same code-quality analysis on this pipeline's "
                "own output confirmed live that this scores 1-2/5 on error handling, because it masks "
                "genuine failures (a package that doesn't exist, a permission error) as if they never "
                "happened, with no visible signal anything went wrong. Only use 'ignore_errors: yes' "
                "when the task's failure mode is SPECIFICALLY expected and SAFE to ignore (e.g. "
                "removing something that may already be absent) — and prefer a targeted 'failed_when' "
                "condition scoped to the ACTUAL safe-to-ignore case over a blanket 'ignore_errors: "
                "yes' that also swallows unrelated, real failures. If you genuinely need to tolerate "
                "one specific failure mode, register the result and add an explicit 'failed_when' "
                "that only treats THAT case as non-fatal, rather than ignoring everything. "
                "X. When the CIS remediation text lists MULTIPLE, SEQUENTIAL steps (e.g. 'systemctl "
                "stop X.service' THEN 'dnf remove X'), your fix must perform ALL of them, not just "
                "the last/most-visible one — removing a package does NOT, by itself, guarantee an "
                "already-running instance of its service was stopped first (a systemd service can "
                "keep running with removed binaries until the next stop/reboot). This is a recurring "
                "pattern across every 'service not in use' rule in this benchmark family "
                "(stop-then-remove, and often mask-the-socket-too as an alternative branch when the "
                "package has dependents) — include the stop step (and mask, if the text offers that "
                "branch) BEFORE/ALONGSIDE the removal, not removal alone. CONCRETE TEMPLATE — use "
                "THIS 2-task shape (stop task ALWAYS listed first, package name ends up unused by the "
                "stop task itself so ordering, not a 'when' condition, is what guarantees it runs "
                "first):\n"
                "  - name: Stop <service> before removal\n"
                "    ansible.builtin.systemd:\n"
                "      name: \"{{ item }}\"\n"
                "      state: stopped\n"
                "    loop:\n"
                "      - <service>.socket\n"
                "      - <service>.service\n"
                "    failed_when: false\n"
                "  - name: Remove <package>\n"
                "    ansible.builtin.dnf:\n"
                "      name: <package>\n"
                "      state: absent\n"
                "(replace <service>/<package> with the actual unit/package names from the CIS text; "
                "'failed_when: false' on the stop task tolerates a unit that was never installed/"
                "running — do NOT omit the stop task's loop over BOTH the .socket and .service units "
                "when the CIS text names both, and do NOT reorder the tasks so removal comes first). "
                "Y. For auditd rules (auditctl lines / /etc/audit/rules.d/*.rules entries), copy "
                "EVERY '-F'/'-S' field shown in the retrieved CIS example line EXACTLY, field by "
                "field — do not drop 'arch=b64', '-S all', or any other field because the rule 'looks "
                "similar' to a simpler pattern you've seen; a generated rule missing a field the CIS "
                "example includes is incomplete even if it otherwise resembles the right shape. Also "
                "include any 'augenrules --load' (or equivalent load/merge) step the remediation text "
                "specifies to actually activate the new rule, not just write it to a .rules file. "
                "Z. For GDM/dconf lockdown rules (the 1.8.x family — banner, disable-user-list, "
                "screen-lock idle/lock-delay, automount, autorun-never, etc.): these rules are "
                "STRUCTURALLY nearly identical (same profile/directory/lock-file/keyfile shape) but "
                "each targets a DIFFERENT gsettings schema path and key (e.g. 'org/gnome/desktop/"
                "media-handling'/'autorun-never' vs 'org/gnome/desktop/session'/'idle-delay' vs "
                "'org/gnome/login-screen'/'banner-message-enable') — do NOT pattern-match the schema "
                "path/key from a similar-looking sibling rule's structure; read the EXACT schema path "
                "and key from the 'CIS Rule: <id> - <title>' header and the retrieved remediation "
                "text for THIS rule specifically, every single time, even when the surrounding YAML "
                "shape is nearly copy-paste identical to a neighboring rule you may recall. "
                "AA. A code-quality reviewer running the same analysis on this pipeline's own output "
                "confirmed live that using 'shell:'/'command:'/'script:' for logic a DEDICATED module "
                "already handles scores 1-2/5 on 'automation' — before writing a shell task, check "
                "whether one of these covers your case: enumerate/list files matching criteria -> "
                "'ansible.builtin.find' (not 'ls'/'find' in shell); read a single file's owner/mode/"
                "size -> 'ansible.builtin.stat' (not 'ls -l'/'stat' in shell); load/reload a kernel "
                "module or sysctl value -> 'ansible.builtin.modprobe' / 'ansible.posix.sysctl' (not "
                "'modprobe'/'sysctl' via shell); manage SELinux enforcing mode -> "
                "'ansible.posix.selinux' (not 'setenforce' via shell); inspect/lock/expire a user "
                "account (shell, password age, lock state) -> 'ansible.builtin.user' (not raw "
                "'chage'/'usermod' parsing via shell); set file owner/group/mode -> 'ansible.builtin."
                "file' with a loop (not 'chmod'/'chown' via shell or find -exec). Reach for 'shell:' "
                "only when the logic genuinely has no dedicated-module equivalent (e.g. a multi-stage "
                "pipeline of greps/awk with no single-module analogue) — and even then, prefer the "
                "template in fact R above over a raw one-off script. "
                "BB. CRITICAL YAML SYNTAX RULE — ALWAYS use SINGLE quotes for any 'regexp:'/'replace:' "
                "value that contains a backslash (\\s, \\b, \\., \\d, etc.): a YAML DOUBLE-quoted "
                "string interprets backslash sequences as ESCAPE CODES, and \\s/\\b/\\. are NOT valid "
                "YAML escapes — this is a guaranteed parse error ('found unknown escape character'), "
                "not a style preference. Confirmed live: this exact mistake ('regexp: \"^net\\.ipv4"
                "\\.conf...\"' — double-quoted, with literal dots escaped) wasted a full attempt on "
                "rule 3.3.1.2 before the next attempt correctly switched to single quotes. Rule of "
                "thumb: 'regexp:'/'replace:' values are regex — use single quotes for them, always; "
                "reserve double quotes for plain strings needing Jinja interpolation ('{{ var }}') or "
                "an embedded literal single-quote. When a 'replace:' string does NOT reference a "
                "capture group, never include '\\1' in it (it would be empty/undefined) — only use "
                "'\\1' (or '\\0') when the 'regexp:' pattern actually has a matching '(...)' group. "
                "CC. Standard Unix exit-code convention: 'grep -q PATTERN file' (or similar) exits 0 "
                "when the PATTERN WAS FOUND (the checked condition IS present/already true), non-zero "
                "when NOT found. When your fix registers such a check and gates a LATER task with "
                "'when: <var>.rc == 0' to decide whether to APPLY a fix for something MISSING, verify "
                "the DIRECTION before writing it: a task that ENABLES/ADDS/FIXES a missing feature "
                "must gate on 'rc != 0' (not found -> needs fixing); a task that REMOVES/DISABLES a "
                "present, unwanted setting must gate on 'rc == 0' (found -> needs removing). Read your "
                "OWN grep pattern and your OWN task's stated purpose together before writing the "
                "condition — this exact inversion (gating an ENABLE step on 'rc == 0' after grepping "
                "FOR the feature, so it only fires when the feature is already present) has been "
                "confirmed live, repeatedly, across multiple generation attempts for rule 5.3.1.3 "
                "even after being told the specific error — write the condition deliberately, don't "
                "copy the shape of a nearby check without re-verifying which direction it needs. "
                "STRICT FORMAT RULES: "
                "1. Start with dash name and include the CIS rule ID in the task name. "
                "2. NEVER include handlers section inside a task - it is invalid Ansible syntax. "
                "3. Use become true for tasks requiring root privileges. "
                "4. Output ONLY valid YAML, no explanation, no markdown code blocks."
            )
        },
        {
            "role": "user",
            "content": f"""Fix this Ansible task to ensure CIS compliance.
Remember: use ONLY what is explicitly stated below, do not invent additional values.

{rule_ids_header}
CIS Remediation Instructions:
{rem_text[:4000]}

AUDITOR'S FINDING (why this SPECIFIC task was flagged {result.status} — this is
frequently about a PROJECT-SPECIFIC bug the generic CIS text above cannot describe,
e.g. broken 'when'/toggle-variable logic, a wrong loop, a typo — read it carefully
and fix THIS concrete problem, not just re-derive generic compliance from the CIS
text alone):
{result.explanation or "(no specific explanation recorded — rely on the CIS text and RAW YAML below)"}
{extra_note}
{sibling_block}
{handlers_block}
Original non-compliant task, RAW YAML (exactly as it appears in the project,
including task-level keys like loop/with_items, failed_when, changed_when,
register, when, ignore_errors — this is the ONLY representation given, so
read the whole task below, not just its module call):
{task.raw_yaml}
{_resolved_params_note(task)}
{_template_content_note(task)}
CRITICAL: your fix MUST target the exact same file path / resource shown in the
RAW YAML above (e.g. if the original task has path=/tmp, your fix must also
be about /tmp - NOT a different, similarly-named path like /var/tmp, even if
the CIS text nearby discusses a similar rule for a different path). Do not
substitute a different target based on pattern-matching against training data.

2. If the RAW YAML above uses a loop ('loop:'/'with_items:') or Jinja variables
(e.g., '{{{{ item }}}}'), your fix MUST use the EXACT loop key, variable name,
and list contents shown in the raw YAML — do NOT guess or invent a variable
name, and do NOT hardcode example values from the CIS text instead of the
real loop.{"" if len(result.rule_ids) <= 1 else f'''
3. This ONE task must satisfy ALL {len(result.rule_ids)} CIS rules listed above
({', '.join(result.rule_ids)}) at once — do NOT narrow the fix down to only one
of them. Every rule_* toggle variable and every option value belonging to any
of these {len(result.rule_ids)} rules must remain covered in your fix — unless
the AUDITOR'S FINDING above specifically says one of them (e.g. a rule_*
variable used twice, or one missing from a condition) is itself the bug, in
which case FIX that specific defect rather than copying it forward
unchanged.'''}
Generate ONLY a valid Ansible YAML task starting with dash name. No markdown, no explanation."""
        }
    ]
    return _call_llm(messages, identifier=f"FIX {rule_id} | {task.task_name}")


# ── Mode 2: Creating a brand-new task (LLM) ───────────────────────────────────
def _llm_create(rule_id: str, rule_title: str, rem_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior Ansible hardening engineer. Your job is to create a new Ansible task "
                "from scratch based on CIS Benchmark remediation instructions, STRICTLY GROUNDED in the "
                "provided text. "
                "ANTI-HALLUCINATION RULES (critical): "
                "A. Use ONLY concrete values (module names, file paths, parameters) EXPLICITLY written in "
                "the provided CIS text below. NEVER invent, extend, or complete a list beyond what is "
                "literally stated in the source text, even if it seems incomplete. "
                "B. If the CIS text describes a DYNAMIC discovery procedure (e.g. run this script to "
                "generate a list of currently loaded or mounted items) rather than a FIXED list, your task "
                "MUST implement that dynamic discovery using shell or command module running the audit "
                "logic and registering output, NOT a hardcoded static list you invented. "
                "C. If the CIS text gives only a FEW illustrative examples rather than an exhaustive list, "
                "do NOT treat those examples as complete - implement the discovery logic shown in the "
                "Audit section instead, or use a loop variable placeholder the engineer must populate. "
                "D. If unsure whether a list is exhaustive or illustrative, prefer the SAFER, SHORTER "
                "interpretation and add a YAML comment noting the list should be reviewed by the engineer. "
                "GENERAL ANSIBLE BEST PRACTICES (apply these on every task, not just the CIS-specific "
                "cases below): (1) Prefer a dedicated Ansible module (mount, lineinfile, systemd, "
                "sysctl, ...) over shell/command whenever one exists — dedicated modules are idempotent "
                "and safer. (2) NEVER guess a module's parameter name or a fact's field name by "
                "analogy or because it 'sounds right' — a fact's field name and a module's parameter "
                "for the same underlying data are often NOT spelled the same. (3) Every task must be "
                "idempotent (safe to re-run without side effects or duplicated entries) — use "
                "unique/select filters and precise 'when' guards rather than blindly re-applying. "
                "(4) Prefer built-in Ansible facts (ansible_facts, e.g. ansible_mounts, "
                "ansible_distribution) over parsing shell command output when the same information is "
                "already available as a fact. (5) Prefer safe, well-known Jinja2 filters/tests (split, "
                "join, unique, selectattr, regex_search) over a custom regex_replace() for string "
                "manipulation — regex is powerful but easy to get subtly wrong. "
                "E. To add a value to an EXISTING comma-separated string (e.g. appending a mount option "
                "to ansible_mounts' 'options' field) use safe list operations: "
                "\"{{ (item.options.split(',') + ['newvalue']) | unique | join(',') }}\" — NEVER a custom "
                "regex_replace() for this. A zero-width regex like '(?<!^)(?<!,)' matches at EVERY "
                "character boundary, not just where you intend — confirmed live to turn "
                "'rw,relatime,seclabel' into 'r,w,,r,e,l,a,t,i,m,e,,s,e,c,l,a,b,e,l,' (a comma inserted "
                "between every letter), silently corrupting the fstab entry. That 'unique' filter "
                "already makes the append idempotent by itself — do NOT also add an extra 'when' "
                "condition (e.g. a regex_search checking whether the value is already present) to "
                "'guard' against re-adding it; such conditions are unnecessary and easy to get "
                "subtly wrong (confirmed live: a regex like '(?<!^)(?<!,)(?<!,nodev)(?<!nodev,)nodev' "
                "searching a comma-separated options string for 'nodev' NEVER matches ANYTHING, "
                "because 'nodev' in such a string is always either at the very start or right after "
                "a comma, and both are excluded by the lookbehinds — making the 'when' condition "
                "silently, permanently True, a dead check that looks like a safeguard but does "
                "nothing). "
                "F. When looping over 'ansible_mounts | selectattr(...)' (the standard fact for mount "
                "info), 'item' has ONLY these fields: mount, device, fstype, options, uuid, size_total, "
                "size_available — there is NO 'src' field. Use 'item.device' for the mount module's "
                "'src' PARAMETER, never 'item.src' (confirmed live: a fix used 'item.src', which does "
                "not exist on ansible_mounts and silently evaluates to Undefined at runtime). "
                "G. When using the mount module to CHANGE OPTIONS on an ALREADY-MOUNTED filesystem "
                "(e.g. adding nodev/nosuid/noexec), use state: mounted, NEVER state: present. "
                "'present' only writes /etc/fstab — the change takes effect only after a reboot. CIS "
                "Audit checks (e.g. 'findmnt | grep -v nodev') read the LIVE mount table, not fstab, so "
                "a 'present'-only fix leaves the system NON_COMPLIANT until reboot even though fstab is "
                "correct — the CIS Remediation text itself confirms this by requiring BOTH steps ('edit "
                "/etc/fstab' AND 'run mount -o remount'); 'state: mounted' does both in one task. "
                "The mount MODULE's parameter for the options string is 'opts' — NEVER 'options' "
                "(that is the ansible_mounts FACT's key name for the same data, an easy mix-up; "
                "confirmed live: 'options' fails ansible-lint with 'Unsupported parameters'). "
                "I. NEVER use a Jinja lookup/query() (e.g. query('shell', ...), lookup('shell', ...), "
                "query('pipe', ...)) inside a 'when:' condition to check REMOTE host state — lookups "
                "execute on the ANSIBLE CONTROL NODE, not the managed host, so the result would reflect "
                "the WRONG machine entirely, regardless of which lookup plugin name is used. There is "
                "also no built-in 'shell' lookup plugin at all (confirmed: the real ones are 'pipe', "
                "'file', 'env', 'password', 'lines', 'ini' — 'shell' does not exist and would fail at "
                "runtime with a lookup-plugin-not-found error). To check remote command output before "
                "deciding whether to run something, use a SEPARATE preceding task with 'register:', then "
                "reference that registered variable's .stdout/.rc in this task's own 'when:' — never an "
                "inline lookup/query(). "
                "J. GNOME dconf LOCK files (/etc/dconf/db/*.d/locks/*, used to prevent a setting from "
                "being overridden) use a DIFFERENT, SIMPLER format than dconf VALUE files: each line is "
                "JUST the full GSettings key path starting with '/', one per line — e.g. "
                "'/org/gnome/desktop/session/idle-delay' — NO '[section]' header and NO 'lock' keyword "
                "(confirmed via official GNOME documentation: help.gnome.org/admin/system-admin-guide, "
                "'Lock down specific settings'). The '[section]' + 'key = value' format is ONLY for dconf "
                "VALUE files (e.g. /etc/dconf/db/*.d/*, setting the actual default) — do NOT mix the two "
                "formats or invent a 'lock <key>' directive, which does not exist. Unless the CIS "
                "remediation text EXPLICITLY names a different database (e.g. 'gdm.d' specifically "
                "for GDM login-screen-only settings), the standard, machine-wide dconf database is "
                "'local.d' — write to /etc/dconf/db/local.d/ and /etc/dconf/db/local.d/locks/, NOT "
                "/etc/dconf/db/gdm.d/, even for a rule whose title mentions 'GDM' (many GDM-related "
                "CIS rules, e.g. screen-lock and idle-delay settings, are enforced via the general "
                "'local.d' database, not a GDM-specific one) — verify against the exact path the "
                "retrieved CIS remediation text itself gives, never assume 'gdm.d' from the rule's "
                "title alone. "
                "K. Ansible file-module 'mode:' accepts SYMBOLIC notation (u/g/o/a + +/-/= + rwx), "
                "e.g. 'go-rwx' means 'remove read/write/execute for Group and Other' — this TIGHTENS "
                "permissions, it is NOT the octal value 0777 and does not grant broad access. Never "
                "read a symbolic mode string as if its letters were octal digits, and never 'fix' a "
                "correctly-tightened symbolic mode by rewriting the task's actual target/content. This "
                "ALSO applies to multi-clause mode strings, e.g. 'o-x,g-wx,o-rwx' — every '-' clause "
                "still REMOVES no matter how many clauses there are or whether one re-targets the same "
                "letter as another; a '-' clause never adds a permission back. "
                "L. The 'ansible.builtin.replace'/'lineinfile'/'blockinfile' modules' 'path:' "
                "parameter takes EXACTLY ONE file path — it does NOT accept a glob pattern like "
                "'/etc/rsyslog.d/*.conf'. To apply the same edit across a directory of files, use "
                "a preceding task with 'ansible.builtin.find' (or 'with_fileglob') to enumerate the "
                "real files first, then loop the edit task over that discovered list. A loop "
                "variable only has dict-style attributes ('item.somekey') when its SOURCE is a "
                "list of dicts — a loop built from 'stdout_lines' or a 'regex_replace' filter is a "
                "plain string list with NO such attributes; referencing 'item.something' on it is "
                "Undefined and fails at runtime. "
                "N. For an AUDIT/CHECK-style task (verifying state, not a remediation command): "
                "'ignore_errors: yes' and NO 'register:' (or one never checked by a "
                "'failed_when:'/'assert:') does not actually verify anything — never generate a "
                "'compliance check' this way; always pair a check with a 'register:' + 'failed_when:' "
                "(or 'assert:') so its result is actually evaluated. This does NOT apply to a "
                "state-changing remediation command (dnf/package, a mutating firewall-cmd/systemctl "
                "call) merely swallowing its own failure — that is a separate, minor robustness point. "
                "This concern applies ONLY when failure-suppression is ACTUALLY PRESENT (ignore_errors: "
                "yes / failed_when: false) — a plain task with normal failure semantics needs no "
                "separate verification step. "
                "M. The 'ansible.posix.sysctl' module's 'state:' parameter accepts ONLY 'present' "
                "or 'absent' — there is NO 'immediate'/'applied'/'reloaded' value. The module "
                "ALREADY applies the value immediately via 'sysctl -w' by default — 'state: "
                "present' (or omitting 'state') is already the correct, complete fix; never invent "
                "a state value to express 'apply now'. "
                "O. A CIS rule about DUPLICATE identifiers (duplicate UIDs, GIDs, user names, or "
                "group names) requires ESTABLISHING A NEW, UNIQUE value for the duplicate — NEVER "
                "'state: absent' / deleting the account or group (destructive, confirmed live rule "
                "7.2.6: remediation text says 'establish unique user names... file ownerships will "
                "automatically reflect the change', a rename, not a deletion). BUT ALSO: the CIS "
                "text itself typically phrases this as a MANUAL review step ('based on the results "
                "of the audit script, establish unique...') — there is USUALLY no safe, generic way "
                "to auto-compute what the new unique value should be without knowing which specific "
                "GID/UID a set of files is actually supposed to belong to. Do NOT invent a variable "
                "(e.g. a 'duplicate_x_mapping' fact) that no other task in the project actually "
                "defines or populates — a task referencing an undefined/always-empty variable is a "
                "silent no-op, not a fix. When you cannot safely auto-compute the correct new value, "
                "the CORRECT implementation is a task that DETECTS the duplicate and FAILS LOUDLY "
                "with a clear message directing to manual remediation (same pattern as an "
                "'ansible.builtin.fail' with a descriptive msg) — an honest, visible non-fix is "
                "better than a fabricated one that silently does nothing or breaks the system. "
                "P. Never use 'authselect select <profile> ...' to 'fix' a rule about a SPECIFIC PAM "
                "module's presence (e.g. 'ensure pam_unix/pam_pwquality/pam_faillock module is "
                "enabled') — 'authselect select' REPLACES THE ENTIRE ACTIVE PROFILE with a named "
                "one, which can silently drop unrelated, already-correct settings and features the "
                "system currently has enabled; it is not a targeted fix for one module's presence, "
                "and guessing which profile/features to select is fabricating configuration state "
                "with no basis in the task's own context. 'authselect select <profile> --with-X' "
                "additionally only accepts REAL, existing authselect feature flags (e.g. "
                "'--with-faillock', '--with-sudo') — there is no '--with-pam' flag; pam_unix is part "
                "of every baseline profile already, not an optional feature. A CIS remediation "
                "phrased as a verification script (e.g. grep for 'pam_unix.so' inside the profile's "
                "own template files under /etc/authselect/ or /usr/share/authselect/default/) should "
                "be implemented as EXACTLY that check — a shell/command task running the equivalent "
                "grep and failing if not found — never as a profile-replacing 'authselect select'. "
                "Q. The 'CIS Rule: <id> - <title>' header above is the AUTHORITATIVE description of "
                "what is required — trust it over the original (possibly typo'd, garbled, or "
                "inverted) task name shown in the project's own source. Confirmed live: a project "
                "task named 'Ensure nodexec option set on /var/log partition' (a typo — 'nodexec', "
                "not 'noexec') produced a fix that added 'exec' to the mount options, the OPPOSITE "
                "of what CIS rule 1.1.2.6.4 ('Ensure noexec option set...') actually requires — the "
                "task's own garbled name was mistakenly treated as authoritative over the correct "
                "rule title provided above it. Before writing a fix whose effect is the OPPOSITE of "
                "a positive/prohibitive framing in the rule title (set vs. not set, enable vs. "
                "disable, present vs. absent), re-read the CIS Rule title and Remediation text above "
                "and resolve any conflict in ITS favor, not the source project's own (possibly "
                "mistagged) task name. "
                "R. The CIS Remediation text sometimes includes an 'Example Script' — a dense bash "
                "one-liner CIS itself provides as ONE way to implement the step (e.g. finding and "
                "commenting out conflicting sysctl entries across multiple candidate files). Do NOT "
                "copy that script verbatim into a 'shell:' task — a code-quality reviewer running "
                "the SAME analysis on this pipeline's own output confirmed live (24 sysctl rules, "
                "3.3.1.x/3.3.2.x family) that doing so scores 1-2/5 on readability/maintainability "
                "(obscure regex, dense variable manipulation) AND 1-2/5 on error handling (no real "
                "failure detection — 'changed_when: result.rc == 0' does NOT stop the play on "
                "failure the way 'failed_when' does, it only affects the changed/ok label). Extract "
                "the INTENT from the example script (e.g. 'find files matching X pattern, comment "
                "out lines matching Y') and implement it with Ansible-native constructs where "
                "practical (a loop over a small, explicit file list with 'lineinfile'/'replace', or "
                "a SHORTER shell command with an explicit 'failed_when' guarding real failure "
                "conditions) rather than reproducing the CIS script's own control flow line-for-line "
                "— the goal is the SAME end state, not a byte-identical script. CONCRETE TEMPLATE for "
                "the extremely common 'find and comment out conflicting sysctl entries across "
                "candidate files' pattern (sysctl parameter families) — use THIS shape instead of the "
                "bash one-liner. DELIBERATELY A SINGLE TASK (not find+register+loop) — confirmed live "
                "that the 2-task version was frequently only PARTIALLY generated (the find task alone, "
                "with the actual cleanup replace task missing) across many sysctl rules in one "
                "evaluation run; one task is less surface area to omit half of:\n"
                "  - name: Comment out conflicting <param> entries across sysctl config files\n"
                "    ansible.builtin.replace:\n"
                "      path: \"{{ item }}\"\n"
                "      regexp: '^\\s*(<param>\\s*=\\s*(?!<compliant_value>\\b).*)$'\n"
                "      replace: '# \\1'\n"
                "    with_fileglob:\n"
                "      - /etc/sysctl.d/*.conf\n"
                "      - /run/sysctl.d/*.conf\n"
                "      - /usr/lib/sysctl.d/*.conf\n"
                "      - /usr/local/lib/sysctl.d/*.conf\n"
                "    failed_when: false\n"
                "This ONE task IS the complete cleanup step — do not split it into a separate find + "
                "loop; 'with_fileglob' already enumerates matching files directly, no 'register'/"
                "'loop' over a discovery task needed. Follow it with your OWN separate task that sets "
                "the compliant value in the authoritative drop-in file — that is TWO tasks total for "
                "the whole rule (cleanup + set), not three. "
                "(replace <param>/<compliant_value> with the actual sysctl key/value for this rule; "
                "'replace' module's 'regexp'/negative-lookahead already skips lines that are ALREADY "
                "correct, so this is naturally idempotent) — this achieves the SAME effect as CIS's "
                "example script, is idempotent, uses dedicated modules (better 'automation' score), "
                "and is readable (better 'maintainability' score), which the raw bash one-liner is not. "
                "CRITICAL — the cleanup/find-and-comment-out step is MANDATORY whenever the CIS text "
                "requires it, REGARDLESS of which form you use for it: confirmed live that switching "
                "to the Ansible-native template sometimes caused the step to be DROPPED ENTIRELY "
                "instead of rewritten (rule 3.3.1.6's fix skipped it completely) — that is WORSE than "
                "keeping the original bash script, since a missing step is a genuine compliance gap, "
                "not just a style issue. If you are not fully confident the Ansible-native template "
                "correctly covers this specific rule's cleanup requirement, INCLUDE IT via the "
                "shorter shell-command form instead (with explicit failed_when) rather than omitting "
                "it — completeness always outranks style; never trade away a required step for "
                "cleaner code. "
                "S. When the CIS Remediation text explicitly NAMES SEVERAL target files that must "
                "all be kept consistent/updated together (e.g. '/etc/pam.d/password-auth AND /etc/"
                "pam.d/system-auth', or 'both system-auth and password-auth'), the fix (or its "
                "sibling fixes together) must cover EVERY named file, not just one — a fix that only "
                "targets one of two explicitly-named files is incomplete even if it correctly handles "
                "the one it does target. Confirmed live: rule 5.3.2.4.3's fix updated only "
                "password-auth, leaving system-auth untouched, despite the CIS text explicitly naming "
                "both. "
                "T. When the CIS Remediation text says a setting is acceptable as 'X OR Y' (e.g. "
                "'sha512 or yescrypt' are BOTH acceptable strong values), a regex/fix meant to correct "
                "a WEAK/wrong value must NOT also match and overwrite the OTHER already-acceptable "
                "alternative — a pattern like matching '(sha512|yescrypt|md5|...)' to replace "
                "everything with 'sha512' would incorrectly downgrade/rewrite an already-compliant "
                "'yescrypt' setting. Build the match pattern from ONLY the values that are actually "
                "non-compliant (exclude every value the CIS text itself lists as already acceptable). "
                "Confirmed live: rule 5.3.2.4.3's fix regex included 'yescrypt' among the values to "
                "replace with 'sha512', which would silently overwrite a compliant setting. "
                "U. When the CIS Remediation text gives MULTIPLE explicit steps — e.g. 'add "
                "<directive>' AND SEPARATELY 'remove any occurrence of <negated-directive>' (a "
                "negated form like '!use_pty' that would otherwise CANCEL the positive setting) — "
                "implement BOTH steps, not just the ADD step. Adding the positive directive while "
                "leaving a negated form of the SAME directive in place can leave the setting "
                "ineffective (later occurrences frequently override earlier ones in config files "
                "like sudoers). Confirmed live: rule 5.2.2's fix added 'Defaults use_pty' but never "
                "searched for/removed '!use_pty', despite the CIS text explicitly listing removal of "
                "the negated form as its own separate step. "
                "V. When a fix must assign a NEW UNIQUE value (UID/GID) to EACH of potentially "
                "SEVERAL accounts/groups sharing a duplicate identifier, computing ONE new value "
                "(e.g. 'max existing UID + 1') ONCE and then applying that SAME value to every "
                "detected account in a loop is WRONG — if more than one account shares the "
                "duplicate, they would all receive the SAME new value, creating a fresh duplicate "
                "instead of fixing one. Each account in the loop needs its OWN unique value — e.g. "
                "increment a counter per iteration, or recompute the 'next free UID' after each "
                "assignment (so the next iteration sees the previous one's change). Confirmed live: "
                "rule 5.4.2.1's fix computed a single 'next_uid' before the loop and reused it for "
                "every non-root UID-0 account found. "
                "W. Do NOT add 'ignore_errors: yes' to a REMEDIATION task (one that changes state — "
                "installs/removes a package, stops/masks a service, edits a file) just as a generic "
                "safety net — a reviewer running the same code-quality analysis on this pipeline's "
                "own output confirmed live that this scores 1-2/5 on error handling, because it masks "
                "genuine failures (a package that doesn't exist, a permission error) as if they never "
                "happened, with no visible signal anything went wrong. Only use 'ignore_errors: yes' "
                "when the task's failure mode is SPECIFICALLY expected and SAFE to ignore (e.g. "
                "removing something that may already be absent) — and prefer a targeted 'failed_when' "
                "condition scoped to the ACTUAL safe-to-ignore case over a blanket 'ignore_errors: "
                "yes' that also swallows unrelated, real failures. If you genuinely need to tolerate "
                "one specific failure mode, register the result and add an explicit 'failed_when' "
                "that only treats THAT case as non-fatal, rather than ignoring everything. "
                "X. When the CIS remediation text lists MULTIPLE, SEQUENTIAL steps (e.g. 'systemctl "
                "stop X.service' THEN 'dnf remove X'), your fix must perform ALL of them, not just "
                "the last/most-visible one — removing a package does NOT, by itself, guarantee an "
                "already-running instance of its service was stopped first (a systemd service can "
                "keep running with removed binaries until the next stop/reboot). This is a recurring "
                "pattern across every 'service not in use' rule in this benchmark family "
                "(stop-then-remove, and often mask-the-socket-too as an alternative branch when the "
                "package has dependents) — include the stop step (and mask, if the text offers that "
                "branch) BEFORE/ALONGSIDE the removal, not removal alone. CONCRETE TEMPLATE — use "
                "THIS 2-task shape (stop task ALWAYS listed first, package name ends up unused by the "
                "stop task itself so ordering, not a 'when' condition, is what guarantees it runs "
                "first):\n"
                "  - name: Stop <service> before removal\n"
                "    ansible.builtin.systemd:\n"
                "      name: \"{{ item }}\"\n"
                "      state: stopped\n"
                "    loop:\n"
                "      - <service>.socket\n"
                "      - <service>.service\n"
                "    failed_when: false\n"
                "  - name: Remove <package>\n"
                "    ansible.builtin.dnf:\n"
                "      name: <package>\n"
                "      state: absent\n"
                "(replace <service>/<package> with the actual unit/package names from the CIS text; "
                "'failed_when: false' on the stop task tolerates a unit that was never installed/"
                "running — do NOT omit the stop task's loop over BOTH the .socket and .service units "
                "when the CIS text names both, and do NOT reorder the tasks so removal comes first). "
                "Y. For auditd rules (auditctl lines / /etc/audit/rules.d/*.rules entries), copy "
                "EVERY '-F'/'-S' field shown in the retrieved CIS example line EXACTLY, field by "
                "field — do not drop 'arch=b64', '-S all', or any other field because the rule 'looks "
                "similar' to a simpler pattern you've seen; a generated rule missing a field the CIS "
                "example includes is incomplete even if it otherwise resembles the right shape. Also "
                "include any 'augenrules --load' (or equivalent load/merge) step the remediation text "
                "specifies to actually activate the new rule, not just write it to a .rules file. "
                "Z. For GDM/dconf lockdown rules (the 1.8.x family — banner, disable-user-list, "
                "screen-lock idle/lock-delay, automount, autorun-never, etc.): these rules are "
                "STRUCTURALLY nearly identical (same profile/directory/lock-file/keyfile shape) but "
                "each targets a DIFFERENT gsettings schema path and key (e.g. 'org/gnome/desktop/"
                "media-handling'/'autorun-never' vs 'org/gnome/desktop/session'/'idle-delay' vs "
                "'org/gnome/login-screen'/'banner-message-enable') — do NOT pattern-match the schema "
                "path/key from a similar-looking sibling rule's structure; read the EXACT schema path "
                "and key from the 'CIS Rule: <id> - <title>' header and the retrieved remediation "
                "text for THIS rule specifically, every single time, even when the surrounding YAML "
                "shape is nearly copy-paste identical to a neighboring rule you may recall. "
                "AA. A code-quality reviewer running the same analysis on this pipeline's own output "
                "confirmed live that using 'shell:'/'command:'/'script:' for logic a DEDICATED module "
                "already handles scores 1-2/5 on 'automation' — before writing a shell task, check "
                "whether one of these covers your case: enumerate/list files matching criteria -> "
                "'ansible.builtin.find' (not 'ls'/'find' in shell); read a single file's owner/mode/"
                "size -> 'ansible.builtin.stat' (not 'ls -l'/'stat' in shell); load/reload a kernel "
                "module or sysctl value -> 'ansible.builtin.modprobe' / 'ansible.posix.sysctl' (not "
                "'modprobe'/'sysctl' via shell); manage SELinux enforcing mode -> "
                "'ansible.posix.selinux' (not 'setenforce' via shell); inspect/lock/expire a user "
                "account (shell, password age, lock state) -> 'ansible.builtin.user' (not raw "
                "'chage'/'usermod' parsing via shell); set file owner/group/mode -> 'ansible.builtin."
                "file' with a loop (not 'chmod'/'chown' via shell or find -exec). Reach for 'shell:' "
                "only when the logic genuinely has no dedicated-module equivalent (e.g. a multi-stage "
                "pipeline of greps/awk with no single-module analogue) — and even then, prefer the "
                "template in fact R above over a raw one-off script. "
                "BB. CRITICAL YAML SYNTAX RULE — ALWAYS use SINGLE quotes for any 'regexp:'/'replace:' "
                "value that contains a backslash (\\s, \\b, \\., \\d, etc.): a YAML DOUBLE-quoted "
                "string interprets backslash sequences as ESCAPE CODES, and \\s/\\b/\\. are NOT valid "
                "YAML escapes — this is a guaranteed parse error ('found unknown escape character'), "
                "not a style preference. Confirmed live: this exact mistake ('regexp: \"^net\\.ipv4"
                "\\.conf...\"' — double-quoted, with literal dots escaped) wasted a full attempt on "
                "rule 3.3.1.2 before the next attempt correctly switched to single quotes. Rule of "
                "thumb: 'regexp:'/'replace:' values are regex — use single quotes for them, always; "
                "reserve double quotes for plain strings needing Jinja interpolation ('{{ var }}') or "
                "an embedded literal single-quote. When a 'replace:' string does NOT reference a "
                "capture group, never include '\\1' in it (it would be empty/undefined) — only use "
                "'\\1' (or '\\0') when the 'regexp:' pattern actually has a matching '(...)' group. "
                "CC. Standard Unix exit-code convention: 'grep -q PATTERN file' (or similar) exits 0 "
                "when the PATTERN WAS FOUND (the checked condition IS present/already true), non-zero "
                "when NOT found. When your fix registers such a check and gates a LATER task with "
                "'when: <var>.rc == 0' to decide whether to APPLY a fix for something MISSING, verify "
                "the DIRECTION before writing it: a task that ENABLES/ADDS/FIXES a missing feature "
                "must gate on 'rc != 0' (not found -> needs fixing); a task that REMOVES/DISABLES a "
                "present, unwanted setting must gate on 'rc == 0' (found -> needs removing). Read your "
                "OWN grep pattern and your OWN task's stated purpose together before writing the "
                "condition — this exact inversion (gating an ENABLE step on 'rc == 0' after grepping "
                "FOR the feature, so it only fires when the feature is already present) has been "
                "confirmed live, repeatedly, across multiple generation attempts for rule 5.3.1.3 "
                "even after being told the specific error — write the condition deliberately, don't "
                "copy the shape of a nearby check without re-verifying which direction it needs. "
                "STRICT FORMAT RULES: "
                "1. Start with dash name and include the CIS rule ID in the task name. "
                "2. Use standard ansible.builtin modules such as lineinfile, file, systemd. "
                "3. Use become true if modifying system configs. "
                "4. Output ONLY valid YAML, no explanation, no markdown."
            )
        },
        {
            "role": "user",
            "content": f"""Create a brand new Ansible task to satisfy this CIS rule.
Remember: use ONLY what is explicitly stated below, do not invent additional values.

CIS Rule: {rule_id} - {rule_title}
CIS Remediation Instructions:
{rem_text[:2500]}

Generate ONLY a valid Ansible YAML task starting with dash name. No markdown, no explanation."""
        }
    ]
    return _call_llm(messages, identifier=f"CREATE {rule_id}")


# ── Shared API interface for the LLM ──────────────────────────────────────────
def _call_llm(messages: list, identifier: str = "") -> str:
    payload = json.dumps({
        "model": agent_a2_llm.API_MODEL,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        # Token limit large enough for multi-task fixes. Truncated responses
        # are additionally detected via finish_reason below.
        "max_tokens": 2000,
        "temperature": 0
    }).encode()

    t0 = time.time()
    try:
        req = urllib.request.Request(
            agent_a2_llm.API_URL,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {agent_a2_llm.API_TOKEN}",
                "User-Agent":    "curl/7.88.1"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read())
        # Success is recorded once, after text is extracted, so that a
        # malformed response is not counted twice.
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "").strip()
        # A response cut off by the token limit (finish_reason=="length") is
        # treated as a failed generation, because truncated YAML can still be
        # syntactically valid.
        if choice.get("finish_reason") == "length":
            agent_a2_llm._record_request(time.time() - t0, success=True)
            print(f"  [A3] LLM response truncated by max_tokens (finish_reason=length)")
            return "# LLM_CALL_FAILED: response truncated by max_tokens (finish_reason=length)"
        agent_a2_llm._record_request(time.time() - t0, success=True)

        # Strip markdown fencing from the code
        m = re.search(r"```(?:yaml)?\s*(.*?)```", text, re.DOTALL)
        final = m.group(1).strip() if m else text
        dump_llm_call("A3_generate", identifier, messages, text, {"final_yaml": final})
        return final

    except Exception as e:
        agent_a2_llm._record_request(time.time() - t0, success=False)
        print(f"  [A3] LLM error: {e}")
        # On failure, no placeholder task is returned: a valid-looking dummy
        # task could pass validation and be counted as a fix.
        #
        # A comment-only marker parses to an empty YAML document, so Agent 4
        # treats it as a format failure. Newlines are removed so that the
        # marker stays a single comment line and is identical across repeated
        # errors (which lets the stagnation guard stop the loop).
        return "# LLM_CALL_FAILED: " + " ".join(str(e).split())


# ── Heuristic (Fallback Mock) ──────────────────────────────────────────────────
def _mock_fix(task: TaskDescription, result: ComplianceResult, rem_text: str) -> str:
    rule_id    = result.rule_ids[0]    if result.rule_ids    else "?"
    rule_title = result.rule_titles[0] if result.rule_titles else ""
    merged     = task.params.copy()

    task_dict = {
        "name": f"[CIS {rule_id}] {rule_title[:55]}",
        task.module: merged,
        "become": True
    }
    return yaml.dump([task_dict], default_flow_style=False, allow_unicode=True, sort_keys=False).strip()


# ── Main operational functions ─────────────────────────────────────────────────

def _extract_target_path(params: dict) -> str:
    """Extract the main path/resource from the task parameters (path/dest/name),
    or, for shell/command tasks without such a key, the first absolute path
    found in the command string."""
    for key in ("path", "dest", "name"):
        val = params.get(key)
        if val and isinstance(val, str) and ("/" in val):
            return val
    for key in ("_raw_params", "cmd", "argv"):
        val = params.get(key)
        if val and isinstance(val, str):
            m = re.search(r"(/[\w][\w./\-]*)", val)
            if m:
                return m.group(1)
    return ""


def _extract_target_path_from_raw(raw_yaml: str) -> str:
    """Same purpose as _extract_target_path(), but scans the task's raw
    (unresolved) YAML text instead of the Jinja-resolved parameters, so that
    a fix reusing the same '{{ variable }}' path is recognised as correct."""
    for key in ("path", "dest", "name", "src"):
        # The value can be a bare literal ('/etc/foo') OR contain a Jinja
        # expression with internal spaces ('/{{ some_var }}') — a plain
        # \S+ stops at the first space and would truncate the latter to
        # just '/{{'. Alternate between '{{ ... }}' blocks (spaces allowed
        # inside) and plain non-space/non-quote runs.
        m = re.search(rf"{key}:\s*['\"]?((?:\{{\{{.*?\}}\}}|[^\s'\"])+)", raw_yaml)
        if m and "/" in m.group(1):
            return m.group(1)
    return ""


def _path_guardrail_check(yaml_fix: str, original_path: str) -> bool:
    """
    Check whether the original path appears in the generated YAML.
    Returns True if OK (the path is present, or there was nothing to check).

    For a templated path (e.g. '/etc/modprobe.d/{{ item }}.conf') only the
    static segments are compared, with every {{ ... }} expression treated as
    a wildcard.
    """
    if not original_path:
        return True

    def _boundary_match(needle: str, haystack: str) -> bool:
        # Plain substring matching is not enough ('/tmp' is contained in
        # '/var/tmp'), so the character before the match must not extend a
        # path segment.
        #
        # The character after the match is checked as well, so that
        # '/etc/cron.d' does not match '/etc/cron.daily'. A following '/'
        # (a subpath) is accepted.
        for m in re.finditer(re.escape(needle), haystack):
            before_ok = m.start() == 0 or not (haystack[m.start() - 1].isalnum() or haystack[m.start() - 1] in "._-")
            end = m.end()
            # A needle ending in '/' already has its own boundary, so a file
            # inside that directory is accepted.
            after_ok = (end == len(haystack) or needle.endswith("/")
                       or not (haystack[end].isalnum() or haystack[end] in "._-"))
            if before_ok and after_ok:
                return True
        return False

    if _boundary_match(original_path, yaml_fix):
        return True

    # A match on the directory alone is accepted, because some CIS rules
    # require renaming a file within the same directory (e.g. rule 1.6.2,
    # upper-case .pmod file names).
    dir_end = original_path.rfind("/")
    if dir_end > 0:
        original_dir = original_path[:dir_end + 1]  # keep trailing '/'
        if _boundary_match(original_dir, yaml_fix):
            return True

    if "{{" not in original_path:
        return False
    skeleton = re.sub(r"\{\{.*?\}\}", "\x00", original_path)
    fix_skeleton = re.sub(r"\{\{.*?\}\}", "\x00", yaml_fix)
    return _boundary_match(skeleton, fix_skeleton)


# Severity threshold below which Agent 3 does not generate a fix. The default
# (LOW) means that every NON_COMPLIANT/PARTIAL finding gets a fix attempt.
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
DEFAULT_MIN_SEVERITY = "LOW"  # LOW = the lowest level, so effectively we generate a fix
                              # for every NON_COMPLIANT/PARTIAL finding regardless of
                              # severity. The principle "more attention for more serious
                              # findings" itself is realized in Agent 4
                              # (ATTEMPTS_BY_SEVERITY), not by skipping generation for
                              # LOW findings entirely.


_CLEANUP_HINT_RE = re.compile(
    r"comment\s+out|review\s+all\s+files.{0,80}(sysctl|conf)|remove\s+all\s+.{0,40}(lines|entries)",
    re.IGNORECASE | re.DOTALL,
)
_CLEANUP_PRESENT_RE = re.compile(
    r"ansible\.builtin\.replace|comment\s*out|regexp.{0,200}\\1|with_fileglob",
    re.IGNORECASE | re.DOTALL,
)


def _ensure_cleanup_step_present(rule_id: str, rem_text: str, yaml_fix: str) -> str:
    """Second, narrow LLM call for rules whose CIS remediation requires
    commenting out conflicting entries. The first call (_llm_fix()) often
    omits this step. This function checks (by regex on the CIS text) whether
    the step is required and missing from yaml_fix, and only then asks the
    model for that single task and appends it. Never raises: on failure
    yaml_fix is left unchanged."""
    if not _CLEANUP_HINT_RE.search(rem_text or ""):
        return yaml_fix
    if _CLEANUP_PRESENT_RE.search(yaml_fix or ""):
        return yaml_fix

    messages = [
        {"role": "system", "content": (
            "You write EXACTLY ONE additional Ansible task, no more, no less. "
            "The CIS Remediation text below requires a 'find and comment out any "
            "conflicting/non-compliant pre-existing entries' step, which a colleague's "
            "already-generated fix (shown below) is MISSING — it only sets the new value, "
            "it never cleans up old conflicting entries in other candidate config files. "
            "Write ONLY that missing cleanup task. Use this exact shape: a single "
            "'ansible.builtin.replace' task with 'with_fileglob:' listing the plausible "
            "candidate file location(s) for this specific config family (infer them from "
            "the CIS text — e.g. sysctl parameters use /etc/sysctl.d/*.conf, /run/sysctl.d/"
            "*.conf, /usr/lib/sysctl.d/*.conf, /usr/local/lib/sysctl.d/*.conf; other "
            "families use whatever directory the CIS text itself names), a 'regexp:' in "
            "SINGLE quotes matching the non-compliant form of the setting with a negative "
            "lookahead for the compliant value, 'replace: \"# \\\\1\"', and 'failed_when: "
            "false'. Output ONLY that one YAML task (a single '- name:' list item), no "
            "explanation, no markdown, no other tasks."
        )},
        {"role": "user", "content": f"""CIS rule: {rule_id}

CIS REMEDIATION TEXT (source of truth for the cleanup step's target file/parameter):
{rem_text[:2000]}

ALREADY-GENERATED FIX (missing the cleanup step — do not repeat any of this, just write the missing task):
{yaml_fix[:1500]}"""},
    ]
    cleanup_task = _call_llm(messages, identifier=f"CLEANUP {rule_id}")
    if not cleanup_task or cleanup_task.startswith("# LLM_CALL_FAILED"):
        return yaml_fix
    print(f"  [A3] ↳ second-pass cleanup task generated for {rule_id} "
          f"({len(cleanup_task)} chars) — CIS text requires it, first pass omitted it")
    return cleanup_task.strip() + "\n" + yaml_fix


_STOP_HINT_RE = re.compile(
    r"systemctl\s+stop\s+\S+\.(service|socket)|stop\s+(the\s+)?\S+\.(service|socket)",
    re.IGNORECASE,
)
_STOP_PRESENT_RE = re.compile(
    r"state:\s*stopped|systemd:.{0,80}stopped|command:.{0,40}systemctl\s+stop",
    re.IGNORECASE | re.DOTALL,
)


def _ensure_stop_step_present(rule_id: str, rem_text: str, yaml_fix: str) -> str:
    """The same second-pass mechanism as _ensure_cleanup_step_present(),
    applied to 'service not in use' rules, which require
    'systemctl stop X.socket X.service' before 'dnf remove X'. The first
    call often omits the stop step."""
    if not _STOP_HINT_RE.search(rem_text or ""):
        return yaml_fix
    if _STOP_PRESENT_RE.search(yaml_fix or ""):
        return yaml_fix

    messages = [
        {"role": "system", "content": (
            "You write EXACTLY ONE additional Ansible task, no more, no less. "
            "The CIS Remediation text below requires stopping a systemd service/socket "
            "BEFORE removal, which a colleague's already-generated fix (shown below) is "
            "MISSING — it only removes the package, it never stops the running service "
            "first. Write ONLY that missing stop task. Use this exact shape: a single "
            "'ansible.builtin.systemd' task with 'state: stopped', looping over the "
            "specific '.socket'/'.service' unit name(s) named in the CIS text (usually "
            "both, if both are named), and 'failed_when: false' to tolerate a unit that "
            "was never installed. Output ONLY that one YAML task (a single '- name:' list "
            "item), no explanation, no markdown, no other tasks."
        )},
        {"role": "user", "content": f"""CIS rule: {rule_id}

CIS REMEDIATION TEXT (source of truth for the exact unit name(s) to stop):
{rem_text[:2000]}

ALREADY-GENERATED FIX (missing the stop step — do not repeat any of this, just write the missing task):
{yaml_fix[:1500]}"""},
    ]
    stop_task = _call_llm(messages, identifier=f"STOP {rule_id}")
    if not stop_task or stop_task.startswith("# LLM_CALL_FAILED"):
        return yaml_fix
    print(f"  [A3] ↳ second-pass stop task generated for {rule_id} "
          f"({len(stop_task)} chars) — CIS text requires it, first pass omitted it")
    return stop_task.strip() + "\n" + yaml_fix


_AUTH_BOTH_FILES = ("password-auth", "system-auth")


def _expand_auth_file_mentions(text: str) -> str:
    """Best-effort textual expansion of shell brace expressions and Ansible
    loop/with_items + '{{ item }}' patterns, so that a fix covering both
    password-auth and system-auth (e.g. '{password,system}-auth') is
    recognised. Used only to decide whether the second-pass call is needed,
    never to alter the fix."""
    text = text or ""
    expanded = [text]
    for m in re.finditer(r"\{([\w,\s-]+)\}", text):
        alts = [a.strip() for a in m.group(1).split(",") if a.strip()]
        if len(alts) > 1:
            # Substitute the brace group IN PLACE (keeping prefix/suffix
            # context, e.g. '/etc/pam.d/' + alt + '-auth') — one variant per
            # alternative — not just the bare alternative words, which would
            # lose the surrounding '-auth' suffix the coverage check needs.
            for a in alts:
                expanded.append(text[:m.start()] + a + text[m.end():])
    loop_items = []
    for m in re.finditer(r"(?:loop|with_items)\s*:\s*\[([^\]]+)\]", text):
        loop_items += [i.strip(" '\"") for i in m.group(1).split(",") if i.strip()]
    for m in re.finditer(r"(?:loop|with_items)\s*:\s*\n((?:\s*-\s*.+\n?)+)", text):
        for line in m.group(1).splitlines():
            line = line.strip()
            if line.startswith("-"):
                loop_items.append(line[1:].strip(" '\""))
    # A lambda is used as the replacement, so backslashes in a loop item are
    # inserted literally. Implausible items (very long, containing a
    # backslash or newline) are skipped.
    for item in loop_items:
        if not item or len(item) > 80 or "\\" in item or "\n" in item:
            continue
        expanded.append(re.sub(r"\{\{\s*item\s*\}\}", lambda _m, _v=item: _v, text))
    return "\n".join(expanded)


def _ensure_both_auth_files_covered(rule_id: str, rem_text: str, yaml_fix: str,
                                    sibling_tasks: List["TaskDescription"] = None) -> str:
    """The same second-pass mechanism as _ensure_cleanup_step_present(),
    applied to the PAM/authselect rules (5.3.x), whose CIS remediation must
    update both 'password-auth' and 'system-auth'. Scoped to rule IDs
    starting with '5.3.'."""
    if not rule_id.startswith("5.3."):
        return yaml_fix
    rt = (rem_text or "").lower()
    if not all(f in rt for f in _AUTH_BOTH_FILES):
        return yaml_fix
    yf = _expand_auth_file_mentions(yaml_fix or "").lower()
    if all(f in yf for f in _AUTH_BOTH_FILES):
        return yaml_fix  # already covers both (directly, via brace-expr, or via loop) — nothing to do
    missing = [f for f in _AUTH_BOTH_FILES if f not in yf]
    if sibling_tasks:
        sib_text = _expand_auth_file_mentions(
            " ".join((s.raw_yaml or "") for s in sibling_tasks)
        ).lower()
        if all(m in sib_text for m in missing):
            return yaml_fix  # a sibling task already covers the file(s) this one is missing

    messages = [
        {"role": "system", "content": (
            "You write EXACTLY ONE additional Ansible task, no more, no less. "
            "The CIS Remediation text below requires the SAME edit to be applied to "
            "BOTH '/etc/pam.d/password-auth' AND '/etc/pam.d/system-auth' (or the "
            "equivalent authselect profile paths for those two files), but a "
            "colleague's already-generated fix (shown below) only covers ONE of "
            "them. Write ONLY the missing task: the SAME kind of edit the existing "
            "fix already makes, but looping over BOTH files (e.g. "
            "'loop: [\"password-auth\", \"system-auth\"]' with the appropriate path "
            "prefix, or two explicit tasks — whichever matches the existing fix's "
            "own style) so the file the existing fix is MISSING gets covered too. "
            "Output ONLY that one YAML task (a single '- name:' list item), no "
            "explanation, no markdown, no other tasks."
        )},
        {"role": "user", "content": f"""CIS rule: {rule_id}

CIS REMEDIATION TEXT (source of truth — confirms both files must be covered):
{rem_text[:2000]}

ALREADY-GENERATED FIX (only covers one of the two required files — do not repeat any of this, just write the missing coverage):
{yaml_fix[:1500]}"""},
    ]
    extra_task = _call_llm(messages, identifier=f"AUTH_BOTH_FILES {rule_id}")
    if not extra_task or extra_task.startswith("# LLM_CALL_FAILED"):
        return yaml_fix
    print(f"  [A3] ↳ second-pass password-auth/system-auth coverage task generated for "
          f"{rule_id} ({len(extra_task)} chars) — CIS text requires both files, first pass "
          f"covered only one")
    return yaml_fix.rstrip() + "\n" + extra_task.strip() + "\n"


_AUTHSELECT_PATH_VAR_RE = re.compile(r"\bpam_profile_path\b")
# Plain substring check — deliberately simple. By the time this is checked
# (see caller), we've already confirmed _AUTHSELECT_PATH_VAR_RE does NOT
# match yaml_fix, so any literal occurrence of these path prefixes here
# is necessarily a hardcoded path, not one built from pam_profile_path.
_HARDCODED_AUTHSELECT_PATH_RE = re.compile(
    r"/etc/authselect/|/usr/share/authselect/default/"
)


def _ensure_authselect_profile_path_reused(rule_id: str, sibling_tasks: List["TaskDescription"],
                                           yaml_fix: str) -> str:
    """For the authselect/PAM rules (5.3.2.4.x): when a sibling task already
    registers the fact 'pam_profile_path' and the generated fix hardcodes an
    '/etc/authselect/...' or '/usr/share/authselect/default/...' path, the
    fix is regenerated to use that fact. A hardcoded path works only for one
    type of authselect profile."""
    if not rule_id.startswith("5.3.2.4"):
        return yaml_fix
    sib_text = render_sibling_block(sibling_tasks) if sibling_tasks else ""
    if not _AUTHSELECT_PATH_VAR_RE.search(sib_text):
        return yaml_fix  # no sibling has already computed this fact — nothing to reuse
    if _AUTHSELECT_PATH_VAR_RE.search(yaml_fix):
        return yaml_fix  # already reusing it correctly
    if not _HARDCODED_AUTHSELECT_PATH_RE.search(yaml_fix):
        return yaml_fix  # doesn't hardcode an authselect path at all — not this bug

    messages = [
        {"role": "system", "content": (
            "You fix EXACTLY ONE thing in the Ansible task shown below, nothing else. "
            "The task hardcodes a literal authselect profile path (e.g. '/etc/authselect/"
            "<profile>/...'), but a SIBLING task (shown below) already computed the "
            "CORRECT path for the ACTIVE profile — custom or default — and registered it "
            "as the fact 'pam_profile_path'. Replace the hardcoded literal path prefix "
            "with a Jinja reference to that existing fact (e.g. '{{ pam_profile_path }}/"
            "system-auth'), keeping every other line of the task UNCHANGED (same module, "
            "same regexp, same loop, same when — only the path changes). Output ONLY the "
            "corrected task as YAML (a single '- name:' list item), no explanation, no "
            "markdown."
        )},
        {"role": "user", "content": f"""CIS rule: {rule_id}

SIBLING TASK(S) — one of these already computed 'pam_profile_path' correctly:
{sib_text[:2000]}

TASK TO FIX (hardcodes its own path instead of using 'pam_profile_path'):
{yaml_fix[:1500]}"""},
    ]
    fixed = _call_llm(messages, identifier=f"AUTHSELECT_PATH_REUSE {rule_id}")
    if not fixed or fixed.startswith("# LLM_CALL_FAILED") or not _AUTHSELECT_PATH_VAR_RE.search(fixed):
        return yaml_fix  # didn't converge — keep original rather than risk a worse result
    print(f"  [A3] ↳ second-pass authselect path-reuse fix generated for {rule_id} "
          f"({len(fixed)} chars) — sibling already computed pam_profile_path, first pass "
          f"hardcoded its own path instead")
    return fixed.strip() + "\n"


def generate_remediations(tasks: List[TaskDescription], results: List[ComplianceResult],
                          use_llm=True, min_severity: str = DEFAULT_MIN_SEVERITY,
                          project_path: str = "") -> List[Remediation]:
    """Generates fixes for incorrectly implemented tasks (FIX).

    min_severity: threshold below which a finding does NOT automatically get
    a generated fix (see SEVERITY_RANK above). Set to None/"" to disable
    filtering and generate for all severities as before.

    project_path: used to discover existing handlers (see discover_handlers())
    so a fix can 'notify:' one instead of missing the activation step
    entirely — optional; without it, fixes are generated exactly as before
    (no handler context, same as prior behavior)."""
    remediations = []
    min_rank = SEVERITY_RANK.get(min_severity, -1) if min_severity else -1
    skipped_low_severity = 0
    unknown_severity_count = 0
    handlers = discover_handlers(project_path) if project_path else []
    if handlers:
        print(f"  [A3] Discovered {len(handlers)} existing project handler(s) — "
              f"fixes needing an activation/reload step can 'notify:' one instead of missing it")

    # Same tag-based grouping as agent_a2_llm.py::audit_tasks() (rule_id ->
    # task indices), recomputed here from `tasks`.
    tasks_by_rule: Dict[str, List[int]] = {}
    for idx, t in enumerate(tasks):
        for rid in t.cis_rule_ids:
            tasks_by_rule.setdefault(rid, []).append(idx)

    for idx, (task, result) in enumerate(zip(tasks, results)):
        if result.status not in ("NON_COMPLIANT", "PARTIAL"):
            continue

        # Tasks of rules marked "(Manual)" by CIS get no automatic fix: such
        # rules require a decision of the organisation.
        if getattr(result, "manual_control", False):
            print(f"  [A3] Skipping '{task.task_name}' — tied to a CIS rule marked "
                  f"\"(Manual)\" in the Benchmark (status={result.status}); Manual "
                  f"controls require human judgment, not an automatic fix — still "
                  f"visible in Agent 5's report as informational, not a scored finding.")
            continue

        sibling_tasks = [
            tasks[j] for rid in task.cis_rule_ids
            for j in tasks_by_rule.get(rid, [])
            if j != idx
        ]

        # An unknown severity (e.g. "" or "NONE") is never filtered out by the
        # threshold, so that such findings still get a fix.
        known_severity = result.severity in SEVERITY_RANK
        if known_severity and SEVERITY_RANK[result.severity] < min_rank:
            skipped_low_severity += 1
            print(f"  [A3] Skipping '{task.task_name}' — severity={result.severity} "
                  f"is below threshold {min_severity}, not generating an automatic fix")
            continue
        if not known_severity:
            unknown_severity_count += 1
            print(f"  [A3] Note: '{task.task_name}' has unknown/empty severity "
                  f"({result.severity!r}) — generating a fix anyway (not filtered "
                  f"by the threshold for an unrecognized value)")

        print(f"\n  [A3] Generating fix for: '{task.task_name}' (status={result.status}, severity={result.severity})")
        rule_id    = result.rule_ids[0]    if result.rule_ids    else "?"
        rule_title = result.rule_titles[0] if result.rule_titles else ""

        # Remediation text is fetched for every rule ID of the task, not only
        # the first one.
        rem_text = get_remediation_chunks_multi(result.rule_ids) or f"See CIS {rule_id}"
        if len(result.rule_ids) > 1:
            print(f"  [A3] Task covers {len(result.rule_ids)} CIS rules at once "
                  f"({', '.join(result.rule_ids)}) — fix must satisfy ALL of them")

        yaml_fix = _llm_fix(task, result, rem_text, sibling_tasks=sibling_tasks, handlers=handlers) if use_llm else _mock_fix(task, result, rem_text)
        if use_llm:
            yaml_fix = _ensure_cleanup_step_present(rule_id, rem_text, yaml_fix)
            yaml_fix = _ensure_stop_step_present(rule_id, rem_text, yaml_fix)
            yaml_fix = _ensure_both_auth_files_covered(rule_id, rem_text, yaml_fix, sibling_tasks=sibling_tasks)
            yaml_fix = _ensure_authselect_profile_path_reused(rule_id, sibling_tasks, yaml_fix)

        # Programmatic guardrail: checks that the fix targets the same
        # path/resource as the original task, because the prompt alone did
        # not prevent confusing paths of similar rules (e.g. /tmp vs /var/tmp).
        #
        # original_path is also stored on the Remediation, so that Agent 4
        # can re-check it in the repair rounds.
        original_path = _extract_target_path(task.params) if use_llm else ""
        if use_llm:
            raw_path = _extract_target_path_from_raw(task.raw_yaml or "")
            if raw_path and "{{" in raw_path:
                # See _extract_target_path_from_raw() docstring: prefer the
                # unresolved form when the original task's own path was
                # templated, so the guardrail's skeleton-collapse comparison
                # can recognize a fix that legitimately re-uses the same
                # Jinja variable instead of a literal.
                original_path = raw_path
            if original_path and not _path_guardrail_check(yaml_fix, original_path):
                # A path mismatch is only logged, not forced back, because the
                # original path may itself be wrong. Agent 4's semantic check
                # decides whether the change is a correct fix.
                print(f"  [A3] ⚠ path guardrail (non-blocking): fix does not contain "
                      f"original path '{original_path}' — deferring to Agent 4's "
                      f"knowledge-base check rather than forcing a reversion")

        remediations.append(Remediation(
            original_task=task.task_name,
            rule_id=rule_id,
            rule_title=rule_title,
            yaml_fix=yaml_fix,
            explanation=f"Fix updating the incorrect configuration to meet the requirements of rule {rule_id}.",
            action_type="FIX",
            source_file=task.source_file,
            severity=result.severity,
            cvss_vector=result.cvss_vector,
            cvss_score=result.cvss_score,
            # All rule IDs are passed on, so that Agent 4's semantic check
            # sees every rule the task must satisfy.
            rule_ids=list(result.rule_ids),
            # handlers_block appended here (not a separate field) so it
            # automatically survives into Agent 4's repair_via_agent3*()
            # prompts too — those already forward Remediation.sibling_context
            # verbatim, and a repair round needs the SAME handler awareness
            # as the first generation, not just the initial attempt.
            sibling_context=render_sibling_block(sibling_tasks) + render_handlers_block(handlers),
            original_path=original_path,
        ))
        print(f"  [A3] Generated YAML code ({len(yaml_fix)} characters)")

    if skipped_low_severity:
        print(f"\n  [A3] Skipped {skipped_low_severity} findings below the severity threshold "
              f"({min_severity}) — still visible in Agent 5's report, without an automatic fix.")
    if unknown_severity_count:
        print(f"  [A3] {unknown_severity_count} findings had unknown/empty severity — "
              f"fixes were generated for them anyway (not skipped).")

    return remediations


def detect_and_create_missing_steps(tasks: List["TaskDescription"], coverage: list,
                                    use_llm=True) -> List[Remediation]:
    """Handles rules that already have tasks, but where the CIS text requires
    an additional step that none of these tasks implements (e.g. 'comment out
    any incorrect settings' in rules 1.8.4/1.8.5). Such a step is not visible
    to the audit, because there is no task to flag. Uses the same regex
    hints as _ensure_cleanup_step_present(), applied to all tasks of the rule
    together, and writes the new task through the CREATE path
    (agent_a4.apply_create())."""
    remediations = []
    by_rule: Dict[str, List["TaskDescription"]] = {}
    for t in tasks:
        for rid in (t.cis_rule_ids or []):
            by_rule.setdefault(rid, []).append(t)

    implemented = {c.rule_id: c.title for c in coverage if c.status == "IMPLEMENTED"}

    for rule_id in sorted(implemented):
        rule_tasks = by_rule.get(rule_id, [])
        if not rule_tasks:
            continue
        rem_text = get_remediation_chunk(rule_id) or ""
        if not _CLEANUP_HINT_RE.search(rem_text):
            continue
        combined_yaml = "\n".join(t.raw_yaml or "" for t in rule_tasks)
        if _CLEANUP_PRESENT_RE.search(combined_yaml):
            continue  # some existing sibling task already covers it

        rule_title = implemented[rule_id]
        print(f"  [A3] ↳ rule {rule_id} has {len(rule_tasks)} existing task(s) but NONE cover "
              f"the cleanup/comment-out step CIS text requires — generating a NEW task for it "
              f"(gap between CREATE and FIX)")

        if not use_llm:
            continue
        existing_block = "\n\n".join(
            f"- {t.task_name}:\n{t.raw_yaml or ''}" for t in rule_tasks[:8]
        )
        messages = [
            {"role": "system", "content": (
                "You write EXACTLY ONE new Ansible task, no more, no less. A CIS rule already has "
                "several existing tasks (shown below) covering MOST of its remediation steps, but "
                "the CIS text describes an additional 'comment out'/'review all files' style step "
                "that NONE of the existing tasks implement. Write ONLY that missing task, matching "
                "the style/conventions of the existing sibling tasks shown (module choice, naming, "
                "'when:'/'notify:' patterns). Output ONLY that one YAML task (a single '- name:' "
                "list item), no explanation, no markdown."
            )},
            {"role": "user", "content": f"""CIS rule: {rule_id}

CIS REMEDIATION TEXT (full):
{rem_text[:3000]}

EXISTING TASKS FOR THIS RULE (already implemented, do not duplicate any of this):
{existing_block[:2500]}"""},
        ]
        new_task = _call_llm(messages, identifier=f"MISSING_STEP {rule_id}")
        if not new_task or new_task.startswith("# LLM_CALL_FAILED"):
            print(f"  [A3]   ! LLM call failed for {rule_id}'s missing step — skipped")
            continue

        remediations.append(Remediation(
            original_task="[NO TASK IN PROJECT — rule partially implemented, one step missing]",
            rule_id=rule_id,
            rule_title=rule_title,
            yaml_fix=new_task.strip(),
            explanation=(f"Rule {rule_id} has {len(rule_tasks)} existing task(s) but none cover a "
                        f"required cleanup/comment-out step described in the CIS text — generated "
                        f"the missing task."),
            action_type="CREATE",
            severity="LOW",
            cvss_vector="",
            cvss_score=None,
        ))
        print(f"  [A3] Generated YAML code ({len(new_task)} characters) for missing step in {rule_id}")

    return remediations


def generate_missing_tasks(missing_rules: List[dict], use_llm=True) -> List[Remediation]:
    """Generates brand-new tasks for rules missing from the project (CREATE)."""
    remediations = []

    for rule_data in missing_rules:
        rule_id = rule_data.get("rule_id", "UNKNOWN")
        rule_title = rule_data.get("title", "Missing CIS Rule")
        chunk_text = rule_data.get("text", "")

        # CREATE has no audit result, so severity is assessed separately with
        # the same mechanism as for FIX (8 CVSS metrics from the LLM, score
        # computed in cvss_calculator.py).
        severity, cvss_vector, cvss_score = assess_missing_rule_cvss(
            rule_id, rule_title, chunk_text) if use_llm else ("", "", None)

        sev_label = f"{severity}" + (f", CVSS {cvss_score}" if cvss_score is not None else "")
        print(f"\n  [A3] Generating a NEW task for missing rule: CIS {rule_id} (severity={sev_label or 'none'})")
        rem_text = get_remediation_chunk(rule_id) or extract_remediation_text(chunk_text)

        yaml_fix = _llm_create(rule_id, rule_title, rem_text) if use_llm else "- name: Mock new task"

        remediations.append(Remediation(
            original_task="[NO TASK IN PROJECT]",
            rule_id=rule_id,
            rule_title=rule_title,
            yaml_fix=yaml_fix,
            explanation=f"Created a brand new task to address the detected missing rule {rule_id}.",
            action_type="CREATE",
            severity=severity,
            cvss_vector=cvss_vector,
            cvss_score=cvss_score,
        ))
        print(f"  [A3] Generated YAML code ({len(yaml_fix)} characters)")

    return remediations


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TEST Agent A3 (FIX and CREATE modes) ===\n")

    # 1. Test FIX mode
    print("--- 1. TEST FIX MODE (Fixing a task) ---")
    task = TaskDescription(
        task_name="Set SSH PermitRootLogin",
        module="lineinfile",
        params={"path": "/etc/ssh/sshd_config", "line": "PermitRootLogin yes"},
        semantic="SSH PermitRootLogin yes configuration in sshd_config"
    )
    result = ComplianceResult(
        task_name="Set SSH PermitRootLogin",
        status="NON_COMPLIANT",
        rule_ids=["5.2.3"],
        rule_titles=["Ensure SSH PermitRootLogin is disabled"],
        retrieved_chunks=[{"text": "Remediation:\nEdit /etc/ssh/sshd_config and set:\nPermitRootLogin no\n"}]
    )

    fix_rems = generate_remediations([task], [result], use_llm=USE_LLM)
    print(f"\n[FIX RESULT]\n{fix_rems[0].yaml_fix}")


    # 2. Test CREATE mode
    print("\n\n--- 2. TEST CREATE MODE (Generating a new rule from scratch) ---")
    missing_rule_mock = [
        {
            "rule_id": "1.1.1.2",
            "title": "Ensure freevxfs filesystem is disabled",
            "text": "Remediation:\nRun the following command to disable freevxfs:\n# echo 'install freevxfs /bin/true' >> /etc/modprobe.d/freevxfs.conf\n"
        }
    ]

    create_rems = generate_missing_tasks(missing_rule_mock, use_llm=USE_LLM)
    print(f"\n[CREATE RESULT]\n{create_rems[0].yaml_fix}")
