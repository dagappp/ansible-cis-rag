"""
Agent A2 — RAG Auditor with the university LLM
Task: for an Ansible task description, retrieve CIS rules from ChromaDB
and assess compliance via the LLM. Handles rules intentionally disabled in the project.
"""

import os
import re
import chromadb
import json
import time
import torch
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer

from cvss_calculator import calculate_base_score
from debug_dump import dump_llm_call, extract_json_object

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION  = "cis_flat"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K       = 5
# A rule split into several chunk types can fill the raw nearest-neighbour
# pool; fetching a wider pool keeps top_k distinct rules after deduplication.
_RAW_MULTIPLIER = 4
_RAW_CAP        = 30

# ── University LLM server (OpenAI-compatible) ─────────────────────────────────
API_URL     = os.environ.get("LLM_API_URL", "http://localhost:8080/v1/chat/completions")
API_TOKEN   = os.environ.get("LLM_API_TOKEN", "")
API_MODEL   = "qwen3.5-122b-a10b"
USE_LLM     = True   # True = university LLM, False = mock

# When True, llm_verify() never falls back to mock_verify() and keeps
# retrying until the model responds. Used in model comparisons, so that
# every result comes from the model under test.
DISABLE_MOCK_FALLBACK = False

# ── LLM call statistics (request count, time, failures) ──────────────────────
# Shared counter for the ENTIRE pipeline (all agents call
# _record_request here, dynamically via "import agent_a2_llm" — the same
# pattern as API_MODEL). For every model run we record not only total time and quality, but also the COST (number of
# requests, total time spent in LLM calls, number of failed attempts).
REQUEST_COUNT     = 0
REQUEST_TIME_TOTAL = 0.0
REQUEST_FAILURES  = 0

# Average request time excludes the first successful request of a run,
# which is slower because of model loading on the server.
#
# These counters accumulate only successful requests. Failed attempts
# return almost instantly and would distort the average model latency.
REQUEST_TIME_TOTAL_EXCL_FIRST = 0.0
REQUEST_COUNT_EXCL_FIRST      = 0
_first_success_seen           = False


def reset_request_stats() -> None:
    """Reset the counters at the start of a run, so that statistics of
    different models run in the same process are not mixed."""
    global REQUEST_COUNT, REQUEST_TIME_TOTAL, REQUEST_FAILURES
    global REQUEST_TIME_TOTAL_EXCL_FIRST, REQUEST_COUNT_EXCL_FIRST, _first_success_seen
    REQUEST_COUNT = 0
    REQUEST_TIME_TOTAL = 0.0
    REQUEST_FAILURES = 0
    REQUEST_TIME_TOTAL_EXCL_FIRST = 0.0
    REQUEST_COUNT_EXCL_FIRST = 0
    _first_success_seen = False


def _record_request(elapsed: float, success: bool = True) -> None:
    """Called after EVERY attempt to connect to the LLM API (successful or
    not), from all agents (A1/A2/A3/A4-repair/A5/judge) — tracks the real
    cost of requests, not just 'how many tasks were processed'."""
    global REQUEST_COUNT, REQUEST_TIME_TOTAL, REQUEST_FAILURES
    global REQUEST_TIME_TOTAL_EXCL_FIRST, REQUEST_COUNT_EXCL_FIRST, _first_success_seen
    REQUEST_COUNT += 1
    REQUEST_TIME_TOTAL += elapsed
    if not success:
        REQUEST_FAILURES += 1
        return  # failed attempts never feed the average-latency counters
    if _first_success_seen:
        REQUEST_TIME_TOTAL_EXCL_FIRST += elapsed
        REQUEST_COUNT_EXCL_FIRST += 1
    else:
        _first_success_seen = True


def average_request_time(exclude_first: bool = True) -> float:
    """Average seconds per LLM request for the current run. exclude_first=True
    (default) drops the first request's timing — see the comment above
    REQUEST_TIME_TOTAL_EXCL_FIRST for why. Returns 0.0 if there's nothing to
    average yet (e.g. only one request has been made and exclude_first=True)."""
    if exclude_first:
        return (REQUEST_TIME_TOTAL_EXCL_FIRST / REQUEST_COUNT_EXCL_FIRST) if REQUEST_COUNT_EXCL_FIRST else 0.0
    return (REQUEST_TIME_TOTAL / REQUEST_COUNT) if REQUEST_COUNT else 0.0


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class TaskDescription:
    """Kept in sync with the output of Parser A0."""
    task_name:    str
    module:       str
    params:       Dict[str, Any]
    semantic:     str
    raw_yaml:     str = ""
    cis_rule_ids: List[str] = field(default_factory=list)
    cis_level:    str = ""
    source_file:  str = ""
    when_rules:   List[str] = field(default_factory=list)

@dataclass
class ComplianceResult:
    task_name:        str
    status:           str   # COMPLIANT / NON_COMPLIANT / PARTIAL / UNKNOWN / RISK_ACCEPTED
    severity:         str = ""   # LOW / MEDIUM / HIGH / CRITICAL — derived from cvss_score (see cvss_calculator.py)
    unknown_reason:   str = ""   # NOT_MAPPED / LOW_CONFIDENCE / LLM_UNCERTAIN / API_ERROR — only when status=UNKNOWN
    cvss_vector:      str = ""   # e.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N" — only when the LLM (not mock) assessed the metrics
    cvss_score:       Optional[float] = None  # 0.0-10.0, computed deterministically per the official CVSS v3.1 formula
    rule_ids:         List[str] = field(default_factory=list)
    rule_titles:      List[str] = field(default_factory=list)
    explanation:      str = ""
    retrieved_chunks: List[dict] = field(default_factory=list)
    retrieval_ms:     float = 0.0
    manual_control:   bool = False  # CIS marks this rule "(Manual)" — see parser_a0_final.get_manual_rule_ids().
                                     # The LLM verdict is still computed and shown, but agent_a5_report.py
                                     # keeps it OUT of the scored compliance % and the findings list — per
                                     # CIS's own definition, a Manual control needs human validation, so an
                                     # automated pass/fail here is informational, never a penalized "finding".


# ── Embedding model (singleton) ────────────────────────────────────────────
_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Uses Apple Silicon MPS when available (much faster encode()),
        # with a CPU fallback for other machines.
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  [A2] Loading model {EMBED_MODEL} (device={device})...")
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
        print("  [A2] Model loaded.")
    return _model


# ── ChromaDB ──────────────────────────────────────────────────────────────────
# Cached collection (singleton), so that every retrieve() call does not open
# a new database client.
_collection = None
_collection_name = None

def get_collection():
    """Cache keyed by collection name (not just a plain singleton) — the
    orchestrator sets agent_a2_llm.COLLECTION BEFORE the first retrieve()
    call (same pattern as for USE_LLM), so in normal operation this still
    behaves like a plain singleton. Keying by name only guards against a
    silent mistake if someone changed COLLECTION mid-process."""
    global _collection, _collection_name
    if _collection is None or _collection_name != COLLECTION:
        client = chromadb.PersistentClient(CHROMA_PATH)
        _collection = client.get_collection(COLLECTION)
        _collection_name = COLLECTION
    return _collection


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K):
    t0 = time.time()
    try:
        model   = get_model()
        q_emb   = model.encode([query], normalize_embeddings=True)
        col     = get_collection()
        # A rule above the chunking threshold is stored as up to three
        # embeddings, which can crowd out other rules. A wider raw pool
        # (top_k * RAW_MULTIPLIER, capped) keeps top_k distinct rules
        # after deduplication by rule_id.
        raw_n = min(top_k * _RAW_MULTIPLIER, _RAW_CAP)
        results = col.query(
            query_embeddings=q_emb.tolist(),
            n_results=raw_n,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        # Embedding or database errors do not stop the run: the task is
        # audited without retrieved context.
        print(f"  [A2] Retrieval error (embedding/ChromaDB): {e} — "
              f"continuing without RAG context for this task")
        return [], round((time.time() - t0) * 1000, 1)

    chunks = []
    seen   = set()
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        rid = meta.get("rule_id", "?")
        if rid in seen:
            continue
        if len(chunks) >= top_k:
            break  # enough DISTINCT rules already — raw_n only widened the candidate pool
        seen.add(rid)
        chunks.append({
            "rule_id":    rid,
            "title":      meta.get("title", ""),
            "chunk_type": meta.get("chunk_type", "full"),
            "text":       doc,
            "score":      round(1 - dist, 4),
        })

    # The top match is always shown in full: if the rule is split into
    # context/audit/remediation chunks, all of them are merged, so the model
    # also sees the Audit section with the pass criteria.
    if chunks and chunks[0]["chunk_type"] != "full":
        try:
            col = get_collection()
            all_parts = col.get(where={"rule_id": chunks[0]["rule_id"]},
                                include=["documents", "metadatas"])
            by_type = {m["chunk_type"]: d for d, m in zip(all_parts["documents"], all_parts["metadatas"])}
            order = ["context", "audit", "remediation"]
            merged_text = "\n\n".join(by_type[t] for t in order if t in by_type)
            if merged_text:
                chunks[0] = {**chunks[0], "chunk_type": "+".join(t for t in order if t in by_type),
                            "text": merged_text}
        except Exception as e:
            print(f"  [A2] Could not merge split-rule chunks for {chunks[0]['rule_id']}: {e} — "
                  f"using the single retrieved chunk as-is")

    return chunks, round((time.time() - t0) * 1000, 1)


# ── Severity rubric (shared by the LLM and mock_verify) ──────────────────────
# Used only for NON_COMPLIANT/PARTIAL results: describes how severe the gap
# is, based on the CVSS v3.1 standard.
#
# The levels correspond to the CVSS v3.1 Qualitative Severity Rating Scale
# (specification, section 5), and the criteria refer to the CVSS base
# metrics (AV, PR, UI and C/I/A impact).
SEVERITY_RUBRIC = """SEVERITY (only when status is NON_COMPLIANT or PARTIAL, otherwise null) —
grounded in the CVSS v3.1 Qualitative Severity Rating Scale (FIRST.org):
  CRITICAL (equivalent to CVSS 9.0-10.0) — no privileges required to exploit the gap
    (Privileges Required: None) and/or a remote attack vector with no user interaction
    (Attack Vector: Network, User Interaction: None), full impact on confidentiality
    and/or integrity (Confidentiality/Integrity Impact: High) — e.g. authentication,
    passwords/keys, privilege escalation, remote root access
  HIGH (equivalent to CVSS 7.0-8.9) — low requirements to exploit (Attack Complexity:
    Low, Privileges Required: Low), significant impact on confidentiality OR integrity —
    e.g. firewall, SELinux, SSH, package GPG signatures
  MEDIUM (equivalent to CVSS 4.0-6.9) — requires elevated privileges or user interaction
    (Privileges Required: High or User Interaction: Required), partial impact on C/I/A —
    e.g. logging/auditing, file permissions, service configuration
  LOW (equivalent to CVSS 0.1-3.9) — minimal or no direct impact on
    confidentiality/integrity/availability (Impact: Low/None) — e.g. login banners,
    documentation settings"""

# Kept as readable documentation and for mock_verify(). The LLM receives
# CVSS_METRICS_GUIDE instead: it returns the 8 base metrics and the score
# is computed in cvss_calculator.py.

# The same letters as in the official FIRST.org calculator
# (https://www.first.org/cvss/calculator/3.1). The LLM returns only these
# letters, the score and label are computed deterministically in
# cvss_calculator.py.
CVSS_METRICS_GUIDE = """Assess the following CVSS v3.1 base metrics for this gap
(only when status is NON_COMPLIANT or PARTIAL — for COMPLIANT/UNKNOWN, cvss_metrics: null):
  AV (Attack Vector) — where it can be exploited from: N=network/remote, A=adjacent network,
    L=local (requires access to the machine), P=physical
  AC (Attack Complexity) — L=easy/repeatable, H=requires special conditions
  PR (Privileges Required) — N=no privileges, L=regular user, H=administrator privileges
  UI (User Interaction) — N=does not require victim action, R=requires it (e.g. a click)
  S (Scope) — U=the gap affects only the same component, C=it can affect other components
  C/I/A (Confidentiality/Integrity/Availability Impact) — for each of these three: N=no impact,
    L=partial impact, H=full/severe impact
IMPORTANT: assess the C/I/A metrics under the ASSUMPTION that an attacker ACTUALLY exploits
this SPECIFIC gap (that's how CVSS works — it always assumes successful exploitation, it does
not ask about the probability of an attack). The question is "what happens IF someone exploits
this", not "does the mere existence of this misconfiguration already harm someone". Almost no
CIS rule should get C:N/I:N/A:N all at once — since the rule exists in the benchmark, violating
it has some real security consequence (even if minor, that's L, not N on all three at once).

Assess realistically relative to the SPECIFIC gap in this task (e.g. missing package signature
verification is usually AV:N/PR:N — the attacker needs no access or privileges to exploit it via
a poisoned package/repository), not relative to a hypothetical worst-case scenario."""

# Severity is always assessed by the LLM (8 CVSS metrics, score computed in
# cvss_calculator.py), both for audited tasks and for missing rules. There is
# no keyword-based fallback: without the LLM, severity stays empty.


# ── University LLM ────────────────────────────────────────────────────────────
def _template_content_note(task: TaskDescription) -> str:
    """Surfaces task.template_content (see parser_a0_final.TaskDescription /
    extract_template_rule_block()): the Jinja2 block from the role's .j2
    templates that holds this rule's real enforcement content.

    Some roles (e.g. ansible-lockdown) enforce many rules through one shared
    template, while the per-rule tasks only set a fact. Without this block
    the model cannot judge such rules.
    """
    if not task.template_content:
        return ""
    return (
        "\nACTUAL ENFORCEMENT CONTENT (from this project's Jinja2 template file(s) — "
        "this task's raw_yaml above may be a mere placeholder/fact-setter, e.g. "
        "'set_fact: update_audit_template: true', with the REAL content rendered "
        "later by a shared template task; this is that real content, specifically "
        "the block gated behind this task's own rule_id):\n"
        f"{task.template_content[:2500]}\n"
    )


def _resolved_params_note(task: TaskDescription) -> str:
    """Surfaces task.params (Jinja-resolved at parse time, see
    parser_a0_final.resolve_jinja()) whenever raw_yaml still contains an
    unresolved '{{ ... }}' expression, so the model gets the actual
    configured value instead of guessing it from the template text.
    """
    note = ""
    if "{{" in (task.raw_yaml or "") and task.params:
        note += (
            "\nRESOLVED PARAMETER VALUES (raw_yaml above may show an unresolved "
            "'{{ variable }}' — these are the ACTUAL values after substituting "
            "the project's defaults/ variables, use these to judge compliance, "
            "not the literal template text):\n"
            f"{json.dumps(task.params, indent=2, default=str)}\n"
        )
    if getattr(task, "resolved_loop", None):
        # Shows the loop's resolved per-iteration values, which the task's
        # own parameters only reference as '{{ item.xxx }}'.
        note += (
            "\nRESOLVED LOOP ITERATIONS (this task's 'loop:'/'with_items:' list, with each "
            "item's own '{{ project variable }}' references ALSO substituted — 'item.xxx' in "
            "the params above refers to each of these in turn, one task execution per item):\n"
            f"{json.dumps(task.resolved_loop, indent=2, default=str)}\n"
        )
    return note


def _rendered_template_note_safe(raw_yaml: str, project_path: str) -> str:
    """Local-import wrapper around judge_combined._rendered_template_note()
    (a module-level import would be circular).

    Gives Agent 2 the same rendered template content that the judge sees.
    project_path="" makes it a no-op."""
    if not project_path:
        return ""
    try:
        from judge_combined import _rendered_template_note
        return _rendered_template_note(raw_yaml, project_path)
    except Exception:
        return ""


def llm_verify(task: TaskDescription, chunks: list, sibling_tasks: List[str] = None,
               project_path: str = "") -> tuple:
    if not USE_LLM or not chunks:
        return mock_verify(task, chunks)

    # Pure embedding-similarity retrieval, without fetching rules by tag.
    # A wrong tag would otherwise inject the text of an unrelated rule.
    #
    # Retrieved chunks are shown whole, without extra truncation: their size
    # is already bounded by the chunking strategy (1024-token threshold).
    context = "\n\n".join(
        f"=== CIS Rule {c['rule_id']}: {c['title']} ===\n{c['text']}"
        for c in chunks[:3]
    )

    # When no retrieved rule matches the task's own declared rule ID, the
    # model is told explicitly, so that it treats the case as a rule-mapping
    # problem instead of inventing claims about the code.
    tag_mismatch_note = ""
    if task.cis_rule_ids and chunks and not any(c["rule_id"] in task.cis_rule_ids for c in chunks[:3]):
        tag_mismatch_note = (
            f"\nWARNING — RULE MAPPING MISMATCH: this task is tagged/declared as rule(s) "
            f"{task.cis_rule_ids}, but NONE of the CIS rules retrieved above (by text "
            f"similarity) share that rule number — the retrieved rules are simply the "
            f"closest topical match, not necessarily the SAME control. This can mean the "
            f"project mistagged the task, or references a rule number that was renumbered/"
            f"removed in the current CIS Benchmark version. Do NOT use this mismatch as "
            f"grounds to claim the task's own code doesn't exist or doesn't do what it says "
            f"— you can see the task's actual raw_yaml above the rules; trust that over the "
            f"retrieved rule text. If none of the retrieved rules genuinely describe what "
            f"this task does, prefer status \"UNKNOWN\" (the reference text for this specific "
            f"declared rule number could not be retrieved) over a confident NON_COMPLIANT "
            f"built on a rule that may not even be the right one.\n"
        )

    # Projects often split one CIS rule across several tasks. The model gets
    # the list of sibling tasks for the same rule and assesses this task in
    # that context, while still checking the task's own logic.
    sibling_block = ""
    if sibling_tasks:
        sib_list = "\n".join(f"  - {s}" for s in sibling_tasks[:10])
        sibling_block = f"""
CONTEXT: This task is PART OF A MULTI-STEP implementation of the same CIS rule,
together with other tasks in the project:
{sib_list}
Assess the COMPLIANCE OF THE WHOLE implementation (this task + the listed siblings
together), NOT just this single task in isolation — if this task correctly carries
out its part (e.g. only the preparation/installation, while the actual configuration
is in a sibling task), that is NOT grounds for PARTIAL/NON_COMPLIANT. At the same
time, still flag NON_COMPLIANT/PARTIAL if THIS task itself has an error in its own
logic (e.g. an inverted condition, a wrong parameter) — that is independent of what
the siblings do.
IMPORTANT — what to put in the "status" field below: it must reflect the WHOLE
combined implementation's outcome, not "does this one task alone implement 100% of
the rule". If this task's own part is correct and free of errors, and the sibling(s)
listed above plausibly cover the rest of the requirement (e.g. one sibling's name
indicates it handles the other required directive/step), respond "status":
"COMPLIANT" for THIS task too — do NOT respond "PARTIAL" solely because this task by
itself only implements a fraction of the full rule. Reserve PARTIAL/NON_COMPLIANT
for when THIS task's own logic is wrong, or the listed siblings do NOT plausibly
cover what this task leaves out.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a CIS Benchmark compliance auditor. "
                "Analyze Ansible tasks and determine if they comply with CIS rules. "
                "CONFIRMED TECHNICAL FACT: Ansible file-module 'mode:' accepts SYMBOLIC notation "
                "(u/g/o/a + +/-/= + rwx), e.g. 'go-rwx' means 'remove read/write/execute for Group "
                "and Other' — this TIGHTENS permissions, it is NOT the octal value 0777 and does not "
                "grant broad access. Never read a symbolic mode string as if its letters were octal "
                "digits, and never flag a correctly-tightened symbolic mode as an insecure permission. "
                "This ALSO applies when the mode string has MULTIPLE comma-separated clauses, e.g. "
                "'o-x,g-wx,o-rwx' — EVERY clause using '-' still REMOVES, no matter how many clauses "
                "there are or whether a later clause re-targets the same letter (here 'o-x' then "
                "'o-rwx' both REMOVE from Others — sequentially redundant, not additive/canceling; "
                "the net effect on Others is still 'no read, no write, no execute'). A '-' clause "
                "NEVER adds a permission back, regardless of what other clauses are present. "
                "CONFIRMED TECHNICAL FACT: this applies SPECIFICALLY to AUDIT/CHECK-style tasks — a "
                "shell/command task whose PURPOSE is to VERIFY current state (e.g. 'findmnt | grep "
                "nodev', 'systemctl is-enabled ...', a task literally named/tagged '| AUDIT |' or "
                "'Check ...') — NOT to state-changing remediation commands (dnf/package "
                "install-or-remove, a firewall-cmd/systemctl mutating call, a file/lineinfile/template "
                "edit — those are already 'doing the work', ignore_errors on them is a separate, "
                "MUCH LESS SEVERE robustness note, never a 'the rule isn't implemented at all' issue). "
                "For an AUDIT/CHECK task specifically, check IN THIS ORDER: (1) FIRST, does THIS SAME "
                "task already have its OWN 'failed_when:'/'assert:' that references its OWN "
                "'register:' variable (e.g. 'register: var_status' + \"failed_when: '\\\"x\\\" not in "
                "var_status.stdout'\" on the SAME task)? If yes, the check is COMPLETE and correctly "
                "evaluated regardless of 'ignore_errors: yes' being ALSO present — 'ignore_errors: yes' "
                "there only means 'don't halt the whole play if this fails', it does NOT erase the "
                "failed_when verdict; do not flag this pattern. (2) ONLY if the task has NO "
                "'register:' at all, or a 'register:' that NEITHER its own 'failed_when:'/'assert:' "
                "NOR any sibling task's 'when:'/'failed_when:'/'assert:' ever references, does 'ignore_errors: "
                "yes' mean the "
                "check's result is silently discarded — do NOT mark it COMPLIANT just because the "
                "command LOOKS like the right check ('the task correctly implements the audit logic' "
                "is not sufficient) — a check whose result is never evaluated anywhere is functionally "
                "a no-op for compliance purposes, equivalent to having no check at all. (3) The SAME "
                "no-op verdict applies when a check task expresses its pass/fail condition ONLY "
                "through 'changed_when:' (e.g. \"changed_when: grep_result.stdout != '1'\") with NO "
                "'failed_when:'/'assert:' anywhere — 'changed_when' ONLY controls whether Ansible "
                "reports the task as 'changed' vs 'ok' in its own output; it NEVER halts the play or "
                "signals failure, regardless of its value. A task using changed_when as if it were "
                "failed_when looks like a real check but can never actually fail — flag it the same "
                "way as an ignore_errors-without-failed_when check. Confirmed live: multiple "
                "'duplicate UID/GID/name' detection tasks (grep count against /etc/passwd or /etc/"
                "group) used exactly this pattern and were wrongly rated COMPLIANT despite the check "
                "never being able to actually report the duplicate it found. Confirmed live: "
                "multiple 'findmnt ... | grep nodev' AUDIT tasks with exactly this shape (ignore_errors: "
                "yes, no register, no sibling task anywhere that sets the option) were incorrectly "
                "rated COMPLIANT — the correct verdict there is NON_COMPLIANT, UNLESS a sibling task "
                "elsewhere actually sets the option (e.g. a 'mount:'/'ansible.posix.mount' task with "
                "matching opts, or a register+failed_when pair that WOULD fail the play). Do NOT "
                "extend this same NON_COMPLIANT-strength verdict to a remediation/PATCH task merely "
                "swallowing its own command's potential failure — note that separately as a minor "
                "robustness gap, not as 'the rule is unimplemented'. CRITICAL: this whole concern "
                "applies ONLY when failure-suppression is ACTUALLY PRESENT ('ignore_errors: yes' "
                "and/or 'failed_when: false/no') — a plain remediation task with NO such suppression "
                "(normal Ansible failure semantics: if the shell/command genuinely errors, the play "
                "fails) needs NO separate post-verification step to be COMPLIANT. CIS remediation "
                "scripts themselves rarely include a 'verify after fixing' step either — the "
                "state-changing action itself, under normal failure semantics, IS the implementation. "
                "Do NOT flag a remediation task NON_COMPLIANT just for 'lacking a mechanism to verify "
                "the change succeeded' when it has no ignore_errors/failed_when suppressing that "
                "verification — that invents a requirement the rule doesn't actually have. This same "
                "prohibition covers every REWORDING of that non-issue, all of which are equally "
                "invalid grounds for NON_COMPLIANT on their own: 'performs both the check and the fix "
                "in one command', 'no mechanism to ensure the state was actually achieved', 'does not "
                "report failure', 'lacks proper Ansible module usage for state management', 'uses "
                "shell/command instead of a dedicated module', 'is not idempotent'. Compliance asks "
                "ONE question: does this task, as written, bring the system to the state the CIS rule "
                "requires? A single 'find ... -exec chown/chmod' shell line that correctly enforces "
                "the required ownership/permissions IS a complete, compliant implementation — CIS's "
                "own Remediation sections are themselves largely shell one-liners of exactly this "
                "shape. Style/robustness preferences (dedicated module over shell, separate audit "
                "step, idempotency polish) belong to CODE QUALITY, which a SEPARATE evaluator scores; "
                "they are never, on their own, a compliance failure. "
                "CONFIRMED TECHNICAL FACT — judge ONLY against what the retrieved CIS rule text "
                "ACTUALLY requires, never against a stricter standard you infer yourself. Confirmed "
                "live (rule 6.2.3.4): the CIS Remediation says only \"set $FileCreateMode to 0640 or "
                "more restrictive\" INSIDE the rsyslog config file — it says NOTHING about the "
                "permissions of that .conf file itself; the task correctly wrote '$FileCreateMode "
                "0640' but was wrongly flagged NON_COMPLIANT because its own 'mode: 0644' (the "
                "config file's permissions, a DIFFERENT thing entirely) 'looked too permissive'. "
                "Before flagging, ask: is this requirement written in the rule text above, or am I "
                "adding it because it seems like good practice? A parameter the rule never mentions "
                "is out of scope for THIS rule (it may be another rule's subject — that other rule "
                "has its own separate task). "
                "CONFIRMED TECHNICAL FACT — when the CIS remediation text lists MULTIPLE, "
                "SEQUENTIAL steps (e.g. 'systemctl stop X.service' THEN 'dnf remove X'), verify the "
                "task (plus its listed siblings) actually performs ALL of them, not just the LAST/"
                "most-visible one. Removing a package does NOT, by itself, guarantee an already-"
                "running instance of its service was stopped first — a systemd service can keep "
                "running with removed binaries until the next stop/reboot. Confirmed live (rule "
                "2.1.5): a task doing only 'dnf: name: bind, state: absent' was wrongly rated "
                "COMPLIANT — the CIS remediation is 'systemctl stop named.service' THEN 'dnf remove "
                "bind', two required steps, and the task (with no sibling covering the stop step "
                "either) only ever performs the second. This is a recurring pattern across every "
                "'service not in use' rule in this benchmark family (stop-then-remove) — do not "
                "credit COMPLIANT for the removal step alone unless the stop step is ALSO genuinely "
                "present, in this task or a real sibling. Two MORE confirmed shapes of this same "
                "multi-step pattern: (a) 'add <directive> to file X' AND SEPARATELY 'remove any "
                "occurrence of <negated-directive>' (e.g. rule 5.2.2 requires adding 'Defaults "
                "use_pty' AND removing any '!use_pty' — a task/sibling family that only adds the "
                "positive directive, never searching for/removing the negated form, is INCOMPLETE, "
                "even though it looks done); (b) the text explicitly NAMES SEVERAL target files that "
                "must ALL be updated together (e.g. rule 5.3.2.4.3 names BOTH /etc/pam.d/"
                "password-auth AND /etc/pam.d/system-auth) — a task/sibling family covering only ONE "
                "of the explicitly-named files is INCOMPLETE even if that one file is handled "
                "correctly. "
                "CONFIRMED TECHNICAL FACT — many CIS remediations (sysctl parameters, crypto "
                "sub-policies, dconf/GNOME settings) explicitly include a step to find and COMMENT "
                "OUT or REMOVE any CONFLICTING pre-existing entry for the same setting in OTHER "
                "files, in addition to writing the new value. Writing only the new value, without "
                "this cleanup step, does NOT guarantee it takes effect — sysctl.d/dconf files are "
                "read in a defined order and a leftover conflicting entry elsewhere can silently "
                "override the new one (last-loaded-wins for sysctl.d). Confirmed live across dozens "
                "of rules in the 3.3.x (sysctl) and 1.5.x/1.6.x (crypto policy) families: tasks that "
                "correctly create the new configuration file were wrongly rated COMPLIANT despite "
                "never checking for or removing a conflicting pre-existing entry, when the retrieved "
                "CIS text explicitly required that cleanup as its own numbered step. Check the "
                "retrieved remediation text for an explicit 'find'/'comment out'/'remove conflicting' "
                "step BEFORE crediting COMPLIANT for a task that only writes the new value. "
                "CONFIRMED TECHNICAL FACT — many CIS remediations end with a final ACTIVATION step "
                "('sysctl --system', 'update-crypto-policies', 'dconf update', 'augenrules --load', "
                "'systemctl reload-or-restart X') that must run for the already-written configuration "
                "to actually take effect. This step is satisfied EITHER by a direct command in the "
                "task itself OR by 'notify: \"<handler name>\"' — Ansible runs a notified handler "
                "automatically, once, at the end of the play, which IS a genuine activation step even "
                "though no command appears directly in the task. Do not penalize a task for 'missing "
                "the activation step' if it has a 'notify:' line — but do NOT assume a notify target "
                "does the right thing either; if you cannot tell what a notified handler actually runs "
                "and it matters to your verdict, reflect that uncertainty rather than guessing either "
                "way. A task with NEITHER a direct activation command NOR any 'notify:' genuinely "
                "lacks this step. "
                "CONFIRMED TECHNICAL FACT — a 'sed' substitution that captures the very value it is "
                "supposed to DELETE into a numbered group, and then re-inserts that same group number "
                "in the replacement string, is a NO-OP for that value: it deletes nothing, even though "
                "the task's name/intent says 'remove'. Confirmed live example: "
                "\"s/(^\\s*password\\s+...pam_unix\\.so\\s+.*)(nullok)(\\s*.*)$/\\1\\3/g\" — group 2 "
                "(\"nullok\") is captured but never used in the replacement, while group 3 (whatever "
                "comes after) IS kept — the target value the task claims to remove never actually "
                "leaves the line. When a task's purpose is to remove/strip a specific token via sed, "
                "check which numbered group holds that token and whether the replacement string omits "
                "it — if the group holding the target token IS present in the replacement, the value "
                "survives and the task is NON_COMPLIANT regardless of what the task name claims. "
                "CONFIRMED TECHNICAL FACT — when the CIS Remediation text ITSELF uses language like "
                "'investigate', 'review to determine', 'may be appropriate', 'if appropriate', "
                "'establish unique ... based on the results of the audit script', 'perform the "
                "appropriate action', or explicitly says the correct value/location 'will vary "
                "depending on your environment' "
                "— the rule requires HUMAN JUDGMENT or environment-specific information that cannot "
                "be safely auto-computed generically. A task that DETECTS the condition and FAILS "
                "LOUDLY with a descriptive message (fail/failed_when on the detected condition) IS a "
                "complete, correct, HONEST implementation for such a rule — do NOT mark it PARTIAL/ "
                "NON_COMPLIANT for 'only auditing, not performing the remediation action'. Reserve "
                "PARTIAL/NON_COMPLIANT for 'audit-only, fails loudly' ONLY when the CIS text gives an "
                "EXACT, deterministic, environment-independent command (e.g. 'firewall-cmd "
                "--remove-service=<service>', 'systemctl stop X && dnf remove Y') that the task could "
                "safely have executed using the condition it already detected, but didn't. "
                "CONFIRMED TECHNICAL FACT — when the CIS remediation's own example script for an "
                "auditd rule (auditctl / a line written to /etc/audit/rules.d/*.rules) includes a "
                "specific field (e.g. '-F arch=b64', '-S all'), a generated/existing rule that "
                "reproduces the same '-F path=... -F perm=x -F auid>=... -F auid!=unset -k <key>' "
                "shape but OMITS that field is INCOMPLETE, not just 'differently formatted' — check "
                "each '-F'/'-S' field in the actual rule against what the retrieved CIS example script "
                "for THIS rule specifies, field by field, rather than accepting any rule that merely "
                "resembles the pattern. Confirmed live: rule 6.3.3.10's project rule omitted both "
                "'-F arch=b64' and '-S all' that CIS's own example script includes. "
                "CONFIRMED TECHNICAL FACT — when a CIS remediation's own audit/example script "
                "excludes a SPECIFIC list of accounts (e.g. 'root|halt|sync|shutdown|nfsnobody', or "
                "checks each account's CURRENT shell/lock-state before acting on it), a task that "
                "excludes only PART of that list (e.g. root/sync/shutdown/halt but not nfsnobody), or "
                "acts unconditionally on every account without the same check, is INCOMPLETE — "
                "compare the task's own exclusion list / condition, item by item, against the exact "
                "list or condition in the retrieved CIS text, do not assume they match because the "
                "general shape looks similar. Confirmed live: rule 5.4.2.7's task excluded root/sync/"
                "shutdown/halt but omitted 'nfsnobody' (UID 65534) which CIS explicitly excludes; rule "
                "5.4.2.8's task locked every non-root system account unconditionally, without the "
                "shell-validity check CIS's own script performs first. "
                "CONFIRMED TECHNICAL FACT — when a CIS remediation for a PRIVILEGED shared identifier "
                "(e.g. 'accounts/groups other than root with UID/GID 0') says 'Remove ... OR assign "
                "them a new UID/GID IF APPROPRIATE' — this is a conditional, judgment-dependent "
                "choice (deletion is not always safe: the account/group could be a legitimate, in-use "
                "service account misconfigured with the shared ID rather than an attacker-planted "
                "one). A task using 'state: absent' (user/group module) to UNCONDITIONALLY DELETE "
                "every such account/group it finds, with no way to distinguish 'safe to remove' from "
                "'should be reassigned instead', is a NON_COMPLIANT/risky implementation, NOT a safe "
                "COMPLIANT one — deleting an account outright is more destructive than the situation "
                "may call for, and CIS's own 'if appropriate' phrasing signals this needs judgment, "
                "not blind deletion. Confirmed live: rules 5.4.2.2/5.4.2.3's project task detects "
                "non-root GID-0 accounts/groups correctly but then unconditionally runs 'state: "
                "absent' on all of them. "
                "CONFIRMED TECHNICAL FACT — standard Unix exit-code convention: 'grep -q PATTERN "
                "file' (or similar) exits 0 when the PATTERN WAS FOUND (i.e. the checked condition "
                "IS present/already true), and non-zero when NOT found. When a preceding task "
                "registers such a check and a LATER task is 'when: <var>.rc == 0' to decide whether "
                "to APPLY a fix for something MISSING, that condition is INVERTED — it would only run "
                "the fix when the thing is ALREADY there, and skip it precisely when it's actually "
                "missing (the opposite of the intended effect). Verify the DIRECTION explicitly: a "
                "task that ENABLES/ADDS/FIXES a missing feature must gate on 'rc != 0' (not found -> "
                "needs fixing); a task that REMOVES/DISABLES a present, unwanted setting must gate on "
                "'rc == 0' (found -> needs removing). Read the grep pattern and the task's own stated "
                "purpose together before accepting the condition as correct — do not assume rc==0 "
                "always means 'fix needed' or vice versa. Confirmed live: rule 5.3.1.3's "
                "'Enable authselect with-pwquality feature IF REQUIRED' task uses 'when: "
                "pwquality_with_feature.rc == 0' after a preceding grep FOR 'with-pwquality' — rc==0 "
                "there means the feature IS ALREADY present, so the enable step (meant for when it's "
                "MISSING) fires under exactly the wrong condition. "
                "CONFIRMED TECHNICAL FACT — within a single block/task-family implementing ONE CIS "
                "dconf rule, verify that the dconf DATABASE PATH is used CONSISTENTLY across every "
                "task that writes to it — a value-writing task using a variable (e.g. "
                "'/etc/dconf/db/{{ NAME_OF_DCONF_DATABASE }}.d/...') and a lock-writing task in the "
                "SAME family hardcoding a DIFFERENT literal database name (e.g. '/etc/dconf/db/gdm.d/"
                "locks/...') is a real defect: the lock would apply to a different database than the "
                "one actually holding the value, making the lock ineffective whenever the variable "
                "does not happen to resolve to that same hardcoded name. Confirmed live: rule 1.8.3's "
                "task family sets the idle-delay value via a '{{ NAME_OF_DCONF_DATABASE }}'-templated "
                "path but hardcodes 'gdm.d' for the lock files. "
                "CONFIRMED TECHNICAL FACT — copy-paste mismatch between a task's SEARCH pattern "
                "(regexp:) and its REPLACEMENT value (line:/replace:): when a 'lineinfile'/'replace' "
                "task's 'regexp:' searches for one specific name/parameter (e.g. 'install atm'), "
                "double-check that the 'line:'/'replace:' value it writes uses the SAME name — a task "
                "clearly intended to act on parameter X (per its regexp AND its own task name/rule "
                "title) that actually writes a DIFFERENT parameter Y in its replacement value is a "
                "silent, dangerous copy-paste bug (often from a nearby sibling task for a DIFFERENT "
                "rule that was cloned and only partially edited) — it means the task does NOTHING "
                "useful for the rule it claims to implement, while looking superficially correct. "
                "Confirmed live: rule 3.2.1 ('Ensure atm kernel module is not available')'s task has "
                "'regexp: \"^(#)?install atm(\\\\s|$)\"' (correctly searches for the atm line) but "
                "'line: \"install dccp /bin/false\"' (writes the WRONG module name, dccp instead of "
                "atm) — likely copy-pasted from a neighboring dccp-disabling task without updating "
                "the replacement value. Always cross-check regexp target vs. written value target "
                "for an exact match before crediting COMPLIANT. "
                "CONFIRMED TECHNICAL FACT — authselect profile path must branch custom vs default, "
                "NOT be hardcoded/re-derived inline: authselect profiles live under TWO different "
                "base paths depending on whether the active profile is a 'custom/...' one "
                "(/etc/authselect/<profile>/...) or the far more common built-in/default one "
                "(/usr/share/authselect/default/<profile>/...) — a task that strips a 'custom/' "
                "prefix via regex/string manipulation and then ALWAYS prepends '/etc/authselect/' "
                "(e.g. via 'regex_replace(\"^custom/\", \"\")' or similar) is WRONG for any non-custom "
                "profile: it silently resolves to a path that does not exist, so the edit never "
                "actually happens. If a SIBLING task in the same block already computes this branch "
                "correctly (commonly registered as a fact/variable named like 'pam_profile_path'), a "
                "task that constructs its OWN path inline instead of reusing that fact is NOT "
                "COMPLIANT even if its shell logic looks superficially reasonable — it is solving the "
                "wrong problem (always-custom) instead of the general one. Confirmed live: rule "
                "5.3.2.4.2's 'Remove remember argument' task builds "
                "'l_file=\"/etc/authselect/{{ authselect_profile.stdout | regex_replace(\\'^custom\\/\\', "
                "\\'\\') }}/$l_pam_file\"' — this ALWAYS uses the '/etc/authselect/' base regardless of "
                "whether the active profile is custom or default, even though a sibling task in the "
                "SAME block ('Determine the path of the active profile') already computes the correct "
                "branching path as 'pam_profile_path', which this task ignores. "
                "When the retrieved CIS Benchmark text states a disjunctive requirement using "
                "'EITHER ... OR ...' (or an equivalent 'any ONE of the following') for the SAME "
                "file/directive, crediting COMPLIANT via EITHER option is sufficient — do NOT require "
                "one specific option when the other, equally valid option was chosen. Confirmed live: "
                "rule 1.1.1.1 (and the identical repeated pattern for 1.1.1.2 through 1.1.1.8, plus the "
                "equivalent network-protocol rules 3.2.1-3.2.4) requires 'blacklist <module> -AND "
                "EITHER- install <module> /bin/false -OR- install <module> /bin/true' — a task setting "
                "'install <module> /bin/true' was repeatedly, wrongly marked NON_COMPLIANT for 'using "
                "/bin/true instead of the required /bin/false', even though the retrieved CIS text "
                "itself states both are acceptable. Always check the retrieved text for 'EITHER/OR' "
                "framing before rejecting a choice between two offered alternatives. "
                "When the retrieved CIS Benchmark Remediation text itself defers the corrective action "
                "to human judgment (e.g. 'based on the results of the audit script, determine/"
                "establish/review ... to determine which is correct', 'manually verify/review') rather "
                "than giving one mechanical fix, an AUDIT-only task that correctly detects the "
                "condition and fails loudly (via failed_when/assert on its own register) is the "
                "correct, COMPLETE automated implementation — do NOT require it to also perform the "
                "corrective action itself (e.g. automatically reassigning a UID/renaming an account), "
                "since the benchmark's own text does not define a safe, deterministic way to do so "
                "automatically. Confirmed live: rules 7.2.4/7.2.6/7.2.7 ('no duplicate UIDs/user names/"
                "group names') explicitly instruct 'review all files owned by the shared UIDs to "
                "determine which UID they are supposed to belong to' — a detect-and-fail AUDIT task "
                "was repeatedly marked NON_COMPLIANT for 'only detecting, not implementing the "
                "remediation', even though CIS's own text defers that decision to a human "
                "administrator. "
                "CONFIRMED TECHNICAL FACT — a CIS remediation's 'EITHER/OR' framing can describe "
                "alternative REMEDIATION ACTIONS, not only alternative config VALUES (e.g. 'remove the "
                "package -OR- stop and mask the service' are both valid ways to satisfy 'service not in "
                "use'; 'the file does not exist -OR- contains X' are both valid ways to satisfy an "
                "audit). Do not require one specific action/value when the retrieved CIS text itself "
                "offers it as one of several alternatives and the task correctly implements a "
                "different, equally valid alternative. "
                "CONFIRMED TECHNICAL FACT — this pipeline's own judges have TWICE made an incorrect "
                "permission-math claim (rejecting a correct symbolic chmod as not reaching a target "
                "octal value; rejecting a correct octal mode for not textually matching a symbolic form "
                "used elsewhere). Before asserting a given mode (symbolic OR octal) does or does not "
                "satisfy a required permission level, explicitly compute the resulting permission bits "
                "(owner/group/other read/write/execute) for BOTH forms and compare the actual bits, "
                "never the surface text/format. "
                "CONFIRMED TECHNICAL FACT — the 'ansible.builtin.file' module's 'state:' parameter's "
                "valid values are: absent, directory, file, hard, link, touch. 'state: touch' IS a "
                "valid, correct value (creates an empty file / updates its mtime) — do not reject it "
                "as invalid. "
                "CONFIRMED TECHNICAL FACT — in an authselect custom profile file (under /etc/"
                "authselect/custom/<name>/), the syntax '{include if \"<feature-name>\"}' is "
                "authselect's OWN native conditional-include directive (not Jinja2/Ansible templating) "
                "and is valid, standard syntax — do not flag it as broken or invented Jinja2. "
                "Before finalizing a NON_COMPLIANT or PARTIAL verdict, re-read your OWN stated "
                "reasoning: if your own explanation actually concludes the CIS requirement IS "
                "satisfied, the verdict must be COMPLIANT — never emit a verdict that contradicts your "
                "own stated reasoning. "
                "CONFIRMED against the official CIS Benchmark PDF itself (not just the Remediation "
                "prose excerpt, which may be truncated before the Audit section): for the "
                "kernel-module-disable family (1.1.1.1 cramfs, 1.1.1.2 freevxfs, 1.1.1.3 hfs, 1.1.1.4 "
                "hfsplus, 1.1.1.5 jffs2, 1.1.1.6 squashfs, 1.1.1.7 udf, 1.1.1.8 usb-storage, and the "
                "network-protocol equivalents 3.2.1 dccp, 3.2.2 tipc, 3.2.3 rds, 3.2.4 sctp), the "
                "Benchmark's own AUDIT script explicitly checks "
                "'grep -Pq ...install MODULE /bin/(true|false)...' and its prose states 'An entry "
                "including /bin/true or /bin/false exists' — '/bin/true' and '/bin/false' are BOTH "
                "explicitly, officially compliant values for the 'install MODULE ...' line; the "
                "Remediation section's example merely shows /bin/false as one illustrative choice, it "
                "does not exclude /bin/true. A task using 'install MODULE /bin/true' is NOT "
                "NON_COMPLIANT for that reason — do not flag it. (The upstream ansible-lockdown "
                "RHEL9-CIS project itself uses /bin/true for exactly this family, e.g. "
                "tasks/section_1/cis_1.1.1.x.yml and tasks/section_3/cis_3.2.x.yml.) "
                "CONFIRMED — in the ansible-lockdown RHEL9-CIS role, a per-rule task that only does "
                "'set_fact: rhel9cis_sysctl_update: true' (rules 1.5.1, 1.5.2, 3.3.1-3.3.11) or "
                "'set_fact: update_audit_template: true' (6.3.3.1-6.3.3.19), optionally followed by a "
                "'debug' message, is a DEFERRED-ENFORCEMENT flag, not a no-op — tasks/main.yml always "
                "imports tasks/post.yml, whose 'POST | Update sysctl' task (when: "
                "rhel9cis_sysctl_update) templates /etc/sysctl.d/60-kernel_sysctl.conf, "
                "60-netipv4_sysctl.conf and 60-netipv6_sysctl.conf (each value gated by its own rule "
                "flag) and notifies 'Reload sysctl' (runs sysctl --system); tasks/main.yml also "
                "imports tasks/auditd.yml (when: update_audit_template), which templates "
                "/etc/audit/rules.d/99_auditd.rules and notifies 'augenrules --load'. Any 'ACTUAL "
                "ENFORCEMENT CONTENT' block shown alongside such a task IS its real remediation — do "
                "NOT mark it NON_COMPLIANT for 'only sets a fact / only prints a debug message / does "
                "not write /etc/sysctl.d or /etc/audit/rules.d' when that block is present and "
                "contains the required directive. "
                "CONFIRMED — ansible-lockdown's mount-option tasks (1.1.2.1.2 through 1.1.2.7.4) that "
                "'set_fact: prelim_mount_point_fs_and_options' to append nodev/nosuid/noexec, with "
                "'notify: \"Remount {{ mount_point }}\"', are complete remediations: handlers/main.yml "
                "defines, under 'listen: \"Remount <mount_point>\"', an ansible.posix.mount task with "
                "'state: present' (writes /etc/fstab) followed by one with 'state: remounted' "
                "(applies it live) — together equivalent to 'state: mounted'. Do NOT flag these as "
                "'only updates an in-memory fact / never writes /etc/fstab / never remounts' when a "
                "'WHAT THE NOTIFIED HANDLER(S) ACTUALLY RUN' block confirms this. "
                "CONFIRMED against the official CIS PDF (2.4.2.1 Audit section): /etc/at.allow (and "
                "/etc/at.deny if present) is compliant when owned by root and 'group owned by the "
                "group daemon or group root' — the Audit text itself shows 'Group: (daemon)' -OR- "
                "'Group: (root)' as equally valid. A fix setting 'group: root' is NOT PARTIAL or "
                "NON_COMPLIANT for not using 'daemon' instead. "
                "CONFIRMED against the official CIS PDF (5.4.3.2 Audit script): the audit regex "
                "accepts 'readonly TMOUT' and 'export TMOUT' as separate lines (via "
                "'\\breadonly\\h+TMOUT\\b' / '\\bexport\\b...TMOUT\\b'), in addition to the single-line "
                "'typeset -xr TMOUT=N' form. The 3-line form 'TMOUT=900' / 'readonly TMOUT' / "
                "'export TMOUT' therefore PASSES the benchmark's own audit; the Remediation section's "
                "'deprecated methods' label is a style preference, not a compliance failure — do not "
                "mark it NON_COMPLIANT for using the 3-line form when the value, readonly, and export "
                "are all otherwise correct. "
                "CONFIRMED against the official CIS PDF (5.4.2.4): the remediation is 'passwd root' "
                "(an operator-chosen secret, environment-specific) -OR- 'usermod -L root' (whose own "
                "Impact section warns it can break root-dependent automation) — a site-policy choice, "
                "not a single deterministic command. A task that runs the Audit check itself ('passwd "
                "-S root', accepting either 'Password set' or 'Password locked' as compliant) and "
                "FAILS LOUDLY (assert/failed_when) when neither holds is a complete, honest "
                "implementation — do not mark it NON_COMPLIANT for not itself choosing and applying "
                "one of the two site-policy options. "
                "Always respond with valid JSON only, no markdown."
            )
        },
        {
            "role": "user",
            "content": f"""Analyze this Ansible task for CIS compliance:

TASK:
Name: {task.task_name}
Semantic description: {task.semantic}

Original task RAW YAML (exactly as it appears in the project, including
task-level keys like loop/with_items, failed_when, changed_when, register,
when, ignore_errors — for a shell/command-style check task, the actual
pass/fail LOGIC often lives in failed_when, not in the module's own
arguments, so read the WHOLE task below, not just the module call. If the
task uses a loop, judge compliance against its FULL, real list, not a
guessed/partial one):
{task.raw_yaml}
{_resolved_params_note(task)}
{_template_content_note(task)}
{_rendered_template_note_safe(task.raw_yaml, project_path)}
RELEVANT CIS BENCHMARK RULES:
{context}
{tag_mismatch_note}
IMPORTANT: the rules above were retrieved by TEXT SIMILARITY and may include
topically-related but DISTINCT CIS rules (different rule numbers) alongside
the one this task actually implements — similarity search surfaces neighbors,
it does not mean this ONE task must satisfy all of them. Identify which
rule(s) above this task ACTUALLY implements (put ONLY those in "rule_ids"
below) and judge compliance against THOSE — do not mark this task PARTIAL or
NON_COMPLIANT merely because it doesn't ALSO cover a different, separately-
numbered rule shown above for context; that other rule almost certainly has
its OWN separate task elsewhere in the project (unless the sibling context
below says otherwise for THIS specific rule).
{sibling_block}
{CVSS_METRICS_GUIDE}

Respond ONLY with this JSON structure:
{{
  "status": "COMPLIANT" or "NON_COMPLIANT" or "PARTIAL" or "UNKNOWN",
  "cvss_metrics": {{"AV": "N|A|L|P", "AC": "L|H", "PR": "N|L|H", "UI": "N|R", "S": "U|C", "C": "N|L|H", "I": "N|L|H", "A": "N|L|H"}} or null (null if status is COMPLIANT/UNKNOWN — CVSS only assesses gaps),
  "rule_ids": ["list of matching CIS rule IDs from the context above"],
  "explanation": "brief explanation in English, max 2 sentences, refer to the specific parameter value"
}}"""
        }
    ]

    payload = json.dumps({
        "model": API_MODEL,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        "max_tokens": 550,  # raised from 400 — cvss_metrics adds ~8 key-value pairs to the JSON
        "temperature": 0
    }).encode()

    # Transient server errors (e.g. 502) are retried, so that they are not
    # reported as a mock result. JSON parsing errors are not retried here.
    RETRYABLE_HTTP_CODES = {502, 503, 504}
    max_attempts = 3
    backoff_s = 2
    # DISABLE_MOCK_FALLBACK: when True, this loop keeps retrying until the
    # model responds, so that every result in a model comparison comes from
    # the model under test.
    #
    # Invalid-JSON limit: a response that repeatedly fails to parse is not
    # retried indefinitely. After MAX_INVALID_JSON_RETRIES the task is
    # reported as not audited (UNKNOWN/API_ERROR), without a mock result.
    MAX_INVALID_JSON_RETRIES = 30
    invalid_json_streak = 0
    attempt = 0
    while True:
        attempt += 1
        t0 = time.time()
        try:
            req = urllib.request.Request(
                API_URL,
                data=payload,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {API_TOKEN}",
                    "User-Agent":    "curl/7.88.1"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read())
            # Success is recorded once, after usable content is extracted,
            # so that a malformed response is not counted twice.
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()

            json_str = extract_json_object(text)
            if json_str:
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    _record_request(time.time() - t0, success=False)
                    if DISABLE_MOCK_FALLBACK:
                        print(f"  [A2] Malformed JSON in response (attempt {attempt}) — retrying in {backoff_s}s (mock disabled)...")
                        time.sleep(backoff_s)
                        continue
                    break
                _record_request(time.time() - t0, success=True)
                dump_llm_call("A2_audit", task.task_name, messages, text, result, extra={
                    "retrieved_chunks": [
                        {"rule_id": c.get("rule_id"), "title": c.get("title"), "score": c.get("score")}
                        for c in chunks
                    ]
                })
                status       = result.get("status", "UNKNOWN")
                explanation  = result.get("explanation", "")
                rule_ids     = result.get("rule_ids", [])
                cvss_metrics = result.get("cvss_metrics")

                severity, cvss_vector, cvss_score = "", "", None
                if status in ("NON_COMPLIANT", "PARTIAL"):
                    if isinstance(cvss_metrics, dict):
                        try:
                            cvss_score, severity, cvss_vector = calculate_base_score(
                                cvss_metrics.get("AV", ""), cvss_metrics.get("AC", ""),
                                cvss_metrics.get("PR", ""), cvss_metrics.get("UI", ""),
                                cvss_metrics.get("S", ""), cvss_metrics.get("C", ""),
                                cvss_metrics.get("I", ""), cvss_metrics.get("A", ""),
                            )
                        except (KeyError, ValueError):
                            pass  # invalid letters from the LLM -> fall back below
                    # Without 8 valid CVSS letters, severity stays empty
                    # instead of being guessed.

                cvss_label = f", CVSS {cvss_score} ({cvss_vector})" if cvss_score is not None else ""
                print(f"  [A2] LLM responded: {status}" + (f" ({severity}{cvss_label})" if severity else ""))
                # unknown_reason='LLM_UNCERTAIN' only when the model itself (not
                # mock, not a failure) determined it doesn't know — a real,
                # substantive reason for uncertainty (e.g. no visible sibling for
                # the same rule), distinct from 'network failure' or 'lower-quality
                # mock heuristic'.
                reason = "LLM_UNCERTAIN" if status == "UNKNOWN" else ""
                return status, explanation, rule_ids, severity, reason, cvss_vector, cvss_score
            # Response without valid JSON. The HTTP call itself succeeded
            # (we have a parsed envelope, just no usable {}-delimited
            # content) — recorded as success once, here, since this is a
            # normal fall-through, not an exception path.
            _record_request(time.time() - t0, success=True)
            invalid_json_streak += 1
            if invalid_json_streak >= MAX_INVALID_JSON_RETRIES:
                print(f"  [A2] Response without valid JSON: {invalid_json_streak} consecutive failures "
                      f"for this task — giving up (marking as not actually audited) instead of "
                      f"retrying indefinitely")
                return ("UNKNOWN", "LLM response never contained valid JSON after "
                        f"{invalid_json_streak} consecutive attempts.", [], "", "API_ERROR", None, None)
            # Normally there's no point retrying
            # an identical request (deterministic, temp=0) — but with
            # DISABLE_MOCK_FALLBACK we try anyway (non-determinism has been
            # observed in practice despite temp=0, e.g. under server load).
            if DISABLE_MOCK_FALLBACK:
                print(f"  [A2] Response without valid JSON (attempt {attempt}, "
                      f"{invalid_json_streak}/{MAX_INVALID_JSON_RETRIES} consecutive) — "
                      f"retrying in {backoff_s}s (mock disabled)...")
                time.sleep(backoff_s)
                continue
            break

        except urllib.error.HTTPError as e:
            _record_request(time.time() - t0, success=False)
            if DISABLE_MOCK_FALLBACK or (e.code in RETRYABLE_HTTP_CODES and attempt < max_attempts):
                print(f"  [A2] Error {e.code} (attempt {attempt}) — retrying in {backoff_s}s" + (" (mock disabled)..." if DISABLE_MOCK_FALLBACK else "..."))
                time.sleep(backoff_s)
                continue
            print(f"  [A2] API connection error: {e} — using mock")
            break
        except urllib.error.URLError as e:
            _record_request(time.time() - t0, success=False)
            if DISABLE_MOCK_FALLBACK or attempt < max_attempts:
                print(f"  [A2] Connection error (attempt {attempt}): {e} — retrying in {backoff_s}s" + (" (mock disabled)..." if DISABLE_MOCK_FALLBACK else "..."))
                time.sleep(backoff_s)
                continue
            print(f"  [A2] API connection error: {e} — using mock")
            break
        except json.JSONDecodeError as e:
            _record_request(time.time() - t0, success=False)
            if DISABLE_MOCK_FALLBACK:
                print(f"  [A2] JSON parsing error from LLM (attempt {attempt}): {e} — retrying in {backoff_s}s (mock disabled)...")
                time.sleep(backoff_s)
                continue
            print(f"  [A2] JSON parsing error from LLM: {e} — using mock")
            break
        except Exception as e:
            _record_request(time.time() - t0, success=False)
            if DISABLE_MOCK_FALLBACK:
                print(f"  [A2] Unexpected error (attempt {attempt}): {e} — retrying in {backoff_s}s (mock disabled)...")
                time.sleep(backoff_s)
                continue
            print(f"  [A2] Unexpected error: {e} — using mock")
            break

    # We only get here when the real LLM call failed (after exhausting
    # retries) — if the mock ALSO returns UNKNOWN, we overwrite the reason
    # with 'API_ERROR' (this is more accurate, actionable information than
    # 'low score' — it says "this was NOT actually audited", not "nothing
    # matched").
    mock_result = mock_verify(task, chunks)
    if mock_result[0] == "UNKNOWN":
        return (mock_result[0], mock_result[1], mock_result[2], mock_result[3],
                "API_ERROR", mock_result[5], mock_result[6])
    return mock_result


def mock_verify(task: TaskDescription, chunks: list) -> tuple:
    """Returns a 7-tuple (status, explanation, rule_ids, severity, unknown_reason,
    cvss_vector, cvss_score). The mock does NOT compute CVSS (it has no 8
    metrics from the LLM, only a keyword heuristic) — cvss_vector/cvss_score
    are always empty/None. unknown_reason only makes sense when
    status=='UNKNOWN' — here it's always 'LOW_CONFIDENCE' (score-based
    heuristic, not a network failure; llm_verify() overwrites this to
    'API_ERROR' when the mock was used AFTER failed attempts at a real LLM
    call, see there)."""
    if not chunks:
        return "UNKNOWN", "No matching CIS rules found.", [], "", "LOW_CONFIDENCE", "", None

    best  = chunks[0]
    score = best["score"]
    task_text = (task.semantic + " " + str(task.params)).lower()
    rule_text = (best["title"] + " " + best["text"][:300]).lower()

    negative = [
        "yes" in task_text and "no" in rule_text,
        "disabled" in rule_text and "enable" in task_text,
    ]

    if score < 0.4:
        return "UNKNOWN", f"Low score ({score:.2f}) — no match.", [], "", "LOW_CONFIDENCE", "", None
    elif any(negative):
        # severity intentionally empty — the mock no longer has a fallback
        # keyword heuristic (removed, see comment above this function); this
        # is a rare, degraded path anyway (LLM unavailable/disabled), so
        # honestly unknown is better than false precision.
        return "NON_COMPLIANT", (
            f"Task likely non-compliant with rule {best['rule_id']}. "
            "Conflicting parameter values detected."
        ), [best["rule_id"]], "", "", "", None
    elif score > 0.7:
        return "COMPLIANT", (
            f"Task complies with rule {best['rule_id']}: {best['title'][:55]}. "
            f"Score={score:.2f}."
        ), [best["rule_id"]], "", "", "", None
    else:
        return "PARTIAL", (
            f"Task partially matches rule {best['rule_id']}. "
            f"Manual verification required (score={score:.2f})."
        ), [best["rule_id"]], "", "", "", None


def assess_missing_rule_cvss(rule_id: str, rule_title: str, rule_text: str):
    """The same mechanism as llm_verify() (the LLM assesses 8 CVSS metrics,
    cvss_calculator.py computes the score), for a rule that has no task in
    the project (CREATE in Agent 3), based on the rule text only.
    Returns (severity: str, cvss_vector: str, cvss_score: Optional[float]),
    or ("", "", None) after 2 failed attempts.

    Known limitation: for missing preventive controls (e.g. "Ensure
    firewalld is installed") the model may assign maximal metrics. CVSS was
    designed for vulnerabilities with a clear exploitation path, and there
    is no external reference to calibrate such cases, so no extra prompt
    tuning was applied."""
    if not USE_LLM:
        return "", "", None

    messages = [
        {"role": "system", "content": (
            "You are a CIS Benchmark compliance auditor assessing the severity "
            "of a completely missing security control. Always respond with valid JSON only, no markdown."
        )},
        {"role": "user", "content": f"""This CIS rule has NO implementation at all in the project (complete absence):

CIS {rule_id}: {rule_title}
Rule text: {rule_text[:1000]}

{CVSS_METRICS_GUIDE}

Respond ONLY with this JSON structure:
{{
  "cvss_metrics": {{"AV": "N|A|L|P", "AC": "L|H", "PR": "N|L|H", "UI": "N|R", "S": "U|C", "C": "N|L|H", "I": "N|L|H", "A": "N|L|H"}},
  "explanation": "brief justification in English, max 1 sentence"
}}"""},
    ]
    payload = json.dumps({
        "model": API_MODEL,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0,
    }).encode()
    for attempt in range(1, 3):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {API_TOKEN}",
                         "User-Agent": "curl/7.88.1"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read())
            # Recorded once below, only once we have a usable result — see
            # the identical fix (and rationale) in llm_verify() above.
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
            json_str = extract_json_object(text)
            if not json_str:
                _record_request(time.time() - t0, success=True)
                continue
            result = json.loads(json_str)
            m = result.get("cvss_metrics")
            if not isinstance(m, dict):
                _record_request(time.time() - t0, success=True)
                continue
            try:
                cvss_score, severity, cvss_vector = calculate_base_score(
                    m.get("AV", ""), m.get("AC", ""), m.get("PR", ""), m.get("UI", ""),
                    m.get("S", ""), m.get("C", ""), m.get("I", ""), m.get("A", ""),
                )
            except (KeyError, ValueError):
                _record_request(time.time() - t0, success=True)
                continue
            _record_request(time.time() - t0, success=True)
            dump_llm_call("A2_missing_rule_cvss", rule_id, messages, text,
                          {"severity": severity, "cvss_vector": cvss_vector, "cvss_score": cvss_score})
            return severity, cvss_vector, cvss_score
        except Exception as e:
            _record_request(time.time() - t0, success=False)
            print(f"  [A2] Error assessing CVSS for missing rule {rule_id} "
                  f"(attempt {attempt}/2): {e}")
    return "", "", None


# No audit result cache: the model server already caches identical
# requests, and a cache keyed without the model name would mix results
# of different models in comparisons.


# ── Main function ────────────────────────────────────────────────────────────
# disabled_rules: rule IDs accepted as a risk in Agent A1
# (run_onboarding(...)["disabled_rules"]). Tasks of these rules are not sent
# to the model.
#
# Read-only tasks are not skipped: a check task can itself contain a bug
# (e.g. a missing `-v` in grep). The model sees the sibling tasks of the
# same rule instead.
def _pre_audit_skip(task: TaskDescription, disabled_rules: List[str]) -> Optional[ComplianceResult]:
    """Decision BEFORE the audit: whether this task should go through
    retrieval+LLM at all, or whether it can be decided without that.
    Deliberately separate from audit_task() (which now assumes the audit DOES
    make sense) — audit_tasks() asks this first, for every task, before even
    considering caching or calling audit_task(). Two reasons for a skip:
      1. RISK_ACCEPTED — rule intentionally disabled in Agent 1
      2. no mapped CIS rule — infrastructure/helper tooling
    """
    if task.cis_rule_ids and any(rid in disabled_rules for rid in task.cis_rule_ids):
        print(f"  [A2] ⏭️  Skipping LLM verification: rule intentionally disabled (Risk Accepted)")
        return ComplianceResult(
            task_name=task.task_name,
            status="RISK_ACCEPTED",
            rule_ids=task.cis_rule_ids,
            explanation="Task tied to a rule that was intentionally disabled in the project's configuration variables (e.g. rule_X: false)."
        )

    if not task.cis_rule_ids:
        print(f"  [A2] ⏭️  Skipping LLM verification: task is not mapped to any CIS rule")
        return ComplianceResult(
            task_name=task.task_name,
            status="UNKNOWN",
            unknown_reason="NOT_MAPPED",
            explanation="Task has no CIS rule mapped to it (no tag/when/name pointing "
                        "to rule_X). This does NOT automatically mean 'irrelevant' — it could be "
                        "infrastructure/helper tooling (noise), BUT equally it could be an extra "
                        "safeguard outside the scope of the CIS Benchmark, which simply has nothing "
                        "to compare it against. In both cases, a compliance assessment does not apply "
                        "here — there is no reference rule."
        )

    return None


def audit_task(task: TaskDescription, disabled_rules: List[str] = None,
              sibling_tasks: List[str] = None, manual_control: bool = False,
              project_path: str = "") -> ComplianceResult:
    """Assumes the task has ALREADY passed _pre_audit_skip() (no reason to
    skip it) — always performs a full retrieval + LLM call.
    sibling_tasks: names of other tasks mapped to the same CIS rule (grouped
    by TAG — see audit_tasks() for why content-based grouping was tried and
    reverted) — see llm_verify().
    manual_control: task is tied to a CIS rule marked "(Manual)" in the
    Benchmark. Still fully audited (the LLM verdict is informational and
    worth having), just tagged so the report can keep it out of the scored
    compliance % — see ComplianceResult.manual_control."""
    disabled_rules = disabled_rules or []
    print(f"  [A2] Auditing: '{task.task_name}' (module: {task.module})")

    query = task.semantic or f"{task.module} {' '.join(str(v) for v in task.params.values())}"
    chunks, ret_ms = retrieve(query)

    print(f"  [A2] Retrieval: {len(chunks)} chunks in {ret_ms:.1f}ms")
    for c in chunks[:3]:
        print(f"       score={c['score']:.3f}  rule={c['rule_id']}  {c['title'][:50]}")

    # Cross-check, not a gate: a mismatch between the task's declared rule
    # and the top retrieved rule is logged for human review.
    if chunks and chunks[0]["rule_id"] not in task.cis_rule_ids:
        print(f"  [A2] ⚠ tag/name says {task.cis_rule_ids}, content match says "
              f"'{chunks[0]['rule_id']}' (score={chunks[0]['score']:.3f}) — disagreement, worth a manual look")

    result_llm = llm_verify(task, chunks, sibling_tasks=sibling_tasks, project_path=project_path)
    status         = result_llm[0]
    explanation    = result_llm[1]
    llm_ids        = result_llm[2] if len(result_llm) > 2 else []
    severity       = result_llm[3] if len(result_llm) > 3 else ""
    unknown_reason = result_llm[4] if len(result_llm) > 4 else ""
    cvss_vector    = result_llm[5] if len(result_llm) > 5 else ""
    cvss_score     = result_llm[6] if len(result_llm) > 6 else None

    rule_ids    = llm_ids if llm_ids else [c["rule_id"] for c in chunks[:3]]
    # Each rule ID gets its own title, looked up among the retrieved chunks
    # by rule ID (not by chunk position). Falls back to the bare rule ID.
    _title_by_id = {c["rule_id"]: c["title"] for c in chunks}
    rule_titles = [_title_by_id.get(rid, rid) for rid in rule_ids]

    icon = {"COMPLIANT": "✓", "NON_COMPLIANT": "✗", "PARTIAL": "~", "UNKNOWN": "?", "RISK_ACCEPTED": "⏭️"}.get(status, "?")
    sev_label = f" [{severity}" + (f", CVSS {cvss_score}" if cvss_score is not None else "") + "]" if severity else ""
    manual_label = " [MANUAL — informational only, not scored]" if manual_control else ""
    print(f"  [A2] {icon} {status}{sev_label}{manual_label}: {explanation[:80]}")

    return ComplianceResult(
        task_name=task.task_name,
        status=status,
        severity=severity,
        unknown_reason=unknown_reason,
        cvss_vector=cvss_vector,
        cvss_score=cvss_score,
        rule_ids=rule_ids,
        rule_titles=rule_titles,
        explanation=explanation,
        retrieved_chunks=chunks,
        retrieval_ms=ret_ms,
        manual_control=manual_control,
    )


# Per-task watchdog: a task that hangs outside an HTTP call (e.g. in the
# embedding step) is stopped. TASK_WATCHDOG_MIN applies to the first tasks,
# later tasks are bounded by TASK_WATCHDOG_MULTIPLIER x the rolling average.
TASK_WATCHDOG_MIN = 300.0
TASK_WATCHDOG_MULTIPLIER = 15.0
TASK_WATCHDOG_WINDOW = 20  # rolling average over the last N completed tasks


def audit_tasks(tasks: List[TaskDescription], disabled_rules: List[str] = None,
                 manual_rule_ids: set = None, project_path: str = "") -> List[ComplianceResult]:
    disabled_rules = disabled_rules or []
    manual_rule_ids = manual_rule_ids or set()
    results = []

    # Map rule_id -> INDICES (into `tasks`) of all tasks mapped to that rule
    # across the whole project — needed to give the LLM the context "this
    # task is part of a multi-step implementation together with...", see
    # llm_verify(). Built once, before the loop, not per-task.
    #
    # Siblings are indexed by task position, not by name, because different
    # tasks of the same rule can share an identical name.
    #
    # Siblings are grouped by the project's rule tag, not by content
    # similarity. Content-based grouping was tested and performed worse:
    # auxiliary tasks with low similarity drifted to unrelated rules.
    tasks_by_rule: Dict[str, List[int]] = {}
    for idx, t in enumerate(tasks):
        for rid in t.cis_rule_ids:
            tasks_by_rule.setdefault(rid, []).append(idx)

    task_times: List[float] = []  # completed-task wall-clock times, for the rolling average/watchdog

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError

    for i, task in enumerate(tasks, 1):
        print(f"\n[A2] Task {i}/{len(tasks)}:")

        skip_result = _pre_audit_skip(task, disabled_rules)
        if skip_result:
            results.append(skip_result)
            continue

        this_idx = i - 1
        # Built from task indices, not from a set of names, so that two
        # different tasks with the same name are both shown.
        #
        # Siblings are shown as full raw YAML (including register/when/
        # failed_when), the same as in Agents 3 and 4.
        sibling_idxs = sorted({
            idx for rid in task.cis_rule_ids
            for idx in tasks_by_rule.get(rid, [])
            if idx != this_idx
        })
        siblings = [
            f"{tasks[idx].task_name}:\n"
            + "\n".join(f"    {line}" for line in
                        (tasks[idx].raw_yaml or f"(module: {tasks[idx].module}, params: {json.dumps(tasks[idx].params)})").split("\n"))
            for idx in sibling_idxs
        ]
        is_manual = any(rid in manual_rule_ids for rid in task.cis_rule_ids)

        # The first completed task is excluded from the average (model
        # loading on the server makes it slower).
        times_excl_first = task_times[1:] if len(task_times) > 1 else []
        recent = times_excl_first[-TASK_WATCHDOG_WINDOW:]
        rolling_avg = sum(recent) / len(recent) if recent else None
        watchdog_timeout = max(TASK_WATCHDOG_MIN, TASK_WATCHDOG_MULTIPLIER * rolling_avg) if rolling_avg else TASK_WATCHDOG_MIN

        t0 = time.time()
        # A fresh, single-use executor per task (not a shared pool) — a
        # shared max_workers=1 pool would let one abandoned, still-running
        # timed-out task permanently block every task submitted after it,
        # defeating the whole point of the watchdog.
        task_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"a2-watchdog-{i}")
        future = task_executor.submit(audit_task, task, disabled_rules, siblings, is_manual, project_path)
        try:
            result = future.result(timeout=watchdog_timeout)
            task_executor.shutdown(wait=False)
            elapsed = time.time() - t0
            task_times.append(elapsed)
            times_excl_first_now = task_times[1:] if len(task_times) > 1 else []
            if times_excl_first_now:
                avg_so_far = sum(times_excl_first_now) / len(times_excl_first_now)
                print(f"  [A2] Task time: {elapsed:.1f}s (avg excl. first over {len(times_excl_first_now)} "
                      f"task(s): {avg_so_far:.1f}s)")
            else:
                print(f"  [A2] Task time: {elapsed:.1f}s (first task — excluded from the running average, "
                      f"typically slower due to model warm-up)")
        except _FutureTimeoutError:
            task_executor.shutdown(wait=False)
            elapsed = time.time() - t0
            avg_note = f"{rolling_avg:.1f}s" if rolling_avg is not None else "n/a (too early to have an average)"
            print(f"  [A2] WATCHDOG: task exceeded {watchdog_timeout:.0f}s (no response after {elapsed:.1f}s, "
                  f"vs. recent average {avg_note}) — treating as a stalled infrastructure call, "
                  f"NOT a model-quality signal, and moving on. The stuck worker thread is abandoned (Python "
                  f"cannot forcibly cancel a blocking call) — it may still be consuming a connection in the "
                  f"background, harmless to the rest of this run.")
            result = ComplianceResult(
                task_name=task.task_name, status="UNKNOWN", unknown_reason="API_ERROR",
                rule_ids=list(task.cis_rule_ids),
                explanation=f"Task audit exceeded the {watchdog_timeout:.0f}s watchdog timeout with no response "
                            f"(infrastructure stall, not evaluated by the model — a retry is needed for a real verdict).",
            )
            # Deliberately NOT appended to task_times — an unresolved call has
            # no real duration to average, and would otherwise poison every
            # subsequent watchdog_timeout calculation upward.
        results.append(result)

    return results
"""

Completes the architecture matching the diagram:
  Agent 1 (Parser) -> Agent 2 (RAG Auditor) -> [non-compliances] -> Agent 3 (Remediation)

Input source: Parser A0 (CoverageAnalyzer, function generate_review_file) —
MISSING_TASK rules (enabled in defaults/, but with no task in the playbook) and
high-risk DISABLED_BY_VAR rules (disabled manually, but covering a critical
area per RiskClassifier). By definition these rules have NO task to audit
(that's the whole point — there isn't one), so they don't go through the
regular audit_task(). Instead, A2 treats each of them as a ready-made
"non-compliance" and passes it directly to Agent 3 — the code is generated
"just in case", for manual review, never deployed automatically.

IMPORTANT: Agent A1 (onboarding) does NOT feed this function. When the user
consciously accepts the risk (RISK_ACCEPTED) during onboarding for an
undefined/disabled rule, that decision is terminal — the rule stays as it
is in defaults/, with no further automatic "just in case" code generation
by A2/A3. Overriding a manually accepted risk would undermine the whole
point of the onboarding step. See agent_a1_onboarding.py.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class MissingRuleDecision:
    """Agent A2's decision on what to do with a rule that has no implementation."""
    rule_id:        str
    title:          str
    coverage_status: str   # MISSING_TASK / DISABLED_BY_VAR_HIGH_RISK
    action:         str    # SEND_TO_A3 / SKIP
    reason:         str


def audit_missing_rules(coverage_gaps: List[Dict[str, Any]]) -> List[MissingRuleDecision]:
    """
    Agent A2 — audit of rules with no implementation (data from Parser A0:
    CoverageAnalyzer, generate_review_file).

    Input: a list of dicts from A0, each containing at least
           {"rule_id": str, "title": str, "status": "MISSING_TASK"|"DISABLED_BY_VAR", ...}

    Decision logic:
      - MISSING_TASK: the rule is enabled but has no task -> SEND_TO_A3
      - DISABLED_BY_VAR with high review priority (Parser A0's
        RiskClassifier) -> SEND_TO_A3, for manual review.

    Returns a list of decisions. Decisions with action=SEND_TO_A3 are
    passed to Agent A3's generate_missing_tasks().
    """
    SEND_STATUSES = {"MISSING_TASK", "DISABLED_BY_VAR"}
    REASONS = {
        "MISSING_TASK": "Rule is enabled (rule=true) but has no implementation — "
                         "an unambiguous non-compliance, a task needs to be generated.",
        "DISABLED_BY_VAR": "High-risk rule disabled manually — generates code "
                            "for review, does not deploy automatically.",
    }
    decisions = []

    for gap in coverage_gaps:
        rule_id = gap.get("rule_id", "UNKNOWN")
        title   = gap.get("title", "")
        status  = gap.get("status", "")

        if status in SEND_STATUSES:
            decisions.append(MissingRuleDecision(
                rule_id=rule_id, title=title, coverage_status=status,
                action="SEND_TO_A3", reason=REASONS[status],
            ))
        else:
            decisions.append(MissingRuleDecision(
                rule_id=rule_id, title=title, coverage_status=status,
                action="SKIP",
                reason="Rule outside the scope of the A2 audit (e.g. low risk)."
            ))

    return decisions


def audit_and_remediate_missing_rules(coverage_gaps: List[Dict[str, Any]],
                                       use_llm: bool = True) -> tuple:
    """
    Full A2 -> A3 flow for rules with no implementation.
    1. A2 (audit_missing_rules) decides which rules need a fix
    2. For every SEND_TO_A3 decision, A2 passes it to A3
       (generate_missing_tasks) to generate the code

    Returns (decisions, remediations) — A2's decisions and the fixes
    generated by A3.
    """
    from agent_a3_llm import generate_missing_tasks

    print(f"\n[A2] Auditing {len(coverage_gaps)} rules with no full implementation...")
    decisions = audit_missing_rules(coverage_gaps)

    to_send = [d for d in decisions if d.action == "SEND_TO_A3"]
    print(f"[A2] {len(to_send)}/{len(decisions)} rules passed to Agent 3")

    # Build the input for A3: it needs rule_id, title, text (CIS chunk)
    a3_input = []
    col = get_collection()
    for d in to_send:
        try:
            res = col.get(where={"rule_id": d.rule_id}, include=["documents"])
        except Exception as e:
            print(f"  [A2] ChromaDB error while fetching the chunk for {d.rule_id}: {e} — skipping")
            continue
        if res["documents"]:
            a3_input.append({
                "rule_id": d.rule_id,
                "title":   d.title,
                "text":    "\n".join(res["documents"]),
            })
        else:
            print(f"  [A2] No CIS chunk for {d.rule_id} in ChromaDB — skipping")

    print(f"\n[A2] Passing {len(a3_input)} rules to Agent 3 (generate_missing_tasks)...")
    remediations = generate_missing_tasks(a3_input, use_llm=use_llm)

    return decisions, remediations

# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TEST of Agent A2 with the university LLM ===\n")
    print(f"Model: {API_MODEL}")
    print(f"API:   {API_URL}")
    print(f"LLM:   {'ENABLED' if USE_LLM else 'MOCK'}\n")

    test_tasks = [
        TaskDescription(
            task_name="Set SSH MaxAuthTries",
            module="lineinfile",
            params={"dest": "/etc/ssh/sshd_config.d/hardening.conf", "line": "MaxAuthTries 4"},
            semantic="Ensure sshd MaxAuthTries is configured, modifying parameter /etc/ssh/sshd_config MaxAuthTries 4",
            cis_rule_ids=["5.2.2"]
        ),
        TaskDescription(
            task_name="Disable cramfs kernel module",
            module="modprobe",
            params={"name": "cramfs", "state": "absent"},
            semantic="Ensure cramfs kernel module is not available modprobe cramfs absent",
            cis_rule_ids=["1.1.1.1"]
        )
    ]

    # Simulation: rule 1.1.1.1 was disabled in vars/defaults by the user
    mock_disabled_rules = ["1.1.1.1"]

    results = audit_tasks(test_tasks, disabled_rules=mock_disabled_rules)

    print("\n" + "="*65)
    print("SUMMARY REPORT")
    print("="*65)
    counts = {"COMPLIANT": 0, "NON_COMPLIANT": 0, "PARTIAL": 0, "UNKNOWN": 0, "RISK_ACCEPTED": 0}
    severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

    for r in results:
        icon = {"COMPLIANT": "✓", "NON_COMPLIANT": "✗", "PARTIAL": "~", "UNKNOWN": "?", "RISK_ACCEPTED": "⏭️"}.get(r.status, "?")
        sev_label = f" [{r.severity}]" if r.severity else ""
        print(f"\n{icon} [{r.status:<14}]{sev_label} {r.task_name}")
        print(f"   CIS rules  : {', '.join(r.rule_ids[:3])}")
        if r.status != "RISK_ACCEPTED":
            print(f"   Retrieval  : {r.retrieval_ms:.0f}ms")
        print(f"   Explanation: {r.explanation}")
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.severity:
            severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

    print(f"\nStatus summary: {counts}")
    print(f"Severity summary (NON_COMPLIANT/PARTIAL only): {severity_counts}")
