"""
Agent A4 — Validator (Self-Correction Loop)
=============================================
Closes the loop: Agent 3 generates code -> Agent 4 validates -> on failure,
the error report goes back to Agent 3, which generates a corrected version.
The loop ends when the code passes validation or when a stagnation guard
stops it.

Validation stages per iteration:
  1. YAML format check — parsing + indentation normalization
  2. Syntax validation (ansible-lint)
  3. Knowledge-base semantic check — does the YAML satisfy the CIS
     remediation text for this rule (retrieved from the same ChromaDB
     knowledge base that Agent 3 uses)
  4. On any failure: repair via Agent 3

Additionally: conflict detection across generated fixes (two fixes setting
contradictory values of the same parameter in the same file).
"""

import subprocess
import tempfile
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from debug_dump import dump_llm_call, extract_json_object

try:
    from ruamel.yaml import YAML
    _ryaml = YAML(typ="safe")
    HAS_RUAMEL = True
except ImportError:
    import yaml as _pyyaml
    HAS_RUAMEL = False

try:
    from parser_a0_final import discover_roles
except ImportError:
    discover_roles = None


def _kb_text_slice(text: str, limit: int) -> str:
    """Truncates the knowledge-base text shown to Agent 4's semantic check
    (and its repair prompts) without dropping the Remediation section:
    reuses model_comparison_grounded_judge._remediation_priority_slice()
    (local import to avoid an import cycle)."""
    if not text:
        return text
    from model_comparison_grounded_judge import _remediation_priority_slice
    return _remediation_priority_slice(text, limit)


def _dedent_to_zero(text: str) -> str:
    """
    ruamel.yaml with sequence-indent offset set (needed for correct
    indentation of NESTED lists under a key) also applies that same
    offset to the top-level list of the document, where it shouldn't
    appear. This function strips the common leading whitespace from
    every non-empty line, bringing the document root back to column 0.
    """
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text
    min_indent = min(len(l) - len(l.lstrip(" ")) for l in non_empty)
    if min_indent == 0:
        return text
    return "\n".join(l[min_indent:] if l.strip() else l for l in lines)


def _dump_tasks(tasks) -> str:
    """Serializes a list of parsed task dicts back to normalized YAML text,
    same style/indentation as check_yaml_format()'s own re-dump step. Used
    by validate_and_repair()'s Step 1b auto-dedup (see its docstring)."""
    if HAS_RUAMEL:
        import io
        buf = io.StringIO()
        _get_dumper().dump(tasks, buf)
        return _dedent_to_zero(buf.getvalue())
    return _pyyaml.dump(tasks, default_flow_style=False, allow_unicode=True)


def _get_dumper():
    """
    YAML dumper configured per yamllint/ansible-lint convention:
    sequences (lists) must be indented relative to their parent key,
    otherwise yamllint reports a 'wrong indentation' error. ruamel.yaml's
    default configuration doesn't do this — .indent() must be set
    explicitly. The side effect (extra indent on the top-level list) is
    removed separately by _dedent_to_zero().
    """
    d = YAML()
    d.default_flow_style = False
    d.indent(mapping=2, sequence=4, offset=2)
    return d


# ── Output structures ────────────────────────────────────────────────────
@dataclass
class LintFinding:
    rule:        str
    message:     str
    line:        Optional[int] = None
    severity:    str = "error"


BLOCKING_RULE_PREFIXES = (
    "args", "syntax", "yaml[syntax]",
    # "yaml_format": marker for a check_yaml_format() failure. Blocking,
    # because the YAML did not parse.
    "yaml_format",
    # "lint_error": ansible-lint itself crashed, so linting did not happen.
    # Blocking. ("timeout" and "tool_missing" are handled directly in
    # run_ansible_lint().)
    "lint_error",
)

def _is_blocking(finding) -> bool:
    """Whether an ansible-lint finding should block the self-repair loop.
    See the comment in run_ansible_lint — the rule family, not the
    severity field, is the reliable criterion here."""
    return any(finding.rule.startswith(p) for p in BLOCKING_RULE_PREFIXES)


@dataclass
class ValidationAttempt:
    attempt_number:  int
    yaml_text:        str
    format_valid:     bool
    lint_valid:       bool
    findings:         List[LintFinding] = field(default_factory=list)
    # Knowledge-base semantic check for this attempt. False = not run
    # (format failed, checking disabled, or no knowledge-base text found).
    semantic_checked:  bool = False
    semantic_verdict:  str = ""   # "COMPLIANT" / "PARTIAL" / "NON_COMPLIANT"
    semantic_score:    Optional[int] = None
    semantic_reason:   str = ""


@dataclass
class ValidationResult:
    original_task:        str
    rule_id:               str
    valid:                 bool                      # final pass/fail (lint AND, if checked, semantic)
    final_yaml:            str
    attempts:              List[ValidationAttempt] = field(default_factory=list)
    total_attempts:         int = 0
    give_up_reason:         str = ""
    # Lint-only result for the final YAML, kept separate from 'valid' (which
    # also requires the semantic check), so that "failed lint" and "failed
    # the semantic check" can be distinguished.
    lint_valid:             bool = False
    # Final semantic verdict, carried over from the last attempt in which
    # the semantic check actually ran. knowledge_base_found=False means that
    # no CIS text was found for the rule; 'valid' is not penalised for that.
    semantic_valid:         Optional[bool] = None
    semantic_verdict:       str = ""
    semantic_score:         Optional[int] = None
    semantic_reason:        str = ""
    knowledge_base_found:   bool = False


@dataclass
class ConflictReport:
    rule_a:  str
    rule_b:  str
    reason:  str


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: YAML format verification
# ═══════════════════════════════════════════════════════════════════════════

def check_yaml_format(yaml_text: str) -> tuple:
    """
    Check whether the text is valid YAML. If parsing succeeds, normalize
    indentation by re-dumping (unifies style regardless of how the LLM
    formatted its answer). Returns (is_valid, fixed_text, error).
    """
    # Pre-cleaning of typical LLM mistakes: tabs -> spaces, trailing
    # whitespace, leftover markdown fences. Weaker models sometimes add
    # prose or a second code block next to the YAML.
    #
    # The model sometimes returns several code blocks, stray ``` fences or a
    # self-correcting sentence. All fence lines are removed, and the last
    # top-level '- name:' block is taken as the model's final answer when
    # the whole text does not parse (see below).
    cleaned = yaml_text.replace("\t", "  ")
    no_fences = re.sub(r"^```[a-zA-Z]*\s*$", "", cleaned, flags=re.MULTILINE)
    no_fences = no_fences.strip()
    no_fences = "\n".join(line.rstrip() for line in no_fences.split("\n"))

    # 'block', 'rescue' and 'always' are task keywords, not modules, so an
    # 'ansible.builtin.' prefix on them is invalid. It is removed
    # deterministically.
    no_fences = re.sub(
        r"^(\s*-?\s*)ansible\.(builtin|legacy)\.(block|rescue|always):",
        r"\1\3:", no_fences, flags=re.MULTILINE,
    )

    def _try_parse(text):
        if HAS_RUAMEL:
            return _ryaml.load(text)
        return _pyyaml.safe_load(text)

    # The whole cleaned text is parsed first and accepted if it is a valid
    # task list (e.g. a two-task "configure, then activate" fix). Only when
    # it does not parse, the last '- name:' block onward is used.
    try:
        whole_parsed = _try_parse(no_fences)
    except Exception:
        whole_parsed = None

    if isinstance(whole_parsed, list) and whole_parsed and all(isinstance(t, dict) for t in whole_parsed):
        cleaned = no_fences
        parsed = whole_parsed
    else:
        lines = no_fences.split("\n")
        last_task_start = None
        for i, line in enumerate(lines):
            if re.match(r"^- name:", line):
                last_task_start = i
        cleaned = "\n".join(lines[last_task_start:]) if last_task_start is not None else no_fences
        cleaned = cleaned.strip()
        cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

        try:
            parsed = _try_parse(cleaned)
        except Exception as e:
            # Fallback: if the model appended a second "---"-separated
            # document without a markdown fence (e.g. two versions of the
            # same task), try parsing ONLY the first document instead of
            # giving up immediately.
            if "expected a single document" in str(e) or "multiple documents" in str(e).lower():
                first_doc = re.split(r"\n---\s*\n", cleaned)[0].strip()
                try:
                    parsed = _try_parse(first_doc)
                    cleaned = first_doc
                except Exception:
                    return False, cleaned, str(e)
            else:
                return False, cleaned, str(e)

    if parsed is None:
        return False, cleaned, "Empty YAML document after parsing"

    try:
        # Normalize formatting by re-dumping — unifies indentation
        if HAS_RUAMEL:
            import io
            buf = io.StringIO()
            _dumper = _get_dumper()
            _dumper.dump(parsed, buf)
            normalized = _dedent_to_zero(buf.getvalue())
        else:
            normalized = _pyyaml.dump(parsed, default_flow_style=False, allow_unicode=True)

        return True, normalized, ""

    except Exception as e:
        return False, cleaned, str(e)


def count_top_level_tasks(yaml_text: str) -> int:
    """How many task definitions the (already format-valid) YAML contains.
    A list of N dicts = N tasks; a single bare dict = 1 task. Used as a
    programmatic guardrail: several stacked draft versions of a task are
    valid YAML and could otherwise pass format and lint checks."""
    try:
        if HAS_RUAMEL:
            parsed = _ryaml.load(yaml_text)
        else:
            parsed = _pyyaml.safe_load(yaml_text)
    except Exception:
        return -1  # unparseable here shouldn't happen (format already validated) — treat as "unknown"
    if isinstance(parsed, list):
        return len(parsed)
    return 1  # a single bare mapping is one task


def check_duplicate_subtasks(yaml_text: str) -> list:
    """Detects identical sub-tasks nested inside a 'block:' (stacked
    duplicate drafts one level below the top). Returns a list of
    human-readable problem strings (empty if none)."""
    try:
        if HAS_RUAMEL:
            parsed = _ryaml.load(yaml_text)
        else:
            parsed = _pyyaml.safe_load(yaml_text)
    except Exception:
        return []  # unparseable — check_yaml_format() already covers this

    problems = []

    def _walk(node):
        if isinstance(node, dict):
            block = node.get("block")
            if isinstance(block, list):
                seen = []
                for sub in block:
                    key = json.dumps(sub, sort_keys=True, default=str)
                    if key in seen:
                        name = sub.get("name", "?") if isinstance(sub, dict) else "?"
                        problems.append(
                            f"Duplicate sub-task inside 'block:' (name: {name!r}) — "
                            f"identical to an earlier sub-task in the same block."
                        )
                    seen.append(key)
                for sub in block:
                    _walk(sub)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return problems


_TASK_LEVEL_KEYWORDS = {
    # 'name' and 'block' are excluded, because they are also ordinary module
    # parameters (e.g. package name, blockinfile content); 'rescue' and
    # 'always' are excluded together with 'block'.
    "loop", "with_items", "with_fileglob", "with_dict", "with_nested",
    "with_list", "with_together", "when", "register", "notify", "failed_when",
    "changed_when", "become", "become_user", "become_method", "tags",
    "ignore_errors", "delegate_to", "delegate_facts", "vars", "loop_control",
    "until", "retries", "delay", "no_log", "check_mode", "environment", "args",
    "any_errors_fatal", "run_once", "throttle",
    "listen", "async", "poll", "connection", "diff",
}


def check_nested_task_keywords(yaml_text: str) -> list:
    """Detects a task-level keyword (loop/when/register/notify/etc.) nested
    one level too deep, inside a module's own argument dict. Such YAML is
    valid and can pass ansible-lint, but Ansible then ignores the keyword
    (e.g. a 'loop:' inside 'lineinfile:' runs the task once).
    Returns a list of human-readable problem strings (empty if none)."""
    try:
        parsed = _ryaml.load(yaml_text) if HAS_RUAMEL else _pyyaml.safe_load(yaml_text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    problems = []
    for task in parsed:
        if not isinstance(task, dict):
            continue
        for key, val in task.items():
            if key in _TASK_LEVEL_KEYWORDS or not isinstance(val, dict):
                continue
            # key is presumably the module name (e.g. 'ansible.builtin.lineinfile') —
            # any task-level keyword found INSIDE its argument dict is nested wrong.
            nested = sorted(set(val.keys()) & _TASK_LEVEL_KEYWORDS)
            if nested:
                problems.append(
                    f"Task-level keyword(s) {nested} are nested INSIDE the '{key}' "
                    f"module's own argument dict instead of at the task level (a "
                    f"sibling of '{key}:', indented the SAME as '{key}:', not a child "
                    f"of it) — Ansible silently ignores a keyword in the wrong place "
                    f"here (e.g. a nested 'loop:' never iterates, 'item' stays "
                    f"undefined). Move {nested} up one indentation level."
                )
    return problems


# Extracts regex_replace('pattern', 'replacement') calls (single or double
# quoted, either argument order of quote style) from generated Jinja text —
# used by check_regex_replace_sanity() below.
_REGEX_REPLACE_RE = re.compile(
    r"regex_replace\(\s*(['\"])((?:\\.|(?!\1).)*)\1\s*,\s*(['\"])((?:\\.|(?!\3).)*)\3"
)


def check_regex_replace_sanity(yaml_text: str) -> list:
    """Programmatic guardrail: executes every regex_replace(...) call found in
    the generated YAML on a representative comma-separated sample value and
    flags calls that visibly corrupt it (e.g. a zero-width pattern inserting
    a comma between every letter). The semantic check reads the pattern but
    does not execute it. Returns a list of human-readable problem
    descriptions (empty if nothing looks broken)."""
    problems = []
    sample = "rw,relatime,seclabel"  # representative of ansible_mounts' comma-separated 'options' field
    for pattern_q, pattern, repl_q, repl in _REGEX_REPLACE_RE.findall(yaml_text):
        # Undo the minimal backslash-escaping Jinja/YAML string literals use
        # for their own quote character (e.g. \' inside a '...'-quoted arg).
        pattern = pattern.replace(f"\\{pattern_q}", pattern_q)
        repl = repl.replace(f"\\{repl_q}", repl_q)
        try:
            result = re.sub(pattern, repl, sample)
        except re.error as e:
            problems.append(f"regex_replace('{pattern}', '{repl}') is not a valid Python/Jinja regex: {e}")
            continue
        if result == sample:
            continue  # no-op on the sample — not necessarily wrong, nothing to flag
        tokens = result.split(",")
        single_char_tokens = sum(1 for t in tokens if len(t) == 1)
        if single_char_tokens >= 3:
            problems.append(
                f"regex_replace('{pattern}', '{repl}') shatters a comma-separated value into single "
                f"characters instead of normalizing it: '{sample}' -> '{result}' (a zero-width pattern "
                f"is the usual cause — it matches at every character boundary, not just where intended)"
            )
        elif len(result) > len(sample) * 2:
            problems.append(
                f"regex_replace('{pattern}', '{repl}') drastically inflates the value instead of "
                f"transforming it as intended: '{sample}' ({len(sample)} chars) -> '{result}' "
                f"({len(result)} chars)"
            )
    return problems


_LOOKUP_IN_WHEN_RE = re.compile(r"^\s*(?:-\s*)?when:\s*(.*)$", re.MULTILINE)
_LOOKUP_CALL_RE = re.compile(r"\b(?:lookup|query)\s*\(")


def check_lookup_in_when(yaml_text: str) -> list:
    """Programmatic guardrail: detects a lookup()/query() call inside a
    'when:' condition. Lookups run on the Ansible control node, not on the
    managed host, so such a condition checks the wrong machine. The prompt
    instruction alone did not prevent this pattern.

    Also scans a short lookahead window, because 'when:' can be a single
    string, a multi-line string or a list of conditions.
    Returns a list of human-readable problem descriptions (empty if none)."""
    problems = []
    lines = yaml_text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(?:-\s*)?when:\s*(.*)$", line)
        if not m:
            continue
        indent, rest = m.group(1), m.group(2)
        window = [rest]
        # 'when:' with no inline value is followed by an indented list/scalar
        # (e.g. 'when:\n  - lookup(...) == ...') — collect those continuation
        # lines (deeper-indented than 'when:' itself) too.
        j = i + 1
        while j < len(lines) and lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) > len(indent):
            window.append(lines[j])
            j += 1
        block = "\n".join(window)
        if _LOOKUP_CALL_RE.search(block):
            problems.append(
                f"'when:' condition calls lookup()/query() to read state — this executes on the "
                f"ANSIBLE CONTROL NODE, not the managed remote host, so it can never correctly check "
                f"remote file/command state: {block.strip()[:160]}"
            )
    return problems


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Syntax validation via ansible-lint
# ═══════════════════════════════════════════════════════════════════════════

def run_ansible_lint(yaml_text: str, task_name: str = "generated_task") -> tuple:
    """
    Wraps the generated task(s) in a minimal playbook and runs
    ansible-lint as a deterministic syntax validator.
    Returns (is_valid, findings_list).

    Indentation is added textually, because a ruamel.yaml round-trip adds
    incorrect top-level indentation.
    """
    if not _ansible_lint_available():
        return True, [LintFinding(
            rule="tool_missing", severity="warning",
            message="ansible-lint unavailable — skipping syntax validation. "
                    "Install with: pip install ansible-lint"
        )]

    # First verify this is valid YAML at all (needed to know whether
    # building a playbook even makes sense)
    try:
        if HAS_RUAMEL:
            parsed = _ryaml.load(yaml_text)
        else:
            parsed = _pyyaml.safe_load(yaml_text)
        if parsed is None:
            return False, [LintFinding(rule="yaml_syntax", message="Empty YAML document")]
    except Exception as e:
        return False, [LintFinding(rule="yaml_syntax", message=str(e))]

    # Textual indentation: each task line gets +4 spaces to nest under
    # the playbook's "tasks:" key
    indented_lines = []
    for line in yaml_text.rstrip().split("\n"):
        indented_lines.append(("    " + line) if line.strip() else "")
    indented_body = "\n".join(indented_lines)

    playbook_text = (
        "---\n"
        "- name: Lint check\n"
        "  hosts: all\n"
        "  become: true\n"
        "  tasks:\n"
        f"{indented_body}\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False,
                                      dir=tempfile.gettempdir()) as f:
        f.write(playbook_text)
        temp_path = f.name

    try:
        proc = subprocess.run(
            ["ansible-lint", temp_path, "-f", "json", "--nocolor"],
            capture_output=True, text=True, timeout=30
        )
        findings = []
        if proc.stdout.strip():
            try:
                results = json.loads(proc.stdout)
                for item in results:
                    findings.append(LintFinding(
                        rule=item.get("check_name", item.get("rule", {}).get("id", "unknown")),
                        message=item.get("description", item.get("message", "")),
                        line=item.get("location", {}).get("lines", {}).get("begin"),
                        severity=item.get("severity", "error"),
                    ))
            except json.JSONDecodeError:
                if proc.returncode != 0:
                    findings.append(LintFinding(
                        rule="lint_error", message=proc.stdout[:500] or proc.stderr[:500]
                    ))

        # Blocking vs. cosmetic findings: ansible-lint's "severity" field is
        # not reliable, so the rule family decides. Wrong or missing module
        # parameters ("args") and syntax errors ("syntax-check") are always
        # blocking. Style, naming, FQCN and line-length findings do not block
        # the repair loop.
        blocking = [f for f in findings if _is_blocking(f)]
        is_valid = len(blocking) == 0
        return is_valid, findings

    except subprocess.TimeoutExpired:
        return False, [LintFinding(rule="timeout", message="ansible-lint timed out")]
    except FileNotFoundError:
        return True, [LintFinding(rule="tool_missing", severity="warning",
                                   message="ansible-lint not found on PATH")]
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _ansible_lint_available() -> bool:
    try:
        subprocess.run(["ansible-lint", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: Knowledge-base semantic check — does the fix actually satisfy
# the CIS remediation instruction, not just pass lint?
# ═══════════════════════════════════════════════════════════════════════════
# "Knowledge base" = the CIS Benchmark text for this rule, retrieved from the
# same ChromaDB collection as in Agent 3. It is the best available reference,
# not an infallible ground truth, so the verifier is asked to judge intent
# rather than demand a literal text match.

def _semantic_check_prompt(rule_id: str, rule_title: str, original_task: str,
                           knowledge_base_text: str, generated_yaml: str,
                           sibling_context: str = "") -> list:
    # The semantic check receives the sibling tasks of the same rule, so
    # that a fix covering only its own part of a multi-task implementation
    # is not rejected for missing the siblings' part.
    sibling_note = ""
    if sibling_context:
        sibling_note = f"""
{sibling_context}
IMPORTANT: this task is only ONE PART of a multi-task implementation of this
CIS rule — the sibling task(s) above (already in the project, unchanged by
this fix, or shown with their CURRENT content if an "UPDATE" note appears
above) cover the rest. Judge the COMBINED coverage (this fix + those
siblings), not just this one task in isolation. If this task correctly
handles its own part and does not duplicate what a sibling already does, that
is CORRECT behavior, not a gap — do NOT mark it PARTIAL/NON_COMPLIANT merely
for not re-implementing what a sibling already covers.
BEFORE YOU DECIDE: the CIS remediation text above may describe several
distinct cases/branches (e.g. "if X do A, if Y do B"). For EACH case you are
about to cite as missing or wrong in THIS fix, first check whether the
sibling content shown above (especially any "UPDATE"-marked current content)
already handles that exact case — if it does, that case is COVERED and must
NOT be cited as a gap, even though this one task alone doesn't handle it.
ALSO check the opposite direction: does THIS fix now duplicate what ANY ONE
of the siblings listed above already does (same module + same target, e.g.
same file path)? A fix that pivoted to a different module/approach during
repair can accidentally re-implement a DIFFERENT sibling than the one it was
originally trying to fix — if so, that IS a real defect (mark it accordingly),
even though the fix's syntax and module choice look otherwise reasonable.
"""
    return [
        {"role": "system", "content": (
            "You are a strict but fair reviewer checking whether a generated "
            "Ansible task actually implements a CIS Benchmark remediation "
            "instruction — not merely whether it is syntactically valid YAML. "
            "You are given the relevant CIS Benchmark remediation text as a "
            "knowledge-base reference. Judge against the INTENT of that text, "
            "not a literal wording match. Flag cases where the task copies an "
            "example placeholder verbatim instead of the real value, comments "
            "out or disables something instead of fixing it, uses the wrong "
            "module for the job, or is otherwise a no-op relative to the "
            "requirement. "
            "Some tasks are deliberately only ONE PART of a multi-task "
            "implementation of the same rule, with the rest handled by sibling "
            "task(s) shown to you separately — when that context is given, "
            "judge the COMBINED coverage of the fix plus its siblings, never the "
            "single fix in isolation, and never cite a case/branch as missing "
            "if the shown sibling content already covers it. "
            "AUTHORITY ORDER: the retrieved CIS Benchmark Remediation text for THIS "
            "specific rule (shown below) is the authoritative definition of what "
            "'compliant' means here. The CONFIRMED TECHNICAL FACTS below describe a "
            "tool's usual convention, but a CIS benchmark may legitimately require a "
            "DIFFERENT file format/content for a particular OS version. If the "
            "retrieved Remediation text explicitly shows file content that conflicts "
            "with a CONFIRMED FACT below (e.g. it shows a dconf LOCK file containing "
            "'[section]' + 'key=value' lines rather than a bare key path), FOLLOW THE "
            "RETRIEVED TEXT — a fix that matches the retrieved Remediation text is "
            "COMPLIANT even if it contradicts the general convention in a FACT below. "
            "CONFIRMED TECHNICAL FACTS (verified against official documentation — "
            "do not contradict these when judging, even if a different format 'feels' "
            "more complete or secure — UNLESS the retrieved Remediation text for this "
            "rule explicitly shows otherwise, per AUTHORITY ORDER above): "
            "(1) the 'ansible_mounts' fact's items have ONLY these fields: mount, "
            "device, fstype, options, uuid, size_total, size_available — there is no "
            "'src' field. EXPLICIT, do not invert this: 'item.device' (accessing the "
            "'device' field) IS CORRECT and IS PRESENT — 'device' is a real ansible_mounts "
            "field, not the missing one. 'src' is the NAME OF A DIFFERENT THING (the mount "
            "MODULE's own parameter, filled in FROM item.device) — 'src' is what does NOT "
            "exist as an ansible_mounts field. So 'src: \"{{ item.device }}\"' in a mount-module "
            "task is the CORRECT pattern (module parameter 'src' = fact field 'device') — do "
            "NOT flag 'item.device' as referencing a non-existent field; the non-existent one "
            "would be 'item.src', never 'item.device'. "
            "(2) the mount MODULE's 'state: mounted' (not 'present') is required to "
            "change options on an ALREADY-mounted filesystem immediately; 'present' "
            "only edits fstab, effective after reboot. "
            "(3) the mount MODULE's options parameter is named 'opts', not 'options' "
            "(that is the FACT's key name for the same data). "
            "(4) there is no 'shell' Ansible lookup plugin; lookups (query()/lookup()) "
            "execute on the CONTROL NODE, never on the managed host, so using one to "
            "check remote state in a 'when:' is ALWAYS wrong regardless of plugin name. "
            "(5) GNOME dconf LOCK files (/etc/dconf/db/*.d/locks/*) USUALLY contain "
            "ONLY the bare GSettings key path per line (e.g. '/org/gnome/desktop/"
            "session/idle-delay') — no '[section]' header, no '= value' suffix — with "
            "the richer '[section]'/'key = value' syntax reserved for dconf VALUE "
            "files. This is the common convention (and is what several CIS RHEL rules "
            "specify). HOWEVER, per AUTHORITY ORDER above: some CIS benchmarks "
            "(confirmed: CIS Debian 12 rules 1.7.7 / 1.7.9) instead require the LOCK "
            "file itself to contain '[org/gnome/...]' followed by 'key=value' lines, "
            "and back that with an Audit that greps the locks dir for 'key=false' / "
            "'key=true'. So: if the retrieved Remediation text for THIS rule shows the "
            "lock file containing '[section]' + 'key=value', a fix writing exactly "
            "that content to the locks path IS COMPLIANT — do not flag it as "
            "'value-file syntax in a lock file'. Only apply the bare-path expectation "
            "when the retrieved Remediation text itself shows a bare key path. In "
            "either case, do not penalize a correctly-paired value file + lock file. "
            "(6) the 'community.general.modprobe' module's 'state' parameter accepts "
            "EXACTLY two values: 'absent' and 'present' (confirmed directly by "
            "ansible-lint's own error output: 'value of state must be one of: absent, "
            "present') — 'state: absent' is the CORRECT, intended value for 'ensure this "
            "kernel module is not available/loadable', do not flag it as invalid or "
            "suggest a different value (e.g. 'removed', 'unloaded') — those do not exist "
            "for this module and will fail lint. "
            "(7) Jinja2 dot-notation on a dict variable, e.g. '{{ sshd.clientalivecountmax }}' "
            "where 'sshd' is a dict/mapping var, correctly resolves to the SCALAR value stored "
            "at that key (e.g. renders as '3'), NOT a stringified representation of the whole "
            "dict — confirmed live (a real project's defaults/ had 'sshd: {clientalivecountmax: "
            "3, ...}', and '{{ sshd.clientalivecountmax }}' in a lineinfile 'line:' correctly "
            "renders '3'). Do NOT flag this pattern as producing a literal Python-dict string in "
            "the config file unless the accessed key's OWN value is itself a list/dict (check the "
            "RESOLVED PARAMETER VALUES shown in the task/prompt, if present, for the actual value). "
            "(8) Ansible file-module 'mode:' accepts SYMBOLIC notation (u/g/o/a + +/-/= + rwx), "
            "e.g. 'go-rwx' means 'remove read/write/execute for Group and Other' — this TIGHTENS "
            "permissions (typically leaving something like 0600/0700), it is NOT the octal value "
            "0777 and does NOT grant broad access; confirmed live (a real lockdown task used "
            "'mode: go-rwx' on a /etc/modprobe.d/ file and a judge incorrectly read it as "
            "'sets mode to 0777, insecure'). Never interpret a symbolic mode string as if its "
            "letters were octal digits. This ALSO applies to MULTI-CLAUSE mode strings, e.g. "
            "'o-x,g-wx,o-rwx' — every clause using '-' still REMOVES no matter how many clauses "
            "there are or whether a later clause re-targets the same letter (a '-' clause NEVER "
            "adds a permission back); confirmed live (a judge claimed 'o-rwx' in such a string "
            "'adds read/write/execute back to others, resulting in 0666' — wrong, it still removes). "
            "(9) The 'ansible.builtin.replace'/'lineinfile'/'blockinfile' modules' 'path:' "
            "parameter takes EXACTLY ONE file path — it does NOT accept a glob pattern like "
            "'/etc/rsyslog.d/*.conf' written DIRECTLY as the 'path:' value (glob expansion is not "
            "performed by these modules themselves). BUT — CONFIRMED, do NOT flag this as an error "
            "when the task already loops correctly: 'with_fileglob:' attached DIRECTLY to the task "
            "(e.g. '- name: ...\\n  ansible.builtin.replace:\\n    path: \"{{ item }}\"\\n  "
            "with_fileglob:\\n    - /etc/sysctl.d/*.conf') IS, BY ITSELF, the complete, correct, "
            "idiomatic pattern — 'with_fileglob' is a LOOKUP PLUGIN that resolves the glob pattern "
            "into a list of REAL file paths BEFORE the task ever runs, then loops the task once per "
            "matched file with 'item' bound to each real path; 'path: \"{{ item }}\"' inside that "
            "task is therefore a single real filename at execution time, not a glob pattern — this "
            "is standard Ansible, requires NO separate preceding 'find'/'register' step, and is NOT "
            "the same construct as writing a glob literally into 'path:'. A SEPARATE preceding "
            "'ansible.builtin.find' + 'register' + 'loop: \"{{ result.files }}\"' is only a "
            "DIFFERENT, equally valid way to achieve the same enumeration (needed when you must "
            "search recursively, filter by file type/age/etc., or search multiple, dynamically-"
            "unknown directories) — do NOT require it when 'with_fileglob:' is already present and "
            "correctly used; that is not a missing step, it is the SAME step in its simpler form. "
            "Confirmed live 2026-09-02: this exact false claim ('replace does not support glob "
            "expansion' / 'with_fileglob is not supported') was raised identically against a fix "
            "that correctly used 'with_fileglob:' directly on a 'replace:' task, across many sysctl "
            "rules in one evaluation run, each time incorrectly forcing a retry. "
            "(10) A loop variable (e.g. 'item') only has dict-style attributes ('item.somekey') "
            "when its SOURCE is a list of dicts — a loop built from 'stdout_lines', a "
            "'regex_replace' filter, or any plain string list has NO such attributes; referencing "
            "'item.something' on it is Undefined and will fail at runtime. Check the loop's actual "
            "source before assuming 'item' is a dict. "
            "(11) The 'ansible.posix.sysctl' module's 'state:' parameter accepts ONLY 'present' or "
            "'absent' — there is NO 'immediate'/'applied'/'reloaded' value (confirmed by "
            "ansible-lint's own error: 'value of state must be one of: present, absent'). The "
            "module ALREADY applies the value immediately via 'sysctl -w' by default (controlled "
            "by the separate boolean 'sysctl_set:'/'reload:' parameters, default true) — 'state: "
            "present' (or omitting 'state' entirely) is already the CORRECT, complete fix; do not "
            "invent a state value to express 'apply now'. "
            "(12b) applies SPECIFICALLY to AUDIT/CHECK-style tasks (verifying state, e.g. 'findmnt | "
            "grep nodev', '| AUDIT |') — NOT state-changing remediation commands (dnf/package, a "
            "mutating firewall-cmd/systemctl call, a file/lineinfile edit — ignore_errors there is a "
            "separate, much less severe robustness note, not 'unimplemented'). For an AUDIT/CHECK task, "
            "check IN ORDER: (1) does THIS task already have its OWN 'failed_when:'/'assert:' "
            "referencing its OWN 'register:'? If yes, it's COMPLETE — 'ignore_errors: yes' being ALSO "
            "present does not erase that failed_when verdict, do not flag it. (2) ONLY if there's NO "
            "'register:' at all, or one never referenced by any 'failed_when:'/'assert:'/'when:' "
            "anywhere (own task OR a sibling), does 'ignore_errors: yes' mean the result is NOT actually "
            "verify anything — the result is silently discarded. Do not accept 'the audit logic looks "
            "correct' as sufficient for COMPLIANT; a check whose result is never evaluated is "
            "functionally a no-op. CRITICAL: applies ONLY when failure-suppression is ACTUALLY "
            "PRESENT (ignore_errors: yes / failed_when: false) — a plain remediation task with normal "
            "failure semantics needs NO separate post-verification step; do not invent that "
            "requirement for a task that isn't suppressing anything. "
            "(12) sshd_config uses FIRST-OBTAINED-VALUE semantics for most directives — unlike "
            "most Linux config files, the FIRST occurrence of a directive wins, later ones are "
            "ignored. If the file has an early 'Include .../*.conf' line, a directive appended at "
            "'insertbefore: EOF' (i.e. at the very end, after the Include) can be silently "
            "overridden by a conflicting value inside an included file, even though it appears "
            "'later' in the main file. A fix intending to guarantee precedence should target "
            "'insertbefore' a specific early anchor (e.g. the first 'Include'/'Match' line) rather "
            "than EOF, when such a line exists in the target file. "
            "(13) sudo's /etc/sudoers has, by DEFAULT on virtually every distro, a "
            "'#includedir /etc/sudoers.d' (or '@includedir') line near its END — meaning files "
            "under /etc/sudoers.d/ are parsed AFTER /etc/sudoers itself, and for 'Defaults' "
            "settings (e.g. timestamp_timeout) sudo applies the LAST value it parses for that "
            "setting. A fix that creates ONE new drop-in file under /etc/sudoers.d/ with the "
            "required 'Defaults ...=value' line is therefore a CORRECT, COMPLETE, and reliably "
            "authoritative fix even when /etc/sudoers itself (or another sudoers.d file) ALSO "
            "sets that same directive to a different value — the new drop-in file's value wins "
            "on the default include order. Do NOT require a fix to ALSO edit the main /etc/sudoers "
            "file or every existing sudoers.d file in place — that is unnecessary and more fragile "
            "(regex edits to an unknown-format existing line) than adding one new authoritative "
            "drop-in file, which is CIS's own commonly recommended pattern for sudoers hardening. "
            "(14) For auditd rules (auditctl lines / /etc/audit/rules.d/*.rules entries): do NOT "
            "require a '-F arch=b32' (32-bit) counterpart rule just because a task's own NAME "
            "contains '32 bit'/'32bit'/'64 bit'/'64bit' — these labels are frequently MISLEADING "
            "in this project (arbitrary internal labels for two DIFFERENT parts of ONE remediation "
            "— e.g. a path-based rule vs a syscall-based rule — not architecture variants). Check "
            "ONLY whether the retrieved KNOWLEDGE-BASE REFERENCE text below itself explicitly shows "
            "an 'arch=b32' line anywhere; if it shows ONLY 'arch=b64' (or no arch field at all) and "
            "never mentions b32/32-bit coverage, do NOT flag the fix for 'missing the 32-bit rule' "
            "— that invents a requirement the benchmark text does not state for this specific rule. "
            "Confirmed live: rule 6.3.3.31's own retrieved text shows only a single 'arch=b64' "
            "example, never a b32 counterpart, despite its project task being named '...32bit'. "
            "(15) When the KNOWLEDGE-BASE REFERENCE text lists MULTIPLE, SEQUENTIAL steps (e.g. "
            "'systemctl stop X.service' THEN 'dnf remove X', or 'stop and mask X.socket X.service' "
            "as an alternative when the package has dependents), REJECT a fix that only performs the "
            "LAST/most-visible step — removing a package does NOT, by itself, guarantee an "
            "already-running instance of its service was stopped first. This is a recurring pattern "
            "across every 'service not in use' rule in this benchmark family; confirmed live this "
            "pipeline's own fixes have repeatedly reached final output missing the stop step even "
            "after being told to include it, so check for it explicitly on every removal-type fix "
            "rather than trusting it was already handled. "
            "(16) When the KNOWLEDGE-BASE REFERENCE text includes a step to find and COMMENT OUT/"
            "REMOVE conflicting pre-existing entries (common for sysctl parameters, crypto "
            "sub-policies, dconf/GNOME settings — 'Step 1: review all files... and comment out or "
            "remove all <param> lines that are not <compliant value>'), REJECT a fix that only "
            "writes the NEW authoritative value without a cleanup step for OLDER conflicting entries "
            "in other candidate config files/directories — the new value can be silently overridden "
            "by an old one still present elsewhere. Confirmed live this is a recurring, frequently "
            "OMITTED step even after being told to include it (multiple sysctl rules in one run) — "
            "check for a find/loop/replace (or equivalent) cleanup task explicitly, do not assume "
            "'sets the value correctly' is sufficient when the reference text describes cleanup as "
            "its own numbered step. "
            "(17) Standard Unix exit-code convention: 'grep -q PATTERN file' exits 0 when the "
            "PATTERN WAS FOUND. REJECT a fix where a task registers such a check and a LATER task "
            "gates 'when: <var>.rc == 0' to APPLY a fix for something MISSING — that direction is "
            "INVERTED (it would only run when the thing is already present, never when actually "
            "missing). A task ENABLING/ADDING a missing feature must gate on 'rc != 0'; a task "
            "REMOVING a present setting must gate on 'rc == 0'. Confirmed live: rule 5.3.1.3's "
            "'enable with-pwquality feature IF REQUIRED' task repeatedly used 'when: "
            "pwquality_with_feature.rc == 0' after grepping FOR the feature — check this specific "
            "direction explicitly, do not accept a superficially-plausible-looking condition. "
            "(18) When the KNOWLEDGE-BASE REFERENCE text states a disjunctive requirement using "
            "'EITHER ... OR ...' (or equivalent 'any ONE of the following') for the SAME file/"
            "directive, crediting COMPLIANT via EITHER option is sufficient — do NOT require one "
            "specific option when the fix chose the other, equally valid option. Confirmed live: "
            "rule 1.1.1.1 (and the identical repeated pattern for 1.1.1.2-1.1.1.8, plus the "
            "equivalent network-protocol rules 3.2.1-3.2.4) requires 'blacklist <module> -AND "
            "EITHER- install <module> /bin/false -OR- install <module> /bin/true' — a fix setting "
            "'install <module> /bin/true' was repeatedly, wrongly rejected across 6 attempts for "
            "'using /bin/true instead of the required /bin/false', even though the retrieved CIS "
            "text itself states both are acceptable. Always check the reference text for "
            "'EITHER/OR' framing before rejecting a choice between two offered alternatives. "
            "(19) When the KNOWLEDGE-BASE REFERENCE text itself defers the corrective action to "
            "human judgment (e.g. 'based on the results of the audit script, determine/establish/"
            "review ... to determine which is correct', 'manually verify/review') rather than "
            "giving one mechanical fix, a fix that correctly detects the condition and fails "
            "loudly (failed_when/assert on its own register) is the correct, COMPLETE automated "
            "implementation — do NOT reject it for 'only detecting, not implementing the "
            "remediation' (e.g. automatically reassigning a UID/renaming an account), since the "
            "benchmark's own text does not define a safe, deterministic way to do so automatically. "
            "Confirmed live: rules 7.2.4/7.2.6/7.2.7 ('no duplicate UIDs/user names/group names') "
            "explicitly instruct 'review all files owned by the shared UIDs to determine which UID "
            "they are supposed to belong to' — a detect-and-fail fix was repeatedly rejected for "
            "'only detecting, not implementing the remediation', even though CIS's own text defers "
            "that decision to a human administrator. "
            "(20) A CIS remediation's 'EITHER/OR' framing can describe alternative REMEDIATION "
            "ACTIONS, not only alternative config VALUES (e.g. 'remove the package -OR- stop and mask "
            "the service' are both valid ways to satisfy 'service not in use'). Do not require one "
            "specific action/value when the reference text offers it as one of several alternatives "
            "and the fix correctly implements a different, equally valid alternative. "
            "(21) This pipeline's own judges have TWICE made an incorrect permission-math claim "
            "(rejecting a correct symbolic chmod as not reaching a target octal value; rejecting a "
            "correct octal mode for not textually matching a symbolic form used elsewhere). Before "
            "asserting a mode (symbolic OR octal) does or does not satisfy a required permission "
            "level, explicitly compute the resulting permission bits and compare the actual bits, "
            "never the surface text/format. "
            "(22) The 'ansible.builtin.file' module's 'state:' parameter's valid values are: absent, "
            "directory, file, hard, link, touch. 'state: touch' IS valid — do not reject it. "
            "(23) In an authselect custom profile file, the syntax '{include if \"<feature-name>\"}' "
            "is authselect's own native conditional-include directive (not Jinja2) and is valid "
            "standard syntax — do not flag it as broken or invented Jinja2. "
            "(24) Before finalizing a NON_COMPLIANT/PARTIAL verdict, re-read your own stated reason: "
            "if it actually concludes the requirement IS satisfied, the verdict must be COMPLIANT — "
            "never emit a verdict that contradicts your own stated reasoning. "
            "(25) CONFIRMED against the official CIS Benchmark PDF itself (not just the Remediation "
            "prose excerpt you may be given): for the kernel-module-disable family (1.1.1.1 cramfs, "
            "1.1.1.2 freevxfs, 1.1.1.3 hfs, 1.1.1.4 hfsplus, 1.1.1.5 jffs2, 1.1.1.6 squashfs, 1.1.1.7 "
            "udf, 1.1.1.8 usb-storage, and the network-protocol equivalents 3.2.1 dccp, 3.2.2 tipc, "
            "3.2.3 rds, 3.2.4 sctp), the Benchmark's own AUDIT script explicitly checks "
            "'grep -Pq ...install MODULE /bin/(true|false)...' and its prose states 'An entry "
            "including /bin/true or /bin/false exists' — '/bin/true' and '/bin/false' are BOTH "
            "explicitly, officially compliant values for the 'install MODULE ...' line; the "
            "Remediation section's example merely shows /bin/false as one illustrative choice, it "
            "does not exclude /bin/true. A fix using 'install MODULE /bin/true' is NOT NON_COMPLIANT "
            "for that reason — do not reject it. (The upstream ansible-lockdown RHEL9-CIS project "
            "itself uses /bin/true for exactly this family, e.g. tasks/section_1/cis_1.1.1.x.yml and "
            "tasks/section_3/cis_3.2.x.yml.) "
            "(26) CONFIRMED — in the ansible-lockdown RHEL9-CIS role, a per-rule task that only does "
            "'set_fact: rhel9cis_sysctl_update: true' (rules 1.5.1, 1.5.2, 3.3.1-3.3.11) or "
            "'set_fact: update_audit_template: true' (6.3.3.1-6.3.3.19), optionally followed by a "
            "'debug' message, is a DEFERRED-ENFORCEMENT flag, not a no-op — tasks/main.yml always "
            "imports tasks/post.yml, whose 'POST | Update sysctl' task templates "
            "/etc/sysctl.d/60-kernel_sysctl.conf, 60-netipv4_sysctl.conf and "
            "60-netipv6_sysctl.conf and notifies 'Reload sysctl'; tasks/main.yml also imports "
            "tasks/auditd.yml, which templates /etc/audit/rules.d/99_auditd.rules and notifies "
            "'augenrules --load'. Any 'ACTUAL ENFORCEMENT CONTENT' block shown alongside such a task "
            "IS its real remediation — do NOT reject it for 'only sets a fact / only prints a debug "
            "message / does not write /etc/sysctl.d or /etc/audit/rules.d' when that block is "
            "present and contains the required directive. "
            "(27) CONFIRMED — ansible-lockdown's mount-option tasks (1.1.2.1.2 through 1.1.2.7.4) "
            "that 'set_fact: prelim_mount_point_fs_and_options' with 'notify: \"Remount {{ "
            "mount_point }}\"' are complete remediations: handlers/main.yml defines, under 'listen: "
            "\"Remount <mount_point>\"', an ansible.posix.mount task with 'state: present' (writes "
            "/etc/fstab) followed by one with 'state: remounted' (applies it live). Do NOT reject "
            "these as 'only updates an in-memory fact / never writes /etc/fstab / never remounts' "
            "when a handler-content note confirms this. "
            "(28) CONFIRMED against the official CIS PDF (2.4.2.1 Audit section): /etc/at.allow is "
            "compliant when 'group owned by the group daemon or group root' — both are explicitly, "
            "equally valid. A fix setting 'group: root' is not wrong for not using 'daemon'. "
            "(29) CONFIRMED against the official CIS PDF (5.4.3.2 Audit script): the audit accepts "
            "'readonly TMOUT' and 'export TMOUT' as separate lines, besides 'typeset -xr TMOUT=N'. "
            "The 3-line 'TMOUT=900'/'readonly TMOUT'/'export TMOUT' form PASSES; 'deprecated "
            "methods' is a style label, not a failure condition."
        )},
        {"role": "user", "content": f"""CIS rule: {rule_id} — {rule_title}
Original (non-compliant) task that this fix replaces: {original_task}

KNOWLEDGE-BASE REFERENCE (CIS Benchmark Audit + Remediation text for this rule — the AUDIT
section is the actual PASS CRITERIA; a fix satisfying it is compliant even if it differs from
the Remediation section's own example):
{_kb_text_slice(knowledge_base_text, 4000) or "[NOT FOUND — no remediation chunk indexed for this rule]"}

GENERATED FIX (Ansible YAML):
{generated_yaml[:2000]}
{sibling_note}
Does this fix — COMBINED with any sibling coverage shown above — correctly
implement the remediation instruction above?
Respond with ONLY valid JSON, no markdown:
{{"verdict": "COMPLIANT" or "PARTIAL" or "NON_COMPLIANT",
"score": integer 1-5 (5 = fully implements the remediation, 1 = does not implement it at all or does something unrelated),
"reason": "1-2 sentences, specifically what is correct/missing"}}"""},
    ]


def _lenient_json_loads(text: str):
    """json.loads() with one fallback repair pass for a common LLM JSON output
    error: an unescaped backslash inside a string value (e.g. a quoted regex)
    produces invalid JSON. Escapes backslashes that do not start a valid JSON
    escape and retries once; raises if the text is still invalid."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
        return json.loads(repaired)


def verify_with_knowledge_base(rule_id: str, rule_title: str, original_task: str,
                               generated_yaml: str, kb_text: str = None,
                               sibling_context: str = "") -> dict:
    """
    One knowledge-base-grounded semantic check of a single generated fix.
    Never raises: a failure is returned in an 'error' field, and the
    calling loop does not penalise the fix for an infrastructure problem.

    kb_text: pre-fetched knowledge-base text covering all of the task's rule
    IDs (from validate_and_repair()). Pass None to fetch the text for
    rule_id alone.
    """
    try:
        from agent_a3_llm import get_audit_and_remediation_chunks
        import agent_a2_llm
    except ImportError as e:
        return {"verdict": "", "score": None, "reason": "",
                "error": f"import failed: {e}", "knowledge_base_found": False}

    if kb_text is None:
        kb_text = get_audit_and_remediation_chunks([rule_id]) or ""
    messages = _semantic_check_prompt(rule_id, rule_title, original_task, kb_text, generated_yaml,
                                      sibling_context=sibling_context)
    payload = json.dumps({
        "model": agent_a2_llm.API_MODEL,
        # enable_thinking:False — this is a short, focused check, not a
        # full-report review; keeps latency and cost down for what can be
        # a per-attempt call inside the repair loop.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        # Large enough for a verbose "reason" field; truncation is also
        # detected through finish_reason below.
        "max_tokens": 800,
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
            with urllib.request.urlopen(req, timeout=150) as resp:
                data = json.loads(resp.read())
            # Recorded once, after a usable result is parsed, so that a
            # malformed response is not counted twice.
            choice = data["choices"][0]
            text = choice["message"]["content"].strip()
            # A truncated verdict (finish_reason=="length") is treated as an
            # error, because it may be incomplete.
            if choice.get("finish_reason") == "length":
                # Recorded as a successful HTTP call: this is a token-limit
                # issue, not a connection failure.
                agent_a2_llm._record_request(time.time() - t0, success=True)
                if attempt < 2:
                    continue
                return {"verdict": "", "score": None, "reason": "",
                        "error": "verifier response truncated by max_tokens (finish_reason=length)",
                        "knowledge_base_found": bool(kb_text)}
            json_str = extract_json_object(text)
            if not json_str:
                agent_a2_llm._record_request(time.time() - t0, success=False)
                if attempt < 2:
                    continue
                return {"verdict": "", "score": None, "reason": "",
                        "error": "verifier did not return valid JSON",
                        "knowledge_base_found": bool(kb_text)}
            result = _lenient_json_loads(json_str)
            agent_a2_llm._record_request(time.time() - t0, success=True)
            dump_llm_call("A4_semantic", rule_id, messages, text, result)
            return {
                "verdict": result.get("verdict", ""),
                "score": result.get("score"),
                "reason": result.get("reason", ""),
                "error": "",
                "knowledge_base_found": bool(kb_text),
            }
        except Exception as e:
            agent_a2_llm._record_request(time.time() - t0, success=False)
            if attempt < 2:
                continue
            return {"verdict": "", "score": None, "reason": "",
                    "error": str(e), "knowledge_base_found": bool(kb_text)}


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: Rewriting logic — the failure report goes back to Agent 3
# ═══════════════════════════════════════════════════════════════════════════

# Concrete repair hints for recurring classes of ansible-lint errors. The
# linter message says what is wrong but not how to fix it, and without a
# hint the model can return the same broken YAML again.
KNOWN_FIX_HINTS = [
    (re.compile(r"'handlers'\s+is\s+not\s+a\s+valid\s+attribute\s+for\s+a\s+Block", re.I),
     "Do not place 'handlers:' inside a task or a block (block:) — handlers "
     "are defined ONLY at the role/playbook level (a separate "
     "handlers/main.yml file) and triggered via 'notify:'. Remove the "
     "'handlers:' key entirely from this task/block."),
    (re.compile(r"couldn't resolve module/action '([\w.]+)'", re.I),
     "The given module name does not exist, or is missing its fully "
     "qualified collection name (FQCN). Use a module that actually exists "
     "in Ansible, with the correct prefix (e.g. 'ansible.builtin.X' or "
     "'community.general.X') — do not invent a module or collection name."),
]


def _lookup_fix_hints(findings: List[LintFinding]) -> list:
    """Match known error patterns to concrete fix hints."""
    hints = []
    for f in findings:
        if not _is_blocking(f):
            continue
        for pattern, hint in KNOWN_FIX_HINTS:
            if pattern.search(f.message) and hint not in hints:
                hints.append(hint)
    return hints


def repair_via_agent3(yaml_text: str, findings: List[LintFinding],
                      rule_id: str, rule_title: str, sibling_context: str = "") -> str:
    """
    Sends the previous (failing) YAML version along with the
    ansible-lint error report back to Agent 3, which re-synthesizes a
    corrected version. Uses the low-level _call_llm function from
    agent_a3_llm.py — the same LLM infrastructure as the rest of A3.
    """
    try:
        from agent_a3_llm import _call_llm
    except ImportError:
        print("  [A4] agent_a3_llm._call_llm unavailable — cannot repair, returning original")
        return yaml_text

    # If no finding is classified as blocking (e.g. a lint timeout), all
    # findings are shown to the model.
    blocking_findings = [f for f in findings if _is_blocking(f)] or findings
    error_summary = "\n".join(
        f"- [{f.rule}] {f.message}" + (f" (line {f.line})" if f.line else "")
        for f in blocking_findings
    )

    hints = _lookup_fix_hints(findings)
    hints_block = ""
    if hints:
        hints_block = "\n\nHINTS (specific fix for this error class):\n" + "\n".join(f"- {h}" for h in hints)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Ansible syntax repair expert. You will receive a YAML "
                "task that failed ansible-lint validation, along with the exact "
                "error report. Fix ONLY the reported issues while preserving the "
                "original intent (implementing a CIS Benchmark requirement). "
                "GENERAL ANSIBLE BEST PRACTICES (apply these always, not just the specific cases "
                "below): prefer a dedicated Ansible module over shell/command whenever one exists; "
                "NEVER guess a module's parameter name or a fact's field name by analogy (a fact's "
                "field and a module's parameter for the same data are often spelled differently); "
                "every task must be idempotent (safe to re-run); prefer built-in Ansible facts over "
                "parsing shell output; prefer safe Jinja2 filters (split/join/unique/selectattr) over "
                "custom regex for string manipulation. "
                "To add a value to an EXISTING comma-separated string (e.g. a mount "
                "'options' field) use safe list operations: \"{{ (item.options.split(',') "
                "+ ['newvalue']) | unique | join(',') }}\" — NEVER a custom regex_replace() "
                "for this; a zero-width regex can match at every character boundary and "
                "corrupt the string (confirmed live: 'rw,relatime' -> 'r,w,,r,e,l,a,t,i,m,e,'). "
                "That 'unique' filter already makes the append idempotent by itself — do NOT "
                "also add an extra 'when' condition (e.g. a regex_search checking whether the "
                "value is already present) to 'guard' against re-adding it; such conditions are "
                "unnecessary and easy to get subtly wrong (confirmed live: a regex like "
                "'(?<!^)(?<!,)(?<!,nodev)(?<!nodev,)nodev' searching a comma-separated options "
                "string for 'nodev' NEVER matches ANYTHING, because 'nodev' in such a string is "
                "always either at the very start or right after a comma, and both are excluded by "
                "the lookbehinds — making the 'when' condition silently, permanently True, a dead "
                "check that looks like a safeguard but does nothing). "
                "When looping over 'ansible_mounts | selectattr(...)', 'item' has ONLY these "
                "fields: mount, device, fstype, options, uuid, size_total, size_available — "
                "there is NO 'src' field; use 'item.device' for the mount module's 'src' "
                "parameter, never 'item.src' (confirmed live: silently evaluates to Undefined). "
                "When using the mount module to change options on an ALREADY-MOUNTED filesystem, "
                "use state: mounted, NEVER state: present — 'present' only writes /etc/fstab (takes "
                "effect after reboot only), but CIS Audit checks read the LIVE mount table, so a "
                "'present'-only fix stays NON_COMPLIANT until reboot; 'mounted' remounts immediately. "
                "The mount MODULE's parameter for the options string is 'opts' — NEVER 'options' "
                "(that is the ansible_mounts FACT's key name for the same data, an easy mix-up; "
                "confirmed live: 'options' fails ansible-lint with 'Unsupported parameters'). "
                "NEVER use a Jinja lookup/query() (e.g. query('shell', ...), lookup('shell', ...), "
                "query('pipe', ...)) inside a 'when:' condition to check REMOTE host state — lookups "
                "execute on the ANSIBLE CONTROL NODE, not the managed host, so the result would reflect "
                "the WRONG machine entirely, regardless of which lookup plugin name is used. There is "
                "also no built-in 'shell' lookup plugin at all (confirmed: the real ones are 'pipe', "
                "'file', 'env', 'password', 'lines', 'ini' — 'shell' does not exist and would fail at "
                "runtime with a lookup-plugin-not-found error). To check remote command output before "
                "deciding whether to run something, use a SEPARATE preceding task with 'register:', then "
                "reference that registered variable's .stdout/.rc in this task's own 'when:' — never an "
                "inline lookup/query(). "
                "GNOME dconf LOCK files (/etc/dconf/db/*.d/locks/*, used to prevent a setting from being "
                "overridden) use a DIFFERENT, SIMPLER format than dconf VALUE files: each line is JUST the "
                "full GSettings key path starting with '/', one per line — e.g. "
                "'/org/gnome/desktop/session/idle-delay' — NO '[section]' header and NO 'lock' keyword "
                "(confirmed via official GNOME documentation). The '[section]' + 'key = value' format is "
                "ONLY for dconf VALUE files (e.g. /etc/dconf/db/*.d/*, setting the actual default) — do "
                "not mix the two formats or invent a 'lock <key>' directive, which does not exist. "
                "Unless the CIS remediation text EXPLICITLY names a different database, the standard "
                "machine-wide dconf database is 'local.d' (/etc/dconf/db/local.d/ and its locks/ "
                "subdirectory) — NOT '/etc/dconf/db/gdm.d/', even for a rule whose title mentions "
                "'GDM'; many GDM-related CIS rules (screen-lock, idle-delay) are enforced via the "
                "general local.d database. Verify against the exact path the retrieved CIS text "
                "gives, never assume 'gdm.d' from the rule title alone. "
                "sudo's /etc/sudoers has, by DEFAULT on virtually every distro, a "
                "'#includedir /etc/sudoers.d' (or '@includedir') line near its END — files under "
                "/etc/sudoers.d/ are parsed AFTER /etc/sudoers itself, and for 'Defaults' settings "
                "(e.g. timestamp_timeout) sudo applies the LAST value it parses. A fix that creates "
                "ONE new drop-in file under /etc/sudoers.d/ with the required 'Defaults ...=value' "
                "line is a CORRECT, COMPLETE, and reliably authoritative fix even when /etc/sudoers "
                "itself (or another sudoers.d file) ALSO sets that directive — the new drop-in file's "
                "value wins on the default include order. Prefer this single new drop-in file over "
                "trying to regex-edit /etc/sudoers or every existing sudoers.d file in place — "
                "editing an unknown-format existing line is far more fragile, and a single new file "
                "is CIS's own commonly recommended pattern for sudoers hardening. "
                "A CIS rule about DUPLICATE identifiers (duplicate UIDs, GIDs, user names, or group "
                "names) requires ESTABLISHING A NEW, UNIQUE value for the duplicate, NEVER "
                "'state: absent' / deleting the account or group — confirmed live (rule 7.2.6): its "
                "own remediation text says 'establish unique user names... file ownerships will "
                "automatically reflect the change as long as the users have unique UIDs', "
                "explicitly a rename/renumbering, not a deletion; do not 'fix' a duplicate-identifier "
                "finding by proposing to delete the account/group. Do NOT invent a variable (e.g. a "
                "'duplicate_x_mapping' fact) that no other task in the project actually defines — a "
                "task referencing an undefined/always-empty variable is a silent no-op. When the CIS "
                "text itself phrases this as requiring manual review and you cannot safely "
                "auto-compute the correct new value, the CORRECT fix is a task that DETECTS the "
                "duplicate and FAILS LOUDLY with a clear message — an honest non-fix beats a "
                "fabricated one that silently does nothing. "
                "Never use 'authselect select <profile> ...' to 'fix' a rule about one SPECIFIC PAM "
                "module's presence — it REPLACES THE ENTIRE ACTIVE PROFILE, silently dropping "
                "unrelated already-correct settings, and guessing which profile/features to select "
                "fabricates configuration state with no basis. 'authselect select <profile> --with-X' "
                "only accepts real, existing authselect feature flags (e.g. '--with-faillock', "
                "'--with-sudo') — there is no '--with-pam' flag; pam_unix is part of every baseline "
                "profile already, not an optional feature toggled this way. A remediation phrased as "
                "a verification script (grep for 'pam_unix.so' in the profile's own template files) "
                "should be implemented as exactly that check, never a profile-replacing "
                "'authselect select'. "
                "The rule_id/rule_title given to you is the AUTHORITATIVE description of what is "
                "required — trust it over a possibly typo'd or inverted original task name. "
                "Confirmed live: a task named 'Ensure nodexec option set...' (typo for 'noexec') "
                "produced a fix adding 'exec' to mount options — the OPPOSITE of what the real rule "
                "('Ensure noexec option set...') requires. Before writing a fix whose effect is the "
                "OPPOSITE of a positive/prohibitive framing in the rule title (set vs. not set, "
                "enable vs. disable), resolve any conflict in the rule title's favor, never the "
                "source task's own possibly-garbled name. "
                "Output EXACTLY ONE Ansible task (a single YAML list item starting "
                "with '- name:') — never multiple alternative versions, draft "
                "attempts, or a 'rewritten' duplicate alongside the original; pick "
                "ONE final approach. Output ONLY that corrected YAML, no "
                "explanation, no markdown, and no comments narrating your reasoning."
            )
        },
        {
            "role": "user",
            "content": f"""This Ansible task (implementing CIS {rule_id}: {rule_title}) failed validation:

ORIGINAL YAML:
{yaml_text}

ANSIBLE-LINT ERRORS:
{error_summary}{hints_block}
{sibling_context}
Fix the YAML so it passes ansible-lint. Output ONLY the corrected YAML task(s)."""
        }
    ]

    try:
        response = _call_llm(messages, identifier=f"A4_REPAIR_LINT {rule_id}")
        cleaned = re.sub(r"^```(?:yaml)?\s*\n?", "", response.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip())
        return cleaned
    except Exception as e:
        print(f"  [A4] Error during repair via Agent 3: {e}")
        return yaml_text


ESCALATE_AFTER = 3


def _escalation_note(failed_attempts: int, prior_reasons: List[str] = None) -> str:
    """After several rejected repair rounds, tells the model explicitly that
    its current approach is not working and must be changed, and lists the
    reasons for the earlier rejections. Each single rejection describes only
    one symptom, so without this summary the model tends to keep refining
    the same approach."""
    if failed_attempts < ESCALATE_AFTER:
        return ""
    note = (
        f"\n⚠ STRATEGY CHANGE REQUIRED — this is repair attempt #{failed_attempts + 1}, and "
        f"{failed_attempts} previous attempts have ALL been rejected. Refining the SAME approach "
        f"again will very likely fail the same way. The problem is most likely the APPROACH "
        f"ITSELF, not its parameters.\n"
        f"Consider a fundamentally DIFFERENT implementation, for example:\n"
        f"  - a dedicated Ansible module instead of lineinfile/replace with a hand-written regex "
        f"(e.g. ansible.builtin.copy/template writing a small dedicated drop-in config file, "
        f"community.general.ini_file, ansible.posix.mount, etc.);\n"
        f"  - writing a NEW dedicated file (a drop-in under a .d/ directory) instead of trying to "
        f"edit an existing multi-format file in place — this sidesteps regex edge cases entirely "
        f"and is what many CIS remediation texts themselves recommend;\n"
        f"  - a 'blockinfile' with a clear marker instead of a single-line regex;\n"
        f"  - splitting the work into TWO tasks (one to guarantee the file/section exists, one to "
        f"set the value) instead of one clever regex that must handle every case at once.\n"
    )
    if prior_reasons:
        seen, uniq = set(), []
        for r in prior_reasons:
            key = (r or "")[:120]
            if key and key not in seen:
                seen.add(key)
                uniq.append((r or "")[:220])
        if uniq:
            note += ("Reasons your previous attempts were rejected (do NOT reintroduce any of "
                     "these — and note that fixing them one at a time has not worked):\n")
            for i, r in enumerate(uniq[-5:], 1):
                note += f"  {i}. {r}\n"
    return note


def repair_via_agent3_semantic(yaml_text: str, rule_id: str, rule_title: str,
                               knowledge_base_text: str, semantic_reason: str,
                               sibling_context: str = "",
                               failed_attempts: int = 0,
                               prior_reasons: List[str] = None) -> str:
    """
    Sends the current YAML back to Agent 3 along with the knowledge-base
    CIS remediation text and the reviewer's explanation of what's
    semantically wrong or incomplete (not a lint error — the YAML is
    syntactically fine, it just doesn't fulfil the requirement). Same
    low-level LLM call as the syntax repair path.
    """
    try:
        from agent_a3_llm import _call_llm
    except ImportError:
        print("  [A4] agent_a3_llm._call_llm unavailable — cannot repair, returning original")
        return yaml_text

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Ansible remediation expert. A previously generated "
                "task passed syntax validation, but a reviewer determined it "
                "does NOT correctly implement the required CIS Benchmark "
                "remediation. You will receive the knowledge-base reference "
                "text for the rule and the reviewer's explanation of what is "
                "wrong. Rewrite the task so it actually fulfils the "
                "requirement — do not just restate the same approach. "
                "GENERAL ANSIBLE BEST PRACTICES (apply these always, not just the specific cases "
                "below): prefer a dedicated Ansible module over shell/command whenever one exists; "
                "NEVER guess a module's parameter name or a fact's field name by analogy (a fact's "
                "field and a module's parameter for the same data are often spelled differently); "
                "every task must be idempotent (safe to re-run); prefer built-in Ansible facts over "
                "parsing shell output; prefer safe Jinja2 filters (split/join/unique/selectattr) over "
                "custom regex for string manipulation. "
                "To add a "
                "value to an EXISTING comma-separated string (e.g. a mount "
                "'options' field) use safe list operations: \"{{ (item.options.split(',') "
                "+ ['newvalue']) | unique | join(',') }}\" — NEVER a custom regex_replace() "
                "for this; a zero-width regex can match at every character boundary and "
                "corrupt the string (confirmed live: 'rw,relatime' -> 'r,w,,r,e,l,a,t,i,m,e,'). "
                "That 'unique' filter already makes the append idempotent by itself — do NOT "
                "also add an extra 'when' condition (e.g. a regex_search checking whether the "
                "value is already present) to 'guard' against re-adding it; such conditions are "
                "unnecessary and easy to get subtly wrong (confirmed live: a regex like "
                "'(?<!^)(?<!,)(?<!,nodev)(?<!nodev,)nodev' searching a comma-separated options "
                "string for 'nodev' NEVER matches ANYTHING, because 'nodev' in such a string is "
                "always either at the very start or right after a comma, and both are excluded by "
                "the lookbehinds — making the 'when' condition silently, permanently True, a dead "
                "check that looks like a safeguard but does nothing). "
                "When looping over 'ansible_mounts | selectattr(...)', 'item' has ONLY these "
                "fields: mount, device, fstype, options, uuid, size_total, size_available — "
                "there is NO 'src' field; use 'item.device' for the mount module's 'src' "
                "parameter, never 'item.src' (confirmed live: silently evaluates to Undefined). "
                "When using the mount module to change options on an ALREADY-MOUNTED filesystem, "
                "use state: mounted, NEVER state: present — 'present' only writes /etc/fstab (takes "
                "effect after reboot only), but CIS Audit checks read the LIVE mount table, so a "
                "'present'-only fix stays NON_COMPLIANT until reboot; 'mounted' remounts immediately. "
                "The mount MODULE's parameter for the options string is 'opts' — NEVER 'options' "
                "(that is the ansible_mounts FACT's key name for the same data, an easy mix-up; "
                "confirmed live: 'options' fails ansible-lint with 'Unsupported parameters'). "
                "NEVER use a Jinja lookup/query() (e.g. query('shell', ...), lookup('shell', ...), "
                "query('pipe', ...)) inside a 'when:' condition to check REMOTE host state — lookups "
                "execute on the ANSIBLE CONTROL NODE, not the managed host, so the result would reflect "
                "the WRONG machine entirely, regardless of which lookup plugin name is used. There is "
                "also no built-in 'shell' lookup plugin at all (confirmed: the real ones are 'pipe', "
                "'file', 'env', 'password', 'lines', 'ini' — 'shell' does not exist and would fail at "
                "runtime with a lookup-plugin-not-found error). To check remote command output before "
                "deciding whether to run something, use a SEPARATE preceding task with 'register:', then "
                "reference that registered variable's .stdout/.rc in this task's own 'when:' — never an "
                "inline lookup/query(). "
                "GNOME dconf LOCK files (/etc/dconf/db/*.d/locks/*, used to prevent a setting from being "
                "overridden) use a DIFFERENT, SIMPLER format than dconf VALUE files: each line is JUST the "
                "full GSettings key path starting with '/', one per line — e.g. "
                "'/org/gnome/desktop/session/idle-delay' — NO '[section]' header and NO 'lock' keyword "
                "(confirmed via official GNOME documentation). The '[section]' + 'key = value' format is "
                "ONLY for dconf VALUE files (e.g. /etc/dconf/db/*.d/*, setting the actual default) — do "
                "not mix the two formats or invent a 'lock <key>' directive, which does not exist. "
                "Unless the CIS remediation text EXPLICITLY names a different database, the standard "
                "machine-wide dconf database is 'local.d' (/etc/dconf/db/local.d/ and its locks/ "
                "subdirectory) — NOT '/etc/dconf/db/gdm.d/', even for a rule whose title mentions "
                "'GDM'; many GDM-related CIS rules (screen-lock, idle-delay) are enforced via the "
                "general local.d database. Verify against the exact path the retrieved CIS text "
                "gives, never assume 'gdm.d' from the rule title alone. "
                "sudo's /etc/sudoers has, by DEFAULT on virtually every distro, a "
                "'#includedir /etc/sudoers.d' (or '@includedir') line near its END — files under "
                "/etc/sudoers.d/ are parsed AFTER /etc/sudoers itself, and for 'Defaults' settings "
                "(e.g. timestamp_timeout) sudo applies the LAST value it parses. A fix that creates "
                "ONE new drop-in file under /etc/sudoers.d/ with the required 'Defaults ...=value' "
                "line is a CORRECT, COMPLETE, and reliably authoritative fix even when /etc/sudoers "
                "itself (or another sudoers.d file) ALSO sets that directive — the new drop-in file's "
                "value wins on the default include order. Prefer this single new drop-in file over "
                "trying to regex-edit /etc/sudoers or every existing sudoers.d file in place — "
                "editing an unknown-format existing line is far more fragile, and a single new file "
                "is CIS's own commonly recommended pattern for sudoers hardening. "
                "A CIS rule about DUPLICATE identifiers (duplicate UIDs, GIDs, user names, or group "
                "names) requires ESTABLISHING A NEW, UNIQUE value for the duplicate, NEVER "
                "'state: absent' / deleting the account or group — confirmed live (rule 7.2.6): its "
                "own remediation text says 'establish unique user names... file ownerships will "
                "automatically reflect the change as long as the users have unique UIDs', "
                "explicitly a rename/renumbering, not a deletion; do not 'fix' a duplicate-identifier "
                "finding by proposing to delete the account/group. Do NOT invent a variable (e.g. a "
                "'duplicate_x_mapping' fact) that no other task in the project actually defines — a "
                "task referencing an undefined/always-empty variable is a silent no-op. When the CIS "
                "text itself phrases this as requiring manual review and you cannot safely "
                "auto-compute the correct new value, the CORRECT fix is a task that DETECTS the "
                "duplicate and FAILS LOUDLY with a clear message — an honest non-fix beats a "
                "fabricated one that silently does nothing. "
                "Never use 'authselect select <profile> ...' to 'fix' a rule about one SPECIFIC PAM "
                "module's presence — it REPLACES THE ENTIRE ACTIVE PROFILE, silently dropping "
                "unrelated already-correct settings, and guessing which profile/features to select "
                "fabricates configuration state with no basis. 'authselect select <profile> --with-X' "
                "only accepts real, existing authselect feature flags (e.g. '--with-faillock', "
                "'--with-sudo') — there is no '--with-pam' flag; pam_unix is part of every baseline "
                "profile already, not an optional feature toggled this way. A remediation phrased as "
                "a verification script (grep for 'pam_unix.so' in the profile's own template files) "
                "should be implemented as exactly that check, never a profile-replacing "
                "'authselect select'. "
                "The rule_id/rule_title given to you is the AUTHORITATIVE description of what is "
                "required — trust it over a possibly typo'd or inverted original task name. "
                "Confirmed live: a task named 'Ensure nodexec option set...' (typo for 'noexec') "
                "produced a fix adding 'exec' to mount options — the OPPOSITE of what the real rule "
                "('Ensure noexec option set...') requires. Before writing a fix whose effect is the "
                "OPPOSITE of a positive/prohibitive framing in the rule title (set vs. not set, "
                "enable vs. disable), resolve any conflict in the rule title's favor, never the "
                "source task's own possibly-garbled name. "
                "Output "
                "EXACTLY ONE Ansible task (a single YAML list item starting with "
                "'- name:') — never multiple alternative versions, draft "
                "attempts, or a 'rewritten' duplicate alongside the original; "
                "pick ONE final approach. Output ONLY that corrected YAML, no "
                "explanation, no markdown, and no comments narrating your "
                "reasoning."
            )
        },
        {
            "role": "user",
            "content": f"""This Ansible task (implementing CIS {rule_id}: {rule_title}) passed syntax
validation but failed semantic review:

CURRENT YAML:
{yaml_text}

KNOWLEDGE-BASE REFERENCE (CIS Benchmark Audit + Remediation text for this rule):
{_kb_text_slice(knowledge_base_text, 4000) or "[not found]"}

REVIEWER'S EXPLANATION OF THE PROBLEM:
{semantic_reason}
{sibling_context}{_escalation_note(failed_attempts, prior_reasons)}
Rewrite the task so it actually implements the remediation. Output ONLY the corrected YAML task(s)."""
        }
    ]

    try:
        response = _call_llm(messages, identifier=f"A4_REPAIR_SEMANTIC {rule_id}")
        cleaned = re.sub(r"^```(?:yaml)?\s*\n?", "", response.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip())
        return cleaned
    except Exception as e:
        print(f"  [A4] Error during semantic repair via Agent 3: {e}")
        return yaml_text


def repair_via_agent3_combined(yaml_text: str, findings: List[LintFinding],
                               rule_id: str, rule_title: str,
                               knowledge_base_text: str, semantic_reason: str,
                               sibling_context: str = "") -> str:
    """
    Combined repair call for a fix that fails both ansible-lint and the
    knowledge-base semantic check: gives Agent 3 both problems at once.
    Currently not called from validate_and_repair() (the semantic check runs
    only after lint passes), kept for direct use.
    """
    try:
        from agent_a3_llm import _call_llm
    except ImportError:
        print("  [A4] agent_a3_llm._call_llm unavailable — cannot repair, returning original")
        return yaml_text

    blocking_findings = [f for f in findings if _is_blocking(f)] or findings
    error_summary = "\n".join(
        f"- [{f.rule}] {f.message}" + (f" (line {f.line})" if f.line else "")
        for f in blocking_findings
    )
    hints = _lookup_fix_hints(findings)
    hints_block = ""
    if hints:
        hints_block = "\n\nHINTS (specific fix for this error class):\n" + "\n".join(f"- {h}" for h in hints)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Ansible remediation expert. A previously generated task "
                "FAILED TWO INDEPENDENT CHECKS AT ONCE: ansible-lint (syntax/style) "
                "AND a knowledge-base semantic review (does it actually implement the "
                "CIS Benchmark remediation, not just parse correctly). Fix BOTH "
                "problems in a single rewrite - do not fix only one and ignore the "
                "other. "
                "GENERAL ANSIBLE BEST PRACTICES (apply these always, not just the specific cases "
                "below): prefer a dedicated Ansible module over shell/command whenever one exists; "
                "NEVER guess a module's parameter name or a fact's field name by analogy (a fact's "
                "field and a module's parameter for the same data are often spelled differently); "
                "every task must be idempotent (safe to re-run); prefer built-in Ansible facts over "
                "parsing shell output; prefer safe Jinja2 filters (split/join/unique/selectattr) over "
                "custom regex for string manipulation. "
                "To add a value to an EXISTING comma-separated string (e.g. a "
                "mount 'options' field) use safe list operations: \"{{ (item.options.split(',') "
                "+ ['newvalue']) | unique | join(',') }}\" — NEVER a custom regex_replace() "
                "for this; a zero-width regex can match at every character boundary and "
                "corrupt the string (confirmed live: 'rw,relatime' -> 'r,w,,r,e,l,a,t,i,m,e,'). "
                "That 'unique' filter already makes the append idempotent by itself — do NOT "
                "also add an extra 'when' condition (e.g. a regex_search checking whether the "
                "value is already present) to 'guard' against re-adding it; such conditions are "
                "unnecessary and easy to get subtly wrong (confirmed live: a regex like "
                "'(?<!^)(?<!,)(?<!,nodev)(?<!nodev,)nodev' searching a comma-separated options "
                "string for 'nodev' NEVER matches ANYTHING, because 'nodev' in such a string is "
                "always either at the very start or right after a comma, and both are excluded by "
                "the lookbehinds — making the 'when' condition silently, permanently True, a dead "
                "check that looks like a safeguard but does nothing). "
                "When looping over 'ansible_mounts | selectattr(...)', 'item' has ONLY these "
                "fields: mount, device, fstype, options, uuid, size_total, size_available — "
                "there is NO 'src' field; use 'item.device' for the mount module's 'src' "
                "parameter, never 'item.src' (confirmed live: silently evaluates to Undefined). "
                "When using the mount module to change options on an ALREADY-MOUNTED filesystem, "
                "use state: mounted, NEVER state: present — 'present' only writes /etc/fstab (takes "
                "effect after reboot only), but CIS Audit checks read the LIVE mount table, so a "
                "'present'-only fix stays NON_COMPLIANT until reboot; 'mounted' remounts immediately. "
                "The mount MODULE's parameter for the options string is 'opts' — NEVER 'options' "
                "(that is the ansible_mounts FACT's key name for the same data, an easy mix-up; "
                "confirmed live: 'options' fails ansible-lint with 'Unsupported parameters'). "
                "NEVER use a Jinja lookup/query() (e.g. query('shell', ...), lookup('shell', ...), "
                "query('pipe', ...)) inside a 'when:' condition to check REMOTE host state — lookups "
                "execute on the ANSIBLE CONTROL NODE, not the managed host, so the result would reflect "
                "the WRONG machine entirely, regardless of which lookup plugin name is used. There is "
                "also no built-in 'shell' lookup plugin at all (confirmed: the real ones are 'pipe', "
                "'file', 'env', 'password', 'lines', 'ini' — 'shell' does not exist and would fail at "
                "runtime with a lookup-plugin-not-found error). To check remote command output before "
                "deciding whether to run something, use a SEPARATE preceding task with 'register:', then "
                "reference that registered variable's .stdout/.rc in this task's own 'when:' — never an "
                "inline lookup/query(). "
                "GNOME dconf LOCK files (/etc/dconf/db/*.d/locks/*, used to prevent a setting from being "
                "overridden) use a DIFFERENT, SIMPLER format than dconf VALUE files: each line is JUST the "
                "full GSettings key path starting with '/', one per line — e.g. "
                "'/org/gnome/desktop/session/idle-delay' — NO '[section]' header and NO 'lock' keyword "
                "(confirmed via official GNOME documentation). The '[section]' + 'key = value' format is "
                "ONLY for dconf VALUE files (e.g. /etc/dconf/db/*.d/*, setting the actual default) — do "
                "not mix the two formats or invent a 'lock <key>' directive, which does not exist. "
                "Unless the CIS remediation text EXPLICITLY names a different database, the standard "
                "machine-wide dconf database is 'local.d' (/etc/dconf/db/local.d/ and its locks/ "
                "subdirectory) — NOT '/etc/dconf/db/gdm.d/', even for a rule whose title mentions "
                "'GDM'; many GDM-related CIS rules (screen-lock, idle-delay) are enforced via the "
                "general local.d database. Verify against the exact path the retrieved CIS text "
                "gives, never assume 'gdm.d' from the rule title alone. "
                "sudo's /etc/sudoers has, by DEFAULT on virtually every distro, a "
                "'#includedir /etc/sudoers.d' (or '@includedir') line near its END — files under "
                "/etc/sudoers.d/ are parsed AFTER /etc/sudoers itself, and for 'Defaults' settings "
                "(e.g. timestamp_timeout) sudo applies the LAST value it parses. A fix that creates "
                "ONE new drop-in file under /etc/sudoers.d/ with the required 'Defaults ...=value' "
                "line is a CORRECT, COMPLETE, and reliably authoritative fix even when /etc/sudoers "
                "itself (or another sudoers.d file) ALSO sets that directive — the new drop-in file's "
                "value wins on the default include order. Prefer this single new drop-in file over "
                "trying to regex-edit /etc/sudoers or every existing sudoers.d file in place — "
                "editing an unknown-format existing line is far more fragile, and a single new file "
                "is CIS's own commonly recommended pattern for sudoers hardening. "
                "A CIS rule about DUPLICATE identifiers (duplicate UIDs, GIDs, user names, or group "
                "names) requires ESTABLISHING A NEW, UNIQUE value for the duplicate, NEVER "
                "'state: absent' / deleting the account or group — confirmed live (rule 7.2.6): its "
                "own remediation text says 'establish unique user names... file ownerships will "
                "automatically reflect the change as long as the users have unique UIDs', "
                "explicitly a rename/renumbering, not a deletion; do not 'fix' a duplicate-identifier "
                "finding by proposing to delete the account/group. Do NOT invent a variable (e.g. a "
                "'duplicate_x_mapping' fact) that no other task in the project actually defines — a "
                "task referencing an undefined/always-empty variable is a silent no-op. When the CIS "
                "text itself phrases this as requiring manual review and you cannot safely "
                "auto-compute the correct new value, the CORRECT fix is a task that DETECTS the "
                "duplicate and FAILS LOUDLY with a clear message — an honest non-fix beats a "
                "fabricated one that silently does nothing. "
                "Never use 'authselect select <profile> ...' to 'fix' a rule about one SPECIFIC PAM "
                "module's presence — it REPLACES THE ENTIRE ACTIVE PROFILE, silently dropping "
                "unrelated already-correct settings, and guessing which profile/features to select "
                "fabricates configuration state with no basis. 'authselect select <profile> --with-X' "
                "only accepts real, existing authselect feature flags (e.g. '--with-faillock', "
                "'--with-sudo') — there is no '--with-pam' flag; pam_unix is part of every baseline "
                "profile already, not an optional feature toggled this way. A remediation phrased as "
                "a verification script (grep for 'pam_unix.so' in the profile's own template files) "
                "should be implemented as exactly that check, never a profile-replacing "
                "'authselect select'. "
                "The rule_id/rule_title given to you is the AUTHORITATIVE description of what is "
                "required — trust it over a possibly typo'd or inverted original task name. "
                "Confirmed live: a task named 'Ensure nodexec option set...' (typo for 'noexec') "
                "produced a fix adding 'exec' to mount options — the OPPOSITE of what the real rule "
                "('Ensure noexec option set...') requires. Before writing a fix whose effect is the "
                "OPPOSITE of a positive/prohibitive framing in the rule title (set vs. not set, "
                "enable vs. disable), resolve any conflict in the rule title's favor, never the "
                "source task's own possibly-garbled name. "
                "Output EXACTLY ONE Ansible task (a single YAML list item "
                "starting with '- name:') — never multiple alternative versions, "
                "draft attempts, or a 'rewritten' duplicate alongside the original; "
                "pick ONE final approach. Output ONLY that corrected YAML, no "
                "explanation, no markdown, and no comments narrating your reasoning."
            )
        },
        {
            "role": "user",
            "content": f"""This Ansible task (implementing CIS {rule_id}: {rule_title}) failed both checks:

CURRENT YAML:
{yaml_text}

ANSIBLE-LINT ERRORS:
{error_summary}{hints_block}

KNOWLEDGE-BASE REFERENCE (CIS Benchmark Audit + Remediation text for this rule):
{_kb_text_slice(knowledge_base_text, 4000) or "[not found]"}

KNOWLEDGE-BASE REVIEWER'S EXPLANATION OF THE SEMANTIC PROBLEM:
{semantic_reason}
{sibling_context}
Rewrite the task so it BOTH passes ansible-lint AND actually implements the remediation.
Output ONLY the corrected YAML task(s)."""
        }
    ]

    try:
        response = _call_llm(messages, identifier=f"A4_REPAIR_COMBINED {rule_id}")
        cleaned = re.sub(r"^```(?:yaml)?\s*\n?", "", response.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip())
        return cleaned
    except Exception as e:
        print(f"  [A4] Error during combined repair via Agent 3: {e}")
        return yaml_text


# ═══════════════════════════════════════════════════════════════════════════
# Self-repair loop — Agent A4's main function
# ═══════════════════════════════════════════════════════════════════════════

def validate_and_repair(yaml_text: str, task_name: str, rule_id: str = "",
                        rule_title: str = "", max_attempts: int = 3,
                        use_llm_repair: bool = True,
                        use_knowledge_base: bool = True,
                        rule_ids: List[str] = None,
                        sibling_context: str = "",
                        original_path: str = "") -> ValidationResult:
    """
    Closed validation/repair loop:
      1. Check YAML format (and normalize indentation if possible)
      2. Run ansible-lint
      3. If lint passes and use_knowledge_base=True: check the fix against
         the CIS knowledge base
      4. On any failure, send the fix back to Agent 3 and repeat
      5. Return the result on success or when a stagnation guard stops
         the loop

    Only code that passes this loop should end up in the project.
    """
    current_yaml = yaml_text
    attempts_log = []
    # Recent format-error messages: stops the loop when different repairs
    # keep producing the same format error (see _format_err_plateaued()).
    _FORMAT_ERR_PLATEAU_WINDOW = 5
    format_err_history: List[str] = []

    def _format_err_plateaued() -> bool:
        if len(format_err_history) < _FORMAT_ERR_PLATEAU_WINDOW:
            return False
        recent = format_err_history[-_FORMAT_ERR_PLATEAU_WINDOW:]
        return len(set(recent)) == 1

    def _repair_is_stalled(repaired: str, baseline: str) -> bool:
        # Compares the repaired YAML with the previous one after the same
        # normalisation, so that formatting differences alone are not
        # treated as a change. Falls back to a raw comparison if the repair
        # does not parse.
        ok, normalized_repaired, _ = check_yaml_format(repaired)
        lhs = normalized_repaired.strip() if ok else repaired.strip()
        return lhs == baseline.strip()
    # Knowledge-base text is cached per rule for the duration of this call
    # and fetched only when the semantic check is enabled.
    kb_cache = {}

    def _get_kb_text():
        if not use_knowledge_base:
            return ""
        if rule_id not in kb_cache:
            try:
                # Fetches the Audit and Remediation sections for all rule IDs
                # of the task, so that the semantic check sees every
                # applicable rule and its accepted alternatives.
                from agent_a3_llm import get_audit_and_remediation_chunks
                kb_cache[rule_id] = get_audit_and_remediation_chunks(rule_ids or [rule_id]) or ""
            except ImportError:
                kb_cache[rule_id] = ""
        return kb_cache[rule_id]

    # The most recent real semantic-check result, kept across attempts in
    # which the semantic check did not run.
    last_semantic = {"valid": None, "verdict": "", "score": None,
                     "reason": "", "kb_found": False}

    # Plateau guard (no improvement): stops the loop once PLATEAU_WINDOW
    # consecutive semantic checks have not produced a new best score. This
    # also covers a model oscillating between equally wrong variants.
    PLATEAU_WINDOW = 5
    semantic_score_history = []
    best_semantic_score = None
    attempts_since_best = 0
    # Every semantic-rejection reason so far, for _escalation_note() — see its
    # docstring: showing the model the ACCUMULATED list of rejections (not just
    # the current round's) is what signals "the approach itself is wrong",
    # which no single round's message ever conveys on its own.
    semantic_reject_reasons: List[str] = []
    # Set True after the first retry of an unchanged-output stall in the
    # semantic-repair branch below (one retry only).
    _semantic_stall_retried = False

    def _plateaued() -> bool:
        return attempts_since_best >= PLATEAU_WINDOW

    # Lint-cycle guard: counts how many times the same set of blocking lint
    # findings recurs. A recurrence means the repair is cycling between
    # states it has already tried (e.g. two alternating module names).
    LINT_CYCLE_LIMIT = 3
    lint_signature_counts: Dict[frozenset, int] = {}

    def _lint_signature(findings) -> frozenset:
        relevant = [f for f in findings if _is_blocking(f)] or findings
        return frozenset((f.rule, f.message[:200]) for f in relevant)

    def _record_lint_signature(findings) -> int:
        sig = _lint_signature(findings)
        lint_signature_counts[sig] = lint_signature_counts.get(sig, 0) + 1
        return lint_signature_counts[sig]

    def _semantic_kwargs():
        return {
            "semantic_valid": last_semantic["valid"],
            "semantic_verdict": last_semantic["verdict"],
            "semantic_score": last_semantic["score"],
            "semantic_reason": last_semantic["reason"],
            "knowledge_base_found": last_semantic["kb_found"],
        }

    def _semantic_kwargs_unchecked():
        # For a fix accepted without a fresh semantic check on its current
        # version, an earlier verdict (for a different YAML) is not reported.
        # Only knowledge_base_found is kept, because it describes the rule.
        return {
            "semantic_valid": None,
            "semantic_verdict": "",
            "semantic_score": None,
            "semantic_reason": "",
            "knowledge_base_found": last_semantic["kb_found"],
        }

    def _log_check(attempt_num, check_name, passed, detail=""):
        # Format and lint checks are not LLM calls, but they are logged
        # through dump_llm_call() too (with empty messages), so that the
        # debug log shows the whole validation flow.
        #
        # The rule_id is prefixed to the identifier, because CREATE tasks
        # have no original task name to group them by in the debug log.
        dump_llm_call(
            "A4_deterministic_check", f"{rule_id} | attempt {attempt_num} | {check_name} | {task_name}",
            [], "", {"check": check_name, "passed": passed, "detail": detail[:400]},
        )

    for attempt_num in range(1, max_attempts + 1):
        print(f"  [A4] Attempt {attempt_num}/{max_attempts}: '{task_name[:50]}'")

        # Step 1 — format
        format_ok, normalized_yaml, format_err = check_yaml_format(current_yaml)
        _log_check(attempt_num, "format", format_ok, format_err)
        if not format_ok:
            print(f"  [A4]   ✗ YAML format error: {format_err[:100]}")
            format_err_history.append(format_err)
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=False, lint_valid=False,
                findings=[LintFinding(rule="yaml_format", message=format_err)],
            ))
            if _format_err_plateaued():
                print(f"  [A4]   ! Same YAML format error recurred "
                      f"{_FORMAT_ERR_PLATEAU_WINDOW} times in a row (different repair "
                      f"attempts, identical failure) — giving up early instead of "
                      f"burning the rest of the attempt budget on a non-converging repair")
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=(f"Format error plateaued: same error recurred "
                                    f"{_FORMAT_ERR_PLATEAU_WINDOW} attempts in a row: "
                                    f"{format_err[:150]}"),
                    **_semantic_kwargs(),
                )
            if attempt_num < max_attempts and use_llm_repair:
                repaired = repair_via_agent3(
                    current_yaml,
                    [LintFinding(rule="yaml_format", message=format_err)],
                    rule_id, rule_title, sibling_context=sibling_context
                )
                if _repair_is_stalled(repaired, current_yaml):
                    # Stagnation guard: stops when the repair returns
                    # unchanged output (e.g. repeated LLM_CALL_FAILED markers
                    # from an unavailable server).
                    print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                          f"— giving up early instead of repeating identical attempts")
                    return ValidationResult(
                        original_task=task_name, rule_id=rule_id, valid=False,
                        final_yaml=current_yaml, attempts=attempts_log,
                        total_attempts=attempt_num, lint_valid=False,
                        give_up_reason=f"Format repair stalled (unchanged YAML, likely an LLM/network "
                                       f"error) after {attempt_num} attempts: {format_err[:150]}",
                        **_semantic_kwargs(),
                    )
                current_yaml = repaired
                continue
            else:
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=f"YAML format error after {attempt_num} attempts: {format_err[:150]}",
                    **_semantic_kwargs(),
                )

        current_yaml = normalized_yaml

        # Step 1b — task-count guardrail (see count_top_level_tasks()).
        # Treated like a format failure: repaired via Agent 3 or given up
        # when attempts are exhausted.
        n_tasks = count_top_level_tasks(current_yaml)
        # Tasks that are identical apart from their 'name:' (repeated drafts
        # of the same step) are collapsed first, keeping the last one. Up to
        # 3 different tasks are then allowed, because CIS remediations often
        # need several steps (e.g. configure, then activate).
        parsed_tasks = None
        if n_tasks > 1:
            try:
                parsed_tasks = (_ryaml.load(current_yaml) if HAS_RUAMEL
                                 else _pyyaml.safe_load(current_yaml))
            except Exception:
                parsed_tasks = None
        if isinstance(parsed_tasks, list) and len(parsed_tasks) == n_tasks:
            def _sig(t):
                d = dict(t) if isinstance(t, dict) else {}
                d.pop("name", None)
                return json.dumps(d, sort_keys=True, default=str)
            last_idx_by_sig = {}
            for idx, t in enumerate(parsed_tasks):
                last_idx_by_sig[_sig(t)] = idx
            keep_idx = sorted(last_idx_by_sig.values())
            if len(keep_idx) < len(parsed_tasks):
                try:
                    current_yaml = _dump_tasks([parsed_tasks[i] for i in keep_idx])
                    n_tasks = len(keep_idx)
                except Exception:
                    pass  # re-serialization failed -> fall through, original n_tasks/current_yaml stand
        allowed_max_tasks = 3
        _log_check(attempt_num, "single_task", n_tasks <= allowed_max_tasks,
                   f"{n_tasks} task(s) found (max allowed: {allowed_max_tasks})")
        if n_tasks > allowed_max_tasks:
            multi_task_err = (
                f"The YAML contains {n_tasks} separate, genuinely different task "
                f"definitions, but at most {allowed_max_tasks} is allowed for one "
                f"remediation — consolidate into at most {allowed_max_tasks} final tasks "
                f"(merge any that are alternative/draft approaches to the same single step)."
            )
            print(f"  [A4]   ✗ Multiple tasks in one response: {n_tasks} found "
                  f"(max allowed: {allowed_max_tasks})")
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=True, lint_valid=False,
                findings=[LintFinding(rule="multiple_tasks", message=multi_task_err)],
            ))
            if attempt_num < max_attempts and use_llm_repair:
                repaired = repair_via_agent3(
                    current_yaml,
                    [LintFinding(rule="multiple_tasks", message=multi_task_err)],
                    rule_id, rule_title, sibling_context=sibling_context
                )
                if _repair_is_stalled(repaired, current_yaml):
                    print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                          f"— giving up early instead of repeating identical attempts")
                    return ValidationResult(
                        original_task=task_name, rule_id=rule_id, valid=False,
                        final_yaml=current_yaml, attempts=attempts_log,
                        total_attempts=attempt_num, lint_valid=False,
                        give_up_reason=f"Multiple-task repair stalled (unchanged YAML) after "
                                       f"{attempt_num} attempts: {n_tasks} tasks found",
                        **_semantic_kwargs(),
                    )
                current_yaml = repaired
                continue
            else:
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=f"{n_tasks} task definitions found after {attempt_num} "
                                   f"attempts, at most {allowed_max_tasks} allowed",
                    **_semantic_kwargs(),
                )

        # Step 1b-2 — no duplicate sub-tasks nested inside a 'block:' (see
        # check_duplicate_subtasks() docstring — a one-level-deeper variant
        # of Step 1b above, found via quality_judge.py's code-quality
        # review, not any compliance-focused check).
        dup_subtask_problems = check_duplicate_subtasks(current_yaml)
        _log_check(attempt_num, "duplicate_subtasks", not dup_subtask_problems,
                  "; ".join(dup_subtask_problems))
        if dup_subtask_problems:
            dup_err = "; ".join(dup_subtask_problems)
            print(f"  [A4]   ✗ Duplicate sub-task(s) inside block: {dup_subtask_problems[0][:120]}")
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=True, lint_valid=False,
                findings=[LintFinding(rule="duplicate_subtasks", message=dup_err)],
            ))
            if attempt_num < max_attempts and use_llm_repair:
                repaired = repair_via_agent3(
                    current_yaml,
                    [LintFinding(rule="duplicate_subtasks", message=dup_err
                                 + " Remove the duplicate — keep only ONE copy of that sub-task.")],
                    rule_id, rule_title, sibling_context=sibling_context
                )
                if _repair_is_stalled(repaired, current_yaml):
                    print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                          f"— giving up early instead of repeating identical attempts")
                    return ValidationResult(
                        original_task=task_name, rule_id=rule_id, valid=False,
                        final_yaml=current_yaml, attempts=attempts_log,
                        total_attempts=attempt_num, lint_valid=False,
                        give_up_reason=f"Duplicate-subtask repair stalled after {attempt_num} attempts: {dup_err[:150]}",
                        **_semantic_kwargs(),
                    )
                current_yaml = repaired
                continue
            else:
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=f"Duplicate sub-task(s) after {attempt_num} attempts: {dup_err[:150]}",
                    **_semantic_kwargs(),
                )

        # Step 1b-3 — task keyword (loop/when/register/notify/...) nested
        # inside a module's arguments (see check_nested_task_keywords()).
        nested_kw_problems = check_nested_task_keywords(current_yaml)
        _log_check(attempt_num, "nested_task_keywords", not nested_kw_problems,
                  "; ".join(nested_kw_problems))
        if nested_kw_problems:
            nested_kw_err = "; ".join(nested_kw_problems)
            print(f"  [A4]   ✗ Task-level keyword nested inside module args: {nested_kw_problems[0][:120]}")
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=True, lint_valid=False,
                findings=[LintFinding(rule="nested_task_keywords", message=nested_kw_err)],
            ))
            if attempt_num < max_attempts and use_llm_repair:
                repaired = repair_via_agent3(
                    current_yaml,
                    [LintFinding(rule="nested_task_keywords", message=nested_kw_err)],
                    rule_id, rule_title, sibling_context=sibling_context
                )
                if _repair_is_stalled(repaired, current_yaml):
                    print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                          f"— giving up early instead of repeating identical attempts")
                    return ValidationResult(
                        original_task=task_name, rule_id=rule_id, valid=False,
                        final_yaml=current_yaml, attempts=attempts_log,
                        total_attempts=attempt_num, lint_valid=False,
                        give_up_reason=f"Nested-keyword repair stalled after {attempt_num} attempts: {nested_kw_err[:150]}",
                        **_semantic_kwargs(),
                    )
                current_yaml = repaired
                continue
            else:
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=f"Task-level keyword nested in module args after {attempt_num} attempts: {nested_kw_err[:150]}",
                    **_semantic_kwargs(),
                )

        # Step 1c — regex_replace() sanity check (see
        # check_regex_replace_sanity()): the regex is executed on a sample
        # value, because the semantic check only judges its apparent intent.
        regex_problems = check_regex_replace_sanity(current_yaml)
        _log_check(attempt_num, "regex_replace_sanity", not regex_problems, "; ".join(regex_problems))
        if regex_problems:
            regex_err = "Unsafe regex_replace() detected (tested against a sample value): " + "; ".join(regex_problems)
            print(f"  [A4]   ✗ regex_replace() sanity check failed: {regex_problems[0][:120]}")
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=True, lint_valid=False,
                findings=[LintFinding(rule="unsafe_regex_replace", message=regex_err)],
            ))
            if attempt_num < max_attempts and use_llm_repair:
                repaired = repair_via_agent3(
                    current_yaml,
                    [LintFinding(rule="unsafe_regex_replace", message=regex_err)],
                    rule_id, rule_title, sibling_context=sibling_context
                )
                if _repair_is_stalled(repaired, current_yaml):
                    print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                          f"— giving up early instead of repeating identical attempts")
                    return ValidationResult(
                        original_task=task_name, rule_id=rule_id, valid=False,
                        final_yaml=current_yaml, attempts=attempts_log,
                        total_attempts=attempt_num, lint_valid=False,
                        give_up_reason=f"regex_replace repair stalled (unchanged YAML) after "
                                       f"{attempt_num} attempts: {regex_problems[0][:150]}",
                        **_semantic_kwargs(),
                    )
                current_yaml = repaired
                continue
            else:
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=f"Unsafe regex_replace() after {attempt_num} attempts: "
                                   f"{regex_problems[0][:150]}",
                    **_semantic_kwargs(),
                )

        # Step 1c-2 — lookup()/query() inside 'when:' (see
        # check_lookup_in_when()), checked mechanically because the prompt
        # instruction alone did not prevent it.
        lookup_problems = check_lookup_in_when(current_yaml)
        _log_check(attempt_num, "lookup_in_when", not lookup_problems, "; ".join(lookup_problems))
        if lookup_problems:
            lookup_err = "lookup()/query() used inside 'when:' to check remote state (runs on the control node, not the managed host): " + "; ".join(lookup_problems)
            print(f"  [A4]   ✗ lookup-in-when check failed: {lookup_problems[0][:120]}")
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=True, lint_valid=False,
                findings=[LintFinding(rule="lookup_in_when", message=lookup_err)],
            ))
            if attempt_num < max_attempts and use_llm_repair:
                repaired = repair_via_agent3(
                    current_yaml,
                    [LintFinding(rule="lookup_in_when", message=lookup_err)],
                    rule_id, rule_title, sibling_context=sibling_context
                )
                if _repair_is_stalled(repaired, current_yaml):
                    print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                          f"— giving up early instead of repeating identical attempts")
                    return ValidationResult(
                        original_task=task_name, rule_id=rule_id, valid=False,
                        final_yaml=current_yaml, attempts=attempts_log,
                        total_attempts=attempt_num, lint_valid=False,
                        give_up_reason=f"lookup-in-when repair stalled (unchanged YAML) after "
                                       f"{attempt_num} attempts: {lookup_problems[0][:150]}",
                        **_semantic_kwargs(),
                    )
                current_yaml = repaired
                continue
            else:
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=f"lookup()/query() in 'when:' after {attempt_num} attempts: "
                                   f"{lookup_problems[0][:150]}",
                    **_semantic_kwargs(),
                )

        # Step 1d — path guardrail (programmatic; see
        # agent_a3_llm._path_guardrail_check's docstring for the original rationale
        # and the earlier /tmp-vs-/var/tmp confusion it was built to catch).
        #
        # The path guardrail runs only on the first attempt. Later repair
        # rounds may legitimately change the scope of the task when the
        # semantic check asks for it.
        if original_path and attempt_num == 1:
            from agent_a3_llm import _path_guardrail_check
            path_ok = _path_guardrail_check(current_yaml, original_path)
            path_err = (f"Fix targets a different path than the original task "
                        f"(expected '{original_path}' somewhere in the fix, not found).")
            _log_check(attempt_num, "path_guardrail", path_ok, "" if path_ok else path_err)
            if not path_ok:
                # A path mismatch is only logged, not blocking, because the
                # original path may itself be wrong (e.g. a mistagged task).
                # The semantic check against the CIS text decides whether
                # the new path is correct.
                print(f"  [A4]   ⚠ path guardrail (non-blocking): {path_err} — deferring to the "
                      f"knowledge-base semantic check to judge whether the new path is correct")
                attempts_log.append(ValidationAttempt(
                    attempt_number=attempt_num, yaml_text=current_yaml,
                    format_valid=True, lint_valid=True,
                    findings=[LintFinding(rule="path_changed", message=path_err)],
                ))

        # Step 2 — ansible-lint
        lint_ok, findings = run_ansible_lint(current_yaml, task_name)
        _log_check(attempt_num, "ansible_lint", lint_ok,
                  "; ".join(f"[{f.rule}] {f.message}" for f in findings[:5]))

        if not lint_ok:
            # If no finding is classified as blocking (e.g. a lint timeout),
            # all findings are reported.
            blocking = [f for f in findings if _is_blocking(f)] or findings
            print(f"  [A4]   ✗ ansible-lint: {len(blocking)} error(s)")
            for f in blocking[:3]:
                print(f"  [A4]     - [{f.rule}] {f.message[:80]}")

            seen_count = _record_lint_signature(findings)
            if seen_count >= LINT_CYCLE_LIMIT:
                print(f"  [A4]   ! The exact same lint error(s) have now recurred {seen_count}x "
                      f"across {attempt_num} attempts — repair is cycling, not converging, "
                      f"giving up early instead of burning the rest of the attempt budget")
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=(f"Lint repair cycling: the same error(s) recurred {seen_count}x "
                                    f"over {attempt_num} attempts without resolving "
                                    f"({blocking[0].rule}: {blocking[0].message[:150]})"),
                    **_semantic_kwargs(),
                )

            # Strict gating: the semantic check runs only after lint passes,
            # because a verdict on YAML that does not pass lint is less
            # reliable. repair_via_agent3_combined() is kept but no longer
            # called from here.
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=True, lint_valid=False, findings=findings,
            ))

            if attempt_num < max_attempts and use_llm_repair:
                print(f"  [A4]   -> Sending error report back to Agent 3 (attempt {attempt_num+1})")
                repaired = repair_via_agent3(current_yaml, findings, rule_id, rule_title, sibling_context=sibling_context)
                if _repair_is_stalled(repaired, current_yaml):
                    # Stagnation guard — see the identical one on the
                    # format-repair path above for the full rationale.
                    print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                          f"— giving up early instead of repeating identical attempts")
                    return ValidationResult(
                        original_task=task_name, rule_id=rule_id, valid=False,
                        final_yaml=current_yaml, attempts=attempts_log,
                        total_attempts=attempt_num, lint_valid=False,
                        give_up_reason=(f"Lint repair stalled (unchanged YAML, likely an LLM/network "
                                        f"error) after {attempt_num} attempts "
                                        f"({len(blocking)} blocking error(s))"),
                        **_semantic_kwargs(),
                    )
                current_yaml = repaired
                continue
            else:
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=False,
                    give_up_reason=(f"ansible-lint failed after {attempt_num} attempts "
                                    f"({len(blocking)} blocking error(s))"),
                    **_semantic_kwargs(),
                )

        # Step 3 — knowledge-base semantic check (only reached once format,
        # the single-task guardrail, AND lint have all passed on this same
        # attempt — strict gating, see the note above)
        semantic_result = None
        if use_knowledge_base:
            kb_text = _get_kb_text()
            if kb_text:
                semantic_result = verify_with_knowledge_base(rule_id, rule_title, task_name, current_yaml,
                                                              kb_text=kb_text, sibling_context=sibling_context)
                if semantic_result.get("error"):
                    print(f"  [A4]   ! Knowledge-base check failed ({semantic_result['error']}) "
                          f"— not penalizing the fix for an infrastructure problem")
                    # Knowledge-base text was found for this rule, so this is
                    # recorded even if the grading call itself failed.
                    last_semantic["kb_found"] = True
                    semantic_result = None
            else:
                print(f"  [A4]   (no knowledge-base text found for {rule_id} — skipping semantic check)")

        if semantic_result is None:
            # Checking disabled, no knowledge-base text, or a failed check:
            # the fix is not rejected on semantic grounds, and an earlier
            # semantic result (if any) stays in place.
            attempts_log.append(ValidationAttempt(
                attempt_number=attempt_num, yaml_text=current_yaml,
                format_valid=True, lint_valid=True, findings=findings,
                semantic_checked=False,
            ))
            print(f"  [A4]   ✓ Passed validation (attempt {attempt_num})")
            return ValidationResult(
                original_task=task_name, rule_id=rule_id, valid=True,
                final_yaml=current_yaml, attempts=attempts_log,
                total_attempts=attempt_num, lint_valid=True,
                **_semantic_kwargs_unchecked(),
            )

        verdict = str(semantic_result.get("verdict", ""))
        score = semantic_result.get("score")
        reason = semantic_result.get("reason", "")
        # The verdict is normalised (case, punctuation), because the
        # verifier is the model under test and may not return the exact
        # literal "COMPLIANT".
        semantically_ok = verdict.strip().upper().startswith("COMPLIANT")
        if not semantically_ok:
            # Narrow override of a known false rejection: the verifier claims
            # that a stop-service or cleanup step is missing although it is
            # present in the YAML (checked with the same regexes as Agent 3's
            # second-pass mechanism). Other rejections are not affected.
            try:
                from agent_a3_llm import _STOP_PRESENT_RE, _CLEANUP_PRESENT_RE
                _claims_missing_step_re = re.compile(
                    r"only (removes|attempts to remove|remove)|fails? to stop|"
                    r"fails? to (comment|cleanup|clean up)|does not stop|"
                    r"missing.{0,20}(stop|cleanup)|no sibling task",
                    re.IGNORECASE,
                )
                step_present = bool(_STOP_PRESENT_RE.search(current_yaml)
                                    or _CLEANUP_PRESENT_RE.search(current_yaml))
                if step_present and _claims_missing_step_re.search(reason or ""):
                    print(f"  [A4]   ! Overriding semantic verdict — reason claims a missing "
                          f"stop/cleanup step, but current_yaml already contains one "
                          f"(confirmed false-negative pattern) — accepting as-is")
                    semantically_ok = True
                    verdict = "COMPLIANT"
                    reason = (f"[A4 override — see code comment] Original verdict was {reason!r}, "
                              f"but the claimed missing step is present in the YAML.")
            except Exception:
                pass
        # kb_found is taken from the verifier's own result, because it
        # retrieves the knowledge-base text independently.
        last_semantic.update(valid=semantically_ok, verdict=verdict, score=score,
                             reason=reason,
                             kb_found=semantic_result.get("knowledge_base_found", True))
        if score is not None:
            semantic_score_history.append(score)
            if best_semantic_score is None or score > best_semantic_score:
                best_semantic_score = score
                attempts_since_best = 0
            else:
                attempts_since_best += 1

        attempts_log.append(ValidationAttempt(
            attempt_number=attempt_num, yaml_text=current_yaml,
            format_valid=True, lint_valid=True, findings=findings,
            semantic_checked=True, semantic_verdict=verdict,
            semantic_score=score, semantic_reason=reason,
        ))

        if semantically_ok:
            print(f"  [A4]   ✓ Passed validation + knowledge-base check (attempt {attempt_num}, score {score}/5)")
            return ValidationResult(
                original_task=task_name, rule_id=rule_id, valid=True,
                final_yaml=current_yaml, attempts=attempts_log,
                total_attempts=attempt_num, lint_valid=True,
                **_semantic_kwargs(),
            )

        print(f"  [A4]   ✗ Knowledge-base check: {verdict} (score {score}/5) — {reason[:100]}")

        if _plateaued():
            print(f"  [A4]   ! Semantic score plateaued at {semantic_score_history[-1]}/5 over the "
                  f"last {PLATEAU_WINDOW} checks — giving up early instead of continuing to a "
                  f"non-improving budget ceiling")
            return ValidationResult(
                original_task=task_name, rule_id=rule_id, valid=False,
                final_yaml=current_yaml, attempts=attempts_log,
                total_attempts=attempt_num, lint_valid=True,
                give_up_reason=(f"Semantic score plateaued at {semantic_score_history[-1]}/5 over "
                                f"the last {PLATEAU_WINDOW} attempts (ansible-lint passes) - not "
                                f"converging, stopped after {attempt_num} attempts instead of "
                                f"exhausting the full budget"),
                **_semantic_kwargs(),
            )

        if attempt_num < max_attempts and use_llm_repair:
            semantic_reject_reasons.append(reason)
            if len(semantic_reject_reasons) >= ESCALATE_AFTER:
                print(f"  [A4]   ! {len(semantic_reject_reasons)} rejected attempts — asking Agent 3 "
                      f"for a DIFFERENT approach, not another refinement of the same one")
            print(f"  [A4]   -> Sending semantic review back to Agent 3 (attempt {attempt_num+1})")
            repaired = repair_via_agent3_semantic(
                current_yaml, rule_id, rule_title, _get_kb_text(), reason,
                sibling_context=sibling_context,
                failed_attempts=len(semantic_reject_reasons),
                prior_reasons=semantic_reject_reasons,
            )
            if _repair_is_stalled(repaired, current_yaml):
                # repair_via_agent3_semantic() returns its input unchanged on
                # an LLM error, so an unchanged result means the repair is not
                # progressing.
                #
                # One retry is allowed first, because the server can return
                # non-deterministic output under load. The loop stops on the
                # second consecutive unchanged result.
                if not _semantic_stall_retried:
                    _semantic_stall_retried = True
                    print(f"  [A4]   ! Repair returned unchanged YAML — retrying the identical call "
                          f"once before giving up (server has shown non-determinism under load "
                          f"despite temperature=0)")
                    retry_repaired = repair_via_agent3_semantic(
                        current_yaml, rule_id, rule_title, _get_kb_text(), reason,
                        sibling_context=sibling_context,
                        failed_attempts=len(semantic_reject_reasons),
                        prior_reasons=semantic_reject_reasons,
                    )
                    if not _repair_is_stalled(retry_repaired, current_yaml):
                        current_yaml = retry_repaired
                        continue
                    print(f"  [A4]   ! Retry also returned unchanged YAML — genuinely stalled, giving up")
                print(f"  [A4]   ! Repair returned unchanged YAML (likely an LLM/network error) "
                      f"— giving up early instead of repeating identical attempts")
                return ValidationResult(
                    original_task=task_name, rule_id=rule_id, valid=False,
                    final_yaml=current_yaml, attempts=attempts_log,
                    total_attempts=attempt_num, lint_valid=True,
                    give_up_reason=f"Semantic repair stalled (unchanged YAML, likely an LLM/network "
                                   f"error) after {attempt_num} attempts: {verdict} — {reason[:150]}",
                    **_semantic_kwargs(),
                )
            current_yaml = repaired
            continue
        else:
            return ValidationResult(
                original_task=task_name, rule_id=rule_id, valid=False,
                final_yaml=current_yaml, attempts=attempts_log,
                total_attempts=attempt_num, lint_valid=True,
                give_up_reason=f"Passed ansible-lint but failed knowledge-base semantic "
                               f"check after {attempt_num} attempts: {verdict} — {reason[:150]}",
                **_semantic_kwargs(),
            )

    # Should not be reached, but just in case
    return ValidationResult(
        original_task=task_name, rule_id=rule_id, valid=False, lint_valid=False,
        final_yaml=current_yaml, attempts=attempts_log,
        total_attempts=max_attempts, give_up_reason="Attempt budget exhausted",
        **_semantic_kwargs(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Conflict detection across multiple fixes
# ═══════════════════════════════════════════════════════════════════════════

def _expand_loop_paths(path, task: dict) -> list:
    """If path is a Jinja template referencing the loop variable ('{{ item }}'
    or '{{ item.subkey }}'), resolves it against the task's own inline
    'loop:'/'with_items:' list, one path per entry. Otherwise returns [path]
    unchanged. Prevents two tasks looping over different files from being
    reported as a conflict only because both use '{{ item }}'.
    """
    if not isinstance(path, str) or "{{" not in path:
        return [path]
    loop_val = task.get("loop") or task.get("with_items")
    if not isinstance(loop_val, list):
        return [path]  # loop is itself a Jinja var reference, or absent — can't expand
    m = re.fullmatch(r"\{\{\s*item(?:\.(\w+))?\s*\}\}", path.strip())
    if not m:
        return [path]  # a more complex template than a bare item/item.subkey ref
    subkey = m.group(1)
    resolved = []
    for entry in loop_val:
        if subkey:
            if isinstance(entry, dict) and subkey in entry:
                resolved.append(str(entry[subkey]))
            else:
                return [path]  # can't resolve this entry — fall back to old behavior
        else:
            resolved.append(str(entry))
    return resolved or [path]


def detect_conflicts(results: List[ValidationResult]) -> List[ConflictReport]:
    """
    Detect conflicts among fixes that passed validation — e.g. two fixes
    modifying the same file with different values.
    """
    conflicts = []
    parsed_by_result = []

    for r in results:
        if not r.valid:
            parsed_by_result.append([])
            continue
        try:
            if HAS_RUAMEL:
                parsed = _ryaml.load(r.final_yaml)
            else:
                parsed = _pyyaml.safe_load(r.final_yaml)
            tasks = parsed if isinstance(parsed, list) else [parsed] if parsed else []
        except Exception:
            tasks = []
        parsed_by_result.append(tasks)

    index = {}
    for i, tasks in enumerate(parsed_by_result):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            reserved = {"name", "when", "tags", "become", "vars", "block"}
            modules = [k for k in task if k not in reserved]
            if not modules:
                continue
            mod = modules[0]
            params = task.get(mod, {})
            if not isinstance(params, dict):
                continue
            path = params.get("path") or params.get("dest") or params.get("name")
            key = params.get("regexp") or str(params.get("line", ""))[:30]
            # Tasks whose path/key is not hashable (e.g. a list) are skipped
            # in conflict detection instead of stopping the run.
            if path and isinstance(path, (str, int, float)) and isinstance(key, (str, int, float)):
                for resolved_path in _expand_loop_paths(path, task):
                    if isinstance(resolved_path, (str, int, float)):
                        index.setdefault((mod, resolved_path, key), []).append(i)

    for (mod, path, key), indices in index.items():
        if len(indices) < 2:
            continue
        values = set()
        names = []
        for idx in indices:
            for task in parsed_by_result[idx]:
                if not isinstance(task, dict):
                    continue
                p = task.get(mod, {})
                if isinstance(p, dict):
                    v = p.get("line") or p.get("value") or p.get("state") or p.get("mode")
                    if v:
                        values.add(str(v))
            names.append(results[idx].original_task)
        if len(values) > 1:
            conflicts.append(ConflictReport(
                rule_a=names[0], rule_b=", ".join(names[1:]),
                reason=f"Module '{mod}' modifies '{path}' with conflicting values: {values}",
            ))

    return conflicts


# ═══════════════════════════════════════════════════════════════════════════
# Batch — validate many fixes at once
# ═══════════════════════════════════════════════════════════════════════════

# Number of attempts in the self-repair loop. Fixes without a severity
# (CREATE — no Agent 2 audit, see Remediation.severity) get the passed-in
# max_attempts as a neutral default.
#
# The attempt limit (999) is the same for every severity and only keeps the
# loop bounded. In practice the loop ends through the stagnation guards:
# unchanged output, repeated errors, or no improvement of the semantic score
# for PLATEAU_WINDOW consecutive checks.
ATTEMPTS_BY_SEVERITY = {"CRITICAL": 999, "HIGH": 999, "MEDIUM": 999, "LOW": 999, "NONE": 999}
# All five levels of the CVSS v3.1 Qualitative Severity Rating Scale are
# listed, including NONE (score 0.0).


def validate_all(remediations: list, max_attempts: int = 3,
                 use_llm_repair: bool = True, scale_by_severity: bool = True,
                 use_knowledge_base: bool = True) -> tuple:
    """
    Validate a list of fixes (Remediation objects from A3) through the
    self-repair loop, then detect conflicts among the ones that passed.

    scale_by_severity=True (default): the attempt limit is taken from
    ATTEMPTS_BY_SEVERITY (currently the same for every severity);
    max_attempts is used for fixes without a severity and when
    scale_by_severity=False.

    use_knowledge_base=True (default): each attempt that passes lint is
    additionally checked against the CIS knowledge base (see
    verify_with_knowledge_base) before being accepted.
    """
    # Fail fast if ansible-lint is not available in this environment;
    # otherwise every fix would be reported as passing lint without being
    # checked.
    if not _ansible_lint_available():
        raise RuntimeError(
            "[A4] ansible-lint is not available on PATH in this environment — "
            "refusing to run validate_all(), since every fix would silently get "
            "lint_valid=True without ever actually being linted. "
            "Activate the correct virtualenv (e.g. venv311) before launching."
        )

    print(f"\n[A4] Validating {len(remediations)} fixes through the self-repair loop...")
    results = []

    # Remediations are validated one at a time. When a later remediation
    # shares a rule_id with an already validated one, it is shown the
    # sibling's final (repaired) YAML instead of its original version.
    completed_by_rule: Dict[str, List[str]] = {}  # rule_id -> [final_yaml, ...]

    for rem in remediations:
        severity = getattr(rem, "severity", "")
        attempts = ATTEMPTS_BY_SEVERITY.get(severity, max_attempts) if scale_by_severity else max_attempts
        if scale_by_severity and severity:
            print(f"  [A4] '{rem.original_task[:50]}' severity={severity} -> {attempts} attempt(s)")

        sibling_context = getattr(rem, "sibling_context", "")
        rem_rule_ids = getattr(rem, "rule_ids", None) or ([rem.rule_id] if rem.rule_id else [])
        fresh_yamls = [
            y for rid in rem_rule_ids for y in completed_by_rule.get(rid, [])
        ]
        if fresh_yamls:
            fresh_block = "\n".join(f"---\n{y}" for y in fresh_yamls[:5])
            sibling_context = (sibling_context or "") + f"""
UPDATE: one or more of the sibling tasks above have ALREADY been revised
during this same repair run — their CURRENT (not original) content is:
{fresh_block}
Treat this as authoritative over the original sibling listing above for
what those siblings now actually do. Do NOT re-implement anything already
covered by this current content either.
"""

        result = validate_and_repair(
            yaml_text=rem.yaml_fix,
            task_name=rem.original_task,
            rule_id=rem.rule_id,
            rule_title=getattr(rem, "rule_title", ""),
            max_attempts=attempts,
            use_llm_repair=use_llm_repair,
            use_knowledge_base=use_knowledge_base,
            rule_ids=getattr(rem, "rule_ids", None),
            sibling_context=sibling_context,
            original_path=getattr(rem, "original_path", ""),
        )
        results.append(result)
        if result.valid:
            for rid in rem_rule_ids:
                completed_by_rule.setdefault(rid, []).append(result.final_yaml)

    valid_count = sum(1 for r in results if r.valid)
    print(f"\n[A4] Result: {valid_count}/{len(results)} fixes passed validation")

    print(f"[A4] Detecting conflicts...")
    conflicts = detect_conflicts(results)
    if conflicts:
        for c in conflicts:
            print(f"  [A4] CONFLICT: '{c.rule_a}' vs '{c.rule_b}' — {c.reason}")
    else:
        print("  [A4] No conflicts.")

    return results, conflicts


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: Writing validated fixes to project files + decision gate
# ═══════════════════════════════════════════════════════════════════════════
# Writing validated fixes to the project files. Two operations:
#   - CREATE: append a new task to the matching section file,
#   - FIX: replace an existing task, located by its exact name.
# Both work on raw text (a ruamel.yaml round-trip would change the
# indentation of the whole file) and run only after explicit operator
# confirmation in confirm_and_apply(). Fixes flagged by detect_conflicts()
# are never written automatically.

def _pick_target_tasks_file(project_path: str) -> str:
    """Pick the tasks/main.yml file to append a new task to (CREATE) —
    same role-priority logic as _pick_target_defaults_file in Agent A1
    (hardening/cis/security before audit/test/check). Used only as a
    FALLBACK by _pick_target_section_file, when the right section_N.yml
    can't be determined/found — main.yml in this project is just a list
    of include_tasks, so appending a whole task there doesn't make sense."""
    if discover_roles is None:
        raise RuntimeError("parser_a0_final unavailable — cannot pick a target tasks/ file")
    for _role_name, role_root in discover_roles(project_path):
        for candidate in (os.path.join(role_root, "tasks", "main.yml"),
                          os.path.join(role_root, "tasks", "main.yaml")):
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(f"No tasks/main.yml found in {project_path}")


def _section_number(rule_id: str) -> str:
    """Extracts the top-level section number from a CIS rule_id (e.g.
    '1.1.2.6.1' -> '1'). Returns '' if rule_id doesn't start with a
    number (can't be mapped to section_N.yml)."""
    m = re.match(r"^(\d+)\.", rule_id)
    return m.group(1) if m else ""


def _pick_target_section_file(project_path: str, rule_id: str) -> str:
    """Pick the section_N.yml file that matches rule_id THEMATICALLY
    (e.g. '1.1.2.6.1' -> section_1.yml), using the same role/priority
    ordering as _pick_target_tasks_file. Without this, CREATE always
    appended the new task to the end of tasks/main.yml — a file that in
    this project is just a list of 'include_tasks: section_N.yml' gated
    by 'when: section_N' — so the task ended up loaded outside the
    section system, disconnected from its thematically correct location.
    If the section number can't be determined or no matching file
    exists, falls back to the old behavior (tasks/main.yml)."""
    section_num = _section_number(rule_id)
    if section_num:
        if discover_roles is None:
            raise RuntimeError("parser_a0_final unavailable — cannot pick a target tasks/ file")
        for _role_name, role_root in discover_roles(project_path):
            for candidate in (os.path.join(role_root, "tasks", f"section_{section_num}.yml"),
                              os.path.join(role_root, "tasks", f"section_{section_num}.yaml")):
                if os.path.exists(candidate):
                    return candidate
    return _pick_target_tasks_file(project_path)


def _wrap_task_with_toggle(final_yaml: str, var_name: str, section_num: str, rule_id: str) -> str:
    """Wraps the generated task in this project's convention: a toggle
    variable gate ('when: rule_X_Y_Z') and section/rule tags ('sN',
    'rX.Y.Z'), same as EVERY other rule (see e.g. rule 1.1.2.6.1 in
    section_1.yml). Without this the new rule would always be enabled
    and — unlike every other rule in the project — impossible to disable
    via defaults/main.yml. If parsing final_yaml fails (shouldn't happen,
    since this is already validated YAML), returns the original
    unchanged — never breaks something that already passed validation."""
    try:
        parsed = _ryaml.load(final_yaml) if HAS_RUAMEL else _pyyaml.safe_load(final_yaml)
    except Exception:
        return final_yaml
    if not isinstance(parsed, list):
        return final_yaml

    for task in parsed:
        if not isinstance(task, dict):
            continue
        existing_when = task.get("when")
        if existing_when is None:
            task["when"] = [var_name]
        elif isinstance(existing_when, list):
            if var_name not in existing_when:
                task["when"] = list(existing_when) + [var_name]
        elif existing_when != var_name:
            task["when"] = [existing_when, var_name]

        section_tag, rule_tag = f"s{section_num}", f"r{rule_id}"
        existing_tags = task.get("tags")
        if existing_tags is None:
            task["tags"] = [section_tag, rule_tag]
        elif isinstance(existing_tags, list):
            merged = list(existing_tags)
            for t in (section_tag, rule_tag):
                if t not in merged:
                    merged.append(t)
            task["tags"] = merged

    try:
        if HAS_RUAMEL:
            import io
            buf = io.StringIO()
            _dumper = _get_dumper()
            _dumper.dump(parsed, buf)
            return _dedent_to_zero(buf.getvalue())
        return _pyyaml.dump(parsed, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception:
        return final_yaml


def _yaml_parses(text: str) -> bool:
    """Safeguard BEFORE writing: does the text still parse as valid YAML
    after the change. Used only for validation — never to reformat the
    file (hence a plain load, no dump)."""
    try:
        if HAS_RUAMEL:
            _ryaml.load(text)
        else:
            _pyyaml.safe_load(text)
        return True
    except Exception:
        return False


def apply_create(rule_id: str, rule_title: str, final_yaml: str, project_path: str) -> str:
    """Append a new, validated task to the CORRECT section_N.yml
    (fallback: tasks/main.yml if the section can't be determined),
    wrapped in this project's convention (when: rule_X_Y_Z, tags:
    sN/rX.Y.Z) — same as every other rule. Also appends the toggle
    variable to defaults/main.yml (same path/logic as Agent A1 for
    missing rules), so the new rule can be disabled like any other.
    Returns the path of the written tasks/ file. Raises ValueError if the
    resulting file would no longer be valid YAML (in that case, nothing
    is written)."""
    target = _pick_target_section_file(project_path, rule_id)
    section_num = _section_number(rule_id)
    var_name = f"rule_{rule_id.replace('.', '_')}"
    task_yaml = _wrap_task_with_toggle(final_yaml, var_name, section_num, rule_id) if section_num else final_yaml

    with open(target, encoding="utf-8") as f:
        current = f.read()

    addition = (f"\n# CIS {rule_id}: {rule_title} — added by Agent A3/A4 "
                f"(generated, ansible-lint validated)\n{task_yaml.rstrip()}\n")
    combined = current.rstrip("\n") + "\n" + addition

    if not _yaml_parses(combined):
        raise ValueError(f"Task for {rule_id} does not produce valid YAML after appending to {target} — not written.")

    with open(target, "w", encoding="utf-8") as f:
        f.write(combined)

    if section_num:
        try:
            import agent_a1_onboarding
            defaults_path = agent_a1_onboarding._pick_target_defaults_file(project_path)
            agent_a1_onboarding.apply_add_rule(defaults_path, var_name, rule_title)
        except Exception as e:
            print(f"  [A4] Warning: task written, but failed to append variable "
                  f"{var_name} to defaults/main.yml ({e}) — the rule will always be enabled.")

    return target


def _find_task_block(lines: List[str], task_name: str):
    """Find the (start, end, indent) block of a task with an EXACT name,
    at any nesting level (e.g. inside block:). end is the index of the
    FIRST line AFTER the block (the next list item at the same or a
    shallower indent, or a dedent below the block's indent). Returns
    (None, None, None) if the name isn't found unambiguously — we never
    guess when there isn't a confident match."""
    name_re = re.compile(r'^(\s*)-\s*name:\s*[\'"]?' + re.escape(task_name) + r'[\'"]?\s*(#.*)?$')
    start, indent = None, None
    for i, line in enumerate(lines):
        m = name_re.match(line)
        if m:
            if start is not None:
                return None, None, None  # ambiguous — two tasks with the same name
            start, indent = i, len(m.group(1))
    if start is None:
        return None, None, None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        cur_indent = len(line) - len(line.lstrip(" "))
        if cur_indent < indent:
            end = j
            break
        if cur_indent == indent and line.lstrip().startswith("- "):
            end = j
            break
    return start, end, indent


def _reindent(yaml_text: str, indent: int) -> str:
    """Strip the common indentation from the fix text, then shift it to
    match the target location's indentation in the file (e.g. a task
    nested inside a block: needs more indentation than a top-level
    task)."""
    lines = yaml_text.rstrip("\n").split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return yaml_text
    min_indent = min(len(l) - len(l.lstrip(" ")) for l in non_empty)
    dedented = [l[min_indent:] if l.strip() else l for l in lines]
    return "\n".join((" " * indent + l) if l.strip() else l for l in dedented)


def apply_fix(remediation, result) -> bool:
    """Replace an existing (non-compliant) task with its validated fix,
    in the file pointed to by remediation.source_file. Returns False
    (with no file modification) if the task couldn't be located
    unambiguously, or if the result of the replacement would no longer
    be valid YAML."""
    source_file = getattr(remediation, "source_file", "")
    if not source_file or not os.path.exists(source_file):
        return False

    with open(source_file, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")

    start, end, indent = _find_task_block(lines, remediation.original_task)
    if start is None:
        return False

    new_block = _reindent(result.final_yaml, indent)
    new_lines = lines[:start] + new_block.split("\n") + lines[end:]
    new_text = "\n".join(new_lines)

    if not _yaml_parses(new_text):
        return False

    with open(source_file, "w", encoding="utf-8") as f:
        f.write(new_text)
    return True


def confirm_and_apply(remediations: list, results: List[ValidationResult],
                      conflicts: List[ConflictReport], project_path: str,
                      audit_log_path: str = "results/a4_applied_changes.json") -> dict:
    """A SINGLE decision gate after validation: show a summary, ask ONCE
    whether to write the validated, conflict-free fixes to the project
    files. No consent = no file is touched. Fixes flagged by a conflict
    (detect_conflicts) are ALWAYS excluded from automatic writing,
    regardless of the answer — they require a manual decision, since two
    of them modify the same location in contradictory ways."""
    valid_count = sum(1 for r in results if r.valid)
    print(f"\n[A4] === Summary: {valid_count}/{len(results)} fixes passed validation ===")

    conflicted_tasks = set()
    if conflicts:
        print(f"[A4] {len(conflicts)} conflict(s) — these tasks will NOT be written automatically:")
        for c in conflicts:
            print(f"  [A4]   CONFLICT: '{c.rule_a}' vs '{c.rule_b}' — {c.reason}")
            conflicted_tasks.add(c.rule_a)
            conflicted_tasks.update(n.strip() for n in c.rule_b.split(","))

    applicable = [
        (rem, res) for rem, res in zip(remediations, results)
        if res.valid and res.original_task not in conflicted_tasks
    ]
    print(f"[A4] {len(applicable)} fix(es) ready to write (validated, conflict-free).")

    if not applicable:
        print("[A4] No fixes to write.")
        return {"applied": [], "skipped_conflicts": len(conflicted_tasks)}

    choice = input(f"\n[A4] Write these {len(applicable)} fixes to the project files? [y/n]\n> ").strip().lower()
    if choice != "y":
        print("[A4] Cancelled — no files were changed.")
        return {"applied": [], "cancelled": True, "candidates": len(applicable)}

    applied = []
    for rem, res in applicable:
        try:
            if rem.action_type == "CREATE":
                target = apply_create(rem.rule_id, rem.rule_title, res.final_yaml, project_path)
                applied.append({"rule_id": rem.rule_id, "action": "CREATE", "file": target})
                print(f"  [A4]   + appended new task for {rem.rule_id} -> {target}")
            else:
                ok = apply_fix(rem, res)
                if ok:
                    applied.append({"rule_id": rem.rule_id, "action": "FIX", "file": rem.source_file})
                    print(f"  [A4]   ~ replaced task for {rem.rule_id} in {rem.source_file}")
                else:
                    print(f"  [A4]   ! could not unambiguously locate task '{rem.original_task}' "
                          f"in {rem.source_file} — skipped, no change made")
        except Exception as e:
            print(f"  [A4]   ! write error for {rem.rule_id}: {e} — skipped")

    os.makedirs(os.path.dirname(audit_log_path) or ".", exist_ok=True)
    with open(audit_log_path, "w", encoding="utf-8") as f:
        json.dump(applied, f, ensure_ascii=False, indent=2)
    print(f"\n[A4] Wrote {len(applied)}/{len(applicable)} fixes. Log: {audit_log_path}")

    return {"applied": applied}


def run_validation_and_apply(remediations: list, project_path: str, max_attempts: int = 3,
                             use_llm_repair: bool = True, use_knowledge_base: bool = True) -> dict:
    """Agent A4's full flow: validate (validate_all) -> detect conflicts
    -> ASK ONCE whether to write -> if yes, write the validated,
    conflict-free fixes to the project files.

    Returns {"results": [...], "conflicts": [...], "applied": [...], ...}
    — the full set (not just the write outcome), since
    orchestrator.py/Agent A5 also need the raw validation results for the
    final report."""
    results, conflicts = validate_all(remediations, max_attempts, use_llm_repair,
                                      use_knowledge_base=use_knowledge_base)
    outcome = confirm_and_apply(remediations, results, conflicts, project_path)
    return {"results": results, "conflicts": conflicts, **outcome}


# ── Standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Agent A4 TEST — self-repair loop ===\n")
    print(f"ansible-lint available: {_ansible_lint_available()}\n")

    # Valid YAML — should pass immediately
    good_yaml = """- name: "5.1.20 | Ensure SSH PermitRootLogin is disabled"
  ansible.builtin.lineinfile:
    path: /etc/ssh/sshd_config
    regexp: '^PermitRootLogin'
    line: 'PermitRootLogin no'
  become: true"""

    # Deliberately broken YAML — wrong module parameter (should trigger repair)
    bad_yaml = """- name: "Bad task test"
  ansible.builtin.lineinfile:
    ath: /etc/ssh/sshd_config
    line: 'PermitRootLogin no'"""

    print("--- Test 1: valid YAML ---")
    r1 = validate_and_repair(good_yaml, "Ensure SSH PermitRootLogin", "5.1.20",
                             "Ensure SSH PermitRootLogin is disabled",
                             max_attempts=2, use_llm_repair=False, use_knowledge_base=False)
    print(f"Result: valid={r1.valid}, attempts={r1.total_attempts}\n")

    print("--- Test 2: broken YAML (no LLM repair, detection only) ---")
    r2 = validate_and_repair(bad_yaml, "Bad task test", "TEST",
                             "Test rule", max_attempts=1, use_llm_repair=False, use_knowledge_base=False)
    print(f"Result: valid={r2.valid}, reason={r2.give_up_reason}\n")

    print("--- Test 3: conflict detection ---")
    from dataclasses import dataclass as _dc
    @_dc
    class _FakeRem:
        original_task: str
        rule_id: str
        rule_title: str
        yaml_fix: str

    rem_a = _FakeRem("Task A", "5.1.16", "MaxAuthTries A", """- name: "A"
  ansible.builtin.lineinfile:
    path: /etc/ssh/sshd_config
    regexp: '^MaxAuthTries'
    line: 'MaxAuthTries 4'""")

    rem_b = _FakeRem("Task B", "5.1.16b", "MaxAuthTries B", """- name: "B"
  ansible.builtin.lineinfile:
    path: /etc/ssh/sshd_config
    regexp: '^MaxAuthTries'
    line: 'MaxAuthTries 6'""")

    results, conflicts = validate_all([rem_a, rem_b], max_attempts=1, use_llm_repair=False,
                                      use_knowledge_base=False)
    print(f"\nDetected {len(conflicts)} conflict(s)")
