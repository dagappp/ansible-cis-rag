"""
Agent A1 — Onboarding
======================
Runs first, before the audit. Ensures that every CIS rule of the benchmark
is either defined and enabled in defaults/, or that its absence or
disablement is a conscious, documented human decision.

For every rule that is UNDEFINED (no rule_X_Y_Z variable) or DISABLED
(variable set to false), A1 asks the operator in the terminal to choose:
  1. FIX NOW — add or enable the rule in defaults/main.yml (ruamel.yaml
     round-trip, comments preserved),
  2. ACCEPT THE RISK — shows the rule's Rationale from the CIS Benchmark and
     requires a justification and a typed confirmation. The rule stays
     disabled and its rule_id is passed to Agent A2 as "disabled_rules",
     so it is not audited.

Decisions are logged in results/onboarding_decisions.json. The log is also
A1's memory between runs: an accepted risk is not asked about again unless
the rule's status or benchmark text changes (sha256 fingerprint).
"""

import os
import re
import json
import sys
import time
import hashlib
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from parser_a0_final import (
    discover_defaults_files,
    discover_roles,
    resolve_rule_variables,
)
import agent_a2_llm  # dynamic access (agent_a2_llm.API_MODEL), not "from X import Y" —
# see rationale in agent_a3_llm.py
from debug_dump import dump_llm_call, extract_json_object

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    _ryaml = YAML(typ="rt")
    _ryaml.preserve_quotes = True
    _ryaml.width = 4096
    _ryaml.explicit_start = True
    # ruamel's default indentation scheme (mapping=2, sequence=2, offset=0)
    # doesn't match the typical defaults/main.yml style in CIS hardening
    # projects (nested maps indented by 4 spaces, e.g. sshd:/pwquality:) —
    # without this, every write (even of a single variable) would reformat
    # the entire file on re-dump (hundreds of changed lines instead of one).
    _ryaml.indent(mapping=4, sequence=4, offset=2)
except ImportError:
    _ryaml = None
    print("[A1] Warning: ruamel.yaml unavailable — writing to defaults/ is disabled "
          "(install: pip install ruamel.yaml).")


ACCEPT_WORD = "ACCEPT"  # word the user must type to consciously accept the risk


@dataclass
class OnboardingDecision:
    rule_id:      str
    title:        str
    cis_status:   str   # MISSING_VARIABLE / DISABLED_BY_VAR
    action:       str   # ADDED / ENABLED / RISK_ACCEPTED — always terminal, no PENDING
    reason:       str = ""
    decided_at:   str = ""
    content_hash: str = ""   # fingerprint of rule_id+cis_status+rationale at decision time — see _gap_fingerprint


# ═══════════════════════════════════════════════════════════════════════════
# Extracting the full list of rules + rationale from the CIS Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def _rule_id_key(rule_id: str) -> tuple:
    """Numeric sort key for a CIS rule ID (e.g. '1.1.1.10' should sort AFTER
    '1.1.1.2', not before it as with plain string sorting)."""
    return tuple(int(p) for p in rule_id.split("."))


def _clean_title(title: str) -> str:
    """Some entries in analysis_results.json are section headings pulled
    from the PDF table of contents and end with a long run of filler dots
    (e.g. '...Configure auditd Service .....................'). Strip this
    before showing it to the user and before writing it as a YAML comment."""
    return re.sub(r"[\s.]{4,}$", "", title).strip()


def _load_cis_rules(analysis_json_path: str) -> dict:
    """Load rules from analysis_results.json and filter out section headings.

    A real CIS rule always has both a Description and a Remediation section.
    Grouping headings (e.g. '6.3.3 Configure auditd Rules') and appendix
    entries lack their own Description, so requiring both sections removes
    them without losing any real rule."""
    with open(analysis_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules = {
        r["rule_id"]: r for r in data["rules"]
        if "Description" in r.get("subsections", []) and "Remediation" in r.get("subsections", [])
    }
    skipped = len(data["rules"]) - len(rules)
    if skipped:
        print(f"[A1] Skipped {skipped} entries without a full Description+Remediation set "
              f"(section headings / appendix noise, not real CIS rules).")
    return rules


_RATIONALE_RE = re.compile(
    r"Rationale:\s*(.*?)\s*(?:Audit:|Impact:|Remediation:|Default Value:|References:)",
    re.S,
)


def _extract_rationale(full_text: str, max_len: int = 600) -> str:
    """Extract the 'Rationale' section from a CIS rule's text — this is the
    official description of consequences/risk from the benchmark, shown to
    the user directly, without any LLM mediation."""
    m = _RATIONALE_RE.search(full_text or "")
    text = m.group(1).strip() if m else ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "…"
    return text or "(no Rationale section found in the benchmark text)"


# ═══════════════════════════════════════════════════════════════════════════
# Computing gaps: undefined / disabled rules
# ═══════════════════════════════════════════════════════════════════════════

def compute_gaps(project_path: str, analysis_json_path: str = "analysis_results.json") -> list:
    """
    Compare the FULL list of CIS rules from the benchmark against the
    variable status in defaults/ (Parser A0: resolve_rule_variables).
    Returns a list of dicts:
      {rule_id, title, rationale, cis_status, var_name, source_file}
    only for rules that are UNDEFINED (MISSING_VARIABLE) or
    DISABLED (DISABLED_BY_VAR).
    """
    cis_rules = _load_cis_rules(analysis_json_path)
    rule_status = resolve_rule_variables(project_path)

    gaps = []
    for rule_id, rule in sorted(cis_rules.items(), key=lambda kv: _rule_id_key(kv[0])):
        status_obj = rule_status.get(rule_id)
        rationale = _extract_rationale(rule.get("full_text", ""))

        title = _clean_title(rule.get("title", rule_id))

        if status_obj is None:
            gaps.append({
                "rule_id": rule_id,
                "title": title,
                "rationale": rationale,
                "cis_status": "MISSING_VARIABLE",
                "var_name": f"rule_{rule_id.replace('.', '_')}",
                "source_file": None,
            })
        elif status_obj.enabled is False:
            gaps.append({
                "rule_id": rule_id,
                "title": title,
                "rationale": rationale,
                "cis_status": "DISABLED_BY_VAR",
                "var_name": status_obj.var_name,
                "source_file": status_obj.source_file,
            })

    return gaps


def _gap_fingerprint(rule_id: str, cis_status: str, rationale: str) -> str:
    """Fingerprint of the gap's content at decision time — lets us detect
    whether anything changed since the previous run (a different benchmark
    version, a different rule status) before treating an old RISK_ACCEPTED
    decision as still valid."""
    raw = f"{rule_id}|{cis_status}|{rationale}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _load_prior_decisions(decisions_output: str) -> dict:
    """Load the decision log from the previous run (if it exists) — this is
    Agent A1's 'memory' between runs. Returns {rule_id: OnboardingDecision};
    if there are duplicates in the file (shouldn't happen, since
    run_onboarding always writes an already-merged, unambiguous set) the
    last entry wins."""
    if not os.path.exists(decisions_output):
        return {}
    try:
        with open(decisions_output, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[A1] Warning: failed to load {decisions_output} ({e}) — "
              f"treating it as no history.")
        return {}
    prior = {}
    for d in raw:
        try:
            prior[d["rule_id"]] = OnboardingDecision(**d)
        except TypeError:
            continue
    return prior


# ═══════════════════════════════════════════════════════════════════════════
# Writing changes to defaults/main.yml (ruamel round-trip — preserves comments)
# ═══════════════════════════════════════════════════════════════════════════

def _pick_target_defaults_file(project_path: str) -> str:
    """Pick the defaults/main.yml file to append the new rule to.
    Uses the same role priority order as Parser A0 (discover_roles:
    hardening/cis/security before audit/test/check) — a new toggle
    variable makes sense in the role that actually contains the task
    implementing the rule, not in an audit role."""
    files = discover_defaults_files(project_path)
    if not files:
        raise FileNotFoundError(f"No defaults/main.yml found in {project_path}")

    for _role_name, role_root in discover_roles(project_path):
        for candidate in (os.path.join(role_root, "defaults", "main.yml"),
                          os.path.join(role_root, "defaults", "main.yaml")):
            if candidate in files:
                return candidate

    return files[0]


def apply_add_rule(defaults_path: str, var_name: str, title: str) -> None:
    """Append a new rule_X_Y_Z: true # <title> variable at the end of the
    defaults file, preserving existing comments/formatting."""
    if _ryaml is None:
        raise RuntimeError("ruamel.yaml unavailable — install: pip install ruamel.yaml")
    with open(defaults_path, encoding="utf-8") as f:
        data = _ryaml.load(f)
    if data is None:
        data = CommentedMap()
    data[var_name] = True
    data.yaml_add_eol_comment(title, var_name)
    with open(defaults_path, "w", encoding="utf-8") as f:
        _ryaml.dump(data, f)


def apply_enable_rule(source_file: str, var_name: str) -> None:
    """Set an existing rule_X variable to true in the file that defines it,
    preserving its comment."""
    if _ryaml is None:
        raise RuntimeError("ruamel.yaml unavailable — install: pip install ruamel.yaml")
    with open(source_file, encoding="utf-8") as f:
        data = _ryaml.load(f)
    data[var_name] = True
    with open(source_file, "w", encoding="utf-8") as f:
        _ryaml.dump(data, f)


# ═══════════════════════════════════════════════════════════════════════════
# Interactive onboarding
# ═══════════════════════════════════════════════════════════════════════════

# ── Reason quality assessment (LLM with reasoning) ─────────────────────────
# Checks whether a risk-acceptance justification is substantive rather than
# dismissive ("no time", "n/a"). Uses the same model as the rest of the
# pipeline, with reasoning enabled. It only warns — the final decision
# always belongs to the human.


def _check_reason_quality(reason: str, rule_id: str, title: str, rationale: str):
    """Returns (is_evasive: bool, feedback: str). On any error (network,
    parsing) returns (False, "") — a failure of this OPTIONAL check never
    blocks onboarding."""
    messages = [
        {"role": "system", "content": (
            "You are assessing whether a user's justification for consciously "
            "accepting a security risk (CIS Benchmark) is substantive, or "
            "evasive/empty in content (e.g. 'just because', 'no time', 'I don't "
            "know', a generic statement unrelated to the rule). Respond ONLY "
            "with valid JSON, no markdown."
        )},
        {"role": "user", "content": f"""CIS Rule {rule_id}: {title}
Consequences according to the CIS Benchmark (Rationale): {rationale}

User's justification for leaving this rule unimplemented:
"{reason}"

Is this justification evasive/without substance (does not genuinely address the
risk described in the Rationale)? Respond with JSON:
{{"evasive": true/false, "feedback": "a short, concrete note in English (1 sentence) — what specifically is missing, or an empty string if the justification is OK"}}"""},
    ]
    payload = json.dumps({
        "model": agent_a2_llm.API_MODEL,
        "messages": messages,
        # 500 wasn't enough for qwen with reasoning enabled by default —
        # the model spent the whole budget on internal reasoning and
        # returned EMPTY content (confirmed empirically: an actual test
        # used 883 tokens for the same question that fit in 500 with
        # Gemma).
        "max_tokens": 1500,
        "temperature": 0,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            agent_a2_llm.API_URL, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {agent_a2_llm.API_TOKEN}",
                     "User-Agent": "curl/7.88.1"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read())
        # Success is recorded once, after a usable result is extracted, so
        # that a malformed response is not counted twice.
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
        # extract_json_object() takes the first complete JSON object (see
        # debug_dump.py).
        json_str = extract_json_object(text)
        if not json_str:
            agent_a2_llm._record_request(time.time() - t0, success=True)
            return False, ""
        result = json.loads(json_str)
        agent_a2_llm._record_request(time.time() - t0, success=True)
        dump_llm_call("A1_reason_quality", rule_id, messages, text, result)
        return bool(result.get("evasive")), result.get("feedback", "")
    except Exception:
        agent_a2_llm._record_request(time.time() - t0, success=False)
        return False, ""


def _ask(prompt: str) -> str:
    return input(prompt).strip()


def _prompt_gap(gap: dict, project_path: str) -> OnboardingDecision:
    fingerprint = _gap_fingerprint(gap["rule_id"], gap["cis_status"], gap["rationale"])
    status_label = "UNDEFINED" if gap["cis_status"] == "MISSING_VARIABLE" else "DISABLED (false)"
    fix_label = ("add the rule (set it to true)" if gap["cis_status"] == "MISSING_VARIABLE"
                 else "enable the rule (set it to true)")

    print(f"\n{'─' * 70}")
    print(f"CIS {gap['rule_id']} — {gap['title']}")
    print(f"Status in project : {status_label}  (variable: {gap['var_name']})")
    print(f"Consequences (Rationale from CIS Benchmark):")
    print(f"  {gap['rationale']}")
    print(f"{'─' * 70}")

    while True:
        # Deliberately only 2 options — "skip for now" doesn't exist.
        # defaults/ after onboarding is meant to be the starting point for
        # A0/A2/A3/A4, so every rule must leave this loop in a RESOLVED
        # state.
        choice = _ask(f"[1] {fix_label}   [2] leave it and consciously accept the risk\n> ")

        if choice == "1":
            try:
                if gap["cis_status"] == "MISSING_VARIABLE":
                    target = _pick_target_defaults_file(project_path)
                    apply_add_rule(target, gap["var_name"], gap["title"])
                    action = "ADDED"
                else:
                    apply_enable_rule(gap["source_file"], gap["var_name"])
                    action = "ENABLED"
            except Exception as e:
                print(f"  [A1] Error writing to defaults/: {e} — rule remains unchanged.")
                continue
            reason = _ask("Short comment for the decision log (optional): ")
            return OnboardingDecision(gap["rule_id"], gap["title"], gap["cis_status"], action,
                                       reason, datetime.now(timezone.utc).isoformat(), fingerprint)

        if choice == "2":
            while True:
                reason = ""
                while not reason:
                    reason = _ask("Why is this deliberate? Justification (required): ").strip()

                is_evasive, feedback = _check_reason_quality(
                    reason, gap["rule_id"], gap["title"], gap["rationale"])
                if is_evasive:
                    print(f"\n  [A1] Note (LLM assessment): {feedback}")
                    redo = _ask("  [1] refine the justification   [2] continue anyway\n  > ")
                    if redo == "1":
                        continue  # only re-asks for the justification, not the whole 1/2 choice
                break

            confirm = _ask(f"Type exactly '{ACCEPT_WORD}' to consciously confirm accepting the risk: ")
            if confirm.strip() == ACCEPT_WORD:
                return OnboardingDecision(gap["rule_id"], gap["title"], gap["cis_status"], "RISK_ACCEPTED",
                                           reason, datetime.now(timezone.utc).isoformat(), fingerprint)
            print("  [A1] Confirmation did not match the keyword — decision was not "
                  "saved, returning to the question.")
            continue

        print("  Unknown option, choose 1 or 2.")


def run_onboarding(project_path: str, analysis_json_path: str = "analysis_results.json",
                    decisions_output: str = None) -> dict:
    """
    Main entry point of Agent A1.

    decisions_output defaults to <project_path>/results/onboarding_decisions.json
    (relative to the project, not to the working directory, so that the
    decision log is found regardless of where the pipeline is started).

    Every rule leaves onboarding in a terminal state: ADDED/ENABLED or
    RISK_ACCEPTED. The rule_ids with RISK_ACCEPTED status are returned as
    "disabled_rules" and should be passed to Agent A2:

        result = run_onboarding(project_path)
        tasks = parse_ansible_project(project_path)          # Parser A0
        audit = audit_tasks(tasks, disabled_rules=result["disabled_rules"])  # Agent A2

    Previously accepted risks with an unchanged fingerprint are carried
    over without asking again.

    Returns {"decisions": list of OnboardingDecision (the full, merged set),
    "disabled_rules": list of rule_ids with RISK_ACCEPTED status that are
    still undefined/disabled in the project's current state}.
    """
    if decisions_output is None:
        decisions_output = os.path.join(project_path, "results", "onboarding_decisions.json")

    gaps = compute_gaps(project_path, analysis_json_path)
    prior = _load_prior_decisions(decisions_output)

    print(f"\n[A1] === Onboarding — {len(gaps)} CIS rules undefined or disabled ===")
    if prior:
        print(f"[A1] Loaded {len(prior)} decisions from the previous run ({decisions_output}).")

    if not gaps:
        print("[A1] All rules from the CIS Benchmark are defined and enabled. Onboarding complete.")
        return {"decisions": list(prior.values()), "disabled_rules": []}

    merged = dict(prior)  # start from the full history — new/changed decisions will overwrite it
    reused, asked = 0, 0

    for gap in gaps:
        fingerprint = _gap_fingerprint(gap["rule_id"], gap["cis_status"], gap["rationale"])
        prev = prior.get(gap["rule_id"])

        if prev is not None and prev.action == "RISK_ACCEPTED" and prev.content_hash == fingerprint:
            print(f"\n[A1] CIS {gap['rule_id']} — risk already accepted on {prev.decided_at[:10]} "
                  f"(\"{prev.reason[:60]}\"), skipping the question.")
            reused += 1
            continue

        merged[gap["rule_id"]] = _prompt_gap(gap, project_path)
        asked += 1

    os.makedirs(os.path.dirname(decisions_output) or ".", exist_ok=True)
    with open(decisions_output, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in merged.values()], f, ensure_ascii=False, indent=2)

    # disabled_rules: only RISK_ACCEPTED entries that are STILL in the
    # current gaps — if someone fixed the rule by hand outside of
    # onboarding, the old decision stays in the log as history, but no
    # longer goes onto this list.
    current_gap_ids = {g["rule_id"] for g in gaps}
    disabled_rules = [rid for rid, d in merged.items()
                       if d.action == "RISK_ACCEPTED" and rid in current_gap_ids]

    added   = sum(1 for d in merged.values() if d.action == "ADDED")
    enabled = sum(1 for d in merged.values() if d.action == "ENABLED")
    print(f"\n[A1] Onboarding complete: {asked} rules resolved just now, "
          f"{reused} restored from memory (without asking again).")
    print(f"[A1] Overall state: {added} added, {enabled} enabled, "
          f"{len(disabled_rules)} with accepted risk.")
    print(f"[A1] Decisions saved to: {decisions_output}")
    print(f"[A1] {len(disabled_rules)} rules passed as disabled_rules to Agent A2 (audit_tasks) — "
          f"the LLM will not audit them or propose a fix for them.")

    return {"decisions": list(merged.values()), "disabled_rules": disabled_rules}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    _project_path = sys.argv[1]
    _analysis_json = next((a.split("=")[1] for a in sys.argv if a.startswith("--analysis=")),
                          "analysis_results.json")

    run_onboarding(_project_path, _analysis_json)
