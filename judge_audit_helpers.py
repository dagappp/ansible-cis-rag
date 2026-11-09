"""judge_audit_helpers.py — small, independently testable text-extraction
helper for judge_audit.py. Kept separate so it can be unit-tested without
pulling in the LLM-calling machinery."""

import re

_RAW_YAML_RE = re.compile(
    r"guessed/partial one\):\n(.*?)\n{2,}RELEVANT CIS BENCHMARK RULES:",
    re.DOTALL,
)

_SEMANTIC_RE = re.compile(
    r"Semantic description: (.*?)\n\nOriginal task RAW YAML",
    re.DOTALL,
)


def extract_raw_yaml(user_message_content: str) -> str:
    """Pulls the task's own raw_yaml block back out of the full Agent 2
    audit prompt (the anchor text and the 'RELEVANT CIS BENCHMARK RULES:'
    marker are copied from agent_a2_llm.llm_verify()'s template). The
    captured block also includes the resolved-parameter and enforcement-
    content notes, which the judge should see as well. Returns "" if the
    anchors are not found."""
    m = _RAW_YAML_RE.search(user_message_content)
    if not m:
        return ""
    return m.group(1).strip()


def extract_semantic_description(user_message_content: str) -> str:
    """Pulls the task's 'Semantic description: ...' line back out of the
    full Agent 2 audit prompt. This line contains Jinja-resolved values
    (e.g. a template's rendered content) that raw_yaml does not show, so
    the judge receives the same information Agent 2 had."""
    m = _SEMANTIC_RE.search(user_message_content)
    if not m:
        return ""
    return m.group(1).strip()
