"""
debug_dump.py - optional, full dump of every LLM call (prompt + raw
response + parsed result) to a JSONL file, for step-by-step manual
verification. Enabled ONLY by the DEBUG_DUMP_PATH environment variable -
when it is not set, dump_llm_call() does nothing (zero overhead in normal
production/comparison runs).
"""
import os
import json
import time

_PATH = os.environ.get("DEBUG_DUMP_PATH")


def extract_json_object(text: str) -> str:
    """Extract the first complete top-level {...} JSON object from an LLM
    response by brace-depth counting (respecting string literals and escaped
    quotes). A simple 'first { to last }' slice fails when the model adds
    text after the JSON object. Returns "" if no complete object is found.
    """
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""  # unbalanced — no complete object (e.g. response got cut off)


def dump_llm_call(stage: str, identifier: str, messages, raw_response: str, parsed=None, extra: dict = None):
    if not _PATH:
        return
    entry = {
        "ts": time.time(),
        "stage": stage,           # e.g. "A2_audit", "A3_generate", "A4_validate", "A4_semantic", "judge_holistic", "judge_per_fix"
        "identifier": identifier, # task name / rule_id, for easy lookup
        "messages": messages,     # full prompt (system+user) sent to the model
        "raw_response": raw_response,
        "parsed": parsed,
    }
    if extra:
        entry.update(extra)
    try:
        with open(_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [debug_dump] Warning: could not write dump ({e})")
