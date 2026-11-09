"""
Parser A0 — Ansible project parser
==================================
The only part of the system that operates directly on the raw playbook
files. It is not an agent (it does not call a language model): it is a
deterministic static parser, a pre-processing layer. The agents only see
its structured output.

Task: translate a multi-file Ansible project into a unified, semantically
rich representation from which Agent A2 builds queries against the RAG
knowledge base. Independent of the target operating system.

Consists of four modules run sequentially:

  MODULE 1: FileDiscovery      — auto-discovery of project files
  MODULE 2: YAMLParser         — parsing while preserving comments (ruamel.yaml)
  MODULE 3: VariableResolver   — detection of CIS rule variables and their status
  MODULE 4: SemanticMapper     — mapping tasks to CIS rules (tags/when/name/comment)
  + CoverageAnalyzer          — comparison of the full CIS list vs. implemented tasks
"""

import re
import os
import json
from dataclasses import dataclass, field
from typing import Optional

# ── MODULE 2: YAMLParser — prefer ruamel.yaml (preserves comments) ────────
try:
    from ruamel.yaml import YAML
    _ryaml = YAML(typ="rt")           # round-trip: preserves comments and line numbers
    _ryaml.preserve_quotes = True
    # Duplicate keys in one mapping (e.g. two "when:" entries) are allowed,
    # last value wins as in plain YAML parsers. Otherwise ruamel would refuse
    # to load the whole file and all of its tasks would be lost.
    _ryaml.allow_duplicate_keys = True
    HAS_RUAMEL = True
except ImportError:
    import yaml as _pyyaml
    HAS_RUAMEL = False
    print("[P0] Warning: ruamel.yaml not available — comments with CIS IDs will not be "
          "used. Install with: pip install ruamel.yaml")


def _load_yaml(path: str):
    """
    Load a YAML file. With ruamel.yaml, line numbers are preserved (.lc.line)
    which allows correlating # CIS X.X.X comments with tasks. Without ruamel
    it works identically in function, but without comment support.
    """
    with open(path, "r", encoding="utf-8") as f:
        if HAS_RUAMEL:
            return _ryaml.load(f)
        return _pyyaml.safe_load(f)


def _dump_yaml(data) -> str:
    """Dump data back to YAML (for raw_yaml in TaskDescription)."""
    if HAS_RUAMEL:
        import io
        buf = io.StringIO()
        _ryaml.dump(data, buf)
        return buf.getvalue()
    else:
        return _pyyaml.dump(data, default_flow_style=False, allow_unicode=True)


# ── Output structures ────────────────────────────────────────────────────────
@dataclass
class TaskDescription:
    """Output of the SemanticMapper module — one task ready for Agent A2."""
    task_name:    str
    module:       str
    params:       dict
    semantic:     str
    raw_yaml:     str = ""
    cis_rule_ids: list = field(default_factory=list)
    cis_id_source: str = ""     # where the ID was taken from: tag / when / name / comment
    cis_level:    str = ""
    source_file:  str = ""
    when_rules:   list = field(default_factory=list)
    resolved_loop: list = field(default_factory=list)
                                 # Jinja-resolved items of this task's inline 'loop:'/
                                 # 'with_items:' list, so the model sees the actual
                                 # per-iteration values referenced as '{{ item.x }}'.
    template_content: str = ""  # Resolved content of the referenced .j2 file, when this
                                 # task's module is 'template'/'ansible.builtin.template' and
                                 # its 'src:' could be found under the role's templates/ dir.
                                 # Some roles enforce many rules through one shared template,
                                 # so the rule's real content is only visible there.


@dataclass
class RuleStatus:
    """Output of the VariableResolver module — status of one CIS rule in defaults."""
    rule_id:   str
    enabled:   bool
    var_name:  str
    comment:   str = ""
    source_file: str = ""


@dataclass
class CoverageReport:
    """Output of CoverageAnalyzer — whether a CIS rule is implemented."""
    rule_id: str
    title:   str
    status:  str   # IMPLEMENTED / DISABLED_BY_VAR / MANUAL_CONTROL / MISSING_TASK / MISSING_VARIABLE
    has_variable: bool
    variable_enabled: Optional[bool] = None
    has_task: bool = False
    task_names: list = field(default_factory=list)
    is_manual: bool = False   # CIS Benchmark marks this rule "(Manual)" — see is_manual_rule()
    cis_level: str = ""       # "1" or "2" — CIS Benchmark's OWN official "Profile
                               # Applicability" designation (Level 1 = baseline,
                               # broadly applicable; Level 2 = defense-in-depth,
                               # more environment-specific) — see extract_cis_profile_level()


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1: FileDiscovery
# ═══════════════════════════════════════════════════════════════════════════
# Does not assume a rigid directory structure beyond the standard Ansible
# convention (roles/*/tasks/, roles/*/defaults/) — which covers nearly all real
# hardening projects. Automatically discovers roles, defaults files, tasks files.

def discover_roles(project_path: str) -> list:
    """
    Find all Ansible roles in the project (directories containing tasks/).
    Sorts so that hardening roles are processed before audit roles.
    """
    roles = []

    def _check(rpath, rname):
        if os.path.isdir(os.path.join(rpath, "tasks")):
            roles.append((rname, rpath))

    roles_dir = os.path.join(project_path, "roles")
    if os.path.isdir(roles_dir):
        for name in os.listdir(roles_dir):
            _check(os.path.join(roles_dir, name), name)

    if not roles:
        _check(project_path, os.path.basename(project_path))

    PRIORITY    = ["hardening", "cis", "security", "rhel", "ubuntu", "windows"]
    DEPRIORITY  = ["audit", "test", "check"]

    def sort_key(item):
        name = item[0].lower()
        if any(p in name for p in PRIORITY):
            return 0
        if any(p in name for p in DEPRIORITY):
            return 2
        return 1

    return sorted(roles, key=sort_key)


def discover_defaults_files(project_path: str) -> list:
    """Find all defaults/main.yml files across the entire project (multiple roles)."""
    found = []
    for root, dirs, files in os.walk(project_path):
        if os.path.basename(root) == "defaults":
            for fname in files:
                if fname in ("main.yml", "main.yaml"):
                    found.append(os.path.join(root, fname))
    return found


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2: YAMLParser (parsing + CIS comment extraction)
# ═══════════════════════════════════════════════════════════════════════════

# Patterns for comments containing a CIS rule ID, e.g.:
#   # CIS 1.1.1.1
#   # 1.1.1.1 Ensure ...
#   #1.1.1.1
COMMENT_CIS_PATTERNS = [
    re.compile(r"#.*?CIS[\s_-]*(\d+(?:\.\d+){1,4})", re.I),
    re.compile(r"#\s*(\d+(?:\.\d+){2,4})\b"),
]


def _extract_cis_from_comment_window(raw_lines: list, start_line: int, window: int = 4) -> str:
    """
    Search a few lines around the start of a task for a comment
    containing a CIS rule ID. Only works with ruamel.yaml (needs line numbers).
    """
    if start_line is None:
        return ""
    lo = max(0, start_line - 1)
    hi = min(len(raw_lines), start_line + window)
    for line in raw_lines[lo:hi]:
        for pat in COMMENT_CIS_PATTERNS:
            m = pat.search(line)
            if m:
                return m.group(1)
    return ""


def _get_line(item) -> Optional[int]:
    """Extract the start line number of an element (only available with ruamel.yaml)."""
    if HAS_RUAMEL and hasattr(item, "lc") and item.lc.line is not None:
        return item.lc.line
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3: VariableResolver
# ═══════════════════════════════════════════════════════════════════════════
# Universal detection of variables controlling CIS rules in defaults/.
# Does not assume a single naming convention nor a single project structure.
# Recognizes 5 name patterns and many representations of boolean values.

RULE_VAR_PATTERNS = [
    # The "(\w+cis_)?" prefix handles the ansible-lockdown family, where each
    # variable has a distribution prefix, e.g. rhel9cis_rule_1_1_1_1,
    # rhel8cis_rule_..., ubtu22cis_rule_... — the rule itself (group 1) is
    # identical to that in simpler projects (rule_1_1_1_1).
    re.compile(r"^(?:\w+cis_)?rule_(\d+(?:_\d+)+)$", re.I),
    re.compile(r"^(?:\w+cis_)?cis_rule_(\d+(?:_\d+)+)$", re.I),
    re.compile(r"^cis_(\d+(?:_\d+)+)$"),
    re.compile(r"^enable_(\d+(?:_\d+)+)$"),
    re.compile(r"^section_(\d+(?:_\d+)+)$"),
]

TRUTHY_VALUES = {True, "true", "True", "yes", "Yes", 1, "1", "enabled"}
FALSY_VALUES  = {False, "false", "False", "no", "No", 0, "0", "disabled"}


def _parse_bool(value) -> bool:
    """Normalize various boolean representations. Unknown values -> True (safe default)."""
    if value in TRUTHY_VALUES:
        return True
    if value in FALSY_VALUES:
        return False
    return True


def resolve_rule_variables(project_path: str) -> dict:
    """
    Main function of VariableResolver.
    Search ALL defaults/main.yml files in the project and build
    a {rule_id: RuleStatus} map. Universal — does not assume a number of rules
    or a specific naming convention.
    """
    status_map = {}
    defaults_files = discover_defaults_files(project_path)

    for path in defaults_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            data = _load_yaml(path)
        except Exception as e:
            print(f"  [VariableResolver] Error loading {path}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        # Variable -> comment map from the same line (from the raw text)
        comments = {}
        for line in raw_text.split("\n"):
            m = re.match(r"^\s*([\w]+)\s*:\s*.*?#\s*(.+)$", line)
            if m:
                comments[m.group(1)] = m.group(2).strip()

        for var_name, value in data.items():
            if not isinstance(var_name, str):
                continue
            rule_id = None
            for pat in RULE_VAR_PATTERNS:
                m = pat.match(var_name)
                if m:
                    rule_id = m.group(1).replace("_", ".")
                    break
            if rule_id is None:
                continue

            status_map[rule_id] = RuleStatus(
                rule_id=rule_id,
                enabled=_parse_bool(value),
                var_name=var_name,
                comment=comments.get(var_name, ""),
                source_file=path,
            )

    return status_map


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 4: SemanticMapper
# ═══════════════════════════════════════════════════════════════════════════
# Maps each task to CIS rules using a multi-level strategy:
#   1. Block tags (most reliable, propagated from the parent)
#   2. when: condition rule_X_X_X
#   3. Numeric prefix in the task name
#   4. Comment near the task (# CIS X.X.X) — requires ruamel.yaml

SKIP_MODULES = {
    "debug","set_fact","assert","meta","fail","include_tasks","import_tasks",
    "include_role","import_role","include_vars","gather_facts","wait_for",
    "pause","add_host","group_by","setup",
}
RESERVED_KEYS = {
    "name","when","tags","register","loop","with_items","with_fileglob",
    "with_list","with_dict","with_nested","with_sequence","notify","become",
    "become_user","vars","ignore_errors","failed_when","changed_when","no_log",
    "block","rescue","always","environment","delegate_to","run_once",
    "any_errors_fatal","check_mode","listen","retries","until","delay",
    "timeout","async","poll","throttle","diff","debugger","module_defaults",
}
MODULE_CONTEXT = {
    "lineinfile":     "modification of a configuration parameter in a file",
    "blockinfile":    "insertion of a configuration block in a file",
    "file":           "permissions or ownership of a file/directory",
    "copy":           "copying a configuration file",
    "template":       "deployment of a configuration template",
    "service":        "management of a system service",
    "systemd":        "management of a systemd service",
    "modprobe":       "management of Linux kernel modules",
    "command":        "execution of a system command",
    "shell":          "execution of a bash shell command",
    "script":         "running a shell script",
    "sysctl":         "configuration of sysctl kernel parameters",
    "user":           "management of a system user account",
    "group":          "management of a system group",
    "cron":           "cron scheduled task",
    "mount":          "mounting a filesystem",
    "ansible.posix.mount": "mounting a filesystem (POSIX)",
    "firewalld":      "firewalld firewall configuration",
    "ansible.posix.firewalld": "firewalld firewall configuration",
    "iptables":       "iptables rule configuration",
    "dnf":            "dnf package installation/removal",
    "yum":            "yum package installation/removal",
    "apt":            "apt package installation/removal",
    "package":        "system package installation/removal",
    "rpm_key":        "GPG key for RPM package verification",
    "replace":        "pattern replacement in a configuration file",
    "ini_file":       "modification of an INI configuration file",
    "pamd":           "configuration of PAM authentication modules",
    "find":           "searching for files matching criteria",
    "stat":           "checking file attributes",
    "ufw":            "UFW firewall configuration",
    "selinux":        "SELinux mode configuration",
    "auditd":         "configuration of the system audit daemon",
    "pam_limits":     "PAM resource limits for users",
    "authorized_key": "user SSH authorization key",
    "logrotate":      "log rotation configuration",
    "nftables":       "nftables rule configuration",
}

CIS_ID_TAG_PATTERNS = [
    re.compile(r"^r(\d+(?:\.\d+){1,4})$"),
    # ansible-lockdown convention: tag "rule_1.1.1.1" (dots, no distribution
    # prefix — the tag itself is shared across the whole lockdown role family).
    re.compile(r"^rule_(\d+(?:\.\d+){1,4})$", re.I),
    re.compile(r"^CIS[-_](\d+(?:\.\d+){1,4})$", re.I),
    re.compile(r"^cis_(\d+(?:_\d+){1,4})$", re.I),
]
WHEN_RULE_PATTERNS = [
    # The "(\w+cis_)?" prefix as in RULE_VAR_PATTERNS — in the
    # ansible-lockdown family "when:" refers directly to the variable from defaults/
    # (e.g. when: rhel9cis_rule_1_1_1_1), not to the bare rule without a prefix.
    re.compile(r"^(?:\w+cis_)?rule_(\d+(?:_\d+){1,4})$", re.I),
    re.compile(r"^(?:\w+cis_)?cis_rule_(\d+(?:_\d+){1,4})$", re.I),
    re.compile(r"^r(\d+(?:_\d+){1,4})$"),
    # postgresql-15-main convention: "when: r_6_2" / "when: r_3_1_10"
    # (underscore directly after "r"). All such variables in that project
    # match real rule IDs of CIS PostgreSQL 15 Benchmark v1.0.0.
    re.compile(r"^r_(\d+(?:_\d+){0,4})$"),
]
NAME_CIS_PATTERNS = [
    re.compile(r"^\[?(\d+(?:\.\d+){1,4})\]?\s*[|\-–:]"),
    re.compile(r"^(?:Rule\s+)?(\d+(?:\.\d+){1,4})\s"),
]


def resolve_jinja(value, variables: dict, depth: int = 0):
    if depth > 3:
        return value
    if isinstance(value, str):
        s = value.strip()
        if s in ("{{ item }}", "{{item}}"):
            return "<loop_item>"
        m = re.fullmatch(r"\{\{\s*(\w+)\s*\}\}", s)
        if m and m.group(1) in variables:
            return variables[m.group(1)]

        # Dictionary access: {{ sshd['key'] }} or {{ sshd.key }}, with an
        # optional | default('...') filter (e.g. aide_cron['aide_minute']
        # | default('0')).
        m2 = re.fullmatch(
            r"\{\{\s*(\w+)\[[\'\"](\w+)[\'\"]\]\s*"
            r"(?:\|\s*default\([\'\"]?([^)\'\"]*)[\'\"]?\)\s*)?\}\}",
            s
        )
        if m2:
            var_name, key_name, default_val = m2.group(1), m2.group(2), m2.group(3)
            parent = variables.get(var_name, {})
            if isinstance(parent, dict) and key_name in parent:
                return parent[key_name]
            if default_val is not None:
                return default_val
            return value

        # Analogous case for dot notation: {{ sshd.key }}
        # with an optional | default('...')
        m3 = re.fullmatch(
            r"\{\{\s*(\w+)\.(\w+)\s*"
            r"(?:\|\s*default\([\'\"]?([^)\'\"]*)[\'\"]?\)\s*)?\}\}",
            s
        )
        if m3:
            var_name, key_name, default_val = m3.group(1), m3.group(2), m3.group(3)
            parent = variables.get(var_name, {})
            if isinstance(parent, dict) and key_name in parent:
                return parent[key_name]
            if default_val is not None:
                return default_val
            return value

        # Jinja expressions embedded in a longer string (e.g. "line:
        # 'ClientAliveInterval {{ sshd['clientaliveinterval'] }}'") are
        # resolved with the same dict-access logic as above, so that the
        # scalar value is substituted rather than the whole dictionary.
        def sub(match):
            expr = match.group(1)
            m_embedded = re.fullmatch(
                r"(\w+)(?:\[[\'\"](\w+)[\'\"]\]|\.(\w+))"
                r"(?:\s*\|\s*default\([\'\"]?([^)\'\"]*)[\'\"]?\))?",
                expr.strip()
            )
            if m_embedded:
                var_name = m_embedded.group(1)
                key_name = m_embedded.group(2) or m_embedded.group(3)
                default_val = m_embedded.group(4)
                parent = variables.get(var_name, {})
                if isinstance(parent, dict) and key_name in parent:
                    return str(parent[key_name])
                if default_val is not None:
                    return default_val
                return match.group(0)
            var = re.split(r"[\.\|\[\s]", expr)[0].strip()
            return str(variables.get(var, match.group(0)))
        return re.sub(r"\{\{\s*([\w\.\|\'\"\ \[\]]+)\s*\}\}", sub, value)
    elif isinstance(value, dict):
        return {k: resolve_jinja(v, variables, depth+1) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_jinja(v, variables, depth+1) for v in value]
    return value


def extract_cis_ids(tags=None, when=None, name: str = "",
                    comment_id: str = "") -> tuple:
    """Returns (id_list, source) — source tells where the ID was taken from (for quality auditing)."""
    ids = []
    for tag in (tags or []):
        tag_str = str(tag).strip()
        for pat in CIS_ID_TAG_PATTERNS:
            m = pat.match(tag_str)
            if m:
                raw = m.group(1).replace("_", ".")
                if raw not in ids:
                    ids.append(raw)
    if ids:
        return ids, "tag"

    for cond in (when or []):
        cond_str = str(cond).strip()
        for pat in WHEN_RULE_PATTERNS:
            m = pat.match(cond_str)
            if m:
                raw = m.group(1).replace("_", ".")
                if raw not in ids:
                    ids.append(raw)
    if ids:
        return ids, "when"

    for pat in NAME_CIS_PATTERNS:
        m = pat.match(name.strip())
        if m:
            return [m.group(1)], "name"

    if comment_id:
        return [comment_id], "comment"

    return [], ""


def extract_level(tags) -> str:
    for t in (tags or []):
        s = str(t).strip()
        if re.match(r"^level[12]_", s):
            return s
    return ""


# ── SemanticMapper extension: reading .sh script content ─────────────────
# Some tasks delegate their logic to an external shell script (script:/
# command:/shell: with X.sh). This reads the script and extracts the lines
# that say what is checked or set (variable assignments, file paths, sysctl
# patterns), so the model sees more than the file name.

SCRIPT_LINE_PATTERNS = [
    re.compile(r'^\s*\w+="[^"]{2,60}"'),        # l_option="fs.protected_hardlinks"
    re.compile(r"^\s*\w+='[^']{2,60}'"),        # l_option='fs.protected_hardlinks'
    re.compile(r'/etc/[\w./\-]+'),               # configuration paths
    re.compile(r'\b[a-z]+(?:\.[a-z_]+){1,4}\s*='),  # fs.protected_hardlinks =
]


def extract_script_summary(files_dir: str, script_name: str, max_lines: int = 6) -> str:
    """
    Read a shell script from the role's files/ directory and extract a condensed
    description of its logic — variable assignments carrying parameter names,
    configuration paths, sysctl patterns. Returns an empty string if the
    file does not exist or cannot be meaningfully summarized.
    """
    if not files_dir or not script_name or "{{" in script_name:
        return ""

    script_path = os.path.join(files_dir, script_name)
    if not os.path.exists(script_path):
        return ""

    try:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(4000)  # read limit for very long scripts
    except Exception:
        return ""

    interesting = []
    seen = set()
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pat in SCRIPT_LINE_PATTERNS:
            m = pat.search(stripped)
            if m:
                fragment = m.group(0).strip()
                if fragment not in seen:
                    seen.add(fragment)
                    interesting.append(fragment)
                break
        if len(interesting) >= max_lines:
            break

    if not interesting:
        return ""

    return "script checks: " + " ".join(interesting)


def extract_template_summary(templates_dir: str, template_name: str,
                             variables: dict = None, max_lines: int = 5) -> str:
    """
    Read a Jinja2 template (.j2) from the role's templates/ directory and
    extract a condensed description of its static content: the first N
    non-empty lines that are not Jinja2 control blocks or comments. Each
    line is passed through resolve_jinja, so templates consisting of a single
    variable (e.g. a login banner) show the actual content from defaults.
    """
    if not templates_dir or not template_name or "{{" in template_name:
        return ""

    template_path = os.path.join(templates_dir, template_name)
    if not os.path.exists(template_path):
        return ""

    try:
        with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(4000)
    except Exception:
        return ""

    JINJA_CONTROL = re.compile(r"^\s*\{%.*%\}\s*$")
    JINJA_COMMENT = re.compile(r"^\s*\{#.*#\}\s*$")

    interesting = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or JINJA_CONTROL.match(stripped) or JINJA_COMMENT.match(stripped):
            continue

        # Resolve Jinja2 variables in the line (e.g. the whole line = "{{ banner }}")
        resolved = stripped
        if variables and "{{" in stripped:
            resolved = resolve_jinja(stripped, variables)
            if not isinstance(resolved, str):
                resolved = str(resolved)
            # Multiline string from defaults (e.g. banner content) - flatten
            # to a single line and trim, so it doesn't dominate semantic
            resolved = " ".join(resolved.split())

        interesting.append(resolved[:80])
        if len(interesting) >= max_lines:
            break

    if not interesting:
        return ""

    return "template contains: " + " | ".join(interesting)


# Cache: templates_dir -> {rule_id: block_text}. Building the index requires
# scanning every .j2 file under a role's templates/ directory, which is
# wasteful to redo for every single task — the SAME templates_dir is shared
# by every task in the same role, so this is built once per role and reused.
_TEMPLATE_RULE_INDEX_CACHE: dict = {}

# Matches a CIS-rule-shaped token inside a Jinja condition, tolerant of an
# arbitrary alpha prefix before "rule_" (e.g. 'rhel9cis_rule_6_3_3_7',
# 'rule_6_3_3_7', bare 'cis_6_3_3_7') — at least 2 underscore-separated
# numeric segments, since a single segment ('rule_5') is usually a totally
# different kind of flag, not a CIS rule id.
_TEMPLATE_RULE_TOKEN_RE = re.compile(r"\b(?:\w*_)?rule_(\d+(?:_\d+){1,6})\b", re.IGNORECASE)


def _build_template_rule_index(templates_dir: str) -> dict:
    """Scans every .j2 file under a role's templates/ directory and indexes
    each '{% if <cond> %} ... {% endif %}' block by the CIS rule_id(s) its
    condition references (dots instead of underscores, e.g. '6.3.3.7').

    Some roles (e.g. ansible-lockdown) enforce many rules through one shared
    template, each rule gated by its own '{% if rule_X %}' block, while the
    per-rule tasks only set a fact. The index lets any task tagged with a
    rule ID find that rule's enforcement content. All matching blocks from
    all template files are collected (a rule can be split across files).
    Only single-level blocks are handled; a block that does not parse
    cleanly is skipped.
    """
    if templates_dir in _TEMPLATE_RULE_INDEX_CACHE:
        return _TEMPLATE_RULE_INDEX_CACHE[templates_dir]

    raw_index: dict = {}
    if templates_dir and os.path.isdir(templates_dir):
        for root, _dirs, files in os.walk(templates_dir):
            for fname in files:
                if not fname.endswith(".j2"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(200_000)  # bounded, but generous — these are text config templates
                except Exception:
                    continue

                # Simple single-level {% if COND %} ... {% endif %} scan —
                # tracks nesting depth just enough to find the MATCHING
                # endif for each if, without trying to parse arbitrary
                # elif/else/for nesting inside.
                if_re = re.compile(r"\{%-?\s*if\s+(.+?)\s*-?%\}")
                pos = 0
                for m in if_re.finditer(content):
                    cond = m.group(1)
                    tokens = _TEMPLATE_RULE_TOKEN_RE.findall(cond)
                    if not tokens:
                        continue
                    depth = 1
                    scan_pos = m.end()
                    end_pos = None
                    for m2 in re.finditer(r"\{%-?\s*(if|endif)\b.*?-?%\}", content[scan_pos:]):
                        if m2.group(1) == "if":
                            depth += 1
                        else:
                            depth -= 1
                            if depth == 0:
                                end_pos = scan_pos + m2.end()
                                break
                    if end_pos is None:
                        continue
                    # When an inline '{% if %}' block sits in the middle of a
                    # configuration line, the whole line is captured, so that
                    # the block keeps its context (e.g. 'cipher@SSH = ...').
                    line_start = content.rfind("\n", 0, m.start()) + 1
                    if content[line_start:m.start()].strip():
                        line_end = content.find("\n", end_pos)
                        if line_end < 0:
                            line_end = len(content)
                        block_text = content[line_start:line_end].strip()
                    else:
                        block_text = content[m.start():end_pos].strip()
                    for tok in tokens:
                        rule_id = tok.replace("_", ".")
                        blocks = raw_index.setdefault(rule_id, [])
                        # Every distinct matching block is collected; exact
                        # duplicates (from nested ifs) are skipped.
                        if block_text not in blocks:
                            blocks.append(block_text)

    index = {
        rule_id: "\n\n".join(b[:1500] for b in blocks)[:3000]
        for rule_id, blocks in raw_index.items()
    }
    _TEMPLATE_RULE_INDEX_CACHE[templates_dir] = index
    return index


def extract_template_rule_block(templates_dir: str, rule_id: str) -> str:
    """Looks up rule_id's Jinja conditional block across every .j2 template
    in the role (see _build_template_rule_index's docstring). Returns "" if
    no template in this role gates content behind this specific rule_id."""
    if not templates_dir or not rule_id:
        return ""
    return _build_template_rule_index(templates_dir).get(rule_id, "")


def _humanize_unresolved_jinja(text: str) -> str:
    """
    Replace Jinja2 expressions that remained unresolved (e.g. values
    from register:, only available at Ansible runtime) with a readable,
    descriptive marker instead of the raw {{ }} syntax. This is NOT an attempt
    to guess the actual value — such values are by definition
    unknown without actually running the playbook. The goal is solely
    to clean up the semantic text of syntactic noise (curly braces,
    dots, Ansible variable names) that degrades embedding quality
    without adding semantic information.
    """
    def repl(m):
        expr = m.group(1).strip()
        if re.search(r"\.(stdout|stdout_lines|rc|results|changed|failed|stat)\b", expr):
            return "<command result>"
        if expr.startswith("item"):
            return "<list item>"
        return "<variable>"
    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", repl, text)


def build_semantic(name: str, module: str, params: dict,
                   variables: dict, loop_var: str = "",
                   script_summary: str = "", loop_literal: list = None,
                   parent_block_name: str = "") -> str:
    clean = re.sub(r"^\[?\d+(?:\.\d+)*\]?\s*[|\-–:]\s*", "", name).strip()
    # All segments of names like "NUMBER | STAGE | Rule description |
    # Substep" are kept (separator replaced with a dash), because the rule
    # description is not always in the same segment.
    clean = clean.replace("|", " - ").strip()
    if not clean:
        clean = name
    # A task name inside a loop often contains an unresolved Jinja2 placeholder
    # (e.g. "Ensure {{ item }} kernel module..." or "{{ item.path }}").
    # The actual loop values are added separately below anyway (the
    # "items: ..." section), so a raw placeholder in the name would only add noise
    # to semantic without adding information — we remove it.
    clean = re.sub(r"\{\{\s*item(?:\.\w+)?\s*(?:\|[^}]*)?\}\}", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()

    # The enclosing block's name (usually the full CIS rule title) is
    # prepended when the sub-task's own name does not contain it, which
    # gives retrieval a much more descriptive query.
    if parent_block_name:
        clean_block = re.sub(r"^\[?\d+(?:\.\d+)*\]?\s*[|\-–:]\s*", "", parent_block_name).strip()
        clean_block = clean_block.replace("|", " - ").strip()
        if clean_block and clean_block.lower() not in clean.lower():
            clean = f"{clean_block} - {clean}" if clean else clean_block

    short_module = module.split(".")[-1] if "." in module else module
    ctx = MODULE_CONTEXT.get(module) or MODULE_CONTEXT.get(short_module, f"module {short_module}")
    parts = [clean, ctx]
    if isinstance(params, dict):
        for k in ("path", "dest", "name", "src", "file"):
            v = params.get(k)
            if v and str(v) not in ("<loop_item>", "", "None"):
                parts.append(str(v)[:80]); break
        for k in ("line", "value", "content", "regexp", "insertafter"):
            v = params.get(k)
            if v and "<loop_item>" not in str(v) and str(v) not in ("", "None"):
                parts.append(str(v)[:80]); break
        for k in ("state", "enabled", "mode", "owner"):
            v = params.get(k)
            if v is not None:
                parts.append(f"{k}={v}")
    # "items:" are added only when the loop variable was detected in this
    # task's own with_items:/loop: and exists in the variable context.
    if loop_var and loop_var in variables:
        val = variables[loop_var]
        if isinstance(val, list):
            # No truncation: 'semantic' is both the retrieval query and the
            # "Semantic description" in Agent 2's prompt, so the full loop
            # list is kept.
            parts.append("items: " + ", ".join(str(x) for x in val))
        elif isinstance(val, dict):
            parts.append("params: " + str(val)[:60])
    elif loop_literal:
        # A loop written as a literal inline list (e.g. "loop: [{path:
        # grub.cfg, mode: ...}, ...]") is added directly, without a variable
        # lookup. No truncation, as above.
        parts.append("items: " + ", ".join(str(x)[:60] for x in loop_literal))
    if script_summary:
        parts.append(script_summary[:200])
    result = " ".join(str(p) for p in parts if p)
    # Category 2: clean up remaining unresolved Jinja2 expressions
    # (mainly register: values) - replace with a readable descriptive marker
    if "{{" in result:
        result = _humanize_unresolved_jinja(result)
    return result


def parse_one(task: dict, variables: dict,
              parent_tags=None, parent_when=None,
              source_file: str = "", raw_lines: list = None,
              files_dir: str = "", templates_dir: str = "",
              parent_block_name: str = "", parent_notify=None) -> Optional[TaskDescription]:
    if not isinstance(task, dict):
        return None

    # Variables defined in this task's own vars: are merged into the context,
    # so that loops over them can be resolved.
    local_vars = task.get("vars", {})
    if isinstance(local_vars, dict) and local_vars:
        resolved_local = resolve_jinja(local_vars, variables)
        if isinstance(resolved_local, dict):
            variables = {**variables, **resolved_local}

    raw_name = str(task.get("name", "unnamed"))
    # Resolve Jinja2 in the task name the same way as in the parameters — without this
    # an inconsistency arose: the 'line' parameter correctly substituted
    # '{{ min_uid }}' -> '1000', but the same variable in the name: field stayed
    # as raw, unresolved text (e.g. "Ensure UID_MIN is set to
    # {{min_uid}}..."), even though the value was available in variables.
    resolved_name = resolve_jinja(raw_name, variables)
    name = resolved_name if isinstance(resolved_name, str) else raw_name

    raw_tags = task.get("tags", [])
    if not isinstance(raw_tags, list): raw_tags = [raw_tags] if raw_tags else []
    tags = list(parent_tags or []) + raw_tags

    raw_when = task.get("when", [])
    if isinstance(raw_when, str): raw_when = [raw_when]
    elif not isinstance(raw_when, list): raw_when = []
    when = list(parent_when or []) + raw_when

    module = None
    for key in task:
        if key in RESERVED_KEYS: continue
        module = key; break
    if module is None:
        return None

    # Check both the bare module name and its last segment — modern
    # projects (e.g. ansible-lockdown) use exclusively fully qualified
    # names (ansible.builtin.debug, ansible.builtin.set_fact...), which
    # would otherwise bypass SKIP_MODULES and be incorrectly treated as a
    # real hardening task.
    if module in SKIP_MODULES or module.split(".")[-1] in SKIP_MODULES:
        # Tasks of SKIP_MODULES (debug, assert, set_fact, ...) are skipped
        # only when they carry no CIS rule reference in tags/when/name. Some
        # projects implement remediation through a rule-tagged set_fact and a
        # shared handler, and such tasks must stay in the parsed output.
        early_cis_ids, _ = extract_cis_ids(tags, when, name, "")
        if not early_cis_ids:
            return None

    raw_params = task.get(module, {})
    params = resolve_jinja(raw_params, variables)
    if not isinstance(params, dict):
        params = {"_raw_params": str(params)}

    loop_var = ""
    loop_literal = None
    loop_items = task.get("with_items") or task.get("loop")
    if loop_items and isinstance(loop_items, str):
        # The loop variable name is matched even when a Jinja2 filter is
        # present (e.g. "{{ bootloader_files | list }}").
        m = re.search(r"\{\{\s*(\w+)", loop_items)
        if m: loop_var = m.group(1)
    elif loop_items and isinstance(loop_items, list):
        # A literal inline loop list has no variable name; its items are
        # taken directly from the YAML (see build_semantic()).
        loop_literal = loop_items

    # Resolve each loop item against project variables — see
    # TaskDescription.resolved_loop's docstring for why this closes a real
    # information gap (item.xxx references in the task's own params combined
    # with an unresolved {{ project_var }} inside the loop item itself).
    resolved_loop = []
    if loop_literal:
        resolved = resolve_jinja(loop_literal, variables)
        if isinstance(resolved, list):
            resolved_loop = resolved
            # Ansible flattens 'with_items:' one level when its single item is
            # itself a list ('- "{{ some_list_var }}"'), so the same is done
            # here when unambiguous.
            if len(resolved_loop) == 1 and isinstance(resolved_loop[0], list):
                resolved_loop = resolved_loop[0]

    # SemanticMapper extension: for script:/command:/shell: tasks that
    # delegate logic to an external .sh file, extract a condensed
    # description of what the script does — otherwise semantic contains only the file name
    script_summary = ""
    script_ref = None
    if module == "script":
        script_ref = params.get("_raw_params") or (
            list(params.values())[0] if len(params) == 1 else None
        )
        if isinstance(raw_params, str):
            script_ref = raw_params
    elif module in ("command", "shell"):
        script_ref = params.get("cmd", "")
    if script_ref and isinstance(script_ref, str) and script_ref.strip().endswith(".sh"):
        script_summary = extract_script_summary(files_dir, script_ref.strip().split()[0])

    # Analogous extension for the template: module — the .j2 file often
    # contains the entire actual configuration (e.g. auditd rules), while
    # the task's parameters only have src:/dest: (file names)
    if module == "template" and not script_summary:
        template_src = params.get("src", "")
        if template_src and isinstance(template_src, str) and template_src.strip().endswith(".j2"):
            script_summary = extract_template_summary(templates_dir, template_src.strip(), variables)


    # MODULE 4 addition: try to extract the ID from a comment near the task
    comment_id = ""
    if raw_lines is not None:
        line_no = _get_line(task)
        comment_id = _extract_cis_from_comment_window(raw_lines, line_no)

    cis_ids, cis_source = extract_cis_ids(tags, when, name, comment_id)
    level = extract_level(tags)
    when_rules = []
    for cond in when:
        for pat in WHEN_RULE_PATTERNS:
            m = pat.match(str(cond).strip())
            if m: when_rules.append(m.group(1).replace("_", "."))

    # See extract_template_rule_block()'s docstring: looks up, across EVERY
    # .j2 template in this role (not just whichever task's 'src:' happens to
    # reference one), the Jinja conditional block gated behind THIS task's
    # own rule_id(s) — the real enforcement content for projects where a
    # shared template renders many CIS rules' actual config at once.
    template_content = ""
    if templates_dir and cis_ids:
        blocks = []
        for rid in cis_ids:
            block = extract_template_rule_block(templates_dir, rid)
            if block:
                blocks.append(f"[{rid}]\n{block}")
        template_content = "\n\n".join(blocks)

    return TaskDescription(
        task_name=name, module=module, params=params,
        # The resolved loop items are used (e.g. 'disk_full_action = halt'
        # instead of '{{ ... }}'), which is more informative for retrieval.
        semantic=build_semantic(name, module, params, variables, loop_var, script_summary,
                                 resolved_loop or loop_literal, parent_block_name),
        # Inherit the ENCLOSING block's notify: (see parse_list's
        # this_notify comment) into raw_yaml when this specific task has
        # none of its own — a dict copy so the original 'task' object
        # (still used above for module/param detection) is left untouched.
        raw_yaml=_dump_yaml([{**task, "notify": parent_notify} if (parent_notify and not task.get("notify")) else task]),
        cis_rule_ids=cis_ids,
        cis_id_source=cis_source,
        cis_level=level,
        source_file=source_file,
        when_rules=when_rules,
        template_content=template_content,
        resolved_loop=resolved_loop,
    )


def parse_list(items, variables: dict,
              parent_tags=None, parent_when=None,
              source_file: str = "", raw_lines: list = None,
              files_dir: str = "", templates_dir: str = "",
              parent_block_name: str = "", parent_notify=None) -> list:
    results = []
    for item in (items or []):
        if not isinstance(item, dict): continue

        raw_tags = item.get("tags", [])
        if not isinstance(raw_tags, list): raw_tags = [raw_tags] if raw_tags else []
        merged_tags = list(parent_tags or []) + raw_tags

        raw_when = item.get("when", [])
        if isinstance(raw_when, str): raw_when = [raw_when]
        elif not isinstance(raw_when, list): raw_when = []
        merged_when = list(parent_when or []) + raw_when

        if "block" in item:
            # Category 3: blocks often define local variables in vars:
            # used later by loops (loop:/with_items:) in subtasks,
            # e.g. a list of dicts to iterate over that isn't in defaults/
            # main.yml. Without this merge, these variables were invisible to
            # VariableResolver and loops remained unresolved in semantic.
            block_vars = item.get("vars", {})
            scoped_variables = variables
            if isinstance(block_vars, dict) and block_vars:
                resolved_block_vars = resolve_jinja(block_vars, variables)
                if isinstance(resolved_block_vars, dict):
                    scoped_variables = {**variables, **resolved_block_vars}

            # The block's own name is passed down to its sub-tasks (a block
            # without a name keeps the name inherited from its parent).
            this_block_name = item.get("name") or parent_block_name

            # A block-level 'notify:' applies to the whole block in Ansible,
            # so it is propagated to every sub-task's raw_yaml (a sub-task's
            # own notify:, if present, takes precedence).
            this_notify = item.get("notify") if item.get("notify") else parent_notify

            for section in ("block", "rescue", "always"):
                sub = item.get(section) or []
                results.extend(parse_list(sub, scoped_variables, merged_tags, merged_when,
                                          source_file, raw_lines, files_dir, templates_dir,
                                          this_block_name, this_notify))
        else:
            td = parse_one(item, variables, merged_tags, merged_when,
                           source_file, raw_lines, files_dir, templates_dir,
                           parent_block_name, parent_notify)
            if td: results.append(td)
    return results


def expand_file(filepath: str, tasks_dir: str, variables: dict,
                visited: set, depth: int = 0) -> list:
    if depth > 15 or filepath in visited:
        return []
    visited.add(filepath)

    if not os.path.exists(filepath):
        basename = os.path.basename(filepath)
        candidate = os.path.join(tasks_dir, basename)
        if os.path.exists(candidate):
            filepath = candidate
        else:
            return []

    # files/ and templates/ are sibling directories to tasks/ in the standard
    # Ansible role structure
    files_dir = os.path.join(os.path.dirname(tasks_dir), "files")
    if not os.path.isdir(files_dir):
        files_dir = ""

    templates_dir = os.path.join(os.path.dirname(tasks_dir), "templates")
    if not os.path.isdir(templates_dir):
        templates_dir = ""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        raw_lines = raw_text.split("\n")
        data = _load_yaml(filepath)
    except Exception as e:
        print(f"  [P0] Error loading {os.path.basename(filepath)}: {e}")
        return []

    if not isinstance(data, list):
        return []

    current_dir = os.path.dirname(filepath)

    def _find_include_key(d: dict):
        # short-name match (not only bare "include_tasks"/"import_tasks") —
        # ansible-lockdown and other modern projects use exclusively FQCN
        # (ansible.builtin.import_tasks), which would otherwise be skipped
        # and incorrectly parsed as a regular task.
        return next((k for k in d if k.split(".")[-1] in ("include_tasks", "import_tasks")), None)

    def _resolve_includes(items):
        """Recursively walk `items`, expanding include_tasks/import_tasks
        directives also when they are nested inside block:/rescue:/always:
        (some projects group imports in a block to share one when:
        condition). Returns (expanded_task_descriptions, cleaned_items),
        where cleaned_items is `items` without the resolved include entries."""
        expanded, cleaned = [], []
        for item in (items or []):
            if not isinstance(item, dict):
                continue
            inc_key = _find_include_key(item)
            if inc_key is not None:
                inc = item[inc_key]
                if isinstance(inc, dict):
                    inc = inc.get("file", "") or inc.get("name", "")
                inc_str = str(resolve_jinja(str(inc).strip(), variables))
                if inc_str and "{{" not in inc_str:
                    # A relative include_tasks/import_tasks path is resolved
                    # relative to the including file's directory first (as in
                    # Ansible, e.g. tasks/section_1/main.yml importing
                    # cis_1.1.1.x.yml), then relative to the role's tasks/.
                    if os.path.isabs(inc_str):
                        inc_path = inc_str
                    else:
                        sibling_path = os.path.join(current_dir, inc_str)
                        inc_path = sibling_path if os.path.exists(sibling_path) else os.path.join(tasks_dir, inc_str)
                    expanded.extend(expand_file(inc_path, tasks_dir, variables, visited, depth + 1))
                continue  # drop the include directive itself
            if any(k in item for k in ("block", "rescue", "always")):
                new_item = dict(item)
                for section in ("block", "rescue", "always"):
                    if section in new_item:
                        sub_expanded, sub_cleaned = _resolve_includes(new_item[section])
                        expanded.extend(sub_expanded)
                        new_item[section] = sub_cleaned
                cleaned.append(new_item)
            else:
                cleaned.append(item)
        return expanded, cleaned

    td_from_includes, plain_dicts = _resolve_includes(data)

    plain_parsed = parse_list(plain_dicts, variables, source_file=filepath,
                              raw_lines=raw_lines, files_dir=files_dir,
                              templates_dir=templates_dir)
    return td_from_includes + plain_parsed


def load_variables(role_root: str) -> dict:
    variables = {}
    for path in [
        os.path.join(role_root, "defaults", "main.yml"),
        os.path.join(role_root, "defaults", "main.yaml"),
        os.path.join(role_root, "vars", "main.yml"),
        os.path.join(role_root, "vars", "main.yaml"),
    ]:
        if os.path.exists(path):
            try:
                d = _load_yaml(path)
                if isinstance(d, dict):
                    variables.update(d)
            except Exception:
                pass
    return variables


# ═══════════════════════════════════════════════════════════════════════════
# Main entry function — parses only tasks (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

def parse_ansible_project(path: str, role_filter: str = None) -> list:
    """Parse the project and return ONLY the list of tasks (no coverage/rule_status)."""
    path = os.path.abspath(path)

    if os.path.isfile(path):
        return _parse_single_file(path)

    if not os.path.isdir(path):
        print(f"[P0] Not found: {path}")
        return []

    roles = discover_roles(path)
    if not roles:
        print(f"[P0] No Ansible roles found in: {path}")
        return []

    all_tasks = []
    for role_name, role_root in roles:
        if role_filter and role_filter.lower() not in role_name.lower():
            continue
        if "audit" in role_name.lower() and not role_filter:
            print(f"[P0] Skipping role '{role_name}' (audit — contains no hardening tasks)")
            continue

        variables = load_variables(role_root)
        tasks_dir = os.path.join(role_root, "tasks")
        main_yml  = os.path.join(tasks_dir, "main.yml")
        if not os.path.exists(main_yml):
            main_yml = os.path.join(tasks_dir, "main.yaml")
            if not os.path.exists(main_yml):
                print(f"[P0] No tasks/main.yml in role '{role_name}'")
                continue

        print(f"[P0] Role '{role_name}': {len(variables)} variables")
        role_tasks = expand_file(main_yml, tasks_dir, variables, set())
        print(f"[P0] Role '{role_name}': {len(role_tasks)} tasks")
        all_tasks.extend(role_tasks)

    return all_tasks


def _parse_single_file(path: str) -> list:
    tasks_dir = os.path.dirname(path)
    variables = {}
    for cand in [
        os.path.join(tasks_dir, "..", "defaults", "main.yml"),
        os.path.join(tasks_dir, "..", "vars", "main.yml"),
        os.path.join(tasks_dir, "defaults", "main.yml"),
    ]:
        if os.path.exists(cand):
            d = load_variables(os.path.dirname(os.path.dirname(cand)))
            variables.update(d)
            break

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        raw_lines = raw_text.split("\n")
        data = _load_yaml(path)
    except Exception as e:
        print(f"[P0] Error loading {path}: {e}")
        return []

    if not isinstance(data, list):
        return []

    raw = []
    for item in data:
        if not isinstance(item, dict): continue
        if "tasks" in item:
            raw.extend(item["tasks"] or [])
        elif "hosts" not in item:
            raw.append(item)

    visited = {os.path.abspath(path)}
    td_results, plain_dicts = [], []
    for item in raw:
        if not isinstance(item, dict): continue
        inc_key = next((k for k in item if k.split(".")[-1] in ("include_tasks", "import_tasks")), None)
        if inc_key is not None:
            inc = item[inc_key]
            if isinstance(inc, dict): inc = inc.get("file", "") or inc.get("name", "")
            inc_str = str(resolve_jinja(str(inc).strip(), variables))
            if inc_str and "{{" not in inc_str:
                inc_path = os.path.join(tasks_dir, inc_str)
                td_results.extend(expand_file(inc_path, tasks_dir, variables, visited))
        else:
            plain_dicts.append(item)

    return td_results + parse_list(plain_dicts, variables, source_file=path, raw_lines=raw_lines)


# ═══════════════════════════════════════════════════════════════════════════
# CoverageAnalyzer — comparing the full CIS list with what was implemented
# ═══════════════════════════════════════════════════════════════════════════

def is_manual_rule(rule: dict) -> bool:
    """Whether the CIS Benchmark itself marks this rule "(Manual)" — i.e. its
    own assessment/audit procedure cannot be fully automated and requires a
    human to validate the configured state (per CIS's own definition of the
    category). MUST scan full_text, NOT title: title is truncated wherever
    the PDF happened to break the line, so "(Manual)"/"(Automated)" — which
    sits right after the rule's (possibly multi-line) name — is frequently
    cut off from title and only ever appears intact in full_text. Confirmed
    concretely on rule 1.1.1.11: title-based scanning finds 13/355 Manual
    rules project-wide, full_text-based scanning finds the correct 21/355."""
    return "(manual)" in rule.get("full_text", "")[:250].lower()


_PROFILE_LEVEL_RE = re.compile(r"Profile Applicability:\s*(?:\n|\r\n)?\s*[•\-\*]?\s*(Level\s*[12])", re.IGNORECASE)


def extract_cis_profile_level(rule: dict) -> str:
    """CIS Benchmark's OWN official "Profile Applicability" designation for
    this rule — "1" (baseline, broadly applicable to nearly all systems) or
    "2" (defense-in-depth, more environment-specific) — read directly from
    the document text (full_text), same pattern as is_manual_rule().
    Returns "" if not found (section headings, which are not real rules,
    have no Profile Applicability)."""
    m = _PROFILE_LEVEL_RE.search(rule.get("full_text", ""))
    if not m:
        return ""
    return "2" if "2" in m.group(1) else "1"


def get_manual_rule_ids(analysis_json_path: str) -> set:
    """Set of all rule_ids the CIS Benchmark document marks "(Manual)" — used
    to exclude these rules from Agent A2's automated COMPLIANT/NON_COMPLIANT
    audit scoring (a Manual control needs human judgment per CIS's own
    definition; an LLM verdict on it is informational only, not a pass/fail)."""
    if not os.path.exists(analysis_json_path):
        return set()
    with open(analysis_json_path, "r", encoding="utf-8") as f:
        all_rules = json.load(f)["rules"]
    return {r["rule_id"] for r in all_rules if is_manual_rule(r)}


def analyze_coverage(tasks: list, rule_status: dict, analysis_json_path: str) -> list:
    """
    Compare three data sources:
      1. The full list of CIS rules from the document (analysis_results.json, from the PDF)
      2. The status of rule_X variables from defaults (VariableResolver)
      3. Tasks actually implemented in the playbook (SemanticMapper)

    Returns a CoverageReport for each rule from the CIS document — 5 categories:
      IMPLEMENTED        — there is a task that fulfils it
      DISABLED_BY_VAR    — deliberately disabled (rule_X: false)
      MANUAL_CONTROL     — marked "(Manual)" by CIS and without a task
                           (expected: such rules require an organisational decision)
      MISSING_TASK       — enabled (rule_X: true), Automated, but no task
      MISSING_VARIABLE   — no variable at all
    """
    if not os.path.exists(analysis_json_path):
        print(f"  [CoverageAnalyzer] Missing {analysis_json_path} — skipping coverage analysis")
        return []

    with open(analysis_json_path, "r", encoding="utf-8") as f:
        raw_rules = json.load(f)["rules"]
    # Section headings of the benchmark (e.g. "6.3.3 Configure auditd Rules")
    # are not auditable rules. The same filter as in Agent A1 is used: a real
    # rule has both a Description and a Remediation subsection.
    all_rules = [
        r for r in raw_rules
        if "Description" in r.get("subsections", []) and "Remediation" in r.get("subsections", [])
    ]
    skipped = len(raw_rules) - len(all_rules)
    if skipped:
        print(f"  [CoverageAnalyzer] Skipped {skipped} section-heading/noise entries "
              f"(no Description+Remediation of their own) — not real CIS controls.")
    all_rule_ids = {r["rule_id"]: r.get("title", r["rule_id"]) for r in all_rules}
    # The "(Manual)" designation is read from the full rule text via
    # is_manual_rule(), because the title is often truncated by line breaks
    # in the PDF.
    rules_by_id = {r["rule_id"]: r for r in all_rules}

    tasks_by_rule = {}
    for t in tasks:
        for rid in t.cis_rule_ids:
            tasks_by_rule.setdefault(rid, []).append(t.task_name)

    reports = []
    for rule_id, title in all_rule_ids.items():
        status_obj = rule_status.get(rule_id)
        has_var  = status_obj is not None
        var_val  = status_obj.enabled if status_obj else None
        has_task = rule_id in tasks_by_rule
        task_names = tasks_by_rule.get(rule_id, [])

        is_manual = is_manual_rule(rules_by_id.get(rule_id, {}))
        cis_level = extract_cis_profile_level(rules_by_id.get(rule_id, {}))

        if has_var and var_val is False:
            status = "DISABLED_BY_VAR"
        elif has_task:
            status = "IMPLEMENTED"
        elif is_manual:
            status = "MANUAL_CONTROL"
        elif has_var and var_val is True:
            status = "MISSING_TASK"
        else:
            status = "MISSING_VARIABLE"

        reports.append(CoverageReport(
            rule_id=rule_id, title=title, status=status,
            has_variable=has_var, variable_enabled=var_val,
            has_task=has_task, task_names=task_names,
            is_manual=is_manual, cis_level=cis_level,
        ))

    return reports


# ═══════════════════════════════════════════════════════════════════════════
# RiskClassifier — risk assessment of disabled rules + generating a review file
# ═══════════════════════════════════════════════════════════════════════════
# Not every DISABLED_BY_VAR is a deliberate decision — it could be an accidental
# disabling or copying defaults from another project without a second thought.
# This module flags disabled rules covering high-risk areas
# and generates ready-to-review task skeletons.

# Review priority of a disabled rule is based on CIS Benchmark's own
# "Profile Applicability" level (Level 1 = baseline, broadly applicable;
# Level 2 = defense in depth, environment-specific), read from the benchmark
# text by extract_cis_profile_level(). A disabled Level 1 rule gets higher
# priority, because CIS expects it to apply almost universally.


def classify_disabled_risk(cis_level: str) -> str:
    """
    Assess review priority for a disabled rule using CIS Benchmark's OWN
    official Level 1/Level 2 "Profile Applicability" designation (not an
    invented keyword list — see the module comment above). Level 1 =
    HIGH_RISK (CIS's own definition: baseline, broadly applicable — a
    disabled Level 1 control has no environment-specific justification to
    lean on). Level 2 or unknown = LOW_RISK (defense-in-depth, legitimately
    more context-dependent whether it applies).
    """
    return "HIGH_RISK" if cis_level == "1" else "LOW_RISK"


def annotate_disabled_rules(coverage_reports: list) -> list:
    """Returns a list of (CoverageReport, risk_level) for disabled rules."""
    annotated = []
    for r in coverage_reports:
        if r.status == "DISABLED_BY_VAR":
            risk = classify_disabled_risk(r.cis_level)
            annotated.append((r, risk))
    return annotated


def generate_commented_task_stub(rule_id: str, title: str, reason: str) -> str:
    clean_title = title.replace(rule_id, "").strip()
    clean_title = clean_title.lstrip(".: -")
    
    lines = [
        f"# ─────────────────────────────────────────────────────────",
        f"# CIS {rule_id}: {clean_title}",
        f"# Status: {reason}"
    ]
    
    # Split the comment logic!
    if "DISABLED_BY_VAR" in reason:
        lines.extend([
            f"# NOTE: This rule was disabled by the administrator in defaults/ (Risk Acceptance).",
            f"# Agent 3 will NOT automatically generate a fix for it.",
            f"# Changing the variable to 'true' is solely a human decision."
        ])
    else:
        # For MISSING_TASK
        lines.extend([
            f"# This rule will be passed to Agent 2 (RAG Auditor),",
            f"# which will decide whether to pass it to Agent 3 (Remediation)",
            f"# to generate ready-to-use YAML code."
        ])
        
    lines.extend([
        f"# ─────────────────────────────────────────────────────────",
        f"# - name: \"{rule_id} | {clean_title}\"",
        f"#   tags:",
        f"#     - r{rule_id}",
    ])
    
    return "\n".join(lines)


def generate_review_file(coverage_reports: list, output_path: str = "results/review_needed.yml") -> str:
    """
    Generate an informational YAML file listing the CIS rules that require attention:
      - all MISSING_TASK rules (enabled but without a task)
      - all high-priority DISABLED_BY_VAR rules

    The file is only a list. Fixes for these rules are generated by Agent 3,
    called through audit_missing_rules() in agent_a2_llm.py.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    missing = [r for r in coverage_reports if r.status == "MISSING_TASK"]
    disabled_annotated = annotate_disabled_rules(coverage_reports)
    high_risk_disabled = [r for r, risk in disabled_annotated if risk == "HIGH_RISK"]

    blocks = [
        "# ═══════════════════════════════════════════════════════════",
        "# LIST TO PROCESS — generated by Agent 0 (CoverageAnalyzer)",
        "# This file does NOT contain ready-to-use code. It is a list of CIS rules that",
        "# Parser 0 passes to Agent 1 (onboarding) -> Agent 2 -> Agent 3",
        "# to generate the actual fixes (per the system architecture).",
        "# ═══════════════════════════════════════════════════════════",
        "",
    ]

    if missing:
        blocks.append(f"# ─── BUG: {len(missing)} rules ENABLED but without a task ───")
        blocks.append("")
        for r in sorted(missing, key=lambda x: x.rule_id):
            blocks.append(generate_commented_task_stub(
                r.rule_id, r.title, "MISSING_TASK (rule=true, no implementation)"
            ))
            blocks.append("")

    if high_risk_disabled:
        blocks.append(f"# ─── NEEDS REVIEW: {len(high_risk_disabled)} disabled high-risk rules ───")
        blocks.append("")
        for r in sorted(high_risk_disabled, key=lambda x: x.rule_id):
            blocks.append(generate_commented_task_stub(
                r.rule_id, r.title, "DISABLED_BY_VAR — high risk, verify whether intentional"
            ))
            blocks.append("")

    content = "\n".join(blocks)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[P0] Generated list to process: {output_path}")
    print(f"     {len(missing)} missing tasks + {len(high_risk_disabled)} risky disabled rules")
    print(f"     Pass this list to Agent 2 (audit_missing_rules) for full fixes.")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# FULL entry point — all 4 modules in a single call
# ═══════════════════════════════════════════════════════════════════════════

def parse_ansible_project_full(path: str, analysis_json_path: str = "analysis_results.json",
                                role_filter: str = None,
                                generate_review: bool = True,
                                review_output_path: str = "results/review_needed.yml") -> dict:
    """
    Runs the entire Parser A0 pipeline: FileDiscovery -> YAMLParser ->
    VariableResolver -> SemanticMapper -> CoverageAnalyzer -> RiskClassifier.
    Does not call an LLM and does not generate fixes; it only identifies and
    reports discrepancies (review_needed.yml).

    Returns a dict:
      {
        "tasks":         list of TaskDescription,
        "rule_status":   {rule_id: RuleStatus},
        "coverage":      list of CoverageReport (if analysis_json_path was provided),
        "review_file":   path to the generated list (or None)
      }
    """
    print(f"\n[P0] === Running the full pipeline (4 modules + RiskClassifier) ===")

    print(f"[P0] Module 1+2+4: FileDiscovery + YAMLParser + SemanticMapper...")
    tasks = parse_ansible_project(path, role_filter=role_filter)

    print(f"\n[P0] Module 3: VariableResolver...")
    rule_status = resolve_rule_variables(path)
    print(f"  Found {len(rule_status)} CIS rule variables in defaults/*")

    coverage = []
    review_file = None
    if os.path.exists(analysis_json_path):
        print(f"\n[P0] CoverageAnalyzer: comparing against the full CIS list...")
        coverage = analyze_coverage(tasks, rule_status, analysis_json_path)

        if generate_review and coverage:
            print(f"\n[P0] RiskClassifier: identifying rules to pass to Agent 2...")
            review_file = generate_review_file(coverage, review_output_path)

    return {
        "tasks":       tasks,
        "rule_status": rule_status,
        "coverage":    coverage,
        "review_file": review_file,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reports
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(tasks: list, verbose: bool = False):
    print(f"\n{'='*65}")
    print(f"MODULE 4 (SemanticMapper) — Task parsing result")
    print(f"{'='*65}")
    print(f"Total tasks          : {len(tasks)}")
    with_cis = sum(1 for t in tasks if t.cis_rule_ids)
    print(f"With CIS rule ID     : {with_cis} ({with_cis/len(tasks)*100:.0f}%)" if tasks else "0")

    by_source = {}
    for t in tasks:
        if t.cis_rule_ids:
            by_source[t.cis_id_source] = by_source.get(t.cis_id_source, 0) + 1
    if by_source:
        print(f"\nCIS rule ID source:")
        for src, c in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {src:<10} {c:>4}x")

    modules = {}
    for t in tasks:
        short = t.module.split(".")[-1]
        modules[short] = modules.get(short, 0) + 1
    print(f"\nAnsible modules (top 8):")
    for m, c in sorted(modules.items(), key=lambda x: -x[1])[:8]:
        print(f"  {m:<22} {c:>4}x")


def print_rule_status(rule_status: dict):
    enabled  = [s for s in rule_status.values() if s.enabled]
    disabled = [s for s in rule_status.values() if not s.enabled]
    print(f"\n{'='*65}")
    print(f"MODULE 3 (VariableResolver) — CIS rule variable status")
    print(f"{'='*65}")
    print(f"Rule variables found : {len(rule_status)}")
    print(f"Enabled                       : {len(enabled)}")
    print(f"Disabled                      : {len(disabled)}")
    if disabled:
        print(f"\nDisabled rules ({len(disabled)}):")
        for s in sorted(disabled, key=lambda x: x.rule_id)[:15]:
            title = f" — {s.comment}" if s.comment else ""
            print(f"  {s.rule_id:<15} [{s.var_name}]{title}")


def print_coverage(coverage: list):
    if not coverage:
        return
    counts = {}
    for r in coverage:
        counts[r.status] = counts.get(r.status, 0) + 1
    total = len(coverage)
    print(f"\n{'='*65}")
    print(f"COVERAGE ANALYZER — CIS rule coverage")
    print(f"{'='*65}")
    print(f"CIS rules in document             : {total}")
    print(f"  IMPLEMENTED (implemented)      : {counts.get('IMPLEMENTED', 0)}")
    print(f"  DISABLED_BY_VAR (disabled)     : {counts.get('DISABLED_BY_VAR', 0)}")
    print(f"  MANUAL_CONTROL (outside CIS Auto) : {counts.get('MANUAL_CONTROL', 0)}")
    print(f"  MISSING_TASK (BUG!)            : {counts.get('MISSING_TASK', 0)}")
    print(f"  MISSING_VARIABLE (skipped)     : {counts.get('MISSING_VARIABLE', 0)}")
    coverage_pct = counts.get("IMPLEMENTED", 0) / total * 100
    print(f"\nImplementation coverage: {coverage_pct:.1f}%")

    missing_task = [r for r in coverage if r.status == "MISSING_TASK"]
    if missing_task:
        print(f"\n⚠ CRITICAL — rules ENABLED but WITHOUT an implementation ({len(missing_task)}):")
        for r in sorted(missing_task, key=lambda x: x.rule_id)[:15]:
            print(f"  {r.rule_id:<12} {r.title[:60]}")


def print_disabled_risk_report(coverage: list):
    """Print disabled rules grouped by risk level."""
    annotated = annotate_disabled_rules(coverage)
    if not annotated:
        return

    high = [(r, risk) for r, risk in annotated if risk == "HIGH_RISK"]
    low  = [(r, risk) for r, risk in annotated if risk == "LOW_RISK"]

    print(f"\n{'='*65}")
    print(f"RISK CLASSIFIER — deliberately disabled rules (DISABLED_BY_VAR)")
    print(f"{'='*65}")
    print(f"Total disabled: {len(annotated)}")
    print(f"  ⚠ HIGH RISK (worth verifying manually): {len(high)}")
    print(f"  ○ Low risk (probably deliberate)        : {len(low)}")

    if high:
        print(f"\n⚠ HIGH RISK — manual verification recommended:")
        for r, _ in sorted(high, key=lambda x: x[0].rule_id):
            print(f"  {r.rule_id:<15} {r.title[:60]}")

    if low:
        print(f"\n○ Low risk:")
        for r, _ in sorted(low, key=lambda x: x[0].rule_id)[:10]:
            print(f"  {r.rule_id:<15} {r.title[:60]}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    project_path = sys.argv[1]
    analysis_json = next((a.split("=")[1] for a in sys.argv if a.startswith("--analysis=")),
                        "analysis_results.json")
    role_filter = next((a.split("=")[1] for a in sys.argv if a.startswith("--role=")), None)

    result = parse_ansible_project_full(project_path, analysis_json, role_filter)

    print_summary(result["tasks"])
    print_rule_status(result["rule_status"])
    print_coverage(result["coverage"])
    print_disabled_risk_report(result["coverage"])

    if result["review_file"]:
        print(f"\n{'='*65}")
        print(f"Manual review file saved: {result['review_file']}")
        print(f"{'='*65}")
