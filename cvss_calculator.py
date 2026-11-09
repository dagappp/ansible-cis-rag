"""
cvss_calculator.py — deterministic implementation of the CVSS v3.1 Base Score formula.

Source: CVSS v3.1 Specification Document (FIRST.org), Section 7.1 (formulas)
and Section 7.4 (metric weights), checked directly against the document:
https://www.first.org/cvss/v3.1/specification-document

Purpose: the LLM assesses ONLY the 8 base metrics (AV/AC/PR/UI/S/C/I/A) as
letters — the same inputs a person selects in the official FIRST.org
calculator (https://www.first.org/cvss/calculator/3.1). The NUMERIC SCORE is
computed here, in Python, with the same formula as the calculator, so the
arithmetic does not depend on the LLM. Every vector can be pasted into the
official calculator to obtain the identical score, which makes the result
verifiable and reproducible.

Scope note: this implementation guarantees the correctness of the
CALCULATION itself (the math is exactly that of the official calculator).
The assessment of the INPUT metrics for a given finding remains an LLM
judgment, like the rest of the audit (CVSS was designed for vulnerabilities
with a clear attack path rather than configuration gaps — see the
discussion in agent_a2_llm.py).
"""

import math

# ── Metric weights (CVSS v3.1 Specification, Section 7.4) ────────────────────
AV_WEIGHTS = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
AC_WEIGHTS = {"L": 0.77, "H": 0.44}
UI_WEIGHTS = {"N": 0.85, "R": 0.62}
CIA_WEIGHTS = {"H": 0.56, "L": 0.22, "N": 0.0}

# Privileges Required depends on Scope (different weights for Unchanged/Changed)
PR_WEIGHTS = {
    "U": {"N": 0.85, "L": 0.62, "H": 0.27},
    "C": {"N": 0.85, "L": 0.68, "H": 0.5},
}

# CVSS v3.1 Qualitative Severity Rating Scale (Section 5 of the
# specification) — the same table as SEVERITY_RUBRIC in agent_a2_llm.py,
# used here to map the COMPUTED score to a label.
def score_to_rating(score: float) -> str:
    if score == 0.0:
        return "NONE"
    if score <= 3.9:
        return "LOW"
    if score <= 6.9:
        return "MEDIUM"
    if score <= 8.9:
        return "HIGH"
    return "CRITICAL"


def _roundup(value: float) -> float:
    """Roundup per Appendix A of the CVSS v3.1 specification — deliberately
    integer arithmetic (not plain round()) to avoid floating-point errors that
    would give a different result than the official calculator at boundary values."""
    int_input = round(value * 100000)
    if int_input % 10000 == 0:
        return int_input / 100000.0
    return (math.floor(int_input / 10000) + 1) / 10.0


def calculate_base_score(av: str, ac: str, pr: str, ui: str, s: str,
                         c: str, i: str, a: str):
    """Input: letters exactly as in an official CVSS v3.1 vector
    (AV: N/A/L/P, AC: L/H, PR: N/L/H, UI: N/R, S: U/C, C/I/A: N/L/H).
    Returns (score: float, rating: str, vector: str)."""
    av, ac, pr, ui, s, c, i, a = (x.upper() for x in (av, ac, pr, ui, s, c, i, a))
    if s not in ("U", "C"):
        raise ValueError(f"Scope must be U or C, got: {s}")

    iss = 1 - (1 - CIA_WEIGHTS[c]) * (1 - CIA_WEIGHTS[i]) * (1 - CIA_WEIGHTS[a])

    if s == "U":
        impact = 6.42 * iss
    else:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15

    exploitability = 8.22 * AV_WEIGHTS[av] * AC_WEIGHTS[ac] * PR_WEIGHTS[s][pr] * UI_WEIGHTS[ui]

    if impact <= 0:
        score = 0.0
    elif s == "U":
        score = _roundup(min(impact + exploitability, 10.0))
    else:
        score = _roundup(min(1.08 * (impact + exploitability), 10.0))

    vector = f"CVSS:3.1/AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{c}/I:{i}/A:{a}"
    return score, score_to_rating(score), vector


if __name__ == "__main__":
    # Two well-known reference vectors, widely cited in CVSS training
    # material, used as a self-check of the implementation:

    # Scope: Unchanged, all Impact=High -> the well-known score 9.8
    # (NOT 10.0 — the ceiling for Scope:Unchanged is lower even with maximal
    # inputs, because the Impact formula for Unchanged is more conservative).
    score, rating, vector = calculate_base_score("N", "L", "N", "N", "U", "H", "H", "H")
    print(f"{vector} -> {score} ({rating})")
    assert score == 9.8 and rating == "CRITICAL", f"Implementation error: expected 9.8, got {score}"

    # Scope: Changed, the same maximal inputs -> 10.0 (Scope:Changed can
    # reach the absolute maximum of the scale, unlike Unchanged)
    score2, rating2, vector2 = calculate_base_score("N", "L", "N", "N", "C", "H", "H", "H")
    print(f"{vector2} -> {score2} ({rating2})")
    assert score2 == 10.0 and rating2 == "CRITICAL", f"Implementation error: expected 10.0, got {score2}"

    print("Self-check OK.")
