# Multi-agent LLM+RAG system for auditing and remediating CIS Benchmark compliance of Ansible roles

Source code of the pipeline described in the master's thesis: a static parser, five LLM agents with a validation layer,
construction of a vector knowledge base from CIS Benchmark documents, and an LLM judge. The full system prompts of the
agents and the judge are in the code of the respective modules (text constants passed to the model).

## Structure

| File / directory | Role |
|---|---|
| `parser_a0_final.py` | Parser A0: deterministic analysis of Ansible roles, CIS rule identifiers, semantic task descriptions |
| `agent_a1_onboarding.py` | Agent A1: configuration reconciliation (undefined and disabled rules) |
| `agent_a2_llm.py` | Agent A2: auditing tasks against standard fragments retrieved from the knowledge base (RAG), CVSS assessment |
| `agent_a3_llm.py` | Agent A3: fix generation |
| `agent_a4.py` | Agent A4: validation (syntax, ansible-lint, semantic check), self-repair loop, write-back |
| `agent_a5_report.py` | Agent A5: final report |
| `cvss_calculator.py` | Deterministic CVSS 3.1 score computation from a vector |
| `debug_dump.py` | Logging of model calls to a diagnostic log |
| `orchestrator.py` | Runs the whole pipeline |
| `analyze_pdf_docling.py`, `ingest_benchmark.py` | Rule extraction from CIS Benchmark PDFs and vector knowledge base construction (ChromaDB) |
| `judge_audit.py`, `judge_audit_helpers.py`, `judge_combined.py`, `model_comparison_grounded_judge.py`, `quality_judge.py` | LLM judge: independent audit, assessment of a single fix (compliance and code quality) |

## Not included in the repository

- **Reference projects** (Ansible roles for RHEL 10, Debian 12, PostgreSQL 15). They contain environment
  configuration data and are not published.
- **CIS Benchmark documents (PDF) and the knowledge base built from them.** The CIS license does not permit their
  redistribution. The documents can be downloaded from the Center for Internet Security website and the knowledge base
  can be built with `analyze_pdf_docling.py` and `ingest_benchmark.py`.
- **Intermediate results** containing code of the reference projects.

## Running

Requires Python 3.11 and a model server compatible with the OpenAI API (in the thesis: `llama.cpp` with the
`llama-swap` router).

```bash
pip install -r requirements.txt
export LLM_API_URL="http://<server>:<port>/v1/chat/completions"
export LLM_API_TOKEN="<token>"
python orchestrator.py <ansible_project_directory>
```

The server address and token are read from environment variables.

## Code version

The code corresponds to the version used in the ablation study, the environment validation (OpenSCAP, CIS-Audit)
and the manual verification described in the thesis.
