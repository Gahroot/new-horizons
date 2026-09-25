# New Horizons — Active Discovery Engine (build plan)

## Goal

A working, local, topic-agnostic discovery engine in `/Users/groot/new-horizons`. You write one
`topic.toml` (Target Goal · Allowed Tools · Validation Metric), run `horizons run`, and the engine loops:

**Deconstruct (literature + baseline) → Hypothesize → Execute (sandbox / data / literature) → Evaluate → Remember → repeat**

until the validation metric is met *and replicated*, the budget runs out, or progress stalls.

"Just works" definition, proven before handoff:
- `horizons doctor` reports every prerequisite (Python, Docker, LLM key, network) with a fix hint.
- `horizons run examples/memory_reduction/topic.toml --llm scripted` completes the full loop end-to-end
  **offline, with no API key**, runs real code in the Docker sandbox, and writes a report.
- With `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` set, the same command without `--llm scripted` runs a real
  research loop.
- A stdlib `unittest` suite passes.

## Current state (inspected)

- Project directory is empty apart from `.ezcoder/`; not a git repo.
- Python 3.14.7 (`/opt/homebrew/bin/python3`), `uv` installed, Docker 29.4.3 daemon reachable.
- Nothing to preserve; no existing conventions.

## Which parts come from where (the Frankenstein map)

| Engine part | Borrowed pattern | Source (Steroids corpus, inspected) |
|---|---|---|
| Stage pipeline + REFINE vs PIVOT decisions, capped pivots | Explicit decision states & rollback | `aiming-lab/AutoResearchClaw` `pipeline/runner.py`, `stages.py` |
| Hypothesis generation from literature synthesis + multi-perspective + judge | Debate/tournament | AutoResearchClaw `stage_impls/_synthesis.py` |
| Structured lessons with categories (system / experiment / literature / analysis) | Lesson extraction | AutoResearchClaw `evolution.py` |
| Generate → evaluate loop with parent lineage and failed-attempt feedback in prompt | Discovery controller | `skydiscover-ai/skydiscover` `default_discovery_controller.py` |
| User-owned `evaluate()` contract returning a metrics dict | Evaluator contract | SkyDiscover `optimize/evaluation/evaluator.py` |
| Hypotheses with falsification criteria, alternative explanations, evidence links, readiness level | Research contracts | `OpenNSWM-Lab/FAROS` `contracts/scientific_research.py` |
| Findings graph: `supports` / `challenges` / `derived_from` / `updates`, whitelisted edge types, link caps | Finding graph | `sapientinc/PRAXIST` `finding_graph_mvp/engine.py` |
| Budget guard that fails closed before an action | Budget ledger | PRAXIST `core/execution_guards.py` |
| Hybrid memory recall: vector + BM25 fused with Reciprocal Rank Fusion (k=60), per-arm caps | RRF fusion | `vectorize-io/hindsight` `engine/search/fusion.py` |
| Semantic Scholar + arXiv search with timeouts, 429/Retry-After backoff, quoted arXiv phrases | Paper sources | `openags/paper-search-mcp` `semantic.py`, `arxiv.py` |
| Container hardening: no-new-privileges, cap-drop, pids limit, memory/CPU limits, network mode | Sandbox config | `opensandbox-group/OpenSandbox` `docker/container_ops.py` |

During implementation, each module is written after re-reading its corpus source above with
`steroids show`/`search`, adapting the pattern rather than copying unrelated machinery.

## Key design decisions

1. **Zero third-party runtime dependencies.** Python stdlib only: `tomllib` (templates), `sqlite3`
   with FTS5 (knowledge base), `urllib.request` (LLM + paper APIs), `xml.etree` (arXiv Atom),
   `subprocess` argv lists (Docker). Nothing to install beyond Python + Docker; nothing to
   supply-chain-audit. Tests use stdlib `unittest`. Packaged with a `pyproject.toml` so
   `uv run horizons …` and `python3 -m horizons …` both work.
2. **Pluggable LLM.** `anthropic` (Messages API), `openai` (any OpenAI-compatible endpoint incl.
   local Ollama via `base_url`), and `scripted` (deterministic offline stub used by the demo and tests).
   Keys only from environment; never written to logs, DB, or reports.
3. **The agent never grades itself.**
   - The validation evaluator is a user-owned file named in the template, mounted **read-only** into the
     sandbox and used as the container entrypoint; the agent only writes `candidate.py`.
   - Success status is set by controller code comparing numbers to the template spec — never by LLM text.
   - A "breakthrough" requires the metric to pass on `replications` independent reruns (default 3).
   - Statistical validation: stdlib permutation test + bootstrap CI; Holm correction across all
     hypotheses tested in the run.
   - Residual risk (documented, not hidden): candidate code shares the evaluator's container process,
     so a malicious candidate could forge output. Mitigations: a per-run random result marker the
     evaluator prints, results parsed only from the final marked line, and controller-side sanity
     checks (non-finite / impossible values → invalid).
4. **Sandbox fails closed.** Docker run with `--network none`, `--read-only`, `--cap-drop ALL`,
   `--security-opt no-new-privileges`, `--pids-limit 256`, `--memory`, `--cpus`, non-root `--user`,
   tmpfs `/tmp`, writable scratch only for the attempt dir, hard timeout + `docker kill`. If Docker is
   unavailable the run **stops** with a doctor hint; a host-subprocess runner exists only behind an
   explicit `sandbox = "unsafe-local"` template value plus `--i-accept-unsafe-local` CLI flag.
5. **Tool allowlist is enforced in code.** `allowed_tools` in the template gates the tool registry;
   the controller cannot call a tool not listed. Literature text and model output are treated as data:
   they can influence hypotheses, never permissions, evaluator path, budget, or template.
6. **Knowledge base = SQLite, one file per workspace** (`.horizons/kb.sqlite`): runs, papers,
   hypotheses, experiments (code hash, metrics, artifacts path, status), findings, edges, lessons,
   plus FTS5 + a local hashed-embedding vector (feature hashing, 512 dims, no model download) fused by
   RRF. Optional provider embeddings are a later extension, not in this build.
7. **Three outcome classes kept distinct** (from the research phase): `refuted` (hypothesis tested and
   failed the metric), `errored` (experiment crashed / invalid — REFINE the experiment), `inconclusive`
   (insufficient evidence / not significant — gather more or PIVOT). Each drives a different next step.
8. **Readiness levels (FAROS):** hypotheses the allowed tools cannot test are recorded as
   `needs_resources` with what is missing, instead of being "tested" by guessing.

## Topic Template (`topic.toml`)

```toml
[topic]
name = "memory-reduction"
goal = "Reduce peak memory of the baseline pairwise-distance routine by 50% without changing results"
context = "Optional background, constraints, known baselines."

[tools]
allowed = ["literature", "python_sandbox"]   # literature | python_sandbox | data_query
[tools.literature]
sources = ["semantic_scholar", "arxiv"]
max_papers = 20
[tools.python_sandbox]
baseline = "baseline.py"          # starting point the agent modifies
evaluator = "evaluate.py"         # user-owned; prints metrics JSON
image = "python:3.13-slim"
timeout_s = 120
memory = "1g"
cpus = 1.0
[tools.data_query]                # only if allowed
database = "data.sqlite"          # opened read-only

[validation]
metric = "peak_memory_mb"
direction = "minimize"            # minimize | maximize
target = { relative_to_baseline = -0.5 }   # or { absolute = 0.05 }
guards = [{ metric = "correct", op = "==", value = 1 }]
replications = 3
# for statistical claims instead:
# kind = "significance"; alpha = 0.05; test = "permutation"

[budget]
max_iterations = 12
max_llm_calls = 200
max_sandbox_runs = 60
max_wall_clock_min = 60
patience = 4                      # iterations without improvement before stop
max_pivots = 3
```

## Package layout

```
pyproject.toml            # name=new-horizons, console script `horizons`, requires-python>=3.11
README.md                 # quickstart, template reference, safety model, limits
horizons/
  __init__.py  __main__.py
  cli.py                  # init | doctor | run | status | report | memory
  template.py             # load + validate topic.toml → frozen dataclasses; fail on unknown keys
  llm.py                  # LLMClient protocol; Anthropic, OpenAI-compat, Scripted; JSON extraction; call budget
  budget.py               # Budget guard (fail closed), counters persisted per run
  kb/
    store.py              # SQLite schema, migrations-by-version, CRUD
    vectors.py            # feature-hash embedding + cosine
    recall.py             # FTS5 BM25 arm + vector arm + RRF fusion (Hindsight)
    graph.py              # finding edges with whitelisted types + caps (PRAXIST)
  tools/
    registry.py           # allowlist-gated tool lookup
    literature.py         # SemanticScholar + arXiv, retries/backoff, SQLite response cache
    sandbox.py            # DockerSandbox, UnsafeLocalSandbox; argv-only subprocess; timeout/kill
    data_query.py         # read-only SQLite (URI mode=ro + authorizer denying writes), row caps
  agents/
    deconstructor.py      # queries → papers → boundary map with verified citations; baseline profiling
    hypothesizer.py       # N candidates w/ falsification criteria, dedupe vs memory, judge ranking
    executor.py           # dispatch by hypothesis.test_kind → sandbox / data / literature probe
    evaluator.py          # metric spec check, guards, replications, stats, outcome class
    reflector.py          # lesson extraction (category + cause) → KB; REFINE/PIVOT/STOP decision input
  stats.py                # permutation test, bootstrap CI, Holm correction (stdlib)
  controller.py           # the loop, state machine, decisions, checkpoint/resume
  report.py               # markdown report: boundary map, hypotheses table, best result, lessons, limits
  prompts/                # *.md prompt templates, loaded via importlib.resources
examples/
  memory_reduction/       # real runnable demo: topic.toml, baseline.py, evaluate.py, scripted_llm.json
  literature_question/    # literature-only topic (evidence support/challenge), no sandbox
tests/                    # unittest: template, kb+recall, stats, sandbox(docker, skipped if absent),
                          #           evaluator decisions, controller e2e with scripted LLM
```

## Loop semantics (controller)

1. **Bootstrap:** validate template → check tool prerequisites → create run row → measure baseline
   (run the unmodified baseline through the evaluator `replications` times).
2. **Deconstruct:** LLM proposes search queries → literature tool → store papers → LLM writes a
   boundary map (known approaches, gaps, promising directions) citing paper IDs; citations not in the
   KB are stripped and logged as a literature lesson.
3. **Hypothesize:** input = goal, boundary map, best result so far, recalled lessons (RRF top-k);
   output = JSON list `{statement, rationale, test_kind, falsification, expected_effect, citations}`;
   near-duplicates of refuted/errored hypotheses (vector sim > 0.9) are dropped; judge picks one.
4. **Execute:** for `python_sandbox` the LLM writes `candidate.py` from the current best program
   (lineage recorded, failed attempts fed back as in SkyDiscover); the sandbox runs the user
   evaluator. `data_query`: LLM writes SELECT queries + a Python analysis run in the sandbox over
   the exported rows. `literature`: targeted search + LLM classifies each paper as
   supports/challenges/unrelated with quoted evidence → finding edges.
5. **Evaluate:** controller code decides `supported | refuted | errored | inconclusive`, never the LLM.
   On first pass-of-target, run replications; confirmed → `breakthrough` finding.
6. **Reflect:** LLM explains *why* (for refuted/inconclusive); code attaches category and stores a
   lesson. Decision: `errored` → REFINE (retry experiment, max 2) · `refuted`/`inconclusive` → PIVOT
   (new hypothesis, bounded by `max_pivots` per lineage) · target met + replicated → STOP (success) ·
   budget/patience exhausted → STOP (report best-so-far, clearly labelled as not meeting target).
7. Every step checkpoints to SQLite, so `horizons run --resume <run_id>` continues after a crash.

## Security controls (bulletproof inline gate)

- Generated code runs only in the hardened container; no network; evaluator & baseline mounted `:ro`.
- All subprocess calls are argv lists; no `shell=True`; paths from templates resolved and required to
  sit inside the topic directory (reject `..`/absolute escapes and symlinks leaving it).
- Data DB opened `file:…?mode=ro` + `set_authorizer` allowing only reads; single statement; row cap.
- Paper/LLM text is untrusted data: stored as text, rendered in reports as escaped/fenced Markdown,
  never evaluated; model JSON parsed with `json.loads` + schema checks; unknown fields dropped.
- Secrets: env only; `doctor` shows present/absent, never values; HTTP errors are logged without headers.
- Budgets fail closed (exceeding any cap stops the run).

## Risks & limits (to state in README)

- Scripted mode proves the plumbing, not research quality; real discoveries depend on the LLM and on
  a well-designed evaluator.
- Same-container evaluator can be gamed by adversarial candidate code (mitigated, not eliminated).
- True **VRAM** measurement needs a GPU host + NVIDIA container toolkit; the shipped demo measures
  peak CPU memory. The template supports `gpus = "all"` for later GPU use (untested here: no GPU on Mac).
- Semantic Scholar unauthenticated rate limits are low; `SEMANTIC_SCHOLAR_API_KEY` is optional.
- LiDAR / historical topics require user-supplied data exported into the `data_query` DB.

## Verification

- `python3 -m unittest discover -s tests -v` passes (Docker tests run, not skipped, on this machine).
- `horizons doctor` output reviewed.
- Offline e2e: `horizons run examples/memory_reduction/topic.toml --llm scripted` → run reaches
  `success` with a replicated breakthrough and a written `report.md`; inspect report + KB rows.
- Negative checks: template with `../` path is rejected; candidate that tries network access fails;
  candidate exceeding memory/time is killed and classed `errored`; disallowed tool call is refused.
- Live smoke (only if an API key is present in env): short 2-iteration real run on the literature
  example; otherwise state it was not run.

## Steps

1. Create `pyproject.toml`, `README.md` skeleton, `.gitignore`, and the `horizons/` package skeleton with `__main__.py` and `cli.py` subcommand stubs.
2. Implement `horizons/template.py` (TOML load, strict validation, path containment) and `horizons/budget.py` (fail-closed guard).
3. Implement `horizons/llm.py` with Anthropic, OpenAI-compatible, and Scripted clients plus robust JSON extraction, after re-reading LLM-call patterns in the corpus.
4. Implement the knowledge base (`kb/store.py`, `kb/vectors.py`, `kb/recall.py` with RRF adapted from Hindsight, `kb/graph.py` with PRAXIST edge rules).
5. Implement `horizons/stats.py` (permutation test, bootstrap CI, Holm correction).
6. Implement `tools/registry.py`, `tools/literature.py` (Semantic Scholar + arXiv adapted from paper-search-mcp, with cache and backoff), `tools/sandbox.py` (hardened Docker runner adapted from OpenSandbox flags, plus gated unsafe-local runner), and `tools/data_query.py` (read-only SQLite).
7. Write prompt templates in `horizons/prompts/` for deconstruct, hypothesize, implement-candidate, classify-evidence, and reflect.
8. Implement agents: `deconstructor.py`, `hypothesizer.py`, `executor.py`, `evaluator.py`, `reflector.py`, following the AutoResearchClaw, SkyDiscover, and FAROS patterns listed above.
9. Implement `horizons/controller.py` (state machine, REFINE/PIVOT/STOP, replication, checkpoint/resume) and `horizons/report.py`.
10. Complete `cli.py`: `init`, `doctor`, `run` (incl. `--llm`, `--resume`, `--i-accept-unsafe-local`), `status`, `report`, `memory search`.
11. Build `examples/memory_reduction/` (baseline, user evaluator, topic, scripted LLM responses) and `examples/literature_question/`.
12. Write `tests/` unittest suite covering template validation, KB recall, stats, sandbox hardening (network denied, timeout, memory kill), evaluator outcome classes, and a scripted end-to-end controller run.
13. Pull and pin the sandbox Docker image, run the full test suite and the offline end-to-end demo, fix defects until green.
14. Run negative security checks (path escape, network attempt, disallowed tool) and the live-LLM smoke test if a key is present.
15. Finish README (quickstart, template reference, safety model, limits) and review the full diff against this plan.
