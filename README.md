# New Horizons

A local, topic-agnostic **active discovery engine**. You describe a research target in one
`topic.toml`; the engine loops

**Deconstruct → Hypothesize → Execute → Evaluate → Remember → repeat**

until your validation metric is met *and replicated*, the budget runs out, or progress stalls.

It searches the literature, proposes falsifiable hypotheses, writes and runs experiments in a locked-down
Docker sandbox, judges results with **your** evaluator (never its own opinion), and keeps a SQLite memory
of papers, hypotheses, experiments, findings and lessons that later runs recall.

Python standard library only — nothing to `pip install` besides the package itself.

## Quickstart

Requirements: Python ≥ 3.11 and Docker (Docker Desktop on macOS).

```bash
cd new-horizons
python3 -m horizons doctor          # checks Python, SQLite FTS5, Docker, keys, network
# or: uv run horizons doctor

# Offline demo: no API key, real sandbox, full loop, writes a report
python3 -m horizons run examples/memory_reduction/topic.toml --llm scripted
```

The demo refutes one idea, recovers from a buggy attempt, replicates a 99.8% memory reduction three
times and stops with `success`. The report goes to `.horizons/runs/<run-id>/report.md`.
**The scripted LLM only proves the machinery works; it does no real research.**

Real research loop with your **Claude or ChatGPT subscription** (no API key):

```bash
python3 -m horizons login claude     # Claude Pro/Max: opens claude.ai, paste the code it shows back
python3 -m horizons login chatgpt    # ChatGPT Plus/Pro: opens auth.openai.com, returns automatically
python3 -m horizons run examples/memory_reduction/topic.toml
```

Runs pick up a saved login automatically (Claude first, then ChatGPT, then API keys). Choose one
explicitly with `--llm claude` or `--llm chatgpt`, and a model with `--model`. The defaults are
`claude-sonnet-5` and `gpt-6-sol`; use `--model gpt-6-luna` for cheaper ChatGPT runs. On a remote or
headless machine, add `--no-browser` and paste the redirect URL (ChatGPT) or code (Claude) at the prompt.
Subscription usage counts against your plan's limits; when a limit is hit, the run stops and can be resumed
with `--resume`.

Or with API keys:

```bash
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY=...
python3 -m horizons run examples/literature_question/topic.toml
```

Local models (Ollama, vLLM, anything OpenAI-compatible):

```bash
export HORIZONS_OPENAI_BASE_URL=http://localhost:11434/v1
python3 -m horizons run my_topic/topic.toml --llm openai --model llama3.1
```

## Commands

| Command | What it does |
|---|---|
| `horizons init DIR` | Scaffold `topic.toml`, `baseline.py`, `evaluate.py` |
| `horizons doctor` | Check every prerequisite; prints a fix for each problem |
| `horizons login claude\|chatgpt [--no-browser]` | Sign in with a subscription instead of an API key |
| `horizons logout claude\|chatgpt\|all` | Forget a saved login |
| `horizons run TOPIC [--llm claude\|chatgpt\|anthropic\|openai\|scripted] [--model M] [--offline]` | Run the loop |
| `horizons run TOPIC --resume RUN_ID` | Continue a crashed or interrupted run from its last checkpoint |
| `horizons status [RUN_ID]` | List runs, or show one run's hypotheses and recent events |
| `horizons report RUN_ID` | Rewrite the Markdown report |
| `horizons memory search "query" [--topic T] [--kind lesson]` | Search the knowledge base |

Global: `--workspace DIR` (default `./.horizons`, or `$HORIZONS_WORKSPACE`). Exit codes for `run`:
`0` success · `1` failed · `2` bad template/setup · `3` stopped (budget or stall).

## Writing a topic

A topic is a folder with `topic.toml` plus the files it names. All paths are relative to that folder
and may not leave it.

```toml
[topic]
name = "memory-reduction"
goal = "Reduce peak memory of the baseline routine by 50% without changing results"
context = "Optional background, constraints, known baselines."

[tools]
allowed = ["literature", "python_sandbox"]   # any of: literature, python_sandbox, data_query

[tools.literature]
sources = ["semantic_scholar", "arxiv"]
max_papers = 20
# fixture = "papers.json"          # optional local papers (used offline / merged online)

[tools.python_sandbox]
baseline = "baseline.py"           # the program the engine may rewrite (as candidate.py)
evaluator = "evaluate.py"          # YOUR measurement; mounted read-only, never edited
image = "python:3.13-slim@sha256:…"  # defaults to a digest-pinned python:3.13-slim
timeout_s = 120
memory = "1g"
cpus = 1.0
# gpus = "all"                     # needs an NVIDIA host + container toolkit (untested)
# sandbox = "unsafe-local"         # NO isolation; also needs --i-accept-unsafe-local

[tools.data_query]                 # only if "data_query" is allowed
database = "data.sqlite"           # opened read-only
evaluator = "evaluate_data.py"     # reads $HORIZONS_WORK/rows.json, prints metrics
max_rows = 10000

[validation]
metric = "peak_memory_mb"
direction = "minimize"             # minimize | maximize
target = { relative_to_baseline = -0.5 }   # or { absolute = 0.05 }
guards = [{ metric = "correct", op = "==", value = 1 }]
replications = 3                   # a win must pass on every rerun
# kind = "significance"            # also require a permutation-test win vs baseline,
# alpha = 0.05                     # Holm-corrected over every hypothesis tested in the run

[budget]
max_iterations = 12
max_llm_calls = 200
max_sandbox_runs = 60
max_wall_clock_min = 60
patience = 4                       # iterations without improvement before stopping
max_pivots = 3                     # pivots without improvement before re-mapping the literature
max_refines = 2                    # retries of a crashed experiment
hypotheses_per_round = 3

[llm]                              # all optional
# provider = "claude"              # claude | chatgpt (subscription login) | anthropic | openai (API key)
#                                  # | scripted. Default: saved login first, then env keys
# model = "claude-sonnet-5"
# base_url = "https://…"           # https only (http allowed for localhost)
# scripted_file = "scripted_llm.json"
```

Unknown keys are errors, so a typo can't silently switch off a guard.

### The evaluator contract

Your evaluator is a Python script run as `python -I evaluate.py` inside the sandbox. It must:

1. Read and **remove** the per-run marker first: `MARKER = os.environ.pop("HORIZONS_RESULT_MARKER")`.
2. Load the agent's program from `os.environ["HORIZONS_CANDIDATE"]` (e.g. `runpy.run_path`).
3. Measure it and print one line: `MARKER + json.dumps({"metric": value, ...})`.

Only the last marked line counts. Every value must be a finite number (booleans count as 0/1).
See `examples/memory_reduction/evaluate.py`.

## How a run works

1. **Baseline:** checks the sandbox, then runs your unmodified baseline `replications` times.
2. **Deconstruct:** the model writes search queries → Semantic Scholar + arXiv → a *boundary map*
   (known approaches, gaps, promising directions). Citations to papers that were not retrieved are
   removed and logged as a lesson.
3. **Hypothesize:** several falsifiable hypotheses (statement, rationale, test tool, falsification
   criterion, alternatives), informed by recalled lessons. Near-duplicates of already-tested ideas are
   dropped. Ideas the allowed tools can't test are stored as `needs_resources` with what's missing.
   A judge prompt ranks the rest; one is tested.
4. **Execute:** `python_sandbox` — the model rewrites the current best program and your evaluator
   measures it. `data_query` — the model writes one read-only `SELECT`; your data evaluator scores the
   rows. `literature` — the model labels papers as supporting or challenging; each label must quote the
   abstract word for word, and the quote is checked in code.
5. **Evaluate (code only):** `errored` (crash, timeout, broken guard) → **REFINE** the experiment;
   `refuted` (missed target) or `inconclusive` (didn't replicate / not significant) → **PIVOT**;
   target met on every replication → **STOP: success**.
6. **Remember:** lessons (system / experiment / literature / analysis) go into SQLite with full-text
   and hashed-vector indexes and are recalled via Reciprocal Rank Fusion in later iterations and runs.

Every step is checkpointed, so `--resume` continues after a crash, Ctrl-C, or an API outage.

## Safety model

- **The agent never grades itself.** Success is decided by code comparing your evaluator's numbers to
  the template. The evaluator and topic folder are mounted read-only.
- **Sandbox:** `--network none`, read-only root filesystem, `--cap-drop ALL`, `no-new-privileges`,
  non-root user, pids/memory/CPU/file limits, a hard timeout followed by `docker kill`. Only a per-attempt
  scratch folder is writable; the engine's own workspace is hidden from the container.
- **Fail closed:** no Docker means the run stops before any model call. Exceeding any budget stops the run.
- **Tool allowlist:** only tools listed in `[tools] allowed` exist for a run.
- **Untrusted text:** paper abstracts and model output are treated as data. They are fenced in prompts,
  escaped in reports, parsed with `json.loads` plus schema checks, and can't change permissions.
- **Data access:** SQLite opened `mode=ro`, an authorizer allows reads only, one statement at a time,
  capped rows and query work.
- **Secrets:** API keys come only from environment variables. Subscription logins are stored in
  `~/.horizons/auth.json` (override with `HORIZONS_AUTH_FILE`), outside every project, readable only by
  your user (0600), and replaced atomically. Logins use OAuth with PKCE and a random `state` that must
  match. Tokens refresh automatically and are never written to the knowledge base, logs or reports.
  The sandbox cannot see the login file, and `doctor` shows only whether each login or key is present.

## Limits (read before trusting a result)

- **Research quality depends on the model and on your evaluator.** A weak evaluator gets gamed by
  hard-working optimisation. Guards (like `correct == 1`) are essential.
- **Candidate code runs in the same container process as your evaluator.** Removing the marker from
  the environment and using a random marker per run make forging results harder, but a deliberately
  adversarial program (e.g. one that inspects the caller's stack) could still fake output. Review the
  best program in the report before relying on it.
- **VRAM:** measuring GPU memory needs a GPU host with the NVIDIA container toolkit (`gpus = "all"`).
  This was not tested here; the demo measures peak CPU heap via `tracemalloc`.
- **Literature** is judged from abstracts only. Semantic Scholar's unauthenticated rate limits are low:
  set `SEMANTIC_SCHOLAR_API_KEY` for heavier use. arXiv requests are spaced 3 seconds apart, as arXiv asks.
- **Your own data** (LiDAR, historical records…) has to be exported into a SQLite file for `data_query`.
- The embeddings are local feature hashing: good for duplicate detection and recall, not deep semantics.
- **Subscription logins** reuse the public sign-in apps of Claude Code and the Codex CLI (as ezcoder
  does), so requests identify as those tools. The providers could change or restrict this at any time;
  API keys are the officially supported route. Check your plan's terms if that matters to you.

## Where the ideas come from

Patterns were adapted (not copied) from open-source research agents: AutoResearchClaw (stage
pipeline, REFINE/PIVOT decisions, lesson categories, synthesis + judge), SkyDiscover (generate/evaluate
loop with lineage and failed-attempt feedback, user-owned evaluator), FAROS (falsification criteria,
alternatives, readiness), PRAXIST (finding graph with whitelisted edges and caps, fail-closed budget),
Hindsight (Reciprocal Rank Fusion), paper-search-mcp (Semantic Scholar/arXiv clients with backoff), and
OpenSandbox (container hardening flags).

## Development

```bash
python3 -m unittest discover -s tests -v    # the Docker tests are skipped only when Docker is unavailable
```

Layout: `horizons/` (engine: `controller.py`, `agents/`, `tools/`, `kb/`, `prompts/`), `examples/`,
`tests/`.
