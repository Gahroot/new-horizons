# AGENTS.md

New Horizons is a Python ≥ 3.11 package that uses only the standard library (no third-party runtime dependencies).
See `README.md` for architecture, the topic template and the safety model.

## Commands

```bash
uv sync                                        # or: python3 -m pip install -e .
python3 -m unittest discover -s tests -v       # full suite; Docker tests skip only if Docker is unavailable
python3 -m compileall -q horizons tests        # syntax check (no linter is configured)
python3 -m horizons doctor                     # check local prerequisites
python3 -m horizons run examples/memory_reduction/topic.toml --llm scripted   # offline end-to-end demo
```

## Rules

- CI lives in `.github/workflows/ci.yml` (Python 3.11 and 3.14, with Docker) and must stay green.
- Never commit with `--no-verify`.
- Do not add runtime dependencies. Keep the sandbox hardening flags, the tool allowlist and the read-only evaluator in place.
- Never write API keys or OAuth tokens to logs, the knowledge base, reports or tests. Tests use a temporary `HORIZONS_AUTH_FILE`.
