# Notes for coding agents

This repository is meant to be cloned and used as a local working directory;
personalize it (queries, criteria, prompts, nodes, paths) as needed.

- Python 3.12. Install: `pip install -r requirements.txt`; teaching examples
  only: `pip install -r examples/chapter/requirements-colab.txt` (Linux/Colab)
  or `examples/chapter/requirements.txt` (hashed lock, Windows x86-64).
- Tests: `pytest examples/chapter -q` (offline; network is blocked by a
  pytest-socket plugin). `--run-slow-chapter` adds slow tests (bertopic);
  `--run-live-chapter` allows live calls and spends money.
- Never print, log or commit API keys; read them from environment variables.
- Exercises (`exercise.py`) are intentionally incomplete; `solution.py` files
  and fixture hashes are verified by tests, keep them in sync when editing.
- Fixtures under `examples/chapter/data/` are synthetic; provenance in
  `LICENSE_NOTES.md`. Do not add copyrighted full texts.
- Corporate TLS interception: set `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`
  externally; never commit `.pem` files.

## Test invocation
Run the two suites in separate pytest sessions (`pytest tests`, then
`pytest examples/chapter`); the examples' socket-blocking plugin otherwise
leaks into tests that talk to a local proxy.

## Optional dependency not shipped
`slr_pipeline.nodes.dedup` can use an external `unified_screening_pipeline_v1`
module for semantic dedup when installed; without it, dedup falls back to the
deterministic DOI/title path (`DedupConfig.mode="auto"`).
