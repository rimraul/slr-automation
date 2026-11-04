# slr-automation

Code, notebooks and exercises accompanying the chapter on automating
systematic literature reviews (searching, deduplication, LLM-assisted
screening, full-text retrieval and eligibility, quality appraisal, data
extraction, synthesis and topic modelling, PRISMA-style reporting).

## Contents

| Path | What |
|---|---|
| `examples/chapter/notebooks/` | Colab notebooks s04-s12 (open in Colab; they clone this repo and install pinned dependencies) |
| `examples/chapter/` | exercises, reference solutions, synthetic fixtures, offline tests |
| `slr_pipeline/` | LangGraph pipeline: search backends, screening, full-text, extraction, synthesis, reporting, cost caps, replay |
| `tests/` | offline test suite |

Full-text eligibility screening lives in a companion repository,
[FULL_TEXT_SCREENING](https://github.com/rimraul/FULL_TEXT_SCREENING)
(imported as `ft_screening`).

## Use

Colab: open a notebook from `examples/chapter/notebooks/`.

Local (clone as a working directory, edit freely, or let a coding agent do it;
see `AGENTS.md`):

```bash
git clone https://github.com/rimraul/slr-automation.git
cd slr-automation
pip install -r requirements.txt
pip install -e ".[topics,pdf,viz,dev]"   # optional extras
pytest examples/chapter -q   # run suites separately:
pytest tests -q
```

Nothing calls a paid API by default. Live calls need your own keys in
environment variables (Colab: Secrets). Never commit keys.

## Licences

Code MIT (`LICENSE`); fixtures and exercise text CC BY 4.0 (`LICENSE-CONTENT`).
Third-party notes: PyMuPDF (via FULL_TEXT_SCREENING) is AGPL-3.0.
