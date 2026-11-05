# Model A (home): a small Japanese company-terms language model

A decoder-only Transformer, with its own tokenizer and training loop, trained from scratch on Japanese company-style text.
It continues text; it is not an instruction-following assistant.

> Status: Day 1 (environment, home laptop).

## Machine this repository is for

**Home laptop, RTX 4060 8 GB.** Run logs in this repository describe only this machine.

## Quick start (Windows, PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Machine home -TorchIndexUrl <index url from pytorch.org selector>
```

## Layout

| Folder | Purpose |
| --- | --- |
| `data/` | glossary, synonyms, patterns, corpus files (large files are git-ignored) |
| `src/` | tokenizer, model, training code |
| `configs/` | model and training configs |
| `scripts/` | one script per day (`dayN_*.py`) |
| `runs/` | logs, evidence JSON, checkpoints (weights are git-ignored) |
| `docs/` | decisions, data card, results, weekly reports |
| `tests/` | unit tests |

## Data and licenses

See `LICENSES_AND_SOURCES.md`.

## Honest reporting

`run_log.md` records the date, machine, GPU, config and result of every run. Reports must match it.
