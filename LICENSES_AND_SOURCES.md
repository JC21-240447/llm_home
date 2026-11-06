# Licenses and sources

Rule: the training text is created by the team only. No outside text is used for training:
no Wikipedia, no books, no web pages, no laws. If a new source is ever proposed, it is added
to this file first and used only after its license is checked.

## Training data sources

| ID | Source | Made by | License status | Notes |
| --- | --- | --- | --- | --- |
| D1 | Glossary 用語集 (G001 to G551) | the team, original wording | team's own work | `data/source/用語集.csv` |
| D2 | Synonym dictionary 同義語辞書 (S0001 to S1705) | the team, original wording | team's own work | `data/source/同義語辞書.csv` |
| D3 | Question patterns (`src_1.txt`, `src_2.txt`, `patterns.csv`) | the team | team's own work | 60 question types |
| D4 | Sample employee table, 50 rows | the team | team's own work | names, IDs and value ranges for the slots |
| D5 | Generator `make_questions.py` and its output `questions_corpus.zip` | the team | team's own work | fills the patterns from D1 to D4 |

Training, validation and test files are made from D5 by `scripts/day5_build_corpus.py`.
See `docs/DATA_CARD.md` for sizes and the split.

## Tools and libraries

| Name | Version | License | Use |
| --- | --- | --- | --- |
| Python | (see run_log.md) | PSF | language |
| PyTorch | (see requirements_lock) | BSD-style | tensors, GPU training |
| NumPy | (see requirements_lock) | BSD | arrays |

## Credit lines (final text is written on Day 17)

- Glossary, synonym dictionary, question patterns, sample table and generator: the project team.
- Model, tokenizer and training loop: written by the project team.
