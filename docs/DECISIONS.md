# Decisions (home repository)

## D1. Language and libraries

- **Language:** Python 3.12 (3.11 or newer is acceptable).
- **Installed on the home laptop:** (fill in after setup: Python, PyTorch, driver; copy from run_log.md).
- **Core library:** PyTorch with CUDA. NumPy for arrays.
- **Written by us:** tokenizer, model, training loop, evaluation.

## D2. Repositories

- This repository is for the home laptop (RTX 4060, 8 GB) only. It is private.
- The school PC has its own repository. Each run log describes only its own machine.
- Code that differs per machine carries a SCHOOL-PC SWAP note (EN and JA) at the place to change.

## D3. Precision

- Home (RTX 4060): bf16.
- School (RTX 2060): fp16 with loss scaling (no native bf16).

## D4. Training corpus (Day 5)

- The corpus is short Japanese questions about employee data only, made by the team's own generator
  from the team's glossary, synonym dictionary, patterns and a 50-row sample table.
- Books and laws (old Day 3) are dropped. General text may be reconsidered only if evaluation shows a need.
- Only the question text is used for training; labels are kept in separate files.

## D5. Split (Day 5)

- Split by pattern, not by line: 21 validation and 21 test patterns, one per question type, from different types.
- A type donates a pattern only if it has 100+ questions and 3+ patterns; the donated pattern holds at most 40% of its type.
- Held-out patterns are capped at 500 questions; lines over the cap are dropped, not moved to train.
- Small types are repeated in train only (up to about 200 lines, at most 5 copies). Seed 42.
- P171 has no questions: all its sentences are identical to P026 sentences. Open question for the team.
