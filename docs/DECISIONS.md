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
  All training text is made by the team's own generator from the team's own files; see `docs/DATA_CARD.md`.

## D6. Tokenizer (Day 6)

- Own byte-level BPE with a script-class pre-tokenizer (digits and punctuation always single pieces). No tokenizer library.
- Fixed IDs: <pad></pad>=0, <bos></bos>=1, <eos></eos>=2, <unk></unk>=3, <unused_0..99>=4-103, bytes=104-359, merges from 360.
- Vocabulary 1,674 instead of the planned ~16,000: the training text has only 743 unique pieces, so training stops
  after 1,314 merges. Chosen by a written rule (all achievable key terms 1-3 tokens, validation within 5% of the
  largest size); see docs/DAY6_TOKENIZER_REPORT.md.
- Evidence for later: the longest question is 25 tokens (27 with <bos></bos>/<eos></eos>). The context length is decided later.
- Trained only on train.txt; validation used only for measuring.
