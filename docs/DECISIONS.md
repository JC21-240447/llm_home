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

## D7. Token files (Day 7)

- Each split is one uint16 stream (<bos></bos> question <eos></eos> ...), little-endian, in data/tokens/, with offsets and a meta file
  holding the vocabulary size and the tokenizer SHA-256 (training refuses a mismatched file).
- Real size: train 2,028,601 tokens; average question 13.6 tokens, longest 28 (with <bos></bos>/<eos></eos>).
- Round trip verified on every question and on the files read back from disk.

## D8. Model architecture and trial sizes (Day 8)

- Decoder-only Transformer, own code (src/model.py): pre-norm RMSNorm, rotary positions (RoPE), causal attention
  (manual + fast path, proven equal by a test), GELU MLP 4x, no biases, weight tying, scaled residual init.
- Block size 256 tokens (longest question 28 tokens); a setting, not part of the weights.
- Trial sizes changed from ~25M/60M/100M to ~1.0M/5.2M/14.8M (tiny/small/medium): vocabulary is 1,674 (not 16k)
  and train is 2,028,601 tokens, so the planned sizes would have 12+ parameters per training token.
- Day 11 still chooses the size by validation loss.

## D9. Trainer (Day 9)

- Training examples: random 256-token windows from the train stream (x = window, y = window shifted by one).
  Validation: fixed non-overlapping windows covering the whole validation stream.
- AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), linear warm-up then cosine decay to 10% of the peak,
  gradient clipping at 1.0, bf16 autocast on the RTX 4060 (fp16 + gradient scaler on GPUs without bf16).
- Checkpoints hold the model, optimizer, scaler, step and all random states; a stopped and resumed run gives the
  same losses as an uninterrupted run (test on CPU: identical; smoke run on GPU: see docs/DAY9_SMOKE_REPORT.md).
- Smoke run settings: tiny model, batch 32 x 256, 600 steps, warm-up 60, peak lr 1e-3.
- Day 9 home results: stop/resume identical on GPU (max difference 0.00e+00). Tiny model, batch 32 x 256:
  validation loss 7.446 -> best 2.673 (perplexity 14.5) at step 200, then rises (2.939 at step 600;
  3.277 at step 3000 in smoke_c) while training loss falls to ~1.0. Overfitting to the training sentence
  patterns starts before one full pass; best.pt keeps the step-200 model. Day 11 must evaluate more often
  early and choose settings by validation loss.

## D10. Generation (Day 10)

- Repetition limits measured from the training questions: a token at most 4 times in a row (社員番号100003),
  a 3-token sequence at most twice (1991年から1997年). A generic "no repeated 3-gram" rule would block real questions.
- Settings compared: greedy, greedy+limits, sample (T 0.8, top-k 50, top-p 0.95), sample-hot (T 1.5).
  Default for evaluation: greedy+limits (deterministic) and sample (varied).
- Sampling seed is derived from (seed, prompt): reproducible, and different prompts get independent draws.
- Findings: last.pt (step 600) uses company terms better than best.pt (step 200); T 1.5 breaks text;
  out-of-scope beginnings are turned into company questions.
- Task 6 item A3 corrected to "5+ same tokens in a row" (4 in a row occurs in real questions).

## D11. Model size (Day 11)

- Trials: tiny / small / medium x dropout 0.0 / 0.1, same budget (800 steps, batch 32 x 256, lr 1e-3, seed 42),
  validation every 25 steps; best checkpoints checked on the 12 Day 10 prompts.
- Rule (fixed before the trials): pass task 6 checks (finished >= 90 %, repetitive <= 5 %); within 0.02 of the
  lowest validation loss, choose the fewest parameters.
- Result: chosen <run></run> (<size></size>, dropout <d></d>), best validation loss <x.xxxx> at step <n></n>.
  Lowest overall: <run></run> <x.xxxx>. Observations: <e.g. larger models reached their best earlier and rose more;
  dropout reduced the rise after best by ...>.
- Day 12 trains the chosen configuration (configs/day11_decision.json).
