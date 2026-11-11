# Run log (home laptop)

Every entry records: date, machine, GPU, config, hours, result.
Entries are appended by scripts; add manual entries below them in the same style.
Report only what is written here.

## 2026-11-05 20:53 | home | Day 1 environment check
- Machine label: home (what actually ran: this entry's machine)
- OS: Windows 10 (10.0.26300)
- Python: 3.11.6 (venv: True)
- RAM: 31.63 GB, free disk: 564.31 GB
- nvidia-smi: NVIDIA GeForce RTX 4060 Laptop GPU, 610.47, 8188 MiB
- PyTorch: 2.15.0+cu132 (CUDA build 13.2), cuda_available=True
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB total, free at start (usable budget): 6.92 GB, compute capability 8.9
- VRAM probe: allocated 8.0 GB, spill to system memory suspected: True
- Native bf16: True -> plan: bf16 (no scaler)
- matmul 4096 float32: 6.14 TFLOPS
- matmul 4096 float16: 24.61 TFLOPS
- matmul 4096 bfloat16: 25.32 TFLOPS
- AMP smoke float16 (scaler=True): PASS
- AMP smoke bfloat16 (scaler=False): PASS
- Network: pypi=ok, pytorch_wheels=ok, github=ok
- UTF-8 Japanese round trip: PASS
- Overall: PASS (details: runs/env_check_home_20261105_2053.json)

## 2026-11-06 19:10 | home | Day 4 question generation

- Machine label: home (CPU only; no GPU used)
- Generator: data/source/make_questions.py (team), run by scripts/day4_generate_corpus.py (day4-v1)
- Settings: max 2000 per pattern, seed 42; time 2.7 s
- Result: 169,727 questions, 60 types, 300 patterns with questions, avg 23.51 chars
- Identical to the team's corpus: yes
- Output folder: runs/day4_generation/20261106_191020
- Checks: PASS

## 2026-11-06 19:52 | home | Day 3 question generation

- Machine label: home (CPU only; no GPU used)
- Generator: data/source/make_questions.py (team), run by scripts/day3_generate_questions.py (day3-v1)
- Settings: max 2000 per pattern, seed 42; generation time 2.7 s
- Result: 169,727 questions, 60 types, 300 patterns with questions, avg 23.51 chars
- questions.csv SHA-256: d9ff2a6a6708f4af81ba3cf8c7db17d88d4112c4e823906f4853ebd1d6d62586
- Output: runs/day3_generation/20261106_195246/ (zip for the other PC: questions_corpus_home_20261106_195246.zip)
- Checks: PASS

## 2026-11-08 17:15 | home | Day 5 corpus split

- Machine label: home (CPU only; no GPU used)
- Script: scripts/day5_build_corpus.py (day5-v1), seed 42
- Held-out patterns: validation 21, test 21 (split by pattern, not by line)
- train: 148,860 lines (147,317 unique), 3,482,192 chars, token bounds 1,646,022 to 10,126,458
- val: 4,517 lines (4,517 unique), 95,987 chars, token bounds 42,388 to 279,399
- test: 4,553 lines (4,553 unique), 109,720 chars, token bounds 51,015 to 320,290
- Manifest: data/splits/split_manifest.json
- Result: PASS

## 2026-11-08 17:49 | home | Day 6 tokenizer

- Machine label: home (CPU only; no GPU used)
- Script: scripts/day6_train_bpe.py (day6-v1); Python 3.11.6
- Training: 743 unique pieces, 1,314 merges, 0.9 s, largest vocabulary 1,674
- Chosen vocabulary: 1,674; validation 47,123 tokens (0.491 per char); key terms 1-3 tokens 232/232; longest question 25 tokens
- tokenizer.json SHA-256: 4468c3ab991bfbdd1ee6673f0b945a364b8419062be5c50bb5755a48c6e93570
- Checks: reload PASS, round trip PASS

## 2026-11-08 17:55 | home | Day 6 tokenizer

- Machine label: home (CPU only; no GPU used)
- Script: scripts/day6_train_bpe.py (day6-v1); Python 3.11.6
- Training: 743 unique pieces, 1,314 merges, 0.9 s, largest vocabulary 1,674
- Chosen vocabulary: 1,674; validation 47,123 tokens (0.491 per char); key terms 1-3 tokens 232/232; longest question 25 tokens
- tokenizer.json SHA-256: 53b03761cd1b614972e1b9bfd63d0a66287d6fd069cd0a9fd91419ec249ca94b
- Checks: reload PASS, round trip PASS

## 2026-11-11 18:09 | home | Day 7 encode corpus

- Machine label: home (CPU only; no GPU used)
- Script: scripts/day7_encode_corpus.py (day7-v1); Python 3.11.6
- Tokenizer SHA-256: 53b03761cd1b614972e1b9bfd63d0a66287d6fd069cd0a9fd91419ec249ca94b
- train: 148,860 questions, 2,028,601 tokens (avg 13.6, max 28), SHA-256 2c95990940ea0118ea67c524477e30e02afd693d9e4472b4b758a2c4140e8fdd
- val: 4,517 questions, 56,157 tokens (avg 12.4, max 22), SHA-256 e39257300d284db7f8b0f373e10ebbb7e2e32aa1346a7d00be52e9afbbeae1bf
- test: 4,553 questions, 63,877 tokens (avg 14.0, max 25), SHA-256 ec593af0f94a889011546bd94deea12a4ff8689131587a6848432d97ce4f534b
- Checks: round trip PASS on every question and tricky text; disk read-back PASS
