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
