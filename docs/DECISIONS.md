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
