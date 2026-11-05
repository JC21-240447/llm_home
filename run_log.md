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
