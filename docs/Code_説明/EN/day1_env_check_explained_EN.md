# day1_env_check.py — Simple Line-by-Line Explanation (English)

## 0. The big picture (read this first)

Think of this script as a **health checkup for your computer** before you start training an AI model.

A doctor checks: height, weight, heartbeat, blood pressure, and then asks you to run on a treadmill for a minute..
This script does the same thing:

| Doctor's checkup | What this script does |
|---|---|
| Basic body info | Records Python version, OS, CPU, RAM, free disk |
| Can you read/write properly? | Writes Japanese text to a file and reads it back (UTF-8 test) |
| Can you reach the outside world? | Tries to open pypi.org, pytorch.org, github.com |
| What muscles do you have? | Asks PyTorch for the GPU name, VRAM, compute capability |
| Treadmill speed test | Matrix multiplication benchmark (fp32 / fp16 / bf16) |
| How much can you carry? | Allocates GPU memory until it is full (VRAM probe) |
| Mini workout | A tiny training run using mixed precision (AMP smoke test) |
| Write the medical report | Saves a JSON file and adds a note to `run_log.md` |

The script is run like this:

```
.venv\Scripts\python.exe scripts\day1_env_check.py --machine school
```

`--machine school` or `--machine home` is only a **label** so that your log says which computer produced the result.

---

## 1. Words you need to know (tiny dictionary)

| Word | Simple meaning |
|---|---|
| **venv** | A private box of Python packages just for this project, so it does not mess with other projects. |
| **GPU** | The graphics card. It does huge amounts of simple math at the same time. AI training needs it. |
| **VRAM** | The GPU's own memory (like the GPU's desk). If your model does not fit on the desk, training crashes. |
| **CUDA** | NVIDIA's system that lets programs use the GPU for math. |
| **Compute capability** | A version number of the GPU's abilities. 7.5 = RTX 2060 (Turing). 8.9 = RTX 4060 (Ada). 8.0 or higher means "Ampere or newer". |
| **fp32 / fp16 / bf16** | Three ways to store a decimal number. fp32 = 32 bits, precise but slow and big. fp16 and bf16 = 16 bits, half the size and faster. |
| **fp16 vs bf16** | fp16 is more precise but can only store a small range of numbers (tiny numbers become 0). bf16 has a huge range (like fp32) but is less precise. bf16 needs a newer GPU (8.0+). |
| **matmul** | Matrix multiplication. The core math inside every neural network. |
| **TFLOPS** | Tera floating-point operations per second = "how many trillions of calculations per second". Higher is faster. |
| **AMP / autocast** | Automatic Mixed Precision. PyTorch automatically uses 16-bit where it is safe and 32-bit where it is needed. |
| **GradScaler** | Helper for fp16. Gradients (the learning signals) can be so tiny that fp16 rounds them to 0. GradScaler multiplies the loss by a big number first, then divides it back later. |
| **WDDM** | Windows' graphics driver mode. It may let the GPU "borrow" normal RAM when VRAM is full (slow). |
| **JSON** | A plain-text file format for saving structured data (keys and values). |
| **smoke test** | A very small test that only asks "does it run without catching fire?" |

---

## 2. The docstring (top of the file)

```python
r"""Day 1 - environment check for Phase 1 (Model A).
...
"""
```

- `r"""` ... `"""` — A **triple-quoted string** at the very top of a file is called a *docstring*. It is just documentation. Python does not run it.
- The `r` in front means **raw string**: backslashes are kept as normal characters. This matters because the usage example contains `.venv\Scripts\...` and Windows paths use backslashes. Without `r`, Python might think `\S` is a special code.
- Inside, there is an English part and a Japanese part saying the same things: what the script does (8 steps) and how to run it.

---

## 3. Imports

```python
from __future__ import annotations
```
- Makes type hints (like `str | None`) be treated as plain text instead of being evaluated immediately. This makes modern type-hint style work on more Python versions.

```python
import argparse
import importlib.metadata as md
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
```

| Line | What it gives you |
|---|---|
| `import argparse` | Reads command-line options like `--machine school`. |
| `import importlib.metadata as md` | Lets you ask "which version of package X is installed?". Shortened to `md`. |
| `import json` | Converts Python dictionaries to JSON text. |
| `import math` | Math helpers. Here only `math.isfinite` is used (checks a number is not NaN/infinity). |
| `import os` | Operating system helpers. Here only `os.cpu_count()`. |
| `import platform` | Info about Python and the OS. |
| `import shutil` | File helpers. Here `shutil.disk_usage` (free disk space). |
| `import subprocess` | Runs other programs (like `nvidia-smi`). |
| `import sys` | Python's own info (version, console streams, paths). |
| `import time` | Stopwatch (`time.time()`). |
| `from datetime import datetime` | Current date and time. |
| `from pathlib import Path` | A clean way to work with file paths. |
| `from typing import Any` | The type "anything" for type hints. |
| `from urllib.request import Request, urlopen` | Built-in tools to open web pages (no extra install needed). |

---

## 4. Constants (settings written once at the top)

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
```
- `__file__` = path of this script (`scripts/day1_env_check.py`).
- `.resolve()` = turn it into a full absolute path.
- `.parent` = the folder containing it (`scripts/`).
- `.parent` again = one more level up (the **repo root**).
- `: Path` is a type annotation: "this variable is a Path".

```python
RUN_LOG: Path = REPO_ROOT / "run_log.md"
RUNS_DIR: Path = REPO_ROOT / "runs"
```
- With `Path`, the `/` operator **joins** path pieces. So these are `<repo>/run_log.md` and `<repo>/runs`.

```python
EXPECTED_GPU_TAG: dict[str, str] = {"school": "2060", "home": "4060"}
```
- A dictionary: machine label → the number we expect to see in the GPU's name. Used later to warn you if you typed the wrong `--machine`.

```python
NETWORK_TARGETS: dict[str, str] = {
    "pypi": "https://pypi.org/simple/pip/",
    "pytorch_wheels": "https://download.pytorch.org/whl/",
    "github": "https://github.com",
}
```
- The three websites we must be able to reach: where Python packages come from (PyPI), where PyTorch is downloaded, and GitHub. School networks sometimes block these.

```python
PACKAGES: list[str] = ["torch", "numpy", "psutil", "tqdm", "pytest"]
```
- The list of packages whose versions we want to record.

```python
JAPANESE_PROBE: str = "今日は晴れ。売上・給与・勤怠"
```
- A sample Japanese sentence ("It's sunny today. Sales, salary, attendance") used to test that Japanese text is not corrupted.

---

## 5. Helper functions

### 5.1 `setup_utf8_console`

```python
def setup_utf8_console() -> None:
    """Make printing Japanese safe on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
```
- `def ... -> None` — defines a function that returns nothing.
- `for stream in (sys.stdout, sys.stderr)` — do the same thing for the two output channels: normal output and error output.
- `stream.reconfigure(encoding="utf-8", errors="replace")` — tell the console "use UTF-8". `errors="replace"` means: if a character still cannot be shown, print `?` instead of crashing.
- `# type: ignore[attr-defined]` — a note for the type checker: "I know you can't see `reconfigure` on this type, ignore it."
- `except (AttributeError, ValueError): pass` — if the stream does not support `reconfigure`, quietly do nothing.

### 5.2 `run_cmd`

```python
def run_cmd(cmd: list[str], timeout: float = 15.0) -> str | None:
    """Run a command, return stripped stdout or None on any failure."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "").strip()
    return text if out.returncode == 0 and text else None
```
- Takes a command as a list (e.g. `["nvidia-smi", "--query-gpu=..."]`) and a time limit of 15 seconds.
- `-> str | None` — returns text **or** `None`.
- `subprocess.run(...)` — runs the command.
  - `capture_output=True` — catch what it prints instead of showing it.
  - `text=True` — give me text, not raw bytes.
  - `timeout=timeout` — kill it if too slow.
  - `check=False` — do not raise an error if the command fails; I will check myself.
- `except (OSError, subprocess.SubprocessError): return None` — if the program does not exist (`OSError`) or timed out (`SubprocessError`), return `None`.
- `text = (out.stdout or "").strip()` — take the output (or empty string if there is none) and remove spaces/newlines at both ends.
- Last line — return `text` only if the command succeeded (`returncode == 0`) **and** the text is not empty. Otherwise `None`.

### 5.3 `gb`

```python
def gb(num_bytes: float) -> float:
    return round(num_bytes / 1024**3, 2)
```
- Converts bytes to gigabytes. `1024**3` = 1,073,741,824. `round(..., 2)` keeps 2 decimal places.

---

## 6. Checks that need no GPU

### 6.1 `check_system`

```python
def check_system() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "in_venv": sys.prefix != sys.base_prefix,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine_arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "stdout_encoding": getattr(sys.stdout, "encoding", None),
        "disk_free_gb": gb(shutil.disk_usage(REPO_ROOT).free),
        "repo_root": str(REPO_ROOT),
    }
```
Builds a dictionary (name → value) one line at a time:
- `"python"` — Python version, like `3.11.9`.
- `"python_executable"` — the exact `python.exe` that is running. Shows whether it is the venv's one.
- `"in_venv"` — `sys.prefix != sys.base_prefix`. Inside a venv these two differ, so `True` means "yes, running inside the venv".
- `"os"` — an f-string (text with `{}` holes filled in) like `Windows 11 (10.0.22631)`.
- `"machine_arch"` — CPU type, e.g. `AMD64`.
- `"cpu_count"` — number of CPU threads.
- `"stdout_encoding"` — what character encoding the console uses. `getattr(..., None)` safely returns `None` if it does not exist.
- `"disk_free_gb"` — free disk space on the drive where the repo lives.
- `"repo_root"` — the repo path as text (`str(...)`, because JSON cannot store `Path` objects).

```python
    try:
        import psutil  # type: ignore

        info["ram_total_gb"] = gb(psutil.virtual_memory().total)
        info["ram_available_gb"] = gb(psutil.virtual_memory().available)
    except ImportError:
        info["ram_total_gb"] = None
    return info
```
- Tries to import `psutil` (a package that reads RAM). It is imported **inside** the function so the script does not crash if it is missing.
- `ram_total_gb` = all RAM. `ram_available_gb` = RAM free right now. (The second line is the same idea as the first, just `.available` instead of `.total`.)
- If `psutil` is not installed (`ImportError`), RAM is recorded as `None`.
- `return info` — hand the dictionary back.

### 6.2 `check_packages`

```python
def check_packages() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions
```
- Start with an empty dictionary.
- Loop over each package name in `PACKAGES`.
- `md.version(name)` — ask for the installed version, e.g. `"2.4.0+cu121"`.
- If the package is not installed, Python raises `PackageNotFoundError`, and we store `None` instead.

### 6.3 `check_utf8_roundtrip`

```python
def check_utf8_roundtrip() -> bool:
    """Write and re-read Japanese text explicitly as UTF-8."""
    tmp = RUNS_DIR / "_utf8_probe.txt"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_text(JAPANESE_PROBE, encoding="utf-8")
    ok = tmp.read_text(encoding="utf-8") == JAPANESE_PROBE
    tmp.unlink(missing_ok=True)
    return ok
```
A "round trip" = go out and come back, then check nothing changed.
- `tmp = RUNS_DIR / "_utf8_probe.txt"` — a temporary file path.
- `RUNS_DIR.mkdir(parents=True, exist_ok=True)` — make the `runs` folder. `parents=True` also creates missing parent folders; `exist_ok=True` means "no error if it already exists".
- `tmp.write_text(JAPANESE_PROBE, encoding="utf-8")` — save the Japanese sentence into the file as UTF-8.
- `ok = tmp.read_text(encoding="utf-8") == JAPANESE_PROBE` — read it back and compare. `True` if identical.
- `tmp.unlink(missing_ok=True)` — delete the temp file (no error if it is already gone).
- `return ok`.

### 6.4 `check_url`

```python
def check_url(url: str, timeout: float = 8.0) -> dict[str, Any]:
    start = time.time()
    try:
        req = Request(url, headers={"User-Agent": "model-a-env-check"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read(1)
            return {"ok": True, "status": resp.status, "seconds": round(time.time() - start, 2)}
    except Exception as exc:  # noqa: BLE001 - we want every failure reason
        return {"ok": False, "error": repr(exc)[:200], "seconds": round(time.time() - start, 2)}
```
- `start = time.time()` — start the stopwatch.
- `Request(url, headers={"User-Agent": ...})` — builds a web request and gives it a name tag. Some websites refuse requests that have no name tag.
- `with urlopen(req, timeout=timeout) as resp:` — opens the page. `with` closes the connection automatically at the end.
- `resp.read(1)` — read just **1 byte**. We only need to know the site answers, not download the page.
- `return {"ok": True, "status": resp.status, ...}` — success: status code (200 = OK) and how many seconds it took.
- `except Exception as exc:` — catch **any** error (no internet, blocked, timeout, ...). We want the reason, so we catch everything.
- `repr(exc)[:200]` — the error as text, cut to 200 characters.
- `# noqa: ...` comments tell the code-style checker "I know, this is on purpose."

### 6.5 `check_nvidia_smi`

```python
def check_nvidia_smi() -> dict[str, str | None]:
    line = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {"nvidia_smi": line}
```
- `nvidia-smi` is NVIDIA's own tool, installed with the graphics driver.
- `--query-gpu=name,driver_version,memory.total` — ask only for GPU name, driver version, and total memory.
- `--format=csv,noheader` — one simple comma-separated line, no header row.
- Uses `run_cmd` from above, so you get a text line or `None`.
- Why check this at all? It talks to the **driver directly**, without PyTorch. If `nvidia-smi` works but PyTorch says "no CUDA", then the problem is the PyTorch install, not the GPU.

---

## 7. Checks that need PyTorch + GPU

### 7.1 `torch_info`

```python
def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": repr(exc)[:300]}
```
- Import PyTorch inside a `try`. If it fails for any reason, return a small report saying `importable: False` plus the error, instead of crashing the whole script.

```python
    info: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not info["cuda_available"]:
        return info
```
- `version` — PyTorch version.
- `cuda_build` — which CUDA version this PyTorch was **built** with (`None` means a CPU-only PyTorch, a common mistake).
- `cuda_available` — can PyTorch really see and use the GPU right now?
- If not, return early. Nothing below makes sense without a GPU.

```python
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    free_b, total_b = torch.cuda.mem_get_info(idx)
```
- `idx = 0` — use the first GPU.
- `props` — an object with GPU details (name, number of processors, ...).
- `major, minor` — compute capability split into two numbers (e.g. `8` and `9` for 8.9).
- `free_b, total_b` — free and total VRAM in bytes.

```python
    info.update(
        device_count=torch.cuda.device_count(),
        gpu_name=props.name,
        compute_capability=f"{major}.{minor}",
        sm_count=props.multi_processor_count,
        vram_total_gb=gb(total_b),
        vram_free_gb=gb(free_b),
        native_bf16=major >= 8,  # Ampere (8.x) and newer have real bf16
    )
```
`info.update(...)` adds many entries at once:
- `device_count` — how many GPUs.
- `gpu_name` — e.g. `NVIDIA GeForce RTX 4060 Laptop GPU`.
- `compute_capability` — e.g. `"8.9"`.
- `sm_count` — number of Streaming Multiprocessors (the GPU's "mini-cores groups"). More = more power.
- `vram_total_gb`, `vram_free_gb` — converted with `gb()`.
- `native_bf16` — `True` if `major >= 8`. RTX 2060 is 7.5 → `False`. RTX 4060 is 8.9 → `True`.

```python
    try:
        info["bf16_reported_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        info["bf16_reported_supported"] = None
    info["recommended_precision"] = (
        "bf16 (no scaler)" if info["native_bf16"] else "fp16 + GradScaler"
    )
    return info
```
- Also asks PyTorch directly "do you support bf16?". (On older GPUs PyTorch may say yes by *emulating* it slowly, which is why we also keep our own `native_bf16` check.) If that call fails, store `None`.
- `recommended_precision` — a one-line recommendation: new GPU → bf16 with no scaler. Old GPU → fp16 with GradScaler.

### 7.2 `bench_matmul`

```python
def bench_matmul(dtype_name: str, n: int = 4096, iters: int = 30) -> dict[str, Any]:
    import torch

    dtype = getattr(torch, dtype_name)
```
- Parameters: the number format name (like `"float16"`), matrix size `n` (4096 × 4096), and how many times to repeat (30).
- `getattr(torch, "float16")` turns the **text** into the real object `torch.float16`.

```python
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
```
- Create two 4096×4096 matrices of random numbers directly on the GPU, in the chosen number format. (`b` is the same idea as `a`.)

```python
    for _ in range(5):  # warm-up
        a @ b
    torch.cuda.synchronize()
```
- `a @ b` = matrix multiplication.
- Do it 5 times first as a **warm-up** (the first runs are slower while the GPU "wakes up").
- `_` is a throwaway variable name (we do not need the counter).
- `torch.cuda.synchronize()` — **wait until the GPU has really finished.** The GPU works in the background; Python moves on immediately, so without this the timing would be wrong.

```python
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize()
```
- Create two **GPU stopwatch markers**. Timing on the GPU itself is more accurate than timing with Python.
- `start.record()` drop the first marker → run 30 multiplications → `end.record()` drop the second marker → wait for everything to finish.

```python
    ms = start.elapsed_time(end) / iters
    tflops = 2 * n**3 / (ms / 1000) / 1e12
```
- `elapsed_time` gives total milliseconds between markers; divide by 30 to get **ms per multiplication**.
- TFLOPS formula: multiplying two n×n matrices takes about `2 × n³` operations (each of the n² output numbers needs n multiplies + n adds). Divide by seconds (`ms / 1000`) to get operations per second, then by `1e12` to get *tera* (trillions).

```python
    del a, b
    torch.cuda.empty_cache()
    return {"ms_per_matmul": round(ms, 3), "tflops": round(tflops, 2)}
```
- Delete the big matrices and give GPU memory back, so the next test starts clean.
- Return the speed results.

### 7.3 `probe_vram`

```python
def probe_vram(chunk_mb: int = 256) -> dict[str, Any]:
    import torch

    free_b, total_b = torch.cuda.mem_get_info(0)
    chunks: list[Any] = []
    allocated = 0
```
- Checks how much VRAM you can **really** grab, in 256 MB chunks.
- Measure free/total VRAM **before** doing anything (important, see the docstring: Windows can pretend you have more).
- `chunks` — a list to hold the chunks (so Python does not throw them away immediately). `allocated` — running total in bytes.

```python
    try:
        while allocated < total_b:
            chunks.append(torch.empty(chunk_mb * 1024**2, dtype=torch.uint8, device="cuda"))
            allocated += chunk_mb * 1024**2
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    finally:
        chunks.clear()
        torch.cuda.empty_cache()
```
- `while allocated < total_b` — keep going until we have taken as much as the card's total memory.
- `torch.empty(256 * 1024**2, dtype=torch.uint8, ...)` — reserve 256 MB (each `uint8` is 1 byte) on the GPU without filling it. Append it to the list and add to the total.
- `except (OutOfMemoryError, RuntimeError): pass` — when the GPU says "no more", that is **expected**. Stop quietly.
- `finally:` — runs **no matter what**. Release all chunks (`clear()`) and return the memory to the GPU (`empty_cache()`).

```python
    spill = allocated > free_b + 256 * 1024**2
```
- If we managed to allocate more than the VRAM that was free at the start (plus a 256 MB tolerance), Windows probably moved some data into **normal RAM**. That is slow, so we flag it as `spill`.

```python
    return {
        "usable_vram_gb": gb(min(allocated, free_b)),  # real VRAM-resident budget
        "free_before_probe_gb": gb(free_b),
        "allocated_in_probe_gb": gb(allocated),
        "total_vram_gb": gb(total_b),
        "system_memory_spill_suspected": spill,
    }
```
- `usable_vram_gb` — the **honest** number: the smaller of "what we allocated" and "what was free". This is the budget you should plan your model around.
- The other values are for the record. Each `gb(...)` is the same conversion as before.

### 7.4 `amp_smoke`

```python
def amp_smoke(dtype_name: str, use_scaler: bool, steps: int = 30) -> dict[str, Any]:
    """Tiny regression with autocast; proves the precision path works."""
    import torch

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)
```
- A **mini training run**. The goal is not to make a good model. The goal is to prove that 16-bit training does not crash or produce garbage on this GPU.
- `dtype` — same text → object trick as before.
- `torch.manual_seed(0)` — fix the random numbers so results are repeatable.

```python
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
```
- `Sequential(...)` — a tiny neural network that runs layers one after another:
  - `Linear(256, 512)` — takes 256 numbers, produces 512.
  - `GELU()` — a smooth "activation" function that adds non-linearity (lets the network learn curves, not just straight lines).
  - `Linear(512, 256)` — brings it back to 256 numbers.
- `.cuda()` — move the model onto the GPU.
- `AdamW` — the optimizer: the thing that updates the model's weights. `lr=1e-3` = learning rate 0.001 (how big each step is).

```python
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
```
- Create the GradScaler. `enabled=False` makes it a "do nothing" version (used for bf16, which does not need scaling).
- The `except` handles **older PyTorch**, where the function lives in a different place.

```python
    x = torch.randn(64, 256, device="cuda")
    y = torch.randn(64, 256, device="cuda")
```
- `x` = fake input (64 samples × 256 numbers). `y` = fake target answers (same shape; same idea as `x`). Random data is fine, because we only test the machinery.

```python
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = torch.nn.functional.mse_loss(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.item()))
```
One training step, repeated 30 times:
1. `opt.zero_grad(set_to_none=True)` — erase the old gradients (learning signals) from the last step. `set_to_none=True` is slightly faster and uses less memory.
2. `with torch.autocast(...)` — inside this block, PyTorch automatically uses 16-bit math where safe.
3. `model(x)` — forward pass: the model's guess. `mse_loss(guess, y)` — mean squared error: "how far is the guess from the target?" Smaller = better.
4. `scaler.scale(loss).backward()` — multiply the loss by a big factor (if scaler on), then **backward** = compute gradients (which direction to adjust each weight).
5. `scaler.unscale_(opt)` — divide the gradients back to their real size, so the next line works on true values.
6. `clip_grad_norm_(..., 1.0)` — if the gradients are dangerously large, shrink them so the total size is at most 1.0. Prevents "exploding" training.
7. `scaler.step(opt)` — update the weights (the scaler skips the step if it finds NaN/infinity).
8. `scaler.update()` — adjust the scaling factor for the next step.
9. `losses.append(float(loss.item()))` — save this step's loss as a normal Python number.

```python
    finite = all(math.isfinite(v) for v in losses)
    return {
        "dtype": dtype_name,
        "scaler": use_scaler,
        "first_loss": round(losses[0], 4),
        "last_loss": round(losses[-1], 4),
        "passed": finite and losses[-1] < losses[0],
    }
```
- `finite` — are **all** losses real numbers (no NaN / infinity)?
- `"passed"` — pass only if all are finite **and** the last loss is lower than the first (the model actually learned a bit).

---

## 8. Reporting: `append_run_log`

```python
def append_run_log(machine: str, stamp: str, report: dict[str, Any], json_name: str) -> None:
    t = report["torch"]
    sysinfo = report["system"]
```
- Takes the label, time, the full report, and the JSON file name. Makes short names `t` (torch info) and `sysinfo` (system info) so later lines are shorter.

```python
    lines = [
        "",
        f"## {stamp} | {machine} | Day 1 environment check",
        f"- Machine label: {machine} (what actually ran: this entry's machine)",
        f"- OS: {sysinfo['os']}",
        f"- Python: {sysinfo['python']} (venv: {sysinfo['in_venv']})",
        f"- RAM: {sysinfo.get('ram_total_gb')} GB, free disk: {sysinfo['disk_free_gb']} GB",
        f"- nvidia-smi: {report['nvidia_smi'].get('nvidia_smi')}",
        f"- PyTorch: {t.get('version')} (CUDA build {t.get('cuda_build')}), "
        f"cuda_available={t.get('cuda_available')}",
    ]
```
- Builds a **list of text lines** for a Markdown log entry. An empty `""` first = blank line before the new entry.
- `## ...` makes a Markdown heading with time and machine.
- `sysinfo['os']` uses square brackets (crashes if missing). `sysinfo.get('ram_total_gb')` uses `.get` (returns `None` if missing). Both work here; `.get` is used where a value may be absent.
- The PyTorch line is split into two f-strings that Python **joins automatically** into one line.

```python
    if t.get("cuda_available"):
        vp = report.get("vram_probe", {})
        lines += [
            f"- GPU: {t['gpu_name']}, {t['vram_total_gb']} GB total, "
            f"free at start (usable budget): {vp.get('usable_vram_gb')} GB, "
            f"compute capability {t['compute_capability']}",
            f"- VRAM probe: allocated {vp.get('allocated_in_probe_gb')} GB, "
            f"spill to system memory suspected: {vp.get('system_memory_spill_suspected')}",
            f"- Native bf16: {t['native_bf16']} -> plan: {t['recommended_precision']}",
        ]
```
- Only if the GPU is usable, add GPU lines.
- `report.get("vram_probe", {})` — if the probe was skipped (`--skip-vram-probe`), use an empty dictionary, so `.get(...)` just returns `None` instead of crashing.
- `lines += [...]` — add several lines to the end of the list.

```python
        for name, res in report.get("matmul", {}).items():
            lines.append(f"- matmul 4096 {name}: {res['tflops']} TFLOPS")
        for res in report.get("amp", []):
            lines.append(
                f"- AMP smoke {res['dtype']} (scaler={res['scaler']}): "
                f"{'PASS' if res['passed'] else 'FAIL'}"
            )
```
- First loop: one line per number format (float32, float16, maybe bfloat16) with its speed.
- Second loop: one line per AMP test with `PASS` or `FAIL`.
- `'PASS' if res['passed'] else 'FAIL'` — a one-line if/else.

```python
    net = ", ".join(f"{k}={'ok' if v['ok'] else 'BLOCKED'}" for k, v in report["network"].items())
    lines += [
        f"- Network: {net}",
        f"- UTF-8 Japanese round trip: {'PASS' if report['checks']['utf8_roundtrip'] else 'FAIL'}",
        f"- Overall: {'PASS' if report['overall_pass'] else 'FAIL'} "
        f"(details: runs/{json_name})",
    ]
```
- Build a text like `pypi=ok, pytorch_wheels=ok, github=BLOCKED` by looping over the network results and joining with `", "`.
- Add network, UTF-8 and overall result lines. (`PASS/FAIL` text works the same as before.)

```python
    if not RUN_LOG.exists():
        RUN_LOG.write_text("# Run log\n", encoding="utf-8")
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
```
- If `run_log.md` does not exist yet, create it with a title.
- Open in `"a"` (**append**) mode: add to the end, never erase old entries.
- `"\n".join(lines)` — glue all lines together with line breaks, then add one final newline.

---

## 9. The main program

```python
def main() -> int:
    setup_utf8_console()
```
- `main()` returns an integer: `0` = success, `1` = failure (like a program exit code).
- First thing: make the console safe for Japanese.

```python
    parser = argparse.ArgumentParser(description="Day 1 environment check")
    parser.add_argument("--machine", choices=["school", "home"], required=True)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-vram-probe", action="store_true")
    args = parser.parse_args()
```
- Defines the command-line options:
  - `--machine` must be `school` or `home` and is **required**.
  - `--skip-network` — a flag with no value. If you write it, it becomes `True`, otherwise `False` (`store_true`).
  - `--skip-vram-probe` — same style flag.
- `parse_args()` reads what you typed. Access values like `args.machine`.

```python
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M")
```
- `stamp` — readable time like `2026-10-30 14:05` (for the log).
- `file_stamp` — file-name-safe time like `20261030_1405` (no spaces or colons, which are bad in Windows file names).

```python
    print(f"=== Day 1 environment check | machine={args.machine} | {stamp} ===")
    print(f"Japanese display test: {JAPANESE_PROBE}")
```
- Prints a header, and prints Japanese so **you** can see with your eyes if the console shows it correctly.

```python
    report: dict[str, Any] = {"machine": args.machine, "timestamp": stamp}
    report["system"] = check_system()
    report["packages"] = check_packages()
    report["nvidia_smi"] = check_nvidia_smi()
    report["network"] = (
        {} if args.skip_network else {k: check_url(u) for k, u in NETWORK_TARGETS.items()}
    )
    report["torch"] = torch_info()
```
- Start the report as a dictionary and fill it one section at a time (system, packages, nvidia-smi).
- Network: if `--skip-network` was given → empty `{}`. Otherwise a **dictionary comprehension**: for each name/URL pair, run `check_url` and store the result under the same name.
- Then add the PyTorch info.

```python
    t = report["torch"]
    if t.get("cuda_available"):
        report["matmul"] = {"float32": bench_matmul("float32"), "float16": bench_matmul("float16")}
        if t["native_bf16"]:
            report["matmul"]["bfloat16"] = bench_matmul("bfloat16")
        if not args.skip_vram_probe:
            report["vram_probe"] = probe_vram()
        report["amp"] = [amp_smoke("float16", use_scaler=True)]
        if t["native_bf16"]:
            report["amp"].append(amp_smoke("bfloat16", use_scaler=False))
```
Only runs if a GPU works:
- Speed test fp32 and fp16 always; bf16 **only** if the GPU has native bf16 (the RTX 4060 yes, the RTX 2060 no).
- VRAM probe unless you asked to skip it.
- AMP test with fp16 **with** scaler. If bf16 is native, also run bf16 **without** scaler (`.append` adds to the list).

```python
    checks: dict[str, bool] = {
        "python_3_10_plus": sys.version_info >= (3, 10),
        "running_in_venv": report["system"]["in_venv"],
        "torch_importable": bool(t.get("importable")),
        "cuda_available": bool(t.get("cuda_available")),
        "fp16_amp_ok": any(r["dtype"] == "float16" and r["passed"] for r in report.get("amp", [])),
        "utf8_roundtrip": check_utf8_roundtrip(),
    }
    report["checks"] = checks
    report["overall_pass"] = all(checks.values())
```
The final scorecard: six yes/no questions.
- Python is 3.10 or newer?
- Running inside the venv?
- PyTorch can be imported?
- CUDA is available?
- The fp16 AMP test passed? (`any(...)` = "is at least one item true?")
- Japanese UTF-8 round trip works? (this runs the test right here)
- `all(checks.values())` — overall pass only if **every** check is `True`.
- `bool(...)` forces a clean `True`/`False`.

```python
    tag = EXPECTED_GPU_TAG[args.machine]
    if t.get("gpu_name") and tag not in t["gpu_name"]:
        print(f"NOTE: --machine {args.machine} usually has an RTX {tag}, but found {t['gpu_name']}.")
        print("      Check that you used the right --machine label; the run log must be truthful.")
```
- Look up the expected GPU number for your label (e.g. `"2060"` for school).
- If a GPU name exists **and** it does not contain that number, print a warning: you probably typed the wrong `--machine`. (Second `print` is the same idea, just the second line of the warning.)

```python
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    json_name = f"env_check_{args.machine}_{file_stamp}.json"
    (RUNS_DIR / json_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    append_run_log(args.machine, stamp, report, json_name)
```
- Make sure `runs/` exists.
- Build a file name like `env_check_school_20261030_1405.json`.
- `json.dumps(report, indent=2, ensure_ascii=False)` — turn the report into nicely indented JSON text. `ensure_ascii=False` keeps Japanese characters as real Japanese instead of `\u4eca...` codes.
- Save it as UTF-8, then append the summary to `run_log.md`.

```python
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== SUMMARY ===")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    blocked = [k for k, v in report["network"].items() if not v["ok"]]
    if blocked:
        print(f"NOTE: network targets not reachable: {blocked} (tell me, this affects installs)")
    print(f"\nSaved: runs/{json_name}  and an entry in run_log.md")
    return 0 if report["overall_pass"] else 1
```
- Print the full report on screen.
- Print the summary: one `[PASS]` or `[FAIL]` line per check.
- `blocked = [...]` — a **list comprehension**: collect the names of every website that failed. If the list is not empty, print a warning.
- Print where the files were saved.
- Return `0` if everything passed, otherwise `1`.

```python
if __name__ == "__main__":
    raise SystemExit(main())
```
- `__name__ == "__main__"` is true only when you run the file directly (not when another file imports it). So the checkup starts only when you run it yourself.
- `raise SystemExit(main())` — run `main()` and use its return value (0 or 1) as the program's exit code. Scripts and CI tools can read this to know if it passed.

---

## 10. What the whole run looks like (order of events)

1. Console set to UTF-8 → read `--machine`.
2. System info → package versions → `nvidia-smi` → network → PyTorch info.
3. If GPU works → speed tests → VRAM probe → mini training tests.
4. Build the scorecard (`checks`) → overall pass/fail.
5. Warn if the machine label looks wrong.
6. Save JSON → append to `run_log.md` → print summary.
7. Exit code `0` (pass) or `1` (fail).

## 11. How to read the results

- **`native_bf16: False` (RTX 2060)** → use fp16 + GradScaler when training.
- **`native_bf16: True` (RTX 4060)** → use bf16, no scaler needed.
- **`usable_vram_gb`** → the real memory budget for your model, batch size, and optimizer states.
- **`system_memory_spill_suspected: True`** → Windows may let you "use" more memory than you really have, but it is slow. Trust `usable_vram_gb`, not the raw allocation number.
- **`BLOCKED` network targets** → pip/PyTorch installs may fail on that network.
- **`FAIL` on `fp16_amp_ok`** → mixed-precision training is not safe on this machine yet; fix before training.

## 12. Quick summary in one paragraph

The script gathers facts about the computer, tests if Japanese text is safe, tests internet access, tests the GPU's speed and memory, runs a tiny 16-bit training test, scores everything with PASS/FAIL, and saves the results both as a JSON file and as a note in `run_log.md`, so you can compare the school PC and the home laptop honestly.
