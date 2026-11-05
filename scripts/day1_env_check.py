r"""Day 1 - environment check for Phase 1 (Model A).
Day 1 - フェーズ1(Model A)の環境チェック

==========================  ENGLISH  ==========================

Runs identically on the school PC (RTX 2060, 6 GB) and the home laptop
(RTX 4060, 8 GB). The only thing that changes is the --machine label.

HOME REPOSITORY COPY. SCHOOL-PC SWAP NOTE:
    Nothing in this file has to change for the school PC. Only run it with
    --machine school instead of --machine home.

What it does
------------
1. Records Python, OS, CPU, RAM, free disk, console encoding.
2. Checks that Japanese text survives a UTF-8 file round trip.
3. Checks network reachability (important on a school network).
4. Imports PyTorch and reads GPU name, VRAM, compute capability.
5. Benchmarks a 4096x4096 matmul in fp32 / fp16 (/ bf16 if native).
6. Probes how much VRAM can really be allocated.
7. Runs a tiny mixed-precision training smoke test (autocast + GradScaler).
8. Writes runs/env_check_<machine>_<timestamp>.json and appends an entry
   to run_log.md.

Usage (from the repo root, using the venv's python, no activation needed):
    .venv\Scripts\python.exe scripts\day1_env_check.py --machine school
    .venv\Scripts\python.exe scripts\day1_env_check.py --machine home

==========================  日本語  ==========================

学校PC(RTX 2060, 6 GB)と自宅ノートPC(RTX 4060, 8 GB)のどちらでも、
同じように動作します。変わるのは --machine ラベルだけです。

自宅リポジトリ用のコピーです。学校PCで使う場合の注意:
    このファイルの中身は変更不要です。--machine home の代わりに
    --machine school を付けて実行するだけです。

このスクリプトが行うこと
------------------------
1. Python、OS、CPU、RAM、ディスクの空き容量、コンソールの文字コードを記録する。
2. 日本語テキストを UTF-8 のファイルに保存して読み戻しても壊れないか確認する。
3. ネットワークに接続できるか確認する(学校のネットワークでは重要)。
4. PyTorch を読み込み、GPU名、VRAM、compute capability を取得する。
5. 4096x4096 の行列積を fp32 / fp16(対応していれば bf16 も)でベンチマークする。
6. 実際に確保できる VRAM の量を調べる。
7. 小さな混合精度の学習テストを実行する(autocast + GradScaler)。
8. runs/env_check_<machine>_<timestamp>.json を書き出し、run_log.md に
   エントリを追記する。

使い方(リポジトリのルートで、venv の python を直接指定。有効化は不要):
    .venv\Scripts\python.exe scripts\day1_env_check.py --machine school
    .venv\Scripts\python.exe scripts\day1_env_check.py --machine home
"""
from __future__ import annotations

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

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RUN_LOG: Path = REPO_ROOT / "run_log.md"
RUNS_DIR: Path = REPO_ROOT / "runs"

EXPECTED_GPU_TAG: dict[str, str] = {"school": "2060", "home": "4060"}
NETWORK_TARGETS: dict[str, str] = {
    "pypi": "https://pypi.org/simple/pip/",
    "pytorch_wheels": "https://download.pytorch.org/whl/",
    "github": "https://github.com",
}
PACKAGES: list[str] = ["torch", "numpy", "psutil", "tqdm", "pytest"]
JAPANESE_PROBE: str = "今日は晴れ。売上・給与・勤怠"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def setup_utf8_console() -> None:
    """Make printing Japanese safe on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


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


def gb(num_bytes: float) -> float:
    return round(num_bytes / 1024**3, 2)


# --------------------------------------------------------------------------
# checks that need no GPU
# --------------------------------------------------------------------------
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
    try:
        import psutil  # type: ignore

        info["ram_total_gb"] = gb(psutil.virtual_memory().total)
        info["ram_available_gb"] = gb(psutil.virtual_memory().available)
    except ImportError:
        info["ram_total_gb"] = None
    return info


def check_packages() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def check_utf8_roundtrip() -> bool:
    """Write and re-read Japanese text explicitly as UTF-8."""
    tmp = RUNS_DIR / "_utf8_probe.txt"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_text(JAPANESE_PROBE, encoding="utf-8")
    ok = tmp.read_text(encoding="utf-8") == JAPANESE_PROBE
    tmp.unlink(missing_ok=True)
    return ok


def check_url(url: str, timeout: float = 8.0) -> dict[str, Any]:
    start = time.time()
    try:
        req = Request(url, headers={"User-Agent": "model-a-env-check"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read(1)
            return {"ok": True, "status": resp.status, "seconds": round(time.time() - start, 2)}
    except Exception as exc:  # noqa: BLE001 - we want every failure reason
        return {"ok": False, "error": repr(exc)[:200], "seconds": round(time.time() - start, 2)}


def check_nvidia_smi() -> dict[str, str | None]:
    line = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {"nvidia_smi": line}


# --------------------------------------------------------------------------
# checks that need PyTorch + GPU
# --------------------------------------------------------------------------
def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": repr(exc)[:300]}

    info: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not info["cuda_available"]:
        return info

    idx = 0
    props = torch.cuda.get_device_properties(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    info.update(
        device_count=torch.cuda.device_count(),
        gpu_name=props.name,
        compute_capability=f"{major}.{minor}",
        sm_count=props.multi_processor_count,
        vram_total_gb=gb(total_b),
        vram_free_gb=gb(free_b),
        native_bf16=major >= 8,  # Ampere (8.x) and newer have real bf16
    )
    try:
        info["bf16_reported_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        info["bf16_reported_supported"] = None
    info["recommended_precision"] = (
        "bf16 (no scaler)" if info["native_bf16"] else "fp16 + GradScaler"
    )
    return info


def bench_matmul(dtype_name: str, n: int = 4096, iters: int = 30) -> dict[str, Any]:
    import torch

    dtype = getattr(torch, dtype_name)
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):  # warm-up
        a @ b
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    tflops = 2 * n**3 / (ms / 1000) / 1e12
    del a, b
    torch.cuda.empty_cache()
    return {"ms_per_matmul": round(ms, 3), "tflops": round(tflops, 2)}


def probe_vram(chunk_mb: int = 256) -> dict[str, Any]:
    """Allocate 256 MB chunks until the GPU refuses (or total VRAM is reached).

    英語: On Windows (WDDM) the NVIDIA driver may let CUDA allocations spill
    into shared system memory, so "allocated" can exceed the VRAM that is
    really free. The honest budget is therefore the free VRAM measured BEFORE
    the probe, and a spill is flagged.
    日本語: Windows(WDDM)ではドライバーが共有システムメモリに逃がすことがあり、
    確保できた量が実際の空きVRAMを超えることがある。そのため、本当の予算は
    プローブ前に測った空きVRAMとし、逃げが疑われる場合は印を付ける。
    """
    import torch

    free_b, total_b = torch.cuda.mem_get_info(0)
    chunks: list[Any] = []
    allocated = 0
    try:
        while allocated < total_b:
            chunks.append(torch.empty(chunk_mb * 1024**2, dtype=torch.uint8, device="cuda"))
            allocated += chunk_mb * 1024**2
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    finally:
        chunks.clear()
        torch.cuda.empty_cache()
    spill = allocated > free_b + 256 * 1024**2
    return {
        "usable_vram_gb": gb(min(allocated, free_b)),  # real VRAM-resident budget
        "free_before_probe_gb": gb(free_b),
        "allocated_in_probe_gb": gb(allocated),
        "total_vram_gb": gb(total_b),
        "system_memory_spill_suspected": spill,
    }


def amp_smoke(dtype_name: str, use_scaler: bool, steps: int = 30) -> dict[str, Any]:
    """Tiny regression with autocast; proves the precision path works."""
    import torch

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    x = torch.randn(64, 256, device="cuda")
    y = torch.randn(64, 256, device="cuda")

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

    finite = all(math.isfinite(v) for v in losses)
    return {
        "dtype": dtype_name,
        "scaler": use_scaler,
        "first_loss": round(losses[0], 4),
        "last_loss": round(losses[-1], 4),
        "passed": finite and losses[-1] < losses[0],
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def append_run_log(machine: str, stamp: str, report: dict[str, Any], json_name: str) -> None:
    t = report["torch"]
    sysinfo = report["system"]
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
        for name, res in report.get("matmul", {}).items():
            lines.append(f"- matmul 4096 {name}: {res['tflops']} TFLOPS")
        for res in report.get("amp", []):
            lines.append(
                f"- AMP smoke {res['dtype']} (scaler={res['scaler']}): "
                f"{'PASS' if res['passed'] else 'FAIL'}"
            )
    net = ", ".join(f"{k}={'ok' if v['ok'] else 'BLOCKED'}" for k, v in report["network"].items())
    lines += [
        f"- Network: {net}",
        f"- UTF-8 Japanese round trip: {'PASS' if report['checks']['utf8_roundtrip'] else 'FAIL'}",
        f"- Overall: {'PASS' if report['overall_pass'] else 'FAIL'} "
        f"(details: runs/{json_name})",
    ]
    if not RUN_LOG.exists():
        RUN_LOG.write_text("# Run log\n", encoding="utf-8")
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    setup_utf8_console()
    parser = argparse.ArgumentParser(description="Day 1 environment check")
    parser.add_argument("--machine", choices=["school", "home"], required=True)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-vram-probe", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"=== Day 1 environment check | machine={args.machine} | {stamp} ===")
    print(f"Japanese display test: {JAPANESE_PROBE}")

    report: dict[str, Any] = {"machine": args.machine, "timestamp": stamp}
    report["system"] = check_system()
    report["packages"] = check_packages()
    report["nvidia_smi"] = check_nvidia_smi()
    report["network"] = (
        {} if args.skip_network else {k: check_url(u) for k, u in NETWORK_TARGETS.items()}
    )
    report["torch"] = torch_info()

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

    tag = EXPECTED_GPU_TAG[args.machine]
    if t.get("gpu_name") and tag not in t["gpu_name"]:
        print(f"NOTE: --machine {args.machine} usually has an RTX {tag}, but found {t['gpu_name']}.")
        print("      Check that you used the right --machine label; the run log must be truthful.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    json_name = f"env_check_{args.machine}_{file_stamp}.json"
    (RUNS_DIR / json_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    append_run_log(args.machine, stamp, report, json_name)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== SUMMARY ===")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    blocked = [k for k, v in report["network"].items() if not v["ok"]]
    if blocked:
        print(f"NOTE: network targets not reachable: {blocked} (tell me, this affects installs)")
    print(f"\nSaved: runs/{json_name}  and an entry in run_log.md")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
