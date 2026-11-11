r"""Day 8 - build Model A at the three trial sizes, check it on the real data, and report.
Day 8 - 3つの大きさの Model A を作り、実際のデータで確かめ、報告する

==========================  ENGLISH  ==========================

1. Reads the vocabulary size from data/tokenizer/tokenizer.json (Day 6) and the
   number of training tokens from data/tokens/train_meta.json (Day 7).
2. For the three trial sizes (tiny / small / medium, see src/model.py PRESETS):
   parameter counts, training tokens per parameter, and one timed training step
   (forward + backward) on the GPU with the real block size: milliseconds per step,
   tokens per second and peak GPU memory.
3. Checks on real token windows from train.bin:
     - the loss before training is close to ln(vocabulary size) (the model knows nothing yet),
     - the manual and the fast attention give the same output,
     - a model can memorise one real batch (loss falls close to 0): proof that the
       model, the loss and the gradients are wired correctly.
4. Saves the three configurations to configs/model_{tiny,small,medium}.json (used
   by Day 9 and Day 11), writes docs/DAY8_MODEL_REPORT.md and a run_log.md entry.

Usage (VS Code terminal, repo root):
    .venv\Scripts\python.exe scripts\day8_model_check.py --machine home

SERVER NOTE (Linux, later): same code; use the Linux venv path and the label server:
    .venv/bin/python scripts/day8_model_check.py --machine server
Parameter counts are identical everywhere; speed and memory depend on the GPU.
On a GPU without bf16 (for example an RTX 2060) the timing uses fp16 and the
memorisation check uses fp32; the script chooses this by itself.

==========================  日本語  ==========================

1. 語彙の大きさを data/tokenizer/tokenizer.json(Day 6)から、学習用のトークン数を
   data/tokens/train_meta.json(Day 7)から読む。
2. 3つの大きさ(tiny / small / medium)について、パラメータ数、1パラメータあたりの学習トークン数、
   GPU での学習1ステップ(順伝播+逆伝播)の時間、1秒あたりのトークン数、GPU メモリの最大使用量を測る。
3. train.bin の実際のトークンで確かめる:
     - 学習前の損失が ln(語彙の大きさ) に近いこと(まだ何も知らない状態)
     - 手書きの注意(manual)と高速な注意(fast)が同じ結果になること
     - 1つの実際のバッチを覚えられること(損失が0に近づく): モデル・損失・勾配が正しくつながっている証拠
4. 3つの設定を configs/model_{tiny,small,medium}.json に保存し(Day 9 と Day 11 で使う)、
   docs/DAY8_MODEL_REPORT.md を書き、run_log.md に追記する。

使い方(VS Codeターミナル、リポジトリのルートで):
    .venv\Scripts\python.exe scripts\day8_model_check.py --machine home

サーバーで使う場合の注意(Linux、後日): コードは同じ。Linux の venv のパスと server というラベルを使う:
    .venv/bin/python scripts/day8_model_check.py --machine server
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.model import PRESETS, ModelA, count_parameters, expected_parameters, preset_config  # noqa: E402

SCRIPT_VERSION: str = "day8-v1"


@dataclass
class SizeReport:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    total: int
    embedding: int
    non_embedding: int
    tokens_per_param: float
    ms_per_step: float
    tokens_per_sec: float
    peak_mem_mb: float


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available")
    return torch.device(choice)


def precision(device: torch.device) -> tuple[str, torch.dtype | None]:
    """(label, autocast dtype). bf16 where supported, fp16 on older GPUs, fp32 on the CPU."""
    if device.type != "cuda":
        return "fp32", None
    if torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16


def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def real_batch(tokens: np.ndarray, batch: int, length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """batch windows of length+1 tokens from fixed random positions: x = first length, y = shifted by one."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - length - 1, size=batch)
    win = np.stack([np.asarray(tokens[s:s + length + 1], dtype=np.int64) for s in starts])
    w = torch.from_numpy(win)
    return w[:, :-1], w[:, 1:]


def time_training_step(model: ModelA, x: torch.Tensor, y: torch.Tensor, device: torch.device,
                       dtype: torch.dtype | None, repeats: int) -> tuple[float, float]:
    """Average milliseconds of one forward + backward pass, and the peak GPU memory in MB."""
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def step() -> None:
        with autocast(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    step()  # warm-up (memory allocation, kernel selection)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) * 1000 / repeats
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return ms, peak


def memorise_one_batch(vocab: int, block: int, x: torch.Tensor, y: torch.Tensor, device: torch.device,
                       dtype: torch.dtype | None, steps: int, seed: int) -> list[float]:
    torch.manual_seed(seed)
    model = ModelA(preset_config("tiny", vocab, block)).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    use = dtype if dtype == torch.bfloat16 else None   # fp16 would need a gradient scaler: use fp32 instead
    losses: list[float] = []
    for _ in range(steps):
        with autocast(device, use):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def write_report(path: Path, a: argparse.Namespace, device_name: str, prec: str, vocab: int, train_tokens: int,
                 reports: list[SizeReport], init_loss: float, attn_diff: float, losses: list[float]) -> None:
    lines = [
        "# Day 8 report: Model A at three sizes",
        "",
        f"Generated by `scripts/day8_model_check.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}.",
        f"Machine label: {a.machine}. Device: {device_name}. Precision for timing: {prec}. "
        f"PyTorch {torch.__version__}, Python {platform.python_version()}.",
        f"Vocabulary {vocab:,} (Day 6). Training tokens {train_tokens:,} (Day 7). "
        f"Block size {a.block_size}, timing batch {a.batch} x {a.block_size} tokens.",
        "",
        "## Sizes",
        "",
        "| Preset | Layers | Heads | Width | Parameters | Embedding | Non-embedding | Training tokens per parameter "
        "| ms per step | Tokens per second | Peak GPU memory (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(f"| {r.name} | {r.n_layer} | {r.n_head} | {r.d_model} | {r.total:,} | {r.embedding:,} | {r.non_embedding:,} "
          f"| {r.tokens_per_param:.2f} | {r.ms_per_step:.1f} | {r.tokens_per_sec:,.0f} | {mem_text(r.peak_mem_mb)} |"
          for r in reports),
        "",
        "One step = forward + backward on one batch (no optimizer). Embedding and output layer share one matrix",
        "(weight tying), so it is counted once. Peak GPU memory is shown as - on the CPU.",
        "The size is chosen on Day 11 by validation loss.",
        "",
        "## Checks on real token windows from train.bin",
        "",
        f"- Loss before training (tiny): {init_loss:.3f}; a uniform guess over {vocab:,} tokens gives "
        f"ln({vocab:,}) = {math.log(vocab):.3f}.",
        f"- Manual vs fast attention (tiny, fp32): largest difference in the outputs {attn_diff:.2e}.",
        f"- Memorising one batch ({a.overfit_batch} x {a.overfit_len} tokens, tiny, {len(losses)} steps): "
        f"loss {losses[0]:.3f} -> {losses[-1]:.4f}.",
        "",
        "| Step | 1 | " + " | ".join(str(s) for s in _milestones(len(losses))) + " |",
        "| --- | --- | " + " | ".join("---" for _ in _milestones(len(losses))) + " |",
        f"| Loss | {losses[0]:.3f} | " + " | ".join(f"{losses[s - 1]:.4f}" for s in _milestones(len(losses))) + " |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def mem_text(mb: float) -> str:
    return "-" if math.isnan(mb) else f"{mb:,.0f}"


def _milestones(n: int) -> list[int]:
    return sorted({max(1, n * k // 5) for k in range(1, 6)})


def append_run_log(log_path: Path, a: argparse.Namespace, device_name: str, prec: str,
                   reports: list[SizeReport], init_loss: float, attn_diff: float, losses: list[float]) -> None:
    body = [
        f"- Machine label: {a.machine}; device: {device_name}; precision for timing: {prec}",
        f"- Script: scripts/day8_model_check.py ({SCRIPT_VERSION}); PyTorch {torch.__version__}; "
        f"Python {platform.python_version()}",
        *(f"- {r.name}: {r.total:,} parameters, {r.ms_per_step:.1f} ms/step, {r.tokens_per_sec:,.0f} tokens/s, "
          f"peak GPU memory {mem_text(r.peak_mem_mb)} MB" for r in reports),
        f"- Checks: initial loss {init_loss:.3f}; manual vs fast attention max diff {attn_diff:.2e}; "
        f"memorise one batch {losses[0]:.3f} -> {losses[-1]:.4f}",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {a.machine} | Day 8 model check\n\n" + "\n".join(body) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 8: build Model A at three sizes and check it.")
    ap.add_argument("--machine", choices=["home", "server", "school"], required=True)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32, help="batch size for the timing step")
    ap.add_argument("--repeats", type=int, default=10, help="timed steps per size")
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--overfit-batch", type=int, default=4)
    ap.add_argument("--overfit-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--tokens-dir", type=Path, default=REPO_ROOT / "data/tokens")
    ap.add_argument("--configs-dir", type=Path, default=REPO_ROOT / "configs")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY8_MODEL_REPORT.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("tokenizer", "tokens_dir", "configs_dir", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    meta_path = a.tokens_dir / "train_meta.json"
    if not a.tokenizer.exists():
        raise SystemExit(f"Missing {a.tokenizer}. Run Day 6 first.")
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}. Run Day 7 first.")

    device = pick_device(a.device)
    prec, dtype = precision(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Day 8 model check | machine={a.machine} | {device_name} | {prec} | PyTorch {torch.__version__} ===")

    vocab = ByteBPETokenizer.load(a.tokenizer).vocab_size
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["vocab_size"] != vocab:
        raise SystemExit("train_meta.json was made with a different tokenizer. Rerun Day 7.")
    train_tokens = int(meta["tokens"])
    tokens = np.memmap(a.tokens_dir / "train.bin", dtype=np.dtype("<u2"), mode="r")
    print(f"Vocabulary {vocab:,}; training tokens {train_tokens:,}; block size {a.block_size}")

    # ---- 1. three sizes: parameters and one timed training step
    a.configs_dir.mkdir(parents=True, exist_ok=True)
    x, y = (t.to(device) for t in real_batch(tokens, a.batch, a.block_size, a.seed))
    reports: list[SizeReport] = []
    print(f"\n{'preset':<7} {'params':>11} {'tokens/param':>12} {'ms/step':>9} {'tokens/s':>11} {'peak MB':>8}")
    for name in PRESETS:
        cfg = preset_config(name, vocab, a.block_size)
        cfg.save(a.configs_dir / f"model_{name}.json")
        torch.manual_seed(a.seed)
        model = ModelA(cfg).to(device)
        counts = count_parameters(model)
        if counts["total"] != expected_parameters(cfg):
            raise SystemExit(f"{name}: parameter count does not match the formula")
        ms, peak = time_training_step(model, x, y, device, dtype, a.repeats)
        r = SizeReport(name, cfg.n_layer, cfg.n_head, cfg.d_model, counts["total"], counts["embedding"],
                       counts["non_embedding"], train_tokens / counts["total"], ms,
                       a.batch * a.block_size / (ms / 1000), peak)
        reports.append(r)
        print(f"{name:<7} {r.total:>11,} {r.tokens_per_param:>12.2f} {r.ms_per_step:>9.1f} "
              f"{r.tokens_per_sec:>11,.0f} {mem_text(r.peak_mem_mb):>8}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 2. loss before training, and manual vs fast attention (fp32)
    torch.manual_seed(a.seed)
    tiny = ModelA(preset_config("tiny", vocab, a.block_size)).to(device).eval()
    with torch.no_grad():
        fast_logits, init = tiny(x, y)
        tiny.set_attention("manual")
        manual_logits, _ = tiny(x, y)
    init_loss = init.item()
    attn_diff = (fast_logits - manual_logits).abs().max().item()
    print(f"\nLoss before training: {init_loss:.3f}  (ln {vocab} = {math.log(vocab):.3f})")
    print(f"Manual vs fast attention: max difference {attn_diff:.2e}")
    if abs(init_loss - math.log(vocab)) > 0.5 or attn_diff > 1e-3:
        raise SystemExit("Check FAILED: initial loss or attention difference out of range")
    del tiny

    # ---- 3. memorise one real batch
    ox, oy = (t.to(device) for t in real_batch(tokens, a.overfit_batch, a.overfit_len, a.seed + 1))
    losses = memorise_one_batch(vocab, a.block_size, ox, oy, device, dtype, a.overfit_steps, a.seed)
    print(f"Memorise one batch ({a.overfit_batch} x {a.overfit_len}): loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    for s in _milestones(len(losses)):
        print(f"   step {s:>4}: {losses[s - 1]:.4f}")
    if losses[-1] > 0.5:
        raise SystemExit("Check FAILED: the model could not memorise one batch")

    write_report(a.report, a, device_name, prec, vocab, train_tokens, reports, init_loss, attn_diff, losses)
    print(f"\nAll checks PASS. Report: {a.report}\nConfigs: {a.configs_dir}")
    if not a.no_log:
        append_run_log(a.run_log, a, device_name, prec, reports, init_loss, attn_diff, losses)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
