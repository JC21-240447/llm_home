r"""The training loop for Model A: resumable, logged, mixed precision (used by Day 9, 11 and 12).
Model A の学習ループ: 再開でき、記録を残し、混合精度で動く(Day 9・11・12 で使う)

==========================  ENGLISH  ==========================

One optimizer STEP:
    1. learning rate for this step: linear warm-up, then cosine decay to min_lr
    2. grad_accum micro-batches: forward (bf16/fp16 autocast), loss, backward
    3. clip the gradient norm to grad_clip
    4. AdamW update (weight decay only on matrices, not on norm gains)
Every log_every steps   : one row in log.csv (loss, perplexity, lr, tokens/s, GPU memory)
Every eval_every steps  : validation loss on the fixed validation windows, a greedy
                          continuation of fixed prompts (samples.md), checkpoint
                          last.pt, and best.pt when validation improved.

Resuming: last.pt holds the model, the optimizer, the gradient scaler, the step,
the random states and the batch generator state. A resumed run therefore uses the
same batches and learning rates as an uninterrupted one, and the loss curve
continues without a jump.

Stopping safely: Ctrl+C (once) finishes the current step, saves last.pt and exits.
Checkpoints are written to a temporary file first and then renamed, so an
interruption during saving never leaves a broken last.pt.

==========================  日本語  ==========================

1回の最適化ステップ:
    1. このステップの学習率: 最初は直線的に上げ(ウォームアップ)、その後コサインで min_lr まで下げる
    2. grad_accum 個の小さなバッチ: 順伝播(bf16/fp16)、損失、逆伝播
    3. 勾配の大きさを grad_clip までに抑える
    4. AdamW で更新(重み減衰は行列だけにかけ、正規化の係数にはかけない)
log_every ステップごと : log.csv に1行(損失、パープレキシティ、学習率、トークン/秒、GPU メモリ)
eval_every ステップごと: 検証用の損失、決まった書き出しの続き(samples.md)、
                         チェックポイント last.pt、検証の損失が良くなったら best.pt
再開: last.pt にモデル・最適化・スケーラー・ステップ数・乱数の状態・バッチの乱数の状態を保存するので、
再開しても中断しなかった場合と同じバッチと学習率で続き、損失の曲線は途切れない。
安全な停止: Ctrl+C を1回押すと、今のステップを終えて last.pt を保存してから終わる。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from src.batching import RandomWindows, open_tokens, validation_batches, validation_windows
from src.model import ModelA, ModelConfig
from src.token_data import read_meta

CHECKPOINT_FORMAT: str = "model-a-checkpoint-v1"
LOG_COLUMNS: tuple[str, ...] = ("step", "event", "tokens_seen", "lr", "train_loss", "train_ppl", "val_loss",
                                "val_ppl", "grad_norm", "tokens_per_sec", "peak_vram_mb", "elapsed_s")
DEFAULT_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "大阪支社で", "社員の平均")
# Settings that may differ between the first run and a resumed run (they do not change the batches or the curve).
RESUME_FREE_KEYS: frozenset[str] = frozenset({"stop_after", "resume", "device", "machine", "overwrite"})


@dataclass
class TrainConfig:
    run_dir: str
    model_config: str = "configs/model_tiny.json"
    tokens_dir: str = "data/tokens"
    tokenizer: str | None = "data/tokenizer/tokenizer.json"
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 600
    warmup_steps: int = 60
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float | None = None          # None = keep the value in the model config
    log_every: int = 10
    eval_every: int = 100
    eval_max_windows: int | None = None   # None = the whole validation stream
    sample_tokens: int = 40
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    precision: str = "auto"               # auto | bf16 | fp16 | fp32
    device: str = "auto"
    seed: int = 42
    stop_after: int | None = None         # stop at this step (simulates an interruption)
    resume: bool = False
    overwrite: bool = False
    machine: str = "home"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- small pieces


def lr_at(step: int, max_lr: float, warmup: int, max_steps: int, min_ratio: float) -> float:
    """Learning rate for the step about to be taken (step counts from 0)."""
    min_lr = max_lr * min_ratio
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on matrices (embedding, attention, MLP); none on vectors (norm gains)."""
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return torch.device(choice)


def pick_precision(choice: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if choice == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if choice == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("this GPU does not support bf16; use --precision fp16")
    return choice


def autocast_ctx(device: torch.device, precision: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def evaluate(model: ModelA, tokens, block_size: int, batch_size: int, device: torch.device, precision: str,
             max_windows: int | None) -> float:
    """Average next-token loss over the fixed validation windows."""
    model.eval()
    total, count = 0.0, 0
    for x, y in validation_batches(tokens, block_size, batch_size, max_windows):
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device, precision):
            _, loss = model(x, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    model.train()
    return total / max(1, count)


@torch.no_grad()
def greedy_continue(model: ModelA, ids: list[int], max_new: int, eos_id: int, device: torch.device) -> list[int]:
    """Always take the highest-scoring next token (Day 10 adds proper sampling)."""
    model.eval()
    out = list(ids)
    for _ in range(max_new):
        x = torch.tensor([out[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt == eos_id:
            break
    model.train()
    return out[len(ids):]


def save_checkpoint(path: Path, state: dict) -> None:
    """Write to a temporary file, then rename: an interruption never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a Model A checkpoint")
    return state


class _StopFlag:
    """Ctrl+C once = stop after the current step and save; twice = stop immediately."""

    def __init__(self) -> None:
        self.requested = False
        self._old = None

    def __enter__(self) -> "_StopFlag":
        try:
            self._old = signal.signal(signal.SIGINT, self._handler)
        except ValueError:          # not in the main thread (for example inside some test runners)
            self._old = None
        return self

    def _handler(self, signum, frame) -> None:  # noqa: ARG002
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\nCtrl+C: finishing this step, saving last.pt, then stopping (press again to stop at once).")

    def __exit__(self, *exc) -> None:
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)


def _truncate_after(log_path: Path, samples_path: Path, step: int) -> None:
    """If the program was killed (not stopped with Ctrl+C), log.csv and samples.md may contain steps after the
    last checkpoint. Those steps will be trained again, so their old lines are removed before resuming."""
    if log_path.exists():
        with log_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        keep = [r for r in rows if int(r["step"]) <= step]
        if len(keep) != len(rows):
            with log_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                w.writeheader()
                w.writerows(keep)
            print(f"Removed {len(rows) - len(keep)} log rows after step {step} (they will be trained again)")
    if samples_path.exists():
        text = samples_path.read_text(encoding="utf-8")
        parts = text.split("\n## step ")
        kept = [parts[0]] + [p for p in parts[1:] if int(p.split(" ", 1)[0]) <= step]
        if len(kept) != len(parts):
            samples_path.write_text("\n## step ".join(kept), encoding="utf-8", newline="\n")


def _fmt(v: float | None, digits: int = 4) -> str:
    return "" if v is None else f"{v:.{digits}f}"


# --------------------------------------------------------------------------- the loop


def train(cfg: TrainConfig, repo_root: Path) -> dict:
    """Train (or resume) one run. Returns a short summary."""
    root = repo_root
    run_dir = (root / cfg.run_dir).resolve()
    tokens_dir = (root / cfg.tokens_dir).resolve()
    device = pick_device(cfg.device)
    precision = pick_precision(cfg.precision, device)

    # ---- model settings and data, checked against each other
    mcfg = ModelConfig.load((root / cfg.model_config).resolve())
    if cfg.dropout is not None:
        mcfg.dropout = cfg.dropout
    train_meta = read_meta(tokens_dir, "train")
    if train_meta.vocab_size != mcfg.vocab_size:
        raise SystemExit("model config and token files use different vocabulary sizes")
    tok = None
    if cfg.tokenizer:
        from src.bpe_tokenizer import ByteBPETokenizer
        tok_path = (root / cfg.tokenizer).resolve()
        if sha256_file(tok_path) != train_meta.tokenizer_sha256:
            raise SystemExit("data/tokens was made with a different tokenizer; rerun Day 7")
        tok = ByteBPETokenizer.load(tok_path)
    train_tokens = open_tokens(tokens_dir, "train")
    val_tokens = open_tokens(tokens_dir, "val")

    # ---- run folder: new run or resume
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    log_path, samples_path = run_dir / "log.csv", run_dir / "samples.md"
    if cfg.resume:
        if not last_path.exists():
            raise SystemExit(f"nothing to resume: {last_path} does not exist")
        saved = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        diff = [k for k, v in cfg.to_dict().items() if k not in RESUME_FREE_KEYS and saved.get(k) != v]
        if diff:
            raise SystemExit(f"resume refused: settings differ from the first run: {diff}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()) and not cfg.overwrite:
            raise SystemExit(f"{run_dir} already has files. Use a new --run-name, --resume, or --overwrite.")
        run_dir.mkdir(parents=True, exist_ok=True)
        for p in (last_path, best_path, log_path, samples_path):
            p.unlink(missing_ok=True)
        (run_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
                                                   encoding="utf-8", newline="\n")
        mcfg.save(run_dir / "model_config.json")

    # ---- build everything
    torch.manual_seed(cfg.seed)
    model = ModelA(mcfg).to(device)
    use_fused = device.type == "cuda"
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            fused=use_fused)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))
    windows = RandomWindows(train_tokens, mcfg.block_size, cfg.batch_size, cfg.seed)

    step, best_val, elapsed_before, tokens_seen = 0, float("inf"), 0.0, 0
    if cfg.resume:
        st = load_checkpoint(last_path, device)
        if st["model_config"] != mcfg.to_dict():
            raise SystemExit("resume refused: the model settings differ from the checkpoint")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        windows.load_state_dict(st["windows"])
        torch.set_rng_state(st["torch_rng"].cpu())
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        step, best_val = st["step"], st["best_val"]
        elapsed_before, tokens_seen = st["elapsed_s"], st["tokens_seen"]
        _truncate_after(log_path, samples_path, step)
        print(f"Resumed from {last_path.name} at step {step} (best validation loss so far {best_val:.4f})")

    tokens_per_step = cfg.batch_size * cfg.grad_accum * mcfg.block_size
    n_val = validation_windows(val_tokens, mcfg.block_size)
    if cfg.eval_max_windows is not None:
        n_val = min(n_val, cfg.eval_max_windows)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device {device_name} | precision {precision} | model {mcfg.name} | "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Batch {cfg.batch_size} x accum {cfg.grad_accum} x {mcfg.block_size} tokens = {tokens_per_step:,} "
          f"tokens/step | steps {cfg.max_steps} | validation windows {n_val}")

    def state_for_save() -> dict:
        return {
            "format": CHECKPOINT_FORMAT, "step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(), "windows": windows.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "model_config": mcfg.to_dict(), "train_config": cfg.to_dict(), "best_val": best_val,
            "elapsed_s": elapsed_before + (time.perf_counter() - t_start), "tokens_seen": tokens_seen,
            "tokenizer_sha256": train_meta.tokenizer_sha256, "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
        }

    new_log = not log_path.exists()
    log_f = log_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(log_f, fieldnames=LOG_COLUMNS)
    if new_log:
        writer.writeheader()

    def write_row(event: str, lr: float, train_loss: float | None, val_loss: float | None,
                  grad_norm: float | None, tps: float | None) -> None:
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        writer.writerow({
            "step": step, "event": event, "tokens_seen": tokens_seen, "lr": f"{lr:.6g}",
            "train_loss": _fmt(train_loss), "train_ppl": _fmt(math.exp(train_loss) if train_loss is not None else None, 2),
            "val_loss": _fmt(val_loss), "val_ppl": _fmt(math.exp(val_loss) if val_loss is not None else None, 2),
            "grad_norm": _fmt(grad_norm), "tokens_per_sec": _fmt(tps, 0), "peak_vram_mb": _fmt(peak, 0),
            "elapsed_s": f"{elapsed_before + time.perf_counter() - t_start:.1f}",
        })
        log_f.flush()

    def checkpoint_and_sample(lr: float) -> float:
        nonlocal best_val
        val = evaluate(model, val_tokens, mcfg.block_size, cfg.batch_size, device, precision, cfg.eval_max_windows)
        improved = val < best_val
        best_val = min(best_val, val)
        write_row("eval", lr, None, val, None, None)
        if tok is not None and cfg.prompts:
            lines = [f"\n## step {step} (validation loss {val:.4f})\n", "| Prompt | Greedy continuation |",
                     "| --- | --- |"]
            for p in cfg.prompts:
                cont = greedy_continue(model, [1] + tok.encode(p), cfg.sample_tokens, 2, device)
                text = tok.decode(cont).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {p} | {text}{' ⟨eos⟩' if cont and cont[-1] == 2 else ''} |")
            with samples_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
        state = state_for_save()
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        print(f"step {step:>6} | validation loss {val:.4f} (perplexity {math.exp(val):.2f})"
              f"{'  * best, saved best.pt' if improved else ''} | saved last.pt")
        return val

    t_start = time.perf_counter()
    model.train()
    interval_loss = torch.zeros((), device=device)
    interval_steps, interval_t0 = 0, time.perf_counter()
    stopped_early = False
    last_lr = cfg.lr
    try:
        with _StopFlag() as stop:
            if step == 0 and not cfg.resume:
                checkpoint_and_sample(lr_at(0, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio))
                interval_t0 = time.perf_counter()
            while step < cfg.max_steps:
                lr = lr_at(step, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
                last_lr = lr
                for g in opt.param_groups:
                    g["lr"] = lr
                step_loss = torch.zeros((), device=device)
                for _ in range(cfg.grad_accum):
                    x, y = windows.next_batch()
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with autocast_ctx(device, precision):
                        _, loss = model(x, y)
                    step_loss += loss.detach() / cfg.grad_accum
                    scaler.scale(loss / cfg.grad_accum).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                tokens_seen += tokens_per_step
                interval_loss += step_loss
                interval_steps += 1

                if step % cfg.log_every == 0 or step == cfg.max_steps:
                    mean_loss = (interval_loss / interval_steps).item()
                    if not math.isfinite(mean_loss):
                        save_checkpoint(run_dir / "diverged.pt", state_for_save())
                        raise SystemExit(f"loss became {mean_loss} at step {step}; saved diverged.pt and stopped")
                    dt = time.perf_counter() - interval_t0
                    tps = interval_steps * tokens_per_step / dt if dt > 0 else None
                    write_row("train", lr, mean_loss, None, float(grad_norm), tps)
                    print(f"step {step:>6} | loss {mean_loss:.4f} | lr {lr:.2e} | "
                          f"{(tps or 0):,.0f} tokens/s")
                    interval_loss.zero_()
                    interval_steps, interval_t0 = 0, time.perf_counter()

                stop_now = stop.requested or (cfg.stop_after is not None and step >= cfg.stop_after)
                if stop_now and interval_steps > 0:      # log the unfinished interval before stopping
                    mean_loss = (interval_loss / interval_steps).item()
                    write_row("train", lr, mean_loss, None, float(grad_norm), None)
                    interval_loss.zero_()
                    interval_steps = 0
                if step % cfg.eval_every == 0 or step == cfg.max_steps:
                    checkpoint_and_sample(lr)
                    interval_t0 = time.perf_counter()
                elif stop_now:
                    save_checkpoint(last_path, state_for_save())
                    print(f"step {step:>6} | saved last.pt")
                if stop_now and step < cfg.max_steps:
                    stopped_early = True
                    print(f"Stopped at step {step}. Continue with --resume.")
                    break
    finally:
        log_f.close()

    return {"run_dir": str(run_dir), "step": step, "max_steps": cfg.max_steps, "best_val": best_val,
            "stopped_early": stopped_early, "device": device_name, "precision": precision, "last_lr": last_lr,
            "elapsed_s": elapsed_before + time.perf_counter() - t_start}
