r"""Day 9 - train Model A (or resume a run). The same script is used for the size trials and the main run.
Day 9 - Model A を学習する(または学習を再開する)。大きさの試験と本番の学習でも同じスクリプトを使う。

==========================  ENGLISH  ==========================

A run lives in runs/day9/<run-name>/ (git-ignored):
    train_config.json, model_config.json   the settings (a resumed run must match them)
    log.csv                                 one row per log/eval step
    samples.md                              greedy continuations of fixed prompts at every evaluation
    last.pt, best.pt                        checkpoints (latest, and lowest validation loss)

Usage (VS Code terminal, repo root):
    # a new run
    .venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_a --max-steps 600
    # stop early on purpose (simulates an interruption), then resume
    .venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --max-steps 600 --stop-after 300
    .venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --resume
Ctrl+C during training: finishes the step, saves last.pt, stops. Resume with --resume.

A resumed run takes all settings from the run's train_config.json; only
--stop-after, --machine and --device may be given again.

SERVER NOTE (Linux, later): same code;  .venv/bin/python scripts/day9_train.py --machine server ...
On a GPU without bf16 (RTX 2060) precision "auto" chooses fp16 with a gradient scaler.

==========================  日本語  ==========================

1つの学習は runs/day9/<run-name>/ に保存される(Git には入れない):
    train_config.json, model_config.json   設定(再開するときは同じでなければならない)
    log.csv                                 記録とテストのたびに1行
    samples.md                              評価のたびに、決まった書き出しの続きを書く
    last.pt, best.pt                        チェックポイント(最新と、検証の損失が最も低いもの)

使い方は上の英語の欄と同じ(VS Code ターミナル、リポジトリのルートで)。
学習中に Ctrl+C: そのステップを終えて last.pt を保存して止まる。--resume で再開する。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day9-v1"


def parse_args() -> argparse.Namespace:
    d = TrainConfig(run_dir="")
    ap = argparse.ArgumentParser(description="Day 9: train or resume Model A.")
    ap.add_argument("--run-name", required=True, help="folder name under --runs-root")
    ap.add_argument("--runs-root", default="runs/day9")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--overwrite", action="store_true", help="start again in an existing run folder")
    g = ap.add_argument_group("settings of a new run (ignored with --resume)")
    g.add_argument("--model-config", default=d.model_config)
    g.add_argument("--batch-size", type=int, default=d.batch_size)
    g.add_argument("--grad-accum", type=int, default=d.grad_accum)
    g.add_argument("--max-steps", type=int, default=d.max_steps)
    g.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    g.add_argument("--lr", type=float, default=d.lr)
    g.add_argument("--min-lr-ratio", type=float, default=d.min_lr_ratio)
    g.add_argument("--weight-decay", type=float, default=d.weight_decay)
    g.add_argument("--grad-clip", type=float, default=d.grad_clip)
    g.add_argument("--dropout", type=float, default=None)
    g.add_argument("--log-every", type=int, default=d.log_every)
    g.add_argument("--eval-every", type=int, default=d.eval_every)
    g.add_argument("--eval-max-windows", type=int, default=None)
    g.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default=d.precision)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--prompt", action="append", help="prompt for the samples (repeatable)")
    return ap.parse_args()


def build_config(a: argparse.Namespace) -> TrainConfig:
    run_dir = f"{a.runs_root}/{a.run_name}"
    if a.resume:
        saved_path = REPO_ROOT / run_dir / "train_config.json"
        if not saved_path.exists():
            raise SystemExit(f"nothing to resume: {saved_path} does not exist")
        cfg = TrainConfig.from_dict(json.loads(saved_path.read_text(encoding="utf-8")))
        cfg.resume, cfg.stop_after, cfg.machine, cfg.device, cfg.overwrite = True, a.stop_after, a.machine, a.device, False
        return cfg
    cfg = TrainConfig(
        run_dir=run_dir, model_config=a.model_config, batch_size=a.batch_size, grad_accum=a.grad_accum,
        max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, min_lr_ratio=a.min_lr_ratio,
        weight_decay=a.weight_decay, grad_clip=a.grad_clip, dropout=a.dropout, log_every=a.log_every,
        eval_every=a.eval_every, eval_max_windows=a.eval_max_windows, precision=a.precision, device=a.device,
        seed=a.seed, stop_after=a.stop_after, machine=a.machine, overwrite=a.overwrite,
    )
    if a.prompt:
        cfg.prompts = a.prompt
    return cfg


def append_run_log(summary: dict, cfg: TrainConfig) -> None:
    status = "stopped early (resume later)" if summary["stopped_early"] else "finished"
    body = [
        f"- Machine label: {cfg.machine}; device: {summary['device']}; precision: {summary['precision']}",
        f"- Script: scripts/day9_train.py ({SCRIPT_VERSION}); PyTorch {torch.__version__}",
        f"- Run: {cfg.run_dir}; model config {cfg.model_config}; {'resumed' if cfg.resume else 'new run'}",
        f"- Step {summary['step']} of {summary['max_steps']} ({status}); best validation loss "
        f"{summary['best_val']:.4f}; total training time {summary['elapsed_s']:.0f} s",
    ]
    with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {cfg.machine} | Day 9 train {Path(cfg.run_dir).name}\n\n"
                + "\n".join(body) + "\n")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    cfg = build_config(a)
    print(f"=== Day 9 train | run {cfg.run_dir} | machine={cfg.machine} | {'resume' if cfg.resume else 'new run'} ===")
    summary = train(cfg, REPO_ROOT)
    print(f"\n{'Stopped early' if summary['stopped_early'] else 'Finished'} at step {summary['step']} of "
          f"{summary['max_steps']}; best validation loss {summary['best_val']:.4f}")
    print(f"Run folder: {summary['run_dir']}")
    append_run_log(summary, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
