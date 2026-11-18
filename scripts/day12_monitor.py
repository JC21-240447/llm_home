r"""Day 12 - monitor a training run from its log: progress, best point, early-stopping counter, time left.
Day 12 - 学習の記録から、進み具合・一番良い点・早期終了までの残り・残り時間を表示する

==========================  ENGLISH  ==========================

Reads runs/.../log.csv and train_config.json (it never touches the training itself, so it
is safe to run while training is going on, from a second terminal).

Usage (VS Code terminal, repo root; open a second terminal with the + button):
    .venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42
    .venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5    (refresh every 5 s; Ctrl+C to quit)

SERVER NOTE: same code;  .venv/bin/python scripts/day12_monitor.py --run runs/day12/main_s42 --watch 5

==========================  日本語  ==========================

runs/.../log.csv と train_config.json を読むだけで、学習そのものには触れないので、学習中に
2つ目のターミナルから安全に実行できる。--watch 5 で5秒ごとに更新(Ctrl+C で終わる)。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def summary(run_dir: Path) -> str:
    log, cfg_path = run_dir / "log.csv", run_dir / "train_config.json"
    if not log.exists() or not cfg_path.exists():
        return f"{run_dir}: no log yet"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    with log.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    evals = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["event"] == "eval" and r["val_loss"]]
    trains = [r for r in rows if r["event"] == "train" and r["train_loss"]]
    if not evals:
        return f"{run_dir}: started, no validation yet"
    step = max(int(r["step"]) for r in rows)
    best_step, best_val = min(evals, key=lambda e: (e[1], e[0]))
    last_val_step, last_val = evals[-1]
    tps = [float(r["tokens_per_sec"]) for r in trains[-5:] if r["tokens_per_sec"]]
    tokens_per_step = cfg["batch_size"] * cfg.get("grad_accum", 1) * 256
    left = cfg["max_steps"] - step
    eta = left * tokens_per_step / (sum(tps) / len(tps)) if tps and left > 0 else 0.0
    patience = cfg.get("early_stop_patience")
    status_file = run_dir / "status.json"
    status = json.loads(status_file.read_text(encoding="utf-8"))["status"] if status_file.exists() else "running"
    bar_n = int(30 * step / max(1, cfg["max_steps"]))
    lines = [
        f"Run {run_dir.relative_to(REPO_ROOT).as_posix() if run_dir.is_relative_to(REPO_ROOT) else run_dir}  [{status}]",
        f"  progress   [{'#' * bar_n}{'.' * (30 - bar_n)}] step {step} / {cfg['max_steps']}",
        f"  train loss {float(trains[-1]['train_loss']):.4f} (last)" if trains else "  train loss -",
        f"  val loss   {last_val:.4f} at step {last_val_step}   |   best {best_val:.4f} "
        f"(perplexity {math.exp(best_val):.2f}) at step {best_step}",
    ]
    if patience:
        waited = last_val_step - best_step
        lines.append(f"  early stop {waited} / {patience} steps without improvement"
                     + ("  -> will stop at the next check" if waited >= patience else ""))
    if tps:
        lines.append(f"  speed      {sum(tps) / len(tps):,.0f} tokens/s   |   time left at most {eta:,.0f} s "
                     f"(less if early stopping triggers)")
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Day 12: monitor a training run.")
    ap.add_argument("--run", required=True, help="run folder, for example runs/day12/main_s42")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds (0 = print once)")
    a = ap.parse_args()
    run_dir = (REPO_ROOT / a.run).resolve()
    if not a.watch:
        print(summary(run_dir))
        return 0
    try:
        while True:
            print("\033[2J\033[H" + summary(run_dir), flush=True)   # clear the screen, then print
            time.sleep(a.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
