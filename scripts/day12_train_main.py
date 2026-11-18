r"""Day 12 - the main training run: the Day 11 configuration, early stopping, three seeds, one final model.
Day 12 - 本番の学習: Day 11 で決めた設定、早期終了、3つの乱数の種、最終のモデル1つ

==========================  ENGLISH  ==========================

Reads configs/day11_decision.json (size, dropout) and trains that configuration with:
    - 3 seeds (42, 43, 44): Day 11 showed that one seed can be off by up to 0.06 in
      validation loss, so the result is reported as a mean and a spread, not one number;
    - a 400-step schedule (the learning rate has decayed by the time Day 11's best point
      (steps 125-225) is reached), warm-up 60, peak learning rate 0.001;
    - validation every 10 steps and early stopping after 100 steps without improvement.
The FINAL model is the seed with the lowest validation loss. The test set is not used
here at all; it is kept for the final evaluation (Day 13).

Then each seed's best checkpoint is checked on the 12 fixed prompts (task 6 items), the
result is compared with the Day 11 trials, and these are written:
    runs/day12/main_s<seed>/           one run folder per seed (git-ignored)
    docs/DAY12_MAIN_TRAINING.md        results, comparison, sample sentences
    docs/figures/day12_main.png        training and validation curves of the 3 seeds
    configs/day12_final.json           which checkpoint is the final model (Day 13 reads it)

Resumable: run the same command again after Ctrl+C or a crash.

Usage (VS Code terminal, repo root):
    .venv\Scripts\python.exe scripts\day12_train_main.py --machine home
Watch a run from a second terminal (optional):
    .venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5

SERVER NOTE (Linux, later): same code;  .venv/bin/python scripts/day12_train_main.py --machine server

==========================  日本語  ==========================

configs/day11_decision.json(大きさとドロップアウト)を読み、その設定を次の条件で学習させる:
    - 3つの乱数の種(42、43、44): Day 11 で、種1つでは検証の損失が最大0.06ずれると分かったので、
      結果を1つの数ではなく、平均とばらつきで示す
    - 400ステップの予定(Day 11 の一番良い点 125〜225 ステップの頃には学習率が下がっている)
    - 10ステップごとに検証し、100ステップ改善しなければ早期終了する
最終のモデルは、検証の損失が一番低い種のもの。テスト用のデータはここでは使わず、
最後の評価(Day 13)のために取っておく。
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.generation import PRESET_SETTINGS, load_model, model_logits_fn, quick_checks  # noqa: E402
from src.model import ModelConfig, expected_parameters  # noqa: E402
from src.selection import MAX_REPETITIVE, MIN_FINISHED, RunSummary, summarize_run  # noqa: E402
from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day12-v1"
KEY_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "勤続年数が", "大阪支社で", "課長の", "今日は")


def read_prompts(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def run_status(run_dir: Path) -> str:
    """'new', 'done' (finished or early-stopped) or 'unfinished' (stopped by Ctrl+C or killed)."""
    status = run_dir / "status.json"
    if status.exists() and json.loads(status.read_text(encoding="utf-8"))["status"] in ("finished", "early_stopped"):
        return "done"
    return "unfinished" if (run_dir / "last.pt").exists() else "new"


def draw_figure(runs: list[RunSummary], runs_root: Path, path: Path) -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for c, r in zip(colours, runs):
        with (runs_root / r.name / "log.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        ev = [(int(x["step"]), float(x["val_loss"])) for x in rows if x["event"] == "eval" and x["val_loss"]
              and int(x["step"]) > 0]
        tr = [(int(x["step"]), float(x["train_loss"])) for x in rows if x["event"] == "train" and x["train_loss"]]
        ax.plot([s for s, _ in tr], [v for _, v in tr], color=c, lw=1.0, alpha=0.45, ls="--")
        ax.plot([s for s, _ in ev], [v for _, v in ev], color=c, lw=1.8, label=f"{r.name} validation")
        ax.plot([r.best_step], [r.best_val], "o", color=c, ms=6)
    ax.set_ylim(0.8, 4.0)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Day 12 main training: validation (solid) and training (dashed); dots = best.pt")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 12: main training run.")
    ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
    ap.add_argument("--decision", type=Path, default=REPO_ROOT / "configs/day11_decision.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--warmup-steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--eval-max-windows", type=int, default=None, help="(testing only) fewer validation windows")
    ap.add_argument("--runs-root", default="runs/day12")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY12_MAIN_TRAINING.md")
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day12_main.png")
    ap.add_argument("--final", type=Path, default=REPO_ROOT / "configs/day12_final.json")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    decision = json.loads(a.decision.read_text(encoding="utf-8"))
    size, dropout, model_config = decision["size"], float(decision["dropout"]), decision["model_config"]
    runs_root = REPO_ROOT / a.runs_root
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Day 12 main training | {size}, dropout {dropout} (from {a.decision.name}) | seeds {a.seeds} | "
          f"{device_name} ===")

    # ---- 1. train each seed (skip done ones, resume unfinished ones)
    statuses: dict[str, dict] = {}
    for i, seed in enumerate(a.seeds, 1):
        name = f"main_s{seed}"
        run_dir = runs_root / name
        state = run_status(run_dir)
        print(f"\n[{i}/{len(a.seeds)}] {name}: {state}")
        if state != "done":
            cfg = TrainConfig(
                run_dir=f"{a.runs_root}/{name}", model_config=model_config, batch_size=a.batch_size,
                max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, dropout=dropout,
                log_every=a.eval_every, eval_every=a.eval_every, eval_max_windows=a.eval_max_windows,
                early_stop_patience=a.patience, prompts=[], device=a.device, seed=seed, machine=a.machine,
                resume=(state == "unfinished"),
            )
            result = train(cfg, REPO_ROOT)
            if result["stopped_early"]:
                print("\nStopped. Run the same command again to continue from here.")
                return 0
        statuses[name] = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    # ---- 2. summarise and check generation on each best.pt (test set NOT used)
    tok = ByteBPETokenizer.load(REPO_ROOT / "data/tokenizer/tokenizer.json")
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    params = expected_parameters(ModelConfig.load(REPO_ROOT / model_config))
    runs: list[RunSummary] = []
    checks: dict[str, dict] = {}
    print("\n--- generation check on each best.pt ---")
    for seed in a.seeds:
        name = f"main_s{seed}"
        r = summarize_run(runs_root / name / "log.csv", name, size, dropout, params)
        model, _ = load_model(runs_root / name / "best.pt", device)
        fn = model_logits_fn(model, device)
        checks[name] = {s: quick_checks(fn, tok, prompts, PRESET_SETTINGS[s], train_set)
                        for s in ("greedy+limits", "sample")}
        gl = checks[name]["greedy+limits"]
        r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
        runs.append(r)
        print(f"  {name}: best val {r.best_val:.4f} at step {r.best_step} ({statuses[name]['status']} at step "
              f"{statuses[name]['step']}) | finished {gl['finished']:.0%} | repetitive {gl['repetitive']:.0%} | "
              f"distinct-2 {gl['distinct2']:.2f}")
        del model

    vals = [r.best_val for r in runs]
    mean_val = statistics.mean(vals)
    spread = max(vals) - min(vals)
    passing = [r for r in runs if r.passes_checks()]
    final = min(passing or runs, key=lambda r: r.best_val)
    d11 = [decision.get("best_val_seed42"), decision.get("best_val_seed43")]
    d11 = [v for v in d11 if v is not None]
    d11_mean = statistics.mean(d11) if d11 else None

    print(f"\nBest validation loss per seed: {', '.join(f'{v:.4f}' for v in vals)}")
    print(f"Mean {mean_val:.4f} (perplexity {math.exp(mean_val):.2f}), spread {spread:.4f}")
    if d11_mean is not None:
        print(f"Day 11 trials of the same configuration (800-step schedule): mean {d11_mean:.4f} -> "
              f"change {mean_val - d11_mean:+.4f}")
    print(f"Final model: {final.name} (best val {final.best_val:.4f} at step {final.best_step})"
          + ("" if final.passes_checks() else "  WARNING: does not pass the task 6 checks"))

    # ---- 3. outputs
    draw_figure(runs, runs_root, a.figure)
    final_ckpt = f"{a.runs_root}/{final.name}/best.pt"
    a.final.parent.mkdir(parents=True, exist_ok=True)
    a.final.write_text(json.dumps({
        "final_checkpoint": final_ckpt, "run": final.name, "size": size, "dropout": dropout,
        "model_config": model_config, "parameters": params, "best_val": round(final.best_val, 4),
        "best_step": final.best_step, "seeds": a.seeds, "best_val_per_seed": [round(v, 4) for v in vals],
        "mean_best_val": round(mean_val, 4), "spread": round(spread, 4), "test_set_used": False,
        "decided": f"{datetime.now():%Y-%m-%d %H:%M}", "script": SCRIPT_VERSION,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    write_report(a, runs, statuses, checks, final, mean_val, spread, d11, d11_mean, device_name, size, dropout,
                 params, len(prompts))
    print(f"\nReport: {a.report}\nFigure: {a.figure}\nFinal model file: {a.final}  ->  {final_ckpt}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {a.machine} | Day 12 main training\n\n"
                    f"- Script: scripts/day12_train_main.py ({SCRIPT_VERSION}); {device_name}; PyTorch {torch.__version__}\n"
                    f"- Configuration: {size}, dropout {dropout}; {a.max_steps}-step schedule, patience {a.patience}\n"
                    + "".join(f"- {r.name}: best val {r.best_val:.4f} at step {r.best_step}; "
                              f"{statuses[r.name]['status']} at step {statuses[r.name]['step']}\n" for r in runs)
                    + f"- Mean {mean_val:.4f}, spread {spread:.4f}; final model {final_ckpt}\n")
        print("Appended an entry to run_log.md")
    return 0


def _cell(text: str) -> str:
    """Make a generated sentence safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def write_report(a, runs, statuses, checks, final, mean_val, spread, d11, d11_mean, device_name, size, dropout,
                 params, n_prompts) -> None:
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
    gl_final = checks[final.name]["greedy+limits"]
    md = [
        "# Day 12 report: main training run",
        "",
        f"Generated by `scripts/day12_train_main.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}. "
        f"Machine: {a.machine}, {device_name}, PyTorch {torch.__version__}, Python {platform.python_version()}.",
        "",
        f"Configuration from Day 11: **{size}, dropout {dropout}** ({params:,} parameters). Batch {a.batch_size} x 256 "
        f"tokens, {a.max_steps}-step schedule (warm-up {a.warmup_steps}, peak learning rate {a.lr}, cosine to 10 %), "
        f"validation every {a.eval_every} steps, early stopping after {a.patience} steps without improvement, "
        f"seeds {', '.join(map(str, a.seeds))}. The test set was not used.",
        "",
        "## Final model",
        "",
        f"**`{a.runs_root}/{final.name}/best.pt`**: best validation loss {final.best_val:.4f} "
        f"(perplexity {math.exp(final.best_val):.2f}) at step {final.best_step}. "
        f"On the {n_prompts} fixed prompts (greedy + limits): finished {gl_final['finished']:.0%}, "
        f"repetitive {gl_final['repetitive']:.0%}, distinct-2 {gl_final['distinct2']:.2f} "
        f"(task 6 guidelines: finished >= {MIN_FINISHED:.0%}, repetitive <= {MAX_REPETITIVE:.0%}).",
        "",
        f"![training and validation curves]({fig_rel})",
        "",
        "## All seeds",
        "",
        "| Run | How it ended | Best val loss | Perplexity | At step | Train loss there | Gap | Finished | Repetitive | "
        "distinct-2 | New sentences |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(f"| {r.name} | {statuses[r.name]['status']} at step {statuses[r.name]['step']} | {r.best_val:.4f} | "
          f"{math.exp(r.best_val):.2f} | {r.best_step} | "
          f"{f'{r.train_at_best:.4f}' if r.train_at_best is not None else '-'} | "
          f"{f'{r.gap:.3f}' if r.gap is not None else '-'} | {r.finished_rate:.0%} | {r.repetitive_rate:.0%} | "
          f"{r.distinct2:.2f} | {checks[r.name]['greedy+limits']['new']}/"
          f"{round(r.finished_rate * n_prompts)} |" for r in runs),
        "",
        f"Mean best validation loss {mean_val:.4f} (perplexity {math.exp(mean_val):.2f}); spread between seeds "
        f"{spread:.4f}.",
        "",
        "## Compared with the Day 11 trials of the same configuration",
        "",
        "| | Schedule | Seeds | Mean best validation loss |",
        "| --- | --- | --- | --- |",
        f"| Day 11 trials | 800 steps | {len(d11)} | {f'{d11_mean:.4f}' if d11_mean is not None else '-'} |",
        f"| Day 12 main | {a.max_steps} steps + early stopping | {len(runs)} | {mean_val:.4f} |",
        "",
        (f"Change: {mean_val - d11_mean:+.4f}. Read it against the seed spread above: a change smaller than the spread "
         "is not evidence of improvement." if d11_mean is not None else ""),
        "",
        "## Sample sentences (final model)",
        "",
        "| Beginning | Greedy + limits | Sampling (T 0.8, top-k 50, top-p 0.95) |",
        "| --- | --- | --- |",
        *(f"| {p} | {_cell(checks[final.name]['greedy+limits']['texts'][p])} | "
          f"{_cell(checks[final.name]['sample']['texts'][p])} |"
          for p in KEY_PROMPTS if p in checks[final.name]["sample"]["texts"]),
        "",
    ]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
