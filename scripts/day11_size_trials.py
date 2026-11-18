r"""Day 11 - size trials: train every (size, dropout) pair on the same budget, check, and choose by a written rule.
Day 11 - 大きさの試験: すべての(大きさ, ドロップアウト)を同じ量だけ学習させ、確かめ、決めたルールで選ぶ

==========================  ENGLISH  ==========================

Trials (6 runs by default): sizes tiny / small / medium (Day 8 configs) x dropout 0.0 / 0.1.
Every run uses the SAME settings, so only size and dropout differ:
    batch 32 x 256 tokens, 800 steps (about 3 passes over the training text), warm-up 60,
    peak learning rate 0.001, seed 42, validation every 25 steps (Day 9 showed the best
    point can come within the first few hundred steps, so checks must be frequent).
For each run: the best checkpoint (lowest validation loss) is also checked on the
12 fixed Day 10 prompts (greedy + repetition limits, and sampling) with the task 6
automatic items. Then src/selection.py applies the decision rule written down in advance.

Outputs:
    runs/day11/<size>_d<dropout>/        one Day 9 run folder per trial (git-ignored)
    docs/DAY11_SIZE_TRIALS.md            table, decision, sample sentences
    docs/figures/day11_val_loss.png      validation curves of all runs
    configs/day11_decision.json          the chosen size and dropout (Day 12 reads it)
    run_log.md                           one entry

Resumable: run the same command again after Ctrl+C or a crash. Finished runs are
skipped, an unfinished run continues from its last.pt.

Usage (VS Code terminal, repo root):
    .venv\Scripts\python.exe scripts\day11_size_trials.py --machine home

SERVER NOTE (Linux, later): same code;  .venv/bin/python scripts/day11_size_trials.py --machine server

==========================  日本語  ==========================

試験(初期設定で6つ): 大きさ tiny / small / medium × ドロップアウト 0.0 / 0.1。
すべて同じ設定で学習するので、違うのは大きさとドロップアウトだけ。検証は25ステップごと
(Day 9 で、一番良い点は最初の数百ステップで来ることが分かったため)。
各試験の一番良いチェックポイントを、Day 10 の12の書き出しで、作業6の自動の確認にかける。
その後、src/selection.py が、前もって決めたルールで選ぶ。
途中で止めても、同じコマンドをもう一度実行すれば続きから再開する。
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import EOS_ID, ByteBPETokenizer  # noqa: E402
from src.generation import (  # noqa: E402
    PRESET_SETTINGS,
    distinct_2,
    generate,
    is_repetitive,
    load_model,
    model_logits_fn,
)
from src.model import ModelConfig, expected_parameters  # noqa: E402
from src.selection import TOLERANCE, RunSummary, choose, summarize_run  # noqa: E402
from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day11-v1"
KEY_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "勤続年数が", "今日は")


def run_name(size: str, dropout: float) -> str:
    return f"{size}_d{int(round(dropout * 100)):02d}"


def read_prompts(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def run_state(run_dir: Path, max_steps: int) -> str:
    """'new', 'finished' or 'unfinished' (judged from the log: the last evaluation must be at max_steps)."""
    log = run_dir / "log.csv"
    if not (run_dir / "last.pt").exists() or not log.exists():
        return "new"
    steps = [int(line.split(",", 1)[0]) for line in log.read_text(encoding="utf-8").splitlines()[1:]
             if ",eval," in line]
    return "finished" if steps and steps[-1] >= max_steps else "unfinished"


def generation_check(ckpt: Path, tok: ByteBPETokenizer, prompts: list[str], device: torch.device,
                     train_set: set[str]) -> dict:
    model, _ = load_model(ckpt, device)
    fn = model_logits_fn(model, device)
    out: dict = {}
    for setting in ("greedy+limits", "sample"):
        gens = [generate(fn, tok, p, PRESET_SETTINGS[setting]) for p in prompts]
        n = len(gens)
        out[setting] = {
            "finished": sum(g.finished for g in gens) / n,
            "repetitive": sum(is_repetitive(tok.encode(g.prompt) + [i for i in g.ids if i != EOS_ID])
                              for g in gens) / n,
            "distinct2": distinct_2([[i for i in g.ids if i != EOS_ID] for g in gens]),
            "new": sum((g.prompt + g.text) not in train_set for g in gens if g.finished),
            "texts": {g.prompt: (g.text + (" ⟨eos⟩" if g.finished else " …")) for g in gens},
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def draw_figure(runs: list[RunSummary], root: Path, path: Path) -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"tiny": "#1f77b4", "small": "#2ca02c", "medium": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for r in runs:
        with (root / r.name / "log.csv").open(encoding="utf-8", newline="") as f:
            ev = [(int(x["step"]), float(x["val_loss"])) for x in csv.DictReader(f)
                  if x["event"] == "eval" and x["val_loss"] and int(x["step"]) > 0]
        ax.plot([s for s, _ in ev], [v for _, v in ev], color=colours.get(r.size, "gray"),
                ls="-" if r.dropout == 0 else "--", lw=1.6, label=f"{r.size}, dropout {r.dropout}")
        ax.plot([r.best_step], [r.best_val], "o", color=colours.get(r.size, "gray"), ms=5)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("Day 11 size trials: validation loss (dots = lowest point of each run)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 11: size trials and the size decision.")
    ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
    ap.add_argument("--sizes", nargs="+", default=["tiny", "small", "medium"])
    ap.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.1])
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--warmup-steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--eval-max-windows", type=int, default=None, help="(testing only) fewer validation windows")
    ap.add_argument("--runs-root", default="runs/day11")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY11_SIZE_TRIALS.md")
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day11_val_loss.png")
    ap.add_argument("--decision", type=Path, default=REPO_ROOT / "configs/day11_decision.json")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    runs_root = REPO_ROOT / a.runs_root
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    trials = [(s, d) for s in a.sizes for d in a.dropouts]
    print(f"=== Day 11 size trials | machine={a.machine} | {device_name} | {len(trials)} runs x {a.max_steps} steps ===")

    # ---- 1. train every trial (skip finished ones, resume unfinished ones)
    for i, (size, dropout) in enumerate(trials, 1):
        name = run_name(size, dropout)
        run_dir = runs_root / name
        state = run_state(run_dir, a.max_steps)
        print(f"\n[{i}/{len(trials)}] {name}: {state}")
        if state == "finished":
            continue
        cfg = TrainConfig(
            run_dir=f"{a.runs_root}/{name}", model_config=f"configs/model_{size}.json", batch_size=a.batch_size,
            max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, dropout=dropout,
            log_every=a.eval_every, eval_every=a.eval_every, eval_max_windows=a.eval_max_windows,
            prompts=[], device=a.device, seed=a.seed, machine=a.machine, resume=(state == "unfinished"),
        )
        summary = train(cfg, REPO_ROOT)
        if summary["stopped_early"]:
            print("\nStopped. Run the same command again to continue from here.")
            return 0

    # ---- 2. summarise, check generation on each best checkpoint, decide
    tok = ByteBPETokenizer.load(REPO_ROOT / "data/tokenizer/tokenizer.json")
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    runs: list[RunSummary] = []
    gen: dict[str, dict] = {}
    print("\n--- generation check on each best.pt ---")
    for size, dropout in trials:
        name = run_name(size, dropout)
        params = expected_parameters(ModelConfig.load(REPO_ROOT / f"configs/model_{size}.json"))
        r = summarize_run(runs_root / name / "log.csv", name, size, dropout, params)
        g = generation_check(runs_root / name / "best.pt", tok, prompts, device, train_set)
        gl = g["greedy+limits"]
        r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
        runs.append(r)
        gen[name] = g
        print(f"  {name:<10} best val {r.best_val:.4f} at step {r.best_step:>4} | finished {gl['finished']:.0%} | "
              f"repetitive {gl['repetitive']:.0%} | distinct-2 {gl['distinct2']:.2f}")
    decision = choose(runs)

    # ---- 3. print, figure, report, decision file, run log
    print(f"\n{'run':<10} {'params':>11} {'best val':>9} {'ppl':>6} {'step':>5} {'train@best':>10} "
          f"{'gap':>6} {'final val':>9} {'tok/s':>9} {'peak MB':>8}")
    for r in runs:
        print(f"{r.name:<10} {r.parameters:>11,} {r.best_val:>9.4f} {math.exp(r.best_val):>6.2f} {r.best_step:>5} "
              f"{(r.train_at_best or float('nan')):>10.4f} {(r.gap or float('nan')):>6.3f} {r.final_val:>9.4f} "
              f"{(r.tokens_per_sec or 0):>9,.0f} {(r.peak_vram_mb or 0):>8,.0f}")
    print(f"\nDecision: {decision.chosen.name if decision.chosen else 'NONE'} - {decision.reason}")

    draw_figure(runs, runs_root, a.figure)
    write_report(a, runs, gen, decision, device_name, prompts)
    if decision.chosen:
        c = decision.chosen
        a.decision.parent.mkdir(parents=True, exist_ok=True)
        a.decision.write_text(json.dumps({
            "chosen_run": c.name, "size": c.size, "dropout": c.dropout, "model_config": f"configs/model_{c.size}.json",
            "parameters": c.parameters, "best_val": round(c.best_val, 4), "best_step": c.best_step,
            "tokens_per_step": a.batch_size * 256, "trial_steps": a.max_steps, "tolerance": TOLERANCE,
            "reason": decision.reason, "decided": f"{datetime.now():%Y-%m-%d %H:%M}", "script": SCRIPT_VERSION,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Report: {a.report}\nFigure: {a.figure}\nDecision file: {a.decision}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {a.machine} | Day 11 size trials\n\n"
                    f"- Script: scripts/day11_size_trials.py ({SCRIPT_VERSION}); {device_name}; PyTorch {torch.__version__}\n"
                    f"- Trials: {', '.join(r.name for r in runs)}; {a.max_steps} steps each, batch {a.batch_size} x 256\n"
                    + "".join(f"- {r.name}: best val {r.best_val:.4f} at step {r.best_step}, final {r.final_val:.4f}\n"
                              for r in runs)
                    + f"- Decision: {decision.chosen.name if decision.chosen else 'none'} ({decision.reason})\n")
        print("Appended an entry to run_log.md")
    return 0


def write_report(a: argparse.Namespace, runs: list[RunSummary], gen: dict, decision, device_name: str,
                 prompts: list[str]) -> None:
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
    c = decision.chosen
    md = [
        "# Day 11 report: size trials and the size decision",
        "",
        f"Generated by `scripts/day11_size_trials.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}. "
        f"Machine: {a.machine}, {device_name}, PyTorch {torch.__version__}, Python {platform.python_version()}.",
        "",
        f"Every run: batch {a.batch_size} x 256 tokens, {a.max_steps} steps, warm-up {a.warmup_steps}, peak learning "
        f"rate {a.lr}, seed {a.seed}, validation every {a.eval_every} steps. Only the size and the dropout differ.",
        "",
        "## Decision",
        "",
        f"**Chosen: `{c.name}`** ({c.size}, dropout {c.dropout}, {c.parameters:,} parameters), best validation loss "
        f"{c.best_val:.4f} (perplexity {math.exp(c.best_val):.2f}) at step {c.best_step}." if c else "**No run chosen.**",
        "",
        f"Reason: {decision.reason}.",
        "",
        f"Rule (written before the trials): keep runs whose best checkpoint passes the task 6 automatic guidelines "
        f"(finished >= 90 %, repetitive <= 5 % with greedy + limits on the {len(prompts)} fixed prompts); among them, "
        f"every run within {TOLERANCE} of the lowest validation loss counts as equally good, and the one with the "
        "fewest parameters is chosen.",
        "",
        f"Runs within the tolerance: {', '.join(decision.as_good) or '-'}. "
        f"Rejected by the checks: {', '.join(f'{k} ({v})' for k, v in decision.rejected.items()) or 'none'}.",
        "",
        f"![validation loss of all runs]({fig_rel})",
        "",
        "## Results",
        "",
        "| Run | Parameters | Best val loss | Perplexity | At step | Train loss there | Gap | Final val loss | "
        "Rise after best | Tokens/s | Peak MB | Finished | Repetitive | distinct-2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(f"| {r.name} | {r.parameters:,} | {r.best_val:.4f} | {math.exp(r.best_val):.2f} | {r.best_step} | "
          f"{f'{r.train_at_best:.4f}' if r.train_at_best is not None else '-'} | "
          f"{f'{r.gap:.3f}' if r.gap is not None else '-'} | {r.final_val:.4f} | {r.rise_after_best:+.3f} | "
          f"{f'{r.tokens_per_sec:,.0f}' if r.tokens_per_sec else '-'} | "
          f"{f'{r.peak_vram_mb:,.0f}' if r.peak_vram_mb else '-'} | {r.finished_rate:.0%} | "
          f"{r.repetitive_rate:.0%} | {r.distinct2:.2f} |" for r in runs),
        "",
        "Gap = validation loss minus training loss at the best step (how much harder unseen sentence patterns are). "
        "Rise after best = final validation loss minus the best (how much the run overfitted afterwards). "
        "Finished / Repetitive / distinct-2: task 6 items A2-A4 on the best checkpoint, greedy + limits.",
        "",
        "## Sample sentences from each best checkpoint",
        "",
        "Greedy + repetition limits (`⟨eos⟩` = finished, `…` = cut at 40 tokens):",
        "",
        "| Run | " + " | ".join(KEY_PROMPTS) + " |",
        "| --- | " + " | ".join("---" for _ in KEY_PROMPTS) + " |",
        *(f"| {r.name} | " + " | ".join(gen[r.name]["greedy+limits"]["texts"].get(p, "-").replace("|", "\\|")
                                        for p in KEY_PROMPTS) + " |" for r in runs),
        "",
        "Sampling (temperature 0.8, top-k 50, top-p 0.95, fixed seed):",
        "",
        "| Run | " + " | ".join(KEY_PROMPTS) + " |",
        "| --- | " + " | ".join("---" for _ in KEY_PROMPTS) + " |",
        *(f"| {r.name} | " + " | ".join(gen[r.name]["sample"]["texts"].get(p, "-").replace("|", "\\|")
                                        for p in KEY_PROMPTS) + " |" for r in runs),
        "",
    ]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
