r"""Day 9 - compare an uninterrupted run with a stopped-and-resumed run: the proof that resuming works.
Day 9 - 中断しなかった学習と、止めて再開した学習を比べる(再開が正しく動く証拠)

==========================  ENGLISH  ==========================

Reads runs/day9/<a>/log.csv (uninterrupted) and runs/day9/<b>/log.csv (stopped at
the step saved as stop_after in its train_config.json, then resumed). It checks:
    1. both runs reached the same final step,
    2. the training loss at every logged step is the same in both runs
       (on a GPU the last digits may differ; the difference is reported),
    3. the validation losses are the same,
    4. the loss curve of run b has no jump at the resume step.
It draws docs/figures/day9_smoke_loss.png and writes docs/DAY9_SMOKE_REPORT.md
(with the greedy samples of the first and the last evaluation).

Usage (VS Code terminal, repo root):
    .venv\Scripts\python.exe scripts\day9_compare_runs.py --a smoke_a --b smoke_b

==========================  日本語  ==========================

runs/day9/<a>/log.csv(中断なし)と runs/day9/<b>/log.csv(train_config.json の stop_after で止めてから
再開)を読み、次を確かめる:
    1. 両方が同じ最後のステップまで進んだ
    2. 記録したすべてのステップで、学習の損失が同じ(GPU では最後の桁が違うことがあるので、差を報告する)
    3. 検証の損失が同じ
    4. b の損失の曲線に、再開したステップで飛びが無い
docs/figures/day9_smoke_loss.png を描き、docs/DAY9_SMOKE_REPORT.md を書く。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

SAME_TOL: float = 1e-6        # "identical"
CONTINUOUS_TOL: float = 0.02  # "the same curve" (GPU arithmetic may differ in the last digits)


def read_log(path: Path) -> tuple[dict[int, float], dict[int, float], list[dict]]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    train = {int(r["step"]): float(r["train_loss"]) for r in rows if r["event"] == "train" and r["train_loss"]}
    val = {int(r["step"]): float(r["val_loss"]) for r in rows if r["event"] == "eval" and r["val_loss"]}
    return train, val, rows


def sample_sections(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [s.strip() for s in re.split(r"\n(?=## step )", text) if s.strip().startswith("## step")]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Day 9: compare an uninterrupted and a resumed run.")
    ap.add_argument("--a", required=True, help="run name of the uninterrupted run")
    ap.add_argument("--b", required=True, help="run name of the stopped-and-resumed run")
    ap.add_argument("--runs-root", default="runs/day9")
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day9_smoke_loss.png")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY9_SMOKE_REPORT.md")
    a = ap.parse_args()

    dir_a, dir_b = REPO_ROOT / a.runs_root / a.a, REPO_ROOT / a.runs_root / a.b
    train_a, val_a, rows_a = read_log(dir_a / "log.csv")
    train_b, val_b, _ = read_log(dir_b / "log.csv")
    cfg_a = json.loads((dir_a / "train_config.json").read_text(encoding="utf-8"))
    cfg_b = json.loads((dir_b / "train_config.json").read_text(encoding="utf-8"))
    resume_step = cfg_b.get("stop_after")
    if resume_step is None:
        raise SystemExit(f"{a.b} was not stopped early (no stop_after in its train_config.json)")
    keys = [k for k in cfg_a if k not in {"run_dir", "stop_after", "resume", "overwrite", "machine", "device"}]
    differing = [k for k in keys if cfg_a.get(k) != cfg_b.get(k)]
    if differing:
        raise SystemExit(f"the two runs use different settings: {differing}")

    common = sorted(set(train_a) & set(train_b))
    diffs = [abs(train_a[s] - train_b[s]) for s in common]
    vcommon = sorted(set(val_a) & set(val_b))
    vdiffs = [abs(val_a[s] - val_b[s]) for s in vcommon]
    max_diff = max(diffs + vdiffs) if diffs else float("nan")
    final_a, final_b = max(train_a), max(train_b)
    before = max(s for s in train_b if s <= resume_step)
    after = min(s for s in train_b if s > resume_step)
    jump_b = abs(train_b[after] - train_b[before])
    jump_a = abs(train_a.get(after, math.nan) - train_a.get(before, math.nan))

    checks = {
        "same final step": final_a == final_b == cfg_a["max_steps"],
        "training losses match": max(diffs) < CONTINUOUS_TOL,
        "validation losses match": max(vdiffs) < CONTINUOUS_TOL,
        "no jump at the resume step": abs(jump_b - jump_a) < CONTINUOUS_TOL,
    }
    verdict = "identical" if max_diff < SAME_TOL else ("the same curve" if all(checks.values()) else "DIFFERENT")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(list(train_a), list(train_a.values()), lw=2.4, color="#9db7d5", label=f"training loss, {a.a} (no stop)")
    ax.plot(list(train_b), list(train_b.values()), lw=1.2, ls="--", color="#1f4e79",
            label=f"training loss, {a.b} (stopped and resumed)")
    ax.plot(list(val_a), list(val_a.values()), "o", ms=6, color="#e3a8a0", label=f"validation loss, {a.a}")
    ax.plot(list(val_b), list(val_b.values()), "x", ms=6, color="#a33a2c", label=f"validation loss, {a.b}")
    ax.axvline(resume_step, color="gray", lw=1, ls=":")
    ax.text(resume_step, ax.get_ylim()[1] * 0.97, f"  stopped / resumed at step {resume_step}", va="top",
            fontsize=8, color="gray")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title("Day 9 smoke run: uninterrupted vs stopped and resumed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.figure)
    plt.close(fig)

    # ---- report
    train_meta = json.loads((REPO_ROOT / "data/tokens/train_meta.json").read_text(encoding="utf-8"))
    last_row = [r for r in rows_a if r["event"] == "train"][-1]
    tps = [float(r["tokens_per_sec"]) for r in rows_a if r["event"] == "train" and r["tokens_per_sec"]]
    epochs = int(last_row["tokens_seen"]) / int(train_meta["tokens"])
    secs = sample_sections(dir_a / "samples.md")
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
    lines = [
        "# Day 9 report: smoke run, stop and resume",
        "",
        f"Generated by `scripts/day9_compare_runs.py` on {datetime.now():%Y-%m-%d %H:%M}.",
        f"Runs: `{a.runs_root}/{a.a}` (no stop) and `{a.runs_root}/{a.b}` (stopped at step {resume_step}, then resumed).",
        f"Model config `{cfg_a['model_config']}`; batch {cfg_a['batch_size']} x accum {cfg_a['grad_accum']}; "
        f"{cfg_a['max_steps']} steps; peak learning rate {cfg_a['lr']}; warm-up {cfg_a['warmup_steps']} steps.",
        "",
        f"![loss curves]({fig_rel})",
        "",
        "## Resume check",
        "",
        "| Check | Result |",
        "| --- | --- |",
        *(f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in checks.items()),
        f"| largest difference between the runs (loss) | {max_diff:.2e} |",
        f"| verdict | **{verdict}** |",
        "",
        "## Learning",
        "",
        "| | Step 0 (before training) | Final |",
        "| --- | --- | --- |",
        f"| validation loss | {val_a[min(val_a)]:.4f} | {val_a[max(val_a)]:.4f} |",
        f"| validation perplexity | {math.exp(val_a[min(val_a)]):.1f} | {math.exp(val_a[max(val_a)]):.2f} |",
        f"| training loss | - | {train_a[max(train_a)]:.4f} |",
        "",
        f"Training tokens seen: {int(last_row['tokens_seen']):,} ({epochs:.2f} passes over the training text). "
        f"Median speed {sorted(tps)[len(tps) // 2]:,.0f} tokens/s. Peak GPU memory "
        f"{last_row['peak_vram_mb'] + ' MB' if last_row['peak_vram_mb'] else '- (CPU run)'}.",
        "",
        "## Greedy samples (run " + a.a + ")",
        "",
        *([secs[0], "", secs[-1]] if len(secs) >= 2 else secs),
        "",
    ]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"Final step: {a.a} {final_a}, {a.b} {final_b} (resumed at {resume_step})")
    for k, v in checks.items():
        print(f"  {k:<28} {'PASS' if v else 'FAIL'}")
    print(f"Largest difference between the runs: {max_diff:.2e} -> {verdict}")
    print(f"Validation loss: {val_a[min(val_a)]:.4f} (step 0) -> {val_a[max(val_a)]:.4f} (step {max(val_a)})")
    print(f"Figure: {a.figure}\nReport: {a.report}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
