r"""Day 10 - next-word candidates and text generation with Model A (completes school plan task 10).
Day 10 - Model A による次の単語の候補と文の生成(作業計画の作業10を完了する)

==========================  ENGLISH  ==========================

Three commands (VS Code terminal, repo root):

  1. candidates : the top 10 next tokens and their probabilities for a beginning
       .venv\Scripts\python.exe scripts\day10_generate.py candidates --text 今日は --text 営業部の
  2. generate   : continue every line of a prompt file with one setting
       .venv\Scripts\python.exe scripts\day10_generate.py generate --setting sample
  3. report     : the full Day 10 evidence. For best.pt AND last.pt of a Day 9 run:
                  candidates for four beginnings, all four settings on the prompt file,
                  and the automatic checks from task 6 (finished, repetitive, distinct-2, new)
       .venv\Scripts\python.exe scripts\day10_generate.py report --run smoke_a --machine home

Settings (src/generation.py PRESET_SETTINGS):
  greedy         always the most likely token, no controls (what Day 9 samples.md shows)
  greedy+limits  greedy + repetition limits measured from the training questions
  sample         temperature 0.8, top-k 50, top-p 0.95, limits, fixed seed
  sample-hot     temperature 1.5, limits, fixed seed (to show what too much randomness does)

SERVER NOTE (Linux, later): same code;  .venv/bin/python scripts/day10_generate.py report --run smoke_a --machine server
Generation is light, so --device cpu also works on any PC; results with the same seed are the same.

==========================  日本語  ==========================

3つのコマンド(VS Code ターミナル、リポジトリのルートで):
  1. candidates : 書き出しに対する次のトークンの上位10個と、その確率
  2. generate   : プロンプトファイルの各行を、1つの設定で続ける
  3. report     : Day 10 の証拠一式。Day 9 の学習の best.pt と last.pt について、4つの書き出しの候補、
                  4つの設定での生成、作業6の自動の確認(完結・繰り返し・distinct-2・新しい文)を行う
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.model import ModelA, ModelConfig  # noqa: E402
from src.generation import (  # noqa: E402
    PRESET_SETTINGS,
    Generation,
    SamplingConfig,
    distinct_2,
    generate,
    is_repetitive,
    load_model,
    model_logits_fn,
    next_token_candidates,
    token_text,
)
from src.training import sha256_file  # noqa: E402

SCRIPT_VERSION: str = "day10-v2"
CANDIDATE_PROMPTS: tuple[str, ...] = ("今日は", "営業部の", "ボーナスが", "大阪支社で働く社員の")


def read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def rel(path: Path) -> str:
    """A path relative to the repository for reports (no personal folder names in committed files)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def val_loss_at(run_dir: Path, step: int) -> str:
    """The validation loss logged at `step` in the run's log.csv (Day 9), or '-' if it is not there."""
    log = run_dir / "log.csv"
    if log.exists():
        with log.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["event"] == "eval" and int(row["step"]) == step and row["val_loss"]:
                    return f"{float(row['val_loss']):.4f}"
    return "-"


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def show(g: Generation) -> str:
    """Prompt + continuation for tables: ⟨eos⟩ = finished, … = cut at the length limit."""
    end = " ⟨eos⟩" if g.finished else " …"
    return (g.text.replace("|", "\\|").replace("\n", " ") + end).strip()


def candidate_rows(logits_fn, tok: ByteBPETokenizer, prompt: str, k: int) -> list[tuple[str, float]]:
    ids = [1] + tok.encode(prompt)
    return [(token_text(tok, i), p) for i, p in next_token_candidates(logits_fn(ids), k)]


def load_checked(path: Path, tok_path: Path, device: torch.device):
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run Day 9 first.")
    model, info = load_model(path, device)
    if info["tokenizer_sha256"] != sha256_file(tok_path):
        raise SystemExit(f"{path.name} was trained with a different tokenizer")
    return model, info


# --------------------------------------------------------------------------- commands


def cmd_candidates(a: argparse.Namespace, tok: ByteBPETokenizer, device: torch.device) -> int:
    model, info = load_checked(a.checkpoint, a.tokenizer, device)
    fn = model_logits_fn(model, device)
    md = [f"# Next-token candidates ({rel(a.checkpoint)}, step {info['step']})", ""]
    for prompt in a.text or list(CANDIDATE_PROMPTS):
        rows = candidate_rows(fn, tok, prompt, a.k)
        print(f"\n{prompt}  ->  pieces: {' / '.join(tok.pieces(prompt))}")
        for r, (t, p) in enumerate(rows, 1):
            print(f"  {r:>2}. {t:<12} {p:7.2%}")
        print(f"      top {a.k} total: {sum(p for _, p in rows):.2%}")
        md += [f"## {prompt}", "", "| Rank | Next token | Probability |", "| --- | --- | --- |",
               *(f"| {r} | {t} | {p:.2%} |" for r, (t, p) in enumerate(rows, 1)),
               f"| | top {a.k} total | {sum(p for _, p in rows):.2%} |", ""]
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(md), encoding="utf-8", newline="\n")
        print(f"\nSaved: {a.out}")
    return 0


def settings_from_args(a: argparse.Namespace) -> SamplingConfig:
    cfg = PRESET_SETTINGS[a.setting]
    over = {k: v for k, v in (("temperature", a.temperature), ("top_k", a.top_k), ("top_p", a.top_p),
                              ("repetition_penalty", a.repetition_penalty), ("seed", a.seed),
                              ("max_new_tokens", a.max_new_tokens)) if v is not None}
    return replace(cfg, **over, name=cfg.name + ("*" if over else ""))


def cmd_generate(a: argparse.Namespace, tok: ByteBPETokenizer, device: torch.device) -> int:
    model, info = load_checked(a.checkpoint, a.tokenizer, device)
    fn = model_logits_fn(model, device)
    cfg = settings_from_args(a)
    prompts = a.text or read_prompts(a.prompts)
    print(f"Checkpoint {rel(a.checkpoint)} (step {info['step']}) | setting {cfg.to_dict()}")
    md = [f"# Generation ({rel(a.checkpoint)}, step {info['step']}, setting {cfg.name})", "",
          f"Settings: `{cfg.to_dict()}`", "", "| Prompt | Continuation |", "| --- | --- |"]
    for prompt in prompts:
        g = generate(fn, tok, prompt, cfg)
        print(f"  {prompt} | {show(g)}")
        md.append(f"| {prompt} | {show(g)} |")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
        print(f"Saved: {a.out}")
    return 0


def cmd_report(a: argparse.Namespace, tok: ByteBPETokenizer, device: torch.device) -> int:
    run_dir = REPO_ROOT / a.runs_root / a.run
    ckpts = {"best": run_dir / "best.pt", "last": run_dir / "last.pt"}
    loaded = {name: load_checked(p, a.tokenizer, device) for name, p in ckpts.items()}
    fns = {name: model_logits_fn(m, device) for name, (m, _) in loaded.items()}
    infos = {name: info for name, (_, info) in loaded.items()}
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Day 10 report | run {a.run} | machine={a.machine} | {device_name} ===")
    for name, info in infos.items():
        print(f"{name}.pt: step {info['step']}, validation loss at that step {val_loss_at(run_dir, info['step'])}")

    md = [
        "# Day 10 report: next-word candidates and generation",
        "",
        f"Generated by `scripts/day10_generate.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}. "
        f"Machine: {a.machine}, {device_name}, PyTorch {torch.__version__}, Python {platform.python_version()}.",
        "",
        "| Checkpoint | Step | Validation loss at that step |",
        "| --- | --- | --- |",
        f"| `{a.runs_root}/{a.run}/best.pt` | {infos['best']['step']} | {val_loss_at(run_dir, infos['best']['step'])} (lowest) |",
        f"| `{a.runs_root}/{a.run}/last.pt` | {infos['last']['step']} | {val_loss_at(run_dir, infos['last']['step'])} |",
        "",
        "## 1. Next-token candidates (school plan task 10)",
        "",
        "The probability the model gives to each possible next token, before any adjustment. "
        "`<xx..>` is part of a character the tokenizer never learned; `<eos>` means \"the question ends here\".",
        "",
    ]
    print("\n--- next-token candidates (best | last) ---")
    for prompt in CANDIDATE_PROMPTS:
        rb = candidate_rows(fns["best"], tok, prompt, 10)
        rl = candidate_rows(fns["last"], tok, prompt, 10)
        print(f"\n{prompt}  (pieces: {' / '.join(tok.pieces(prompt))})")
        for r, ((tb, pb), (tl, pl)) in enumerate(zip(rb, rl), 1):
            print(f"  {r:>2}. {tb:<10} {pb:7.2%}   | {tl:<10} {pl:7.2%}")
        md += [f"### {prompt}", "", f"Pieces: {' / '.join(tok.pieces(prompt))}", "",
               "| Rank | best.pt token | Probability | last.pt token | Probability |", "| --- | --- | --- | --- | --- |",
               *(f"| {r} | {tb} | {pb:.2%} | {tl} | {pl:.2%} |" for r, ((tb, pb), (tl, pl)) in enumerate(zip(rb, rl), 1)),
               f"| | top 10 total | {sum(p for _, p in rb):.2%} | top 10 total | {sum(p for _, p in rl):.2%} |", ""]

    # ---- generation with every setting on both checkpoints
    results: dict[tuple[str, str], list[Generation]] = {}
    for sname, cfg in PRESET_SETTINGS.items():
        for cname in ckpts:
            results[(cname, sname)] = [generate(fns[cname], tok, p, cfg) for p in prompts]

    md += ["## 2. Generation", "", "Settings (all with a 40-token limit; sampling uses seed 42):", "",
           "| Setting | Method | Temperature | top-k | top-p | Repetition limits |", "| --- | --- | --- | --- | --- | --- |",
           *(f"| {s} | {c.method} | {c.temperature} | {c.top_k or '-'} | {c.top_p if c.top_p < 1 else '-'} | "
             f"{'on' if c.max_run else 'off'} |" for s, c in PRESET_SETTINGS.items()),
           "", "`⟨eos⟩` = the model ended the question; `…` = cut at 40 tokens.", ""]
    for prompt_i, prompt in enumerate(prompts):
        md += [f"### {prompt}", "", "| Setting | best.pt | last.pt |", "| --- | --- | --- |"]
        for sname in PRESET_SETTINGS:
            gb, gl = results[("best", sname)][prompt_i], results[("last", sname)][prompt_i]
            md.append(f"| {sname} | {show(gb)} | {show(gl)} |")
        md.append("")

    # ---- automatic checks (task 6 items A2, A3, A4, A5)
    md += ["## 3. Automatic checks (task 6 items)", "",
           f"Over the {len(prompts)} prompts. Finished = ended with `<eos>` (A2). Repetitive = more repetition than any "
           "training question has (A3). distinct-2 = variety (A4). New = prompt + continuation is not a training "
           "question (A5, reference only).", "",
           "| Checkpoint | Setting | Finished | Repetitive | distinct-2 | New |", "| --- | --- | --- | --- | --- | --- |"]
    print("\n--- automatic checks ---")
    print(f"{'ckpt':<5} {'setting':<14} {'finished':>9} {'repetitive':>11} {'distinct-2':>11} {'new':>6}")
    for (cname, sname), gens in results.items():
        n = len(gens)
        fin = sum(g.finished for g in gens)
        rep = sum(is_repetitive(tok.encode(g.prompt) + [i for i in g.ids if i != 2]) for g in gens)
        d2 = distinct_2([[i for i in g.ids if i != 2] for g in gens])
        new = sum((g.prompt + g.text) not in train_set for g in gens if g.finished)
        md.append(f"| {cname} | {sname} | {fin}/{n} | {rep}/{n} | {d2:.2f} | {new}/{fin} |")
        print(f"{cname:<5} {sname:<14} {fin:>6}/{n:<2} {rep:>8}/{n:<2} {d2:>11.2f} {new:>3}/{fin:<2}")
    md.append("")

    # ---- repetition limits on the untrained model (step 0), where loops really happen
    md += untrained_limits_section(run_dir, tok, device)

    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"\nReport: {a.report}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {a.machine} | Day 10 generation report\n\n"
                    f"- Script: scripts/day10_generate.py ({SCRIPT_VERSION}); device {device_name}; PyTorch {torch.__version__}\n"
                    f"- Run {a.runs_root}/{a.run}: best.pt step {infos['best']['step']}, last.pt step {infos['last']['step']}\n"
                    f"- Prompts: {a.prompts.relative_to(REPO_ROOT).as_posix()} ({len(prompts)}); settings: "
                    f"{', '.join(PRESET_SETTINGS)}; report docs/{a.report.name}\n")
        print("Appended an entry to run_log.md")
    return 0


def untrained_limits_section(run_dir: Path, tok: ByteBPETokenizer, device: torch.device) -> list[str]:
    """Rebuild the step-0 model of the run (same seed as Day 9, so the same starting weights) and compare greedy
    with and without the repetition limits. Trained checkpoints rarely loop, so this shows the limits at work."""
    cfg_path, train_cfg_path = run_dir / "model_config.json", run_dir / "train_config.json"
    if not cfg_path.exists() or not train_cfg_path.exists():
        return []
    seed = json.loads(train_cfg_path.read_text(encoding="utf-8"))["seed"]
    torch.manual_seed(seed)
    untrained = ModelA(ModelConfig.load(cfg_path)).to(device).eval()
    fn = model_logits_fn(untrained, device)
    lines = ["## 4. Repetition limits on the untrained model (step 0)", "",
             f"The step-0 model is rebuilt with the run's seed ({seed}), so it starts from the same weights as Day 9. "
             "Untrained, it repeats one piece; the limits (at most 4 same tokens in a row, a 3-token sequence at most "
             "twice, as in the training questions) force it out of the loop.", "",
             "| Prompt | greedy | greedy+limits |", "| --- | --- | --- |"]
    print("\n--- repetition limits on the untrained model (step 0) ---")
    for prompt in ("営業部の", "大阪支社で"):
        g0 = generate(fn, tok, prompt, PRESET_SETTINGS["greedy"])
        g1 = generate(fn, tok, prompt, PRESET_SETTINGS["greedy+limits"])
        print(f"  {prompt} | {show(g0)[:30]} | {show(g1)[:30]}")
        lines.append(f"| {prompt} | {show(g0)} | {show(g1)} |")
    return lines + [""]


# --------------------------------------------------------------------------- arguments


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 10: next-word candidates and generation.")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    sub = ap.add_subparsers(dest="command", required=True)
    default_ckpt = REPO_ROOT / "runs/day9/smoke_a/best.pt"

    c = sub.add_parser("candidates", help="top-k next tokens for beginnings")
    c.add_argument("--checkpoint", type=Path, default=default_ckpt)
    c.add_argument("--text", action="append", help="beginning (repeatable); default: four examples")
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--out", type=Path, default=None, help="also save a Markdown table")

    g = sub.add_parser("generate", help="continue prompts with one setting")
    g.add_argument("--checkpoint", type=Path, default=default_ckpt)
    g.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
    g.add_argument("--text", action="append", help="beginning (repeatable) instead of the prompt file")
    g.add_argument("--setting", choices=list(PRESET_SETTINGS), default="sample")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--top-p", type=float)
    g.add_argument("--repetition-penalty", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--out", type=Path, default=None)

    r = sub.add_parser("report", help="full Day 10 report for best.pt and last.pt of a run")
    r.add_argument("--run", default="smoke_a")
    r.add_argument("--runs-root", default="runs/day9")
    r.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
    r.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY10_GENERATION_REPORT.md")
    r.add_argument("--machine", choices=["home", "server", "school"], default="home")
    r.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    for name in ("tokenizer", "checkpoint", "prompts", "report", "out"):
        if getattr(a, name, None) is not None:
            setattr(a, name, getattr(a, name).resolve())
    return a


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    tok = ByteBPETokenizer.load(a.tokenizer)
    device = pick_device(a.device)
    return {"candidates": cmd_candidates, "generate": cmd_generate, "report": cmd_report}[a.command](a, tok, device)


if __name__ == "__main__":
    sys.exit(main())
