# Day 10 — day10_generate.py — Simple Line-by-Line Explanation (English)

> **How the three Day 10 files fit together**
> - **Day 10.1 `generation.py`** — the engine (**read it first**).
> - **Day 10 (this file) `day10_generate.py`** — the **command-line driver** with three commands: `candidates`, `generate`, `report`. It completes **school plan task 10**.
> - **Day 10.2 `test_generation.py`** — automatic tests for the engine.

---

## 0. The big picture (read this first)

After Day 9 you have trained checkpoints (`best.pt`, `last.pt`). Day 10 lets you **look at what the model has learned**, in three ways:

| Command | What it does | Typical use |
|---|---|---|
| **`candidates`** | Shows the **top 10 next tokens and their probabilities** for a beginning | "What does the model think comes after `今日は`?" |
| **`generate`** | **Continues every line of a prompt file** with one setting | Quick experiments |
| **`report`** | The **full Day 10 evidence** for `best.pt` **and** `last.pt` of a run: candidates for four beginnings, **all four settings** on the prompt file, automatic checks, and a demo of the repetition limits | The report you hand in |

```
# 1. candidates
.venv\Scripts\python.exe scripts\day10_generate.py candidates --text 今日は --text 営業部の
# 2. generate
.venv\Scripts\python.exe scripts\day10_generate.py generate --setting sample
# 3. report
.venv\Scripts\python.exe scripts\day10_generate.py report --run smoke_a --machine home
```
The four settings (from Day 10.1): `greedy`, `greedy+limits`, `sample` (temperature 0.8, top-k 50, top-p 0.95), `sample-hot` (temperature 1.5, to show what too much randomness does).

Generation is light, so `--device cpu` works on any PC. Same seed → same text. (Linux server later: `.venv/bin/python scripts/day10_generate.py report --run smoke_a --machine server`.)

---

## 1. Words you need to know

All words are from Day 10.1 (logits, greedy, sampling, temperature, top-k/top-p, `<eos>`, distinct-2). New Python ideas:
- **Sub-commands** (`argparse` `add_subparsers`): one script with several "commands", each with its own options (like `git commit`, `git push`).
- **`dataclasses.replace`**: make a **copy** of a dataclass object with some fields changed.

---

## 2. The docstring, imports and constants

- English + Japanese: the three commands, the four settings, the server note. (The prompt file `prompts/day10_prompts.txt` is the default input for `generate` and `report`; it must exist in your repo.)

```python
from __future__ import annotations

import argparse, csv, json, platform, sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.model import ModelA, ModelConfig  # noqa: E402
from src.generation import (PRESET_SETTINGS, Generation, SamplingConfig, distinct_2, generate, is_repetitive,
                            load_model, model_logits_fn, next_token_candidates, token_text)  # noqa: E402
from src.training import sha256_file  # noqa: E402

SCRIPT_VERSION: str = "day10-v2"
CANDIDATE_PROMPTS: tuple[str, ...] = ("今日は", "営業部の", "ボーナスが", "大阪支社で働く社員の")
```
- The usual setup (`sys.path` trick as in earlier days). From the other files: the tokenizer (**Day 6.1**), the model (**Day 8.1**), the whole generation engine (**Day 10.1**), and `sha256_file` (**Day 9.2**).
- `CANDIDATE_PROMPTS` — the four beginnings used for the candidate tables in the report: a general one (`今日は` = "today…"), and company-style ones.

---

## 3. Helper functions

### 3.1 `read_prompts`

```python
def read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
```
- Reads the prompt file: one prompt per line; **blank lines and lines starting with `#` (comments) are skipped**.

### 3.2 `rel`

```python
def rel(path: Path) -> str:
    """A path relative to the repository for reports (no personal folder names in committed files)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name
```
- Shows a path **relative to the repo** with `/` separators; if the file is outside the repo, only the **file name**. So reports never contain your personal folder names (e.g. `C:\Users\...`) — important because some repositories are public.

### 3.3 `val_loss_at`

```python
def val_loss_at(run_dir: Path, step: int) -> str:
    log = run_dir / "log.csv"
    if log.exists():
        with log.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["event"] == "eval" and int(row["step"]) == step and row["val_loss"]:
                    return f"{float(row['val_loss']):.4f}"
    return "-"
```
- Finds the **validation loss logged at a given step** in the run's `log.csv` (Day 9.2) so the report can say "this checkpoint was saved at step N, where the validation loss was X". Returns `"-"` if not found.

### 3.4 `pick_device`, `show`, `candidate_rows`, `load_checked`

```python
def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)
```
- As in Day 8 (without the "CUDA missing" message).

```python
def show(g: Generation) -> str:
    """Prompt + continuation for tables: ⟨eos⟩ = finished, … = cut at the length limit."""
    end = " ⟨eos⟩" if g.finished else " …"
    return (g.text.replace("|", "\\|").replace("\n", " ") + end).strip()
```
- Formats a generation for a Markdown table: the **continuation text** (`g.text`; the prompt is shown in its own column/heading, despite the docstring) with `|` escaped and line breaks replaced (as in Day 9.2), plus a mark: **⟨eos⟩** = the model ended the question; **…** = cut at the length limit.

```python
def candidate_rows(logits_fn, tok, prompt, k) -> list[tuple[str, float]]:
    ids = [1] + tok.encode(prompt)
    return [(token_text(tok, i), p) for i, p in next_token_candidates(logits_fn(ids), k)]
```
- For a prompt: build the model input (`[1]` = `<bos>` + the prompt tokens), get the scores, take the top `k` candidates, and give each token a **readable label** (`token_text`). Returns `(text, probability)` pairs.

```python
def load_checked(path: Path, tok_path: Path, device: torch.device):
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run Day 9 first.")
    model, info = load_model(path, device)
    if info["tokenizer_sha256"] != sha256_file(tok_path):
        raise SystemExit(f"{path.name} was trained with a different tokenizer")
    return model, info
```
- Load a checkpoint **and verify the tokenizer**: the fingerprint stored in the checkpoint must equal the SHA-256 of the tokenizer file now on disk. Otherwise the numbers would mean different words → stop. (The same kind of check as Day 9.2 §5.2.)

---

## 4. Command 1: `candidates`

```python
def cmd_candidates(a, tok, device) -> int:
    model, info = load_checked(a.checkpoint, a.tokenizer, device)
    fn = model_logits_fn(model, device)
    md = [f"# Next-token candidates ({rel(a.checkpoint)}, step {info['step']})", ""]
```
- Load and check the checkpoint, wrap the model as a scores function (Day 10.1 §4.3), start the Markdown lines.

```python
    for prompt in a.text or list(CANDIDATE_PROMPTS):
        rows = candidate_rows(fn, tok, prompt, a.k)
        print(f"\n{prompt}  ->  pieces: {' / '.join(tok.pieces(prompt))}")
        for r, (t, p) in enumerate(rows, 1):
            print(f"  {r:>2}. {t:<12} {p:7.2%}")
        print(f"      top {a.k} total: {sum(p for _, p in rows):.2%}")
        md += [f"## {prompt}", "", "| Rank | Next token | Probability |", "| --- | --- | --- |",
               *(f"| {r} | {t} | {p:.2%} |" for r, (t, p) in enumerate(rows, 1)),
               f"| | top {a.k} total | {sum(p for _, p in rows):.2%} |", ""]
```
- For each beginning (yours with `--text`, or the four defaults — `a.text or …`):
  - get the top-`k` candidates;
  - print the prompt with its **tokenizer pieces**, then a ranked list: rank, token, **probability as a percentage** (`:7.2%` = a percentage with 2 decimals in a field 7 wide; `:<12` = left-aligned in 12), and the **total probability of the top `k`**;
  - add the same as a Markdown table.
- The total shows how "sure" the model is: if the top 10 cover 95%, the model is concentrated; if only 30%, it is unsure.

```python
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(md), encoding="utf-8", newline="\n")
        print(f"\nSaved: {a.out}")
    return 0
```
- With `--out`, also save the Markdown table. Return 0.

---

## 5. Command 2: `generate`

```python
def settings_from_args(a: argparse.Namespace) -> SamplingConfig:
    cfg = PRESET_SETTINGS[a.setting]
    over = {k: v for k, v in (("temperature", a.temperature), ("top_k", a.top_k), ("top_p", a.top_p),
                              ("repetition_penalty", a.repetition_penalty), ("seed", a.seed),
                              ("max_new_tokens", a.max_new_tokens)) if v is not None}
    return replace(cfg, **over, name=cfg.name + ("*" if over else ""))
```
- Start from the chosen **preset** (`--setting`). `over` — a dictionary of the **overrides you typed** (`--temperature`, `--top-k`, …), built from `(name, value)` pairs by keeping those that are **not `None`**.
- `replace(cfg, **over, name=…)` — a **copy** of the preset with your overrides applied; the name gets a `*` if anything was overridden (so tables show it is a modified setting). `replace` runs the validation again, so a bad value (`--temperature 0`) is rejected.

```python
def cmd_generate(a, tok, device) -> int:
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
```
- Load/check, build the settings, take the prompts (your `--text`s, or the lines of the **prompt file**), print the settings, then for each prompt: **generate** (Day 10.1 `generate`), print `prompt | continuation`, add a table row. Optionally save.

---

## 6. Command 3: `report`

### 6.1 Setup

```python
def cmd_report(a, tok, device) -> int:
    run_dir = REPO_ROOT / a.runs_root / a.run
    ckpts = {"best": run_dir / "best.pt", "last": run_dir / "last.pt"}
    loaded = {name: load_checked(p, a.tokenizer, device) for name, p in ckpts.items()}
    fns = {name: model_logits_fn(m, device) for name, (m, _) in loaded.items()}
    infos = {name: info for name, (_, info) in loaded.items()}
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
```
- Both checkpoints are loaded (`best` and `last`), each wrapped as a scores function; their info (steps etc.) is kept. Three **dictionary comprehensions** build `loaded`, `fns`, `infos` keyed by `"best"` / `"last"`.
- `prompts` — from the prompt file. `train_set` — **all training questions as a set** (fast lookup), used later for the "new?" check (reference only). `device_name` as in Day 8.
- Then it prints a header and, for each checkpoint, its step and the validation loss at that step; the report starts with a table of the two checkpoints (**best = lowest validation loss**).

### 6.2 Section 1: next-token candidates (task 10)

```python
    for prompt in CANDIDATE_PROMPTS:
        rb = candidate_rows(fns["best"], tok, prompt, 10)
        rl = candidate_rows(fns["last"], tok, prompt, 10)
        ...
        for r, ((tb, pb), (tl, pl)) in enumerate(zip(rb, rl), 1):
            print(f"  {r:>2}. {tb:<10} {pb:7.2%}   | {tl:<10} {pl:7.2%}")
        md += [... table "Rank | best.pt token | Probability | last.pt token | Probability" ...]
```
- For each of the four beginnings: top-10 candidates from **both** checkpoints, shown **side by side** (`zip(rb, rl)` pairs rank 1 of best with rank 1 of last, …) with probabilities and the top-10 total. This shows how the model's beliefs differ between the best and the last checkpoint.

### 6.3 Section 2: generation with every setting on both checkpoints

```python
    results: dict[tuple[str, str], list[Generation]] = {}
    for sname, cfg in PRESET_SETTINGS.items():
        for cname in ckpts:
            results[(cname, sname)] = [generate(fns[cname], tok, p, cfg) for p in prompts]
```
- For **every preset × both checkpoints**, generate a continuation for **every prompt**; store in a dictionary keyed by `(checkpoint name, setting name)`.
- Then the report writes: a **table of the four settings** (method, temperature, top-k or `-`, top-p or `-` if it is 1, repetition limits `on`/`off` — `c.max_run` is truthy when set), and **one table per prompt**: rows = settings, columns = `best.pt` and `last.pt` continuations (`show(...)`). `⟨eos⟩` = finished, `…` = cut at 40 tokens.

### 6.4 Section 3: automatic checks (task 6 items A2–A5)

```python
    for (cname, sname), gens in results.items():
        n = len(gens)
        fin = sum(g.finished for g in gens)
        rep = sum(is_repetitive(tok.encode(g.prompt) + [i for i in g.ids if i != 2]) for g in gens)
        d2 = distinct_2([[i for i in g.ids if i != 2] for g in gens])
        new = sum((g.prompt + g.text) not in train_set for g in gens if g.finished)
        md.append(f"| {cname} | {sname} | {fin}/{n} | {rep}/{n} | {d2:.2f} | {new}/{fin} |")
        print(f"{cname:<5} {sname:<14} {fin:>6}/{n:<2} {rep:>8}/{n:<2} {d2:>11.2f} {new:>3}/{fin:<2}")
```
- For each (checkpoint, setting):
  - **Finished (A2)** — how many outputs ended with `<eos>` (`sum` of True/False counts the Trues);
  - **Repetitive (A3)** — how many outputs have more repetition than any real question (`is_repetitive` on **prompt + generated tokens, `<eos>` removed** — the `2` is `<eos>`'s ID);
  - **distinct-2 (A4)** — the variety over all outputs;
  - **New (A5, reference only)** — among **finished** outputs, how many are **not** a training question (`prompt + text` compared with `train_set`). New is *not* the same as good.
- Printed as a neat table (alignment codes as before) and added to the report as `finished/n`, `repetitive/n`, `distinct-2`, `new/finished`.

### 6.5 Section 4: repetition limits on the **untrained** model

```python
    md += untrained_limits_section(run_dir, tok, device)
```
- Adds a demo section (next function).

### 6.6 Save and log

```python
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"\nReport: {a.report}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(...)   # heading + script/device/PyTorch; run and best/last steps; prompt file, settings, report name
        print("Appended an entry to run_log.md")
    return 0
```
- Save `docs/DAY10_GENERATION_REPORT.md`, and (unless `--no-log`) append a short entry to `run_log.md`.

### 6.7 `untrained_limits_section` — the "limits at work" demo

```python
def untrained_limits_section(run_dir, tok, device) -> list[str]:
    cfg_path, train_cfg_path = run_dir / "model_config.json", run_dir / "train_config.json"
    if not cfg_path.exists() or not train_cfg_path.exists():
        return []
    seed = json.loads(train_cfg_path.read_text(encoding="utf-8"))["seed"]
    torch.manual_seed(seed)
    untrained = ModelA(ModelConfig.load(cfg_path)).to(device).eval()
    fn = model_logits_fn(untrained, device)
```
- **The problem:** a *trained* checkpoint rarely falls into loops, so the limits would never visibly do anything. **The trick:** rebuild the **untrained (step 0) model** of the run: read the run's saved model config and its **seed**, set the same seed, and build the model — **Day 9 did exactly this** (`torch.manual_seed(seed)` then `ModelA(...)`), so the weights are **identical to the starting weights of the real run**. (If the run's config files are missing, the section is skipped: `return []`.)

```python
    lines = ["## 4. Repetition limits on the untrained model (step 0)", "", f"The step-0 model is rebuilt with the run's seed ({seed}), ... ", "",
             "| Prompt | greedy | greedy+limits |", "| --- | --- | --- |"]
    for prompt in ("営業部の", "大阪支社で"):
        g0 = generate(fn, tok, prompt, PRESET_SETTINGS["greedy"])
        g1 = generate(fn, tok, prompt, PRESET_SETTINGS["greedy+limits"])
        print(f"  {prompt} | {show(g0)[:30]} | {show(g1)[:30]}")
        lines.append(f"| {prompt} | {show(g0)} | {show(g1)} |")
    return lines + [""]
```
- Two prompts; for each, greedy **without** limits (the untrained model repeats one piece over and over) versus **with** limits (at most 4 identical tokens in a row, a 3-token sequence at most twice — forced out of the loop). Printed (first 30 characters) and written as a table. This gives **visible evidence that the limits work**.

---

## 7. Arguments (sub-commands)

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 10: next-word candidates and generation.")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    sub = ap.add_subparsers(dest="command", required=True)
    default_ckpt = REPO_ROOT / "runs/day9/smoke_a/best.pt"
```
- Two options shared by all commands (`--tokenizer`, `--device`), and `add_subparsers(dest="command", required=True)` — you **must** name a command; its name is stored in `a.command`. `default_ckpt` — the default checkpoint for `candidates`/`generate`.

```python
    c = sub.add_parser("candidates", help="top-k next tokens for beginnings")
    c.add_argument("--checkpoint", type=Path, default=default_ckpt)
    c.add_argument("--text", action="append", help="beginning (repeatable); default: four examples")
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--out", type=Path, default=None, help="also save a Markdown table")
```
- The **`candidates`** command: `--checkpoint`, `--text` (repeatable with `action="append"`), `--k` (how many candidates, 10), `--out`.

```python
    g = sub.add_parser("generate", help="continue prompts with one setting")
    g.add_argument("--checkpoint", ...)  g.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
    g.add_argument("--text", action="append", help="beginning (repeatable) instead of the prompt file")
    g.add_argument("--setting", choices=list(PRESET_SETTINGS), default="sample")
    g.add_argument("--temperature", type=float)  g.add_argument("--top-k", type=int)  g.add_argument("--top-p", type=float)
    g.add_argument("--repetition-penalty", type=float)  g.add_argument("--seed", type=int)  g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--out", type=Path, default=None)
```
- The **`generate`** command: the checkpoint, the prompt file (or `--text`), the preset (`--setting`, default `sample`), and **optional overrides** with no default (so `None` = "not given", which `settings_from_args` relies on).

```python
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
```
- The **`report`** command: which run (`--run`, folder `runs/day9/<run>`), the prompt file, the report path, the machine label, `--no-log`.
- Finally, make the path options **absolute** with the `getattr`/`setattr` loop (as in earlier days); `getattr(a, name, None)` returns `None` for options that this command does not have, so only the existing ones are resolved.

---

## 8. `main`

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    tok = ByteBPETokenizer.load(a.tokenizer)
    device = pick_device(a.device)
    return {"candidates": cmd_candidates, "generate": cmd_generate, "report": cmd_report}[a.command](a, tok, device)


if __name__ == "__main__":
    sys.exit(main())
```
- Console safety, parse, load the tokenizer, choose the device, then **dispatch**: a dictionary maps the command name to its function; `[a.command]` picks one and `(a, tok, device)` calls it. Its return value (0) is the exit code.

---

## 9. How to read the results

- **Candidates:** a trained model should put high probability on sensible next tokens; the **top-10 total** tells how concentrated it is. `<eos>` appearing high after a complete-looking question = the model learned to end.
- **Greedy vs sampling:** greedy is deterministic and may repeat; `sample` is more varied; `sample-hot` is deliberately too random (nonsense, broken characters) — a good illustration of temperature.
- **Automatic checks:** want **Finished** high, **Repetitive** low, **distinct-2** reasonably high; **New** is only for reference.
- **Section 4** shows the limits breaking loops in the untrained model.
- **best.pt vs last.pt:** usually similar after a short smoke run; they differ when validation got worse after the best step.

## 10. Small things worth knowing

- **The prompt file must exist** (`prompts/day10_prompts.txt` by default).
- **`show()` shows only the continuation**, although its docstring says "prompt + continuation"; the prompt is displayed separately in the tables.
- **The report command needs both `best.pt` and `last.pt`** (a Day 9 run creates both at its first evaluation).
- **The log entry uses `a.prompts.relative_to(REPO_ROOT)`**, so it fails if the prompt file is outside the repo (the report itself works).
- **Hard-coded `2`** is `<eos>`'s ID (same as Day 9.2).
- **"New" is only a reference:** a sentence not in the training set can still be nonsense.

## 11. Quick summary in one paragraph

`day10_generate.py` is the command-line front end for generation: `candidates` shows the top next tokens with probabilities, `generate` continues prompts with a chosen (or overridden) setting, and `report` produces the full evidence for the best and last checkpoints — side-by-side candidates, all four settings on every prompt, automatic checks (finished / repetitive / distinct-2 / new) and a demonstration on the untrained step-0 model that the repetition limits stop loops — after verifying that each checkpoint's tokenizer fingerprint matches the tokenizer file.
