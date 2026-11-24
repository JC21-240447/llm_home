r"""Day 13 (part A) - the final automatic evaluation of the final model, and the human scoring sheet.
Day 13(前半) - 最終のモデルの、プログラムによる最終評価と、人が採点するためのシート

==========================  ENGLISH  ==========================

Follows the task 6 document ("How to check generated text, and the evaluation criteria"):
    - A1 perplexity on the validation AND the TEST text. This is the first and only time the test
      text is used: every choice (size, dropout, seed) was already made with the validation text.
      References: uniform guessing (~1,674), the untrained model, and a counting (bigram) baseline.
    - A2-A5 on the fixed list of 50 beginnings (6 kinds, task 6 Table 7), two generation methods
      (greedy + limits, sampling with a fixed seed), max 40 tokens, both best.pt and last.pt (6-2).
    - Extra automatic checks for the errors Day 12 showed: wrong unit, broken number, unseen value
      (src/evaluation.py). All are checked on the real test questions first.
    - The human scoring workbook (6-4): the 100 outputs of the final model (50 beginnings x 2
      methods), shuffled, with the method hidden. Each scorer fills in a copy; part B
      (scripts/day13_score_summary.py) reads the copies and writes the final evaluation table.

Writes:
    docs/DAY13_AUTO_EVALUATION.md          the automatic results
    docs/figures/day13_perplexity.png      A1 against the references
    docs/day13/auto_results.json           the numbers (part B reads them)
    docs/day13/generations.csv             all 200 outputs with their checks (opens in Excel)
    docs/day13/scoring_sheet.xlsx          the sheet to hand to the scorers
    docs/day13/scoring_key.json            which method made each scored output (do not show scorers)

Usage (VS Code terminal, repo root):
    .venv\Scripts\python.exe scripts\day13_final_eval.py --machine home

SERVER NOTE (Linux, later): same code;  .venv/bin/python scripts/day13_final_eval.py --machine server

==========================  日本語  ==========================

作業6の文書「生成した文章の確認方法と評価基準」のとおりに評価する:
    - A1 パープレキシティを検証用とテスト用の文章で測る。テスト用の文章を使うのは、これが最初で最後
      (大きさ・ドロップアウト・種の選択は、すべて検証用の文章で済んでいる)
    - 固定の50の書き出し、2つの生成方法、最大40トークン、best.pt と last.pt の両方で A2〜A5
    - Day 12 で見つかった誤り(単位の誤り、壊れた数字)を数える自動の確認2つ
    - 人の採点用の Excel(最終のモデルの100の出力。順番をばらばらにし、生成方法を隠す)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.batching import open_tokens  # noqa: E402
from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.evaluation import (PROMPT_KINDS, BigramBaseline, ScoringRow, UnitChecker, has_broken_number,  # noqa: E402
                            load_eval_prompts, max_digit_run, term_phrases, write_scoring_workbook)
from src.generation import (EOS_ID, PRESET_SETTINGS, distinct_2, generate, is_repetitive, load_model,  # noqa: E402
                            model_logits_fn)
from src.model import ModelA, ModelConfig  # noqa: E402
from src.selection import MAX_REPETITIVE, MIN_FINISHED  # noqa: E402
from src.training import evaluate, pick_precision  # noqa: E402

SCRIPT_VERSION: str = "day13-v1"
METHODS: tuple[str, ...] = ("greedy+limits", "sample")
SHUFFLE_SEED: int = 13


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 13 part A: final automatic evaluation + scoring sheet.")
    ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
    ap.add_argument("--final", type=Path, default=REPO_ROOT / "configs/day12_final.json")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day13_eval_prompts.csv")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-max-windows", type=int, default=None, help="(testing only) fewer windows")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs/day13")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY13_AUTO_EVALUATION.md")
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day13_perplexity.png")
    ap.add_argument("--force", action="store_true", help="replace the scoring sheet even if scores already exist")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    final = json.loads(a.final.read_text(encoding="utf-8"))
    best_path = REPO_ROOT / final["final_checkpoint"]
    run_dir = best_path.parent
    last_path = run_dir / "last.pt"
    for p in (best_path, last_path):
        if not p.exists():
            raise SystemExit(f"{p} not found. Day 13 needs the Day 12 run folder ({run_dir}); do not delete runs/day12.")
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    precision = pick_precision("auto", device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    mcfg = ModelConfig.load(REPO_ROOT / final["model_config"])
    vocab = mcfg.vocab_size
    print(f"=== Day 13 final evaluation | final model {final['final_checkpoint']} | {device_name} ===")

    # ---- data
    tok = ByteBPETokenizer.load(REPO_ROOT / "data/tokenizer/tokenizer.json")
    tokens_dir = REPO_ROOT / "data/tokens"
    train_tok, val_tok, test_tok = (open_tokens(tokens_dir, s) for s in ("train", "val", "test"))
    train_lines = read_lines(REPO_ROOT / "data/corpus/train.txt")
    test_lines = read_lines(REPO_ROOT / "data/corpus/test.txt")
    train_set = set(train_lines)
    prompts = load_eval_prompts(a.prompts)
    counts = {k: sum(p.kind == k for p in prompts) for k in PROMPT_KINDS}
    print(f"Beginnings: {len(prompts)} ({', '.join(f'{k} {n}' for k, n in counts.items())})")

    def loss_of(model: torch.nn.Module, tokens) -> float:
        v = evaluate(model, tokens, mcfg.block_size, a.batch_size, device, precision, a.eval_max_windows)
        model.eval()     # evaluate() switches back to training mode at the end; generation needs eval mode
        return v

    # ---- A1 perplexity
    print("\n--- A1 perplexity (validation and test) ---")
    a1: dict[str, dict] = {}
    a1["uniform"] = {"label": "Uniform guessing (every token equally likely)", "val": math.log(vocab),
                     "test": math.log(vocab)}
    torch.manual_seed(0)
    untrained = ModelA(mcfg).to(device).eval()
    a1["untrained"] = {"label": "Untrained model (same size, random weights)", "val": loss_of(untrained, val_tok),
                       "test": loss_of(untrained, test_tok)}
    del untrained
    bigram = BigramBaseline.fit(train_tok, vocab)
    lam = bigram.tune(val_tok)
    a1["bigram"] = {"label": f"Counting baseline: bigram (previous token only, weight {lam} chosen on validation)",
                    "val": bigram.loss(val_tok), "test": bigram.loss(test_tok)}
    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    for name, path in (("best.pt", best_path), ("last.pt", last_path)):
        model, info = load_model(path, device)
        models[name] = (model, info)
        a1[name] = {"label": f"Final model {name} (step {info['step']})", "val": loss_of(model, val_tok),
                    "test": loss_of(model, test_tok), "step": info["step"]}
    others = {}
    for seed in final.get("seeds", []):
        p = run_dir.parent / f"main_s{seed}" / "best.pt"
        if p.exists() and p.resolve() != best_path.resolve():
            m, info = load_model(p, device)
            others[f"main_s{seed}"] = {"val": loss_of(m, val_tok), "test": loss_of(m, test_tok), "step": info["step"]}
            del m
    for k, r in a1.items():
        print(f"  {r['label']:<80} val {r['val']:.4f} ({math.exp(r['val']):,.2f})  test {r['test']:.4f} "
              f"({math.exp(r['test']):,.2f})")
    for k, r in others.items():
        print(f"  other seed {k} best.pt (step {r['step']}): val {r['val']:.4f}  test {r['test']:.4f}")

    # ---- detectors, checked on the real test questions first
    unit_checker = UnitChecker(term_phrases(REPO_ROOT / "data/source/columns.csv")).learn(train_lines)
    digit_limit = max_digit_run(train_lines)
    calib = {"questions": len(test_lines),
             "wrong_unit": sum(bool(unit_checker.mismatches(q)) for q in test_lines),
             "broken_number": sum(has_broken_number(q, digit_limit) for q in test_lines),
             "unseen_value": sum(bool(unit_checker.unseen_values(q)) for q in test_lines),
             "digit_limit": digit_limit, "items_learned": len(unit_checker.allowed)}
    print(f"\nDetectors on the {calib['questions']} real test questions (should be ~0): wrong unit "
          f"{calib['wrong_unit']}, broken number {calib['broken_number']} (digit limit {digit_limit}), "
          f"unseen value {calib['unseen_value']}")
    unseen = [p.prompt for p in prompts if p.kind == "unseen"]
    unseen_new = sum(not any(q.startswith(u) for q in train_lines) for u in unseen)

    # ---- A2-A5 (+ extra checks) on the 50 beginnings
    print("\n--- A2-A5 on the fixed beginnings ---")
    outputs: list[dict] = []
    agg: dict[str, dict] = {}
    for mname, (model, _) in models.items():
        fn = model_logits_fn(model, device)
        for method in METHODS:
            body_ids = []
            rows = []
            for p in prompts:
                g = generate(fn, tok, p.prompt, PRESET_SETTINGS[method])
                body = [i for i in g.ids if i != EOS_ID]
                body_ids.append(body)
                full = p.prompt + g.text
                rows.append({
                    "model": mname, "method": method, "kind": p.kind, "prompt": p.prompt, "text": g.text,
                    "finished": g.finished, "repetitive": is_repetitive(tok.encode(p.prompt) + body),
                    "copy": g.finished and full in train_set,
                    "wrong_unit": "; ".join(f"{t}+{u}" for t, u in unit_checker.mismatches(full)),
                    "broken_number": has_broken_number(full, digit_limit),
                    "unseen_value": "; ".join(unit_checker.unseen_values(full)),
                })
            n = len(rows)
            agg[f"{mname}|{method}"] = {
                "finished": sum(r["finished"] for r in rows) / n, "repetitive": sum(r["repetitive"] for r in rows) / n,
                "distinct2": distinct_2(body_ids), "copy": sum(r["copy"] for r in rows) / n,
                "wrong_unit": sum(bool(r["wrong_unit"]) for r in rows) / n,
                "broken_number": sum(r["broken_number"] for r in rows) / n,
                "unseen_value": sum(bool(r["unseen_value"]) for r in rows) / n,
            }
            outputs.extend(rows)
            s = agg[f"{mname}|{method}"]
            print(f"  {mname} {method:<14} finished {s['finished']:.0%} | repetitive {s['repetitive']:.0%} | "
                  f"distinct-2 {s['distinct2']:.2f} | copy {s['copy']:.0%} | wrong unit {s['wrong_unit']:.0%} | "
                  f"broken number {s['broken_number']:.0%} | unseen value {s['unseen_value']:.0%}")

    # ---- the human scoring sheet: final model only, both methods, shuffled, method hidden
    scored = [r for r in outputs if r["model"] == "best.pt"]
    random.Random(SHUFFLE_SEED).shuffle(scored)
    key = [{"id": f"S{i:03d}", "method": r["method"], "kind": r["kind"], "prompt": r["prompt"],
            "text": r["text"] + ("" if r["finished"] else " …"), "out_of_scope": r["kind"] == "out_of_scope"}
           for i, r in enumerate(scored, 1)]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    key_path, sheet_path = a.out_dir / "scoring_key.json", a.out_dir / "scoring_sheet.xlsx"
    scores_dir = a.out_dir / "scores"
    has_scores = scores_dir.exists() and any(scores_dir.glob("scores_*.xlsx"))
    old_key = json.loads(key_path.read_text(encoding="utf-8"))["rows"] if key_path.exists() else None
    if has_scores and old_key is not None and old_key != key and not a.force:
        raise SystemExit("The new outputs differ from the sheet the scorers already used (docs/day13/scores has "
                         "files). Nothing was overwritten. Use --force only if you intend to score again.")
    key_path.write_text(json.dumps({"final_checkpoint": final["final_checkpoint"], "shuffle_seed": SHUFFLE_SEED,
                                    "rows": key}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_scoring_workbook(sheet_path, [ScoringRow(k["id"], k["prompt"], k["text"], k["out_of_scope"]) for k in key])
    scores_dir.mkdir(exist_ok=True)

    # ---- files
    with (a.out_dir / "generations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(outputs[0].keys()))
        w.writeheader()
        w.writerows(outputs)
    results = {
        "script": SCRIPT_VERSION, "date": f"{datetime.now():%Y-%m-%d %H:%M}", "machine": a.machine,
        "device": device_name, "final": final, "a1": a1, "other_seeds": others, "generation": agg,
        "calibration": calib, "prompt_counts": counts, "unseen_never_in_training": unseen_new,
        "methods": {m: PRESET_SETTINGS[m].to_dict() for m in METHODS}, "scored_outputs": len(key),
        "guidelines": {"min_finished": MIN_FINISHED, "max_repetitive": MAX_REPETITIVE},
    }
    (a.out_dir / "auto_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")
    draw_figure(a1, a.figure)
    write_report(a, results, outputs, prompts, sheet_path)
    print(f"\nReport: {a.report}\nScoring sheet: {sheet_path}  ({len(key)} outputs)\nNext: part B (human scoring).")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {a.machine} | Day 13 final automatic evaluation\n\n"
                    f"- Script: scripts/day13_final_eval.py ({SCRIPT_VERSION}); {device_name}; PyTorch {torch.__version__}\n"
                    f"- Final model {final['final_checkpoint']}: test loss {a1['best.pt']['test']:.4f} "
                    f"(perplexity {math.exp(a1['best.pt']['test']):.2f}); bigram baseline test "
                    f"{math.exp(a1['bigram']['test']):.2f}\n"
                    + "".join(f"- {k}: finished {v['finished']:.0%}, repetitive {v['repetitive']:.0%}, wrong unit "
                              f"{v['wrong_unit']:.0%}\n" for k, v in agg.items())
                    + f"- Scoring sheet: {len(key)} outputs (docs/day13/scoring_sheet.xlsx)\n")
        print("Appended an entry to run_log.md")
    return 0


def draw_figure(a1: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["uniform", "untrained", "bigram", "last.pt", "best.pt"]
    names = ["uniform\nguessing", "untrained\nmodel", "bigram\n(counting)", "final model\nlast.pt",
             "final model\nbest.pt"]
    vals = [math.exp(a1[k]["test"]) for k in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    bars = ax.bar(names, vals, color=["#bbbbbb", "#bbbbbb", "#8c8c8c", "#6baed6", "#1f77b4"])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:,.1f}", ha="center", fontsize=9)
    ax.set_ylabel("test perplexity (log scale, lower is better)")
    ax.set_title("A1: perplexity on the test text")
    ax.set_ylim(1, max(vals) * 3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def write_report(a, res: dict, outputs: list[dict], prompts, sheet_path: Path) -> None:
    a1, agg, cal, final = res["a1"], res["generation"], res["calibration"], res["final"]
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
    best_ppl = math.exp(a1["best.pt"]["test"])
    md = [
        "# Day 13 report (part A): final automatic evaluation",
        "",
        f"Generated by `scripts/day13_final_eval.py` ({SCRIPT_VERSION}) on {res['date']}. Machine: {res['machine']}, "
        f"{res['device']}, PyTorch {torch.__version__}, Python {platform.python_version()}.",
        "",
        f"Final model: **`{final['final_checkpoint']}`** ({final['size']}, dropout {final['dropout']}, "
        f"{final['parameters']:,} parameters, step {final['best_step']}). Evaluated as in the task 6 document. "
        "**The test text is used here for the first time**; every earlier choice used only the validation text.",
        "",
        "## A1. Perplexity",
        "",
        f"![test perplexity]({fig_rel})",
        "",
        "| Model | Validation loss | Validation perplexity | Test loss | Test perplexity |",
        "| --- | --- | --- | --- | --- |",
        *(f"| {r['label']} | {r['val']:.4f} | {math.exp(r['val']):,.2f} | {r['test']:.4f} | {math.exp(r['test']):,.2f} |"
          for r in a1.values()),
        *(f"| (reference) other seed {k} best.pt, step {r['step']} | {r['val']:.4f} | {math.exp(r['val']):,.2f} | "
          f"{r['test']:.4f} | {math.exp(r['test']):,.2f} |" for k, r in res["other_seeds"].items()),
        "",
        f"- Against the untrained model: test perplexity {best_ppl:,.2f} vs "
        f"{math.exp(a1['untrained']['test']):,.0f}.",
        f"- Against counting (bigram): {math.exp(a1['bigram']['test']):,.2f}. The bigram model predicts each token "
        "from the previous token only. The difference shows what the Transformer adds by reading the whole "
        "beginning of the sentence.",
        "- The other seeds were evaluated on the test text only after the final model was fixed; they show how much "
        "the test result depends on the seed, and played no part in any choice.",
        "",
        f"## A2-A5 (and two extra checks) on the {len(prompts)} fixed beginnings",
        "",
        f"Generation: max 40 tokens, seed 42; greedy + limits, and sampling (temperature 0.8, top-k 50, top-p 0.95). "
        f"Guidelines (task 6, Table 10): finished >= {MIN_FINISHED:.0%}, repetitive <= {MAX_REPETITIVE:.0%}.",
        "",
        "| Model | Method | A2 finished | A3 repetitive | A4 distinct-2 | A5 copy (reference) | Wrong unit | "
        "Broken number | Unseen value (reference) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(f"| {k.split('|')[0]} | {k.split('|')[1]} | {v['finished']:.0%} | {v['repetitive']:.0%} | "
          f"{v['distinct2']:.2f} | {v['copy']:.0%} | {v['wrong_unit']:.0%} | {v['broken_number']:.0%} | "
          f"{v['unseen_value']:.0%} |" for k, v in agg.items()),
        "",
        f"The extra checks were first run on the {cal['questions']:,} real test questions: wrong unit "
        f"{cal['wrong_unit']}, broken number {cal['broken_number']}, unseen value {cal['unseen_value']}. So a flag "
        f"on a generated sentence is a real problem, not a false alarm. (Allowed units learned for "
        f"{cal['items_learned']} item names; digit limit {cal['digit_limit']}, the longest number in the training "
        "questions.) Unseen value is a reference: an invented value such as 30000万円 is almost always wrong, but "
        "a new value is not wrong in itself.",
        "",
        f"\"Wording not in training\": {res['unseen_never_in_training']} of "
        f"{sum(p.kind == 'unseen' for p in prompts)} of these beginnings never start a training question.",
        "",
        "### By kind of beginning (final model, best.pt)",
        "",
        "| Kind | Count | Finished (greedy / sample) | Wrong unit (greedy / sample) | Copied (greedy / sample) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind, (en, ja, _) in PROMPT_KINDS.items():
        g = [r for r in outputs if r["model"] == "best.pt" and r["kind"] == kind and r["method"] == "greedy+limits"]
        s = [r for r in outputs if r["model"] == "best.pt" and r["kind"] == kind and r["method"] == "sample"]
        if not g:
            continue
        md.append(f"| {en} ({ja}) | {len(g)} | {sum(r['finished'] for r in g)} / {sum(r['finished'] for r in s)} | "
                  f"{sum(bool(r['wrong_unit']) for r in g)} / {sum(bool(r['wrong_unit']) for r in s)} | "
                  f"{sum(r['copy'] for r in g)} / {sum(r['copy'] for r in s)} |")
    md += [
        "",
        "## All outputs of the final model",
        "",
        "Flags: **U** wrong unit, **B** broken number, **V** unseen value, **R** repetitive, **C** exact copy of a "
        "training question, … = did not finish.",
        "",
        "| Kind | Beginning | Greedy + limits | Sampling |",
        "| --- | --- | --- | --- |",
    ]

    def show(r: dict) -> str:
        flags = "".join(f for f, on in (("U", r["wrong_unit"]), ("B", r["broken_number"]),
                                        ("V", r["unseen_value"]), ("R", r["repetitive"]), ("C", r["copy"])) if on)
        return _cell(r["text"] + ("" if r["finished"] else " …")) + (f" **[{flags}]**" if flags else "")

    for p in prompts:
        g = next(r for r in outputs if r["model"] == "best.pt" and r["prompt"] == p.prompt and r["method"] == METHODS[0])
        s = next(r for r in outputs if r["model"] == "best.pt" and r["prompt"] == p.prompt and r["method"] == METHODS[1])
        md.append(f"| {p.kind} | {_cell(p.prompt)} | {show(g)} | {show(s)} |")
    md += [
        "",
        "## Next: human scoring (part B)",
        "",
        f"`{sheet_path.relative_to(REPO_ROOT).as_posix() if sheet_path.is_relative_to(REPO_ROOT) else sheet_path}` "
        f"holds the {res['scored_outputs']} outputs of the final model (both methods), shuffled, with the method "
        "hidden. Two or more scorers each fill in their own copy (docs/day13/scores/scores_<initials>.xlsx); then "
        "`scripts/day13_score_summary.py` writes the final evaluation table (`docs/DAY13_EVALUATION.md`).",
        "",
        "last.pt is evaluated automatically only (A1-A5); human scoring covers the final model, so that each scorer "
        "reads 100 outputs, not 200.",
        "",
    ]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
