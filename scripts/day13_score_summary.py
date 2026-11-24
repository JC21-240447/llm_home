r"""Day 13 (part B) - read the human scores and write the FINAL EVALUATION TABLE (completes task 11).
Day 13(後半) - 人の採点を読み、最終の評価の表を書く(作業11の完了)

==========================  ENGLISH  ==========================

Reads every docs/day13/scores/scores_<initials>.xlsx (one per scorer, copied from
docs/day13/scoring_sheet.xlsx and filled in), checks them (every score present and 1-5,
codes valid, text unchanged), and reports ALL problems at once, so a scorer can fix
everything in one go.

Then, following task 6 (sections 6-4 and 7-3):
    - H1-H4 averages (each output's score = the mean over scorers), per method and overall
    - share of outputs with H3 <= 2 (an output that makes no sense is a failure)
    - agreement between scorers: share within 1 point, and Cohen's kappa, per item
    - problem types (Table 9), by kind of beginning
    - the final evaluation table against the pass guidelines (Table 10), with part A's numbers

Writes docs/DAY13_EVALUATION.md and docs/figures/day13_human_scores.png.

Usage (VS Code terminal, repo root):
    .venv\Scripts\python.exe scripts\day13_score_summary.py

SERVER NOTE: same code;  .venv/bin/python scripts/day13_score_summary.py

==========================  日本語  ==========================

docs/day13/scores/scores_<イニシャル>.xlsx(採点者ごとに1つ)をすべて読み、確かめ(すべての点が
1〜5であるか、記号が正しいか、文が変えられていないか)、問題をまとめて一度に表示する。
その後、作業6(6-4、7-3)のとおりに、H1〜H4の平均、H3が2以下の割合、採点者の一致率と
カッパ係数、問題の種類を計算し、合格の目安と比べた最終の評価の表を書く。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (H_ITEMS, PROBLEM_TYPES, PROMPT_KINDS, Score, ScoreFileError, ScoringRow,  # noqa: E402
                            agreement_within_one, cohen_kappa, mean, read_scores)

SCRIPT_VERSION: str = "day13-v1"
MIN_H_MEAN: float = 3.5
MAX_H3_LOW: float = 0.10
MIN_AGREEMENT: float = 0.70
H_NAMES: dict[str, str] = {"H1": "H1 correct Japanese", "H2": "H2 natural", "H3": "H3 makes sense",
                           "H4": "H4 company terms"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 13 part B: human scores -> final evaluation table.")
    ap.add_argument("--scores-dir", type=Path, default=REPO_ROOT / "docs/day13/scores")
    ap.add_argument("--key", type=Path, default=REPO_ROOT / "docs/day13/scoring_key.json")
    ap.add_argument("--auto", type=Path, default=REPO_ROOT / "docs/day13/auto_results.json")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY13_EVALUATION.md")
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day13_human_scores.png")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def yes_no(ok: bool | None) -> str:
    return "-" if ok is None else ("**yes**" if ok else "**no**")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    key = json.loads(a.key.read_text(encoding="utf-8"))["rows"]
    auto = json.loads(a.auto.read_text(encoding="utf-8"))
    expected = [ScoringRow(k["id"], k["prompt"], k["text"], k["out_of_scope"]) for k in key]
    files = sorted(a.scores_dir.glob("scores_*.xlsx"))
    if not files:
        raise SystemExit(f"No scores_*.xlsx in {a.scores_dir}. Each scorer copies docs/day13/scoring_sheet.xlsx "
                         "there as scores_<initials>.xlsx and fills it in.")
    scorers: dict[str, dict[str, Score]] = {}
    errors: list[str] = []
    for f in files:
        try:
            scorers[f.stem.removeprefix("scores_")] = read_scores(f, expected)
        except ScoreFileError as e:
            errors.append(str(e))
    if errors:
        print("The score files have problems; nothing was written. Fix these and run again:\n")
        print("\n\n".join(errors))
        return 1
    names = list(scorers)
    print(f"=== Day 13 part B | {len(names)} scorer(s): {', '.join(names)} | {len(key)} outputs ===")
    if len(names) < 2:
        print("WARNING: task 6 asks for two or more scorers; agreement cannot be measured with one.")

    # ---- per output: mean over scorers
    per_out: dict[str, dict[str, float | None]] = {}
    for k in key:
        per_out[k["id"]] = {}
        for h in H_ITEMS:
            vals = [scorers[n][k["id"]].h[h] for n in names if scorers[n][k["id"]].h[h] is not None]
            per_out[k["id"]][h] = mean(vals) if vals else None

    def h_means(rows: list[dict]) -> dict[str, float]:
        return {h: mean(per_out[r["id"]][h] for r in rows if per_out[r["id"]][h] is not None) for h in H_ITEMS}

    def h3_low(rows: list[dict]) -> float:
        return sum(per_out[r["id"]]["H3"] <= 2 for r in rows) / len(rows)

    groups = {"greedy+limits": [k for k in key if k["method"] == "greedy+limits"],
              "sample": [k for k in key if k["method"] == "sample"], "both": key}
    means = {g: h_means(rows) for g, rows in groups.items()}
    lows = {g: h3_low(rows) for g, rows in groups.items()}

    # ---- agreement between every pair of scorers
    agreement: dict[str, dict[str, float]] = {}
    if len(names) >= 2:
        for h in H_ITEMS:
            ag, kp = [], []
            for x, y in itertools.combinations(names, 2):
                ids = [k["id"] for k in key if scorers[x][k["id"]].h[h] is not None]
                ax = [scorers[x][i].h[h] for i in ids]
                ay = [scorers[y][i].h[h] for i in ids]
                ag.append(agreement_within_one(ax, ay))
                kp.append(cohen_kappa(ax, ay))
            agreement[h] = {"within_one": mean(ag), "kappa": mean(kp)}

    # ---- problem types (an output counts if any scorer gave the type)
    types_of = {k["id"]: set().union(*(scorers[n][k["id"]].types for n in names)) for k in key}
    type_counts = {code: sum(code in types_of[k["id"]] for k in key) for code in PROBLEM_TYPES}

    for g in groups:
        print(f"  {g:<14} " + " | ".join(f"{h} {means[g][h]:.2f}" for h in H_ITEMS) + f" | H3<=2 {lows[g]:.0%}")
    for h, v in agreement.items():
        print(f"  agreement {h}: within 1 point {v['within_one']:.0%}, kappa {v['kappa']:.2f}")

    draw_figure(means, a.figure)
    write_report(a, key, auto, names, scorers, per_out, means, lows, agreement, type_counts, types_of)
    print(f"\nFinal evaluation table: {a.report}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | Day 13 human scoring summary\n\n"
                    f"- Script: scripts/day13_score_summary.py ({SCRIPT_VERSION}); scorers: {len(names)}\n"
                    "- H1-H4 (both methods): " + ", ".join(f"{h} {means['both'][h]:.2f}" for h in H_ITEMS)
                    + f"; H3<=2: {lows['both']:.0%}\n"
                    + ("- Agreement within 1 point: " + ", ".join(f"{h} {v['within_one']:.0%}"
                                                                  for h, v in agreement.items()) + "\n"
                       if agreement else "- Agreement: not measured (one scorer)\n"))
        print("Appended an entry to run_log.md")
    return 0


def draw_figure(means: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    width = 0.38
    xs = range(len(H_ITEMS))
    for i, (g, c) in enumerate((("greedy+limits", "#1f77b4"), ("sample", "#ff7f0e"))):
        vals = [means[g][h] for h in H_ITEMS]
        bars = ax.bar([x + (i - 0.5) * width for x in xs], vals, width, label=g, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(MIN_H_MEAN, color="#d62728", ls="--", lw=1, label=f"guideline {MIN_H_MEAN}")
    ax.set_xticks(list(xs), [H_NAMES[h] for h in H_ITEMS])
    ax.set_ylim(1, 5.4)
    ax.set_ylabel("mean score (1-5)")
    ax.set_title("Human scores of the final model")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def write_report(a, key, auto, names, scorers, per_out, means, lows, agreement, type_counts, types_of) -> None:
    a1, gen, final = auto["a1"], auto["generation"], auto["final"]
    g_best = {m: gen[f"best.pt|{m}"] for m in ("greedy+limits", "sample")}
    test_ppl = math.exp(a1["best.pt"]["test"])
    untrained_ppl = math.exp(a1["untrained"]["test"])
    bigram_ppl = math.exp(a1["bigram"]["test"])
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
    finished_ok = all(v["finished"] >= auto["guidelines"]["min_finished"] for v in g_best.values())
    rep_ok = all(v["repetitive"] <= auto["guidelines"]["max_repetitive"] for v in g_best.values())
    agree_ok = (all(v["within_one"] >= MIN_AGREEMENT for v in agreement.values()) if agreement else None)
    rows = [
        ("A1 Perplexity (test)", "far below the untrained value",
         f"{test_ppl:.2f} (untrained {untrained_ppl:,.0f}; counting baseline {bigram_ppl:.2f})",
         test_ppl < untrained_ppl / 10),
        ("A2 Completion rate", ">= 90 %", " / ".join(f"{v['finished']:.0%}" for v in g_best.values()), finished_ok),
        ("A3 Repetition rate", "<= 5 %", " / ".join(f"{v['repetitive']:.0%}" for v in g_best.values()), rep_ok),
        *((H_NAMES[h] + " (mean)", f">= {MIN_H_MEAN}",
           f"{means['greedy+limits'][h]:.2f} / {means['sample'][h]:.2f} (all {means['both'][h]:.2f})",
           means["both"][h] >= MIN_H_MEAN) for h in H_ITEMS),
        ("Outputs with H3 <= 2", "<= 10 %", f"{lows['greedy+limits']:.0%} / {lows['sample']:.0%} "
                                            f"(all {lows['both']:.0%})", lows["both"] <= MAX_H3_LOW),
        ("Agreement (within 1 point)", ">= 70 % per item",
         (", ".join(f"{h} {v['within_one']:.0%}" for h, v in agreement.items()) if agreement
          else "not measured (one scorer)"), agree_ok),
    ]
    passed = sum(r[3] is True for r in rows)
    judged = sum(r[3] is not None for r in rows)
    md = [
        "# Day 13: final evaluation of Model A (task 11)",
        "",
        f"Written by `scripts/day13_score_summary.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}, from "
        f"part A (`docs/DAY13_AUTO_EVALUATION.md`, {auto['date']}) and {len(names)} scorer(s).",
        "",
        f"Final model: `{final['final_checkpoint']}` ({final['size']}, dropout {final['dropout']}, "
        f"{final['parameters']:,} parameters). Criteria: the task 6 document (Tables 5-10), unchanged.",
        "",
        "## Final evaluation table",
        "",
        "Values for generation are shown as greedy + limits / sampling.",
        "",
        "| Item | Guideline | Result | Meets the guideline |",
        "| --- | --- | --- | --- |",
        *(f"| {r[0]} | {r[1]} | {r[2]} | {yes_no(r[3])} |" for r in rows),
        "",
        f"**{passed} of {judged} judged items meet the guidelines.**",
        "",
        "## Human scores",
        "",
        f"![human scores]({fig_rel})",
        "",
        f"{len(key)} outputs of the final model (50 beginnings x 2 methods), shuffled, method hidden; each output's "
        f"score is the mean over the scorers ({', '.join(names)}). H4 is not scored for out-of-scope beginnings.",
        "",
        "| Method | " + " | ".join(H_NAMES[h] for h in H_ITEMS) + " | H3 <= 2 |",
        "| --- | --- | --- | --- | --- | --- |",
        *(f"| {g} | " + " | ".join(f"{means[g][h]:.2f}" for h in H_ITEMS) + f" | {lows[g]:.0%} |"
          for g in ("greedy+limits", "sample", "both")),
        "",
        "### By kind of beginning (both methods)",
        "",
        "| Kind | Outputs | " + " | ".join(H_ITEMS) + " | H3 <= 2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for kind, (en, ja, _) in PROMPT_KINDS.items():
        rs = [k for k in key if k["kind"] == kind]
        if not rs:
            continue
        hm = {h: mean(per_out[r["id"]][h] for r in rs if per_out[r["id"]][h] is not None) for h in H_ITEMS}
        md.append(f"| {en} ({ja}) | {len(rs)} | " + " | ".join("-" if math.isnan(hm[h]) else f"{hm[h]:.2f}"
                                                               for h in H_ITEMS)
                  + f" | {sum(per_out[r['id']]['H3'] <= 2 for r in rs) / len(rs):.0%} |")
    md += ["", "### Agreement between scorers", ""]
    if agreement:
        md += ["| Item | Within 1 point | Cohen's kappa |", "| --- | --- | --- |",
               *(f"| {H_NAMES[h]} | {v['within_one']:.0%} | {v['kappa']:.2f} |" for h, v in agreement.items()),
               "", "Within 1 point is the guideline (task 6, 6-4). Kappa counts only exact agreement and removes "
               "agreement expected by chance (1 = perfect, 0 = chance), so it is a stricter reference. When most "
               "scores are 4 or 5, two scorers are often within 1 point by chance alone, so read both together."]
        low_k = [h for h, v in agreement.items() if v["kappa"] < 0.2]
        if low_k:
            md += ["", f"**Kappa is below 0.2 for {', '.join(low_k)}**: beyond chance, the scorers agree little on "
                   "these items, even where the within-1 rate meets the guideline. As task 6 says, the scorers "
                   "should compare a few of the outputs where they differed and make the criteria wording clearer."]
    else:
        md.append("Not measured: only one scorer. Task 6 asks for two or more.")
    md += [
        "",
        "### Problem types (outputs given the type by at least one scorer)",
        "",
        "| Type | Outputs | Automatic check (part A, final model, greedy / sample) |",
        "| --- | --- | --- |",
    ]
    auto_note = {"T": " / ".join(f"wrong unit {v['wrong_unit']:.0%}" for v in g_best.values()),
                 "R": " / ".join(f"repetitive {v['repetitive']:.0%}" for v in g_best.values()),
                 "U": " / ".join(f"unfinished {1 - v['finished']:.0%}" for v in g_best.values())}
    for code, (en, ja) in PROBLEM_TYPES.items():
        md.append(f"| {code} {en} ({ja}) | {type_counts[code]} | {auto_note.get(code, '-')} |")

    def mean_all(i: str) -> float:
        v = [x for x in per_out[i].values() if x is not None]
        return sum(v) / len(v)

    ranked = sorted(key, key=lambda k: (mean_all(k["id"]), k["id"]))
    md += ["", "### Lowest and highest scored outputs", "",
           "| | Method | Beginning | Text | Mean score | Types | Notes |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, part in (("low", ranked[:8]), ("high", ranked[-5:][::-1])):
        for k in part:
            notes = "; ".join(scorers[n][k["id"]].note for n in names if scorers[n][k["id"]].note)
            md.append(f"| {label} | {k['method']} | {_cell(k['prompt'])} | {_cell(k['text'])} | "
                      f"{mean_all(k['id']):.2f} | {','.join(sorted(types_of[k['id']])) or '-'} | {_cell(notes) or '-'} |")
    md += [
        "",
        "## Notes",
        "",
        "- Generation values are for the final model (best.pt). last.pt is in part A (A1-A5 only).",
        "- The guidelines were set in task 6 before this evaluation and are not changed here. If a guideline is "
        "revised later, the change is recorded with its date, because scores under different criteria cannot be "
        "compared directly.",
        "",
    ]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
