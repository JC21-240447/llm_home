r"""Choosing the model: summarise trial runs and apply the written decision rule (Day 11).
モデルの選び方: 試験の学習をまとめ、決めておいたルールで選ぶ(Day 11)

==========================  ENGLISH  ==========================

summarize_run(): reads one run's log.csv (Day 9 format) and returns the numbers the
decision needs: lowest validation loss and its step, the training loss at that step
(the gap between them shows memorising), the final validation loss (did it rise
afterwards?), speed and GPU memory.

choose(): the decision rule, written down BEFORE looking at the results:
  1. A run is a candidate only if its best checkpoint passes the task 6 automatic
     guidelines on the fixed prompts: finished >= 90 %, repetitive <= 5 %.
  2. Among candidates, find the lowest validation loss.
  3. Every candidate within `tolerance` (0.02) of that loss counts as "as good".
  4. Among those, take the one with the FEWEST parameters (smaller is faster, cheaper
     and memorises less); if still tied, the lower validation loss.
Why prefer smaller: a difference smaller than 0.02 in loss (about 2 % in perplexity)
is within run-to-run noise, so it is not evidence that the bigger model is better.

==========================  日本語  ==========================

summarize_run(): 1つの学習の log.csv(Day 9 の形式)を読み、決めるのに必要な数をまとめる:
検証の損失の最低値とそのステップ、そのときの学習の損失(差が「覚えすぎ」を表す)、
最後の検証の損失(その後に上がったか)、速さ、GPU メモリ。

choose(): 結果を見る前に決めておいたルール:
  1. 一番良いチェックポイントが、決まった書き出しで作業6の目安(完結 90%以上、繰り返し 5%以下)を
     満たすものだけを候補にする
  2. 候補の中で、検証の損失の最低値を見つける
  3. それとの差が 0.02 以内の候補は「同じくらい良い」とみなす
  4. その中で、パラメータが一番少ないものを選ぶ(同じなら検証の損失が低い方)
小さい方を選ぶ理由: 損失の差 0.02(パープレキシティで約2%)は、実行ごとのゆれの範囲なので、
大きいモデルの方が良いという証拠にならない。
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOLERANCE: float = 0.02
MIN_FINISHED: float = 0.90
MAX_REPETITIVE: float = 0.05


@dataclass
class RunSummary:
    name: str
    size: str
    dropout: float
    parameters: int
    best_val: float
    best_step: int
    train_at_best: float | None
    final_val: float
    final_step: int
    tokens_per_sec: float | None       # median over the run
    peak_vram_mb: float | None
    elapsed_s: float
    finished_rate: float | None = None  # filled in after the generation check
    repetitive_rate: float | None = None
    distinct2: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float | None:
        """Validation minus training loss at the best step: how much worse unseen wording is."""
        return None if self.train_at_best is None else self.best_val - self.train_at_best

    @property
    def rise_after_best(self) -> float:
        """How much the validation loss rose from its best to the end (0 = never rose)."""
        return self.final_val - self.best_val

    def passes_checks(self) -> bool:
        if self.finished_rate is None or self.repetitive_rate is None:
            return False
        return self.finished_rate >= MIN_FINISHED and self.repetitive_rate <= MAX_REPETITIVE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gap"], d["rise_after_best"] = self.gap, self.rise_after_best
        return d


def _f(x: str) -> float | None:
    return float(x) if x not in ("", None) else None


def summarize_run(log_path: Path, name: str, size: str, dropout: float, parameters: int) -> RunSummary:
    with log_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    evals = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["event"] == "eval" and r["val_loss"]]
    if not evals:
        raise ValueError(f"{log_path}: no validation results")
    trains = {int(r["step"]): float(r["train_loss"]) for r in rows if r["event"] == "train" and r["train_loss"]}
    best_step, best_val = min(evals, key=lambda e: (e[1], e[0]))
    final_step, final_val = evals[-1]
    tps = sorted(v for v in (_f(r["tokens_per_sec"]) for r in rows if r["event"] == "train") if v)
    vram = [v for v in (_f(r["peak_vram_mb"]) for r in rows) if v is not None]
    elapsed = max((float(r["elapsed_s"]) for r in rows if r["elapsed_s"]), default=0.0)
    return RunSummary(
        name=name, size=size, dropout=dropout, parameters=parameters, best_val=best_val, best_step=best_step,
        train_at_best=trains.get(best_step), final_val=final_val, final_step=final_step,
        tokens_per_sec=tps[len(tps) // 2] if tps else None, peak_vram_mb=max(vram) if vram else None,
        elapsed_s=elapsed,
    )


@dataclass
class Decision:
    chosen: RunSummary | None
    lowest_val: float | None
    as_good: list[str]
    rejected: dict[str, str]
    reason: str


def choose(runs: list[RunSummary], tolerance: float = TOLERANCE) -> Decision:
    rejected = {r.name: (f"finished {r.finished_rate:.0%}, repetitive {r.repetitive_rate:.0%}"
                         if r.finished_rate is not None else "no generation check")
                for r in runs if not r.passes_checks()}
    candidates = [r for r in runs if r.passes_checks()]
    if not candidates:
        return Decision(None, None, [], rejected, "no run passed the task 6 automatic guidelines")
    lowest = min(r.best_val for r in candidates)
    as_good = [r for r in candidates if r.best_val - lowest <= tolerance]
    chosen = min(as_good, key=lambda r: (r.parameters, r.best_val, r.dropout))
    best_overall = min(candidates, key=lambda r: r.best_val)
    if chosen is best_overall:
        reason = f"{chosen.name} has the lowest validation loss ({chosen.best_val:.4f})"
        if len(as_good) > 1:
            reason += " and is also the smallest of the runs within the tolerance"
    else:
        reason = (f"{best_overall.name} has the lowest validation loss ({best_overall.best_val:.4f}), but "
                  f"{chosen.name} ({chosen.best_val:.4f}) is within {tolerance} of it with fewer parameters "
                  f"({chosen.parameters:,} vs {best_overall.parameters:,}), so the smaller model is chosen")
    return Decision(chosen, lowest, [r.name for r in as_good], rejected, reason)
