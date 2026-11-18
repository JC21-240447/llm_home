"""Unit tests for src/selection.py (Day 11): run summaries and the size decision rule. CPU only, instant.
src/selection.py の単体テスト(Day 11): 学習のまとめと、大きさを選ぶルール。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.selection import RunSummary, choose, summarize_run  # noqa: E402
from src.training import LOG_COLUMNS  # noqa: E402


def run(name: str, params: int, best: float, finished: float = 1.0, repetitive: float = 0.0,
        dropout: float = 0.0) -> RunSummary:
    return RunSummary(name=name, size=name.split("_")[0], dropout=dropout, parameters=params, best_val=best,
                      best_step=100, train_at_best=1.0, final_val=best + 0.1, final_step=800, tokens_per_sec=None,
                      peak_vram_mb=None, elapsed_s=10.0, finished_rate=finished, repetitive_rate=repetitive,
                      distinct2=0.5)


def test_summary_reads_best_point_gap_and_rise(tmp_path: Path) -> None:
    log = tmp_path / "log.csv"
    rows = [
        {"step": 0, "event": "eval", "val_loss": "7.40"},
        {"step": 25, "event": "train", "train_loss": "3.00", "tokens_per_sec": "100"},
        {"step": 25, "event": "eval", "val_loss": "2.80"},
        {"step": 50, "event": "train", "train_loss": "1.50", "tokens_per_sec": "300", "peak_vram_mb": "400"},
        {"step": 50, "event": "eval", "val_loss": "2.60"},
        {"step": 75, "event": "train", "train_loss": "1.20", "tokens_per_sec": "200"},
        {"step": 75, "event": "eval", "val_loss": "2.90", "elapsed_s": "12.5"},
    ]
    with log.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LOG_COLUMNS})
    s = summarize_run(log, "tiny_d00", "tiny", 0.0, 1000)
    assert (s.best_step, s.best_val, s.train_at_best) == (50, 2.60, 1.50)
    assert s.gap == pytest.approx(1.10)
    assert (s.final_step, s.final_val) == (75, 2.90)
    assert s.rise_after_best == pytest.approx(0.30)
    assert s.tokens_per_sec == 200 and s.peak_vram_mb == 400 and s.elapsed_s == 12.5


def test_lowest_loss_wins_when_clearly_better() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.70), run("small_d00", 5_000_000, 2.60)])
    assert d.chosen.name == "small_d00"


def test_smaller_model_wins_within_tolerance() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.615), run("small_d00", 5_000_000, 2.600),
                run("medium_d00", 15_000_000, 2.640)])
    assert d.chosen.name == "tiny_d00"            # 0.015 worse than the best: within 0.02, and 5x smaller
    assert set(d.as_good) == {"tiny_d00", "small_d00"}
    assert "smaller" in d.reason


def test_runs_failing_the_task6_checks_are_never_chosen() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.50, finished=0.80), run("small_d00", 5_000_000, 2.60),
                run("medium_d00", 15_000_000, 2.55, repetitive=0.10)])
    assert d.chosen.name == "small_d00"
    assert set(d.rejected) == {"tiny_d00", "medium_d00"}


def test_no_candidate_gives_no_choice() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.50, finished=0.5)])
    assert d.chosen is None and "no run passed" in d.reason


def test_same_size_tie_prefers_lower_loss_then_no_dropout() -> None:
    d = choose([run("tiny_d10", 1_000_000, 2.605, dropout=0.1), run("tiny_d00", 1_000_000, 2.600)])
    assert d.chosen.name == "tiny_d00"
