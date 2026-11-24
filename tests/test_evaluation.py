"""Tests for src/evaluation.py (Day 13). Run: .venv\\Scripts\\python.exe -m pytest tests -q"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import (PROMPT_KINDS, BigramBaseline, ScoreFileError, ScoringRow, UnitChecker,
                            agreement_within_one, cohen_kappa, has_broken_number, load_eval_prompts, max_digit_run,
                            read_scores, write_scoring_workbook)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHRASES = ["月平均残業時間", "残業時間", "勤続年数", "年収", "年齢", "賞与を除いた年収"]
TRAIN = ["年収が800万円以上の社員は？", "勤続年数が10年以上の人", "年齢が30歳未満の社員",
         "月平均残業時間が20時間を超える人", "賞与を除いた年収が500万円以上の社員", "社員番号100003の人"]


def test_eval_prompt_list_matches_task6_table7() -> None:
    prompts = load_eval_prompts(REPO_ROOT / "prompts/day13_eval_prompts.csv")
    assert len(prompts) == 50
    for kind, (_, _, count) in PROMPT_KINDS.items():
        assert sum(p.kind == kind for p in prompts) == count


def test_units_are_learned_from_the_training_questions() -> None:
    uc = UnitChecker(PHRASES).learn(TRAIN)
    assert uc.allowed["年収"] == {"円"} and uc.allowed["勤続年数"] == {"年"} and uc.allowed["年齢"] == {"歳"}
    # the longest name wins: 月平均残業時間 is not read as 残業時間, 賞与を除いた年収 not as 年収
    assert "月平均残業時間" in uc.allowed and "残業時間" not in uc.allowed
    assert "賞与を除いた年収" in uc.allowed


def test_wrong_unit_is_flagged_and_correct_units_are_not() -> None:
    uc = UnitChecker(PHRASES).learn(TRAIN)
    assert uc.mismatches("勤続年数が300万円以上の社員を教えてください。") == [("勤続年数", "円")]
    assert uc.mismatches("年齢が30年以上の社員") == [("年齢", "年")]
    assert uc.mismatches("年収が700万円未満で勤続年数が5年以上の人") == []
    assert uc.mismatches("残業時間が10時間") == []          # never seen with a unit in training: not judged
    assert all(uc.mismatches(q) == [] for q in TRAIN)       # no false alarms on the training questions
    assert uc.unseen_values("年収が30000万円以下の社員") == ["30000万円"]   # a value no question has
    assert uc.unseen_values("年収が800万円以上で年齢が30歳未満") == []


def test_broken_number_uses_the_longest_number_in_training() -> None:
    limit = max_digit_run(TRAIN)
    assert limit == 6                                        # 社員番号100003
    assert has_broken_number("ボーナスが30000100万円未満の社員は？", limit)
    assert not has_broken_number("年収が800万円以上", limit)
    assert not has_broken_number("社員番号100003の人", limit)


def test_bigram_baseline_is_a_proper_distribution_and_beats_uniform() -> None:
    rng = np.random.default_rng(0)
    vocab = 12
    stream = np.array([1, 4, 5, 6, 2] * 200 + list(rng.integers(3, vocab, 200)), dtype=np.uint16)
    b = BigramBaseline.fit(stream, vocab)
    lam = b.tune(stream[:300])
    for prev in range(vocab):            # probabilities of every next token sum to 1
        p = [math.exp(-b.loss(np.array([prev, nxt]), lam)) for nxt in range(vocab)]
        assert sum(p) == pytest.approx(1.0)
    assert b.loss(stream[:500]) < math.log(vocab)


def test_agreement_and_kappa_known_values() -> None:
    a = [5, 4, 3, 2, 1, 5]
    assert agreement_within_one(a, a) == 1.0 and cohen_kappa(a, a) == pytest.approx(1.0)
    b = [4, 4, 1, 2, 3, 5]               # differences 1, 0, 2, 0, 2, 0
    assert agreement_within_one(a, b) == pytest.approx(4 / 6)
    # exact agreement 3/6; chance = sum over scores 1-5 of share_a * share_b
    #   score 5: 2/6 x 1/6, 4: 1/6 x 2/6, 3: 1/6 x 1/6, 2: 1/6 x 1/6, 1: 1/6 x 1/6  ->  7/36
    p_e = 7 / 36
    assert cohen_kappa(a, b) == pytest.approx((0.5 - p_e) / (1 - p_e))      # = 11/29


def test_scoring_workbook_round_trip_and_problem_report(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    rows = [ScoringRow("S001", "年収が", "高い社員は？", False), ScoringRow("S002", "今日は", "何人いますか。", True)]
    sheet = tmp_path / "scoring_sheet.xlsx"
    write_scoring_workbook(sheet, rows)

    wb = openpyxl.load_workbook(sheet)
    ws = wb["採点 Scoring"]
    assert ws["G3"].value == "-"                         # out of scope: H4 is not scored
    for col, v in zip("DEFG", (5, 4, 5, 4)):
        ws[f"{col}2"] = v
    for col, v in zip("DEF", (3, 2, 1)):
        ws[f"{col}3"] = v
    ws["H3"] = "O, t"
    filled = tmp_path / "scores_ab.xlsx"
    wb.save(filled)
    got = read_scores(filled, rows)
    assert got["S001"].h == {"H1": 5, "H2": 4, "H3": 5, "H4": 4}
    assert got["S002"].h["H4"] is None and got["S002"].types == {"O", "T"}

    ws["E2"] = None                                      # a missing score, a bad value and a bad code
    ws["F3"] = 7
    ws["H2"] = "X"
    wb.save(filled)
    with pytest.raises(ScoreFileError) as e:
        read_scores(filled, rows)
    msg = str(e.value)
    assert "S001: H2 is empty" in msg and "S002: H3 must be a whole number 1-5" in msg and "['X']" in msg
