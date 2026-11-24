# Day 13.2 — test_evaluation.py — やさしい1行ずつの解説(日本語)

> **Day 13 の4つのファイルの関係**
> - **Day 13.1 `evaluation.py`** — テストされるコード(**先に読んでください**)。
> - **Day 13 `day13_final_eval.py`**(前半)と **`day13_score_summary.py`**(後半)— 2つの運転手。
> - **Day 13.2 `test_evaluation.py`(このファイル)** — 道具箱の**7個の自動テスト**。CPUのみで数秒。
>
> pytestの基本は、Day 6.2〜12.3で説明しました。ここでは新しいことだけを説明します。

---

## 0. 全体像(まずここから)

最終評価は、**教員の前で説明する**数字を作ります。このテストは、手で確かめられる小さな例で、道具が約束どおりに動くことを証明します:

| 約束 | テスト |
|---|---|
| 書き出しのリストが作業6の表7と合う(50個、種類ごとの数) | `test_eval_prompt_list_matches_task6_table7` |
| 単位は学習用の質問から**学ばれ**、一番長い名前が勝つ | `test_units_are_learned_from_the_training_questions` |
| 誤った単位は警告され、正しいもの(と学習用の質問そのもの)は警告されない | `test_wrong_unit_is_flagged_and_correct_units_are_not` |
| 壊れた数字の上限は、学習用のデータから来る | `test_broken_number_uses_the_longest_number_in_training` |
| バイグラムの基準は、正しい確率の分布で、一様より良い | `test_bigram_baseline_is_a_proper_distribution_and_beats_uniform` |
| 一致率とカッパが、手計算の既知の値になる | `test_agreement_and_kappa_known_values` |
| Excelのシートが読み書きでき、悪い採点ファイルの問題が**一度にすべて**報告される | `test_scoring_workbook_round_trip_and_problem_report` |

実行:`.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. 準備

```python
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
```
- **Day 13.1** の道具を読み込む。(`sys.path` の行は無い:このテストのファイルは、前に設定したとおり、pytestがプロジェクトのルートから `src` を見つけると想定している。)
- `PHRASES` — `columns.csv` のような項目名;**ほかの名前を含む名前**がわざと入っている(`月平均残業時間` の中の `残業時間`、`賞与を除いた年収` の中の `年収`)。`TRAIN` — 4つの単位と長い数字(社員番号100003)を含む、6つの小さな「学習用の質問」。

---

## 2. テスト

### 2.1 `test_eval_prompt_list_matches_task6_table7`

```python
    prompts = load_eval_prompts(REPO_ROOT / "prompts/day13_eval_prompts.csv")
    assert len(prompts) == 50
    for kind, (_, _, count) in PROMPT_KINDS.items():
        assert sum(p.kind == kind for p in prompts) == count
```
- **本物のプロジェクトのファイル**を読む唯一のテスト:本物の `prompts/day13_eval_prompts.csv` には、ちょうど **50個**の書き出しがあり、**各種類が作業6の求める数**(10/10/10/10/5/5)でなければならない。読み込みでは安全確認(既知の種類、重複なし)も動く。

### 2.2 `test_units_are_learned_from_the_training_questions`

```python
    uc = UnitChecker(PHRASES).learn(TRAIN)
    assert uc.allowed["年収"] == {"円"} and uc.allowed["勤続年数"] == {"年"} and uc.allowed["年齢"] == {"歳"}
    assert "月平均残業時間" in uc.allowed and "残業時間" not in uc.allowed
    assert "賞与を除いた年収" in uc.allowed
```
- `TRAIN` から学んだ後:年収 → 円だけ、勤続年数 → 年だけ、年齢 → 歳だけ(だれもこのルールを打ち込んでいない)。
- **一番長い名前が勝つ:**`月平均残業時間が20時間` は、短い `残業時間` ではなく `月平均残業時間` の下に保存される;同じく `賞与を除いた年収` は、ただの `年収` とは読まれない。(短い名前が勝ってしまうと、検出器は間違った項目と単位を比べてしまう。)

### 2.3 `test_wrong_unit_is_flagged_and_correct_units_are_not`

```python
    assert uc.mismatches("勤続年数が300万円以上の社員を教えてください。") == [("勤続年数", "円")]
    assert uc.mismatches("年齢が30年以上の社員") == [("年齢", "年")]
    assert uc.mismatches("年収が700万円未満で勤続年数が5年以上の人") == []
    assert uc.mismatches("残業時間が10時間") == []          # never seen with a unit in training: not judged
    assert all(uc.mismatches(q) == [] for q in TRAIN)       # no false alarms on the training questions
    assert uc.unseen_values("年収が30000万円以下の社員") == ["30000万円"]
    assert uc.unseen_values("年収が800万円以上で年齢が30歳未満") == []
```
- 勤続年数が円 → **警告**。年齢が年 → **警告**。正しい組が2つの文 → **警告なし**。
- `残業時間が10時間` → 警告なし。`残業時間` は**学ばれていない**(長い名前の中にしか出てこない)ので:「見たことのない項目は判定しない」。
- どの学習用の質問にも**誤警報なし** — 前半の、本物のテスト用の質問での較正と同じ考え方。
- `30000万円` は学習になかった → **学習にない値**として挙がる;`800万円` と `30歳` はあった → 空のリスト。

### 2.4 `test_broken_number_uses_the_longest_number_in_training`

```python
    limit = max_digit_run(TRAIN)
    assert limit == 6                                        # 社員番号100003
    assert has_broken_number("ボーナスが30000100万円未満の社員は？", limit)
    assert not has_broken_number("年収が800万円以上", limit)
    assert not has_broken_number("社員番号100003の人", limit)
```
- `TRAIN` の数字の並びの最長は `100003` → **6**。`30000100` は8桁 → **壊れている**。`800` は問題ない。`100003`(ちょうど6、本物の社員番号)は壊れて**いない**(ルールは「上限より長い」)。

### 2.5 `test_bigram_baseline_is_a_proper_distribution_and_beats_uniform`

```python
    rng = np.random.default_rng(0)
    vocab = 12
    stream = np.array([1, 4, 5, 6, 2] * 200 + list(rng.integers(3, vocab, 200)), dtype=np.uint16)
    b = BigramBaseline.fit(stream, vocab)
    lam = b.tune(stream[:300])
    for prev in range(vocab):
        p = [math.exp(-b.loss(np.array([prev, nxt]), lam)) for nxt in range(vocab)]
        assert sum(p) == pytest.approx(1.0)
    assert b.loss(stream[:500]) < math.log(vocab)
```
- 偽のトークンの列:`1 4 5 6 2` の並びを200回繰り返し(とても予測しやすい)、さらにランダムなトークン200個;語彙は12。
- **正しい分布:****どの**前のトークンについても、12個の次のトークンの確率は**合計が1**にならなければならない。コツ:2トークンの列 `[prev, nxt]` の損失はちょうど `−log P(nxt | prev)` なので、`exp(−loss)` が確率になる。(これで、add-oneのunigramと `lam` の混ぜ方が、見たことのない前のトークンの場合も含めて、正しいことも証明される。)
- **一様に勝つ:**予測しやすい列で、損失が `ln(12)`(一様な当てずっぽう)より低い。

### 2.6 `test_agreement_and_kappa_known_values`

```python
    a = [5, 4, 3, 2, 1, 5]
    assert agreement_within_one(a, a) == 1.0 and cohen_kappa(a, a) == pytest.approx(1.0)
    b = [4, 4, 1, 2, 3, 5]               # differences 1, 0, 2, 0, 2, 0
    assert agreement_within_one(a, b) == pytest.approx(4 / 6)
    p_e = 7 / 36
    assert cohen_kappa(a, b) == pytest.approx((0.5 - p_e) / (1 - p_e))      # = 11/29
```
- 同じ採点者 → 一致率1.0、カッパ1.0。
- 採点者 `b` は `a` と 1、0、2、0、2、0 だけ違う → **6個中4個**で1点以内。
- **カッパの手計算:**完全な一致 = 3/6 = 0.5(2番目、4番目、6番目)。偶然の一致(aの割合 × bの割合を点ごとに足す):5点 → 2/6 × 1/6、4点 → 1/6 × 2/6、3・2・1点 → それぞれ 1/6 × 1/6 → **7/36**。カッパ = (0.5 − 7/36) / (1 − 7/36) = (11/36) / (29/36) = **11/29 ≈ 0.38**。テストは手計算をコメントに書いているので、コードが**独立した計算**と照らされる。

### 2.7 `test_scoring_workbook_round_trip_and_problem_report`

```python
    openpyxl = pytest.importorskip("openpyxl")
    rows = [ScoringRow("S001", "年収が", "高い社員は？", False), ScoringRow("S002", "今日は", "何人いますか。", True)]
    sheet = tmp_path / "scoring_sheet.xlsx"
    write_scoring_workbook(sheet, rows)
```
- `pytest.importorskip("openpyxl")` — openpyxlが入っていなければ、このテストは**スキップ**される(失敗ではない)。2行、2つ目は対象外。`tmp_path` = pytestの一時フォルダー。

```python
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
```
- 空のシートを開く;対象外の行は、H4にもう `-` が入っている。**採点者のふりをする:**2行目のH1〜H4、3行目のH1〜H3を埋め、3行目に問題の記号 `"O, t"`(わざと乱れている:空白と小文字)を入れる。`scores_ab.xlsx` として保存し、**読み戻す**。
- S001はちょうど 5、4、5、4 で戻る;S002のH4は `None`、記号は `{"O", "T"}`(小文字の `t` は `T` に直された)。

```python
    ws["E2"] = None                                      # a missing score, a bad value and a bad code
    ws["F3"] = 7
    ws["H2"] = "X"
    wb.save(filled)
    with pytest.raises(ScoreFileError) as e:
        read_scores(filled, rows)
    msg = str(e.value)
    assert "S001: H2 is empty" in msg and "S002: H3 must be a whole number 1-5" in msg and "['X']" in msg
```
- **3種類の違う間違いで、わざとファイルを壊す:**S001のH2を消す、S002のH3を7にする、S001の問題の記号を `X` にする。読み込みは `ScoreFileError` を出し、**1つのメッセージに3つすべてが含まれなければならない**(これが「問題を一度にすべて」の約束)。

---

## 3. テストはいくつ?

**7個のテスト**。期待される結果:`7 passed`(openpyxlが無ければ `6 passed, 1 skipped`)、約1秒で。

## 4. このテストが確かめて*いない*こと

- 本物のデータでの、壊れた数字や単位のルールの**ちょうどの**境界(前半の、本物のテスト用の質問での較正がそれを行う)。
- `term_phrases` が本物の `columns.csv` を読むこと。
- `p_e = 1`(全員がいつも同じ点)という特別な場合のカッパ。
- Excelシートの**見た目**(色、幅、入力規則のポップアップ):値だけを確かめている。
- **文が編集された**採点ファイル(`the beginning or text differs`)や、40件の表示の上限。
- `day13_final_eval.py` と `day13_score_summary.py` 自体(本物のモデルと採点が必要)。

## 5. 一段落でまとめ

この7個の一瞬のテストは、本物の書き出しのリストが作業6と合うこと(50個、種類ごとの正しい数)、単位が学習用の質問から学ばれ一番長い名前が勝つこと、誤った単位と作り出された値が誤警報なしで警告されること、壊れた数字の上限がデータから取られること、バイグラムの基準が一様な当てずっぽうより良い本物の確率の分布であること、一致率とカッパが手計算(カッパ = 11/29)と一致すること、そしてExcelの採点シートが読み書きでき、壊れた採点ファイルがすべての問題を並べた1つのメッセージになることを証明します。
