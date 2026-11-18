# Day 11.2 — test_selection.py — やさしい1行ずつの解説(日本語)

> **Day 11 の3つのファイルの関係**
> - **Day 11.1 `selection.py`** — テストされるコード(**先に読んでください**)。
> - **Day 11 `day11_size_trials.py`** — 運転手。
> - **Day 11.2(このファイル)`test_selection.py`** — **6個の自動テスト**。CPUのみで**一瞬**(モデルも学習も無し。小さな数だけ)。
>
> pytestの基本は、Day 6.2〜10.2で説明しました。ここでは新しいことだけを説明します。

---

## 0. 全体像(まずここから)

決めるルールは、Day 11で一番「議論される」部分です — 教員のレビューで「なぜこの大きさを選んだのか?」と聞かれます。このテストは、プログラムがルールを**書かれたとおりに**適用することを、手で答えを計算できる小さな作り物の学習で証明します:

| 約束 | テスト |
|---|---|
| まとめが、最良の点、gap、最良の後の上昇、速さの中央値、ピークメモリ、経過時間を見つける | `test_summary_reads_best_point_gap_and_rise` |
| 明らかに良い(損失の低い)大きなモデルが勝つ | `test_lowest_loss_wins_when_clearly_better` |
| 許容範囲内なら、**小さい**モデルが勝つ | `test_smaller_model_wins_within_tolerance` |
| 作業6の確認に不合格の学習は、損失が最良でも**絶対に**選ばれない | `test_runs_failing_the_task6_checks_are_never_chosen` |
| 誰も合格しなければ、選ばれる学習は**無い** | `test_no_candidate_gives_no_choice` |
| 同じ大きさ:損失の低い方が勝つ | `test_same_size_tie_prefers_lower_loss_then_no_dropout` |

実行(前と同じ):`.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. 準備と補助関数 `run`

```python
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.selection import RunSummary, choose, summarize_run  # noqa: E402
from src.training import LOG_COLUMNS  # noqa: E402
```
- いつもの準備。テストされるコード(**Day 11.1**)と、**Day 9.2**の `LOG_COLUMNS` — `log.csv` の12個の列名 — を読み込む。偽のログが、本物とまったく同じ形式になる。

```python
def run(name: str, params: int, best: float, finished: float = 1.0, repetitive: float = 0.0,
        dropout: float = 0.0) -> RunSummary:
    return RunSummary(name=name, size=name.split("_")[0], dropout=dropout, parameters=params, best_val=best,
                      best_step=100, train_at_best=1.0, final_val=best + 0.1, final_step=800, tokens_per_sec=None,
                      peak_vram_mb=None, elapsed_s=10.0, finished_rate=finished, repetitive_rate=repetitive,
                      distinct2=0.5)
```
- **偽の学習を作る工場**。大事なこと(名前、パラメータ、最良の検証の損失、そして任意で完結・繰り返しの割合とドロップアウト)だけを指定し、他は無害な初期値になる。`name.split("_")[0]` は、`tiny_d00` のような名前から大きさを取る。初期値では、偽の学習は確認に**合格する**(完結100%、繰り返し0%)。

---

## 2. テスト

### 2.1 `test_summary_reads_best_point_gap_and_rise`

```python
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
```
- 一時フォルダーに**偽の学習のログ**を作る:小さな物語を語る7つの行(検証の損失 7.40 → 2.80 → **2.60** → 2.90:ステップ50で最良になり、その後**上がる**)。各行は大事な列だけを書き、`{k: r.get(k, "") for k in LOG_COLUMNS}` が**他のすべての列を空文字で埋める**ので、ファイルは本物と同じ12列を持つ。

```python
    s = summarize_run(log, "tiny_d00", "tiny", 0.0, 1000)
    assert (s.best_step, s.best_val, s.train_at_best) == (50, 2.60, 1.50)
    assert s.gap == pytest.approx(1.10)
    assert (s.final_step, s.final_val) == (75, 2.90)
    assert s.rise_after_best == pytest.approx(0.30)
    assert s.tokens_per_sec == 200 and s.peak_vram_mb == 400 and s.elapsed_s == 12.5
```
- まとめを実行し、各数を**手計算**と照らし合わせる:
  - **最良の点:**ステップ50、検証の損失2.60、そのステップでの学習の損失1.50;
  - **gap** = 2.60 − 1.50 = **1.10**;
  - **最後の点:**ステップ75、損失2.90;**最良の後の上昇** = 2.90 − 2.60 = **0.30**;
  - **速さの中央値:**速さは100、300、200 → 並べると100、200、300 → 真ん中は **200**;
  - **ピークメモリ** 400(ある値はこれだけ)、**経過時間** 12.5(ログの中で最大の経過時間)。

### 2.2 `test_lowest_loss_wins_when_clearly_better`

```python
def test_lowest_loss_wins_when_clearly_better() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.70), run("small_d00", 5_000_000, 2.60)])
    assert d.chosen.name == "small_d00"
```
- tiny 2.70 と small 2.60:差は **0.10** で、許容範囲0.02よりずっと大きいので、small だけが「同じくらい良い」→ 大きくても **small が勝つ**。(ルールは、いつも小さいモデルを好むわけではない — 差がゆれの範囲内のときだけ。)

### 2.3 `test_smaller_model_wins_within_tolerance`

```python
def test_smaller_model_wins_within_tolerance() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.615), run("small_d00", 5_000_000, 2.600),
                run("medium_d00", 15_000_000, 2.640)])
    assert d.chosen.name == "tiny_d00"            # 0.015 worse than the best: within 0.02, and 5x smaller
    assert set(d.as_good) == {"tiny_d00", "small_d00"}
    assert "smaller" in d.reason
```
- 最低の損失は small の2.600。tiny は0.015悪い(**0.02以内**)、medium は0.040悪い(**範囲外**)。
- **tiny が選ばれる**(small よりパラメータが5分の1);「同じくらい良い」集合は、ちょうど `{tiny, small}`(**集合**として比べるので、順番は関係ない);そして理由の文に**小さい**モデルのことが出てくる — 報告書に表示される説明。

### 2.4 `test_runs_failing_the_task6_checks_are_never_chosen`

```python
def test_runs_failing_the_task6_checks_are_never_chosen() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.50, finished=0.80), run("small_d00", 5_000_000, 2.60),
                run("medium_d00", 15_000_000, 2.55, repetitive=0.10)])
    assert d.chosen.name == "small_d00"
    assert set(d.rejected) == {"tiny_d00", "medium_d00"}
```
- tiny は**最良の損失(2.50)**だが、文を80%しか終えない(90%未満)→ **却下**;medium は良い損失(2.55)だが、繰り返しが10%(5%超)→ **却下**。small だけが合格 → 損失は一番悪いのに、**small が選ばれる**。`d.rejected` は、不合格の2つの学習をちょうど挙げる。
- 要点:**生成の質は、損失を比べる前の、絶対的なふるい**。

### 2.5 `test_no_candidate_gives_no_choice`

```python
def test_no_candidate_gives_no_choice() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.50, finished=0.5)])
    assert d.chosen is None and "no run passed" in d.reason
```
- 唯一の学習が不合格(完結50%)→ **候補なし → `chosen` は `None`** で、理由は「no run passed…」と言う。運転手は、悪いものを黙って選ぶ代わりに「No run chosen」と報告し、決定のファイルを書かない。

### 2.6 `test_same_size_tie_prefers_lower_loss_then_no_dropout`

```python
def test_same_size_tie_prefers_lower_loss_then_no_dropout() -> None:
    d = choose([run("tiny_d10", 1_000_000, 2.605, dropout=0.1), run("tiny_d00", 1_000_000, 2.600)])
    assert d.chosen.name == "tiny_d00"
```
- **同じ大きさ**(同じパラメータ)の2つの学習:ドロップアウト0.1では損失2.605、ドロップアウトなしでは2.600。どちらも許容範囲内で、大きさも同じなので、同点の決め方(Day 11.1の§5)は**検証の損失の低い方**へ → `tiny_d00`。
- (テストの*名前*には「then no dropout」— 損失がちょうど等しいときの、さらなる決め方 — も入っている。このテストは、損失が決める前半だけを確かめる。)

---

## 3. テストはいくつ?

**6個のテスト**。期待される結果:`6 passed`(一瞬で)。

## 4. このテストが確かめて*いない*こと

- **ちょうどの境界:**最良よりちょうど0.02高い損失(ルールは `<=`)、ちょうど90%の完結、ちょうど5%の繰り返し。
- **割合が `None`** の学習(生成の確認なし)— `passes_checks` は `False` を返すが、ここでは直接テストされない(割合が低い場合の「候補なし」だけ)。
- **ちょうど同点のときのドロップアウトのルール**(パラメータも損失も等しい → ドロップアウトなし)。
- `generation_check`、図、報告書、決定のファイル(モデルやファイルが必要。スクリプトを動かして確かめる)。

## 5. 一段落でまとめ

この6個の一瞬のテストは、小さな偽の学習と偽のログで、まとめが、最良の点、gap、上昇、速さの中央値、メモリを正しく計算すること、そして決めるルールが、明らかに良いなら損失の最も低い学習を選び、0.02の許容範囲内なら小さいモデルを好み、完結/繰り返しの確認に不合格の学習は決して選ばず、誰も合格しなければ選ばず、同じ大きさの同点は損失の低い方で決めることを証明します。
