# Day 13 — day13_score_summary.py — やさしい1行ずつの解説(日本語)

> **Day 13 の4つのファイルの関係**
> - **Day 13.1 `evaluation.py`** — 道具箱(先に読む)。
> - **Day 13 `day13_final_eval.py`** — 前半:プログラムによる評価 + 採点シート。
> - **Day 13(このファイル)`day13_score_summary.py`** — **後半**:人の採点を読み、**最終の評価の表**を書く(作業11の完了)。
> - **Day 13.2 `test_evaluation.py`** — 道具箱のテスト。
>
> おすすめの読む順番:**13.1 → 13 final_eval → 13 score_summary(これ)→ 13.2**。

---

## 0. 全体像(まずここから)

前半のあと、2人以上の人が、それぞれExcelシートのコピーに記入しました。後半は:

1. `docs/day13/scores/scores_<イニシャル>.xlsx` をすべて**読み**、**確かめる**(すべての点があるか、1〜5の整数か、記号が正しいか、文が変えられていないか)— 問題は**すべて一度に**報告する。
2. (作業6の 6-4 と 7-3 のとおりに)**計算する**:生成方法ごとのH1〜H4の平均;H3が2以下の出力の割合;採点者の一致(1点以内、コーエンのカッパ);問題の種類。
3. 合格の目安(作業6の表10)と比べた**最終の評価の表**を、前半の数字も入れて**書く**:`docs/DAY13_EVALUATION.md` と `docs/figures/day13_human_scores.png`。

```
.venv\Scripts\python.exe scripts\day13_score_summary.py
```

---

## 1. 先に知っておきたい用語

| 用語 | かんたんな意味 |
|---|---|
| **H1〜H4** | 人が付ける4つの点(1〜5):H1 正しい日本語、H2 自然さ、H3 意味が分かる、H4 社内用語。 |
| **採点者** | シートに記入する人。 |
| **1点以内** | 2人の採点者の差が1以下(例:4と5)。目安:70%以上。 |
| **コーエンのカッパ** | 完全な一致から、偶然の一致を引いたもの(Day 13.1)。1 = 完全、0 = 偶然。 |
| **目安(ガイドライン)** | 評価の**前に**作業6で決めた合格のライン。後から変えない。 |

---

## 2. importと定数

```python
import argparse, itertools, json, math, sys
from datetime import datetime
from pathlib import Path
...
from src.evaluation import (H_ITEMS, PROBLEM_TYPES, PROMPT_KINDS, Score, ScoreFileError, ScoringRow,
                            agreement_within_one, cohen_kappa, mean, read_scores)      # Day 13.1

SCRIPT_VERSION = "day13-v1"
MIN_H_MEAN = 3.5
MAX_H3_LOW = 0.10
MIN_AGREEMENT = 0.70
H_NAMES = {"H1": "H1 correct Japanese", "H2": "H2 natural", "H3": "H3 makes sense", "H4": "H4 company terms"}
```
- 新しい標準のモジュール:**`itertools`**(`combinations` = 採点者のすべての組)。
- **合格の目安**が定数:Hの平均はすべて **3.5 以上**;H3が2以下の出力は **10%以下**;1点以内の一致は項目ごとに **70%以上**。(完結 90%以上、繰り返し 5%以下は、前半が保存した目安から来る。)
- `H_NAMES` — 報告書と図の表示名。

---

## 3. `parse_args` と `yes_no`

- オプション:`--scores-dir`(初期値 `docs/day13/scores`)、`--key`(答えの表)、`--auto`(前半のJSON)、`--report`、`--figure`、`--no-log`。
- `yes_no(ok)` → `None`(判定しない)なら `"-"`、そうでなければ `**yes**` か `**no**`(Markdownで太字)。

---

## 4. `main`

### 4.1 すべての採点ファイルを読み、確かめる

```python
    key = json.loads(a.key.read_text(encoding="utf-8"))["rows"]
    auto = json.loads(a.auto.read_text(encoding="utf-8"))
    expected = [ScoringRow(k["id"], k["prompt"], k["text"], k["out_of_scope"]) for k in key]
    files = sorted(a.scores_dir.glob("scores_*.xlsx"))
    if not files:
        raise SystemExit("No scores_*.xlsx in ... Each scorer copies docs/day13/scoring_sheet.xlsx there as scores_<initials>.xlsx and fills it in.")
```
- 答えの表と前半の結果を読む;答えの表を、**配った**行のリスト(ファイルの確認に使う)にする。採点ファイルを探す(並べ替え済み);1つも無ければ、手順を示して止まる。

```python
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
```
- ファイルごとに、`read_scores`(Day 13.1)が `{出力のID → Score}` を返す。採点者の名前 = ファイル名から `scores_` を除いたもの(`scores_ab.xlsx` → `ab`)。
- **どれかの**ファイルに問題があれば、(**すべての**ファイルの)問題を**全部**表示して、**何も書かずに止まる**(戻り値1)。だから、最終の表が不完全な採点に基づくことは無い。
- `names` = 採点者のリスト。**採点者が1人だけなら警告が出る**(一致を測れない)。

### 4.2 出力ごとの点 = 採点者の平均

```python
    per_out = {}
    for k in key:
        per_out[k["id"]] = {}
        for h in H_ITEMS:
            vals = [scorers[n][k["id"]].h[h] for n in names if scorers[n][k["id"]].h[h] is not None]
            per_out[k["id"]][h] = mean(vals) if vals else None
```
- 出力ごと、H項目ごとに:**すべての採点者の点の平均**(例:4と5 → 4.5)。`None` の点(対象外の出力のH4)は飛ばす;誰も採点していなければ `None`。

```python
    def h_means(rows): return {h: mean(per_out[r["id"]][h] for r in rows if per_out[r["id"]][h] is not None) for h in H_ITEMS}
    def h3_low(rows):  return sum(per_out[r["id"]]["H3"] <= 2 for r in rows) / len(rows)
    groups = {"greedy+limits": [...], "sample": [...], "both": key}
    means = {g: h_means(rows) for g, rows in groups.items()}
    lows  = {g: h3_low(rows) for g, rows in groups.items()}
```
- `h_means` — 行の1グループについて、出力ごとの平均の平均;`h3_low` — (平均の)H3が**2以下**の出力の割合(意味が分からない文は、それだけで失敗)。
- **3つのグループ:**greedyの出力(50)、sampleの出力(50)、両方(100)— 採点者には生成方法を隠す必要があったが、ここでは(答えの表から)分かっている。

### 4.3 採点者の一致

```python
    agreement = {}
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
```
- H項目ごと、**採点者の組ごと**(`combinations(names, 2)`:3人なら3組)に:その項目が採点された出力を取り、両者の点を並べ、1点以内の一致とカッパを計算する。そして**組で平均**して、項目ごとに1つの数にする。(採点者が1人なら、`agreement` は空のまま。)

### 4.4 問題の種類

```python
    types_of = {k["id"]: set().union(*(scorers[n][k["id"]].types for n in names)) for k in key}
    type_counts = {code: sum(code in types_of[k["id"]] for k in key) for code in PROBLEM_TYPES}
```
- 出力ごとに:すべての採点者が付けた問題の記号の**和集合**(「**誰か1人でも**付ければ数える」)。そして記号ごと(G, R, T, O, U, N)に、それを持つ出力の数。

### 4.5 表示、図、報告書、ログ

- グループごとに1行表示:`H1 … H4` の平均とH3が2以下の割合;一致は項目ごとに1行。そして `draw_figure`、`write_report`、`run_log.md` への1項目(採点者、Hの平均、H3が2以下、一致または「not measured」)。

---

## 5. `draw_figure`

- 群になった棒グラフ:H1〜H4それぞれに、青い棒(greedy+limits)とオレンジの棒(sample)、**各棒の上に値**、**目安3.5の赤い破線**。y軸は1〜5.4。赤い線より下の棒は不合格。

---

## 6. `write_report` — 最終の評価の表

### 6.1 表の行

```python
    g_best = {m: gen[f"best.pt|{m}"] for m in ("greedy+limits", "sample")}
    test_ppl, untrained_ppl, bigram_ppl = best.pt のテストの損失の exp, 学習前の exp, バイグラムの exp
    finished_ok = all(v["finished"] >= min_finished ...)
    rep_ok = all(v["repetitive"] <= max_repetitive ...)
    agree_ok = all(v["within_one"] >= MIN_AGREEMENT ...) if agreement else None
```
- 前半から、**最終のモデル**のプログラムの数字(両方の方法の `best.pt`)を取り、目安ごとに確かめる(**両方の**方法が合格でなければならない)。採点者が1人なら `agree_ok` は `None`。

| 項目 | 目安 | 判定の仕方 |
|---|---|---|
| A1 パープレキシティ(テスト) | 学習前よりずっと低い | **テストのパープレキシティ < 学習前 ÷ 10**(「ずっと」のしきい値は、このスクリプト自身の選択) |
| A2 完結の割合 | 90%以上 | 両方の方法 |
| A3 繰り返しの割合 | 5%以下 | 両方の方法 |
| H1〜H4(平均) | 3.5以上 | **両方の**方法を合わせた平均 |
| H3が2以下の出力 | 10%以下 | 両方の方法を合わせて |
| 一致(1点以内) | 項目ごとに70%以上 | すべての項目 |

- `passed` = ちょうど `True` の行;`judged` = `None` でない行。報告書は**「N of M judged items meet the guidelines.」**と書く。

### 6.2 セクション

| セクション | 内容 |
|---|---|
| ヘッダー | バージョン、日付、元(前半の報告書の日付、採点者の数)、最終のモデル、「基準:作業6の文書、**変更なし**」 |
| **最終の評価の表** | 上の行;生成の値は **greedy / sampling** で表示 |
| 人の採点 | 図、説明(対象外の書き出しではH4を採点しない)、方法ごとの表(greedy、sample、両方) |
| **書き出しの種類ごと** | 6種類それぞれに:出力の数、H1〜H4の平均(`nan` なら `-`。例:対象外のH4)、H3が2以下の割合 |
| **採点者の一致** | 項目ごとの1点以内とカッパ + **カッパのほうが厳しい**こと、ほとんどの点が4〜5だと、2人の採点者は偶然だけでも1点以内になりやすいことの説明。カッパが **0.2** 未満の項目があれば、太字の警告:採点者は、違った出力をいくつか見比べて、基準の言葉をはっきりさせるべき |
| **問題の種類** | 記号(G/R/T/O/U/N)ごとに:出力の数、そして関連する種類については**前半のプログラムの確認**(T ↔ 単位の誤り%、R ↔ 繰り返し%、U ↔ 未完了%) |
| **点の最も低い/高い出力** | H1〜H4と採点者の平均で、**最低8個**と**最高5個**(同点はIDで):方法、書き出し、文、平均点、問題の種類、**採点者のメモ** |
| 注意 | 生成の値は best.pt のもの;目安はここでは変えない。あとで改訂するなら、日付とともに記録する |

```python
    def mean_all(i): v = [x for x in per_out[i].values() if x is not None]; return sum(v) / len(v)
    ranked = sorted(key, key=lambda k: (mean_all(k["id"]), k["id"]))
    ... ranked[:8] ... ranked[-5:][::-1]
```
- `mean_all` — 出力ごとに1つの数(H1〜H4の平均の平均)。`ranked` = 低い順;`ranked[:8]` = 最悪の8個、`ranked[-5:][::-1]` = 最良の5個(良い順)。**実際の最悪の文**と採点者のメモが読めるので、平均だけより役に立つ。

---

## 7. 実行の流れ(順番)

1. 答えの表と前半の結果を読み、採点ファイルを探す。
2. すべてのファイルを確かめる;問題があれば、全部表示して止まる。
3. 出力ごとに採点者の平均;greedy / sample / 両方のHの平均;H3が2以下の割合。
4. 一致(採点者の組)と問題の種類の数。
5. 図 → 最終の評価の表の報告書 → 実行ログ。

## 8. 結果の読み方

- **まず表:**「N of M items meet the guidelines」とyes/noの列。
- **どの項目が不合格か?**H1〜H4を3.5と比べる(図の赤い線);種類ごとの表で、**どの種類の書き出し**が弱いかを見る(*学習にない言い方*や*対象外*が多い)。
- **一致:**1点以内が高くてカッパが低い = 採点者は似た高い点を付けるが、同じ数字になることは少ない;違った出力を話し合う。
- **最も低い出力 + メモ:**次に直すこと(単位、繰り返し、作り出した用語)。

## 9. 知っておくと良い小さなこと

- **「採点者の平均」でH3が2.5になることがある**。これは2以下**ではない**ので失敗に数えられない;ルールは各人の点ではなく、採点者の平均を使う。
- **H4は、対象内の出力だけで平均する**(対象外 = `None`)。
- **A1のしきい値(÷10)は作業6のものではない** — 「ずっと低い」をこのスクリプトが解釈したもの。聞かれたらそう答える。
- **問題の種類は和集合**(誰か1人でも)なので、1人の採点者が出力に種類を加えられる。
- **生成方法が明かされるのは、採点が終わった後の、ここだけ**(答えの表を通して)。
- **採点者が1人だと**:一致と一致の行は「not measured」で、表の判定される項目が減る。
- **後半は目安を決して変えない** — 評価の前に固定されている。
- **matplotlibが必要**;採点ファイルには **openpyxl** が必要(Day 13.1を通して)。

## 10. 一段落でまとめ

`day13_score_summary.py` は、すべての採点者のExcelファイルを読み、すべて確かめて、1つでも悪ければ問題の一覧を出して止まります。そうでなければ、出力ごとの点を採点者で平均し、greedy・sampling・両方のH1〜H4、H3が2以下の出力の割合、採点者の1点以内の一致率とコーエンのカッパ、問題の種類、点の最低/最高の出力を報告し、前半のプログラムの数字と合わせて、最終の評価の表(各合格の目安に対するyes/no)、図、実行ログの項目にします。
