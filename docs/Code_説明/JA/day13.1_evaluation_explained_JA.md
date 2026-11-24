# Day 13.1 — evaluation.py — やさしい1行ずつの解説(日本語)

> **Day 13 の4つのファイルの関係**
> - **Day 13.1 `evaluation.py`(このファイル)** — **道具箱**:書き出しのリスト、問題の検出器、バイグラムの基準、一致率とカッパ、Excelの採点シート。**最初に読んでください。**
> - **Day 13 `day13_final_eval.py`** — 前半:最終のプログラムによる評価をして、採点シートを作る(この道具箱を使う)。
> - **Day 13 `day13_score_summary.py`** — 後半:人の採点を読んで、最終の評価の表を書く(この道具箱を使う)。
> - **Day 13.2 `test_evaluation.py`** — この道具箱の**7個の自動テスト**。
>
> おすすめの読む順番:**13.1 → 13 final_eval → 13 score_summary → 13.2**。

---

## 0. 全体像(まずここから)

Day 9〜12では、**損失**と少しの生成の確認でモデルを測りました。Day 13は**最終試験**で、作業6の文書が求めることをすべて行います。文章を生成するだけでは測れない項目があるので、このファイルが4つの道具を足します:

| # | 道具 | 答える問い |
|---|---|---|
| 1 | 固定の約50の書き出し(6種類) | 「どのプロンプトでテストするか。毎回同じものか?」 |
| 2 | 問題の検出器(単位の誤り、壊れた数字、学習にない値) | 「*勤続年数が300万円* のような文を書いていないか?」 |
| 3 | バイグラム(数えるだけ)の基準 | 「Transformerは、ただ数えるだけの方法より、どれだけ良いか?」 |
| 4 | 人の採点(Excelシート、読み戻し、一致率、カッパ) | 「本物の人は文を良いと思うか。採点者同士は一致するか?」 |

---

## 1. 先に知っておきたい用語

| 用語 | かんたんな意味 |
|---|---|
| **基準(ベースライン)** | 比べるための単純な方法。モデルがそれにほとんど勝てないなら、すごくない。 |
| **バイグラム** | 隣り合う2つのトークンの組。バイグラムモデルは、**1つ前のトークンだけ**から、数えるだけで次を予測する。 |
| **正規表現(regex)** | 文字を探すためのパターン。例:「数字+円」。 |
| **コーエンのカッパ** | 2人の採点者の一致を、偶然の一致を引いて測る数。1 = 完全、0 = 偶然と同じ。 |
| **入力規則(Excel)** | セルのルール。例:「1〜5の整数だけ」。 |
| **較正(キャリブレーション)** | 良いと分かっているデータで検出器を確かめること(ほぼ0件になるはず)。 |

---

## 2. import

```python
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
```
- `csv` は書き出しのリストと `columns.csv` を読む;`re` = 正規表現;`dataclass`/`field` は小さな記録のクラス(Day 6.1);`Iterable`/`Sequence` は型のヒント;`numpy` は数える表のため。`openpyxl`(Excel)は、2つのExcel用の関数の**中**でimportするので、無くてもそれ以外は動く。

---

## 3. 1つ目 — 固定の書き出し

```python
PROMPT_KINDS: dict[str, tuple[str, str, int]] = {
    "dept": ("Department or branch", "部署・勤務地で始まる", 10),
    "item": ("Item", "項目で始まる", 10),
    "count_rank": ("Count or rank", "人数・順位を聞く", 10),
    "unseen": ("Wording not in training", "学習にない言い方", 10),
    "key_term": ("Key term variant", "用語とその言い換え", 5),
    "out_of_scope": ("Out of scope", "対象外", 5),
}
```
- 辞書:**種類 → (英語名、日本語名、作業6の表7が求める書き出しの数)**。10+10+10+10+5+5 = **50**。
- `unseen` = 学習用の質問にない言い方(一般化のテスト);`out_of_scope` = モデルが本来知らないはずの書き出し(例:今日は)。

```python
@dataclass(frozen=True)
class EvalPrompt:
    kind: str
    prompt: str
```
- テスト用の書き出し1つ。`frozen=True` = 作った後は変更できない。

```python
def load_eval_prompts(path: Path) -> list[EvalPrompt]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [EvalPrompt(r["kind"].strip(), r["prompt"].strip()) for r in csv.DictReader(f) if r["prompt"].strip()]
    unknown = {r.kind for r in rows} - set(PROMPT_KINDS)
    if unknown:
        raise ValueError(f"unknown prompt kinds: {sorted(unknown)}")
    if len({r.prompt for r in rows}) != len(rows):
        raise ValueError("a beginning appears twice")
    return rows
```
- CSV(列 `kind`、`prompt`)を読む。`utf-8-sig` = Excelが先頭に付ける見えない印を許すUTF-8。空のプロンプトは飛ばし、前後の空白は取る。
- **2つの安全確認:**`kind` はすべて既知でなければならない(打ち間違いで止まる)、そして**同じ書き出しが2回あってはならない**(結果はプロンプトごとに保存するので、重複すると上書きされる)。

---

## 4. 2つ目 — 問題の検出器

### 4.1 数字と単位のパターン

```python
_NUMBER_UNIT = r"([0-9][0-9,]*(?:万[0-9]*)?)\s*(円|歳|年|時間|つ|%|％)"
```
**2つのグループ**を持つ正規表現:
- グループ1 = **数字**:数字1つ、続けて数字かカンマ、任意で `万` とさらに数字(`800`、`1,000`、`300万`、`1万5000`)。
- グループ2 = **単位**:円、歳、年、時間、つ、%、％(間の空白は許す:`\s*`)。

### 4.2 `term_phrases`

```python
def term_phrases(columns_csv: Path) -> list[str]:
    phrases: set[str] = set()
    with columns_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            phrases.add(r["列名"].strip())
            for col in ("質問で使う言い方", "平均と一緒に使う言い方"):
                phrases.update(p.strip() for p in r.get(col, "").split("、") if p.strip())
    return sorted(phrases, key=lambda p: (-len(p), p))
```
- `columns.csv` から、**学習用の質問が項目を呼ぶすべての言い方**を集める:列名(列名)と、言い換え(`、` 区切り)。`set` が重複を除く。
- **長い順**(`-len(p)`)、同じ長さなら五十音順で返す。なぜ長い順か?`月平均残業時間` が、短い `残業時間` より先に当てはまるように。

### 4.3 `UnitChecker`

```python
class UnitChecker:
    def __init__(self, phrases: Sequence[str]) -> None:
        alts = "|".join(re.escape(p) for p in sorted(phrases, key=lambda p: (-len(p), p)))
        self.pattern = re.compile(rf"({alts})(?:が|は)?{_NUMBER_UNIT}")
        self.allowed: dict[str, set[str]] = {}
        self.values: set[str] = set()
        self._value = re.compile(_NUMBER_UNIT)
```
- 大きな1つのパターンを作る:**(項目名)(任意の が/は)(数字)(単位)** — 例:`年収が800万円`。`re.escape` は名前の中の特殊文字をそのままの文字として扱う;`|` は「または」;一番長い名前が先に来て勝つ。
- `allowed`:項目 → 学習用で、その項目の後に見た単位の集合(`learn` が埋める)。`values`:学習用で見た「数字+単位」の文字すべて。`_value`:数字+単位だけのパターン。

```python
    @staticmethod
    def _unit(u: str) -> str:
        return "%" if u == "％" else u
```
- 全角の `％` と普通の `%` を同じ単位として扱う。

```python
    def pairs(self, text: str) -> list[tuple[str, str]]:
        return [(m.group(1), self._unit(m.group(3))) for m in self.pattern.finditer(text)]
```
- 文の中の(項目、単位)の組をすべて見つける。グループ1 = 項目、グループ2 = 数字、グループ3 = 単位。

```python
    def learn(self, lines: Iterable[str]) -> "UnitChecker":
        for line in lines:
            for phrase, unit in self.pairs(line):
                self.allowed.setdefault(phrase, set()).add(unit)
            self.values.update(self.value_strings(line))
        return self
```
- **学習用の質問から学ぶ**(手で書いたルールは無い):組ごとに、その項目の許す単位の集合に単位を足す(`setdefault` は最初のときに空の集合を作る);「数字+単位」の文字も全部覚える。`return self` で `UnitChecker(...).learn(lines)` と1行で書ける。

```python
    def value_strings(self, text: str) -> list[str]:
        return [m.group(1) + self._unit(m.group(2)) for m in self._value.finditer(text)]

    def unseen_values(self, text: str) -> list[str]:
        return [v for v in self.value_strings(text) if v not in self.values]

    def mismatches(self, text: str) -> list[tuple[str, str]]:
        return [(p, u) for p, u in self.pairs(text) if p in self.allowed and u not in self.allowed[p]]
```
- `value_strings` — 「数字+単位」の文字すべて。例:`["800万円", "30歳"]`。
- `unseen_values` — **どの学習用の質問にもない**もの(参考の確認:`30000万円` = 3億円は、決まった値のリストになかったので、作り出された値)。
- `mismatches` — **単位の誤り**:項目は既知で、**かつ**その単位が学習で一度もその項目に続かなかった(`勤続年数 + 円`)。どの単位とも一緒に見たことのない項目は**判定しない**(どちらの証拠も無い)。

### 4.4 壊れた数字

```python
def max_digit_run(lines: Iterable[str]) -> int:
    return max((len(m) for line in lines for m in re.findall(r"[0-9]+", line)), default=0)

def has_broken_number(text: str, limit: int) -> bool:
    return any(len(m) > limit for m in re.findall(r"[0-9]+", text))
```
- `max_digit_run` — 学習用の質問の中の、**数字の並びの最長**(例:`社員番号100003` で6)。`re.findall(r"[0-9]+", …)` は数字の並びをすべて挙げる;`default=0` は空の入力のため。
- `has_broken_number` — その上限より**長い並び**(例:`30000100`)があればTrue。上限は推測ではなく、データから測る。

---

## 5. 3つ目 — バイグラムの基準

```python
@dataclass
class BigramBaseline:
    pair_counts: np.ndarray      # (V, V)
    row_totals: np.ndarray       # (V,)
    unigram: np.ndarray          # (V,)
    lam: float = 0.9
```
- モデルは**表**:`pair_counts[a][b]` = 学習用の文章で、トークン `b` がトークン `a` の後に来た回数。`row_totals[a]` = `a` が「1つ前」として現れた回数。`unigram` = 各トークンの全体の確率。`lam` = 組の表をどれだけ信じるか(0.9 = 90%)。
- 式:**P(次 | 前) = lam × count(前,次)/count(前) + (1 − lam) × P_unigram(次)**。unigramの部分は**+1の数え方**(add-one)なので、どのトークンも確率0にならない(0だと損失が無限大になる)。

```python
    @classmethod
    def fit(cls, tokens: np.ndarray, vocab_size: int) -> "BigramBaseline":
        t = np.asarray(tokens, dtype=np.int64)
        pairs = np.bincount(t[:-1] * vocab_size + t[1:], minlength=vocab_size * vocab_size)
        pair_counts = pairs.reshape(vocab_size, vocab_size).astype(np.float64)
        uni = np.bincount(t, minlength=vocab_size).astype(np.float64) + 1.0
        return cls(pair_counts, pair_counts.sum(axis=1), uni / uni.sum())
```
- `int64` に変える(トークンのファイルは `uint16` で、掛け算するとあふれる)。
- **すべての組を一度に数えるコツ:**組 (a, b) を `a × V + b` という1つの数にし、`np.bincount` が各数の出現回数を数え、`reshape(V, V)` で回数を表にする。
- `uni` = 1トークンの回数 **+ 1**(add-one)を合計で割って確率に。`pair_counts.sum(axis=1)` = 行の合計。
- 大きさ:1,674 × 1,674 個の数 ≈ 22 MB — 問題ない。

```python
    def loss(self, tokens: np.ndarray, lam: float | None = None) -> float:
        lam = self.lam if lam is None else lam
        t = np.asarray(tokens, dtype=np.int64)
        prev, nxt = t[:-1], t[1:]
        totals = self.row_totals[prev]
        p_bi = np.divide(self.pair_counts[prev, nxt], totals, out=np.zeros(len(prev)), where=totals > 0)
        w = np.where(totals > 0, lam, 0.0)
        p = w * p_bi + (1.0 - w) * self.unigram[nxt]
        return float(-np.log(p).mean())
```
- 文章の中の(前、次)の組ごとに、その回数と前のトークンの合計を調べる。**合計 > 0 のところだけ**割る(`where=`。他は0のまま)。
- `w` = 前のトークンが学習で見られたなら `lam`、そうでなければ **0**(unigramだけを使う)。
- 2つの部分を混ぜ、**−log** をとり、平均する → Transformerの損失と**同じ種類**(自然対数)なので、直接比べられる(`exp(損失)` = パープレキシティ)。

```python
    def tune(self, val_tokens, grid=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)) -> float:
        self.lam = min(grid, key=lambda g: self.loss(val_tokens, g))
        return self.lam
```
- 各 `lam` を試し、**検証の損失が最低**のものを残す。**テスト用の文章は決して使わない** — 最終の数字が出るまで、手つかずにしておく。

---

## 6. 4つ目 — 採点者の一致

```python
def agreement_within_one(a, b) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty score lists")
    return sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)
```
- 2人の採点者の差が**1点以内**の出力の割合(作業6の目安:70%以上)。`sum` では `True` が1と数えられる。

```python
def cohen_kappa(a, b, labels=(1, 2, 3, 4, 5)) -> float:
    ...
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)
```
- `p_o` = 観察された**完全な**一致。`p_e` = **偶然**に期待される一致:各点について(Aがそれを付けた割合)×(Bがそれを付けた割合)を足したもの。
- **カッパ = (p_o − p_e) / (1 − p_e)。**例:2人とも常に「5」なら p_e = 1 でゼロ割りになるので、特別な場合として、本当に一致していれば1.0、そうでなければ0.0を返す。
- 注意:`a.count(k)` はリスト用(numpyの配列ではない)。呼ぶ側はリストを渡す。

---

## 7. 5つ目 — 採点のExcel

### 7.1 定数と記録

```python
H_ITEMS = ("H1", "H2", "H3", "H4")
PROBLEM_TYPES = {"G": ("Grammar error", "文法の誤り"), "R": ("Repetition", "繰り返し"), "T": ("Wrong term use", "用語の誤り"),
                 "O": ("Off topic", "話題がずれる"), "U": ("Unfinished", "途中で終わる"), "N": ("Invented name", "存在しない名前")}
```
- H1 正しい日本語、H2 自然さ、H3 意味が分かる、H4 社内用語(作業6の表8);問題の種類の記号6つ(表9)。
- `_HEADERS` = 列の見出し(2か国語);`_CRITERIA` = 各項目の1〜5の基準。説明のシートに出るので、全員が同じ「3」の意味で採点できる。

```python
@dataclass
class ScoringRow:  id: str; prompt: str; text: str; out_of_scope: bool
@dataclass
class Score:       h: dict[str, int | None]; types: set[str] = field(default_factory=set); note: str = ""
class ScoreFileError(ValueError): pass
```
- `ScoringRow` = 採点される1行(採点者が見るもの)。`Score` = 1人の採点者が1行に付けたもの:4つの点(`None` = 採点しない)、問題の記号、メモ。`field(default_factory=set)` は、各Scoreに**自分専用の**空の集合を与える。`ScoreFileError` = 悪い採点ファイルのための特別なエラーの種類。

### 7.2 `write_scoring_workbook`

- **2つのシート**のExcelを作る:
  1. **「説明 Instructions」** — ルール(一人で採点、1〜5の整数、対象外の行はH4を採点しない、問題の記号、「…」= 途中で終わった、行はばらばら)、基準の表、記号の一覧。折り返すのは表だけ。
  2. **「採点 Scoring」** — 出力1つにつき1行:`ID、書き出し、続き、H1、H2、H3、H4、問題の種類、メモ`。見出しの行は青で太字。対象外の行は、**H4 に「-」が最初から入り、灰色**。
- **入力規則:**D〜F列(H1〜H3)は**1〜5の整数だけ**を受け付ける(それ以外はエラーが出る);H4(G列)は、対象内の行にだけ規則がある。
- 列の幅、ウィンドウ枠の固定(`D2` で、スクロールしてもID・書き出し・続きが見える)、折り返し、`wb.active = 1` で採点のシートが最初に開く。

### 7.3 `read_scores` — 採点者1人のファイルを読み、**問題をすべて集める**

- 採点者のファイルを開き(`data_only=True` = 式ではなく値)、各行を確かめる:
  - IDが、配ったIDのどれかであること;
  - **書き出しと文が、元のシートから変えられていない**こと;
  - 各H点が入っていて、**1〜5の整数**であること(`3.5` は拒否:`isinstance(value, float) and value != v`);対象外の行のH4は飛ばす(`None`として保存);
  - 問題の記号が正しいこと:カンマや空白で分け(`[,、\s]+`)、大文字にし(`t` → `T`)、`-` は無視し、知らない記号は報告;
  - 行が欠けていないこと。
- **問題はまず全部集めてから、まとめて出す**(最初の40件まで表示)ので、採点者は、1回に1つのエラーではなく、一度にすべて直せる。

### 7.4 `mean`

```python
def mean(values): v = list(values); return sum(v) / len(v) if v else math.nan
```
- 平均。空のリストでは、落ちずに `nan`(「数ではない」)を返す。

---

## 8. 知っておくと良い小さなこと

- **単位の誤りは、既知の項目だけを判定する**:学習で単位と一緒に見たことのない項目は飛ばすので、検出器は慎重(誤警報が少ない)だが、網羅的ではない。
- **検出器は、プロンプト+続きの全体を見る**(呼ぶ側が文全体を渡す)ので、*プロンプト*に書かれた学習にない値が「学習にない値」として出ることがある。
- **学習にない値は参考だけ**:新しい値が必ず誤りとは限らない。
- **バイグラムの表は 1,674² 個の数**で、**学習用**のトークンだけから作り、**検証用**で調整する。
- **`％`/`%` の扱い**で、同じ単位を2つの違う単位と数えることを避ける。
- **ほとんどの点が4〜5だと、一致が高くてもカッパは低く見える**(「カッパのパラドックス」)— 2つの数字を一緒に読む(Day 13 score_summary の報告書がそう書く)。
- **Excelは文を変えることがある**(例:末尾の空白);`read_scores` は文を厳密に比べるので、採点者はB〜C列を編集してはいけない。

## 9. 一段落でまとめ

`evaluation.py` はDay 13の道具箱です:6種類・50個の固定の書き出しのリスト(安全確認つき)、**学習用の質問から学んだ**ルールによる、単位の誤り・壊れた数字・学習にない値の検出器、検証だけで調整する、パープレキシティのための数えるだけ(バイグラム)の基準、2人の採点者の、1点以内の一致率とコーエンのカッパ、そして、1〜5を強制するExcelの採点シートと、すべての採点ファイルを確かめて問題を一度にまとめて報告する読み込みの関数です。
