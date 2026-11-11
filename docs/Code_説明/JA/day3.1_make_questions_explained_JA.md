# Day 3.1 — make_questions.py — やさしい1行ずつの解説(日本語)

> **Day 3 の2つのファイルの関係**
> - **Day 3.1(このファイル)`make_questions.py`** — チームの**質問生成プログラム**:文型に言葉を当てはめて、日本語の質問を何千個も作り、各質問の*意味*も保存する。
> - **Day 3 `day3_generate_questions.py`** — *実行役*:入力をきれいなフォルダーにコピーし、固定した設定でこの生成プログラムを動かし、結果を検査して保存する(動画で記録する実行)。
>
> おすすめの読む順番:**3.1 → 3**。(Day 4・5は、あとでこの生成プログラムの出力を使います。)

---

## 0. 全体像(まずここから)

**穴埋めプリント**のように動く**質問の工場**を想像してください。

空欄のある文を書きます:

```
{所属}の社員の{列}の{集計}を教えてください。
```
工場には、各空欄に入れる言葉のリストがあります:
- `{所属}` → 営業部、開発部、大阪支社、…
- `{列}` → 年収、年俸、月収、…
- `{集計}` → 平均、合計、最高額、…

それを**組み合わせて**、本物の質問を作ります:

```
営業部の社員の年収の平均を教えてください。
開発部の社員の月収の最高額を教えてください。
...
```

さらに、**質問ごとに、その意味**を小さなJSONのラベルとして保存します(例:`{"パターン":"P001","質問タイプ":"T05","条件":"部署","条件値":"営業部","列":"年収","集計":"平均"}`)。後の日がこのラベルを使います(Day 4・5が検査します。学習用の文章としては**絶対に**使いません)。

*(上の例は、考え方を見せるために作ったものです。本物の文型は、チームの `src_*.txt` にあります。)*

### 大事な考え方(この5つの言葉を覚える)

| 考え方 | 意味 |
|---|---|
| **文型(パターン)** | `{スロット}` のある文のひな形。`P001`、`P002`、… というIDが付く。 |
| **質問タイプ** | 同じ種類のことを聞く文型のグループ(元のファイルの `#T..` の見出しの行)。 |
| **スロット** | `{列}` や `{所属}` のような空欄。 |
| **選択肢(option)** | スロットの埋め方1つ:**(入れる文字、意味)**。例:`("営業部", {"条件":"部署","条件値":"営業部"})`。 |
| **要素(factor)** | 独立した選択肢のリスト1つ(例:「部署・勤務地の選択肢すべて」)。質問 = **各要素から選択肢を1つずつ**。作れる質問の数 = **各要素の大きさの積**。 |

---

## 1. 入力と出力

**入力**(スクリプトを動かすフォルダーにあること):

| ファイル | 中身 |
|---|---|
| `用語集.csv` | チームの用語集 |
| `同義語辞書.csv` | 同義語辞書(言葉の言い換え) |
| `DB.xlsx` | 50行のサンプルの社員表(氏名、ID、部署、勤務地など) |
| `src_*.txt` | 文型の元ファイル:`#` で始まる行は**質問タイプの見出し**(`#T..|名前`)、他の行は文型 `文型|列グループ|列グループ2` |

**出力**(同じフォルダーに書かれる):

| ファイル | 中身 |
|---|---|
| `patterns.csv` | 文型ごとに、タイプ、文体、作れる質問の数 |
| `question_types.csv` | 質問タイプごとの合計 |
| `columns.csv` | 数値の列ごとのルール |
| `samples.csv` | 文型ごとに例の質問5個と意味(チームの確認用) |
| `questions.csv` | (`--corpus N` のときだけ)**文型ごとに最大 N 問**と意味 |

このスクリプトは、ファイルを**今いるフォルダーで**読み書きするので、Day 3は全部を新しいフォルダーにコピーして、そこで動かします(チームの `data/source` を上書きしないため)。

使い方:

```
python make_questions.py                  # 文型、件数、サンプル
python make_questions.py --corpus 2000    # バランスの取れた質問コーパスも
```

---

## 2. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **スロット / プレースホルダー** | 文のひな形の中の `{...}` の空欄。 |
| **置換** | スロットを本物の文字に置き換えること(`text.replace("{列}", "年収")`)。 |
| **直積(Cartesian product)** | 各リストから1つずつ選ぶ、すべての組み合わせ(`itertools.product`)。 |
| **順列(permutation)** | 順番のある並び。(A,B) と (B,A) は別(`itertools.permutations`)。 |
| **組み合わせ(combination)** | 順番のない選び方。(A,B) は1回だけ(`itertools.combinations`)。 |
| **シード / `random.Random`** | 再現できる乱数(Day 5を参照)。 |
| **バランス調整** | 文型ごとに最大 N 問までにして、巨大な文型が支配しないようにすること。 |
| **pandas / DataFrame** | 表を扱うライブラリ。`pd.read_excel` が `DB.xlsx` を表として読む(`openpyxl` が必要)。 |
| **正規表現(`re`)** | 文字のパターンを書くミニ言語(Day 5を参照)。 |
| **意味(意味ラベル)** | 各質問と一緒に保存するJSONのラベル。 |

---

## 3. docstringとimport

```python
"""Model A question generator (Phase 1 data), with the meaning of every question saved for later use.
Inputs (same folder): ...
Outputs: ...
Usage: ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import random
import re
from dataclasses import dataclass, field

import pandas as pd
```
- docstringには、入力、出力、使い方が載っている(このファイルには日本語の部分は無く、rawの文字列でもない)。
- `argparse`、`csv`、`json`、`random`、`re`、`dataclass`/`field`、`from __future__` — これまでの日と同じ。新しいもの:
  - `glob` — 名前のパターンでファイルを探す(`glob.glob("src_*.txt")`)。
  - `itertools` — 組み合わせの道具(`product`、`permutations`、`combinations`)。
  - `pandas as pd` — **外部の**表ライブラリ(Day 3が `pandas` と `openpyxl` のインストールを確認する理由)。

```python
GLOSSARY, SYNONYMS, DB = "用語集.csv", "同義語辞書.csv", "DB.xlsx"
```
- 3つのファイル名を1行で代入(タプルの展開代入)。相対的な名前 → ファイルは今いるフォルダーにある必要がある。

---

## 4. 列の表(`Col` と `COLS`)

### 4.1 `Col` データクラス

```python
@dataclass
class Col:
    name: str
    kind: str
    words: list[str]
    vals: list[str]
    big: str = ""
    small: str = ""
    ask: str = ""
    aggs: list[str] = field(default_factory=list)
    avg_words: list[str] = field(default_factory=list)
    ops: tuple = ("以上", "以下", "未満", "超")
```
**データの列1つ**と、それについての人々の言い方の説明:

| 項目 | 意味 | 例(年収) |
|---|---|---|
| `name` | 正式な列名 | 年収 |
| `kind` | 種類:金額、数値、評価、計算、比率 | 金額 |
| `words` | 言い方いろいろ | 年収、年俸、年間給与、年間収入 |
| `vals` | 比較に使う値の例 | 400万円 … 800万円 |
| `big` / `small` | 「大きい/小さい」の言葉 | 高い / 低い |
| `ask` | その値を聞くときの疑問詞 | いくら |
| `aggs` | 使える集計 | 平均、合計、最高額、最低額 |
| `avg_words` | **「平均」の後**で使う言い方(違うとき) | (なし) |
| `ops` | 使える比較の言葉 | 以上、以下、未満、超 |

`field(default_factory=list)` は、オブジェクトごとに自分専用の空のリストを与える(Day 5の `SplitResult` を参照)。

### 4.2 `MONEY`、`NUM`、`COLS` のリスト

```python
MONEY = ["平均", "合計", "最高額", "最低額"]
NUM = ["平均", "最大値", "最小値"]
```
- 金額の列と、数値の列に使える集計。

`COLS` は**21個の列**のリスト。どの行も同じ形(`Col(name, kind, words, vals, big, small, ask, aggs, ...)`)なので、グループと特別な場合をまとめる:

| 種類 | 列 | 備考 |
|---|---|---|
| 金額 | 年収、月収、基本給、役職手当、住宅手当、通勤手当、残業手当、賞与、昇給 | `ask`=いくら、集計 = `MONEY` |
| 数値 | 年齢、勤続年数、月平均残業時間、保有資格数、有給取得率 | `ask` は何歳 / 何年 / 何時間 / いくつ / 何パーセントと違う。集計 = `NUM` |
| 評価 | 直近総合評価、直近健康診断結果、直近メンタル診断結果 | 値は小さな数。健康診断・メンタル診断は、big/small の言葉も集計も**無い**(「健康診断結果が高い」とは言えない) |
| 計算 | 手当合計、賞与を除いた年収 | 複数の列から計算した値。集計は平均だけ |
| 比率 | 賞与の年収比、残業手当の月収比、基本給の月収比 | パーセント。集計は平均だけ |

特別な場合(文を自然にするためのもの):
- **勤続年数** は `avg_words` = 勤続年数、勤務年数、在籍年数 — つまり社歴は**入れない**。「平均社歴」は不自然だから。
- **月平均残業時間** は `avg_words` = 残業時間 — 「平均」+「月平均残業時間」だと、壊れた言葉**平均平均**(Day 3・4が探す `BROKEN_MARKS` の1つ)になるから。
- **保有資格数** は `ops` = 以上、以下、未満だけ(超は無い)。
- **直近総合評価** は `avg_words` = 総合評価、評価。

### 4.3 `BASE` と `GROUPS`

```python
BASE = [c for c in COLS if c.kind in ("金額", "数値", "評価")]
GROUPS = {
    "金額": [c for c in BASE if c.kind == "金額"],
    "大小": [c for c in BASE if c.big],
    "平均": [c for c in BASE if c.aggs],
    "比較": [c for c in BASE if c.kind in ("金額", "数値")],
    "評価": [c for c in BASE if c.kind == "評価"],
    "計算": [c for c in COLS if c.kind == "計算"],
    "比率": [c for c in COLS if c.kind == "比率"],
}
```
- `BASE` — 本物のデータの列(金額、数値、評価)。
- `GROUPS` — 名前の付いた**列のグループ**。文型の行の `...|金額|` は、「列のスロットに金額の列を使う」の意味。グループ:金額、大小(big/small の言葉がある列)、平均(集計がある列)、比較(金額か数値の列。「〜以上」の比較用)、評価、計算、比率。
- `if c.big` は、文字列が空でないときに真。

### 4.4 `to_number`

```python
def to_number(v: str) -> float:
    s = v.replace("%", "").replace("円", "").replace("歳", "").replace("年", "").replace("時間", "").replace("つ", "")
    total, m = 0, re.match(r"(?:(\d+)万)?(\d+)?$", s)
    if m and (m.group(1) or m.group(2)):
        total = int(m.group(1) or 0) * 10000 + int(m.group(2) or 0)
        return total
    return float(s)
```
- 値の文字を、意味ラベル用の**本物の数**に変える。
- まず単位の言葉(`%`、円、歳、年、時間、つ)を取り除く。
- 正規表現 `(?:(\d+)万)?(\d+)?$` は「*N*万がある場合もあり、その後に*M*がある場合もある」と読む。例:
  - `500万円` → `500万` → 500 × 10,000 = **5,000,000**
  - `1万5000円` → 1×10,000 + 5,000 = **15,000**
  - `5000円` → **5,000**
  - `30%`、`25歳` → 最後の行 `float(s)` により、普通の数(30、25)。
- `m.group(1) or 0` — その部分が無ければ0を使う。

---

## 5. チームのファイルから言葉を読む

```python
gl = list(csv.reader(open(GLOSSARY, encoding="utf-8-sig")))[1:]
syn: dict[str, list[str]] = {}
for r in list(csv.reader(open(SYNONYMS, encoding="utf-8-sig")))[1:]:
    syn.setdefault(r[2], []).append(r[3])
db = pd.read_excel(DB)
```
- これらは、ファイルが**読み込まれたとき**、今いるフォルダーで実行される。
- `csv.reader(open(...))` はCSVを1行ずつ読む。`[1:]` は見出しの行を飛ばす。
- `gl` — 用語集の行。(このファイルでは、`gl` は読み込まれるが、その後使われない。)
- `syn` — **標準用語 → 言い換えのリスト**の辞書:同義語の各行について、`r[2]`(3列目)が標準用語、`r[3]`(4列目)が言い換え。`setdefault(キー, [])` は最初にリストを作り、`.append` が言い換えを足す。
- `db = pd.read_excel(DB)` — 社員表をpandasのDataFrameとして読む(`db["部署"]` は部署の列)。

```python
BAD_END = ("課", "チーム", "グループ", "組織", "担当部署", "本部", "拠点", "支店", "部署", "者", "社員")
def value_words(term: str) -> list[str]:
    """Standard value + synonyms that read naturally in 「{X}の社員」."""
    out = [term]
    for v in syn.get(term, []):
        if "の" not in v and not v.endswith(BAD_END) and not re.search(r"[たるい]$", v):
            out.append(v)
    return out
```
- 標準の言葉**と、「〇〇の社員」で自然に読める言い換えだけ**を返す。
- 言い換えが**除かれる**のは、`の` を含む、`BAD_END` の言葉(例:「…課」「…チーム」「…社員」。不自然になる)で終わる、`た`・`る`・`い`(動詞や形容詞の語尾)で終わる、のどれか。
- `v.endswith(BAD_END)` はタプルを受け付ける:どれかで終われば真。

```python
DEPTS = sorted(db["部署"].unique())
RANKS = ["一般", "主任", "係長", "課長", "次長", "部長"]
PLACES = sorted(db["勤務地"].unique())
DEPT_W = {d: value_words(d) for d in DEPTS}
PLACE_W = {p: value_words(p) for p in PLACES}
NAMES = db["氏名"].tolist()
IDS = [str(i) for i in db["社員番号"]]
SURNAMES = sorted({next((s for s in ("佐々木", "原", "林", "森") if n.startswith(s)), n[:2]) for n in NAMES})
```
- `DEPTS`、`PLACES` — 表にある異なる部署・勤務地(`.unique()`)を並べ替えたもの。
- `RANKS` — 固定の役職のリスト(低い順)。
- `DEPT_W`、`PLACE_W` — 各部署・勤務地 → 自然な言い方のリストの辞書。
- `NAMES` — 全員の氏名(`.tolist()`)。`IDS` — 社員番号を文字列にしたもの。
- `SURNAMES` — **各人の名字の推定**(日本語の氏名には空白がない):佐々木、原、林、森で始まればそれを、そうでなければ**最初の2文字**を使う。`next(ジェネレーター, 既定値)` は、最初に一致したものか既定値を返す。集合が重複を除き、`sorted` が順番を固定する。

```python
REASONS = {...}   MARRIAGE = {...}   MISSING = {...}   DISPLAY = {...}
```
- **言い方**の小さな辞書が4つ:
  - `REASONS` — 休職の事由(私傷病 → 私傷病/病気/傷病、育児、産前産後、介護、労災);
  - `MARRIAGE` — 既婚 / 未婚の言い方;
  - `MISSING` — 空欄になりうる列と、その名前;
  - `DISPLAY` — **表示**できる列(「氏名と部署を一覧にして」)と言い方。

---

## 6. 選択肢とスロットの領域

```python
O = tuple  # option = (surface, meta)
def opts(pairs) -> list[O]:
    return [(s, m) for s, m in pairs]
```
- `O` は、タプルの短い名前で、**選択肢 = (表の文字、意味の辞書)** だと示すもの。`opts` は小さな補助関数で、このファイルの他では使われていない。

### 6.1 `attr_options`

```python
def attr_options(kinds: list[str], person: bool = False) -> list[O]:
    out = []
    for k in kinds:
        if k == "部署":
            out += [(w, {"条件": "部署", "条件値": d}) for d, ws in DEPT_W.items() for w in ws]
        elif k == "勤務地":
            out += [(w, {"条件": "勤務地", "条件値": p}) for p, ws in PLACE_W.items() for w in ws]
        elif k == "役職":
            for r in RANKS:
                if person and r == "一般":
                    continue
                if not person:
                    out.append((f"役職が{r}", {"条件": "役職", "条件値": r}))
                if r != "一般":
                    out.append((r, {"条件": "役職", "条件値": r}))
        elif k == "性別":
            for g, w in (("男", "男性"), ("女", "女性")):
                out.append((w, {"条件": "性別", "条件値": g}))
                if not person:
                    out.append((f"性別が{g}", {"条件": "性別", "条件値": g}))
    return out
```
- 「**どの社員のグループか**」のスロットの選択肢を作る。各選択肢は、言い方とその**意味**(`条件` = どの属性か、`条件値` = どの値か)の組。
- 部署 / 勤務地:各部署・勤務地の、すべての言い方。
- 役職:各役職について、「人」の形でなければ長い形 `役職が課長` を足し、(一般を除いて)短い形 `課長` も足す。`person=True` なら、長い形と一般は飛ばす(「…の人」の言い方が自然になるように)。
- 性別:`男性` / `女性`、そして人の形でなければ `性別が男` / `性別が女` も。意味には、短い値の 男/女 が入る。

### 6.2 `pair_options`

```python
def pair_options(word_map: dict[str, list[str]], key: str) -> list[O]:
    return [((a, b), {key: x, key + "2": y}) for x, y in itertools.permutations(word_map, 2) for a in word_map[x] for b in word_map[y]]
```
- **異なる2つのもの**が要るスロット用(例:「営業部と開発部」)。`permutations(word_map, 2)` は、**異なる**キー(x, y)の、すべての順序付きの組を返す。そして x の言い方 `a` と y の言い方 `b` のすべて。意味には `部署` と `部署2`(または指定した `key`)が入る。

### 6.3 `SLOT_DOMAINS` — 単独のスロットの辞書

各項目:スロット名 → 選択肢のリスト。スロット(すべて意味が保存される):

| スロット | 入るもの |
|---|---|
| `{属性}` | 部署、役職、勤務地、性別(長い形と短い形) |
| `{所属}` | 部署または勤務地 |
| `{人の属性}` | 「人」の形の性別または役職 |
| `{性別語}` / `{役職語}` | 男性・女性 / 一般を除く役職 |
| `{グループ}` | 「何別に」:部署/部門、役職、勤務地/勤務先、性別/男女 |
| `{グループ所}` / `{グループ人}` | 絞ったグループのリスト(場所系 / 人系) |
| `{区分列}` | 区分の列:部署/部門、役職、勤務地/勤務先 |
| `{理由}` / `{休職事由}` | 休職の事由(言い方付き / 正式名 + その他) |
| `{婚姻}` | 婚姻の言い方 |
| `{欠損列}` | 空欄になりうる列 |
| `{氏名}` / `{姓}` / `{社員番号}` | 人の氏名 / 名字 / ID |
| `{年}` / `{年2}` | 1990〜2025の年 / 2021〜2025の年 |
| `{月}` | 1〜12の月 |
| `{年代}` | 20代 … 60代 |
| `{N}` / `{人数}` | 上位Nの数(3、5、10)/ 人数(2人、3人、5人、10人) |
| `{経過}` | 経過年数(1、2、3、5年) |

各選択肢は `("20代", {"年代": 20})` のような形。

### 6.4 `PAIR_DOMAINS` — 組になるスロット

```python
PAIR_DOMAINS = {
    ("{部署}", "{部署2}"): pair_options(DEPT_W, "部署"),
    ("{勤務地}", "{勤務地2}"): pair_options(PLACE_W, "勤務地"),
    ("{役職}", "{役職2}"): pair_options({r: [r] for r in RANKS}, "役職"),
    ("{氏名}", "{氏名2}"): pair_options({n: [n] for n in NAMES}, "氏名"),
    ("{表示列}", "{表示列2}"): pair_options(DISPLAY, "表示列"),
    ("{年}", "{年3}"): [((str(a), str(b)), {"年から": a, "年まで": b}) for a in range(1990, 2026) for b in range(a + 1, min(a + 11, 2027))],
}
```
- キー = 2つのスロット名。値 = **2つ一緒の**選択肢のリスト。**異なる**2つの部署 / 勤務地 / 役職 / 氏名 / 表示列と、**年の範囲**(年 `a` から、それより後の年 `b` まで。最大10年後まで、2026年を超えない)。

---

## 7. 列に結びついた選択肢:`col_options`

```python
def col_options(col: Col, pat: str, second: bool = False) -> list[O]:
    sfx = "2" if second else ""
    out = []
    avg_ctx_pat = "平均" in pat or "中央値" in pat
    aggs = col.aggs if "{集計}" in pat else [""]
```
- 文型の文 `pat` で使われる、**1つの列**の選択肢をすべて作る。
- `second=True` は、これが**2つ目**の列のスロットだという意味で、スロット名に `2` が付く(`{列2}`、`{大2}` …)。
- `avg_ctx_pat` — 文型が平均や中央値の話をしているか?
- `aggs` — 文型に `{集計}` のスロットがあれば、その列で使える集計をループする。なければ、空の1つだけ。

```python
    for agg in aggs:
        words = col.avg_words if ((avg_ctx_pat or agg == "平均") and col.avg_words) else col.words
        for w in words:
            base = {"{列" + sfx + "}": w, "{大" + sfx + "}": col.big, "{小" + sfx + "}": col.small, "{疑問詞}": col.ask}
            meta = {"列" + sfx: col.name}
            if agg:
                base["{集計}"] = agg; meta["集計"] = agg
            variants = [(base, meta)]
```
- 言い方のリストを選ぶ:**平均の文脈**で、列が特別な `avg_words` を持っていればそれを使う(「平均平均」を避ける)。そうでなければ普通の言い方。
- 各言い方 `w` について、`base` =「スロット → 文字」の置換を作る:列の言い方、big/small の言葉、疑問詞。`meta` = 意味(`列` = 正式な列名)。集計があれば、両方に足す。
- `variants` は、その(置換、意味)の1組から始まる。

```python
            if "{値" + sfx + "}" in pat:
                variants = [({**b, "{値" + sfx + "}": v, "{比較" + sfx + "}": c}, {**m, "値" + sfx: to_number(v), "比較" + sfx: c})
                            for b, m in variants for v in col.vals for c in col.ops]
```
- 文型に**値のスロット**(「年収が{値}{比較}」のような)があれば、各variantを、**値の例すべて × 比較の言葉すべて**で広げる。意味には、数(`to_number`)と比較が入る。`{**b, ...}` は辞書 `b` をコピーしてキーを足す。

```python
            if "{下}" in pat and not second:
                variants = [({**b, "{下}": col.vals[i], "{上}": col.vals[j]}, {**m, "下限": to_number(col.vals[i]), "上限": to_number(col.vals[j])})
                            for b, m in variants for i, j in itertools.combinations(range(len(col.vals)), 2)]
```
- 文型が**範囲**(「{下}から{上}まで」)なら、値のすべての組で広げる(下 < 上 — `combinations` は各組を1回だけ選ぶ)。

```python
            if "{評価列}" in pat:
                vals = col.vals[1:-1] if ("以上" in pat or "以下" in pat) else col.vals
                variants = [({**b, "{評価列}": w, "{評価値}": v}, {**m, "値": int(v)}) for b, m in variants for v in vals]
            out += variants
    return out
```
- 文型が**評価**の文型なら:評価の値すべてを使う。ただし文型が以上/以下を含むときは、**最初と最後を除く**(「評価が最低値以上」が自明にならないように)。意味には整数が入る。
- variantを `out` に足し、全部のループの後で `out` を返す。

---

## 8. 文型

### 8.1 `Pattern` データクラスと文体

```python
@dataclass
class Pattern:
    pid: str
    tid: str
    tname: str
    text: str
    group: str
    group2: str

    @property
    def style(self) -> str:
        t = self.text
        if t.endswith("？") and len(t) <= 22 and "ください" not in t:
            return "短い" if not re.search(r"(る|た|の)？$", t) else "くだけた"
        if re.search(r"(ください|ですか|ますか|です)。$", t):
            return "丁寧"
        return "くだけた"
```
- 文型:ID、質問タイプのIDと名前、ひな形の文、そして最大2つの**列グループ**(`group` は1つ目の列のスロット用、`group2` は2つ目用)。
- `style` — ひな形からの**文体の推定**(Day 5が数える `文体` はこれ):
  - `？` で終わり、短く(22文字以下)、「ください」が無い:**短い**。ただし る/た/の + `？` で終わるなら **くだけた**;
  - ください / ですか / ますか / です + `。` で終わる:**丁寧**;
  - それ以外は **くだけた**。

### 8.2 `load_patterns`

```python
def load_patterns() -> list[Pattern]:
    out, tid, tname, n = [], "", "", 0
    for f in sorted(glob.glob("src_*.txt")):
        for line in open(f, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                tid, tname = line[1:].split("|")
                continue
            text, g1, g2 = (line.split("|") + ["", ""])[:3]
            n += 1
            out.append(Pattern(f"P{n:03d}", tid, tname, text, g1, g2))
    return out
```
- すべての `src_*.txt` を読む(並べ替えるので、順番が固定)。
- 各行について:改行を取り、**空行を飛ばす**。
- `#` で始まる行は**質問タイプの見出し** `#T05|名前`:`line[1:]` が `#` を落とし、`.split("|")` がタイプIDと名前を返す — 後に続く文型のために覚えておく。
- そうでなければ文型の行:`文|グループ|グループ2`。`(line.split("|") + ["", ""])[:3]` は、空文字を2つ足して最初の3つを取るので、無いグループは空になる。
- `n += 1` とID `f"P{n:03d}"` → `P001`、`P002`、…(`:03d` = 0埋めの3桁)。
- **注意:**IDは行・ファイルの**順番**に依存する。途中に文型を挿入すると、後ろの文型の番号が全部変わる。

### 8.3 `factors` — 独立した選択肢のリスト

```python
def factors(p: Pattern) -> tuple[list[list], bool]:
    """Independent choice lists for a pattern; bool = two columns that must differ."""
    fs: list[list] = []
    if p.group:
        fs.append([(c, o) for c in GROUPS[p.group] for o in col_options(c, p.text)])
    if p.group2:
        fs.append([(c, o) for c in GROUPS[p.group2] for o in col_options(c, p.text, second=True)])
```
- `fs` — 要素のリスト。各**選択**は、`(列または None, (置換, 意味))` の組。
- 1つ目の列のスロット:文型のグループの各列と、その各選択肢 → 選択が1つ。2つ目の列のスロット:同じことを `second=True` で。

```python
    for (a, b), dom in PAIR_DOMAINS.items():
        if a in p.text and b in p.text:
            dom2 = dom
            if a == "{表示列}" or "{表示列}" in p.text and a == "{表示列}":
                pass
            if a == "{表示列}" and "{氏名}" in p.text:
                dom2 = [o for o in dom if "氏名" not in (o[1]["表示列"], o[1]["表示列2"])]
            fs.append([(None, ({a: x, b: y}, m)) for (x, y), m in dom2])
```
- 各**組のスロット**について、2つのスロットが両方ひな形にあれば:その選択肢を取る。(`if … pass` の行は何もしない。)表示列のスロットで、**かつ**文に `{氏名}` が既にあれば、氏名をもう一度表示してしまう選択肢を除く。要素を1つ追加する(列グループではないので、列は `None`)。

```python
    used_pairs = {s for (a, b) in PAIR_DOMAINS for s in (a, b) if a in p.text and b in p.text}
    for slot, dom in SLOT_DOMAINS.items():
        if slot in p.text and slot not in used_pairs:
            fs.append([(None, ({slot: s}, m)) for s, m in dom])
```
- `used_pairs` — 組として既に処理したスロット名(二重に埋めないように)。
- ひな形にあり、まだ使われていない**単独のスロット**ごとに、その選択肢すべてを持つ要素を追加する。

```python
    if "{表示列}" in p.text and "{表示列}" not in used_pairs:
        excl = {"氏名"} if ("氏名" in p.text or "{氏名}" in p.text) else set()
        fs.append([(None, ({"{表示列}": w}, {"表示列": c})) for c, ws in DISPLAY.items() if c not in excl for w in ws])
    return fs, bool(p.group and p.group2)
```
- 単独の `{表示列}`(表示する列)のスロット:文に既に氏名があれば `氏名` を除く。
- 要素と、フラグを返す:**列のスロットが2つあるとき True**(異なる列でなければならない)。

### 8.4 `render` — 文を1つ作る

```python
def render(p: Pattern, combo) -> tuple[str, dict] | None:
    cols = [c for c, _ in combo if c is not None]
    if len(cols) == 2 and cols[0].name == cols[1].name:
        return None
```
- `combo` = 各要素から1つずつの選択。2つの列が選ばれて、それが**同じ**列なら → `None` を返す(飛ばす。例:「年収と年収の比較」は意味がない)。

```python
    text, meta = p.text, {"パターン": p.pid, "質問タイプ": p.tid}
    for _, (subs, m) in combo:
        for k, v in subs.items():
            text = text.replace(k, v)
        meta.update(m)
```
- ひな形と、**文型IDと質問タイプID**を入れた意味の辞書(Day 4・5が確認する2つのキー)から始める。
- 各選択について:各スロットを文字に置き換え(`text.replace("{列}", "年収")`)、その意味を `meta` に合わせる(`dict.update`)。

```python
    if "表示列" in meta and "表示列2" not in meta and ("氏名" in p.text or "{氏名}" in p.text) and meta["表示列"] == "氏名":
        return None
```
- 文が既に人を名指ししているのに、氏名の列を表示してしまう文は飛ばす。

```python
    text = text.replace("性別別", "男女別").replace("男女ごと", "性別ごと")
    text = re.sub(r"(?<!が)(主任|係長|課長|次長|部長)の社員(?!数)", r"\1", text)   # 課長の社員 -> 課長
    text = re.sub(r"性別が(男|女)(ではない|以外)", r"\1性\2", text)            # 性別が女ではない -> 女性ではない
```
- 文を自然にする**手直し**:
  - `性別別` → `男女別`、`男女ごと` → `性別ごと`。
  - 「課長の社員」→「課長」(正規表現の `(?<!が)` は「直前が が でない」、`(?!数)` は「直後が 数 でない」の意味なので、「課長の社員数」は残る)。`\1` は、かっこで取った役職を戻す。
  - 「性別が女ではない/以外」→「女性ではない/以外」。

```python
    if "{" in text or "のの" in text or "社員の社員" in text or "平均平均" in text:
        raise ValueError(f"bad output {p.pid}: {text}")
    return text, meta
```
- **安全確認:**埋まっていないスロット(`{`)や、既知の悪いつながりがあれば、壊れた質問を静かに作る代わりに、**エラーで止まる**(raise)。(同じ印が、Day 3・4の `BROKEN_MARKS`。)そうでなければ、完成した文と意味を返す。

### 8.5 `count` — 文型が作れる質問の数

```python
def count(p: Pattern) -> int:
    fs, two = factors(p)
    total = 1
    for f in fs:
        total *= len(f)
    if two:  # remove combinations where both column slots use the same column
        from collections import Counter
        c1, c2 = Counter(c.name for c, _ in fs[0]), Counter(c.name for c, _ in fs[1])
        same = sum(c1[k] * c2[k] for k in c1)
        total = total // (len(fs[0]) * len(fs[1])) * (len(fs[0]) * len(fs[1]) - same)
    return total
```
- **すべての要素の大きさをかけ算**する = 組み合わせの数。
- 列のスロットが2つあるときは、両方が同じ列になる組み合わせを引く:`c1`/`c2` は、各要素が列名ごとに持つ選択肢の数を数え、`same` = 同じ列の組み合わせの数。2つの列の要素の分は `(len0 × len1 − same)` に置き換えられる。
- これが `patterns.csv` の数(「作れる質問の数」)。**組み合わせの数**であり、違う組み合わせが同じ文を作ることもあるので、本当に異なる文の数は少し少ないことがある(だから、重複を除いた後に、質問が少なくなる、または無くなる文型がある)。

### 8.6 `generate` — 文型ごとに、最大 `limit` 問を選ぶ

```python
def generate(p: Pattern, limit: int, rng: random.Random) -> list[tuple[str, dict]]:
    fs, _ = factors(p)
    n = count(p)
    seen: dict[str, dict] = {}
    if n <= limit * 3 and n <= 200000:
        for combo in itertools.product(*fs) if fs else [()]:
            r = render(p, combo)
            if r and r[0] not in seen:
                seen[r[0]] = r[1]
        items = list(seen.items())
        rng.shuffle(items)
        return items[:limit]
```
- **小さな文型**(組み合わせが `limit` の3倍以下で、かつ200,000以下):**すべて**の組み合わせを試す(`itertools.product(*fs)` = 要素ごとに1つ選ぶすべての方法。要素が無ければ、空の組み合わせ `()` が1つ)。新しい文は1回だけ残す(`seen` は文をキーにしているので、重複は落ちる)。シャッフルして、最初の `limit` 個を返す。

```python
    tries = 0
    while len(seen) < limit and tries < limit * 20:
        tries += 1
        r = render(p, [rng.choice(f) for f in fs])
        if r and r[0] not in seen:
            seen[r[0]] = r[1]
    return list(seen.items())
```
- **大きな文型**:全部は調べず、**ランダムに抜き取る** — 要素ごとにランダムに選択を1つ選び、`render` し、新しければ残す — `limit` 個見つかるか、`limit × 20` 回試すまで(だから `limit` より少し少なく返ることもある)。
- これが**バランス調整**:文型ごとに最大 `limit` 問なので、巨大な文型が小さな文型を埋もれさせない。共有されたシード付きの `rng` が、結果を再現可能にする。

---

## 9. `main`

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=0, help="max questions per pattern for questions.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pats = load_patterns()
    counts = {p.pid: count(p) for p in pats}
```
- オプション:`--corpus N`(0 = コーパスを書かない)と `--seed`。全体で共有する、シード付きの乱数生成器を1つ作る。全文型を読み込み、それぞれの数を計算する。

```python
    with open("patterns.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["パターンID", "質問タイプID", "質問タイプ", "文体", "文型", "列グループ", "列グループ2", "作れる質問の数"])
        for p in pats:
            w.writerow([p.pid, p.tid, p.tname, p.style, p.text, p.group, p.group2, counts[p.pid]])
```
- **`patterns.csv`**:文型ごとに1行(ID、タイプID、タイプ名、文体、ひな形、グループ、作れる質問の数)。`utf-8-sig` でExcelでも日本語が見える。これがDay 5が読むファイル。

```python
    types: dict[str, list] = {}
    for p in pats:
        t = types.setdefault(p.tid, [p.tname, 0, 0])
        t[1] += 1; t[2] += counts[p.pid]
    with open("question_types.csv", ...) as f:
        ... w.writerow(["質問タイプID", "質問タイプ", "パターン数", "作れる質問の数"])
        for k, (n, c, q) in types.items():
            w.writerow([k, n, c, q])
```
- 質問タイプごとの合計:`[名前, 文型の数, 作れる質問の合計]`(Day 4の報告書と同じ `setdefault` の技)。**`question_types.csv`** に書く。

```python
    with open("columns.csv", ...) as f:
        w = csv.writer(f); w.writerow(["列名", "種類", "質問で使う言い方", ...])
        for c in COLS:
            w.writerow([c.name, c.kind, "、".join(c.words), "、".join(c.avg_words or c.words), "、".join(c.vals), c.big, c.small, c.ask, "、".join(c.aggs)])
```
- **`columns.csv`**:列ごとのルール(言い方は `、` でつなぐ。`c.avg_words or c.words` は、平均用の言い方、無ければ普通の言い方)。

```python
    with open("samples.csv", ...) as f:
        w = csv.writer(f); w.writerow(["パターンID", ..., "生成された質問", "意味"])
        for p in pats:
            for q, m in generate(p, 5, rng):
                w.writerow([p.pid, p.tid, p.tname, p.text, q, json.dumps(m, ensure_ascii=False)])
```
- **`samples.csv`**:文型ごとに5つの例と、その意味(JSON、日本語はそのまま)。チームの確認用。

```python
    if a.corpus:
        total = 0
        with open("questions.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(["質問", "パターンID", "質問タイプID", "意味"])
            seen: set[str] = set()
            for p in pats:
                for q, m in generate(p, a.corpus, rng):
                    if q in seen:      # the same sentence can come from two patterns; keep the first
                        continue
                    seen.add(q)
                    w.writerow([q, p.pid, p.tid, json.dumps(m, ensure_ascii=False)]); total += 1
        print("questions.csv:", total, "questions")
```
- `--corpus N` のときだけ:**`questions.csv`**(Day 3〜7で使うファイル)を書く。ヘッダーは、Day 3・4の `QUESTION_HEADER`。
- 文型は順番に処理し、それぞれ最大 N 問。**先の文型**がすでに書いた文は飛ばす(`seen`)ので、ファイル全体に**重複する文はない** — そして、文がすべて既に使われていた後ろの文型は、質問が**0個**になる(Day 4・5で見た「質問が無い文型」)。

```python
    print("patterns:", len(pats), "| types:", len(types), "| all distinct questions:", sum(counts.values()))
    for cap in (500, 1000, 2000, 3000):
        print(f"  max {cap} per pattern -> {sum(min(v, cap) for v in counts.values())}")

if __name__ == "__main__":
    main()
```
- 文型の数、タイプの数、作れる質問の総数を表示する。次に**計画用の表**:文型ごとの上限を500 / 1000 / 2000 / 3000問にしたら、コーパスがどれくらいの大きさになるか(文型ごとに `min(v, cap)`)。Day 3は2000を使う。
- 「直接実行したときだけ動かす」のお決まりの終わり方(ここでは `sys.exit` は無し)。

---

## 10. 作り話の流れの例

ひな形(説明用の例):`{所属}の社員の{列}の{集計}を教えてください。|金額|`

1. `load_patterns` が `P001`(上の見出しからタイプ、グループは `金額`)を作る。
2. `factors`:要素1 = (金額の列、言い方、集計)の選択肢すべて。要素2 = `{所属}` の選択肢すべて(部署と勤務地と、その言い方)。
3. `count` = 大きさ1 × 大きさ2。`generate` が組み合わせを選ぶ(全部、またはランダムに抜き取り)。
4. `render` が、例えば(年収、平均)+(営業部)を取り、スロットを置き換えて、`営業部の社員の年収の平均を教えてください。` と、意味 `{"パターン":"P001","質問タイプ":"T..","列":"年収","集計":"平均","条件":"部署","条件値":"営業部"}` を作る。
5. 新しい文なら、`questions.csv` に書かれる。

## 11. 知っておくと良い小さなこと

- **すべてが、今いるフォルダー基準。**入力ファイルは、スクリプトを動かしているフォルダーから、読み込み時に読まれる。だからDay 3は `cwd=<新しいフォルダー>` で動かす。
- **シード付き + 並べ替え = 再現できる。**`random.Random(seed)` と、ファイル・部署・氏名などの `sorted(...)` により、どのマシンでも同じ出力になる(Day 3のSHA-256の比較は、これに頼っている)。
- **文型のIDは順番に依存する。**元の行を足したり消したりすると、後ろのIDが変わり、(文型IDで動く)Day 5の分割も変わる。
- **悪い出力は実行を止める。**`render` は飛ばさずに `ValueError` を出すので、間違ったひな形はすぐ見つかる。
- **`count` は組み合わせの数**で、異なる文の数が保証されたものではない。
- **`generate` は、大きな文型では `limit` より少なく返すことがある。**ランダムな抜き取りが重複に当たり続ける場合(`limit × 20` 回で諦める)。
- **使われない部分:**用語集のリスト `gl` と補助関数 `opts` は、このファイルでは後で使われない。`factors` の `pass` の分岐は何もしない。害はない。
- **`with` なしで開いたファイル。**一部の `open(...)` は明示的に閉じられていない。Pythonが終了時に閉じる。スクリプトなら問題ない。

## 12. 一段落でまとめ

`make_questions.py` は、チームの用語集、同義語リスト、サンプルの社員表、文型を読み込みます。文型ごとに、独立した選択肢のリスト(どの列か、どの部署か、どの年か、…)を作り、各リストから1つずつ選んで本物の質問に組み合わせ、言い方を手直しし、すべての質問に、JSONの**意味**を付けて保存します。コーパスは、文型ごとに最大 N 問で、同じ文は繰り返さないようにバランスを取り、`patterns.csv`、`question_types.csv`、`columns.csv`、`samples.csv`、そして(`--corpus` があれば)`questions.csv` を書き出します。
