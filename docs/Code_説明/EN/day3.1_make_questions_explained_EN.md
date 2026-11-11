# Day 3.1 — make_questions.py — Simple Line-by-Line Explanation (English)

> **How the two Day 3 files fit together**
> - **Day 3.1 (this file) `make_questions.py`** — the team's **question generator**: fills sentence patterns with words and makes thousands of Japanese questions, each with its *meaning* saved.
> - **Day 3 `day3_generate_questions.py`** — the *runner*: copies the inputs to a clean folder, runs this generator with fixed settings, checks the result and saves it (the run shown on video).
>
> Suggested reading order: **3.1 → 3**. (Days 4–5 later reuse the same generator's output.)

---

## 0. The big picture (read this first)

Imagine a **question factory** that works like a **fill-in-the-blank worksheet**.

You write a sentence with blanks:

```
{所属}の社員の{列}の{集計}を教えてください。
```
and the factory has lists of words for every blank:
- `{所属}` → 営業部, 開発部, 大阪支社, …
- `{列}` → 年収, 年俸, 月収, …
- `{集計}` → 平均, 合計, 最高額, …

It then tries **combinations** and produces real questions:

```
営業部の社員の年収の平均を教えてください。
開発部の社員の月収の最高額を教えてください。
...
```

And for **every question it also saves its meaning** as a small JSON label (e.g. `{"パターン":"P001","質問タイプ":"T05","条件":"部署","条件値":"営業部","列":"年収","集計":"平均"}`). Later days use these labels (Day 4/5 check them; they are **never** used as training text).

*(The example above is made up to show the idea; the real patterns are in the team's `src_*.txt` files.)*

### Key ideas (learn these 5 words)

| Idea | Meaning |
|---|---|
| **Pattern (文型)** | A sentence template with `{slots}`. Gets an ID `P001`, `P002`, … |
| **Question type** | A group of patterns that ask the same kind of thing (header lines `#T..` in the source files). |
| **Slot** | A blank like `{列}` or `{所属}`. |
| **Option** | One way to fill a slot: **(text to insert, meaning)**. Example: `("営業部", {"条件":"部署","条件値":"営業部"})`. |
| **Factor** | One independent list of options (e.g. "all department/workplace options"). A question = **one option from each factor**. The number of possible questions = **product of the factor sizes**. |

---

## 1. The inputs and outputs

**Inputs** (must be in the folder where the script runs):

| File | What it is |
|---|---|
| `用語集.csv` | The team's glossary |
| `同義語辞書.csv` | The synonym dictionary (variants of words) |
| `DB.xlsx` | The 50-row sample employee table (names, IDs, departments, workplaces, …) |
| `src_*.txt` | Pattern sources: lines starting with `#` are **question-type headers** (`#T..|name`), other lines are patterns `文型|列グループ|列グループ2` |

**Outputs** (written into the same folder):

| File | What it holds |
|---|---|
| `patterns.csv` | Every pattern with its type, style, and how many questions it can make |
| `question_types.csv` | Totals per question type |
| `columns.csv` | The rules per numeric column |
| `samples.csv` | 5 example questions per pattern, with meaning (for team review) |
| `questions.csv` | (only with `--corpus N`) up to **N questions per pattern**, with meanings |

Because the script reads and writes files **in the current folder**, Day 3 copies everything into a fresh folder and runs it there (so the team's `data/source` is never overwritten).

Usage:

```
python make_questions.py                  # patterns, counts and samples
python make_questions.py --corpus 2000    # also the balanced question corpus
```

---

## 2. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Slot / placeholder** | A `{...}` blank in a sentence template. |
| **Substitution** | Replacing a slot with real text (`text.replace("{列}", "年収")`). |
| **Cartesian product** | All combinations of choosing one item from each list (`itertools.product`). |
| **Permutation** | An ordered pair/arrangement, e.g. (A,B) and (B,A) are different (`itertools.permutations`). |
| **Combination** | An unordered choice, e.g. (A,B) only once (`itertools.combinations`). |
| **Seed / `random.Random`** | Repeatable randomness (see Day 5). |
| **Balancing** | Keeping at most N questions per pattern so huge patterns do not dominate. |
| **pandas / DataFrame** | A library for tables; `pd.read_excel` reads `DB.xlsx` into a table (needs `openpyxl`). |
| **Regular expression (`re`)** | A text-pattern mini-language (see Day 5). |
| **Meaning (意味)** | The JSON label saved with each question. |

---

## 3. The docstring and imports

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
- The docstring lists inputs, outputs and usage (this file has no Japanese part and is not a raw string).
- `argparse`, `csv`, `json`, `random`, `re`, `dataclass`/`field`, `from __future__` — same as earlier days. New:
  - `glob` — find files by a name pattern (`glob.glob("src_*.txt")`).
  - `itertools` — tools for combinations (`product`, `permutations`, `combinations`).
  - `pandas as pd` — the **third-party** table library (this is why Day 3 checks that `pandas` and `openpyxl` are installed).

```python
GLOSSARY, SYNONYMS, DB = "用語集.csv", "同義語辞書.csv", "DB.xlsx"
```
- Three file names assigned in one line (tuple unpacking). Relative names → files must be in the current folder.

---

## 4. The columns table (`Col` and `COLS`)

### 4.1 The `Col` dataclass

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
A description of **one data column** and how people talk about it:

| Field | Meaning | Example (年収) |
|---|---|---|
| `name` | The official column name | 年収 |
| `kind` | Type: 金額 (money), 数値 (number), 評価 (rating), 計算 (calculated), 比率 (ratio) | 金額 |
| `words` | Different ways to say it | 年収, 年俸, 年間給与, 年間収入 |
| `vals` | Example values for comparisons | 400万円 … 800万円 |
| `big` / `small` | Words for "large/small" | 高い / 低い |
| `ask` | The question word for its value | いくら |
| `aggs` | Allowed summaries | 平均, 合計, 最高額, 最低額 |
| `avg_words` | Wording to use **after "平均"** (when different) | (none) |
| `ops` | Comparison words allowed | 以上, 以下, 未満, 超 |

`field(default_factory=list)` gives each object its own empty list (see Day 5's `SplitResult`).

### 4.2 The lists `MONEY`, `NUM`, `COLS`

```python
MONEY = ["平均", "合計", "最高額", "最低額"]
NUM = ["平均", "最大値", "最小値"]
```
- The summaries allowed for money columns and for number columns.

`COLS` is the list of **21 columns**. Every line has the same shape (`Col(name, kind, words, vals, big, small, ask, aggs, ...)`), so here are the groups and the special cases:

| Kind | Columns | Notes |
|---|---|---|
| 金額 (money) | 年収, 月収, 基本給, 役職手当, 住宅手当, 通勤手当, 残業手当, 賞与, 昇給 | `ask`=いくら, summaries = `MONEY` |
| 数値 (number) | 年齢, 勤続年数, 月平均残業時間, 保有資格数, 有給取得率 | `ask` varies: 何歳 / 何年 / 何時間 / いくつ / 何パーセント. Summaries = `NUM` |
| 評価 (rating) | 直近総合評価, 直近健康診断結果, 直近メンタル診断結果 | values are small numbers; health/mental results have **no** big/small word and no summaries (you cannot say "a high health result") |
| 計算 (calculated) | 手当合計, 賞与を除いた年収 | computed from several columns; summary = 平均 only |
| 比率 (ratio) | 賞与の年収比, 残業手当の月収比, 基本給の月収比 | percentages; summary = 平均 only |

Special cases (they exist to keep sentences natural):
- **勤続年数** has `avg_words` = 勤続年数, 勤務年数, 在籍年数 — i.e. **not** 社歴, because "平均社歴" sounds odd.
- **月平均残業時間** has `avg_words` = 残業時間 — because "平均" + "月平均残業時間" would make the broken word **平均平均** (one of the `BROKEN_MARKS` that Day 3/4 look for).
- **保有資格数** has `ops` = 以上, 以下, 未満 only (no 超).
- **直近総合評価** has `avg_words` = 総合評価, 評価.

### 4.3 `BASE` and `GROUPS`

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
- `BASE` — the real data columns (money, number, rating).
- `GROUPS` — named **groups of columns**. A pattern line like `...|金額|` means "use the money columns for the column slot". The groups: 金額 (money), 大小 (columns that have a big/small word), 平均 (columns with summaries), 比較 (money or number columns, for "above X" comparisons), 評価 (rating columns), 計算, 比率.
- `if c.big` is true when the string is not empty.

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
- Turns a value text into a **real number** for the meaning label.
- First remove the unit words (`%`, 円, 歳, 年, 時間, つ).
- The regular expression `(?:(\d+)万)?(\d+)?$` reads "optionally *N*万, then optionally *M*". Examples:
  - `500万円` → `500万` → 500 × 10,000 = **5,000,000**
  - `1万5000円` → 1×10,000 + 5,000 = **15,000**
  - `5000円` → **5,000**
  - `30%`, `25歳` → plain numbers (30, 25) by the last line `float(s)`.
- `m.group(1) or 0` — use 0 if that part is missing.

---

## 5. Words from the team's files

```python
gl = list(csv.reader(open(GLOSSARY, encoding="utf-8-sig")))[1:]
syn: dict[str, list[str]] = {}
for r in list(csv.reader(open(SYNONYMS, encoding="utf-8-sig")))[1:]:
    syn.setdefault(r[2], []).append(r[3])
db = pd.read_excel(DB)
```
- These run **when the file loads**, in the current folder.
- `csv.reader(open(...))` reads the CSV row by row; `[1:]` skips the header row.
- `gl` — the glossary rows. (In this file `gl` is loaded but not used again.)
- `syn` — a dictionary **standard term → list of variants**: for each synonym row, `r[2]` (3rd column) is the standard term and `r[3]` (4th column) the variant. `setdefault(key, [])` creates the list the first time; `.append` adds the variant.
- `db = pd.read_excel(DB)` — the employee table as a pandas DataFrame (`db["部署"]` is the department column).

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
- Returns the standard word **plus only those synonyms that sound natural** in "〇〇の社員".
- A synonym is **rejected** if it contains `の`, ends with one of the `BAD_END` words (e.g. "…課", "…チーム", "…社員", which would read oddly), or ends with `た`, `る`, `い` (verb/adjective endings).
- `v.endswith(BAD_END)` accepts a tuple: true if it ends with any of them.

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
- `DEPTS`, `PLACES` — the different departments / workplaces in the table (`.unique()`), sorted.
- `RANKS` — fixed rank list, lowest to highest.
- `DEPT_W`, `PLACE_W` — dictionaries: each department / workplace → its list of natural wordings.
- `NAMES` — all names (`.tolist()`); `IDS` — employee numbers as text.
- `SURNAMES` — a **guess of each person's surname** (Japanese names have no space): if the name starts with 佐々木, 原, 林 or 森 use that; otherwise use the **first two characters**. `next(generator, default)` returns the first match or the default. A set removes duplicates; `sorted` fixes the order.

```python
REASONS = {...}   MARRIAGE = {...}   MISSING = {...}   DISPLAY = {...}
```
- Four small dictionaries of **wordings**:
  - `REASONS` — leave reasons (私傷病 → 私傷病/病気/傷病, 育児, 産前産後, 介護, 労災);
  - `MARRIAGE` — 既婚 / 未婚 wordings;
  - `MISSING` — columns that may be blank, with their names;
  - `DISPLAY` — columns that can be **shown** ("list their name and department") with wordings.

---

## 6. Options and slot domains

```python
O = tuple  # option = (surface, meta)
def opts(pairs) -> list[O]:
    return [(s, m) for s, m in pairs]
```
- `O` is just a short name for a tuple, documenting that an **option = (surface text, meta dictionary)**. `opts` is a small helper that is not used elsewhere in this file.

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
- Builds the options for "**which group of employees**" slots. Each option pairs a wording with its **meaning** (`条件` = which attribute, `条件値` = which value).
- 部署 / 勤務地: every wording of every department / workplace.
- 役職 (rank): for each rank: when it is not "person" style add the long form `役職が課長`; and (except 一般) add the short form `課長`. With `person=True` the long form and 一般 are skipped (so "…の人" phrasing sounds right).
- 性別: `男性` / `女性`, plus `性別が男` / `性別が女` when not person style. The meaning stores the short value 男/女.

### 6.2 `pair_options`

```python
def pair_options(word_map: dict[str, list[str]], key: str) -> list[O]:
    return [((a, b), {key: x, key + "2": y}) for x, y in itertools.permutations(word_map, 2) for a in word_map[x] for b in word_map[y]]
```
- For slots that need **two different items** (e.g. "営業部と開発部"). `permutations(word_map, 2)` gives every ordered pair of **different** keys (x, y); then every wording `a` of x and `b` of y. The meaning stores `部署` and `部署2` (or the given `key`).

### 6.3 `SLOT_DOMAINS` — the dictionary of single slots

Each entry: slot name → list of options. The slots (all with their meaning stored):

| Slot | What fills it |
|---|---|
| `{属性}` | department, rank, workplace or gender (long and short forms) |
| `{所属}` | department or workplace |
| `{人の属性}` | gender or rank in "person" style |
| `{性別語}` / `{役職語}` | 男性・女性 / ranks except 一般 |
| `{グループ}` | "group by what": 部署/部門, 役職, 勤務地/勤務先, 性別/男女 |
| `{グループ所}` / `{グループ人}` | restricted group-by lists (place-like / person-like) |
| `{区分列}` | a division column: 部署/部門, 役職, 勤務地/勤務先 |
| `{理由}` / `{休職事由}` | leave reasons (with wordings / canonical + その他) |
| `{婚姻}` | marital-status wordings |
| `{欠損列}` | columns that may be blank |
| `{氏名}` / `{姓}` / `{社員番号}` | a person's name / surname / ID |
| `{年}` / `{年2}` | a year 1990–2025 / 2021–2025 |
| `{月}` | a month 1–12 |
| `{年代}` | 20代 … 60代 |
| `{N}` / `{人数}` | top-N number (3, 5, 10) / a head-count (2人, 3人, 5人, 10人) |
| `{経過}` | years elapsed (1, 2, 3, 5年) |

Each option looks like `("20代", {"年代": 20})`.

### 6.4 `PAIR_DOMAINS` — slots that come in pairs

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
- Key = the two slot names; value = the list of options for **both together**. Two **different** departments / workplaces / ranks / names / display columns, and a **year range** (from year `a` to a later year `b`, at most 10 years later and not beyond 2026).

---

## 7. Column-bound options: `col_options`

```python
def col_options(col: Col, pat: str, second: bool = False) -> list[O]:
    sfx = "2" if second else ""
    out = []
    avg_ctx_pat = "平均" in pat or "中央値" in pat
    aggs = col.aggs if "{集計}" in pat else [""]
```
- Makes all options for **one column** used in pattern text `pat`.
- `second=True` means this is the **second** column slot, so slot names get a `2` suffix (`{列2}`, `{大2}` …).
- `avg_ctx_pat` — does the pattern talk about an average/median?
- `aggs` — if the pattern has a `{集計}` slot, loop over the column's allowed summaries; otherwise one empty entry.

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
- Pick the wording list: in an **average context** use the special `avg_words` if the column has them (avoids "平均平均"), otherwise the normal words.
- For each wording `w`, build `base` = "slot → text" substitutions: the column word, the big/small words and the question word. `meta` = the meaning (`列` = official column name). If there is a summary, add it to both.
- `variants` starts with that one (substitutions, meaning) pair.

```python
            if "{値" + sfx + "}" in pat:
                variants = [({**b, "{値" + sfx + "}": v, "{比較" + sfx + "}": c}, {**m, "値" + sfx: to_number(v), "比較" + sfx: c})
                            for b, m in variants for v in col.vals for c in col.ops]
```
- If the pattern has a **value slot** (like "年収が{値}{比較}"), expand every variant with **every example value × every comparison word**; the meaning stores the number (`to_number`) and the comparison. `{**b, ...}` copies dictionary `b` and adds keys.

```python
            if "{下}" in pat and not second:
                variants = [({**b, "{下}": col.vals[i], "{上}": col.vals[j]}, {**m, "下限": to_number(col.vals[i]), "上限": to_number(col.vals[j])})
                            for b, m in variants for i, j in itertools.combinations(range(len(col.vals)), 2)]
```
- If the pattern is a **range** ("{下}から{上}まで"), expand with every pair of values (lower < upper — `combinations` picks each pair once).

```python
            if "{評価列}" in pat:
                vals = col.vals[1:-1] if ("以上" in pat or "以下" in pat) else col.vals
                variants = [({**b, "{評価列}": w, "{評価値}": v}, {**m, "値": int(v)}) for b, m in variants for v in vals]
            out += variants
    return out
```
- If the pattern is a **rating** pattern: use all rating values, but **drop the first and last** when the pattern says 以上/以下 (so "rating ≥ lowest" is not trivial). Meaning stores the integer.
- Add the variants to `out`; return `out` after all loops.

---

## 8. Patterns

### 8.1 The `Pattern` dataclass and its style

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
- A pattern: its ID, its question type ID and name, the template text, and up to two **column groups** (`group` for the first column slot, `group2` for the second).
- `style` — a **guess of the writing style** from the template (this is the `文体` that Day 5 counts):
  - ends with `？`, short (≤ 22 characters) and no "ください": **短い**, unless it ends with る/た/の + `？` → **くだけた** (casual);
  - ends with ください / ですか / ますか / です + `。` → **丁寧** (polite);
  - otherwise **くだけた**.

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
- Reads every `src_*.txt` file (sorted, so the order is fixed).
- For each line: remove the line break; **skip blank lines**.
- A line starting with `#` is a **question-type header** `#T05|name`: `line[1:]` drops the `#`, `.split("|")` gives type ID and name — remembered for the patterns that follow.
- Otherwise it is a pattern line: `text|group|group2`. `(line.split("|") + ["", ""])[:3]` pads with two empty strings and keeps the first three, so missing groups become empty.
- `n += 1` and the ID `f"P{n:03d}"` → `P001`, `P002`, … (`:03d` = 3 digits with leading zeros).
- **Note:** IDs depend on the **order** of lines/files. Inserting a pattern in the middle renumbers all later patterns.

### 8.3 `factors` — the independent choice lists

```python
def factors(p: Pattern) -> tuple[list[list], bool]:
    """Independent choice lists for a pattern; bool = two columns that must differ."""
    fs: list[list] = []
    if p.group:
        fs.append([(c, o) for c in GROUPS[p.group] for o in col_options(c, p.text)])
    if p.group2:
        fs.append([(c, o) for c in GROUPS[p.group2] for o in col_options(c, p.text, second=True)])
```
- `fs` — the list of factors. Each **choice** is a pair `(column or None, (substitutions, meaning))`.
- First column slot: for each column in the pattern's group and each of its options → one choice. Second column slot: the same, with `second=True`.

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
- For each **pair-slot** whose two slots are both in the template: take its options. (The `if … pass` line does nothing.) If the slots are display columns **and** the sentence already contains a `{氏名}`, remove options that would display the name again. Add one factor (column `None` because it is not a column group).

```python
    used_pairs = {s for (a, b) in PAIR_DOMAINS for s in (a, b) if a in p.text and b in p.text}
    for slot, dom in SLOT_DOMAINS.items():
        if slot in p.text and slot not in used_pairs:
            fs.append([(None, ({slot: s}, m)) for s, m in dom])
```
- `used_pairs` — the slot names already handled as pairs (so they are not filled twice).
- For every **single slot** present in the template and not already used: add a factor with all its options.

```python
    if "{表示列}" in p.text and "{表示列}" not in used_pairs:
        excl = {"氏名"} if ("氏名" in p.text or "{氏名}" in p.text) else set()
        fs.append([(None, ({"{表示列}": w}, {"表示列": c})) for c, ws in DISPLAY.items() if c not in excl for w in ws])
    return fs, bool(p.group and p.group2)
```
- A single `{表示列}` (display column) slot: exclude `氏名` if the sentence already mentions a name.
- Return the factors plus a flag: **True when the pattern has two column slots** (they must be different columns).

### 8.4 `render` — build one sentence

```python
def render(p: Pattern, combo) -> tuple[str, dict] | None:
    cols = [c for c, _ in combo if c is not None]
    if len(cols) == 2 and cols[0].name == cols[1].name:
        return None
```
- `combo` = one choice from each factor. If two columns were chosen and they are the **same** column → return `None` (skip; e.g. "年収と年収の比較" makes no sense).

```python
    text, meta = p.text, {"パターン": p.pid, "質問タイプ": p.tid}
    for _, (subs, m) in combo:
        for k, v in subs.items():
            text = text.replace(k, v)
        meta.update(m)
```
- Start from the template and a meaning dictionary containing the **pattern ID and question-type ID** (the two keys Day 4/5 verify).
- For every choice: replace each slot with its text (`text.replace("{列}", "年収")`) and merge its meaning into `meta` (`dict.update`).

```python
    if "表示列" in meta and "表示列2" not in meta and ("氏名" in p.text or "{氏名}" in p.text) and meta["表示列"] == "氏名":
        return None
```
- Skip a sentence that would show the name column while the sentence already names a person.

```python
    text = text.replace("性別別", "男女別").replace("男女ごと", "性別ごと")
    text = re.sub(r"(?<!が)(主任|係長|課長|次長|部長)の社員(?!数)", r"\1", text)   # 課長の社員 -> 課長
    text = re.sub(r"性別が(男|女)(ではない|以外)", r"\1性\2", text)            # 性別が女ではない -> 女性ではない
```
- **Clean-ups** to make sentences natural:
  - `性別別` → `男女別`; `男女ごと` → `性別ごと`.
  - "課長の社員" → "課長" (the regex `(?<!が)` means "not preceded by が"; `(?!数)` means "not followed by 数", so "課長の社員数" is kept). `\1` puts back the rank that was captured in the brackets.
  - "性別が女ではない/以外" → "女性ではない/以外".

```python
    if "{" in text or "のの" in text or "社員の社員" in text or "平均平均" in text:
        raise ValueError(f"bad output {p.pid}: {text}")
    return text, meta
```
- **Safety check:** if a slot was left unfilled (`{`) or a known bad join appears, **stop with an error** (raise) instead of silently producing a broken question. (The same marks are `BROKEN_MARKS` in Day 3/4.) Otherwise return the finished sentence and its meaning.

### 8.5 `count` — how many questions a pattern can make

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
- Multiply the **sizes of all factors** = number of combinations.
- With two column slots, subtract the combinations where both use the same column: `c1`/`c2` count how many options each factor has per column name; `same` = number of same-column combinations; the two column-factors' share is replaced by `(len0 × len1 − same)`.
- This is the number in `patterns.csv` ("作れる質問の数"). It is a **combination count**; different combinations can still make the same sentence, so the real number of distinct sentences can be a little lower (which is why some patterns end with fewer or no questions after duplicate removal).

### 8.6 `generate` — pick up to `limit` questions for a pattern

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
- **Small pattern** (≤ 3 × limit combinations and ≤ 200,000): try **every** combination (`itertools.product(*fs)` = all ways to pick one choice per factor; if there are no factors, one empty combination `()`). Keep each new sentence once (`seen` is keyed by text, so duplicates are dropped), shuffle, and return the first `limit`.

```python
    tries = 0
    while len(seen) < limit and tries < limit * 20:
        tries += 1
        r = render(p, [rng.choice(f) for f in fs])
        if r and r[0] not in seen:
            seen[r[0]] = r[1]
    return list(seen.items())
```
- **Big pattern**: don't enumerate; **randomly sample** — pick one random choice from each factor, render it, keep it if new — until `limit` questions are found or `limit × 20` tries are used (so it can return slightly fewer than `limit`).
- This is the **balancing**: at most `limit` questions per pattern, so huge patterns do not drown small ones. The shared seeded `rng` makes the result repeatable.

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
- Options: `--corpus N` (0 = do not write the corpus) and `--seed`. One seeded random generator shared by everything. Load all patterns and compute each one's count.

```python
    with open("patterns.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["パターンID", "質問タイプID", "質問タイプ", "文体", "文型", "列グループ", "列グループ2", "作れる質問の数"])
        for p in pats:
            w.writerow([p.pid, p.tid, p.tname, p.style, p.text, p.group, p.group2, counts[p.pid]])
```
- **`patterns.csv`**: one row per pattern (ID, type ID, type name, style, template, groups, possible questions). `utf-8-sig` so Excel shows Japanese. This is the file Day 5 reads.

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
- Totals per question type: `[name, number of patterns, total possible questions]` (same `setdefault` trick as Day 4's report). Written to **`question_types.csv`**.

```python
    with open("columns.csv", ...) as f:
        w = csv.writer(f); w.writerow(["列名", "種類", "質問で使う言い方", ...])
        for c in COLS:
            w.writerow([c.name, c.kind, "、".join(c.words), "、".join(c.avg_words or c.words), "、".join(c.vals), c.big, c.small, c.ask, "、".join(c.aggs)])
```
- **`columns.csv`**: the rules per column (wordings joined with `、`, and `c.avg_words or c.words` = the average wording or, if none, the normal words).

```python
    with open("samples.csv", ...) as f:
        w = csv.writer(f); w.writerow(["パターンID", ..., "生成された質問", "意味"])
        for p in pats:
            for q, m in generate(p, 5, rng):
                w.writerow([p.pid, p.tid, p.tname, p.text, q, json.dumps(m, ensure_ascii=False)])
```
- **`samples.csv`**: 5 examples per pattern with their meaning (as JSON, Japanese kept) for the team to review.

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
- Only with `--corpus N`: write **`questions.csv`** (the file used by Days 3–7). Header = `QUESTION_HEADER` from Day 3/4.
- Patterns are processed in order; for each, up to N questions. A sentence already written by an **earlier pattern** is skipped (`seen`), so the whole file has **no duplicate sentences** — and a later pattern whose sentences were all already used ends with **zero questions** (that is the "patterns with no questions" you saw in Day 4/5).

```python
    print("patterns:", len(pats), "| types:", len(types), "| all distinct questions:", sum(counts.values()))
    for cap in (500, 1000, 2000, 3000):
        print(f"  max {cap} per pattern -> {sum(min(v, cap) for v in counts.values())}")

if __name__ == "__main__":
    main()
```
- Print the number of patterns and types and the total possible questions. Then a **planning table**: how big the corpus would be if you limited each pattern to 500 / 1000 / 2000 / 3000 questions (`min(v, cap)` per pattern). Day 3 uses 2000.
- Standard "run only when executed directly" ending (here without `sys.exit`).

---

## 10. A made-up walk-through

Template (illustration only): `{所属}の社員の{列}の{集計}を教えてください。|金額|`

1. `load_patterns` creates `P001` (type from the header above it, group `金額`).
2. `factors`: factor 1 = every (money column, wording, summary) option; factor 2 = every `{所属}` option (departments and workplaces with their wordings).
3. `count` = size1 × size2. `generate` picks combinations (all, or a random sample).
4. `render` takes e.g. (年収, 平均) + (営業部) → replaces the slots → `営業部の社員の年収の平均を教えてください。` with meaning `{"パターン":"P001","質問タイプ":"T..","列":"年収","集計":"平均","条件":"部署","条件値":"営業部"}`.
5. The sentence is written to `questions.csv` if it is new.

## 11. Small things worth knowing

- **Everything is relative to the current folder.** The input files are read at import time from the folder the script runs in, which is why Day 3 runs it with `cwd=<fresh folder>`.
- **Seeded and sorted = repeatable.** `random.Random(seed)`, `sorted(...)` for files, departments, names… give the same output on every machine (Day 3's SHA-256 comparison relies on this).
- **Pattern IDs depend on order.** Adding or removing a source line changes later IDs, which also changes Day 5's split (it works on pattern IDs).
- **Bad output stops the run.** `render` raises `ValueError` instead of skipping, so a faulty template is found immediately.
- **`count` is a combination count**, not a guaranteed number of distinct sentences.
- **`generate` may return fewer than `limit`** for big patterns if random sampling keeps hitting duplicates (it gives up after `limit × 20` tries).
- **Unused pieces:** the glossary list `gl` and the helper `opts` are not used further in this file; the `pass` branch in `factors` does nothing. They are harmless.
- **Files opened without `with`.** Some `open(...)` calls are not closed explicitly; Python closes them at exit. Fine for a script.

## 12. Quick summary in one paragraph

`make_questions.py` reads the team's glossary, synonym list, sample employee table and sentence patterns; for each pattern it builds independent lists of options (which column, which department, which year, …), combines one option from each list into real questions, cleans up the wording, saves a JSON **meaning** with every question, balances the corpus to at most N questions per pattern with no repeated sentences, and writes `patterns.csv`, `question_types.csv`, `columns.csv`, `samples.csv` and (with `--corpus`) `questions.csv`.
