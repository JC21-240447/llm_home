# day5_build_corpus.py — やさしい1行ずつの解説(日本語)

## 0. 全体像(まずここから)

大量の**練習問題**を持つ先生が、学校の試験の準備をするところを想像してください。

- ほとんどの問題は**宿題の山**(train = 学習用)に入れます。生徒はこれで練習します。
- 一部は**小テストの山**(val = 検証用)に入れます。先生が途中で上達を確かめるために使います。
- 一部は**期末試験の山**(test = テスト用)に入れます。生徒は最後まで絶対に見てはいけません。

危険なのは**うっかりカンニング**です。練習した文と「ほんの少し違うだけの文」が試験に出ると、生徒は賢く見えますが、実は暗記しただけです。このスクリプトは、それを防ぐために作られています。

やることを一覧にすると:

| 手順 | 関数 | やさしく言うと |
|---|---|---|
| 1 | `load_questions` | 質問を読み込み、壊れたものを弾く |
| 2 | `choose_held_out_patterns` | 行ではなく**文型を丸ごと**、val と test 用に選ぶ |
| 3 | `build_splits` | 3つの山を作り、ほぼ同じ文を除き、大きなグループに上限を付け、小さなタイプを train で繰り返す |
| 4 | `split_stats`、`coverage` | 行数、文字数、バイト数、トークンの範囲、語彙カバー率を数える |
| 5 | `write_lines`、`write_labels` | 山をファイルとして保存する |
| 6 | `write_data_card`、マニフェストJSON、`append_run_log` | 説明書と、指紋入りの「レシート」を書く |
| 7 | `main` の中の安全確認 | 山と山の間で何も漏れていないか確かめる |

実行方法:

```
.venv\Scripts\python.exe scripts\day5_build_corpus.py --machine home
```

結果は、入力ファイルと `--seed` だけで**完全に決まります**。だから学校PCと自宅ノートPCで、`train.txt`、`val.txt`、`test.txt` が1バイトも違わず同じになります(マニフェストの SHA-256 で証明できます)。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **train / val / test** | 練習の山 / 途中確認の山 / 期末試験の山。 |
| **データリーク(漏れ)** | テストの情報が学習に紛れ込むこと。点数が実力より良く見えてしまう。 |
| **文型(パターン)** | 文のひな形1つ(= 言い回し1つ)。同じ文型から作られた質問は、ほとんど同じ見た目になる。 |
| **質問タイプ** | 同じ種類のことを聞く文型のグループ(例:「給与を聞く」)。 |
| **取り分ける(held out)** | 学習から離して、別に保管すること。 |
| **ほぼ重複(near-duplicate)** | 句読点や空白が違うだけの2つの文。 |
| **NFKC正規化** | Unicodeの整理。全角の英数字(`ＡＢＣ１２３`)などの変種を標準の形(`ABC123`)にそろえる。 |
| **オーバーサンプリング** | 小さなグループの項目を繰り返して、モデルがそれを見る回数を増やすこと。 |
| **シード** | 乱数の出発点。同じシードなら、同じ「ランダム」な結果になる。 |
| **マニフェスト** | 何を作ったかを、設定や指紋と一緒に正確に記した「レシート」ファイル。 |
| **データカード** | データセットの説明書:出典、作り方、大きさ、限界など。 |
| **SHA-256** | ファイルの指紋(Day 4を参照)。 |
| **トークナイザー / トークン** | モデルのために文を小さな断片に切る道具。断片1つがトークン。 |
| **BPE(バイトレベル)** | バイトから始めて、よく出る組み合わせをくっつけていく、よく使われるトークナイザーの方式。 |
| **プリトークナイザー** | BPEの**前**に行う最初の切り分け。BPEは、この切れ目をまたいでは絶対に結合できない。 |
| **frozen dataclass(凍結データクラス)** | 作った後に中身を変更できない、データ用クラス。 |
| **正規表現(regex)** | 文字のパターンを書き表すミニ言語(ここでは「これらの句読点のどれか」)。 |

---

## 2. ファイル先頭のdocstring(説明文)

```python
r"""Day 5 - corpus split, light balancing, statistics and data card (Model A).
...
"""
```
- Day 1・Day 4と同じ形式:raw(`r`)の説明文で、英語と日本語があり、実行はされない。
- 入力(チームのコーパスZIP、`patterns.csv`、任意の用語集と同義語辞書)と、6つの手順を載せている。
- 大事な一文:**学習に使うのは 質問 の文だけ。意味 のラベルは別の `*_labels.csv` に書き出す。**(モデルが答えの一覧を見ないようにするため。)
- 学校PCの注意:CPUと標準ライブラリだけを使うので、両方のマシンで同じ出力になる。

---

## 3. import(道具を持ってくる)

```python
from __future__ import annotations
```
- Day 1・Day 4と同じ。

```python
import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
```

`argparse`、`csv`、`hashlib`、`io`、`json`、`sys`、`zipfile`、`Counter`、`defaultdict`、`dataclass`、`datetime`、`Path` は **Day 4と同じ**。新しいものは次のとおり:

| 行 | 何が使えるか |
|---|---|
| `import math` | 数学の道具。ここでは `math.ceil`(**切り上げ**)。 |
| `import random` | 乱数(shuffle、choice、sample)。 |
| `import re` | 正規表現。 |
| `import unicodedata` | Unicodeの道具。NFKC正規化に使う。 |
| `from dataclasses import ... field` | `field(...)` で、データクラスの属性に特別な設定ができる(ここでは「その属性専用の空の辞書」)。 |
| `from typing import Iterable` | 「ループで回せるもの何でも」(リスト、ジェネレーターなど)を表す型。 |

---

## 4. 定数

```python
SCRIPT_VERSION: str = "day5-v1"
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SPLITS: tuple[str, str, str] = ("train", "val", "test")
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
```
- `SCRIPT_VERSION` と `REPO_ROOT` — Day 4と同じ。
- `SPLITS` — 3つの山の名前を、決まった順番で入れた、ちょうど3つの文字列のタプル。スクリプトのあちこちのループで使う。
- `QUESTION_HEADER` — `questions.csv` が持つべきヘッダー(Day 4と同じ)。

---

## 5. データクラス(データを入れる「箱」)

### 5.1 `Question`

```python
@dataclass(frozen=True)
class Question:
    """One generated question and its internal labels."""

    text: str
    pattern_id: str
    type_id: str
    meaning: str  # JSON string, kept aside, never used as training text
```
- `@dataclass`(Day 4を参照)は、コンストラクターを自動で書いてくれる。
- `frozen=True` — `Question` を作った後は、**中身を変更できない**。うっかり書き換えるのを防ぎ、おまけに、凍結されたオブジェクトは集合に入れたり辞書のキーにしたりできる。
- 4つの項目:文、どの文型で作られたか、どの質問タイプか、JSONのラベル(`meaning`)。コメントは、ラベルが学習用の文章に**絶対**使われないことを思い出させる。

### 5.2 `PatternInfo`

```python
@dataclass(frozen=True)
class PatternInfo:
    """One row of patterns.csv."""

    pattern_id: str
    type_id: str
    type_name: str
    style: str
    template: str
```
- `patterns.csv` の1行を、変更できない箱にしたもの:パターンID、タイプID、タイプ名、文体、文のひな形(文型)。

### 5.3 `SplitConfig` — 設定を1か所に集めたもの

```python
@dataclass
class SplitConfig:
    """All knobs of the split. Every value is written into the manifest."""

    seed: int = 42
    val_fraction: float = 0.07
    test_fraction: float = 0.07
    min_type_lines: int = 100
    min_pattern_lines: int = 20
    max_donate_share: float = 0.4
    eval_cap_per_pattern: int = 500
    oversample_target: int = 200
    max_repeat: int = 5
```
各行は「名前: 型 = 初期値」。各つまみの意味:

| 設定 | 意味 |
|---|---|
| `seed = 42` | 乱数の出発点(結果を再現できる)。 |
| `val_fraction = 0.07` | (質問がある)文型の約7%を検証用に取り分ける。 |
| `test_fraction = 0.07` | 約7%をテスト用に取り分ける。 |
| `min_type_lines = 100` | 質問タイプは、文型を1つ差し出すために、最低100問が必要。 |
| `min_pattern_lines = 20` | 差し出す文型は、最低20問が必要(小さすぎて何も測れないものを避ける)。 |
| `max_donate_share = 0.4` | ...そして、そのタイプの質問の最大40%まで(train にタイプのほとんどが残るように)。 |
| `eval_cap_per_pattern = 500` | 取り分けた文型ごとに、最大500問まで残す。 |
| `oversample_target = 200` | train の小さなタイプは、約200行になるまで繰り返す... |
| `max_repeat = 5` | ...ただし、最大5コピーまで。 |

凍結されていないので、値を置き換えられる(コマンドラインのオプションがそうする)。docstringにある通り、全部の値がマニフェストに保存されるので、どの設定で分割したかをいつでも確認できる。

### 5.4 `SplitResult`

```python
@dataclass
class SplitResult:
    assignment: dict[str, str]
    lines: dict[str, list[Question]]
    train_unique: list[Question]
    repeat_factor: dict[str, int]
    dropped_near_dup: dict[str, int] = field(default_factory=dict)
    dropped_by_cap: dict[str, int] = field(default_factory=dict)
```
分割の処理が返す「ひとまとめ」:
- `assignment` — パターンID → どの山か(`"train"`、`"val"`、`"test"`)。
- `lines` — 山の名前 → その山の質問のリスト(train は増やした後のもの)。
- `train_unique` — 繰り返す**前**の train(本当に異なる質問の数)。
- `repeat_factor` — タイプID → train でのコピー数。
- `dropped_near_dup`、`dropped_by_cap` — 山ごとに除いた行数。
- `field(default_factory=dict)` —「値が無ければ、**このオブジェクト専用の新しい空の辞書**を作る」。(`= {}` と直接書くのはPythonでは危険で、全オブジェクトが同じ辞書を共有してしまう。)最後の2つは初期値を持つので、初期値のない項目の後ろに置かれている。

---

## 6. 読み込み

### 6.1 `sha256_of`

```python
def sha256_of(path: Path) -> str:
    """SHA-256 of a file, so anyone can prove the inputs and outputs match."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```
目的はDay 4の `sha256_file` と同じだが、巨大なファイルがメモリに収まらなくてもよいように、**少しずつ**処理する。
- `h = hashlib.sha256()` — 空のハッシュ計算機を作る。
- `path.open("rb")` — 生のバイトで読むために開く(`b` = バイナリ)。
- `iter(lambda: f.read(1 << 20), b"")` — `f.read(1 << 20)`(1MiB = 1,048,576バイトを読む)を、空の `b""`(ファイルの終わり)が返るまで繰り返し呼ぶ。`lambda` は1行の小さな関数。`1 << 20` は「1を20ビット左にずらす」= 2²⁰。
- `h.update(chunk)` — 読んだ断片を計算機に渡す。
- `h.hexdigest()` — 最終的な指紋を文字列で返す。

### 6.2 `read_questions_text`

```python
def read_questions_text(source: Path) -> str:
    """Return the text of questions.csv, read from a .zip or a plain .csv."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            members = [n for n in zf.namelist() if n.endswith("questions.csv")]
            if len(members) != 1:
                raise SystemExit(f"{source}: expected exactly one questions.csv inside, found {members}")
            return zf.read(members[0]).decode("utf-8-sig")
    return source.read_text(encoding="utf-8-sig")
```
- 考え方はDay 4の `read_corpus_bytes` と同じ(ZIPなら、ちょうど1つの `questions.csv` を探す。そうでなければ普通のファイルを読む)。違いは1つだけ:生のバイト列ではなく**文字列**(BOMを取り除く `utf-8-sig` でデコード)を返すこと。ここではバイト単位の比較が要らないため。

### 6.3 `load_questions`

```python
def load_questions(source: Path) -> list[Question]:
    """Load and validate the question corpus. Stops with a clear message on bad data."""
    reader = csv.reader(io.StringIO(read_questions_text(source)))
    header = next(reader)
    if header != QUESTION_HEADER:
        raise SystemExit(f"Unexpected header {header}; expected {QUESTION_HEADER}")
```
- テキストを読み、ファイルのように扱えるようにし、CSVリーダーを作り、最初の行をヘッダーとして確認する(Day 4の `parse_corpus` と同じ考え方)。

```python
    questions: list[Question] = []
    seen: set[str] = set()
    type_of_pattern: dict[str, str] = {}
    problems: list[str] = []
```
- 4つの入れ物:
  - `questions` — 受け入れた良い質問。
  - `seen` — すでに受け入れた文の集合(重複をすばやく見つけるため)。
  - `type_of_pattern` — パターンID → そのタイプID(1つの文型が2つのタイプを持たないことの確認用)。
  - `problems` — 見つけた問題の説明。

```python
    for line_no, row in enumerate(reader, start=2):
```
- `enumerate(reader, start=2)` は、カウンターと行を**同時に**くれる。1行目はヘッダーなので2から始める。メッセージの「57行目」が、実際のファイルの行と合う。

```python
        if len(row) != 4:
            problems.append(f"line {line_no}: {len(row)} fields")
            continue
```
- **チェックA:** ちょうど4列か。違えば問題として記録し、`continue`(次の行へ飛ぶ)。*Day 4より慎重:* 悪い行は安全に飛ばすので、4つに分けて受け取るときにスクリプトが落ちることがない。

```python
        text, pid, tid, meaning = (c.strip() for c in row)
```
- 4つの列を4つの名前に受け取る。`.strip()` で各セルの前後の空白を取る(ジェネレーター式を使った受け取り)。

```python
        if not text or "\n" in text or "\r" in text:
            problems.append(f"line {line_no}: empty or multi-line question")
            continue
```
- **チェックB:** 質問は空でなく、改行(`\n` や `\r`)を含まないこと。学習用ファイルは**1行に1問**だから。

```python
        if text in seen:
            problems.append(f"line {line_no}: duplicate question {text!r}")
            continue
```
- **チェックC:** 重複。`{text!r}` は文を引用符付き(`repr`)で表示するので、見えない空白に気づきやすい。

```python
        if type_of_pattern.setdefault(pid, tid) != tid:
            problems.append(f"line {line_no}: pattern {pid} has two types")
            continue
```
- **チェックD:** 1つの文型は1つのタイプ。`setdefault(pid, tid)` は、この文型が初めてなら `tid` を保存し、保存されている値を返す。保存済みのタイプがこの行のタイプと違えば、その文型は矛盾している。

```python
        try:
            label = json.loads(meaning)
        except json.JSONDecodeError:
            problems.append(f"line {line_no}: 意味 is not valid JSON")
            continue
        if label.get("パターン") != pid or label.get("質問タイプ") != tid:
            problems.append(f"line {line_no}: 意味 does not match its pattern/type columns")
            continue
```
- **チェックE:** ラベルは正しいJSONで、その中のパターンとタイプが、行の列と一致すること(Day 4と同じチェック)。

```python
        seen.add(text)
        questions.append(Question(text, pid, tid, meaning))
```
- **全部の**チェックに合格した行だけ:その文を覚え、`Question` として追加する。

```python
    if problems:
        print(f"{len(problems)} problem(s) in the corpus; first 10:")
        for p in problems[:10]:
            print("  ", p)
        raise SystemExit("Fix the corpus (or regenerate it with the team's generator) and run again.")
    return questions
```
- 問題があれば:件数を表示し、最初の10件(`problems[:10]` = 先頭10個のスライス)を見せて、止める。なければ、きれいなリストを返す。

### 6.4 `load_patterns`

```python
def load_patterns(path: Path) -> dict[str, PatternInfo]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["パターンID"]: PatternInfo(r["パターンID"], r["質問タイプID"], r["質問タイプ"], r["文体"], r["文型"])
        for r in rows
    }
```
- `patterns.csv` を辞書として読み(Day 4の `load_pattern_table` と同じ)、**辞書内包表記**で、パターンID → `PatternInfo` の箱にする。`patterns["P012"].type_id` のようにすばやく引ける。

### 6.5 `load_terms`

```python
def load_terms(glossary: Path, synonyms: Path) -> tuple[list[str], list[str]]:
    """Standard terms and their variants, for the coverage report. Missing files -> empty lists."""
    terms: list[str] = []
    variants: list[str] = []
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            terms = sorted({r["標準用語"].strip() for r in csv.DictReader(f) if r["標準用語"].strip()})
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            variants = sorted({r["変形・同義語"].strip() for r in csv.DictReader(f) if r["変形・同義語"].strip()})
    return terms, variants
```
- 空のリストを2つ作って始める。これらのファイルは**任意**なので、無ければ空のまま。
- `{ ... for r in ... if ...}` は**集合内包表記**:`標準用語`(標準用語)の列を取り、空白を取り、空のセルを飛ばし、重複を自動で除く(集合)。`sorted(...)` で、決まった順番のリストにする。
- 2つ目のブロック(`変形・同義語` の列、変形)は、同義語辞書について同じことをしている。

---

## 7. 分割

### 7.1 ほぼ重複のキー

```python
_NEAR_DUP_STRIP = re.compile(r"[\s。、，．,.!！?？・「」『』（）()〜~ー－\-]")
```
- `re.compile(...)` は正規表現を一度だけ作って再利用する。先頭の `_` は「このファイルの内部用」の印。
- `[ ... ]` の中は**文字の集合**:次のどれか1文字に一致する:空白(`\s`)、日本語と欧文の句読点、かっこ、波ダッシュ、長音記号 `ー`、ハイフン。

```python
def near_dup_key(text: str) -> str:
    """Text with punctuation, spaces and long-vowel marks removed (after NFKC)."""
    return _NEAR_DUP_STRIP.sub("", unicodedata.normalize("NFKC", text))
```
- まず `unicodedata.normalize("NFKC", text)` で変種をそろえる(全角 → 通常)。
- 次に `.sub("", ...)` で、一致した文字を全部空に置き換える(削除する)。
- 結果 = **言い回しの指紋**。例:`給与は？` と `給与は` は同じキーになり、ほぼ重複と見なされる。

### 7.2 `choose_held_out_patterns` — どの文型を val/test に回すか

```python
def choose_held_out_patterns(
    questions: list[Question], patterns: dict[str, PatternInfo], cfg: SplitConfig
) -> dict[str, str]:
```
- 入力:全質問、文型の表、設定。出力:質問がある**すべて**の文型について、パターンID → 山の名前の辞書。
- docstringがルールを説明している(同じシードなら常に同じ結果):
  - 質問が `min_type_lines` 以上で、空でない文型が3つ以上あるタイプだけが、文型を差し出せる → どのタイプも、train に**最低2つの言い回し**が残る。
  - 選ばれたタイプは**ちょうど1つ**の文型を差し出す。val 用のタイプと test 用のタイプは**別のタイプ** → 取り分けた分が、多くのタイプにわたる。
  - 差し出す文型は、`min_pattern_lines` 問以上で、タイプの `max_donate_share` 以下であること。

```python
    rng = random.Random(cfg.seed)
```
- シード付きの**自分専用**の乱数生成器を作る。Pythonのグローバルな乱数の状態には触れないので、結果はシードだけで決まる。

```python
    per_pattern = Counter(q.pattern_id for q in questions)
    per_type = Counter(q.type_id for q in questions)
```
- 文型ごと、タイプごとの質問数を数える。(2行は同じ考えで、キーが違うだけ。)

```python
    patterns_of_type: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(per_pattern):
        patterns_of_type[patterns[pid].type_id].append(pid)
```
- 「タイプ → そのタイプの、空でない文型のリスト」を作る。`sorted(...)` で回すと順番が固定され、ランダムな選び方も再現できる。`defaultdict(list)` は、無いキーを空のリストから始める。

```python
    def candidates_of(tid: str) -> list[str]:
        return [
            p for p in patterns_of_type[tid]
            if cfg.min_pattern_lines <= per_pattern[p] <= cfg.max_donate_share * per_type[tid]
        ]
```
- 補助関数:タイプ `tid` の文型のうち、差し出して**よい**ものはどれか? 連鎖比較 `a <= x <= b` は「x が a と b の間」の意味:20問以上で、かつタイプ全体の40%以下。

```python
    eligible: list[str] = []
    for tid in sorted(patterns_of_type):
        candidates = candidates_of(tid)
        if per_type[tid] >= cfg.min_type_lines and len(patterns_of_type[tid]) >= 3 and candidates:
            eligible.append(tid)
```
- **対象になるタイプ**を集める:質問が十分(100以上)、文型が3つ以上、差し出してよい候補が1つ以上。(`and candidates` はリストが空でないときだけ真。)

```python
    n_patterns = len(per_pattern)
    n_val = round(cfg.val_fraction * n_patterns)
    n_test = round(cfg.test_fraction * n_patterns)
```
- 取り分ける文型の数:空でない文型全体の7%を四捨五入して val 用に、同様に test 用に。(2行目は、test の割合を使った同じ考え方。)

```python
    if n_val + n_test > len(eligible):
        print(f"WARNING: only {len(eligible)} eligible types; reducing held-out patterns.")
        n_val = len(eligible) // 2
        n_test = len(eligible) - n_val
```
- 安全策:取り分けたい文型の数が、対象タイプの数(各タイプは1つしか差し出さない)より多ければ、警告を出し、対象タイプを半分に分ける(`//` は整数の割り算)。

```python
    rng.shuffle(eligible)
    assignment = {pid: "train" for pid in per_pattern}
    for split, chosen_types in (("val", eligible[:n_val]), ("test", eligible[n_val:n_val + n_test])):
        for tid in sorted(chosen_types):
            assignment[rng.choice(candidates_of(tid))] = split
    return assignment
```
- `rng.shuffle(eligible)` — 対象タイプをその場でシャッフルする(シード付きで再現できる)。
- まず**全部**の文型を `"train"` にしておく。
- 先頭の `n_val` 個のタイプを検証用、次の `n_test` 個を test 用にする(重ならない2つのスライス → タイプの集合は別々になる)。
- 選ばれた各タイプ(これも再現のために並べ替えた順で)について、`rng.choice(...)` で差し出してよい文型を1つランダムに選び、山を `val` か `test` に変える。
- 割り当てを返す。

### 7.3 `build_splits` — 3つの山を作る

```python
def build_splits(questions: list[Question], assignment: dict[str, str], cfg: SplitConfig) -> SplitResult:
    """Apply the pattern assignment, remove near-duplicates, cap eval patterns, oversample train."""
    rng = random.Random(cfg.seed + 1)
```
- 新しいシード付き乱数生成器(`seed + 1`なので、前の関数とは別の乱数の流れだが、再現はできる)。

```python
    by_split: dict[str, list[Question]] = {s: [] for s in SPLITS}
    for q in questions:
        by_split[assignment[q.pattern_id]].append(q)
```
- 空のリストを3つ作り、全質問を、その**文型**が割り当てられた山に入れる。(文型ごと丸ごと移動する。これが「文型単位の分割」。)

**ステップA:ほぼ重複のガード**

```python
    dropped_near_dup: dict[str, int] = {}
    blocked = {near_dup_key(q.text) for q in by_split["train"]}
    for split in ("val", "test"):
        kept = [q for q in by_split[split] if near_dup_key(q.text) not in blocked]
        dropped_near_dup[split] = len(by_split[split]) - len(kept)
        by_split[split] = kept
        blocked |= {near_dup_key(q.text) for q in kept}
```
- `blocked` = train の全質問の、言い回しの指紋(「生徒がもう見たもの」)。
- val、次に test について:指紋が `blocked` に**無い**質問だけを残し、除いた数を数え、山を残したものに置き換える。
- `blocked |= {...}` — 残した質問の指紋も `blocked` に足す。結果:**train が勝ち、val は train に負け、test は train と val に負ける。**どの山も、他の山と同じ言い回しを持たない。

**ステップB:取り分けた文型に上限を付ける**

```python
    dropped_by_cap: dict[str, int] = {}
    for split in ("val", "test"):
        grouped: dict[str, list[Question]] = defaultdict(list)
        for q in by_split[split]:
            grouped[q.pattern_id].append(q)
        capped: list[Question] = []
        dropped = 0
        for pid in sorted(grouped):
            items = grouped[pid]
            if len(items) > cfg.eval_cap_per_pattern:
                dropped += len(items) - cfg.eval_cap_per_pattern
                items = rng.sample(items, cfg.eval_cap_per_pattern)
            capped.extend(items)
        dropped_by_cap[split] = dropped
        by_split[split] = capped
```
- val と test について:質問をパターンIDでグループにする。
- 各文型(並べ替えた順)について:500問を超えていたら、超えた分を除いた数として数え、ちょうど500問を**ランダムに抜き出して**残す(`rng.sample` は重複なしで選ぶ)。残したものを `capped` に追加する。
- 除いた数を保存し、山を置き換える。コードのコメントにある通り、除いた行は train に**移さない**。ただ捨てる。移すと、取り分けた文型が学習に漏れてしまうため。
- なぜ上限を付けるのか? 1つの巨大な文型が試験の大部分を占めて、点数をゆがめるのを防ぐため。

**ステップC:train の中だけで、小さなタイプを増やす**

```python
    train_unique = list(by_split["train"])
    per_type = Counter(q.type_id for q in train_unique)
    repeat_factor = {
        tid: min(cfg.max_repeat, max(1, math.ceil(cfg.oversample_target / n))) for tid, n in per_type.items()
    }
```
- `train_unique` — 繰り返す前の train の**コピー**(`list(...)` でコピーを作る)。
- train の質問をタイプごとに数える。
- 各タイプのコピー数の式:
  `ceil(200 / n)` を切り上げ → 最低1(`max(1, ...)`)→ 最大5(`min(5, ...)`)。
  例:40問のタイプ → ceil(5.0)=5コピー、150問 → ceil(1.33)=2、500問 → ceil(0.4)=1 → 1のまま、10問 → ceil(20)=20 → 5に制限。

```python
    train: list[Question] = []
    for q in train_unique:
        train.extend([q] * repeat_factor[q.type_id])
    by_split["train"] = train
```
- train の各質問を、`repeat_factor` の数だけ追加する。`[q] * 3` は同じオブジェクト3つのリストを作り、`extend` が全部を追加する。train の山を、大きくなったリストに置き換える。

```python
    for split in SPLITS:
        rng.shuffle(by_split[split])
    return SplitResult(assignment, by_split, train_unique, repeat_factor, dropped_near_dup, dropped_by_cap)
```
- 各山をシャッフルする(シード付き)。行が文型ごとに固まらず、繰り返したコピーも混ざる。
- すべてを `SplitResult` に入れて返す。

---

## 8. 統計

### 8.1 `script_class`

```python
def script_class(ch: str) -> str:
    """Script class of one character, matching the planned Day 6 pre-tokenizer."""
    o = ord(ch)
```
- 文字 `ch` が**どの種類の文字**かを答える。`ord(ch)` はそのUnicodeの番号(コードポイント)。Day 6で予定しているプリトークナイザーも、同じ分類で文を切る。

```python
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
```
- 0x3041〜0x309F(16進数)の番号は**ひらがな**。

```python
    if (0x30A1 <= o <= 0x30FA) or o in (0x30FC, 0x30FD, 0x30FE) or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "katakana"
```
- **カタカナ**:本体の範囲に加えて、長音記号 `ー` と繰り返し記号(0x30FC〜0x30FE)、小書きカタカナの拡張、半角カタカナ。

```python
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or o in (0x3005, 0x3006, 0x30F6):
        return "kanji"
```
- **漢字**:主要なCJK漢字の範囲、拡張Aの範囲、そして漢字のように扱う特別な文字 `々`、`〆`、`ヶ`。

```python
    if ch.isascii() and ch.isalpha() or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
        return "latin"
```
- **ラテン文字**:普通のASCIIの英字、または全角のA–Z / a–z。(Pythonは `and` を `or` より先に評価するので、「(ASCIIかつ英字)、または全角の大文字、または全角の小文字」と読む。)

```python
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return "other"
```
- 数字(全角の数字も含む)、空白、それ以外のすべて(句読点、記号)→ `"other"`。

### 8.2 `pretoken_chunks`

```python
def pretoken_chunks(text: str) -> int:
    chunks = 0
    previous = ""
    for ch in text:
        cls = script_class(ch)
        if cls != previous or cls == "other":
            chunks += 1
        previous = cls
    return chunks
```
- **同じ種類の文字が続く塊(ラン)**を数える。種類が変わったとき、**または**文字が `"other"` のとき(つまり句読点は1つずつが別の塊)に、新しい塊が始まる。
- 例:`給与はいくら？` → 漢字 `給与`(1)→ ひらがな `はいくら`(2)→ other `？`(3)→ **3塊**。
- なぜ重要か(docstringより):このプリトークナイザーを使うバイトレベルBPEは、**塊をまたいで結合できない**ので、どの塊も最低1トークンになる。つまり塊の数は、トークン数の確実な**下限**。UTF-8のバイト数は**上限**(最悪でも1バイトにつき1トークン)。本当のトークン数は、Day 7でチームのトークナイザーで測る。

### 8.3 `split_stats`

```python
def split_stats(qs: list[Question]) -> dict[str, int]:
    texts = [q.text for q in qs]
    return {
        "lines": len(texts),
        "unique_lines": len(set(texts)),
        "characters": sum(len(t) for t in texts),
        "utf8_bytes": sum(len(t.encode("utf-8")) for t in texts),
        "token_lower_bound": sum(pretoken_chunks(t) for t in texts),
        "patterns": len({q.pattern_id for q in qs}),
        "types": len({q.type_id for q in qs}),
        "max_chars": max((len(t) for t in texts), default=0),
    }
```
- 文を集めて、次を計算する:
  - `lines` — 行数、`unique_lines` — 異なる行の数(集合が繰り返しを除く。train では、オーバーサンプリングの効果が見える)。
  - `characters` — 総文字数。`utf8_bytes` — 総バイト数(`t.encode("utf-8")` は文字列をバイトに変える。日本語は1文字3バイト)。
  - `token_lower_bound` — 全文の塊の数の合計。
  - `patterns`、`types` — 出てくる文型・タイプの種類数。
  - `max_chars` — 一番長い行。`default=0` は、山が空でもエラーにならないための指定。

### 8.4 `coverage`

```python
def coverage(texts: Iterable[str], words: list[str]) -> int:
    """How many of `words` occur at least once in the texts (substring match)."""
    blob = "\n".join(texts)
    return sum(1 for w in words if w in blob)
```
- すべての文を1つの大きな文字列(`blob`)につなぐ。
- 各単語について、`w in blob` は、その単語が中のどこかに現れれば真。`sum(1 for ...)` で真の数を数える。
- **部分文字列**の検査なので、短い単語は長い単語の中にも一致しうる(データカードにも注意書きがある)。

---

## 9. ファイルの書き出し

### 9.1 `write_lines`

```python
def write_lines(path: Path, qs: list[Question]) -> None:
    """One question per line, UTF-8 without BOM, LF line endings (what the tokenizer reads)."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for q in qs:
            f.write(q.text + "\n")
```
- `"w"` = 書き込み(上書き)。`encoding="utf-8"` = BOMなしのUTF-8。
- `newline="\n"` — Windows(普通はCRLFで書く)でも、常に単純なLFの改行にする。これで両方のマシンでファイルが同一(同じSHA-256)になる。
- 各質問を書いて改行する。書くのは質問の文だけ — **ラベルは入れない**。

### 9.2 `write_labels`

```python
def write_labels(path: Path, qs: list[Question], split: str) -> None:
    """Internal label file (not training text). Same order as the .txt file."""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["質問", "パターンID", "質問タイプID", "split", "意味"])
        for q in qs:
            w.writerow([q.text, q.pattern_id, q.type_id, split, q.meaning])
```
- ラベル付きのCSV。`.txt` と**同じ順番**なので、テキストのN行目とラベルのN行目が対応する。
- `utf-8-sig` はBOMを付けるので、Excelで日本語が正しく開ける。`newline=""` は `csv` モジュールが必要とする指定。
- `csv.writer` は、セルの中の引用符やカンマを自動で処理する(`meaning` はカンマと引用符だらけのJSONなので重要)。
- まずヘッダー行を書き、次に質問ごとに1行(`split` の名前も含む)を書く。

### 9.3 `fmt`

```python
def fmt(n: int | float) -> str:
    return f"{n:,}" if isinstance(n, int) else f"{n:,.2f}"
```
- 報告用に数を整える:整数は3桁区切りのカンマ(`12,345`)、小数はカンマと小数第2位(`1,234.50`)。

### 9.4 `write_data_card`

```python
def write_data_card(path, stats, per_type_rows, result, patterns, empty_patterns, cov, n_terms, n_variants, cfg, source_name, source_sha) -> None:
    """docs/DATA_CARD.md. Every number comes from this run; nothing is typed by hand."""
```
- データカード(Markdown文書)を書く。引数は必要なものすべて:統計、タイプごとの表の行、分割結果、文型の表、空の文型、カバー率、用語の数、設定、入力ファイル名とその指紋。

```python
    held = {s: sorted(p for p in ... if sp == s) for s in ("val", "test")}
    style_counts = {s: Counter(patterns[p].style for p in held[s]) for s in held}
```
- `held` — val と test それぞれについて、その山に入ったパターンIDの並べ替えたリスト。(1つの辞書内包表記。)
- `style_counts` — それぞれの山で、取り分けた文型の文体ごとの数。

```python
    lines: list[str] = [ ... ]
```
- そして、Markdownの行の大きなリスト。説明文書のテキストなので、1文ずつではなく、セクションの地図で説明する:

| セクション | 書いてあること | 数字の出どころ |
|---|---|---|
| タイトルと日付 | 誰がいつ作ったか | `SCRIPT_VERSION`、`datetime.now()` |
| 1. What the corpus is | 会社の社員データについての短い日本語の質問 | 固定の文 |
| 2. Sources and license | チームが作ったものの表、「外部のテキストは使っていない」、入力ファイル名とSHA-256 | `source_name`、`source_sha` |
| 3. How it was split | 行でなく文型単位、取り分けた文型の数、ルール、除いたほぼ重複、上限、シード、取り分けた文体 | `held`、`cfg`、`result.dropped_*`、`style_counts` |
| 4. Balancing | train だけのオーバーサンプリング | `cfg.oversample_target`、`cfg.max_repeat` |
| 5. Size | 山ごとの表 | `stats` |
| 6. Per question type | タイプごとの表 | `per_type_rows` |
| 7. Vocabulary coverage | 用語集の用語・変形がいくつ現れるか | `cov`、`n_terms`、`n_variants` |
| 8. Known limits and biases | 弱点の正直なリスト(狭い領域、テンプレート製、名前の繰り返し、タイプの偏り、空の文型、あいまいな言葉) | 固定の文 + `empty_patterns` |
| 9. Reproduce | もう一度実行するコマンド | 固定の文 |

注目したい書式のテクニック:
- `{cfg.max_donate_share:.0%}` — 0.4 を `40%` と表示する。
- `{', '.join(empty_patterns) or 'none'}` — IDをつなぐ。リストが空なら `none`(Day 4と同じ技)。
- `{dict(style_counts['val'])}` — `Counter` を普通の辞書にして表示する。
- `".venv\\Scripts\\python.exe ..."` — 普通の(rawでない)文字列では、バックスラッシュを `\` 1つとして出すために `\\` と2つ書く。
- 一部の行は `f"` で始まるのに `{}` が無い。害はない。

```python
    for s in SPLITS:
        st = stats[s]
        lines.append(
            f"| {s} | {fmt(st['lines'])} | ... |"
        )
```
- **Size** の表に、山ごとに1行を追加する。数の整形には `fmt` を使う。

```python
    lines += [ ... ]
```
- 残りの文章でリストを続ける:トークンの範囲の説明、タイプごとの表(`*("| " + " | ".join(r) + " |" for r in per_type_rows)` — `*` が展開し、行の文字列を ` | ` でつないだMarkdownの行をタイプごとに作る)、カバー率の表(`*(f"| {s} | ..." for s in SPLITS)`)、限界、再現のコマンド。

```python
    path.write_text("\n".join(lines), encoding="utf-8")
```
- 全部の行を改行でつないで、ファイルとして保存する。

### 9.5 `append_run_log`

```python
def append_run_log(log, machine, stats, manifest_rel, cfg, n_val, n_test) -> None:
    entry = [
        "",
        f"## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 5 corpus split",
        "",
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Script: scripts/day5_build_corpus.py ({SCRIPT_VERSION}), seed {cfg.seed}",
        f"- Held-out patterns: validation {n_val}, test {n_test} (split by pattern, not by line)",
    ]
    for s in SPLITS:
        st = stats[s]
        entry.append(
            f"- {s}: {st['lines']:,} lines ({st['unique_lines']:,} unique), {st['characters']:,} chars, "
            f"token bounds {st['token_lower_bound']:,} to {st['utf8_bytes']:,}"
        )
    entry += [f"- Manifest: {manifest_rel}", "- Result: PASS", ""]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))
```
- Day 1・Day 4と同じ形:Markdownの行のリストを作り、`run_log.md` に追記する。
- ループは、山ごとに、数字とトークンの範囲(下限 `to` 上限)を1行追加する。
- `"- Result: PASS"` は常に書かれる。失敗していれば、ここに来る前に `SystemExit` で止まっているため。

---

## 10. メイン部分

### 10.1 `parse_args`

```python
def parse_args() -> argparse.Namespace:
    d = SplitConfig()
    ap = argparse.ArgumentParser(description="Day 5: split the question corpus by pattern and write the data card.")
```
- `d = SplitConfig()` — **初期値を読むためだけ**に設定オブジェクトを作る。初期値は(`SplitConfig` に)一度だけ書き、コマンドラインの初期値としても使い回す。重複がない。

```python
    ap.add_argument("--machine", choices=["school", "home"], required=True, help="label for run_log.md")
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "data/source/questions_corpus.zip", help="questions_corpus.zip or questions.csv")
    ap.add_argument("--patterns", type=Path, default=REPO_ROOT / "data/source/patterns.csv")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/corpus")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/splits/split_manifest.json")
    ap.add_argument("--data-card", type=Path, default=REPO_ROOT / "docs/DATA_CARD.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--no-log", action="store_true", help="do not append to run_log.md (for dry runs)")
```
- Day 1・Day 4と同じ `argparse` の考え方。`--machine` は必須。
- `Path` のオプション(`--source`、`--patterns`、`--glossary`、`--synonyms`、`--out-dir`、`--manifest`、`--data-card`、`--run-log`)は、どれも同じ働き:初期の場所を持つパス。違うのは名前と初期の場所だけ。
- `--no-log` — フラグ:付ければ `run_log.md` に追記しない。helpにあるとおり**試し実行(dry run)**用。

```python
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--val-fraction", type=float, default=d.val_fraction)
    ap.add_argument("--test-fraction", type=float, default=d.test_fraction)
    ap.add_argument("--min-type-lines", type=int, default=d.min_type_lines)
    ap.add_argument("--min-pattern-lines", type=int, default=d.min_pattern_lines)
    ap.add_argument("--max-donate-share", type=float, default=d.max_donate_share)
    ap.add_argument("--eval-cap", type=int, default=d.eval_cap_per_pattern)
    ap.add_argument("--oversample-target", type=int, default=d.oversample_target)
    ap.add_argument("--max-repeat", type=int, default=d.max_repeat)
    return ap.parse_args()
```
- `SplitConfig` のつまみ1つにつき、オプションが1つ。9行とも同じ考え方:`type=int` か `type=float` で、初期値は `d` から取る。
- `ap.parse_args()` が、入力された内容を読み取って返す。

### 10.2 `main` — 準備と読み込み

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
```
- Day 4と同じ。コメントが理由を説明している:日本語のエラーメッセージで、UTF-8でないコンソール(日本語Windowsの `cp932` など)が落ちてはいけない。

```python
    a = parse_args()
    for name in ("source", "patterns", "glossary", "synonyms", "out_dir", "manifest", "data_card", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
```
- Day 4と同じ `setattr`/`getattr` のループ。今回は8つのパスのオプション:それぞれを絶対パスにして、コマンドラインで相対パスを入力しても動くようにする。

```python
    cfg = SplitConfig(a.seed, a.val_fraction, a.test_fraction, a.min_type_lines, a.min_pattern_lines,
                      a.max_donate_share, a.eval_cap, a.oversample_target, a.max_repeat)
```
- コマンドラインの値から、本番の設定オブジェクトを作る。値は `SplitConfig` の項目と**同じ順番**で渡している。

```python
    for p in (a.source, a.patterns):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")
```
- コーパスと `patterns.csv` は**必須**。どちらかが無ければ止める。(用語集と同義語辞書は任意。)

```python
    print(f"=== Day 5 corpus split | machine={a.machine} | seed={cfg.seed} ===")
    questions = load_questions(a.source)
    patterns = load_patterns(a.patterns)
```
- 見出しを表示し、質問を読み込んで検査し、文型の表を読み込む。

```python
    unknown = sorted({q.pattern_id for q in questions} - patterns.keys())
    if unknown:
        raise SystemExit(f"Pattern IDs not in patterns.csv: {unknown[:10]}")
```
- 集合の引き算:質問で使われているパターンID **から** `patterns.csv` にあるIDを引く。残ったものは未知のID → 止める(最大10個を表示)。

```python
    wrong_type = sorted({q.pattern_id for q in questions if patterns[q.pattern_id].type_id != q.type_id})
    if wrong_type:
        raise SystemExit(f"Type in corpus differs from patterns.csv for: {wrong_type[:10]}")
```
- 突き合わせ:各質問について、コーパスのタイプが `patterns.csv` のタイプと等しいこと。違えば止める。

```python
    used = {q.pattern_id for q in questions}
    empty_patterns = sorted(set(patterns) - used)
    print(f"Loaded {len(questions):,} questions, {len(used)} patterns with questions, "
          f"{len({q.type_id for q in questions})} types.")
    if empty_patterns:
        print(f"Patterns with no questions (all their sentences already came from another pattern): {empty_patterns}")
```
- `used` — 質問がある文型。`empty_patterns` — `patterns.csv` にはあるが質問が無い文型(その文が全部、先の文型の重複だった)。
- 要約を表示し、空の文型があれば注意を表示する。

```python
    assignment = choose_held_out_patterns(questions, patterns, cfg)
    result = build_splits(questions, assignment, cfg)
```
- **2つの要の処理:**どの文型を取り分けるかを選び、山を作る。

### 10.3 `main` — 出力の書き出し

```python
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.data_card.parent.mkdir(parents=True, exist_ok=True)
```
- 3つの出力先フォルダーがあることを確認する(これまでと同じ `mkdir`。マニフェストとデータカードは、親フォルダーが必要)。

```python
    outputs: dict[str, str] = {}
    for s in SPLITS:
        txt = a.out_dir / f"{s}.txt"
        lab = a.out_dir / f"{s}_labels.csv"
        write_lines(txt, result.lines[s])
        write_labels(lab, result.lines[s], s)
        outputs[str(txt.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_of(txt)
        outputs[str(lab.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_of(lab)
```
- 山ごとに、`train.txt` / `train_labels.csv`(val、test も同様)を書く。
- 各ファイルの指紋を `outputs` に記録する。キーは、`\` ではなく `/` を使った、リポジトリからの相対パス。Windowsでもそれ以外でも同じ見た目になる(Day 4の `work_rel` と同じ技)。

```python
    stats = {s: split_stats(result.lines[s]) for s in SPLITS}
    stats["train_unique"] = split_stats(result.train_unique)
```
- 山ごとの統計に加えて、train を繰り返す**前**の分の項目を追加。

```python
    tr_u = Counter(q.type_id for q in result.train_unique)
    tr = Counter(q.type_id for q in result.lines["train"])
    va = Counter(q.type_id for q in result.lines["val"])
    te = Counter(q.type_id for q in result.lines["test"])
    type_names = {p.type_id: p.type_name for p in patterns.values()}
    per_type_rows = [
        [t, type_names.get(t, ""), str(tr_u[t]), str(result.repeat_factor.get(t, 1)), str(tr[t]), str(va[t]), str(te[t])]
        for t in sorted({q.type_id for q in questions})
    ]
```
- 4つの `Counter` が、タイプごとの質問数を数える:train-unique、train(繰り返し後)、val、test。(4行は同じ考え方。)`Counter` は、無いキーに0を返すので、そのタイプに val の質問が無くても `va[t]` は安全。
- `type_names` — タイプID → 読みやすい名前。
- `per_type_rows` — データカード用の行のリスト。各行は**文字列**のリスト:タイプID、名前、train unique、コピー数、train の行数、val、test。

```python
    terms, variants = load_terms(a.glossary, a.synonyms)
    cov = {
        s: {"terms": coverage((q.text for q in result.lines[s]), terms),
            "variants": coverage((q.text for q in result.lines[s]), variants)}
        for s in SPLITS
    }
    if not terms:
        print("NOTE: glossary/synonym files not found; the coverage section will show 0.")
```
- 用語集と同義語のリストを読み込む。
- `cov` — 山ごとに、用語集の用語と変形がいくつ現れるか。入れ子の辞書内包表記で、`(q.text for q in ...)` は `coverage` に渡すジェネレーター。
- 用語が1つも読み込めなければ、注意を表示する。

```python
    held = {s: sorted(p for p, sp in assignment.items() if sp == s) for s in SPLITS}
```
- 山ごと(train も含む)の、パターンIDの並べ替えたリスト。

```python
    manifest = {
        "script": "scripts/day5_build_corpus.py",
        "script_version": SCRIPT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "machine": a.machine,
        "python": sys.version.split()[0],
        "config": cfg.__dict__,
        "inputs": {
            str(a.source.name): sha256_of(a.source),
            str(a.patterns.name): sha256_of(a.patterns),
        },
        "outputs_sha256": outputs,
        "patterns_per_split": held,
        "patterns_without_questions": empty_patterns,
        "repeat_factor_per_type": dict(sorted(result.repeat_factor.items())),
        "dropped_near_duplicates": result.dropped_near_dup,
        "dropped_by_eval_cap": result.dropped_by_cap,
        "stats": stats,
        "coverage": {"glossary_terms": len(terms), "synonym_variants": len(variants), **cov},
    }
```
- **マニフェスト**:この実行の「レシート」。中身:どのスクリプトとバージョンか、いつ(`isoformat(timespec="seconds")` → `2026-11-08T14:05:33`)、どのマシンか、Pythonのバージョン(`sys.version.split()[0]` で番号だけ)、全設定(`cfg.__dict__` = データクラスの項目を辞書にしたもの)、**入力**と**出力**の指紋、どの文型がどこに入ったか、繰り返しの倍率、除いたもの、統計、カバー率。
- `**cov` は、`cov` 辞書を作成中の辞書に展開して合体させる。
- 注意:`created`、`machine`、`python` はマシンによって違うが、`outputs_sha256` は同じになるはず。だから、ファイル全体ではなく `outputs_sha256` を比べる。

```python
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_data_card(a.data_card, stats, per_type_rows, result, patterns, empty_patterns, cov,
                    len(terms), len(variants), cfg, a.source.name, manifest["inputs"][a.source.name])
```
- マニフェストを読みやすいJSONで保存する(Day 1・Day 4と同じ)。次にデータカードを書く。入力ファイルの指紋はマニフェストから渡す。

### 10.4 `main` — 安全確認、要約、ログ

```python
    train_keys = {near_dup_key(q.text) for q in result.train_unique}
    leak = sum(1 for s in ("val", "test") for q in result.lines[s] if near_dup_key(q.text) in train_keys)
    overlap = set(held["val"]) & set(held["test"]) | (set(held["val"]) | set(held["test"])) & set(held["train"])
    if leak or overlap:
        raise SystemExit(f"Split check FAILED: {leak} leaked lines, {len(overlap)} shared patterns.")
```
- **書き出した結果の最終検査。**
- `train_keys` — train の言い回しの指紋。
- `leak` — 指紋が train にもある val/test の行を数える。0であるべき。
- `overlap` — 複数の山に現れる文型。演算子 `&`(共通部分)は `|`(和集合)より先に計算されるので、意味は **(val ∩ test) ∪ ((val ∪ test) ∩ train)**。空であるべき。
- どちらかが0でなければ、エラーで止める(「分割が漏れていたら、大きな音で失敗する」)。

```python
    print()
    print(f"{'split':<6} {'lines':>9} {'unique':>9} {'chars':>11} {'token bounds':>25} {'patterns':>9} {'types':>6}")
    for s in SPLITS:
        st = stats[s]
        bounds = f"{st['token_lower_bound']:,} - {st['utf8_bytes']:,}"
        print(f"{s:<6} {st['lines']:>9,} {st['unique_lines']:>9,} {st['characters']:>11,} {bounds:>25} "
              f"{st['patterns']:>9} {st['types']:>6}")
```
- コンソールにきれいな表を表示する。f文字列の `<6` は「幅6で左寄せ」、`>9` は「幅9で右寄せ」、`,` は3桁区切りのカンマ。`bounds` は「下限 - 上限」のトークン範囲。

```python
    print(f"Near-duplicates removed: {result.dropped_near_dup}; over eval cap: {result.dropped_by_cap}")
    oversampled = {t: f for t, f in sorted(result.repeat_factor.items()) if f > 1}
    print(f"Oversampled types (copies): {oversampled}")
    print(f"Coverage {cov}")
    print("Split check: PASS (no shared patterns, no near-duplicate leaks)")
    print(f"Wrote {a.out_dir}, {a.manifest}, {a.data_card}")
```
- 除いたもの、オーバーサンプリングされたタイプ(`if f > 1` で、コピーが2つ以上のものだけ)、カバー率、PASSのメッセージ(確認に合格したときだけ到達)、書き出し先を表示する。

```python
    if not a.no_log:
        append_run_log(a.run_log, a.machine, stats, str(a.manifest.relative_to(REPO_ROOT)).replace("\\", "/"),
                       cfg, len(held["val"]), len(held["test"]))
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- Day 4と同じ終わり方:`--no-log` が無ければ `run_log.md` に追記し、`0` を返す。ファイルを直接実行したときだけ `main()` を動かし、その結果を終了コードにする。

---

## 11. 実行の流れ(順番)

1. オプションを読む → 絶対パスにする → `SplitConfig` を作る。
2. 必須の入力2つがあるか確認する。
3. `load_questions`(検査)+ `load_patterns` → IDとタイプを突き合わせる。
4. `choose_held_out_patterns` → `assignment`。
5. `build_splits` → ほぼ重複のガード → 上限 → train のオーバーサンプリング → シャッフル。
6. `train/val/test` の `.txt` と `_labels.csv` を書き、指紋を取る。
7. 統計、タイプごとの表、カバー率。
8. `split_manifest.json` と `docs/DATA_CARD.md` を書く。
9. 漏れの確認 → コンソールの表 → `run_log.md` に追記。

## 12. 結果の読み方

- **`Split check: PASS`** — どの文型も2つの山に現れず、val/test の文で train の文と同じ言い回しのものは無い。
- **取り分けた文型の数** — val と test それぞれ、文型のおよそ7%(対象タイプが少なければそれより少なく。WARNINGに注意)。
- **train の `lines` と `unique`** — オーバーサンプリングのため `lines` のほうが大きい。`unique` が、本当に異なる質問の数。
- **トークンの範囲** — 本当のトークン数は、下限の数字とUTF-8のバイト数の間のどこか。Day 7で測る。
- **カバー率が100%未満** — 正常。用語集の言葉の多くは、質問には使われない。
- **マシンの比較** — 学校と自宅で、マニフェストの `outputs_sha256` を比べる。同じ値 = 1バイトも違わない出力。

## 13. 知っておくと良い小さなこと

- **Day 4からの良い改善:** `load_questions` は悪い行を `continue` で飛ばすので、列の数が違う行は、落ちずにきれいに報告される。
- **漏れの確認は書き出しの後。** 確認が失敗しても、ファイルは(マニフェストとデータカードも)すでにディスク上にある。無効なものとして扱い、原因を直してから使うこと。
- **`relative_to(REPO_ROOT)`**(Day 4と同じ注意):`--out-dir` や `--manifest` をリポジトリの外にすると、この呼び出しがエラーになる。
- **有効なJSONだがオブジェクトでないラベル**(例:`123` やリスト)があると、`label.get(...)` が親切なメッセージではなくPythonの `AttributeError` を出す。チームの生成プログラムがこれを作ることは無いはず。
- **データカードの固定の文。** 数字は実行結果から来るが、「at least 3 phrasings」や「at least 2 phrasings in train」のような文は、今のコードのルールに合わせて手で書いたテキスト。コードのルールを変えたら、カードの文も直すこと。
- **丸め:** Pythonの `round` は、ちょうど半分の値を最も近い偶数に丸める(例:`round(2.5)` は `2`)ので、取り分ける数が、手計算の予想と1つずれることがある。それでも再現はできる。

## 14. 一段落でまとめ

このスクリプトは、チームの質問を読み込んで検査し、選んだ質問タイプごとに**文型を丸ごと**検証用・テスト用に取り分けて、モデルが一度も見たことのない言い回しで採点されるようにし、ほぼ同じ文を除き、大きな取り分け文型に上限を付け、小さな質問タイプを学習用の中だけで数回繰り返し、質問の文だけのテキストファイルと別のラベルファイルを書き出し、すべて(設定、指紋、統計、カバー率)をマニフェスト、データカード、実行ログに記録して、最後に山の間で何も漏れていないことを確認します。
