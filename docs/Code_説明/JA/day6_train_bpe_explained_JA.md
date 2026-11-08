# Day 6 — day6_train_bpe.py — やさしい1行ずつの解説(日本語)

> **Day 6 の3つのファイルの関係**
> - **Day 6.1 `bpe_tokenizer.py`** — *エンジン*(トークナイザー本体)。
> - **Day 6(このファイル)`day6_train_bpe.py`** — *運転手*:エンジンを学習させ、いくつかの大きさを試し、決まったルールで1つ選び、保存し、検証し、報告する。
> - **Day 6.2 `test_bpe_tokenizer.py`** — *検査員*(自動テスト)。
>
> **先に Day 6.1 を読んでください。**このスクリプトは、その関数(`train_bpe`、`count_pieces`、`pretokenize`、`ByteBPETokenizer`)を何度も使います。

---

## 0. 全体像(まずここから)

Day 6の大きな問い:**トークナイザーの語彙は、どれくらい大きくすべきか?**

- **小さすぎる** → 単語が細かい断片にたくさん切られる → トークン列が長くなり、遅く、モデルも学びにくい。
- **大きすぎる** → とてもまれなトークンが増え、モデルがほとんど学べない上に、モデルの埋め込み表が、利益なく大きくなる。

このスクリプトは、勘で決める代わりに**測って**、**書き下したルール**で選びます。賢いコツ(Day 6.1より):結合は順番に学ぶので、**最初の N 個の結合が、大きさ N のトークナイザー**です。だから1回の学習で、*すべて*のより小さい大きさがただで手に入ります。

| 手順 | 何が起こるか |
|---|---|
| 1 | `train.txt`(学習用)と `val.txt`(測るためだけ)を読む |
| 2 | 1回学習して、文章が許すすべての結合を学ぶ → 「フル」のトークナイザー |
| 3 | それを多くの候補の大きさ(400、500、…、16000)に切り詰めて、それぞれを**測る** |
| 4 | ルール `choose_size()` を当てはめる → 条件を満たす一番小さい大きさを選ぶ |
| 5 | 選んだトークナイザーとフルのものを保存 |
| 6 | 読み込み直して**検証**(同じ動き、ロスレスな往復) |
| 7 | デモを表示し、報告書と実行ログを書く |

実行:

```
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home
```

あとで、**学習し直さずに**別の大きさを保存するには:

```
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home --resave-size 4000
```

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **語彙サイズ** | トークナイザーが持つ、異なるトークンの数。 |
| **圧縮(compression)** | 同じ文を表すのに必要なトークンが、どれだけ少ないか。少ないほど良い圧縮。 |
| **1文字あたりのトークン数** | `トークン数 ÷ 文字数`。低いほど良い圧縮。 |
| **重要用語(key terms)** | 用語集の、列名・値・検索語(列名、値、検索語)のような大事な言葉。どれも1〜3トークンにしたい。 |
| **達成可能な重要用語** | 学習が1〜3トークンに*できる*重要用語(セクション6.3を参照)。 |
| **文脈の長さ(context length)** | モデルが一度に読む最大のトークン数。一番長い質問(トークン数)が、決める材料になる。 |
| **許容度(tolerance)** | 一番良い(最大の)トークナイザーより、どれだけ悪くても許すか(初期値5%)。 |
| **切り詰め(truncate)** | 最初の N 個の結合だけを残すこと(Day 6.1の `truncated`)。 |
| **取り分けた検証用** | 学習に使わない文(Day 5)。ここでは*測る*ためだけに使う。 |
| **往復(round trip)** | `decode(encode(text)) == text`。 |

---

## 2. ファイル先頭のdocstring(説明文)

- これまでと同じ形式。5つの手順と、**大きさを選ぶルール**を載せている:
  > **1〜3トークンになり得る重要用語がすべて1〜3トークン**で、かつ**検証用のトークン数が、一番大きいトークナイザーより `--tolerance` 以上は増えない**、一番小さい候補。
- 大事な点:検証用とテスト用の文は学習に**使わない**。検証用は、候補を測るためだけに使う。
- 学校PCの注意:コードは同じでCPUのみ。学校PCでも**同じ `tokenizer.json` のSHA-256**にならなければならない。

---

## 3. importと `sys.path` の行

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
```
- 新しい `platform`(Pythonのバージョンを返す。例:`platform.python_version()`)以外は、すべて **Day 4・Day 5と同じ**。

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- `REPO_ROOT` — 前と同じ。
- `sys.path` は、`import` するときにPythonが探すフォルダーのリスト。`insert(0, ...)` はリポジトリのルートを**先頭に**入れる。これで、このスクリプトが `scripts/` にあっても、`from src.bpe_tokenizer import ...` が動く。

```python
from src.bpe_tokenizer import (  # noqa: E402
    FIRST_MERGE_ID,
    ByteBPETokenizer,
    TrainingLog,
    count_pieces,
    pretokenize,
    train_bpe,
)
```
- Day 6.1のエンジンを読み込む。`# noqa: E402` は、スタイルチェッカーへの「このimportがファイルの先頭にないのは承知の上。先に `sys.path` を設定する必要があったから」という伝言。

---

## 4. 定数

```python
SCRIPT_VERSION: str = "day6-v1"
KEY_DB_GROUPS: frozenset[str] = frozenset({"列名", "値", "検索語"})
```
- `SCRIPT_VERSION` — 前と同じ。
- `KEY_DB_GROUPS` — 用語集の `DB区分`(カテゴリ)のうち、**重要**とみなすもの:列名、値、検索語。`frozenset` は変更できない集合。

```python
CANDIDATE_SIZES: tuple[int, ...] = (400, 500, 600, 800, 1000, 1200, 1400, 1500, 1600, 2000, 4000, 8000, 16000)
```
- 試す語彙サイズ(小さい順 — `choose_size` はこの順番に頼っている)。

```python
DEMO_SENTENCES: tuple[str, ...] = (
    "今日は晴れ",
    "営業部の社員の平均年収を教えてください。",
    "ボーナスが100万円以上の人は何人ですか。",
    "大阪支社で勤続年数が一番長い人は誰？",
    "Model A tokenizer test 2026",
)
```
- 最後に表示するサンプルの文。選んだトークナイザーが実際の文をどう切るかを*見る*ため(数字を含む英語の文も入っている)。

---

## 5. `SizeResult` データクラス

```python
@dataclass
class SizeResult:
    """Measurements for one candidate vocabulary size."""

    vocab_size: int
    val_tokens: int
    tokens_per_char: float
    key_ok: int          # achievable key terms that are 1-3 tokens
    key_achievable: int  # key terms that training can make 1-3 tokens (see achievable_terms())
    max_val_tokens: int
    max_train_tokens: int
```
- 候補の大きさ1つについての「結果の1行」:
  - `vocab_size` — 測った大きさ。
  - `val_tokens` — 検証用の全文に必要なトークンの合計(少ないほど良い)。
  - `tokens_per_char` — それを文字数で割ったもの。
  - `key_ok` — この大きさで3トークン以下になる、達成可能な重要用語の数。
  - `key_achievable` — そうなり*得る*重要用語の数(目標)。
  - `max_val_tokens`、`max_train_tokens` — 一番長い質問のトークン数(検証用と、train の抜き取り)。

---

## 6. ヘルパー関数

### 6.1 `sha256_file` と `read_lines`

```python
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
```
- `sha256_file` — Day 4と同じ。
- `read_lines` — テキストファイルを読み、改行で分け、空でない行だけを残す(`if ln`)。

### 6.2 `load_key_terms`

```python
def load_key_terms(glossary: Path, synonyms: Path, train_blob: str) -> list[str]:
    """Key terms (列名, 値, 検索語 and their variants) that actually occur in the training text."""
    terms: set[str] = set()
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            terms |= {r["標準用語"].strip() for r in csv.DictReader(f) if r["DB区分"].strip() in KEY_DB_GROUPS}
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            terms |= {r["変形・同義語"].strip() for r in csv.DictReader(f) if r["DB区分"].strip() in KEY_DB_GROUPS}
    return sorted(t for t in terms if t and t in train_blob)
```
- 空の集合から始める。
- 用語集(あれば)から:`DB区分` が重要グループのいずれかである行の `標準用語` を取る。`terms |= {...}` は、新しい集合を `terms` に**足す**(集合の和)。
- 同義語辞書から:同じことを、`変形・同義語`(変形)の列で行う。
- 空でなく、**かつ学習用の文に実際に現れる**(`t in train_blob`)用語を、並べ替えて返す。学習で一度も見ない用語は学べないので、測っても意味がない。

### 6.3 `achievable_terms`

```python
def achievable_terms(key_terms: list[str], piece_counts: dict[str, int], min_frequency: int) -> list[str]:
    out: list[str] = []
    for t in key_terms:
        parts = pretokenize(t)
        if len(parts) <= 3 and all(piece_counts.get(p, 0) >= min_frequency for p in parts):
            out.append(t)
    return out
```
- どの重要用語が、現実的に1〜3トークンに*なれる*か?
- `parts = pretokenize(t)` — 用語をプリトークナイザーの断片に切る(Day 6.1)。
- 次の**両方**が成り立てば、達成可能:
  1. 断片が**最大3つ**で、かつ
  2. **どの断片も、学習用の文の中で単独で**、少なくとも `min_frequency` 回出てくる(`piece_counts.get(p, 0)`。`all(...)` =「全部の断片で真」)。
- なぜか? BPEは断片の境界をまたいで結合できないので、断片ごとに最低1トークンが必要 → 断片3つ以下なら、トークンも3つ以下になれる。そして断片が**1つの**トークンになれるのは、断片全体として十分な回数出てくるときだけ。docstringの例:`勤続` は `勤続年数`(1つの漢字の連続)の*中*にしか現れないので、`勤続` 単独は学習用の文の中で断片では**ない** → 単独のトークンにはなれない → 達成不可能。これでルールが公平になる:不可能なことを要求しない。

### 6.4 `measure`

```python
def measure(tok, val, train_sample, key_terms, achievable) -> SizeResult:
    val_lengths = [len(tok.encode(ln)) for ln in val]
    key_ok = sum(1 for t in achievable if len(tok.encode(t)) <= 3)
    chars = sum(len(ln) for ln in val)
    return SizeResult(
        vocab_size=tok.vocab_size,
        val_tokens=sum(val_lengths),
        tokens_per_char=sum(val_lengths) / chars,
        key_ok=key_ok,
        key_achievable=len(achievable),
        max_val_tokens=max(val_lengths),
        max_train_tokens=max(len(tok.encode(ln)) for ln in train_sample),
    )
```
- `val_lengths` — 検証用の各行のトークン数。
- `key_ok` — 3トークン以下で符号化される、達成可能な重要用語の数。
- `chars` — 検証用の文の総文字数。
- `SizeResult` を返す:大きさ、検証用の総トークン数、1文字あたりのトークン数、重要用語の数、一番長い質問(検証用と train の抜き取り)。
- (注:`key_terms` 引数は渡されるが、中では使われない。使うのは `achievable` だけ。害はない。)

### 6.5 `choose_size` — 書き下したルール

```python
def choose_size(results: list[SizeResult], tolerance: float) -> SizeResult:
    best_val = results[-1].val_tokens
    for r in results:
        if r.key_ok == r.key_achievable and r.val_tokens <= (1 + tolerance) * best_val:
            return r
    return results[-1]
```
- `results[-1]` は**一番大きい**トークナイザー(結果は小さい順)なので、`best_val` = 得られる最少の検証用トークン数。
- 一番小さい大きさから上へ見ていき、次の条件を満たす**最初の**ものを返す:
  - `key_ok == key_achievable`(達成可能な重要用語がすべて1〜3トークン)、**かつ**
  - `val_tokens <= (1 + tolerance) * best_val`(最良の圧縮の5%以内)。
- どれも満たさなければ、一番大きいものに戻る。
- このルールは**コードに書き下され**、報告書にも載るので、選択は「根拠を満たす一番小さいもの」と説明できる。感覚ではない。

### 6.6 `roundtrip_failures`

```python
def roundtrip_failures(tok: ByteBPETokenizer, lines: list[str]) -> int:
    return sum(1 for ln in lines if tok.decode(tok.encode(ln)) != ln)
```
- 符号化→復号で元に戻**らない**行を数える。**0**であるべき。

### 6.7 `write_report`

```python
def write_report(path, log, results, chosen, tolerance, min_frequency, seconds, demos,
                 key_terms, achievable, tok, hashes, n_train, n_val, new_val_pieces) -> None:
    long_terms = [t for t in achievable if len(tok.encode(t)) > 3]
    short_all = sum(1 for t in key_terms if len(tok.encode(t)) <= 3)
    long_other = [f"{t} ({len(tok.encode(t))})" for t in key_terms if t not in achievable and len(tok.encode(t)) > 3]
```
- `docs/DAY6_TOKENIZER_REPORT.md` を書く。まず3つのリストを準備する:
  - `long_terms` — 選んだ大きさでも**3トークンを超える**、達成可能な重要用語(無いはず)。
  - `short_all` — **すべての**重要用語のうち、3トークン以下の数。
  - `long_other` — 達成不可能で3トークンを超える重要用語。`勤続 (2)` のように、トークン数付きで書く。

Markdownの `lines` リストには、次のセクションが入る:

| セクション | 内容 |
|---|---|
| タイトルと日付 | バージョンと時刻 |
| **Training** | 学習用・検証用の行数。`log` があれば(`--resave-size` では無い):断片の総数と異なる数、結合の上限、**学んだ結合の数**、**最大の語彙サイズ**(`FIRST_MERGE_ID + merges_learned`)、止まった理由、時間 |
| **Size candidates** | 大きさごとに1行の表:語彙、検証用トークン数、1文字あたりのトークン数、1〜3トークンの重要用語(`ok/achievable`)、検証用で一番長い質問、train で一番長い質問。選んだ行には `**(chosen)**` の印が付く(`r is chosen` は**同じオブジェクト**かを比べる) |
| **Rule** | ルールを言葉で、許容度をパーセント(`{tolerance:.0%}`)で、そして選んだ大きさ |
| **重要用語の説明** | 重要用語の数、達成可能な数、長すぎるもの、1〜3トークンにできないものとその理由 |
| **New validation pieces** | 検証用にあって、**学習では一度も出ない**断片(それでも、より小さなトークンで符号化できる) |
| **Demo** | デモの文と、その切り方の表(`' / '.join(p)`) |
| **Files (SHA-256)** | トークナイザーのファイルと学習の入力の指紋 |

- `*( ... for r in results)` は、大きさごとに表の1行を展開する(Day 4・5と同じ技)。
- `{' **(chosen)**' if r is chosen else ''}` — 選んだ行にだけ印を付ける。
- 最後は `path.write_text("\n".join(lines), encoding="utf-8")` — 前と同じ。

### 6.8 `append_run_log`

```python
def append_run_log(log_path: Path, machine: str, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 6 tokenizer\n\n{text}\n")
```
- Day 4・5より単純:呼び出し側が本文を作り、これは見出しを付けて `run_log.md` に追記するだけ(追記モード `"a"`)。

---

## 7. `parse_args`

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 6: train the byte-level BPE tokenizer and choose its size.")
    ap.add_argument("--machine", choices=["school", "home"], required=True)
    ap.add_argument("--train", type=Path, default=REPO_ROOT / "data/corpus/train.txt")
    ap.add_argument("--val", type=Path, default=REPO_ROOT / "data/corpus/val.txt")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/tokenizer")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY6_TOKENIZER_REPORT.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--max-merges", type=int, default=100_000, help="safety cap; training normally stops earlier")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--resave-size", type=int, default=None,
                    help="skip training: cut tokenizer_full.json to this size and save it as tokenizer.json")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()
```
- これまでと同じ `argparse` の考え方。`--machine` は必須。7つの `Path` オプションはどれも同じ働き(名前 + 初期の場所)。
- `--min-frequency` — 組が結合されるために最低この回数出る必要がある(初期値2)。
- `--max-merges` — 安全上限。`100_000` は `100000` と同じ(アンダースコアは大きな数を読みやすくする)。
- `--tolerance` — 0.05 = 5%。
- `--resave-size` — 初期値は `None`(= 通常の学習)。指定すると、**学習を飛ばして**、保存済みのフルのトークナイザーをその大きさに切り直す。
- `--no-log` — Day 4・5と同じフラグ。

---

## 8. `main`

### 8.1 準備

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("train", "val", "glossary", "synonyms", "out_dir", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    for p in (a.train, a.val):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run Day 5 first.")
```
- コンソールと `resolve()` は、Day 4・5と同じ形。2つの入力ファイル(Day 5から来る)が必要で、無ければ止める。

```python
    print(f"=== Day 6 tokenizer | machine={a.machine} | Python {platform.python_version()} ===")
    train = read_lines(a.train)
    val = read_lines(a.val)
    full_path = a.out_dir / "tokenizer_full.json"
    chosen_path = a.out_dir / "tokenizer.json"
```
- 見出し(Pythonのバージョン付き)を表示し、2つのテキストファイルを読み、2つの出力ファイルのパスを決める:**フル**のトークナイザーと、**選んだ大きさ**のトークナイザー。

### 8.2 学習する(または読み込む)

```python
    log: TrainingLog | None = None
    seconds = 0.0
    if a.resave_size is None:
        start = time.perf_counter()
        merges, log = train_bpe(count_pieces(train), a.max_merges, a.min_frequency)
        seconds = time.perf_counter() - start
        full = ByteBPETokenizer(merges)
        full.save(full_path)
        print(f"Unique pre-token pieces: {log.unique_pieces:,} (total {log.total_pieces:,})")
        print(f"Merges learned: {log.merges_learned:,} -> largest vocabulary {full.vocab_size:,} "
              f"({log.stop_reason}); {seconds:.1f} s")
    else:
        if not full_path.exists():
            raise SystemExit(f"--resave-size needs {full_path}. Run without it first.")
        full = ByteBPETokenizer.load(full_path)
        print(f"Loaded {full_path.name}: largest vocabulary {full.vocab_size:,}")
```
- `log` と `seconds` は空から始める(`--resave-size` の道には学習ログがない)。
- **通常モード:**ストップウォッチを始める。`count_pieces(train)` が断片を数え、`train_bpe` が結合を(自分で止まるまで全部)学ぶ。ストップウォッチを止める。**フル**のトークナイザーを作って保存する。断片の数と、止まった理由付きで最大の語彙サイズを表示する。
- **再保存モード:**フルのファイルが既に必要(無ければ分かりやすいメッセージで止める)。それを読み込む。

### 8.3 測定の準備

```python
    train_blob = "\n".join(train)
    piece_counts = count_pieces(train)
    key_terms = load_key_terms(a.glossary, a.synonyms, train_blob)
    achievable = achievable_terms(key_terms, piece_counts, a.min_frequency)
    new_val_pieces = sorted({p for ln in val for p in pretokenize(ln)} - set(piece_counts))
    train_sample = train[::10]  # every 10th line: enough to find the longest questions quickly
```
- `train_blob` — 学習用の全文を1つの文字列にしたもの(`t in train_blob` のため)。
- `piece_counts` — 断片の数をもう一度(再保存モードでも必要。通常モードでは数え直しだが、害はない)。
- `key_terms` と `achievable` — ルールの目標。
- `new_val_pieces` — 集合の引き算:検証用の断片 **引く** 学習で見た断片 = 学習で一度も見ていない断片。
- `train[::10]` — 刻み付きのスライス:10行おきに1行。「train で一番長い質問」の列にだけ使う(注意のセクションを参照)。

```python
    sizes = [s for s in CANDIDATE_SIZES if s < full.vocab_size] + [full.vocab_size]
    if a.resave_size is not None and a.resave_size not in sizes:
        sizes = sorted(set(sizes) | {min(a.resave_size, full.vocab_size)})
```
- フルの大きさより小さい候補の大きさに、**フルの大きさ自身**を足す(最大のものが必ず測られるように)。
- 再保存モードで、指定した大きさがリストに無ければ、足す(フルの大きさを超えない)。

### 8.4 大きさごとに測る

```python
    print(f"Key terms found in the training text: {len(key_terms)}; achievable as 1-3 tokens: {len(achievable)}")
    print(f"\n{'vocab':>7} {'val tokens':>11} {'tok/char':>9} {'key terms 1-3':>14} {'max val':>8} {'max train':>10}")
    results: list[SizeResult] = []
    for s in sizes:
        r = measure(full.truncated(s), val, train_sample, key_terms, achievable)
        results.append(r)
        print(f"{r.vocab_size:>7,} {r.val_tokens:>11,} {r.tokens_per_char:>9.3f} "
              f"{f'{r.key_ok}/{r.key_achievable}':>14} {r.max_val_tokens:>8} {r.max_train_tokens:>10}")
```
- 数と表の見出しを表示する(`>7` などの書式は、Day 5と同じ)。
- 各大きさについて:**`full.truncated(s)`** が大きさ `s` のトークナイザーをくれる。`measure` がそれを評価し、結果を保存して1行表示する。入れ子のf文字列は `35/38` のような文字を作って右寄せにする。

### 8.5 選ぶ

```python
    if a.resave_size is None:
        chosen = choose_size(results, a.tolerance)
        print(f"\nRule: smallest size with all achievable key terms at 1-3 tokens and validation within "
              f"{a.tolerance:.0%} of the largest -> chosen {chosen.vocab_size:,}")
    else:
        chosen = next(r for r in results if r.vocab_size == min(a.resave_size, full.vocab_size))
        print(f"\nManual choice (--resave-size): {chosen.vocab_size:,}")
```
- 通常モード:書き下したルールを当てはめて、表示する。
- 再保存モード:**あなた**が選ぶ。`next(ジェネレーター)` は、大きさが一致する最初の結果を返す。

```python
    tok = full.truncated(chosen.vocab_size)
    tok.save(chosen_path)
```
- 選んだトークナイザーを作り、`tokenizer.json` として保存する。

### 8.6 検証

```python
    reloaded = ByteBPETokenizer.load(chosen_path)
    reloaded_full = ByteBPETokenizer.load(full_path)
    same = reloaded.merges == tok.merges and reloaded_full.merges == full.merges
    same = same and all(reloaded.encode(ln) == tok.encode(ln) for ln in val)
    extra = ["", "今日は晴れ", "ABC abc 123", "絵文字😀と記号①②", "\t改行\nを含む", "<bos>は普通の文字"]
    failures = roundtrip_failures(reloaded, val + extra)
    if not same or failures:
        raise SystemExit(f"Reload check FAILED: identical={same}, round-trip failures={failures}")
    print(f"Reload check: PASS (identical merges and encodings); round trip: PASS on {len(val) + len(extra):,} texts")
```
- 保存した2つのファイルを、ディスクから**読み込み直す**(保存と読み込みで何も失われないことの証明)。
- `same` — 読み込み直した結合が元と等しく、**かつ**検証用のすべての行が、読み込み直しの前後で同じように符号化される。
- `extra` — 意地悪なテスト文:空文字、英語と数字、絵文字と丸数字、タブと改行、そして文字どおりの `<bos>`(普通の文字として扱われなければならない)。
- `failures` — 検証用の行とextraでの、往復の失敗の数(0であるべき)。
- どこか悪ければ、エラーで止める。そうでなければPASSを表示する。

### 8.7 デモ、指紋、報告書

```python
    demos = [(t, reloaded.pieces(t)) for t in DEMO_SENTENCES]
    print("\nDemo (chosen tokenizer):")
    for t, p in demos:
        print(f"  {t}\n    -> {' / '.join(p)}  ({len(p)} tokens)")
```
- 各デモの文について、読める断片を取り、` / ` でつないで、トークン数と一緒に表示する。

```python
    hashes = {
        "data/tokenizer/tokenizer.json": hashlib.sha256(chosen_path.read_bytes()).hexdigest(),
        "data/tokenizer/tokenizer_full.json": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "data/corpus/train.txt (input)": sha256_file(a.train),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(a.report, log, results, chosen, a.tolerance, a.min_frequency, seconds, demos, key_terms,
                 achievable, reloaded, hashes, len(train), len(val), new_val_pieces)
    print(f"\ntokenizer.json SHA-256: {hashes['data/tokenizer/tokenizer.json']}")
    print(f"Report: {a.report}")
```
- 2つのトークナイザーのファイルと、学習の入力の指紋(最初の2つは `hashlib` を直接、3つ目は `sha256_file` を使う — 結果は同じ種類)。**学校と自宅で `tokenizer.json` のSHA-256を比べる** — 一致しなければならない。
- 報告書を書き、大事な指紋と場所を表示する。

### 8.8 ログと終了

```python
    if not a.no_log:
        text = "\n".join([
            "- Machine label: " + a.machine + " (CPU only; no GPU used)",
            f"- Script: scripts/day6_train_bpe.py ({SCRIPT_VERSION}); Python {platform.python_version()}",
            (f"- Training: {log.unique_pieces:,} unique pieces, {log.merges_learned:,} merges, {seconds:.1f} s, "
             f"largest vocabulary {full.vocab_size:,}" if log else "- Re-saved from tokenizer_full.json (no training)"),
            f"- Chosen vocabulary: {chosen.vocab_size:,}; validation {chosen.val_tokens:,} tokens "
            f"({chosen.tokens_per_char:.3f} per char); key terms 1-3 tokens {chosen.key_ok}/{chosen.key_achievable}; "
            f"longest question {max(chosen.max_val_tokens, chosen.max_train_tokens)} tokens",
            f"- tokenizer.json SHA-256: {hashes['data/tokenizer/tokenizer.json']}",
            "- Checks: reload PASS, round trip PASS",
        ])
        append_run_log(a.run_log, a.machine, text)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- 箇条書きの行から、ログの本文を作る。3つ目の項目は**条件式**:`log` があれば学習の統計を、無ければ「Re-saved ... (no training)」と書く。
- 「longest question」は、検証用と、抜き取った train の一番長い質問のうち、大きいほう。
- `"Checks: ... PASS"` は常に書かれる。失敗していれば、もっと前でプログラムが止まるため。
- 終わり方はDay 4・5と同じ。

---

## 9. 実行の流れ(順番)

1. オプションを読む → パス → Day 5のファイルがあるか確認。
2. 学習用と検証用の行を読む。
3. 1回学習する(`--resave-size` モードではフルのトークナイザーを読み込む)。`tokenizer_full.json` を保存。
4. 重要用語を探す → どれが達成可能か。
5. 候補の大きさごとに:フルのトークナイザーをその大きさに切って**測る**。
6. ルールで大きさを選ぶ(または手動で指定した大きさを使う)。
7. `tokenizer.json` を保存。2つのファイルを読み込み直して、同一性と往復を検証。
8. デモ → 指紋 → 報告書 → 実行ログ。

## 10. 結果の読み方

- **表:**大きさが増えるにつれて `val tokens` は減る(圧縮が良くなる)が、効果は次第に小さくなる。選ばれた行は、**両方の基準で十分良い**最初の大きさ。
- **`key terms 1-3` = `ok/achievable`:**選んだ大きさでは、2つの数が等しくなければならない(例:`38/38`)。
- **`max val / max train`:**一番長い質問のトークン数 → 文脈の長さを決める材料(余裕と、`<bos>`/`<eos>` の分を残す)。
- **`Reload check: PASS`:**保存と読み込みでロスが無く、トークナイザーが検証用の全行と意地悪なテスト文でロスレス。
- **`tokenizer.json SHA-256`:**もう一方のPCと比べる。同じハッシュ = 同一のトークナイザー。
- **「学習で一度も見ない検証用の断片」:**取り分けた文型なら正常。それでも、より小さなトークンで符号化される。

## 11. 知っておくと良い小さなこと

- **「train で一番長い質問」は抜き取り。** `max_train_tokens` は、学習用の10行おきの1行(`train[::10]`)でしか計算されないので、本当に一番長い学習用の質問は、少し長いかもしれない。文脈の長さを決めるときは、余裕を足すか、全行で最大を計算する(速い)こと。
- **検証は保存の後。** 選んだトークナイザーは、読み込み直しの確認の*前*にディスクへ書かれる。確認が失敗しても、ファイルは既に存在する。無効なものとして扱うこと。
- **`measure` の中で `key_terms` は未使用。** 渡されるが、使われるのは `achievable` だけ。
- **断片の数え上げが2回。** 通常モードでは `count_pieces(train)` が2回(学習用と `piece_counts` 用)走る。害はなく、少し無駄なだけ。
- **ルールは検証用に依存する。** 検証用は取り分けた*文型*(Day 5)なので、見たことのない言い回しをトークナイザーがどう扱うかの、公平なテストになる。
- **Day 5のトークンの範囲。** 選んだ大きさでの検証用の本当のトークン数は、Day 5の下限とUTF-8のバイト数の間に入るはず。

## 12. 一段落でまとめ

このスクリプトは、`train.txt` でバイトレベルBPEを1回学習し、結果を多くの候補の語彙サイズに切り、取り分けた検証用の文で、それぞれを測り(圧縮、用語集の重要な用語がいくつ1〜3トークンに収まるか、一番長い質問)、書き下したルールを満たす一番小さい大きさを選んで保存し、何も失われないことを証明するために読み込み直し、デモを見せ、すべて(SHA-256の指紋を含む)を報告書と実行ログに記録します。
