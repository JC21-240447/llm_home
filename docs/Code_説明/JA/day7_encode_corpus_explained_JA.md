# Day 7 — day7_encode_corpus.py — やさしい1行ずつの解説(日本語)

> **Day 7 の4つのファイルの関係**
> - **Day 7.1 `token_data.py`** — ファイル形式のライブラリ(このスクリプトが使うので、**先に**読んでください)。
> - **Day 7(このファイル)`day7_encode_corpus.py`** — *メインのスクリプト*:**すべての**質問でトークナイザーを確かめ、トークンファイルを書き、読み直し、実際のトークン数を報告する。
> - **Day 7 `day7_tokenizer_demo.py`** — 小さなデモのスクリプト。
> - **Day 7.2 `test_token_data.py`** — ライブラリの自動テスト。

---

## 0. 全体像(まずここから)

Day 6でトークナイザーを作りました。Day 7の問いは:**「コーパス全体でこれを信頼できるか? そして結果を安全に保存できるか?」**

**出荷前の品質管理の段階**だと考えてください:

| 手順 | 何が起こるか |
|---|---|
| 1 | トークナイザー(Day 6)と3つの分割(Day 5)を読み込む |
| 2 | **質問1つ1つで往復テスト**(と難しい文でも):`decode(encode(text)) == text`。1つでも失敗したら止まる |
| 3 | **重要な用語**が何トークンかかるかを測る → `docs/DAY7_TERM_TOKENS.csv` を書く |
| 4 | 各分割を `<bos> 質問 <eos> ...` の1本の列に符号化 → `data/tokens/{train,val,test}.bin`(+ オフセット + メタ)|
| 5 | **ファイルをディスクから読み直して**復号する:すべての質問が完全に元に戻ること |
| 6 | **実際の**トークン数を報告する(Day 5では下限と上限しか分からなかった) |

実行:

```
.venv\Scripts\python.exe scripts\day7_encode_corpus.py --machine home
```

後日Linuxのサーバーで:`.venv/bin/python scripts/day7_encode_corpus.py --machine server`(スラッシュとLinuxのvenvのパス)。トークンファイルは両方のマシンで**同一**になるはず(報告書のSHA-256で比べる)。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **往復(round trip)** | 符号化してから復号し、元と比べること。 |
| **ストリーム(列)** | すべての質問を、IDの1本の長いリストにつなげたもの。各質問は `<bos> … <eos>` で包まれる(Day 7.1を参照)。 |
| **読み直し(read-back)** | 書き込んだファイルをディスクから読み直して、(プログラムのメモリだけでなく)ファイルそのものが正しいことを証明すること。 |
| **文字の途中のトークン** | 文字の*一部*だけのトークン。トークナイザーが学習していない文字(1バイトずつ書かれる)に現れる。 |
| **重要用語(key terms)** | 用語集(Day 6を参照)の、列名・値・検索語(列名、値、検索語)と、その標準用語・言い換え。 |
| **1文字あたりのトークン数** | `トークン数 ÷ 文字数`。低いほど良い圧縮。 |

---

## 2. docstring(説明文)

- これまでと同じ形式。上の6つの手順、使い方のコマンド、**サーバーの注意**(Linuxでもコードは同じだが、スラッシュと `server` のラベルを使う)を載せている。
- 手順4の形式 `<bos> 質問 <eos> <bos> 質問 <eos> ...` に注目 — Day 7.1が書くものと同じ。

---

## 3. importと準備

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- すべて **Day 4〜6と同じ**(`Counter` は数え上げに、`platform` はPythonのバージョンに使う)。`REPO_ROOT` と `sys.path.insert` の行も、Day 6と同じ技(`from src... import` が動くように)。

```python
from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer  # noqa: E402
from src.token_data import (  # noqa: E402
    FORMAT_VERSION, TokenFileMeta, encode_lines, read_token_files, sha256_bytes, split_stream, write_token_files,
)
```
- トークナイザーのエンジン(Day 6.1)から:印のID、バイトのオフセット(104)、トークナイザーのクラス。
- ファイル形式のライブラリ(**Day 7.1**)から:形式の名前、メタの箱、書き込み・読み込み・切り分けの補助関数。

---

## 4. 定数

```python
SCRIPT_VERSION: str = "day7-v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")
KEY_DB_GROUPS: frozenset[str] = frozenset({"列名", "値", "検索語"})
```
- `SCRIPT_VERSION` は前と同じ。`SPLITS` はDay 5と同じ。`KEY_DB_GROUPS` はDay 6と同じ(「重要」とみなす用語集のカテゴリ)。

```python
TRICKY_TEXTS: tuple[str, ...] = (
    "", "今日は晴れ", "Hello, world! 123", "絵文字😀👍と記号①②③※", "改行\nとタブ\tを含む文",
    "ＡＢＣ全角とﾊﾝｶｸ", "<bos><eos><unused_7> は普通の文字", "「」『』【】・〜…", "𠮷野家(4バイトの漢字)",
)
```
- Day 6.2のリストに似た、9つの意地悪な文:空文字、普通の日本語、数字付きの英語、**絵文字**と記号、**改行とタブ**、**全角・半角**の形、**特殊トークンに見える**文(普通の文字として扱われなければならない)、日本語のかっこと句読点、そして `𠮷` — **4バイトの漢字**(普通の漢字の範囲の外なので、トークナイザーは「その他」の文字として扱うが、それでも往復できなければならない)。

---

## 5. `SplitReport` データクラス

```python
@dataclass
class SplitReport:
    split: str
    questions: int
    characters: int
    tokens: int                # including <bos>/<eos>
    content_tokens: int        # without <bos>/<eos>
    avg_tokens: float          # per question, including <bos>/<eos>
    max_tokens: int
    partial_tokens: int        # tokens that are only part of a character (a character never learned)
    questions_with_partial: int
    tokens_sha256: str
```
- 分割ごとの「結果カード」(Day 6の `SizeResult` と同じ考え方):質問と文字の数、印あり・なしの総トークン数、質問の平均と最長(トークン数)、文字の途中のトークンの数とそれを含む質問の数、`.bin` ファイルの指紋。

---

## 6. ヘルパー関数

### 6.1 `read_lines`

```python
def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
```
- Day 6と同じ。

### 6.2 `load_key_terms`

```python
def load_key_terms(glossary: Path, synonyms: Path) -> list[tuple[str, str, str]]:
    """(term, group, source) for column names, values and query words, standard terms and variants."""
    out: dict[str, tuple[str, str, str]] = {}
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                g = r["DB区分"].strip()
                t = r["標準用語"].strip()
                if g in KEY_DB_GROUPS and t:
                    out.setdefault(t, (t, g, "用語集"))
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                g = r["DB区分"].strip()
                t = r["変形・同義語"].strip()
                if g in KEY_DB_GROUPS and t:
                    out.setdefault(t, (t, g, "同義語辞書"))
    return sorted(out.values())
```
- Day 6の版に似ているが、**3つ組** `(用語, グループ, 出典)` を返す — CSVに各用語の出どころを書けるように — そして学習用の文での絞り込みは**しない**。
- `out` は用語をキーにした辞書。`setdefault` は用語ごとに**最初の**項目を残すので、両方のファイルにある用語は `用語集` と記される(用語集を先に読むため)。
- 用語集のブロック(標準用語)と同義語のブロック(言い換え)は、列名と出典のラベルが違うだけで、同じ考え方。
- `sorted(out.values())` は、3つ組を(用語の順に)並べ替えて返す。

### 6.3 `partial_token_ids`

```python
def partial_token_ids(tok: ByteBPETokenizer) -> set[int]:
    """IDs whose bytes are not complete characters: used only for characters training never learned.
    (Single ASCII bytes such as digits are complete characters and are NOT counted.)"""
    out: set[int] = set()
    for i in range(BYTE_OFFSET, tok.vocab_size):
        try:
            tok.token_bytes[i].decode("utf-8")
        except UnicodeDecodeError:
            out.add(i)
    return out
```
- **文字の一部だけ**のトークンIDを、全部見つける。
- 104(最初のバイトのトークン)から語彙の終わりまで、全IDをループする。それぞれについて、そのバイト列を**単独で**UTF-8として復号してみる。失敗したら(`UnicodeDecodeError`)、そのトークンは不完全 → 集合に追加する。
- なぜ動くか:`0xE4`(3バイトの日本語の文字の先頭)のような1バイトは、単独では正しいUTF-8ではない。`5` のような1バイトのASCIIは正しいので、数字とASCIIの記号は数えられ**ない**。
- なぜ気にするか:学習した文字には、普通トークンがある。学習で見たことのない文字は**1バイトずつ**書かれる → 途中のトークンになる。それを数えれば、トークナイザーが「未知の」文字にどれだけ出会うかが分かる。

### 6.4 `write_report`

```python
def write_report(path, reports, tok, tok_sha, term_rows, n_tricky) -> None:
    in_train = [r for r in term_rows if r[5] == "yes"]
    dist_all: Counter = Counter(min(int(r[3]), 6) for r in term_rows)
    dist_in: Counter = Counter(min(int(r[3]), 6) for r in in_train)
    long_in = [f"{r[0]} ({r[3]})" for r in in_train if int(r[3]) > 3]
```
- `docs/DAY7_ENCODING_REPORT.md` を書く。`term_rows` の各行は `[用語, グループ, 出典, トークン数, 断片, 学習用の文に有無]` なので、`r[5]` は `"yes"/"no"`、`r[3]` はトークン数(文字列)。
- `in_train` — 学習用の質問に現れる用語。
- `dist_all`、`dist_in` — **1、2、3、…トークンかかる用語がいくつあるか**。`min(数, 6)` で、6以上をまとめて1つの「6」の区分にする。
- `long_in` — 学習用の文に現れる、**3トークンを超える**用語。`勤続 (4)` のように書く。

Markdownの `lines` には、次のセクションが入る:

| セクション | 内容 |
|---|---|
| ヘッダー | バージョン、日付、トークナイザーの語彙とSHA-256 |
| **Round trip** | すべての質問と意地悪な文が合格。トークンファイルの読み直しもOK |
| **Real token counts** | 分割ごとに1行:質問数、文字数、トークン数(印付き)、**1文字あたりのトークン数 = `content_tokens / characters`**、質問の平均トークン数、最長の質問、文字の途中のトークン、それを使う質問。表の下に、「文字の途中のトークン」の意味と、1文字あたりのトークン数が `<bos>/<eos>` を含まないことの説明 |
| **Key terms** | 重要用語の数、学習用に現れる数。「必要なトークン数 → 用語の数」の表(最後の区分は `'6 or more'` のラベル)。3トークンを超える用語の一覧 |
| **Files (SHA-256)** | `.bin` ファイルごとに指紋1つ |

- `*( ... for r in reports)` と `*( ... for k in sorted(dist_all))` は、表の行を展開する(前と同じ技)。区分が無いとき `dist_in[k]` は0を返す(`Counter` の性質)。
- `path.write_text("\n".join(lines), encoding="utf-8", newline="\n")`:Day 5・6と同じLFの改行。

### 6.5 `append_run_log`

```python
def append_run_log(log_path: Path, machine: str, reports: list[SplitReport], tok_sha: str) -> None:
    body = [
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Script: scripts/day7_encode_corpus.py ({SCRIPT_VERSION}); Python {platform.python_version()}",
        f"- Tokenizer SHA-256: {tok_sha}",
        *(f"- {r.split}: {r.questions:,} questions, {r.tokens:,} tokens (avg {r.avg_tokens:.1f}, max {r.max_tokens}), "
          f"SHA-256 {r.tokens_sha256}" for r in reports),
        "- Checks: round trip PASS on every question and tricky text; disk read-back PASS",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 7 encode corpus\n\n" + "\n".join(body) + "\n")
```
- Day 6と同じ形:箇条書きの行(分割ごとに1行、`*( ... )` の展開)を作り、見出し付きの項目を `run_log.md` に追記する。「Checks … PASS」の行は、失敗していればスクリプトが先に止まるので、常に書かれる。

---

## 7. `parse_args`

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 7: test the tokenizer on every question and encode the corpus.")
    ap.add_argument("--machine", choices=["home", "server", "school"], required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "data/corpus")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/tokens")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY7_ENCODING_REPORT.md")
    ap.add_argument("--terms-csv", type=Path, default=REPO_ROOT / "docs/DAY7_TERM_TOKENS.csv")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()
```
- 前と同じ `argparse` の考え方。新しい点:`--machine` が、将来のLinuxマシン用の **`server`** も受け付ける。`Path` のオプション(トークナイザー、コーパスのフォルダー、出力先、用語集、同義語、報告書、用語のCSV、実行ログ)はどれも同じ働き。`--no-log` は前と同じ。

---

## 8. `main`

### 8.1 準備と読み込み

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("tokenizer", "corpus_dir", "out_dir", "glossary", "synonyms", "report", "terms_csv", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    if not a.tokenizer.exists():
        raise SystemExit(f"Missing {a.tokenizer}. Run Day 6 first.")
```
- Day 4〜6と同じ前置き(コンソールの安全対策、絶対パス)。トークナイザーのファイル(Day 6から)が必要。

```python
    print(f"=== Day 7 encode corpus | machine={a.machine} | Python {platform.python_version()} ===")
    tok = ByteBPETokenizer.load(a.tokenizer)
    tok_sha = sha256_bytes(a.tokenizer.read_bytes())
    print(f"Tokenizer: vocabulary {tok.vocab_size:,}, SHA-256 {tok_sha[:12]}...")
```
- 見出しを表示し、トークナイザーを読み込み、その指紋(`tok_sha`。すべてのメタファイルに入る)を計算し、最初の12文字(`[:12]`)を短縮形として表示する。

### 8.2 難しい文での往復

```python
    bad = [t for t in TRICKY_TEXTS if tok.decode(tok.encode(t)) != t]
    if bad:
        raise SystemExit(f"Round trip FAILED on tricky texts: {bad}")
    print(f"Round trip on {len(TRICKY_TEXTS)} tricky texts: PASS")
```
- 符号化→復号で元に戻**らない**難しい文を集める。1つでもあれば止まる。なければPASS。

### 8.3 分割ごとの処理

```python
    partial_ids = partial_token_ids(tok)
    reports: list[SplitReport] = []
    for split in SPLITS:
        path = a.corpus_dir / f"{split}.txt"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run Day 5 first.")
        lines = read_lines(path)
```
- 文字の途中のIDの集合を、一度だけ計算する。そして `train`、`val`、`test` について:テキストファイル(Day 5から。無ければ止まる)を探し、行を読む。

```python
        unique = list(dict.fromkeys(lines))
        failures = [ln for ln in unique if tok.decode(tok.encode(ln)) != ln]
        if failures:
            raise SystemExit(f"Round trip FAILED on {len(failures)} {split} questions, e.g. {failures[0]!r}")
```
- `dict.fromkeys(lines)` は、行をキーにした辞書を作る → **順番を保ったまま、繰り返しを除く**。(train にはオーバーサンプリング(Day 5)による繰り返しの行がある。異なる文を1回ずつ試せば十分で、速い。)
- **異なる質問すべて**が往復できるか確認する。失敗があれば止まり、件数と最初の例(`!r` は引用符付きで表示)を見せる。

```python
        tokens, offsets = encode_lines(lines, tok.encode, BOS_ID, EOS_ID)
        meta = TokenFileMeta(FORMAT_VERSION, split, tok.vocab_size, tok_sha, BOS_ID, EOS_ID, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), sha256_bytes(path.read_bytes()))
        write_token_files(a.out_dir, split, tokens, offsets, meta)
```
- **すべての**行(繰り返しも含む。列には学習用の全行が必要なので)を、`tokens` と `offsets` に符号化する(Day 7.1)。
- メタの「ラベル」を、項目の**順番どおりに**作る:形式、分割、語彙サイズ、トークナイザーの指紋、`<bos>`/`<eos>` のID、質問の数、トークンの数、トークンのバイト列の指紋、元のテキストファイルの指紋。
- この分割の3つのファイルを書く。

```python
        disk_tokens, disk_offsets, _ = read_token_files(a.out_dir, split, memmap=True)
        for ln, ids in zip(lines, split_stream(disk_tokens, disk_offsets)):
            if ids[0] != BOS_ID or ids[-1] != EOS_ID or tok.decode(ids.tolist()) != ln:
                raise SystemExit(f"Disk read-back FAILED in {split}: {ln!r}")
```
- **ディスクからの読み直し**(メモリマップで):今書いたファイルを開き、列を質問に切り分け直し、元の各行について、質問が `<bos>` で始まり、`<eos>` で終わり、**ちょうど**元の行に復号されることを確かめる。これで、メモリ上の配列だけでなく、**ファイル**が正しいことが証明される。(`ids.tolist()` はNumPyの配列を `decode` 用に普通のPythonの整数にする。)`_` は、返されたメタを無視する。

```python
        per_q = [int(e - s) for s, e in zip(offsets, list(offsets[1:]) + [len(tokens)])]
        is_partial = [int(x) in partial_ids for x in tokens.tolist()]
        q_partial = sum(1 for ids in split_stream(tokens, offsets) if any(int(x) in partial_ids for x in ids.tolist()))
```
- `per_q` — 質問ごとのトークン数(印を含む)= 終わりの位置 − 始まりの位置。`split_stream` と同じ「次の始まりが自分の終わり」の考え方。
- `is_partial` — **各トークン**が、文字の途中のトークンか?(True/Falseのリストで、合計すると数になる。)
- `q_partial` — 途中のトークンが1つでも現れる**質問**がいくつあるか(`any(...)` =「1つでもあるか」)。

```python
        r = SplitReport(split, len(lines), sum(len(ln) for ln in lines), len(tokens), len(tokens) - 2 * len(lines),
                        sum(per_q) / len(per_q), max(per_q), sum(is_partial), q_partial,
                        hashlib.sha256((a.out_dir / f"{split}.bin").read_bytes()).hexdigest())
        reports.append(r)
```
- 結果カードを作る:質問数、総文字数、総トークン数、**内容のトークン数 = トークン数 − 質問ごとに2**(`<bos>` と `<eos>` を除く)、平均と最長の質問、途中のトークンの数、途中のトークンを含む質問の数、そして**ディスク上のファイル**のSHA-256。

```python
        print(f"{split:<6} {r.questions:>8,} questions  {r.tokens:>10,} tokens  avg {r.avg_tokens:4.1f}  max {r.max_tokens:>3}  "
              f"partial-char {r.partial_tokens:>4} in {r.questions_with_partial} questions  -> round trip + read-back PASS")
```
- 分割ごとに、整った要約を1行表示する(書式のコードはDay 5・6と同じ)。この行に着いた時点で、2つの確認は合格している。

### 8.4 重要用語

```python
    terms = load_key_terms(a.glossary, a.synonyms)
    train_blob = "\n".join(read_lines(a.corpus_dir / "train.txt"))
    term_rows: list[list[str]] = []
    for term, group, source in terms:
        n = len(tok.encode(term))
        term_rows.append([term, group, source, str(n), " / ".join(tok.pieces(term)), "yes" if term in train_blob else "no"])
```
- 重要用語を読み込み、学習用の文を1つの大きな文字列にする(Day 6と同じ)。
- 各用語について:`n` = 必要なトークン数。行は、用語、グループ、出典、`n`(文字列)、` / ` でつないだ読める断片、そして用語が学習用の文の**どこかに現れるか**(`term in train_blob` — 部分文字列の確認)。

```python
    a.terms_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.terms_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "DB区分", "source", "tokens", "pieces", "in_train_text"])
        w.writerows(term_rows)
```
- 表をCSVとして保存する(`utf-8-sig` でExcelでも日本語が正しく見える。`newline=""` は `csv` モジュール用 — Day 5のラベルファイルと同じ)。`writerow` がヘッダーを書き、`writerows` が全行を一度に書く。

```python
    in_train = [r for r in term_rows if r[5] == "yes"]
    print(f"Key terms: {len(term_rows)} (in training text: {len(in_train)}); "
          f"1-3 tokens: {sum(1 for r in term_rows if int(r[3]) <= 3)} of all, "
          f"{sum(1 for r in in_train if int(r[3]) <= 3)} of those in the training text")
```
- 重要用語がいくつあるか、学習用に現れるのがいくつか、1〜3トークンのものがいくつか(全体と、学習用に現れるものの中で)を表示する。

### 8.5 報告書、ログ、終了

```python
    write_report(a.report, reports, tok, tok_sha, term_rows, len(TRICKY_TEXTS))
    total = sum(r.tokens for r in reports)
    print(f"\nTotal tokens: {total:,} (train {reports[0].tokens:,})")
    print(f"Report: {a.report}\nTerm table: {a.terms_csv}")
    if not a.no_log:
        append_run_log(a.run_log, a.machine, reports, tok_sha)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- 報告書を書き、総トークン数を表示し(`SPLITS` が train から始まるので、train は `reports[0]`)、ファイルの場所を表示し、`--no-log` でなければ実行ログに追記し、0を返す。終わり方は前と同じ。

---

## 9. 実行の流れ(順番)

1. オプション → 絶対パス → トークナイザーが必要。
2. トークナイザーを読み込み、指紋を取る。
3. 9つの難しい文で往復。
4. 分割ごとに:**異なる質問すべて**で往復 → 符号化 → 3つのファイルを書く → **読み直して全行を復号** → トークン・途中のトークンを数える → 指紋。
5. 重要用語:用語ごとのトークン数 → CSV。
6. 報告書 → 実行ログ。

## 10. 結果の読み方

- **PASSの行がすべてそろう** = トークナイザーがコーパス全体でロスレスで、ディスク上のファイルも正しい。
- **実際のトークン数とDay 5の範囲:**各分割の実際の合計は、Day 5の「トークンの下限」とその「UTF-8のバイト数」の間に入るはず。
- **1文字あたりのトークン数**(例:0.4)が圧縮を表す。低いほど良い。
- **最長の質問(トークン数)**は、文脈の長さを決める材料(Day 6も推定値を出したが、こちらは全行を対象にするので正確)。
- **文字の途中のトークン:**train では、0かごく少ないのが理想。val/test では、少しあっても正常(そこにしか出ない文字)。
- **重要用語の表:**3トークンを超える用語は、複数の断片でできた句か、長い単語の中にしか出ない単語(Day 6と同じ)。
- **`.bin` ファイルのSHA-256:**自宅とサーバーで比べる — 同一でなければならない。

## 11. 知っておくと良い小さなこと

- **重要用語の「学習用に有り」は部分文字列の確認**(`term in train_blob`)なので、短い用語は長い単語の中にも一致しうる。Day 6はもっと厳しい「達成可能」の判定を使った。
- **空の分割ファイル**だと、平均(`sum(per_q) / len(per_q)`)がゼロ除算で落ちる。Day 5の本物の出力では起こらない。
- **ファイルは読み直しの確認の前に書かれる。**読み直しが失敗しても、ファイルは既に存在する。無効なものとして扱うこと。
- **train の往復は異なる行だけ、読み直しは全行。**どちらも完全:繰り返しの行は、元の行と同一だから。
- **`memmap=True` の読み直し**は、Windowsでは配列がスコープを出るまでファイルのハンドルを開いたままにする。実行中にファイルを削除する処理はないので、ここでは無害。

## 12. 一段落でまとめ

このスクリプトは、train・検証用・テスト用のすべての質問(と意地悪な文)で、トークナイザーがロスレスであることを証明し、各分割を `<bos> … <eos>` の形で `uint16` のトークンIDの列として、オフセットとメタファイルと一緒に書き込み、ディスクからファイルを読み直して、完全に復号できることを証明し、実際のトークン数、文字の途中のトークン、重要用語に必要なトークン数を測り、すべてをSHA-256の指紋付きで、報告書、CSV、実行ログに記録します。
