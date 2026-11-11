# Day 7.1 — token_data.py — やさしい1行ずつの解説(日本語)

> **Day 7 の4つのファイルの関係**
> - **Day 7.1(このファイル)`token_data.py`** — *ファイル形式のライブラリ*:トークンIDをディスクに書き込み、読み込む方法。
> - **Day 7 `day7_encode_corpus.py`** — *メインのスクリプト*:すべての質問でトークナイザーを確かめ、このライブラリでトークンファイルを書く。
> - **Day 7 `day7_tokenizer_demo.py`** — 小さな*デモ*スクリプト(文 → トークン → 番号 → 文)。
> - **Day 7.2 `test_token_data.py`** — *検査員*:このライブラリの自動テスト。
>
> おすすめの読む順番:**7.1 → 7(encode_corpus)→ 7(demo)→ 7.2**。

---

## 0. 全体像(まずここから)

Day 6のトークナイザーは、文を**数**に変えます。Day 9の学習プログラムは、その数を**何百万個**も読みます。そのため、**速くて安全な、ディスクへの保存方法**が必要です。このファイルは、その保存の形式を定めます。

**1本の長い巻物に印刷した時刻表**を想像してください:
- 各質問は `<bos> トークン トークン トークン ... <eos>` と書き、質問を**次々と**、**1本の長い列**(巻物)に書き込みます。
- 2つ目の小さなファイルは、**各質問がどこから始まるか**(「オフセット」)を覚えています — しおりのようなもの。
- 3つ目の小さなファイル(JSON)は**巻物のラベル**:質問の数、トークンの数、どのトークナイザーが作ったか、指紋 — 間違った巻物を使わないようにするため。

小さな例(IDは仮のもの。`B` = `<bos>`、`E` = `<eos>`):

```
質問1 = "A"  -> トークン  [B, 140, E]
質問2 = "BC" -> トークン  [B, 150, 160, E]

train.bin          : B 140 E B 150 160 E          (1本の列、数が7個)
train_offsets.bin  : 0, 3                          (質問1は位置0から、質問2は位置3から)
train_meta.json    : { questions: 2, tokens: 7, ... }
```

| ファイル(分割ごと) | 中身 | 型 |
|---|---|---|
| `train.bin` | トークンIDの列 | `uint16`(リトルエンディアン) |
| `train_offsets.bin` | 各質問の開始位置 | `int64`(リトルエンディアン) |
| `train_meta.json` | 件数、語彙サイズ、指紋 | JSONテキスト |

なぜ**1本の列**か? 学習プログラムが、**メモリマップ**を使って、ファイル全体をRAMに読み込まずに、列から学習用の「窓」を切り出すため。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **NumPy** | 数の配列を速く扱うPythonのライブラリ。(`np` が通常の短い名前。) |
| **配列 / `ndarray`** | 型が決まった数のリスト。コンパクトに保存され、速く処理される。 |
| **dtype** | 配列の数の型。 |
| **uint16** | 符号なし16ビット整数:0〜65,535の整数で、1個あたり**2バイト**。 |
| **int64** | 符号付き64ビット整数で、1個あたり8バイト。 |
| **リトルエンディアン** | 複数バイトの数を保存するときの順番:小さいバイトが先。例:258(= `0x0102`)は、バイト `02 01` として保存される。順番を固定すると、どのコンピューターでもファイルが同一になる。 |
| **オフセット** | 列の中で、何かが始まる位置。 |
| **ストリーム(列)** | 項目が次々と並んだ長い列。 |
| **メモリマップ(`memmap`)** | ファイルに直接つながった配列:OSが、触れた部分だけを読み込む。 |
| **窓(window)** | 列から切り出した短い部分(学習プログラムがモデルに渡すもの)。 |
| **メタファイル** | 大きなファイルを説明する小さなファイル。 |
| **スライス / ビュー** | `a[s:e]` は、配列の `s` から `e` までの部分。NumPyでは*ビュー*(コピーしない)。 |

---

## 2. docstring(説明文)

- これまでと同じ形式。**形式**(uint16のリトルエンディアンの列、質問ごとに `<bos> ... <eos>`、オフセットのファイル、メタのJSON)と、**1本の列にする理由**を説明している。
- 「サーバーで使う場合の注意」は、普通のPythonとNumPyだけで、ファイルが**リトルエンディアンを明示して**書かれるので、WindowsとLinuxで同じになる、と言っている。

---

## 3. import(道具を持ってくる)

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
```
- `from __future__ ...`、`hashlib`、`json`、`dataclass`、`Path`、`Iterable` — これまでの日と同じ。
- **新しいもの:**`asdict`(データクラスのオブジェクトを普通の辞書にする)と `numpy as np`(トークナイザー関係のコードで初めての**外部の**ライブラリ。venvにインストールされている必要がある)。

---

## 4. 定数

```python
TOKEN_DTYPE = np.dtype("<u2")    # uint16, little-endian
OFFSET_DTYPE = np.dtype("<i8")   # int64, little-endian
FORMAT_VERSION = "model-a-tokens-v1"
```
- `np.dtype("<u2")` — 型コードの読み方:`<` = リトルエンディアン、`u` = 符号なし整数、`2` = 2バイト。つまり、いつもリトルエンディアンのuint16。
- `np.dtype("<i8")` — リトルエンディアン、符号付き整数、8バイト = int64。
- `FORMAT_VERSION` — メタファイルに保存する名札(Day 6.1の `FORMAT_VERSION` と同じ)。`read_meta` が、別の形式のファイルを弾けるようにする。

---

## 5. `TokenFileMeta` データクラス

```python
@dataclass
class TokenFileMeta:
    """Everything needed to trust and use one token file."""

    format: str
    split: str
    vocab_size: int
    tokenizer_sha256: str
    bos_id: int
    eos_id: int
    questions: int
    tokens: int
    tokens_sha256: str
    source_sha256: str
```
- 「巻物のラベル」:
  - `format` — 形式の名札。
  - `split` — `train`、`val`、`test`。
  - `vocab_size` — それを作ったトークナイザーの語彙サイズ。
  - `tokenizer_sha256` — そのトークナイザーのファイルの指紋。
  - `bos_id`、`eos_id` — 開始・終了の印に使ったID。
  - `questions`、`tokens` — 件数。
  - `tokens_sha256` — トークンの列のバイト列の指紋。
  - `source_sha256` — 元になったテキストファイル(例:`train.txt`)の指紋。
- これらがあれば、後のプログラムが「このファイルは、このトークナイザーが、このテキストから作ったものか?」を確かめられる。

---

## 6. 関数

### 6.1 `sha256_bytes`

```python
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```
- Day 4と同じ:バイト列のSHA-256の指紋。

### 6.2 `encode_lines`

```python
def encode_lines(lines: Iterable[str], encode, bos_id: int, eos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode lines into one stream (<bos> ... <eos> per line) and the start offset of each line."""
    stream: list[int] = []
    offsets: list[int] = []
    for line in lines:
        offsets.append(len(stream))
        stream.append(bos_id)
        stream.extend(encode(line))
        stream.append(eos_id)
```
- 入力:行、**`encode` 関数そのもの**(`tok.encode` を渡す。関数は値のように受け渡せる)、2つの印のID。
- `stream` — すべてのトークンIDが入っていく、伸びるリスト。`offsets` — 各質問の開始位置。
- 各行について:
  1. `offsets.append(len(stream))` — **しおり**:次の質問は、今の長さの位置から始まる。
  2. `<bos>` を足す。
  3. その行のトークンIDを足す(`extend` はリストをまるごと足す)。
  4. `<eos>` を足す。

```python
    if stream and max(stream) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("a token ID does not fit in uint16")
    return np.asarray(stream, dtype=TOKEN_DTYPE), np.asarray(offsets, dtype=OFFSET_DTYPE)
```
- 安全確認:`np.iinfo(TOKEN_DTYPE).max` は、uint16が持てる最大の数(65,535)。それより大きいIDがあれば止まる(さもないと、静かに壊れる)。`stream and ...` は、空のリストに `max` を使わないための条件。
- `np.asarray(リスト, dtype=...)` は、Pythonのリストを、正しい型のコンパクトなNumPy配列に変えて、2つとも返す。

### 6.3 `split_stream`

```python
def split_stream(tokens: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """The token IDs of each line, including its <bos> and <eos>."""
    ends = list(offsets[1:]) + [len(tokens)]
    return [tokens[s:e] for s, e in zip(offsets, ends)]
```
- しおりの逆:列を、質問に切り分け直す。
- 各質問は、**次の質問が始まる所**で終わり、最後の質問は列の終わりで終わる:`ends = offsets[1:] + [len(tokens)]`。
- `zip(offsets, ends)` で、各開始位置と終了位置を組にする。`tokens[s:e]` がその質問のスライス(*ビュー*、コピーなし)。

### 6.4 `write_token_files`

```python
def write_token_files(out_dir: Path, split: str, tokens: np.ndarray, offsets: np.ndarray,
                      meta: TokenFileMeta) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tokens": out_dir / f"{split}.bin",
        "offsets": out_dir / f"{split}_offsets.bin",
        "meta": out_dir / f"{split}_meta.json",
    }
    tokens.astype(TOKEN_DTYPE).tofile(paths["tokens"])
    offsets.astype(OFFSET_DTYPE).tofile(paths["offsets"])
    paths["meta"].write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return paths
```
- フォルダーがあることを確認し、3つのファイルのパス(`train.bin`、`train_offsets.bin`、`train_meta.json`)を決める。
- `tokens.astype(TOKEN_DTYPE)` で、配列がぴったり正しい型(リトルエンディアンのuint16)であることを保証する。`.tofile(path)` は、配列の**生のバイト列**を書く — ヘッダーも何もなし。dtypeがバイトの順番を固定するので、ファイルはどのコンピューターでも同じになる。
- オフセット(int64)も同様。
- メタ:`asdict(meta)` → 辞書 → `json.dumps`(読みやすく、日本語はそのまま、インデント2)→ LFの改行で保存(SHA-256が同一になる、Day 5・6.1と同じ考え方)。
- パスの辞書を返す。

### 6.5 `read_meta`

```python
def read_meta(out_dir: Path, split: str) -> TokenFileMeta:
    data = json.loads((out_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
    if data.get("format") != FORMAT_VERSION:
        raise ValueError(f"{split}_meta.json: unknown format {data.get('format')}")
    return TokenFileMeta(**data)
```
- JSONを辞書として読み、形式の名札が自分たちのものでなければ弾く。
- `TokenFileMeta(**data)` — `**` は**辞書をキーワード引数に展開する**:`{"split": "train", ...}` が `split="train", ...` になる。(ファイルにキーの不足や余分があると、Pythonは `TypeError` を出す。)

### 6.6 `read_token_files`

```python
def read_token_files(out_dir: Path, split: str, memmap: bool = True) -> tuple[np.ndarray, np.ndarray, TokenFileMeta]:
    """Open a split. With memmap=True the tokens stay on disk until used (what the trainer needs)."""
    meta = read_meta(out_dir, split)
    path = out_dir / f"{split}.bin"
    tokens = np.memmap(path, dtype=TOKEN_DTYPE, mode="r") if memmap else np.fromfile(path, dtype=TOKEN_DTYPE)
    offsets = np.fromfile(out_dir / f"{split}_offsets.bin", dtype=OFFSET_DTYPE)
    if len(tokens) != meta.tokens or len(offsets) != meta.questions:
        raise ValueError(f"{split}: file sizes do not match {split}_meta.json")
    return tokens, offsets, meta
```
- 最初にメタを読む。
- `memmap=True`(初期値):`np.memmap(path, dtype=..., mode="r")` — **読み取り専用のメモリマップ**:配列は普通のように振る舞うが、データはディスクに残り、触れた部分だけが読み込まれる。巨大なファイルで、学習プログラムが必要とするもの。
- `memmap=False`:`np.fromfile(...)` — ファイル全体をメモリに読み込む。
- オフセットは小さいので、いつも全部を読み込む。
- **整合性の確認:**ファイルのトークン数は `meta.tokens` と等しく、オフセットの数は `meta.questions` と等しくなければならない。そうでなければ `ValueError`(ファイルが途中で切れているか、メタが別のファイルのもの)。
- トークン、オフセット、メタを返す。

---

## 7. 知っておくと良い小さなこと

- **uint16の限界。** 65,535を超えるIDは保存できない。私たちの語彙は小さい(数百〜数千)ので安全で、万一そうなれば `encode_lines` がエラーを出す。
- **読み込み側は、サイズを確認するだけで、指紋は確認しない。** `read_token_files` は件数を確認するが、`tokens_sha256` を計算し直したり、`tokenizer_sha256` / `vocab_size` を比べたりはしない。ファイルが今のトークナイザーに合うかの確認は、**学習プログラム(Day 9)の仕事** — そのための情報は、メタファイルに入っている。
- **書き込み時のメモリ。** `encode_lines` は、まず普通のPythonのリストを作る。このコーパスの大きさなら問題ないが、巨大なものでは重くなる。
- **コピーではなくビュー。** `split_stream` のスライスは、元の配列を指している。元のまま残したいなら、変更しないこと。

## 8. 一段落でまとめ

`token_data.py` は、符号化した文のための、シンプルで固定されたファイル形式を定めます:全質問を `<bos> … <eos>` の形で1本のリトルエンディアンの `uint16` の列にし、各質問の開始位置を別の `int64` のファイルに、件数と指紋を入れたJSONの「ラベル」を別に保存します。これらのファイルを書き、読み込み(必要ならメモリマップとして)、列を質問に切り分け直し、ラベルとサイズが合わないファイルを弾きます。
