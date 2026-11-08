# Day 6.1 — bpe_tokenizer.py — やさしい1行ずつの解説(日本語)

> **Day 6 の3つのファイルの関係**
> - **Day 6.1(このファイル)`bpe_tokenizer.py`** — *エンジン*:トークナイザー本体(文を切る、数に変える、結合を学ぶ)。
> - **Day 6 `day6_train_bpe.py`** — *運転手*:このエンジンを使って学習し、いくつかの大きさを試し、1つ選び、保存し、報告書を書く。
> - **Day 6.2 `test_bpe_tokenizer.py`** — *検査員*:エンジンを自動で検査するテスト。
>
> おすすめの読む順番:**6.1 → 6 → 6.2**。

---

## 0. 全体像(まずここから)

ニューラルネットワークは文字を読めません。読めるのは**数**だけです。**トークナイザー**は、その2つの間の翻訳者です:

```
"営業部の社員"  --符号化(encode)-->  [ 412, 380, 951 ]  --復号(decode)-->  "営業部の社員"
```

**LEGO**で考えてみましょう:
- まず、すべてのバイト(0〜255)に1つずつ、**256個の小さなブロック**があります。これで*何でも*(どんな文字も、絵文字も)組み立てられるので、「未知語」は存在しません。
- 次に、学習用の文を見て、2つのブロックがよく隣り合っているのに気づきます。それを**くっつけて**1つの大きなブロックにします。これを何度も繰り返します。くっつける1回を**結合(merge)**と呼びます。
- **結合の順番付きリスト**が「レシピ本」です。新しい文を符号化するときは、小さなブロックから始めて、レシピを**学んだ順に**当てはめ直します。

このファイルには3つの部分があります:

| 部分 | 何をするか |
|---|---|
| **プリトークナイザー**(`char_class`、`pretokenize`) | まず文字の種類で、文を「単語のような塊」に切る(漢字の連続、ひらがなの連続など) |
| **`ByteBPETokenizer` クラス** | 符号化、復号、保存、読み込み、小さい版を作る |
| **学習**(`count_pieces`、`train_bpe`) | 学習用の文から結合を学ぶ |

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **トークン** | モデルが読む1つの断片。バイトのことも、1文字のことも、営業部のような単語全体のこともある。 |
| **トークンID** | トークンの番号。モデルにはIDしか見えない。 |
| **語彙 / 語彙サイズ(vocab size)** | トークンの全リスト / その数。 |
| **バイト / UTF-8** | コンピューターは文字をバイト(0〜255)で保存する。UTF-8では、日本語1文字はたいてい**3バイト**。 |
| **BPE(バイトペアエンコーディング)** | 隣り合うトークンの組のうち、一番多いものを繰り返し結合する方式。「バイトレベル」= バイトから始める。 |
| **結合(merge)** | 学んだルール1つ:「トークンAの次にトークンBが来たら、新しいトークンCにする」。 |
| **組(pair)** | 隣り合う2つのトークン `(A, B)`。 |
| **ランク(rank)** | 結合リストの中での位置。ランク0が最初に学んだもの。ランクが小さいほど先に使う。 |
| **プリトークナイザー** | BPEの前に行う最初の切り分け。BPEの結合は、この切れ目をまたげない。 |
| **特殊トークン** | 役目のあるトークン:`<pad>`(埋め草)、`<bos>`(開始)、`<eos>`(終了)、`<unk>`(未知)。 |
| **予約ID** | 将来のために空けてある100個の予備ID。 |
| **ロスレス(lossless)** | 何も失われない:`decode(encode(text)) == text`。 |
| **往復(round trip)** | 符号化してから復号し、元の文と比べること。 |
| **決定的(deterministic)** | 同じ入力なら必ず同じ出力(隠れた乱数がない)。 |
| **キャッシュ** | 一度計算した答えを覚えておくメモ帳。同じ仕事を繰り返さないため。 |
| **`frozenset`** | 絶対に変更できない集合。 |

---

## 2. ファイル先頭のdocstring(説明文)

```python
r"""Byte-level BPE tokenizer for Model A, written from scratch (no tokenizer library).
...
"""
```
- これまでと同じ形式(raw文字列、英語+日本語、実行はされない)。
- 3つの手順(プリトークナイザー → バイト → 結合)、**固定のIDの並び**(セクション4)、そして、正規化をせずすべてのバイトを表せるので**ロスレス**だという約束を説明している。
- docstringの例:`営業部の社員は何人ですか。` → `営業部 | の | 社員 | は | 何人 | ですか | 。`。

---

## 3. import(道具を持ってくる)

```python
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
```
- `from __future__ ...`、`json`、`Counter`、`defaultdict`、`dataclass`、`Path`、`Iterable` — すべて **Day 4・Day 5と同じ**。(`csv` や `zipfile` などは無い:このファイルに必要なのはJSONと数え上げだけ。)

---

## 4. 定数とIDの並び

```python
FORMAT_VERSION: str = "model-a-byte-bpe-v1"
PRETOKENIZER_NAME: str = "script_class_v1"
```
- 保存するファイルに書き込む、2つの名札。`load` がこれを確認するので、別の形式で作られたファイルを間違えて読み込むことがない。

```python
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
NUM_RESERVED: int = 100
RESERVED_FIRST_ID: int = len(SPECIAL_TOKENS)                  # 4
BYTE_OFFSET: int = RESERVED_FIRST_ID + NUM_RESERVED           # 104
FIRST_MERGE_ID: int = BYTE_OFFSET + 256                       # 360
```
- `SPECIAL_TOKENS` — 4つの特殊な名前を順番に。
- `PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3` — **タプルの展開代入**:4つの名前に、1行で4つの数を割り当てる。
- `NUM_RESERVED = 100` — 予備IDの数。
- 続く3行で、各ブロックの開始位置を計算している:

| IDの範囲 | そこにあるもの |
|---|---|
| 0 〜 3 | `<pad>`、`<bos>`、`<eos>`、`<unk>` |
| 4 〜 103 | `<unused_0>` 〜 `<unused_99>`(100個の予備ID) |
| 104 〜 359 | 256個の1バイト(`BYTE_OFFSET` = 104 → バイト `b` のIDは `104 + b`) |
| 360 〜 | 学習した結合(学習した順)(`FIRST_MERGE_ID` = 360) |

```python
Pair = tuple[int, int]
```
- **型エイリアス**:`Pair` は「整数2つのタプル」の分かりやすい別名にすぎない。

---

## 5. プリトークナイザー

### 5.1 `char_class`

```python
def char_class(ch: str) -> str:
    """Script class of one character. Digits and 'other' never form runs."""
    o = ord(ch)
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    ...
    return "other"
```
- **Day 5の `script_class` と同じ考え方**(ひらがな / カタカナ / 漢字 / ラテン文字 / 数字 / 空白 / その他を、Unicodeの番号の範囲で判定)。見た目の違いは、ラテン文字の判定に余分なかっこ `(ch.isascii() and ch.isalpha())` が付いていることだけで、意味は同じ。
- Day 5のものは「Day 6のプリトークナイザーに合わせる」ために書いたもので、こちらが本物。

### 5.2 `_RUN_CLASSES`

```python
_RUN_CLASSES: frozenset[str] = frozenset({"hiragana", "katakana", "kanji", "latin", "space"})
```
- **連続してひとかたまりになってよい**文字の種類。`digit`(数字)と `other`(その他)はこの集合に**入っていない**ので、数字1つ、句読点1つが、いつも単独の断片になる。
- なぜか? 数は1桁ずつ書けるので(どんな数でも表せる)、また `。` が単語にくっつかないようにするため。

### 5.3 `pretokenize`

```python
def pretokenize(text: str) -> list[str]:
    """Cut text into script-class runs. Concatenating the result gives the text back exactly."""
    pieces: list[str] = []
    current = ""
    previous = ""
    for ch in text:
        cls = char_class(ch)
        if current and cls == previous and cls in _RUN_CLASSES:
            current += ch
        else:
            if current:
                pieces.append(current)
            current = ch
        previous = cls
    if current:
        pieces.append(current)
    return pieces
```
- `pieces` — 完成した断片。`current` — 今作っている断片。`previous` — 直前の文字の種類。
- 各文字について、種類(`cls`)を調べる。
  - **もし**断片を作っている最中で、**かつ**種類が直前と同じで、**かつ**その種類が連続を許されているなら → その文字を `current` に足す。
  - **そうでなければ** → 断片はここで終わり:`current` を(空でなければ)保存し、この文字で新しい断片を始める。
- `previous = cls` — 次の文字のために覚えておく。
- ループの後、最後の断片を保存する。
- **ロスレス:** 何も削ったり変えたりしないので、断片をつなぐと元の文になる。
- 例:`600万円` → `6`(数字、単独)| `0` | `0` | `万円`(漢字の連続)。

---

## 6. トークナイザーのクラス

### 6.1 作る:`__post_init__`

```python
@dataclass
class ByteBPETokenizer:
    """A trained tokenizer: an ordered list of merges plus the fixed ID layout."""

    merges: list[Pair]
```
- データクラス(Day 4を参照)で、項目は1つだけ:**順番付きの結合リスト**。他はすべて、これから導かれる。

```python
    def __post_init__(self) -> None:
        self.ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [s.encode("utf-8") for s in SPECIAL_TOKENS]
        self.token_bytes += [f"<unused_{i}>".encode("utf-8") for i in range(NUM_RESERVED)]
        self.token_bytes += [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self._cache: dict[str, list[int]] = {}
```
- `__post_init__` は、データクラスが `merges` を入れた**直後に自動で**実行される。ここで補助のデータを作る:
- `self.ranks` — 引き表:組 → ランク(結合リストの中での位置)。`enumerate` が位置 `i` と組をくれる。「この組は結合できるか、どれだけ早く学んだか」をすばやく調べるのに使う。
- `self.token_bytes` — 「ID → そのトークンが表すバイト列」のリスト。**IDの順に**作るので、リストの添字 = トークンID:
  1. 4つの特殊な名前をバイトにしたもの(ID 0〜3)。
  2. `b"<unused_0>"` のような100個の予約名(ID 4〜103)。
  3. 256個の1バイト:`bytes([b])` は b = 0…255 の1バイトの値を作る(ID 104〜359)。
  4. 各結合 `(a, b)` について:新しいトークンのバイト列は、トークン `a` のバイト列**足す**トークン `b` のバイト列(`self.token_bytes[a] + self.token_bytes[b]`)。だから結合の k 番目は、自動的にID `360 + k` になる。
- `self._cache` — `_encode_piece` 用のメモ帳(断片の文字列 → そのトークンID)。先頭の `_` は「内部用」の印。

### 6.2 大きさ

```python
    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)
```
- `@property` により、`tok.vocab_size` を**かっこなしで**、変数のように書ける。値 = 360 + 結合の数。

```python
    def truncated(self, vocab_size: int) -> "ByteBPETokenizer":
        n = max(0, min(len(self.merges), vocab_size - FIRST_MERGE_ID))
        return ByteBPETokenizer(self.merges[:n])
```
- 最初の `n` 個の結合だけを残した、**小さい版**を返す。
- `n = max(0, min(len(self.merges), vocab_size - 360))` — 指定の大きさになるのに必要な結合の数。ただし負にならず、持っている数を超えない。
- `self.merges[:n]` — スライス:最初の `n` 個の結合。
- docstringの肝:結合は順番に学ぶので、最初の N 個の結合は、学習がその大きさで止まっていたら作られていたトークナイザー**そのもの**。だから**1回の学習で、より小さい大きさがすべてただで手に入る**(Day 6が使う)。
- 型が引用符付き `"ByteBPETokenizer"` なのは、その時点ではクラスがまだ定義の途中だから。

### 6.3 断片1つを符号化:`_encode_piece`

```python
    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
```
- まずメモ帳を見る。この断片を前に符号化していたら、保存された答えを返す。

```python
        ids = [BYTE_OFFSET + b for b in piece.encode("utf-8")]
```
- 断片をバイトにし、各バイトをトークンID(`104 + バイト`)にする。例:`万`(3バイト)→ ID3つ。これが「小さなブロック」の出発状態。

```python
        while len(ids) > 1:
            best_rank = None
            best_at = -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_at = r, i
            if best_rank is None:
                break
            ids[best_at:best_at + 2] = [FIRST_MERGE_ID + best_rank]
```
- トークンが2つ以上ある間、繰り返す:
  - `best_rank` / `best_at` — ここまでに見つけた、一番早く学んだ結合可能な組と、その位置。
  - 隣り合う組 `(ids[i], ids[i+1])` を全部見る。`ranks.get(...)` はそのランクを返し、学んだ結合でなければ `None`。
  - **一番ランクが小さい**組(`r < best_rank`)を残す —「一番古いレシピを先に使う」。同点なら最初(一番左)のものが残る。比較が厳密な `<` だから。
  - 結合できる組がなければ(`best_rank is None`)→ 完了、`break`。
  - あれば、その位置の2つのトークンを、ID `360 + ランク` の**1つの**トークンに置き換える。`ids[best_at:best_at+2] = [x]` は**スライスへの代入**で、2つを1つに置き換える。
- これは学習と同じ順番を再生するので、結果は学習が「想定した」ものと一致する。

```python
        self._cache[piece] = ids
        return ids
```
- 答えをメモ帳に保存して返す。

### 6.4 `encode`

```python
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text to token IDs. Special-token strings in the text are treated as ordinary text."""
        ids: list[int] = [BOS_ID] if add_bos else []
        for piece in pretokenize(text):
            ids.extend(self._encode_piece(piece))
        if add_eos:
            ids.append(EOS_ID)
        return ids
```
- 指定があれば `<bos>` から始め、無ければ空のリストから始める。
- `pretokenize` で文を切り、各断片を符号化して、IDをつなげる(`extend`)。
- 指定があれば、最後に `<eos>` を足す。
- docstringの大事な点:文の中に `<bos>` という文字が含まれていても、特殊トークンでは**なく**、普通の文字として符号化される。(つまり、入力に打ち込んで特殊トークンを注入することはできない。)

### 6.5 `decode`

```python
    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Token IDs to text. Special and reserved IDs are skipped by default."""
        out = bytearray()
        for i in ids:
            if i < BYTE_OFFSET:
                if skip_special:
                    continue
            out += self.token_bytes[i]
        return out.decode("utf-8", errors="replace")
```
- `out = bytearray()` — 伸ばせるバイトの入れ物。
- 各IDについて:104未満(特殊または予約)**かつ** `skip_special` が真なら、飛ばす(`continue`)。
- そうでなければ、そのトークンのバイト列(`token_bytes[i]`)を追加する。`skip_special` が偽なら、特殊トークンも名前のバイト列が加わる。
- 最後に、全バイトをUTF-8で文字列に戻す。`errors="replace"` は、バイト列が正しいUTF-8でない場合(たとえば文字の途中)に、落ちずに `�` を入れるという意味。

### 6.6 `pieces` — 人間の目で見るため

```python
    def pieces(self, text: str) -> list[str]:
        result: list[str] = []
        for i in self.encode(text):
            raw = self.token_bytes[i]
            try:
                result.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                result.append("<" + raw.hex() + ">")
        return result
```
- トークンを読める文字で見せる(デモ用)。
- トークンが完全なUTF-8の文字列なら、そのまま見せる。文字の一部だけで単独では正しくないなら、バイトを16進数で `<e5ae>` のように見せる。

### 6.7 保存と読み込み

```python
    def to_dict(self) -> dict[str, object]:
        return {
            "format": FORMAT_VERSION,
            "pre_tokenizer": PRETOKENIZER_NAME,
            "special_tokens": {name: i for i, name in enumerate(SPECIAL_TOKENS)},
            "reserved": {"prefix": "<unused_", "first_id": RESERVED_FIRST_ID, "count": NUM_RESERVED},
            "byte_offset": BYTE_OFFSET,
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for a, b in self.merges],
            "vocab_preview": {
                str(i): self.token_bytes[i].decode("utf-8", errors="replace")
                for i in range(FIRST_MERGE_ID, self.vocab_size)
            },
        }
```
- JSONファイルになる辞書を作る:
  - 2つの名札、特殊トークンのID、予約の情報、`byte_offset`、`vocab_size`;
  - `merges` — 本当のデータ(JSONにタプルは無いので、組は2要素のリスト);
  - `vocab_preview` — 学習したトークンの「ID → 文字」の、**人間が読める**表(JSONのキーは文字列でなければならないので `str(i)`)。人が読むためだけのもので、`load` は無視する。

```python
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": the same bytes on Windows and Linux, so the SHA-256 is the same on every PC
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
```
- フォルダーがあることを確認してから、JSONを書く。
- `ensure_ascii=False` で日本語を読める形で残す(Day 1・4・5と同じ)。`indent=1` は1スペースのインデントで見やすくする。
- `newline="\n"` — 常にLFの改行にする(Day 5の `write_lines` と同じ考え方)。だからファイルのSHA-256が、**学校PCと自宅ノートPCで同一**になる。

```python
    @classmethod
    def load(cls, path: Path) -> "ByteBPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_VERSION or data.get("pre_tokenizer") != PRETOKENIZER_NAME:
            raise ValueError(f"{path}: not a {FORMAT_VERSION} / {PRETOKENIZER_NAME} tokenizer file")
        if data.get("byte_offset") != BYTE_OFFSET:
            raise ValueError(f"{path}: unexpected ID layout")
        tok = cls([(int(a), int(b)) for a, b in data["merges"]])
        if tok.vocab_size != data.get("vocab_size"):
            raise ValueError(f"{path}: vocab_size does not match the merges")
        return tok
```
- `@classmethod` — クラス自身に属する関数(`ByteBPETokenizer.load(path)` と呼ぶ)で、クラスを `cls` として受け取る。
- JSONを読み、**3つの安全確認**をする:形式とプリトークナイザーの名前が正しいか、IDの並びが正しいか、保存された `vocab_size` が結合の数と合うか。失敗すれば、分かりやすいメッセージ付きの `ValueError`。
- 結合からトークナイザーを作り直す。各 `[a, b]` のリストを、整数の `(a, b)` タプルに戻す。

---

## 7. 学習

### 7.1 `TrainingLog`

```python
@dataclass
class TrainingLog:
    unique_pieces: int
    total_pieces: int
    possible_merges_upper_bound: int
    merges_learned: int
    stop_reason: str
    last_merge_count: int
```
- 学習の成績表:異なる断片がいくつあったか、断片が全部でいくつか、理論上の結合の最大数、実際に学んだ数、なぜ止まったか、最後に結合した組が何回出てきたか。

### 7.2 `count_pieces`

```python
def count_pieces(lines: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        counts.update(pretokenize(line))
    return counts
```
- 学習用の各行を断片に切り、カウンターに足す。`Counter.update(リスト)` は、項目ごとに1を足す。結果は「各断片が何回出るか」、例:`{"営業部": 530, "の": 4120, ...}`。
- 学習は、文章全体ではなく、この**重み付きの重複なしの断片**に対して行う → ずっと速い。

### 7.3 `train_bpe` — 心臓部

```python
def train_bpe(piece_counts: Counter[str], max_merges: int, min_frequency: int = 2) -> tuple[list[Pair], TrainingLog]:
```
- 入力:断片の数、結合の数の安全上限(`max_merges`)、結合するために組が最低何回出る必要があるか(`min_frequency`)。出力:結合のリストと成績表。

```python
    pieces = sorted(piece_counts)  # sorted: deterministic order
    words: list[list[int]] = [[BYTE_OFFSET + b for b in p.encode("utf-8")] for p in pieces]
    weights: list[int] = [piece_counts[p] for p in pieces]
    upper_bound = sum(max(0, len(w) - 1) for w in words)
```
- `pieces` — 重複なしの断片を並べ替えたもの(学習が決定的になる)。
- `words` — 各断片を、バイトIDのリストにしたもの(「小さなブロック」の状態)。
- `weights` — 各断片が何回出るか。
- `upper_bound` — 結合の最大数:結合は1回ごとに全体の長さを最低1減らすので、`sum(長さ − 1)` を超えることはない。

```python
    pair_counts: Counter[Pair] = Counter()
    where: dict[Pair, set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += weights[wi]
            where[pair].add(wi)
```
- 隣り合う組を全部数える。断片の出現回数で**重み付け**する。`zip(w, w[1:])` は、各項目を次の項目と組にする。
- `where[pair]` — この組を含む断片の番号の集合。これで後から、全部を数え直さずに、**影響を受けた断片だけ**を更新できる。

```python
    merges: list[Pair] = []
    stop_reason = f"reached max_merges={max_merges}"
    last_count = 0
    while len(merges) < max_merges:
```
- 結合リストは空から始める。`stop_reason` は初期値を持つ(早く止まれば書き換える)。ループは安全上限まで回る。

```python
        if not pair_counts:
            stop_reason = "no pairs left (every piece is a single token)"
            break
        best, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if count < min_frequency:
            stop_reason = f"most frequent pair occurs fewer than {min_frequency} times"
            break
```
- 組がもう無ければ止まる。
- **一番良い組**を選ぶ:`key` 付きの `max(...)`。各項目 `kv = (組, 回数)` について、キーは `(回数, -1つ目のID, -2つ目のID)`:
  - **回数**が一番多いものが勝つ;
  - 同点なら、**トークンIDが小さい**組が勝つ(マイナスにすると、小さいIDのほうが上位になる)。だから同点の決め方がいつも同じ → **決定的**。
- 一番良い組でも `min_frequency` 回未満なら止まる。(まれな組は、学習データを丸暗記するだけだから。)

```python
        new_id = FIRST_MERGE_ID + len(merges)
        merges.append(best)
        last_count = count
```
- 新しいトークンは、次の空きID(360、361、…)を受け取る。結合と回数を記録する。

```python
        for wi in sorted(where.pop(best, set())):
            w = words[wi]
            for pair in zip(w, w[1:]):          # remove this word's old pairs
                pair_counts[pair] -= weights[wi]
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair != best:
                    where[pair].discard(wi)
```
- その一番良い組を含む各断片について(`where.pop` が集合を取り出して項目を消す。`sorted` は決定的にするため):
  - **古い組の回数を取り除く**:断片の各組について重みを引き、0になった項目は消す。
  - この断片を、他の組の `where` の集合からも取り除く(`discard` = 無くてもエラーにならずに削除)。

```python
            merged: list[int] = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(w[i])
                    i += 1
            words[wi] = merged
```
- 断片の結合後の版を、左から右へ見て作る:一番良い組がある所は、新しいIDを入れて2つ進み、そうでなければ1つのトークンをそのままコピーする。(`encode` の「一番左から先に」の動きと一致する。)

```python
            for pair in zip(merged, merged[1:]):  # add its new pairs
                pair_counts[pair] += weights[wi]
                where[pair].add(wi)
```
- 結合後の断片の**新しい組の回数を足す**(新しいトークンを含む組も含む)。
- この「古いものを取り除き、新しいものを足す」更新は**差分更新**:結合した組を含む断片だけを数え直し、コーパス全体は数え直さない。

```python
    log = TrainingLog(len(pieces), sum(weights), upper_bound, len(merges), stop_reason, last_count)
    return merges, log
```
- 成績表を作り、結合と一緒に返す。

### 7.4 小さな実例(バイトの代わりに文字で)

断片:`hug` ×10、`pug` ×5、`pun` ×12。(実際のコードでは、文字はバイトのIDです。)

| ラウンド | 組の回数(上位) | 結合 | 結果 |
|---|---|---|---|
| 1 | (p,u)=17、(u,g)=15、(u,n)=12、(h,u)=10 | **p+u → "pu"** | hug、[pu]g、[pu]n |
| 2 | (pu,n)=12、(h,u)=10、(u,g)=10、(pu,g)=5 | **pu+n → "pun"** | hug、[pu]g、[pun] |
| 3 | (h,u)=10 と (u,g)=10 が同点 → 1つ目のIDが小さいほうが勝つ("h" < "u") | **h+u → "hu"** | [hu]g、… |
| 4 | (hu,g)=10 | **hu+g → "hug"** | [hug]、… |

結合のリストは、順番付きのレシピ `[pu, pun, hu, hug, …]` になる。新しい単語は、これを順番に当てはめ直して符号化する。

---

## 8. 知っておくと良い小さなこと

- **Day 5とDay 6の数字の扱い。** Day 5の `pretoken_chunks` は `600` のような数字の連続を**1つ**の塊と数えるが、この本物のプリトークナイザーは**3つ**の断片に分ける。だからDay 5の「トークンの下限」は、下限としては正しいが、数字を含む文では本当の最小値より少しゆるい。
- **`decode` はIDを確認しない。** 語彙に無いID(例:99999)は、Pythonの `IndexError` になる。普通に使えば起こらない。
- **キャッシュはトークナイザーのオブジェクトごと。** `truncated()` は空のキャッシュを持つ新しいオブジェクトを作る。これは正しい(別のトークナイザーが答えを使い回してはいけない)が、大きさごとにキャッシュを温め直すことになる。
- **学習の速さ。** 差分更新は速いが、`max(pair_counts.items(), ...)` は結合のたびに全部の組を調べる。この規模のコーパスなら問題ない。
- **`min_frequency = 2` が大事な理由。** 1回しか出ない組を結合すると、丸暗記しただけの役に立たないトークンができてしまう。

## 9. 一段落でまとめ

`bpe_tokenizer.py` は、文を同じ種類の文字の連続に切り、各塊をUTF-8のバイト列にして(だからどんな文も表せる)、学んだ結合の順番付きリストを当てはめ直して、より大きなトークンを作ります。正確に復号でき、JSONとして保存・読み込みでき、最初の結合だけを残して小さい版を作れ、さらに、完全に決定的な差分更新のアルゴリズムで、断片の数から結合を学びます。
