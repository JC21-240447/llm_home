# Day 7.2 — test_token_data.py — やさしい1行ずつの解説(日本語)

> **Day 7 の4つのファイルの関係**
> - **Day 7.1 `token_data.py`** — ファイル形式のライブラリ(**先に読んでください**)。
> - **Day 7 `day7_encode_corpus.py`** と **`day7_tokenizer_demo.py`** — スクリプト。
> - **Day 7.2(このファイル)`test_token_data.py`** — *検査員*:Day 7.1が正しいことを確かめる自動テスト。
>
> pytestの基本(テスト関数、`assert`、フィクスチャ、`tmp_path`、`skip`)は **Day 6.2** で説明したので、このファイルでは新しい部分だけを説明します。

---

## 0. 全体像(まずここから)

トークンファイルが間違って書かれると(バイトの順番が違う、質問が欠けている、件数が合わない)、Day 9の学習は、静かに**ごみ**から学んでしまいます。このテストは、ファイル形式のライブラリが約束どおりに動くことを確かめます:

| 約束 | テスト |
|---|---|
| 列が、質問ごとに `<bos> 質問 <eos>` の形に並び、質問に切り分け直せる | `test_stream_layout` |
| ファイルを書いて読み直す(メモリマップの有無にかかわらず)と**同一**のデータになり、ファイルが本当に**リトルエンディアン** | `test_files_round_trip` |
| サイズがメタのラベルと合わないファイルは**弾かれる** | `test_size_mismatch_is_rejected` |
| **本物の**検証用のトークンファイルが、本物のコーパスと一致する | `test_real_token_files_match_corpus` |

実行(Day 6.2と同じ):

```
.venv\Scripts\python.exe -m pytest tests -q
```

---

## 1. 新しい用語

| 用語 | かんたんな意味 |
|---|---|
| **`pytest.raises(ValueError)`** | テストの道具:「このコードは `ValueError` を**出さなければならない**」。出さなければ、テストは**失敗**する。悪い入力が弾かれることを確かめるのに使う。 |
| **`np.array_equal(a, b)`** | 2つのNumPy配列の形と値が同じなら `True`。 |
| **`np.asarray(x)`** | `x` を普通のNumPy配列として扱う(メモリマップを、比べるために普通の配列にする)。 |
| **`bytes[:2]`** | バイト列の最初の2バイト。 |

(フィクスチャ、`scope="module"`、`tmp_path`、`skip` はDay 6.2と同じ。)

---

## 2. docstring、import、準備

```python
"""Unit tests for src/token_data.py (Day 7).
...
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- Day 6.2と同じ形式:rawでないdocstring(だからバックスラッシュは `\\` と2つ)、`sys`、`Path`、`pytest`、`REPO_ROOT` / `sys.path` の技。新しいのは `numpy as np`(Day 7.1と同じ)。

```python
from src.bpe_tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer, count_pieces, train_bpe  # noqa: E402
from src.token_data import (  # noqa: E402
    FORMAT_VERSION, TOKEN_DTYPE, TokenFileMeta, encode_lines, read_token_files, sha256_bytes, split_stream, write_token_files,
)
```
- トークナイザー(Day 6.1)から:印のID、トークナイザーのクラス、学習の補助関数(小さなトークナイザーを作るため)。ライブラリ(Day 7.1)から:テストが試すものすべて。

---

## 3. テスト用のデータとフィクスチャ

```python
LINES = ["営業部の社員は何人ですか。", "開発部の平均年収を教えてください。", "今日は晴れ😀", "A1"]
```
- 4つのテスト用の文:普通の質問2つ、**絵文字**付きの1つ、英字と数字だけのごく短い1つ。

```python
@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    merges, _ = train_bpe(count_pieces(LINES * 5), 10_000, 2)
    return ByteBPETokenizer(merges)
```
- Day 6.2と同じ形のフィクスチャで、小さなトークナイザーを**1回だけ**作る:4つの行を5回繰り返したものが学習用のテキスト。`train_bpe(counts, 10_000, 2)` は `max_merges` と `min_frequency` を位置で渡している。以下のテストはすべて、引数に `tok` と書くことで、これを受け取る。

---

## 4. テスト

### 4.1 `test_stream_layout`

```python
def test_stream_layout(tok: ByteBPETokenizer) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    assert tokens.dtype == TOKEN_DTYPE and len(offsets) == len(LINES)
    parts = split_stream(tokens, offsets)
    for line, ids in zip(LINES, parts):
        assert ids[0] == BOS_ID and ids[-1] == EOS_ID
        assert tok.decode(ids.tolist()) == line
    assert sum(len(p) for p in parts) == len(tokens)
```
- 4つの行を、列とオフセットに符号化する(Day 7.1)。
- トークンの配列は、**ぴったりリトルエンディアンのuint16の型**で、**行ごとにオフセットが1つ**あるはず。
- 列を質問に切り分け直す(`split_stream`)。元の各行と、その部分について:
  - `<bos>` で始まり、`<eos>` で終わる;
  - **元の行に復号される**。
- 最後の確認:すべての部分の長さの**合計が全体の長さと等しい** — 欠けも二重数えもない(隙間も重なりもない)。

### 4.2 `test_files_round_trip`

```python
def test_files_round_trip(tok: ByteBPETokenizer, tmp_path: Path) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    meta = TokenFileMeta(FORMAT_VERSION, "train", tok.vocab_size, "x", BOS_ID, EOS_ID, len(LINES), len(tokens),
                         sha256_bytes(tokens.tobytes()), "y")
    write_token_files(tmp_path, "train", tokens, offsets, meta)
```
- メタのラベルを用意する。`"x"` と `"y"` は仮の指紋(ライブラリは確認しない)で、トークンのバイト列の本物の指紋は `sha256_bytes(tokens.tobytes())` で計算する。
- `tmp_path`(pytest標準のフィクスチャ)は新しい一時フォルダーなので、このテストが本物の `data/tokens` に触れることはない。
- そこに3つのファイルを書く。

```python
    for memmap in (True, False):
        t2, o2, m2 = read_token_files(tmp_path, "train", memmap=memmap)
        assert np.array_equal(np.asarray(t2), tokens) and np.array_equal(o2, offsets)
        assert m2 == meta
```
- ファイルを**両方の方法**(メモリマップと通常の読み込み)で読み直す。それぞれについて:
  - トークンとオフセットが元と**等しい**(`np.asarray` が、比べる前にメモリマップを普通の配列にする);
  - JSONから読んだメタが、書いたメタと**等しい**(データクラスは項目ごとに比べる)。

```python
    # little-endian on disk, whatever the machine
    raw = (tmp_path / "train.bin").read_bytes()
    assert int.from_bytes(raw[:2], "little") == BOS_ID
```
- **ディスク上のバイトの順番**を確かめる。ファイルを生のバイトとして読み、最初の2バイトを**リトルエンディアン**の数として解釈する:`BOS_ID`(1)と等しいはず。リトルエンディアンではバイトは `01 00`。ビッグエンディアンのファイルなら `00 01` = 256 になり、失敗する。これは「ファイルがWindowsとLinuxで同一」という約束を守るためのもの。

### 4.3 `test_size_mismatch_is_rejected`

```python
def test_size_mismatch_is_rejected(tok: ByteBPETokenizer, tmp_path: Path) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    meta = TokenFileMeta(FORMAT_VERSION, "val", tok.vocab_size, "x", BOS_ID, EOS_ID, len(LINES), len(tokens) + 1,
                         "z", "y")
    write_token_files(tmp_path, "val", tokens, offsets, meta)
    with pytest.raises(ValueError):
        read_token_files(tmp_path, "val")
```
- **わざと間違えた**ラベルを作る:`len(tokens) + 1` 個のトークンがあると主張する(1つ多い)。
- ファイルを書き、そして読もうとする。読み込み側は、本当のトークン数がラベルと違うことに気づいて、`ValueError` を出さなければならない。
- `with pytest.raises(ValueError):` — 中のコードがその例外を出したときだけ、テストが**合格**する。平気で読んでしまえば、テストは失敗する。(これはDay 7.1のセクション6.6の**安全装置**のテスト。)

### 4.4 `test_real_token_files_match_corpus`

```python
def test_real_token_files_match_corpus() -> None:
    """Runs only after Day 7 has written data/tokens/ (checks the real validation file)."""
    tok_path = REPO_ROOT / "data/tokenizer/tokenizer.json"
    out_dir = REPO_ROOT / "data/tokens"
    val_txt = REPO_ROOT / "data/corpus/val.txt"
    if not ((out_dir / "val.bin").exists() and tok_path.exists() and val_txt.exists()):
        pytest.skip("run Day 7 first")
    tok = ByteBPETokenizer.load(tok_path)
    tokens, offsets, meta = read_token_files(out_dir, "val")
    lines = [ln for ln in val_txt.read_text(encoding="utf-8").split("\n") if ln]
    assert meta.vocab_size == tok.vocab_size and meta.questions == len(lines)
    assert all(tok.decode(ids.tolist()) == ln for ln, ids in zip(lines, split_stream(tokens, offsets)))
```
- **本物の**ファイルでのテスト(Day 6.2の最後のテストと同じ)。本物の検証用のトークンファイル、トークナイザー、検証用のテキストのどれかが無ければ → `pytest.skip`(失敗ではなく**スキップ**として報告される)。
- あれば:本物のトークナイザーを読み込み、本物の検証用のトークンファイルを読み、本物の検証用の行を読み、次を確かめる:
  - **メタのラベル**が、トークナイザーの語彙サイズと行の数と一致する;
  - トークンファイルから切り出した**すべての質問**が、対応するテキストの行に復号される。

---

## 5. テストはいくつ?

| テスト | 実行回数 |
|---|---|
| `test_stream_layout` | 1 |
| `test_files_round_trip` | 1(中で両方の読み方を確認) |
| `test_size_mismatch_is_rejected` | 1 |
| `test_real_token_files_match_corpus` | 1(スキップされることがある) |
| **合計** | **4** |

期待される結果:`4 passed`、またはDay 7が本物のファイルを作る前なら `3 passed, 1 skipped`。(`tests` フォルダー全体を実行すれば、Day 6.2のテストも一緒に動く。)

## 6. このテストが確かめて*いない*こと

- `encode_lines` の **uint16のオーバーフロー**のエラー(65,535を超えるID)。
- `read_meta` が**間違った形式の名前**を弾くこと。
- **キーが足りない、または余分な**メタファイル。
- `tokens_sha256` / `tokenizer_sha256` の値(ライブラリは保存するだけで、確認はしない。Day 9が比べるべき)。

これらは、後から追加のテストとして足せます。

## 7. 一段落でまとめ

このテストは、トークンの列が質問ごとに `<bos> … <eos>` の形に並び、切り分け直して正確に復号できること、3つのファイルの書き込みと読み込み(メモリマップでも、そうでなくても)が、本当のリトルエンディアンの順番で同一のデータを返すこと、サイズがメタのラベルと食い違うファイルが弾かれること、そして、本物のファイルがあれば、本物の検証用のトークンファイルが本物の検証用のテキストと一致することを確かめます。
