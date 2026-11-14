# Day 9.1 — batching.py — やさしい1行ずつの解説(日本語)

> **Day 9 の5つのファイルの関係**
> - **Day 9.1(このファイル)`batching.py`** — Day 7のトークンファイルから、**学習用・検証用のバッチ**を切り出す。
> - **Day 9.2 `training.py`** — **学習ループ**(Day 9の心臓部)。このファイルを使う。
> - **Day 9 `day9_train.py`** — 学習を始める/再開するコマンドラインのスクリプト。
> - **Day 9 `day9_compare_runs.py`** — 中断しなかった学習と、止めて再開した学習を比べる。
> - **Day 9.3 `test_training.py`** — 9.1と9.2の自動テスト。
>
> おすすめの読む順番:**9.1 → 9.2 → 9(train)→ 9(compare)→ 9.3**。

---

## 0. 全体像(まずここから)

モデルは、1つのゲームを何百万回も繰り返して学びます:**「文章があります。各位置で、次のトークンを当ててください。」**このファイルは、そのゲームの**練習プリント**を用意します。

Day 7のトークンの列を思い出してください:`<bos> 質問 <eos> <bos> 質問 <eos> …`(とても長い数の1列)。

### 学習の例1つ = 「窓」1つ

列から `block_size + 1` 個の連続したトークンを切り出します(ブロックサイズ256なら257トークン):

```
窓 :  t0  t1  t2  t3  ...  t255  t256
x  :  t0  t1  t2  t3  ...  t255            (モデルが読むもの)
y  :      t1  t2  t3  ...  t255  t256      (各位置の次のトークン)
```
位置 `t0` ではモデルは `t1` を、`t1` では `t2` を予測しなければなりません。つまり**窓1つで、256個の「次のトークン」の練習問題が一度に**できます。(Day 8の `real_batch` と同じ考え方。)

### 2種類のバッチ

| 種類 | 窓の始まる位置 | 理由 |
|---|---|---|
| **学習用** | **ランダムな**位置 | ステップごとに違う窓を見るので、モデルが順番を丸暗記しない |
| **検証用** | **決まった**、並んだ位置で、列全体を覆う | 評価のたびにまったく同じものを測るので、数字を時間を通して比べられる |

### なぜ乱数の状態を保存しなければならないか

学習を止めて再開したとき、**中断しなかった場合とまったく同じバッチ**で続けなければなりません。だから、乱数生成器の内部の状態を、チェックポイントごとに保存します(`state_dict` / `load_state_dict`)。

---

## 1. 先に知っておきたい用語

| 用語 | かんたんな意味 |
|---|---|
| **窓(window)** | トークンの列から切り出した短い部分(`block_size + 1` 個のトークン)。 |
| **バッチ** | まとめて処理する、いくつかの窓(`batch_size`)。 |
| **メモリマップ** | ファイルをRAMに読み込まずに開く。使う窓だけが読まれる(Day 7.1)。 |
| **乱数生成器(RNG)の状態** | 生成器の内部の「列の中の位置」。保存して戻すと、次の乱数が**そっくり繰り返される**。 |
| **ジェネレーター関数 / `yield`** | リスト全体を作る代わりに、結果を**1つずつ**渡す関数。 |

---

## 2. docstringとimport

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.token_data import TOKEN_DTYPE, read_meta
```
- `Path`、`numpy`、`torch` — 前と同じ。`Iterator` — 「1項目ずつ回せるもの」を表す型(下のジェネレーターに使う)。
- **Day 7.1**から:`TOKEN_DTYPE`(リトルエンディアンの `uint16`)と `read_meta`(`*_meta.json` のラベルを読む)。

---

## 3. `open_tokens`

```python
def open_tokens(tokens_dir: Path, split: str) -> np.ndarray:
    """The token stream of one split as a read-only memory map (checked against its meta file)."""
    meta = read_meta(tokens_dir, split)
    tokens = np.memmap(tokens_dir / f"{split}.bin", dtype=TOKEN_DTYPE, mode="r")
    if len(tokens) != meta.tokens:
        raise ValueError(f"{split}.bin does not match {split}_meta.json")
    return tokens
```
- 分割のメタのラベルを読み、`train.bin`(または `val.bin`)を、Day 7.1とまったく同じ型で**読み取り専用のメモリマップ**として開き、ファイルのトークン数がラベルと同じか確認する(違えば `ValueError`)。
- メモリマップを直接開いたDay 8と比べて:これは、安全確認が組み込まれた**ライブラリ版**。

---

## 4. `RandomWindows`

```python
class RandomWindows:
    """Random training windows with a saveable random state."""

    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int, seed: int) -> None:
        if len(tokens) <= block_size + 1:
            raise ValueError("token stream is shorter than one window")
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
```
- ランダムな学習用のバッチを渡す、小さなクラス。
- 窓1つ分にも足りない短い列は弾く。
- 列、大きさ、そして**シード付きのNumPyの乱数生成器** `self.rng`(Day 8と同じ `default_rng`)を保存する。

```python
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.block_size - 1, size=self.batch_size)
        win = np.stack([np.asarray(self.tokens[s:s + self.block_size + 1], dtype=np.int64) for s in starts])
        w = torch.from_numpy(win)
        return w[:, :-1], w[:, 1:]
```
- **Day 8の `real_batch` とまったく同じ処理**を、クラスの中に入れたもの:
  1. `batch_size` 個のランダムな開始位置を選ぶ;
  2. 各開始位置から `block_size + 1` 個のトークンを切り出す(メモリマップから64ビット整数としてコピーもする);
  3. `(batch, block_size+1)` の表に積み、PyTorchのテンソルにする;
  4. `x = w[:, :-1]` と `y = w[:, 1:]` を返す。
- 呼び出すたびに生成器の乱数が消費されるので、**次の呼び出しは別の窓**を返す。
- (窓は、ある質問から次の質問にまたがることがあるが、問題ない。)

```python
    def state_dict(self) -> dict:
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["bit_generator"]
```
- `state_dict` — 生成器の**内部の状態**(普通の数の辞書)を返す。チェックポイントに保存できるように。
- `load_state_dict` — 保存した状態を戻す。その後、生成器は、元のものが出したはずの**まったく同じ将来の数**を出す。`state_dict` / `load_state_dict` という名前は、PyTorchの保存の決まりに倣っている。

---

## 5. 検証用の窓

```python
def validation_windows(tokens: np.ndarray, block_size: int) -> int:
    """Number of fixed validation windows (the last incomplete piece is not used)."""
    return (len(tokens) - 1) // block_size
```
- **丸ごと**入る窓の数:`(長さ − 1) // block_size`。`−1` は、各窓が `y` のために余分に1トークン必要だから。窓より短い余りは使わない。例:100トークン、ブロック10 → `99 // 10 = 9` 個の窓。

```python
def validation_batches(tokens, block_size, batch_size, max_windows=None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping windows covering the stream from the start, in batches."""
    n = validation_windows(tokens, block_size)
    if max_windows is not None:
        n = min(n, max_windows)
    for first in range(0, n, batch_size):
        idx = range(first, min(first + batch_size, n))
        win = np.stack([np.asarray(tokens[i * block_size:i * block_size + block_size + 1], dtype=np.int64)
                        for i in idx])
        w = torch.from_numpy(win)
        yield w[:, :-1], w[:, 1:]
```
- `n` = 使う窓の数。`max_windows`(任意)で上限を付け、速い部分的な評価にできる。
- `for first in range(0, n, batch_size)` は、窓を `batch_size` ずつ進む。`idx` = このバッチの窓の番号(最後のバッチは小さいことがある)。
- **窓 `i` はトークン `i × block_size` から始まる**ので、`x` の部分は**隙間も重なりもなく並ぶ**:窓0はトークン0〜9、窓1は10〜19、…(元のスライスが1トークン長いのは、`y` を1つずらすためだけ)。
- `yield` — バッチ `(x, y)` を1つ渡して**一時停止**する。呼び出し側のループが次を求める。これが**ジェネレーター**であるということ:バッチは、必要なときに作られ、一度に全部は作られない。
- 窓が固定なので、**評価のたびに同じ文章を測り**、前の評価と比べられる。

---

## 6. 知っておくと良い小さなこと

- **学習用の窓は、重複を許したランダム。**「エポック」の区切りは無く、偶然、ある トークンが他より多く見られる。Day 9.2は、`tokens_seen ÷ 学習トークン数` を「データを何周したか」として報告する。
- **検証用は、各トークンを多くても1回だけターゲットにする**が、最後の数トークン(窓1つ分に満たない部分)は無視される。
- **`next_batch` はDay 8の `real_batch` と同じ内容**(同じ考え、同じコード)。Day 8が先に書かれたため。
- **ディスク上のファイルと同じ型:**`TOKEN_DTYPE` が、メモリマップが、Day 7.1の書いたリトルエンディアンの `uint16` を読むことを保証する。

## 7. 一段落でまとめ

`batching.py` は、Day 7のトークンの列を、次のトークンの練習問題に変えます:学習用は、ランダムな窓(`x` = トークン、`y` = 同じトークンを1つずらしたもの)で、乱数生成器の状態を保存・復元できるので、再開した学習がまったく同じバッチを見ます。検証用は、列全体を覆う、並んだ固定の窓なので、評価のたびに同じものを測ります。
