# Day 8.1 — model.py — やさしい1行ずつの解説(日本語)

> **Day 8 の3つのファイルの関係**
> - **Day 8.1(このファイル)`model.py`** — *モデル本体*:Model A。PyTorchで1から書いた、小さなGPT型のTransformer。
> - **Day 8 `day8_model_check.py`** — *運転手*:3つの大きさでモデルを作り、時間を測り、実際のデータで確かめ、設定を保存し、報告書を書く。
> - **Day 8.2 `test_model.py`** — *検査員*:モデルの自動テスト。
>
> おすすめの読む順番:**8.1 → 8 → 8.2**。これはここまでのプロジェクトで一番大事なファイルです。セクション0をじっくり読んでください。

---

## 0. 全体像(まずここから)

Model Aは**文章の続きを作る機械**です。トークン番号の列(Day 6・7のトークナイザーが作ったもの)を渡すと、**各位置で**、次の質問に答えます:

> 「ここまで読んだことを踏まえて、**語彙のどのトークンが次に来そうか?**」

答えは**点数(ロジット)**のリストで、語彙のトークン1つにつき点数が1つです。学習(Day 9)で、*本当の*次のトークンに高い点数を付けるように教えます。生成(Day 10)で、点数を文章に変えます。

### 数の旅

```
トークンID  (batch, T)                 例: [1, 412, 380, ...]
   │  1. トークン埋め込み               各ID → d_model 個の数のベクトル(その「身分証」)
   ▼
ベクトル    (batch, T, d_model)
   │  2. N × ブロック                  各ブロック =  前のトークンを見る(Attention)
   │                                                 + 各位置で考える(MLP)
   ▼
ベクトル    (batch, T, d_model)
   │  3. 最後の RMSNorm                大きさを整える
   │  4. 出力層                        ベクトル → 語彙のトークンごとの点数
   ▼
ロジット    (batch, T, vocab_size)
```

### かんたんなたとえ

- **埋め込み** = どのトークンにも、それを表す**座標カード**(数のリスト)がある。
- **Attention(注意)** = 各単語で**前の単語を振り返り**、今どれが大事かを決める読者のようなもの(**先は絶対に覗けない**)。
- **MLP** = 各位置が、集めたことを**自分の中で考える**。
- **残差接続**(`x = x + ...`)= **ベルトコンベア**:各層は、ベルトに載っているものに*メモを足す*だけで、ベルトを捨てたりしない。
- **正規化** = 数を適度な大きさに保ち、何も爆発しないようにする。
- **RoPE** = ベクトルを*回転*させて、2つのトークンが**どれだけ離れているか**をモデルに伝える方法。
- **重み共有** = トークンを読み込むときと、点数を出力するときで、**同じ辞書**を使う。

### このファイルの部品

| 部品 | 何か |
|---|---|
| `ModelConfig` + `PRESETS` + `preset_config` | 設定(大きさ)と、3つの試す大きさ(tiny / small / medium) |
| `RMSNorm` | ベクトルの大きさを抑える |
| `rope_tables`、`apply_rope` | 回転による位置の情報 |
| `CausalSelfAttention` | 「前のトークンを見る」(手書きの版 + 速い版) |
| `MLP` | 「各位置で考える」 |
| `Block` | Attention 1つ + MLP 1つを、正規化と残差で包んだもの |
| `ModelA` | モデル全体 |
| `count_parameters`、`expected_parameters` | 学習できる数を数える(2つの方法で、照らし合わせる) |

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **Transformer** | GPT型のモデルが使う、ニューラルネットワークの設計。 |
| **デコーダーのみ / GPT型** | モデルは左から右にだけ読み、**次の**トークンを予測する。 |
| **埋め込み(embedding)** | トークンIDをベクトル(数のリスト)に変える表。 |
| **ベクトル / `d_model`** | 数のリスト。`d_model` はその長さ(モデルの「幅」)。 |
| **パラメータ** | モデルの中の、学習できる数。「100万パラメータ」= そういう数が100万個。 |
| **ロジット** | 生の点数。高いほど、モデルはそのトークンが次に来そうだと思っている。 |
| **ソフトマックス** | 点数を、合計が1になる確率に変える。 |
| **クロスエントロピー損失** | 本当の次のトークンに、モデルがどれだけ「驚くか」:`−log(そのトークンに与えた確率)`。低いほど良い。何も知らないモデル(一様な当て推量)の損失は `ln(語彙サイズ)`。 |
| **Attention(注意)** | 各位置が、他の位置の情報を、関連の強さで重みを付けて混ぜる。 |
| **Query / Key / Value(q, k, v)** | q = 「何を探している?」、k = 「何を提供できる?」、v = 「選ばれたら渡すもの」。一致の点数 q·k が、v をどれだけ取るかを決める。 |
| **ヘッド / `n_head`** | Attentionを複数回、並行して行う(複数の「ヘッド」)。それぞれ違うものを探す。 |
| **因果マスク(causal mask)** | **後ろの**位置を隠すルール。位置は、自分と前の位置しか見られない。 |
| **MLP** | 各位置に別々にかける、小さな2層のネットワーク。 |
| **GELU** | MLPの2つの層の間で使う、なめらかな「オン/オフ」の関数。 |
| **残差接続** | `x = x + f(x)`:入力を残して、層の結果を足す。 |
| **RMSNorm** | 各ベクトルの平均の大きさが1になるように調整する(その後、学習する倍率)。 |
| **RoPE** | 回転式の位置埋め込み(セクション5.2を参照)。 |
| **重み共有** | 埋め込みの行列と出力の行列が**同じ**行列。 |
| **バイアス(bias)** | 線形層で足される追加の定数。このモデルは使わない(`bias=False`)。 |
| **`block_size`** | モデルが一度に読む最長の列(トークン数)。 |
| **バッファ(buffer)** | モデルに属し(GPUにも一緒に移る)、**学習されない**テンソル。 |

---

## 2. docstring(説明文)

- これまでと同じ形式(英語 + 日本語)。モデルの役割と、入力から出力までの部品を説明し、そして**設計の選択**を挙げる:RoPE(学習する位置の表が無いので、1つの最大長に縛られない)、因果的なAttentionの2つの版(読みやすい「manual」と、速いもの)、RMSNorm、バイアスなし、4倍の幅のGELUのMLP、重み共有、そして残差への射影の**スケールした初期化**(`std / sqrt(2 × n_layer)`)。
- 「サーバーの注意」:PyTorchだけ → WindowsとLinux、CPUとGPUで同じ。

---

## 3. importとAttentionのモード

```python
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_MODES: tuple[str, ...] = ("fast", "manual")
```
- `from __future__`、`json`、`math`、`dataclass`/`asdict`/`field`、`Path` — これまでの日と同じ。
- **新しいもの:**`torch`(PyTorch。ディープラーニングのライブラリ)、`torch.nn as nn`(ネットワークの部品:`Linear`、`Embedding`、`Module` など)、`torch.nn.functional as F`(学習する部分を持たない関数:`softmax`、`gelu`、`cross_entropy` など)。
- `ATTENTION_MODES` — 許される2つのAttentionの版。

---

## 4. `ModelConfig` — すべての設定

```python
@dataclass
class ModelConfig:
    """All shape settings of the model. Saved next to every checkpoint (Day 9)."""

    vocab_size: int = 1674          # from data/tokenizer/tokenizer.json (Day 6)
    block_size: int = 256           # longest sequence the model reads at once (tokens)
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    mlp_ratio: int = 4              # MLP hidden size = mlp_ratio * d_model
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    attention: str = "fast"         # "fast" or "manual" (same result, see tests)
    name: str = field(default="custom")
```

| 設定 | 意味 |
|---|---|
| `vocab_size` | トークンの数(Day 6から)。初期値の1674は仮のもの。 |
| `block_size` | 最長の入力(256トークン)。 |
| `n_layer` | 積み重ねるブロックの数(深さ)。 |
| `n_head` | Attentionのヘッドの数。 |
| `d_model` | ベクトルの幅。 |
| `mlp_ratio` | MLPの隠れ層の幅は `mlp_ratio × d_model`(4倍)。 |
| `dropout` | 過学習を減らすため、学習中に数をランダムに切ること。`0.0` = オフ。 |
| `rope_base` | RoPEの回転の速さを決める定数(10000、普通の値)。 |
| `norm_eps` | RMSNormでゼロ割りを避けるために足す、ごく小さな数。 |
| `init_std` | ランダムな初期の重みの標準偏差(0.02)。 |
| `attention` | `"fast"` か `"manual"`。 |
| `name` | ラベル(`tiny`、`small`、…、または `custom`)。 |

```python
    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("head size must be even for RoPE")
        if self.attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")
```
- 作った直後に実行され(Day 6.1と同じ)、**不可能な設定を弾く**:
  - `d_model` はヘッドに均等に分けられること(各ヘッドは `d_model / n_head` 個の数を受け持つ);
  - **ヘッドの大きさは偶数**であること。RoPEは数を**2つずつの組**で回転させるため;
  - `attention` は既知のモードであること。

```python
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head
```
- `head_dim` = 各ヘッドが扱う数の個数(`d_model ÷ n_head`)。`//` は整数の割り算。

```python
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
```
- `to_dict` / `from_dict` — 普通の辞書との相互変換(`**data` が設定に展開される。Day 7.1の `read_meta` と同じ)。
- `save` / `load` — JSONファイルの書き込み/読み込み(LFの改行。前と同じ)。Day 9とDay 11が読む `configs/model_*.json` はこれ。

### 3つの試す大きさ

```python
PRESETS: dict[str, dict] = {
    "tiny":   {"n_layer": 4, "n_head": 4, "d_model": 128},
    "small":  {"n_layer": 6, "n_head": 4, "d_model": 256},
    "medium": {"n_layer": 8, "n_head": 6, "d_model": 384},
}
```
- **Day 11**用の、3つの候補のモデルの大きさ(どれを使うかは、そこで検証用の損失で決める)。
- セクション7の式で、語彙が1,674トークンのとき、パラメータ数はおよそ次のとおり:

| プリセット | 層 | ヘッド | 幅 | ヘッドの大きさ | パラメータ(約) |
|---|---|---|---|---|---|
| tiny | 4 | 4 | 128 | 32 | **100万** |
| small | 6 | 4 | 256 | 64 | **515万** |
| medium | 8 | 6 | 384 | 64 | **1,480万** |

(`expected_parameters` から私が計算したもの。Day 8の報告書が、本物の語彙での正確な数を表示します。)

```python
def preset_config(name: str, vocab_size: int, block_size: int = 256, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return ModelConfig(vocab_size=vocab_size, block_size=block_size, name=name, **{**PRESETS[name], **overrides})
```
- プリセットの名前から設定を作る。未知の名前なら、分かりやすいエラー。
- `{**PRESETS[name], **overrides}` は2つの辞書を合わせる。**上書きが勝つ**ので、`preset_config("small", 1674, dropout=0.1)` はdropoutだけを変える。そして `**` が、合わせた設定をキーワード引数として渡す。

---

## 5. 部品

### 5.1 `RMSNorm`

```python
class RMSNorm(nn.Module):
    """Scales each vector to a fixed average size (root mean square = 1), then by a learned gain.
    Computed in float32 for stability under bf16/fp16."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)
```
- PyTorchの層はどれも、`nn.Module` を**継承した**クラス。`super().__init__()` を最初に呼ぶ必要がある。`forward` は、層を呼んだときに実際の仕事をする関数。
- `nn.Parameter(torch.ones(dim))` — `dim` 個の数の、**学習できる**リストで、1から始まる(「倍率」)。
- `forward`:
  1. `x32 = x.float()` — 正確さのため32ビットで計算する(モデルは16ビットで動いているかもしれない)。
  2. `x32.pow(2).mean(dim=-1, keepdim=True)` — 最後の次元(各ベクトルを別々に)の**二乗の平均**。`rsqrt` = 1/√。つまり `normed = x / sqrt(mean(x²) + eps)` — 各ベクトルは、平均の大きさ(二乗平均平方根)が**1**になる。
  3. 学習する倍率をかけ、元の数の型に戻す(`.to(x.dtype)`)。
- なぜか? これが無いと、数が層ごとに大きくなったり小さくなったりして、学習が不安定になる。

### 5.2 RoPE:`rope_tables` と `apply_rope`

モデルはトークンの**順番**を知る必要があります(「Aの次にB」は「Bの次にA」と違う)。RoPEは、各トークンのqueryとkeyのベクトルを、位置に応じた角度で**回転**させて、それを実現します。

```python
def rope_tables(head_dim: int, length: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (length, head_dim/2) for rotary position embedding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()
```
- 回転の角度を**一度だけ**用意する。
- `torch.arange(0, head_dim, 2)` = 0、2、4、…(数の**組**1つにつき1項目)。`inv_freq` = `head_dim/2` 個の回転の速さのリストで、速いものから遅いものへ(`1 / base^(2i/head_dim)`)。
- `torch.arange(length)` = 位置 0、1、2、…。`torch.outer(位置, inv_freq)` は、すべての位置とすべての速さをかけ合わせる → 形が `(length, head_dim/2)` の表 `angles`:**各位置・各組の角度**。
- その角度のコサインとサインを返す。

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos[None, None], sin[None, None]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.to(x.dtype)
```
- `x` の形は `(batch, heads, T, head_dim)`。
- `x[..., 0::2]` は**偶数番目**の項目、`x[..., 1::2]` は**奇数番目**の項目を取る — つまり組 `(x[2i], x[2i+1])`。
- `cos[None, None]` は、大きさ1の次元を2つ足して、`(T, head_dim/2)` の表を `(batch, heads, T, head_dim/2)` に合わせる。
- 式 `(x1·c − x2·s, x1·s + x2·c)` は、各組をその角度だけ回転させる、標準的な**2次元の回転**。
- `torch.stack((..., ...), dim=-1).flatten(-2)` は、回転させた組を、元の交互の並びに戻す。
- **大事な性質(docstringにある):**回転の後、queryとkeyの内積は、2つの位置が**どれだけ離れているかだけ**に依存する — モデルが学ぶのは「3つ前」で、「絶対的な位置17」ではない。Day 8.2がこれをテストする。

### 5.3 `CausalSelfAttention`

```python
class CausalSelfAttention(nn.Module):
    """Each position mixes information from itself and EARLIER positions only."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head, self.head_dim, self.mode = cfg.n_head, cfg.head_dim, cfg.attention
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj.is_residual_out = True  # type: ignore[attr-defined]  # scaled init
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
```
- 設定(ヘッドの数、ヘッドの大きさ、モード)を覚える。
- `self.qkv` — query、key、value を**一緒に**作る**1つの**線形層(出力は `3 × d_model`。あとで分ける)。`nn.Linear(入力, 出力, bias=False)` = 学習できる行列をかける。
- `self.proj` — ヘッドの結果を `d_model` に混ぜ戻す、最後の線形層。
- `self.proj.is_residual_out = True` — この層についた**付箋**(「私は残差の流れに書き込む」)。`ModelA._init_weights` がこれを読んで、初期の重みを小さくする。(`# type: ignore` は、型チェッカーを黙らせるだけ。)
- `self.resid_drop` — dropout(`dropout = 0.0` のときは何もしない)。

```python
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_head, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```
- `b, t, d` = バッチの大きさ、列の長さ、幅。
- `self.qkv(x)` → `(b, t, 3d)`。`.split(d, dim=2)` は最後の次元を**大きさ d の3つ**に切る = q、k、v。
- `view(b, t, n_head, head_dim).transpose(1, 2)` は、幅 `d` の各ベクトルを、`head_dim` ずつの `n_head` 個に切り、ヘッドの次元を前に動かす → 形は `(b, heads, t, head_dim)`。(同じ行が、ジェネレーターで q、k、v に適用される。)
- RoPEは **q と k だけ**に適用する(位置は一致を見るために大事。value の v は、内容を運ぶだけ)。

```python
        if self.mode == "manual":
            y = self._manual(q, k, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_drop(self.proj(y))
```
- 同じことを計算する2つの方法:
  - `"manual"`:下の、読みやすい手順どおりの版 `_manual`;
  - それ以外は、PyTorchの**速い組み込み**の `scaled_dot_product_attention`(`is_causal=True` で因果マスクが組み込み済み)。dropoutは学習中だけ(`self.training`)。
- そしてヘッドを元に戻す:ヘッドの次元を戻し、`.contiguous()`(メモリの並びを整える)、`.view(b, t, d)`、`proj` とdropoutをかける。

```python
    def _manual(self, q, k, v):
        t = q.size(-2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)        # (b, h, t, t)
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))                      # hide later positions
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)                 # each row sums to 1
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return weights @ v
```
Attentionのレシピを、手順どおりに:
1. `scores = q @ kᵀ / √head_dim` — 位置の組(i, j)ごとに、「位置 i の **query** は、位置 j の **key** とどれだけ合うか?」。`@` は行列のかけ算。結果は、ヘッドごとの `(t, t)` の表。`√head_dim` で割ると、数が大きくなりすぎない。
2. `future = torch.triu(ones(t, t, bool), diagonal=1)` — **対角線より上**が `True` の表(= j が i より後 = 未来)。
3. `masked_fill(future, -inf)` — 未来の点数を**マイナス無限大**にするので、ソフトマックスの後は**重み0**になる。これが**因果マスク**:先を覗かない。
4. `softmax(..., dim=-1)` — 点数の各行を、**合計が1**になる重みに変える(float32で計算して、型を戻す)。
5. 重みにdropout(`dropout = 0` ならオフ)。
6. `weights @ v` — 各位置が、注目した位置の value の**重み付き平均**を取る。

### 5.4 `MLP`

```python
class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.down.is_residual_out = True  # type: ignore[attr-defined]
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.gelu(self.up(x))))
```
- **各位置に別々に**かける小さなネットワーク:広げる(`up`:d → 4d)、**GELU**をかける、元に狭める(`down`:4d → d)、dropout。(`down` にも同じ付箋。)
- Attentionは位置の**間**で情報を動かし、MLPは各位置の**中**で情報を処理する。

### 5.5 `Block`

```python
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))
```
- Transformerの1層:正規化が2つ、Attentionが1つ、MLPが1つ。
- `forward` = docstringの2行:
  - `x = x + Attention(RMSNorm(x))` — 正規化し、注意し、**結果を x に足す**(残差);
  - `x = x + MLP(RMSNorm(x))` — 正規化し、処理し、足す。
- 「**Pre-norm**」= 各サブ層の*前*に正規化をかける。残差 `x + ...` が、元の情報を全層でまっすぐ流し続け(ベルトコンベア)、深いモデルを学習できるようにする。

---

## 6. `ModelA` — モデル全体

```python
class ModelA(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying: one matrix, two uses
        cos, sin = rope_tables(cfg.head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
```
- `tok_emb` — 埋め込みの表:`vocab_size` 行 × `d_model` 列。行 `i` がトークン `i` のベクトル。
- `blocks` — `ModuleList`(PyTorchが知っているリスト。だからパラメータが見つかる)に入った、`n_layer` 個のブロック。
- `norm_f` — 最後のRMSNorm。
- `lm_head` — **出力層**:ベクトル(`d_model`)→ 語彙のトークンごとの点数(`vocab_size`)。その重みの行列の形は `(vocab_size, d_model)` — **埋め込みの表と、まったく同じ形**。
- `self.lm_head.weight = self.tok_emb.weight` — **重み共有**:トークンを読み込むときも、点数を書き出すときも、*同じ*行列を使う。利点:パラメータがずっと少なくなる(`d_model` に対して語彙が大きいとき、大きな節約)ことと、「トークンがどう見えるか」と「出力されやすさ」が自然につながること。
- `register_buffer(..., persistent=False)` — RoPEの表を**バッファ**として保存する:モデルと一緒にGPUへ移るが、学習はされず、`persistent=False` は、チェックポイントに**保存されない**という意味(いつでも計算し直せるから)。
- `self.apply(self._init_weights)` — 初期化の関数を、すべての下位の層に実行する。

```python
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            std = self.cfg.init_std
            if getattr(m, "is_residual_out", False):
                std /= math.sqrt(2 * self.cfg.n_layer)     # keeps the residual stream from growing with depth
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
```
- すべての重みに、**小さなランダムな初期値**(正規分布、平均0、標準偏差0.02)を与える。
- 「付箋」(`is_residual_out`:Attentionの `proj` とMLPの `down`)が付いた層は、標準偏差を **`√(2 × n_layer)` 分、小さく**する。各ブロックはベルトコンベアに2つの寄与を足すので、深いモデルでは、さもないとベルトの値が深さとともに大きくなる。これで、最初が安定する。
- `getattr(m, "is_residual_out", False)` は付箋を読み、無ければ `False`。
- (RMSNormの倍率は1のまま。共有された行列は、どちらの経路でも同じ標準偏差で初期化される。)
- 小さな初期の重みは、**初期の損失が `ln(語彙)` に近い**理由でもある:初期の点数がほぼすべて等しい、つまり一様な当て推量だから(Day 8がこれを確かめる)。

```python
    def set_attention(self, mode: str) -> None:
        if mode not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")
        self.cfg.attention = mode
        for blk in self.blocks:
            blk.attn.mode = mode
```
- すべての層のAttentionを、**同じ重みのまま** `fast` と `manual` で切り替える。2つが一致することを証明するために、テストとDay 8が使う。

```python
    def forward(self, idx, targets=None, ignore_index=-100):
        """idx: (batch, T) token numbers. Returns logits (batch, T, vocab) and, if targets are
        given, the average cross-entropy loss of predicting targets[:, t] at position t."""
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} is longer than block_size {self.cfg.block_size}")
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
```
- 入力 `idx`:トークンID、形は `(batch, T)`。
- `block_size` より長い列は弾く。
- RoPEの表の最初の `t` 行を取る(`[:t]`)。
- `self.tok_emb(idx)` — 各IDのベクトルを引く → `(batch, T, d_model)`。dropoutは任意。
- すべてのブロックを順番に通す(`for blk in self.blocks`)。
- 最後の正規化、そして出力層 → 形が `(batch, T, vocab_size)` の `logits`:**すべての位置**で、**すべてのトークン**の点数。

```python
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss
```
- `targets`(正しい次のトークン)があれば、**損失**を計算する:
  - `logits.view(-1, vocab)` は `(batch, T, vocab)` を、(位置ごとに1つの)行の長いリストに平らにする。`targets.reshape(-1)` も同じように平らにする;
  - `F.cross_entropy` はソフトマックスをかけ、各行で `−log(正しいトークンの確率)` を取り、平均する。(`.float()` = 32ビットで計算。)
  - `ignore_index=-100`:ターゲットが `-100` の位置は**飛ばされる**(あとでパディングに使う)。
- 両方を返す。`targets` が無ければ `loss` は `None`(生成で使う)。

**学習(Day 9)での使われ方:**入力はトークン `0..T-1`(`x`)、ターゲットはトークン `1..T`(`y`、1つずらしたもの)。だから位置 `t` で、モデルはトークン `t+1` を予測するよう求められる。

---

## 7. パラメータを数える

```python
def count_parameters(model: nn.Module) -> dict[str, int]:
    """Unique parameters (the tied matrix is counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():          # parameters() already yields shared tensors once
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    emb = model.tok_emb.weight.numel() if isinstance(model, ModelA) else 0
    return {"total": total, "embedding": emb, "non_embedding": total - emb}
```
- 学習できる数を数える。`p.numel()` はテンソルの要素数。`id(p)` はオブジェクトの同一性なので、共有された(重み共有した)テンソルは二重に数えられない(PyTorchの `parameters()` は元々1回だけ挙げる。この確認は安全のため)。
- `embedding` = 埋め込みの行列の大きさ(2回使われても1回だけ数える)、`non_embedding` = それ以外のすべて。

```python
def expected_parameters(cfg: ModelConfig) -> int:
    """The same count from the formula, written out (checked by a test)."""
    d, h = cfg.d_model, cfg.mlp_ratio * cfg.d_model
    per_block = (3 * d * d + d * d) + (d * h + h * d) + 2 * d   # attention + MLP + two norm gains
    return cfg.vocab_size * d + cfg.n_layer * per_block + d       # embedding (tied) + blocks + final norm
```
- 同じ数を、**手で書いた式で計算**したもの(照らし合わせ用):
  - Attention:`qkv` = `d × 3d`、`proj` = `d × d` → `3d² + d²`;
  - MLP:`up` = `d × h`、`down` = `h × d`(`h = 4d`);
  - RMSNormの倍率2つ = `2d`;
  - 加えて、埋め込み `vocab × d`(1回だけ)、`n_layer` 個のブロック、最後のnorm `d`。
- `h = 4d` なら、ブロックごとに `12d² + 2d`。バイアスも、学習する位置の表も無いので、他に数えるものはない。
- Day 8とDay 8.2は、どちらも `count_parameters(model) == expected_parameters(cfg)` を確かめる:コードと手書きの式が一致すれば、欠けや二重の数えがない可能性がとても高い。

---

## 8. 形の一覧(batch 32、T 256、tiny、語彙 1,674)

| 手順 | 形 |
|---|---|
| 入力ID | (32, 256) |
| 埋め込みの後 | (32, 256, 128) |
| Attentionの中:q、k、v | それぞれ (32, 4, 256, 32) |
| Attentionの点数の表 | 層ごとに (32, 4, 256, 256) |
| 4つのブロックとnormの後 | (32, 256, 128) |
| ロジット | (32, 256, 1674) |
| 損失 | 数1つ |

## 9. 知っておくと良い小さなこと

- **すべては、行列のかけ算の一直線**に、いくつかの小さな関数を足したもの。繰り返しの構造は無い。
- **Attentionのコストは `T²` で増える**(点数の表が `T × T`)ので、`block_size` は256に抑えている。
- **Dropoutは初期値で0**なので、学習と評価の動作は同じ。`.train()` / `.eval()` の切り替えが効くのは、dropoutをオンにしたときだけ。
- **IDの確認:**モデルは、トークンIDが `< vocab_size` かを確認しない。大きすぎるIDは、PyTorchの添字のエラーになる。
- **共有された重みとチェックポイント:**出力の行列と埋め込みは1つのテンソル。だからパラメータ数でも1回しか数えない。
- **`ModelConfig` の初期値 `vocab_size=1674` は仮のもの。**実際の実行では、語彙はトークナイザーから取る(Day 8がそうしている)。

## 10. 一段落でまとめ

`model.py` はModel Aを定義します:トークンIDがベクトルになり、複数のブロックが、各位置に前の位置を見させ(回転式の位置情報付きの因果的Attention)、集めたことを考えさせ(MLP)、正規化と残差接続を伴います。埋め込みと共有した最後の層が、すべての位置で語彙のすべてのトークンの点数を出力します。このファイルは設定と3つの試す大きさ(tiny/small/medium)も持ち、パラメータを2つの方法で数えて照らし合わせます。
