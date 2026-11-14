# Day 10.1 — generation.py — やさしい1行ずつの解説(日本語)

> **Day 10 の3つのファイルの関係**
> - **Day 10.1(このファイル)`generation.py`** — *エンジン*:モデルの点数を、次のトークンの候補と、生成した文に変える。
> - **Day 10 `day10_generate.py`** — *運転手*:このエンジンを使う3つのコマンド(`candidates`、`generate`、`report`)。
> - **Day 10.2 `test_generation.py`** — *検査員*:12個の自動テスト(学習済みのモデルは不要)。
>
> おすすめの読む順番:**10.1 → 10 → 10.2**。

---

## 0. 全体像(まずここから)

Model A(Day 8)がするのは**1つのこと**だけです:ここまでのトークンを受け取り、語彙の**すべてのトークンに点数**を付ける(「これが次に来る可能性はどれくらい?」)。*文章*を得るには、小さなサイクルを、トークン1つずつ繰り返さなければなりません:

```
ここまでのトークン → モデル → 点数 → ① 点数を整える → ② トークンを1つ選ぶ → 足す → 繰り返す
                                                     <eos>(質問の終わり)か、40トークンで止まる
```

### 手順1 — 点数から確率へ(次のトークンの候補)

`softmax` が、点数を**合計1の確率**に変えます。確率の高い上位10トークンを見せるのが、学校の作業計画の**作業10**:「モデルは、次に何が来ると思っているか?」

### 手順2 — 点数を整える(`process_logits`)

| 調整 | することと理由 |
|---|---|
| **特殊トークンを禁止** | `<pad>`、`<bos>`、`<unk>`、`<unused_*>` は絶対に書かれない(`<eos>` は許される)。本物の文字ではないから |
| **繰り返しのペナルティ**(任意) | すでに使ったトークンを、少し選ばれにくくする。繰り返しを止める |
| **繰り返しの上限** | 同じトークンは**続けて4回まで**。同じ3トークンの並びは**2回まで**。`5000000000…` のような繰り返しを止めるが、本物の質問は**止めない** |

上限(4と2)は、Day 10で**学習用の質問から測りました**:本物の質問に実際に現れる、最大の繰り返しです(例:`社員番号100003` は0が何個か続き、`1991年から1997年` は3トークンの型を繰り返す)。上限を破るトークンは**取り除かれます**(点数 `−∞`)。

### 手順3 — トークンを1つ選ぶ

| 方法 | 選び方 |
|---|---|
| **貪欲法(greedy)** | いつも一番確率の高いトークン。同じ入力 → 同じ出力。 |
| **サンプリング** | 確率に従ったランダムな抽選。**温度**、**top-k**、**top-p** で形を整え、**決まったシード**を使う(再現できる)。 |

#### 温度 — どれくらい「大胆」に選ぶか

softmaxの前に、点数を温度で割ります。点数が2と1の候補2つの例:

| 温度 | 確率 | 効果 |
|---|---|---|
| 0.5(< 1) | 88% / 12% | **鋭い** — 安全だが、繰り返しやすい |
| 1.0 | 73% / 27% | 普通 |
| 2.0(> 1) | 62% / 38% | **平ら** — よりランダムで、間違いも増える |

#### top-kとtop-p — ありそうにない裾を切る

- **top-k = 50:**一番良い50トークンだけを残す。
- **top-p = 0.95(「核」)**:確率の合計が0.95になる、最小の上位グループを残す。例:確率 `0.5, 0.3, 0.15, 0.05`、p = 0.9 → 最初の3つを残し(0.5 + 0.3 + 0.15 = 0.95 ≥ 0.9)、0.05を落とし、正規化し直す。

### 手順4 — 繰り返す

`<eos>` か `max_new_tokens`(40)まで。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **ロジット(logits)** | モデルの生の点数(トークンごとに1つ)。 |
| **ソフトマックス** | 点数 → 合計1の確率。 |
| **貪欲デコード** | いつも一番上のトークンを取る。 |
| **サンプリング** | 与えられた確率で、トークンをランダムに引く。 |
| **温度 / top-k / top-p** | 上の3つの「ランダムさの調整」。 |
| **シード / `torch.Generator`** | ランダムな抽選を再現できるようにする。 |
| **文脈(context)** | ここまでのトークン(プロンプト + 生成した分)。 |
| **`-inf` のマスク** | 点数をマイナス無限大にして、確率をちょうど0にすること。 |
| **バイグラム / トライグラム** | 隣り合う2つ / 3つのトークンの並び。 |
| **distinct-2** | 多様さの尺度:異なる2トークンの並び ÷ すべての2トークンの並び。 |
| **CRC32** | 文字から計算する、決まったチェックサムの数(いつでも、どのコンピューターでも同じ)。 |
| **`<eos>` / ⟨eos⟩** | 「質問の終わり」のトークン。 |

---

## 2. docstring、import、定数

- docstring(英語 + 日本語)が、上の手順1〜4を説明している。

```python
from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer
from src.model import ModelA, ModelConfig
from src.training import load_checkpoint
```
- ほとんどは既知。新しいもの:`zlib`(CRC32用)、`Callable` と `Sequence`(型ヒント:「関数」/「リストのような並び」)。
- 前の日から:IDの定数とトークナイザー(**Day 6.1**)、モデル(**Day 8.1**)、`load_checkpoint`(**Day 9.2**)。

```python
TRAIN_MAX_SAME_TOKEN_RUN: int = 4
TRAIN_MAX_TRIGRAM_REPEATS: int = 2

LogitsFn = Callable[[list[int]], torch.Tensor]
```
- **学習用の質問から測った2つの上限**。
- `LogitsFn` — **型エイリアス**:「トークンIDのリストを受け取り、点数のテンソルを返す関数」。生成のコードは、*どんな*そのような関数でも動く(本物のモデルでも、テストの偽物でも)。

---

## 3. `SamplingConfig` — 生成の設定すべて

```python
@dataclass
class SamplingConfig:
    method: str = "greedy"              # "greedy" or "sample"
    temperature: float = 1.0
    top_k: int = 0                      # 0 = off
    top_p: float = 1.0                  # 1.0 = off
    repetition_penalty: float = 1.0     # 1.0 = off
    max_run: int | None = None          # None = off; TRAIN_MAX_SAME_TOKEN_RUN to switch on
    max_trigram_repeats: int | None = None  # None = off; TRAIN_MAX_TRIGRAM_REPEATS to switch on
    max_new_tokens: int = 40
    seed: int = 42
    name: str = "custom"
```
- どの設定にも「**何もしない**」初期値(`0`、`1.0`、`None`)があるので、初期値のままなら、普通の貪欲な生成になる。
- `max_run` / `max_trigram_repeats` は、学習用の値を設定すると、繰り返しの上限を**オン**にする。
- `max_new_tokens` = 長さの上限(40);`seed` = サンプリング用;`name` = 表のラベル。

```python
    def __post_init__(self) -> None:
        if self.method not in ("greedy", "sample"):
            raise ValueError("method must be 'greedy' or 'sample'")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0 (use method='greedy' for the most likely token)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)
```
- 意味のない設定を弾く(Day 8.1の `ModelConfig` と同じ形):未知の方法;温度が0以下(ゼロ割りになる。代わりに貪欲法を使う);`top_p` が (0, 1] の外;ペナルティが1未満(繰り返しを*促して*しまう)。
- `to_dict` — 設定を辞書として(生成される報告書に表示される)。

### 4つのプリセット

```python
PRESET_SETTINGS: dict[str, SamplingConfig] = {
    "greedy": SamplingConfig(name="greedy"),
    "greedy+limits": SamplingConfig(name="greedy+limits", max_run=TRAIN_MAX_SAME_TOKEN_RUN,
                                    max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
    "sample": SamplingConfig(name="sample", method="sample", temperature=0.8, top_k=50, top_p=0.95,
                             max_run=TRAIN_MAX_SAME_TOKEN_RUN, max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
    "sample-hot": SamplingConfig(name="sample-hot", method="sample", temperature=1.5,
                                 max_run=TRAIN_MAX_SAME_TOKEN_RUN, max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
}
```

| プリセット | 何か | 目的 |
|---|---|---|
| `greedy` | 一番上のトークン、調整なし | Day 9の `samples.md` が示したもの |
| `greedy+limits` | 貪欲 + 繰り返しの上限 | 上限が働くのを見る |
| `sample` | 温度0.8、top-k 50、top-p 0.95、上限 | 「おすすめ」の設定 |
| `sample-hot` | 温度1.5、上限 | **ランダムすぎる**とどうなるかを見せる |

これらは**固定**なので、Day 10の報告書を再現できる。

### 結果の箱 `Generation`

```python
@dataclass
class Generation:
    prompt: str
    ids: list[int]          # generated tokens only (without the prompt; <eos> included if reached)
    text: str               # decoded generated text (without <eos>)
    stop_reason: str        # "eos" or "max_new_tokens"

    @property
    def finished(self) -> bool:
        return self.stop_reason == "eos"
```
- 1回の生成の結果:プロンプト、**新しい**トークンID(届いていれば `<eos>` 付き)、復号した**続きの文**(プロンプトも `<eos>` も含まない)、止まった理由。
- `finished` — モデルが**自分で質問を終えた**(`<eos>`)なら `True`、長さの上限で切られたなら `False`。

---

## 4. モデルの読み込みと包み込み

### 4.1 `token_text`

```python
def token_text(tok: ByteBPETokenizer, token_id: int) -> str:
    """A readable form of one token: the text, <eos>, or <xx..> for a piece of a character."""
    if token_id == EOS_ID:
        return "<eos>"
    raw = tok.token_bytes[token_id]
    if token_id < BYTE_OFFSET:
        return raw.decode("utf-8")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + raw.hex() + ">"
```
- トークン1つの読めるラベル(候補の表に使う):終わりのトークンは `<eos>`;特殊・予約のトークンはASCIIの名前;普通のトークンは文字;そして**文字の一部**(単独ではUTF-8として正しくないバイト)は `<hex>` — `ByteBPETokenizer.pieces`(Day 6.1)と同じ決まり。

### 4.2 `load_model`

```python
def load_model(checkpoint: Path, device: torch.device) -> tuple[ModelA, dict]:
    state = load_checkpoint(checkpoint, device)
    model = ModelA(ModelConfig.from_dict(state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    info = {"path": str(checkpoint), "step": state["step"], "best_val": state["best_val"],
            "tokenizer_sha256": state["tokenizer_sha256"], "model_name": state["model_config"].get("name", "")}
    return model, info
```
- Day 9のチェックポイント(`best.pt` か `last.pt`)を `load_checkpoint`(安全な読み込み。Day 9.2)で読む。
- **チェックポイントの中に保存された設定**(`state["model_config"]`)からモデルを作り直し、重みを読み込み、**評価モード**にする。
- モデルと `info` の辞書(パス、ステップ、最良の検証の損失、**学習時に保存されたトークナイザーの指紋**、モデル名)を返す — 呼び出し側が、トークナイザーが今も合っているか確認できるように。

### 4.3 `model_logits_fn`

```python
def model_logits_fn(model: ModelA, device: torch.device) -> LogitsFn:
    """Wrap the model as 'token IDs so far -> scores for the next token' (crops to the block size)."""
    block = model.cfg.block_size

    @torch.no_grad()
    def fn(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        return logits[0, -1].float().cpu()

    return fn
```
- モデルを、単純な関数の型 `LogitsFn` に変える。これは**クロージャ**:内側の `fn` が `model`、`device`、`block` を覚えている。
- `fn(ids)`:最後の `block_size` 個のトークンを取り、1つだけのバッチを作り、モデルを動かし、**最後の位置**の点数(`logits[0, -1]`)を、**CPU**上の32ビットの小数として返す。(`@torch.no_grad()` — 勾配なし。)
- なぜ関数にするか? 生成のロジック(これ以降のすべて)が、**モデルやGPUについて何も知らなくてよく**、テストでは小さな偽の関数に置き換えられるから。
- (「キャッシュ」は無し:新しいトークンのたびに、列全体をもう一度処理する。単純で、40トークン以下なら十分速い。)

---

## 5. 部品

### 5.1 `next_token_candidates`

```python
def next_token_candidates(logits: torch.Tensor, k: int = 10) -> list[tuple[int, float]]:
    """The k most likely next tokens with their probabilities (the raw model, no adjustments)."""
    probs = F.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)]
```
- `softmax` → 確率;`torch.topk(probs, k)` → 大きい順の `k` 個の値とその位置(添字 = トークンID)。`(トークンID, 確率)` の組を返す。**調整なし** — 生のモデルが何を信じているかを見せる。

### 5.2 `banned_special_ids`

```python
def banned_special_ids(vocab_size: int) -> list[int]:
    """Every ID below the byte range except <eos>: never written in a question."""
    return [i for i in range(min(BYTE_OFFSET, vocab_size)) if i != EOS_ID]
```
- ID 0〜103(`<pad>`、`<bos>`、`<unk>`、100個の `<unused_*>`)の**うち `<eos>`(ID 2)を除く**もの。(`min(...)` は、テストの小さな語彙を守る。)

### 5.3 `run_length_at_end`

```python
def run_length_at_end(ids: Sequence[int]) -> int:
    if not ids:
        return 0
    n = 1
    while n < len(ids) and ids[-1 - n] == ids[-1]:
        n += 1
    return n
```
- `ids` の末尾に、**同じトークンが続けて何個**あるか(例:`[7, 5, 5, 5]` → 3)。`ids[-1]` が最後、`ids[-1-n]` が `n` 個前;最初に違うものが出たら止まる。

### 5.4 `process_logits` — 整える処理

```python
def process_logits(logits, context, generated, cfg) -> torch.Tensor:
    """Apply bans and repetition control. `context` = prompt + generated (without <bos>);
    `generated` = only the new tokens. Returns new logits (removed tokens are -inf)."""
    out = logits.clone().float()
    out[banned_special_ids(out.numel())] = float("-inf")
```
- 点数をコピーし(`clone`。元は触らない)、禁止された特殊トークンを `−∞` にする。

```python
    if cfg.repetition_penalty > 1.0 and generated:
        used = torch.tensor(sorted(set(generated)), dtype=torch.long)
        vals = out[used]
        out[used] = torch.where(vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty)
```
- **繰り返しのペナルティ**(「CTRL方式」)。有効で、すでにトークンが生成されているときだけ:**すでに使った**トークンについて、**正の点数はペナルティで割り**、**負の点数はかける** — どちらも、そのトークンを**選ばれにくく**する。(負の点数を割ると、逆に*選ばれやすく*なってしまうので、2つの場合に分ける。`torch.where(条件, a, b)` は、条件が成り立つ所では `a`、そうでなければ `b` を選ぶ。)使っていないトークンは変わらない。

```python
    if cfg.max_run is not None and run_length_at_end(context) >= cfg.max_run:
        out[context[-1]] = float("-inf")         # one more would make the run too long
```
- **連続の上限:**文脈の末尾が既に `max_run`(4)個の同じトークンなら、そのトークンをもう1回続けることを禁止する(5個目は長すぎる)。

```python
    if cfg.max_trigram_repeats is not None and len(context) >= 2:
        a, b = context[-2], context[-1]
        counts: dict[int, int] = {}
        for x, y, z in zip(context, context[1:], context[2:]):
            if x == a and y == b:
                counts[z] = counts.get(z, 0) + 1
        for z, n in counts.items():
            if n >= cfg.max_trigram_repeats:
                out[z] = float("-inf")
    return out
```
- **3トークンの並びの上限:**`a, b` = 最後の2トークン。文脈の中で `a, b` が現れた、以前のすべての場所を探し、**その後にどのトークン `z` が続いたか**を数える(`counts[z]`)。並び `a b z` が既に `max_trigram_repeats`(2)回起きていたら、`z` を禁止する — 同じ3トークンの並びの3回目は、止められる。整えた点数を返す。
- 例:文脈 `A B C A B C A B` — `A B C` が2回起きているので、`A B` の後の `C` は禁止される。

### 5.3 `choose_token`

```python
def choose_token(logits, cfg, generator) -> int:
    if not torch.isfinite(logits).any():
        return EOS_ID                            # everything was removed: end the question
    if cfg.method == "greedy":
        return int(logits.argmax())
```
- **すべての**トークンが取り除かれていたら(全部 `−∞`)、質問を穏やかに終える(`<eos>`)。貪欲なら → 一番大きい点数の位置。

```python
    scores = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(scores, min(cfg.top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = F.softmax(scores, dim=-1)
```
- **温度:**点数を割る。**top-k:**k番目に良い点数(`kth`。上位kの中で一番小さいもの)を求め、それより**小さい**ものをすべて `−∞` にする。そして `softmax` → 確率。

```python
    if cfg.top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        before = torch.cumsum(sorted_p, dim=0) - sorted_p          # probability of the tokens ranked above
        keep = before < cfg.top_p                                   # always keeps the top token
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[order[keep]] = True
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))
```
- **top-p:**確率を大きい順に並べる。`cumsum − sorted_p` = 各トークンより**上位**のトークンの確率の合計(`before`)。その合計がまだ `p` より**小さい**間は残す — だから一番上のトークン(before = 0)は**必ず残る**。`mask[order[keep]] = True` で、残すトークンを元の並びに印を付け、他を0にして**正規化し直す**(新しい合計で割る)。
- `torch.multinomial(probs, 1, generator=generator)` — `probs` に従って、**トークンを1つランダムに引く**。シード付きの生成器を使う。

### 5.4 `prompt_seed`

```python
def prompt_seed(seed: int, prompt: str) -> int:
    return (seed * 1_000_003 + zlib.crc32(prompt.encode("utf-8"))) % (2**63)
```
- `(seed, プロンプト)` の組ごとに、決まった整数を1つ。`zlib.crc32` はプロンプトの文字の**決まったチェックサム**(Pythonの `hash` は実行ごとに変わるので使わない)。
- なぜプロンプトごと? **共有のシード1つ**だと、違うプロンプトが**同じ乱数の抽選**になってしまう。これなら**プロンプトごとに別のランダムな列**になり、同じプロンプトにはいつも同じものが与えられる(再現できる)。`% 2**63` で、シードとして有効な大きさに保つ。

### 5.5 `generate` — ループ全体

```python
def generate(logits_fn: LogitsFn, tok: ByteBPETokenizer, prompt: str, cfg: SamplingConfig) -> Generation:
    """Continue `prompt` until <eos> or cfg.max_new_tokens. Reproducible: same inputs + seed -> same output."""
    prompt_ids = tok.encode(prompt)
    ids = [BOS_ID] + prompt_ids
    generated: list[int] = []
    generator = torch.Generator().manual_seed(prompt_seed(cfg.seed, prompt)) if cfg.method == "sample" else None
    stop = "max_new_tokens"
    for _ in range(cfg.max_new_tokens):
        logits = process_logits(logits_fn(ids), prompt_ids + generated, generated, cfg)
        nxt = choose_token(logits, cfg, generator)
        ids.append(nxt)
        generated.append(nxt)
        if nxt == EOS_ID:
            stop = "eos"
            break
    text_ids = [i for i in generated if i != EOS_ID]
    return Generation(prompt, generated, tok.decode(text_ids), stop)
```
- `prompt_ids` — プロンプトをトークンにしたもの。`ids = [BOS_ID] + prompt_ids` — モデルへの入力は、学習データ(`<bos> 質問 <eos>`)とまったく同じく、`<bos>` から始まる。
- `generated` — 新しいトークン。サンプリングなら、このプロンプト用の**シード付きの乱数生成器**を作る。貪欲なら不要。
- 毎回:モデルに点数を聞き(`logits_fn(ids)`)、**整え**(文脈 = プロンプト + 生成した分、`<bos>` は*含めない*)、トークンを**選び**、両方のリストに足す。`<eos>` で止まる(`stop = "eos"`)— そうでなければ、`max_new_tokens` 回でループが終わり、`"max_new_tokens"` になる。
- `Generation` を返す:`<eos>` を**含まない**、復号した文。

---

## 6. 簡単な自動チェック(学校の作業計画の作業6の項目)

```python
def max_same_token_run(ids: Sequence[int]) -> int:
    best = run = 0
    for i, t in enumerate(ids):
        run = run + 1 if i and ids[i - 1] == t else 1
        best = max(best, run)
    return best
```
- `ids` のどこかにある、**同じトークンの一番長い連続**:`run` が今の連続を数え(`i and …` = 最初の項目でなく、前と同じ)、`best` が最大を保つ。

```python
def max_trigram_count(ids: Sequence[int]) -> int:
    counts: dict[tuple[int, int, int], int] = {}
    for tri in zip(ids, ids[1:], ids[2:]):
        counts[tri] = counts.get(tri, 0) + 1
    return max(counts.values(), default=0)
```
- すべての3トークンの並びを数え、**一番多く出るものの回数**を返す(`default=0` はごく短い入力用)。

```python
def is_repetitive(ids: Sequence[int]) -> bool:
    return max_same_token_run(ids) > TRAIN_MAX_SAME_TOKEN_RUN or max_trigram_count(ids) > TRAIN_MAX_TRIGRAM_REPEATS
```
- **A3の確認:**「どの本物の学習用の質問よりも繰り返しが多い」— 同じトークンが5個以上続く、または1つの3トークンの並びが3回以上。

```python
def distinct_2(all_ids: Sequence[Sequence[int]]) -> float:
    pairs = [p for ids in all_ids for p in zip(ids, ids[1:])]
    return len(set(pairs)) / len(pairs) if pairs else 0.0
```
- **A4の確認(多様さ):**すべての出力の、すべての2トークンの組を集める。`distinct-2` = **異なる**組の数 ÷ **すべての**組。1に近い = 多様、低い = 繰り返しが多くて退屈。(組が無ければ0.0。)

```python
def perplexity_of(loss: float) -> float:
    return math.exp(loss)
```
- 損失からパープレキシティ(Day 9.2)。このファイルの他では使われない、小さな補助。

---

## 7. 知っておくと良い小さなこと

- **貪欲法もサンプリングも、再現できる。**貪欲には乱数が無く、サンプリングはプロンプトごとのシードを使う。**同じマシン**で、同じ入力なら同じ文になる。(CPUとGPUの間では、モデルの最後の桁が違うことがあり、まれな僅差のとき、トークンが変わりうる。)
- **上限は本物の質問を止めない。**4と2は、学習用の質問で測った最大値だから。
- **繰り返しのペナルティは、どのプリセットでもオフ**(`1.0`)。実験用に使える。
- **KVキャッシュは無し:**モデルは、トークンごとに列全体を読み直す。40トークンなら十分。
- **プロンプトは `<bos>` の後から始まる。**学習は `<bos> 質問 <eos>` を使ったので、生成も同じにする。
- **作られてよい特殊トークンは `<eos>` だけ。**他がすべて禁止/除去されたら、質問は単に終わる。

## 8. 一段落でまとめ

`generation.py` は、学習済みのモデルを「トークン → 点数」の関数として包み、次のトークンの確率の上位を見せ、点数を整え(特殊トークンの禁止、任意で繰り返しのペナルティ、学習用の質問から測った繰り返しの上限)、トークンを1つ選び — 貪欲に、または温度・top-k・top-pで形を整えたシード付きのサンプリングで — `<eos>` か40トークンまで繰り返します。繰り返しと多様さ(distinct-2)の簡単な確認も備えています。
