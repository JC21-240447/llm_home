# Day 8.2 — test_model.py — やさしい1行ずつの解説(日本語)

> **Day 8 の3つのファイルの関係**
> - **Day 8.1 `model.py`** — モデル(**先に読んでください**)。
> - **Day 8 `day8_model_check.py`** — 運転手のスクリプト。
> - **Day 8.2(このファイル)`test_model.py`** — *検査員*:モデルが正しいことを確かめる、12個の自動テスト。すべて**CPUで数秒**で終わる。
>
> pytestの基本(テスト関数、`assert`、`parametrize`、`tmp_path`、`pytest.raises`)は、Day 6.2とDay 7.2で説明したので、このファイルでは新しい部分だけを説明します。

---

## 0. 全体像(まずここから)

モデルには、**何も落とさない**バグがあります:動くし、数も出るが、その数が間違っている(例えば、こっそり未来を覗いている、手書きのAttentionと速いAttentionが違う)。このテストは、モデルの**数学的な約束**を確かめて、そうした静かなバグを見つけます:

| 約束 | テスト |
|---|---|
| 出力の形が正しく、損失はターゲットがあるときだけ出る | `test_output_shapes_and_loss` |
| 学習前の損失 ≈ `ln(語彙)` | `test_initial_loss_is_close_to_uniform_guess` |
| **未来は過去を変えない**(因果性)。両方のAttentionの版で | `test_causality_future_token_does_not_change_the_past`(×2) |
| 手書きのAttention = 速いAttention | `test_manual_and_fast_attention_agree` |
| 埋め込みと出力の行列が、本当に同じメモリ | `test_weight_tying` |
| パラメータ数 = 手書きの式 | `test_parameter_count_matches_formula` |
| 3つのプリセットが、計画した大きさの範囲に入る | `test_presets_are_in_the_planned_size_range` |
| RoPEは**相対的な**位置だけに依存する | `test_rope_depends_only_on_relative_position` |
| 長すぎる入力は弾かれる | `test_too_long_input_is_rejected` |
| 設定を保存・読み込みできる。悪い設定は弾かれる | `test_config_round_trip` |
| モデルが小さな1バッチを覚えられる | `test_overfit_one_batch` |

実行(前と同じ):

```
.venv\Scripts\python.exe -m pytest tests -q
```

---

## 1. 新しい用語

| 用語 | かんたんな意味 |
|---|---|
| **`torch.randint(a, b, shape)`** | `a` から `b−1` までのランダムな整数のテンソル。ここでは、ランダムな偽のトークンID。 |
| **`torch.allclose(a, b, atol=...)`** | 2つのテンソルが、ごく小さな許容範囲(`atol`)で等しければ `True`。小数の計算は完全に同一にはならないので必要。 |
| **`.eval()`** | 評価モード(dropoutオフ、決定的)。 |
| **`torch.no_grad()`** | 勾配を記録しない(速い。確認には不要)。 |
| **`pytest.approx(x, abs=...)`** | 1つの数について「xにおよそ等しい」。 |
| **`data_ptr()`** | テンソルのデータのメモリのアドレス。同じアドレス = 同じメモリ。 |
| **`ndim`** | テンソルの次元の数(0 = 数1つ)。 |

---

## 2. docstring、import、準備

```python
"""Unit tests for src/model.py (Day 8). All run on the CPU in a few seconds.
...
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import (  # noqa: E402
    ATTENTION_MODES, ModelA, ModelConfig, apply_rope, count_parameters, expected_parameters, preset_config, rope_tables,
)
```
- Day 6.2・7.2と同じ形式(rawでないdocstringなのでバックスラッシュは2つ。`REPO_ROOT`/`sys.path` の技)。新しいのは `math` と `torch`。テストが必要とするものを、モデルのファイルから全部読み込む。

---

## 3. 補助

```python
VOCAB = 50


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4, d_model=64)
    return ModelConfig(**{**base, **kw})


def make_model(seed: int = 0, **kw) -> ModelA:
    torch.manual_seed(seed)
    return ModelA(tiny_cfg(**kw)).eval()
```
- `VOCAB = 50` — テストを速くするための、**ごく小さな見立ての語彙**。
- `tiny_cfg(**kw)` — **とても小さな設定**(2層、4ヘッド、幅64、ブロックサイズ32)。追加のキーワード引数が初期値を**上書き**する(`{**base, **kw}`。`preset_config` と同じ合わせ方)。
- `make_model(seed, **kw)` — 乱数のシードを固定し、小さなモデルを作って**評価モード**にする。シードが固定 = 実行のたびに同じ重み → テストが再現できる。

---

## 4. テスト

### 4.1 `test_output_shapes_and_loss`

```python
def test_output_shapes_and_loss() -> None:
    m = make_model()
    idx = torch.randint(0, VOCAB, (3, 10))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 10, VOCAB)
    assert loss is not None and loss.ndim == 0
    assert m(idx)[1] is None
```
- 10個のトークンIDのランダムな列を3つ。ターゲット(ここでは単に同じトークン)付きでモデルを動かす。
- `logits.shape` は `(3, 10, VOCAB)` のはず:**すべての位置で、すべてのトークンの点数**。
- 損失があって、**数1つ**(`ndim == 0`)のはず。
- ターゲット無し(`m(idx)`)なら、要素 `[1]`(損失)は `None` のはず。

### 4.2 `test_initial_loss_is_close_to_uniform_guess`

```python
def test_initial_loss_is_close_to_uniform_guess() -> None:
    """Before training the model knows nothing: loss should be about ln(vocab size)."""
    m = make_model()
    idx = torch.randint(0, VOCAB, (8, 32))
    _, loss = m(idx, torch.randint(0, VOCAB, (8, 32)))
    assert abs(loss.item() - math.log(VOCAB)) < 0.3
```
- ランダムな入力**とランダムなターゲット**。未学習のモデルは一様な当て推量のはずなので、損失は `ln(50) ≈ 3.91` から0.3以内でなければならない。(Day 8の確認の小さな版。悪い初期化やつなぎ間違いを見つける。)

### 4.3 `test_causality_future_token_does_not_change_the_past`(2回実行:`fast` と `manual`)

```python
@pytest.mark.parametrize("mode", ATTENTION_MODES)
def test_causality_future_token_does_not_change_the_past(mode: str) -> None:
    m = make_model(attention=mode)
    idx = torch.randint(0, VOCAB, (2, 20))
    changed = idx.clone()
    k = 12
    changed[:, k] = (changed[:, k] + 1) % VOCAB
    with torch.no_grad():
        a, _ = m(idx)
        b, _ = m(changed)
    assert torch.allclose(a[:, :k], b[:, :k], atol=1e-6)        # earlier positions: identical
    assert not torch.allclose(a[:, k:], b[:, k:], atol=1e-4)    # position k and later: changed
```
- **一番大事なテスト。**モデルが**未来を絶対に使わない**ことを証明する。
- `parametrize("mode", ATTENTION_MODES)` — Attentionの版ごとに1回(`fast`、`manual`)、テストを実行する。
- 20トークンのランダムな列を2つ取り、**コピー**し(`idx.clone()`)、コピーの**位置12の1トークン**を変える(`+1`、`% VOCAB` で一周して戻る)。
- 両方をモデルに通す(`a`、`b` = ロジット)。
- 確認1:**位置12より前**(`[:, :k]`)の出力は、**同一**(`1e-6` 以内)でなければならない — 後ろのトークンを変えても、前の出力に影響してはいけない。もし影響したら、モデルは未来を覗いている。
- 確認2:位置12以降(`[:, k:]`)の出力は、**違う**はず — そうでなければ、テストが無意味になる(変更は、許される場所で、実際に影響しなければならない)。

### 4.4 `test_manual_and_fast_attention_agree`

```python
def test_manual_and_fast_attention_agree() -> None:
    m = make_model()
    idx = torch.randint(0, VOCAB, (2, 32))
    with torch.no_grad():
        m.set_attention("fast")
        fast, _ = m(idx)
        m.set_attention("manual")
        manual, _ = m(idx)
    assert torch.allclose(fast, manual, atol=1e-5)
```
- 同じ重み、同じ入力で、`set_attention` でAttentionの版を切り替えて、出力を比べる:`1e-5` 以内で一致するはず。**読みやすい手書きのコード**とPyTorchの**速いコード**が、同じものを計算していることの証明。

### 4.5 `test_weight_tying`

```python
def test_weight_tying() -> None:
    m = make_model()
    assert m.lm_head.weight is m.tok_emb.weight
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
```
- 出力層の重みと埋め込みの表は、**同じオブジェクト**(`is`)で、**同じメモリのアドレス**(`data_ptr()`)にあるはず。つまり、行列は1回だけ保存され、コピーされていない。

### 4.6 `test_parameter_count_matches_formula`

```python
def test_parameter_count_matches_formula() -> None:
    for cfg in (tiny_cfg(), preset_config("tiny", 1674), preset_config("small", 1674)):
        m = ModelA(cfg)
        counts = count_parameters(m)
        assert counts["total"] == expected_parameters(cfg)
        assert counts["embedding"] == cfg.vocab_size * cfg.d_model
```
- 3つの設定(小さなテスト用の設定、プリセットtiny、プリセットsmall)でモデルを作り、**数えた**パラメータが**手書きの式**(Day 8.1の§7)と等しいこと、埋め込みの数が `vocab_size × d_model` と等しいことを確かめる。(mediumは、テストを速くするために飛ばす。)

### 4.7 `test_presets_are_in_the_planned_size_range`

```python
def test_presets_are_in_the_planned_size_range() -> None:
    sizes = {name: expected_parameters(preset_config(name, 1674)) for name in ("tiny", "small", "medium")}
    assert 0.5e6 < sizes["tiny"] < 2e6
    assert 3e6 < sizes["small"] < 8e6
    assert 10e6 < sizes["medium"] < 20e6
```
- **式だけ**を使って(モデルは作らないので、一瞬)、各プリセットの大きさが、計画した範囲に入ることを確かめる:tinyは50万〜200万、smallは300万〜800万、mediumは1,000万〜2,000万。(`0.5e6` = 500,000。)1,674トークンなら、約100万、515万、1,480万で、範囲の中。誰かがうっかり `PRESETS` を編集して、Day 11の計画を壊すのを防ぐ。

### 4.8 `test_rope_depends_only_on_relative_position`

```python
def test_rope_depends_only_on_relative_position() -> None:
    """Rotary embedding: q·k is the same when both positions shift by the same amount."""
    torch.manual_seed(1)
    cos, sin = rope_tables(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(pq: int, pk: int) -> float:
        rq = apply_rope(q, cos[pq:pq + 1], sin[pq:pq + 1])
        rk = apply_rope(k, cos[pk:pk + 1], sin[pk:pk + 1])
        return float((rq * rk).sum())

    assert score(10, 3) == pytest.approx(score(40, 33), abs=1e-4)
    assert score(10, 3) != pytest.approx(score(10, 9), abs=1e-3)
```
- Day 8.1の§5.2の、RoPEの大事な性質をテストする。
- ヘッドの大きさ16、位置64個分の回転の表を作る。ランダムなquery `q` と、ランダムなkey `k` を1つずつ作る(それぞれ16個の数のベクトル。形は `(1,1,1,16)`)。
- `score(pq, pk)` — `q` を位置 `pq` にあるものとして回転し(表のその1行だけを取る:`cos[pq:pq+1]`)、`k` を位置 `pk` にあるものとして回転し、**内積**を返す(`(rq * rk).sum()`)。
- 確認1:位置(10, 3)と(40, 33)は、**距離が同じ(7)** → 点数は**等しい**はず(`1e-4` 以内)。
- 確認2:位置(10, 3)(距離7)と(10, 9)(距離**1**)→ 点数は**違う**はず。つまり、点数は距離に依存し、それ以外には依存しない。
- `pytest.approx(値, abs=...)` により、`==` / `!=` が「およそ等しい / おおよそ等しくない」の意味になる。

### 4.9 `test_too_long_input_is_rejected`

```python
def test_too_long_input_is_rejected() -> None:
    m = make_model()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))
```
- テスト用のモデルは `block_size = 32`。**33**トークンの列(`torch.zeros(1, 33, dtype=torch.long)` = 整数の0が33個)は、`ValueError` を出さなければならない(Day 8.1の `forward` の確認)。

### 4.10 `test_config_round_trip`

```python
def test_config_round_trip(tmp_path: Path) -> None:
    cfg = preset_config("small", 1674, dropout=0.1)
    path = tmp_path / "config.json"
    cfg.save(path)
    assert ModelConfig.load(path) == cfg
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_head=3)
```
- 設定(プリセットsmallで `dropout=0.1` を上書き)を作り、一時ファイルに保存し、読み込み直す:元と**等しい**はず(データクラスの等価比較は、全項目を比べる)。
- そして**悪い**設定は弾かれなければならない:`d_model=100` は `n_head=3` で割り切れない → `ValueError`(`__post_init__` から)。

### 4.11 `test_overfit_one_batch`

```python
def test_overfit_one_batch() -> None:
    """A correct model and loss must be able to memorise one small batch almost perfectly."""
    torch.manual_seed(0)
    m = ModelA(tiny_cfg()).train()
    idx = torch.randint(0, VOCAB, (4, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(300):
        _, loss = m(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert first > 3.0
    assert loss.item() < 0.1
```
- Day 8の丸暗記の確認の小さな版を、**ランダムな偽のデータ**で行う:17個のランダムなトークンの列を4つ。`x` は最初の16個、`y` は1つずらしたもの。
- **学習モード**でAdamWにより300ステップ学習する(Day 8と同じループ)。
- `first = first if first is not None else loss.item()` — **最初の**損失だけを覚える(一度だけ設定する、簡潔な書き方)。
- 最初の損失が**3.0より大きく**(`ln 50 ≈ 3.9` 近くから始まる。つまり本当に何も知らなかった)、最後の損失が**0.1より小さい**(覚えた)ときだけ合格。ランダムなデータは「理解」できないので、損失を低くする唯一の方法は丸暗記 — それで、勾配、損失、オプティマイザがすべて動くことが証明される。

---

## 5. テストはいくつ?

| テスト | 実行回数 |
|---|---|
| 形と損失 | 1 |
| 初期の損失 | 1 |
| 因果性 | 2(fast、manual) |
| manualとfast | 1 |
| 重み共有 | 1 |
| パラメータ数 | 1 |
| プリセットの大きさの範囲 | 1 |
| RoPEの相対位置 | 1 |
| 長すぎる入力 | 1 |
| 設定の往復 | 1 |
| 1バッチを覚える | 1 |
| **合計** | **12** |

期待される結果:`12 passed`(このファイルには「スキップ」するテストは無い。すべて、本物のデータなしで動くため)。

## 6. このテストが確かめて*いない*こと

- **本物のデータでの学習**と生成(Day 9・10)。
- **GPU / bf16 / fp16 の動作** — すべてのテストは、CPUのfp32で動く(GPUの部分はDay 8のスクリプトが受け持つ)。
- `dropout > 0` のときの動作(学習と評価の違い)。
- 損失の `ignore_index` オプション(パディング用。あとで使う)。
- mediumのプリセット(Day 8自身と、式による大きさの範囲のテストで確認される)。

## 7. 一段落でまとめ

この12個の速いCPUテストは、Model Aが、正しい形の出力を出すこと、一様な当て推量の損失から始まること、**未来が過去に影響することが決してない**こと(両方のAttentionの版で)、手書きのAttentionと速いAttentionが一致すること、重みが本当に共有されていること、パラメータ数が式と合い、プリセットが計画した大きさの範囲に留まること、RoPEが相対的な位置だけに依存すること、長すぎる入力と無効な設定が弾かれること、設定が正確に保存・読み込みできること、そしてモデルが小さな1バッチを覚えられることを確かめます。
