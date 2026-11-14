# Day 10.2 — test_generation.py — やさしい1行ずつの解説(日本語)

> **Day 10 の3つのファイルの関係**
> - **Day 10.1 `generation.py`** — テストされるコード(**先に読んでください**)。
> - **Day 10 `day10_generate.py`** — 運転手のスクリプト。
> - **Day 10.2(このファイル)`test_generation.py`** — **12個の自動テスト**。CPUのみ、数秒。**学習済みのチェックポイントは不要** — テストは、手で作った点数と偽のモデルを使う。
>
> pytestの基本は、Day 6.2〜9.3で説明しました。ここでは新しいことだけを説明します。

---

## 0. 全体像(まずここから)

生成は、*ほぼ*正しくするのは簡単で、*ぴったり*正しくするのは難しい(上限の1つのずれ、ペナルティの符号の間違い…)。このテストは、**正しい答えがはっきり分かる、小さな手作りの数**で、小さな部品を1つずつ確かめます:

| 約束 | テスト |
|---|---|
| 上位の候補が、並べ替えられた確率になっている | `test_candidates_are_sorted_probabilities` |
| 特殊トークンは書かれないが、`<eos>` は許される | `test_special_tokens_are_never_written_but_eos_is_allowed` |
| 連続の上限が、ちょうど「4個はOK、5個目は止める」 | `test_same_token_run_limit_matches_training_data` |
| 3トークンの並びの上限が、ちょうど「2回はOK、3回目は止める」 | `test_trigram_limit_matches_training_data` |
| 繰り返しのペナルティが、使ったトークンだけを下げる(負の数でも向きが正しい) | `test_repetition_penalty_lowers_only_used_tokens` |
| 貪欲法が一番上のトークンを取る | `test_greedy_takes_the_top_token` |
| top-k = 1 と、ごく小さなtop-pが、貪欲法のように振る舞う | `test_top_k_one_and_tiny_top_p_behave_like_greedy` |
| 温度が、2番目の選択の選ばれる頻度を本当に変える | `test_temperature_changes_how_often_the_second_token_is_chosen` |
| シードがプロンプトごとに固定で、プロンプト間で違う | `test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts` |
| `generate` が `<eos>` で止まり、長さの上限で止まり、再現できる | `test_generate_stops_at_eos_and_is_reproducible` |
| 繰り返しの確認とdistinct-2が、正しい値を出す | `test_repetition_check_and_distinct_2` |
| 無効な設定が弾かれる | `test_invalid_settings_are_refused` |

実行(前と同じ):`.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. 新しい用語

| 用語 | かんたんな意味 |
|---|---|
| **偽のモデル(fake model)** | 本物のネットワークの代わりに、手で選んだ点数を返す小さな関数 — テストを速く、予測できるものにする。 |
| **`pytest.mark.skipif(条件, reason=…)`** | デコレーター:条件が真なら、テストを**スキップ**する(ここでは、トークナイザーのファイルが無いとき)。 |
| **`torch.Generator().manual_seed(n)`** | シード付きの乱数生成器(Day 10.1)。 |

---

## 2. 準備と補助関数 `flat`

```python
from __future__ import annotations

import math, sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer  # noqa: E402
from src.generation import (TRAIN_MAX_SAME_TOKEN_RUN, TRAIN_MAX_TRIGRAM_REPEATS, SamplingConfig, banned_special_ids,
                            choose_token, distinct_2, generate, is_repetitive, next_token_candidates, process_logits,
                            prompt_seed)  # noqa: E402

TOKENIZER = REPO_ROOT / "data/tokenizer/tokenizer.json"
V = 400
A, B, C = 300, 301, 302      # three ordinary byte-range token IDs used in the tests
```
- いつもの準備。`TOKENIZER` — Day 6のトークナイザーのファイル(必要なテストは1つだけ)。`V = 400` — **見立ての語彙サイズ**。`A, B, C` — テストで使う**普通のトークンID**3つ(300、301、302。どれも特殊な範囲0〜103より上)で、「トークンA/B/C」の代わり。

```python
def flat(**scores: float) -> torch.Tensor:
    t = torch.zeros(V)
    for k, v in scores.items():
        t[int(k[1:])] = v
    return t
```
- **点数の表を作る**小さな補助:全部0から始め、選んだトークンに設定する。工夫は、**キーワードの名前がトークンの番号を持つ**こと — `flat(t300=10.0)` は添字 `300` を `10.0` にする(`k[1:]` が先頭の `t` を落とし、`int(...)` が数にする)。`**scores` はキーワード引数を辞書に集める。

---

## 3. テスト

### 3.1 `test_candidates_are_sorted_probabilities`

```python
def test_candidates_are_sorted_probabilities() -> None:
    logits = torch.randn(V, generator=torch.Generator().manual_seed(0))
    cands = next_token_candidates(logits, 10)
    probs = [p for _, p in cands]
    assert len(cands) == 10 and probs == sorted(probs, reverse=True)
    assert probs[0] == pytest.approx(torch.softmax(logits, 0).max().item())
    assert 0 < sum(probs) <= 1
```
- 400トークンの、(シード付きの)ランダムな点数。上位10個を求める。
- ちょうど10個で、**大きい順に並んでいる**。最初のものが真の最大の確率と等しい。そして上位10の合計が0から1の間(確率の合計は1を超えない)。

### 3.2 `test_special_tokens_are_never_written_but_eos_is_allowed`

```python
def test_special_tokens_are_never_written_but_eos_is_allowed() -> None:
    logits = torch.zeros(V)
    logits[BOS_ID] = 100.0
    out = process_logits(logits, [A], [], SamplingConfig())
    assert all(out[i] == float("-inf") for i in banned_special_ids(V))
    assert EOS_ID not in banned_special_ids(V) and torch.isfinite(out[EOS_ID])
    assert torch.isfinite(out[BYTE_OFFSET:]).all()
```
- `<bos>` に巨大な点数(100)を与える — 罠。整えた後:**禁止された特殊IDはすべて `−∞`**(だから罠は失敗する)、`<eos>` は禁止リストに**無く**、点数は有限のまま、そして**普通のトークンすべて(ID ≥ 104)**は有限のまま(他のものが、うっかり除かれていない)。

### 3.3 `test_same_token_run_limit_matches_training_data`

```python
def test_same_token_run_limit_matches_training_data() -> None:
    cfg = SamplingConfig(max_run=TRAIN_MAX_SAME_TOKEN_RUN)
    logits = flat(t300=10.0)
    ok = process_logits(logits, [A] * (TRAIN_MAX_SAME_TOKEN_RUN - 1), [], cfg)   # 3 in a row: a 4th is allowed
    assert torch.isfinite(ok[A])
    stop = process_logits(logits, [A] * TRAIN_MAX_SAME_TOKEN_RUN, [], cfg)        # 4 in a row: a 5th is not
    assert stop[A] == float("-inf")
```
- 典型的な**境界のテスト**:上限4で、文脈 `[A, A, A]`(3個続き)は、4個目の `A` を**許し**、`[A, A, A, A]`(4個続き)は5個目を**止める**はず。(`[A] * 3` は `A` が3つのリスト。)1つのずれの間違いを見つける。

### 3.4 `test_trigram_limit_matches_training_data`

```python
def test_trigram_limit_matches_training_data() -> None:
    cfg = SamplingConfig(max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS)
    once = [A, B, C, A, B]                    # A B C seen once: A B C again is allowed (like 1991年から1997年)
    assert torch.isfinite(process_logits(flat(t302=5.0), once, [], cfg)[C])
    twice = [A, B, C, A, B, C, A, B]          # seen twice: a third time is not
    assert process_logits(flat(t302=5.0), twice, [], cfg)[C] == float("-inf")
```
- 上限2。文脈 `A B C A B` — `A B C` は**1回**起きていて、今 `A B` にいるので、`C`(2回目の `A B C`)は**許される**はず。文脈 `A B C A B C A B` — `A B C` は**2回**起きているので、`C` は**3回目**になる → **止められる**。

### 3.5 `test_repetition_penalty_lowers_only_used_tokens`

```python
def test_repetition_penalty_lowers_only_used_tokens() -> None:
    logits = flat(t300=2.0, t301=-1.0, t302=2.0)
    out = process_logits(logits, [A, B], [A, B], SamplingConfig(repetition_penalty=2.0))
    assert out[A] == pytest.approx(1.0)       # positive score divided
    assert out[B] == pytest.approx(-2.0)      # negative score multiplied
    assert out[C] == pytest.approx(2.0)       # unused token unchanged
```
- 点数:A = 2、B = −1、C = 2。AとBは使用済み。ペナルティ2。
- A(正)→ `2 / 2 = 1`;B(負)→ `−1 × 2 = −2`(**もっと負 = 選ばれにくい**、正しい向き);C(未使用)→ `2` のまま。典型的な符号の間違いを防ぐ。

### 3.6 `test_greedy_takes_the_top_token`

```python
def test_greedy_takes_the_top_token() -> None:
    assert choose_token(flat(t300=1.0, t301=3.0), SamplingConfig(), None) == B
```
- 点数:A = 1、B = 3 → 貪欲法は **B**(301)を選ぶはず。(乱数生成器は不要:`None`。)

### 3.7 `test_top_k_one_and_tiny_top_p_behave_like_greedy`

```python
def test_top_k_one_and_tiny_top_p_behave_like_greedy() -> None:
    logits = flat(t300=1.0, t301=3.0, t302=2.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert choose_token(logits, SamplingConfig(method="sample", top_k=1), g) == B
        assert choose_token(logits, SamplingConfig(method="sample", top_p=0.01), g) == B
```
- *サンプリング*のモードでも、`top_k=1`(一番良いものだけ残す)とごく小さな `top_p=0.01`(一番上のトークンだけが残る。それは必ず残る)は、候補を**1つ**にするので、抽選は**毎回B**でなければならない — それぞれ20回確認する。2つの絞り込みが、本当に裾を切ることの証明。

### 3.8 `test_temperature_changes_how_often_the_second_token_is_chosen`

```python
def test_temperature_changes_how_often_the_second_token_is_chosen() -> None:
    logits = torch.full((V,), float("-inf"))
    logits[A], logits[B] = 2.0, 1.0

    def share_b(t: float) -> float:
        g = torch.Generator().manual_seed(1)
        cfg = SamplingConfig(method="sample", temperature=t)
        return sum(choose_token(logits, cfg, g) == B for _ in range(2000)) / 2000

    expected = 1 / (1 + math.e)               # softmax probability of B at temperature 1
    assert share_b(1.0) == pytest.approx(expected, abs=0.04)
    assert share_b(0.5) < share_b(1.0) < share_b(2.0)
```
- 候補は2つだけ(A = 2、B = 1。他は `−∞`)。`share_b(t)` = 温度 `t` のとき、**2000回の抽選のうち、Bを選んだ割合**。
- 温度1では、Bの確率は `e¹ / (e² + e¹) = 1 / (1 + e) ≈ 27%`。測った割合は、これの ±0.04 以内でなければならない(ランダムな抽選は、ぴったりにはならない)。
- そして順序:**冷たい(0.5)→ Bはあまり選ばれない;熱い(2.0)→ より選ばれる**。温度がすべきことを、している。

### 3.9 `test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts`

```python
def test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts() -> None:
    assert prompt_seed(42, "営業部の") == prompt_seed(42, "営業部の")
    assert prompt_seed(42, "営業部の") != prompt_seed(42, "開発部の")
    assert prompt_seed(42, "営業部の") != prompt_seed(7, "営業部の")
```
- 同じシード + 同じプロンプト → **同じ**数(再現できる)。違うプロンプト → 違う数。違うシード → 違う数。

### 3.10 `test_generate_stops_at_eos_and_is_reproducible`(トークナイザーが無ければスキップ)

```python
@pytest.mark.skipif(not TOKENIZER.exists(), reason="needs data/tokenizer/tokenizer.json (Day 6)")
def test_generate_stops_at_eos_and_is_reproducible() -> None:
    tok = ByteBPETokenizer.load(TOKENIZER)
    target = tok.encode("社員を教えてください。")

    def fake_model(ids: list[int]) -> torch.Tensor:      # writes the target, then <eos>
        n = len(ids) - 1 - len(tok.encode("営業部の"))
        logits = torch.zeros(tok.vocab_size)
        logits[target[n] if n < len(target) else EOS_ID] = 20.0
        return logits
```
- `@pytest.mark.skipif(...)` — Day 6のトークナイザーが無いときは、スキップされる。
- **偽のモデル** — 「ここまでのトークン」から点数への関数:`n` = すでに生成されたトークンの数(`len(ids)` から `<bos>` とプロンプトのトークン数を引いたもの)を求め、**決まった文 `社員を教えてください。` の次のトークンに、巨大な点数(20)**を与える。文が完成した後は、`<eos>` に与える。だから「モデル」は、いつもまさにその文を書いて、止まる。

```python
    g = generate(fake_model, tok, "営業部の", SamplingConfig())
    assert g.finished and g.text == "社員を教えてください。" and g.ids[-1] == EOS_ID
    s1 = generate(fake_model, tok, "営業部の", SamplingConfig(method="sample", temperature=1.0))
    s2 = generate(fake_model, tok, "営業部の", SamplingConfig(method="sample", temperature=1.0))
    assert s1.ids == s2.ids
```
- 貪欲な生成は**完結**し、文は決めた文で、最後に生成されたIDは `<eos>` のはず。
- 同じ設定で2回サンプリングすると、**同一のID**になるはず(固定シードによる再現性)。

```python
    def never_ends(ids: list[int]) -> torch.Tensor:
        logits = torch.zeros(tok.vocab_size)
        logits[EOS_ID] = -50.0
        return logits

    cut = generate(never_ends, tok, "営業部の", SamplingConfig(max_new_tokens=7))
    assert cut.stop_reason == "max_new_tokens" and len(cut.ids) == 7
```
- **決して `<eos>` を出さない**(点数 −50)2つ目の偽のモデル。`max_new_tokens=7` なら、生成は**長さの上限で止まる**はず:理由は `"max_new_tokens"`、ちょうど**7**トークン。
- これが可能なのは、`generate` が、本物のモデルだけでなく、どんな点数の**関数**でも受け取れるから(Day 10.1の§4.3)。

### 3.11 `test_repetition_check_and_distinct_2`

```python
def test_repetition_check_and_distinct_2() -> None:
    assert not is_repetitive([A, A, A, A])            # 4 in a row occurs in training questions
    assert is_repetitive([A] * 5)
    assert is_repetitive([A, B, C] * 3)
    assert distinct_2([[A, B, A, B]]) == pytest.approx(2 / 3)
    assert distinct_2([]) == 0.0
```
- `A A A A`(4個続き)は繰り返し**ではない**(本物の質問にもある)。`A` が5個は**繰り返し**。`A B C A B C A B C`(同じ3トークンの並びが3回)は**繰り返し**。
- `distinct_2([[A, B, A, B]])`:組は `(A,B)`、`(B,A)`、`(A,B)` → 3つのうち異なるのは2つ → `2/3`。空の入力は `0.0`(ゼロ割りなし)。

### 3.12 `test_invalid_settings_are_refused`

```python
def test_invalid_settings_are_refused() -> None:
    for bad in ({"method": "beam"}, {"temperature": 0}, {"top_p": 0}, {"repetition_penalty": 0.5}):
        with pytest.raises(ValueError):
            SamplingConfig(**bad)
```
- 4つの無効な設定(未知の方法、温度0、top-p 0、1未満のペナルティ)。それぞれ `ValueError` を出さなければならない(`SamplingConfig(**bad)` は辞書をキーワード引数に展開する)。

---

## 4. テストはいくつ?

**12個のテスト。**期待される結果:`12 passed`、またはトークナイザーのファイルが無ければ `11 passed, 1 skipped`(テスト3.10)。

## 5. このテストが確かめて*いない*こと

- **本物の学習済みモデル**での生成(それは、Day 10の報告書のスクリプトが見せる)。
- `load_model` と `model_logits_fn`(チェックポイント/モデルが必要)。
- `token_text`、`max_same_token_run`、`max_trigram_count` の直接のテスト(`is_repetitive` を通じて動く)。
- コマンドラインのスクリプト `day10_generate.py` と、Markdownの報告書の形式。

## 6. 一段落でまとめ

この12個のCPUテストは、小さな手作りの点数と偽のモデルで、生成のエンジンを確かめます:候補が並べ替えられた確率であること、特殊トークンは禁止されるが `<eos>` は許されること、繰り返しの上限が学習用の質問から測った境界でちょうど止めること、ペナルティが負の点数を正しく扱うこと、貪欲・top-k・top-p・温度が定義どおりに振る舞うこと、シードがプロンプトごとに安定していること、生成が `<eos>` か長さの上限で、再現できる形で止まること、繰り返しとdistinct-2の確認が期待どおりの値を出すこと、そして無効な設定が弾かれること。
