# Day 12.1 — generation.py(更新版)— 変更点 + 1行ずつの解説(日本語)

> **これは、Day 10.1 で最初に見た `generation.py` の更新版です。**
> - **ファイルの先頭から `perplexity_of` までは、Day 10.1 とまったく同じ**です。その1行ずつの解説は **`day10.1_generation_explained_JA.md`**(とEN)にあります:`SamplingConfig`、プリセット、`process_logits`、`choose_token`、`generate`、`is_repetitive`、`distinct_2` などは、そちらを読んでください。
> - **ファイルの一番最後に、新しい関数 `quick_checks` が1つ足されました。**このファイルは、それを詳しく説明します。
>
> **Day 12 の5つのファイルの関係:**`day12_train_main.py`(本番の学習。**`quick_checks` を呼ぶ**)、`day12_monitor.py`(学習を見守る)、**`generation.py`(12.1、このファイル)**、`training.py`(12.2、早期終了)、`test_training.py`(12.3、新しい2つのテスト)。

---

## 0. 全体像(まずここから)

Day 11では、`day11_size_trials.py` が**自分専用の**関数 `generation_check` を持っていて、プロンプトの文章を生成し、**作業6の自動の項目**を計算していました:

| 項目 | 意味 |
|---|---|
| **A2 finished** | `<eos>` で終わった出力の割合 |
| **A3 repetitive** | どの本物の学習用の質問よりも繰り返しが多い出力の割合 |
| **A4 distinct-2** | 出力の多様さ |
| **A5 new**(参考用のみ) | 終えた出力のうち、学習用の質問**ではない**もの |

Day 12の本番の学習のスクリプトも、**同じ**確認が必要です。コードをコピーする代わりに、そのロジックを `quick_checks` として**ライブラリに移し**、どのスクリプトからも呼べるようにしました。(Day 12はこれを使う。あなたが貼ったDay 11のスクリプトには、まだ自分用のコピーがある。)

これはソフトウェアの普通の習慣です:**2つのスクリプトが同じ関数を必要とするなら、共有のライブラリに置く。**

---

## 1. 変わっていない部分(参照の表)

| ファイルの部分 | 説明がある場所 |
|---|---|
| `TRAIN_MAX_SAME_TOKEN_RUN`、`TRAIN_MAX_TRIGRAM_REPEATS`、`LogitsFn` | Day 10.1 §2 |
| `SamplingConfig`、`PRESET_SETTINGS`、`Generation` | Day 10.1 §3 |
| `token_text`、`load_model`、`model_logits_fn` | Day 10.1 §4 |
| `next_token_candidates`、`banned_special_ids`、`run_length_at_end`、`process_logits`、`choose_token`、`prompt_seed`、`generate` | Day 10.1 §5 |
| `max_same_token_run`、`max_trigram_count`、`is_repetitive`、`distinct_2`、`perplexity_of` | Day 10.1 §6 |

---

## 2. 新しい関数:`quick_checks`

```python
def quick_checks(logits_fn: LogitsFn, tok: ByteBPETokenizer, prompts: Sequence[str], cfg: SamplingConfig,
                 train_set: set[str] | None = None) -> dict:
    """Task 6 automatic items on a list of prompts (Day 11/12): finished (A2), repetitive (A3), distinct-2 (A4),
    new sentences (A5, reference), and the generated texts."""
```
- **入力:**
  - `logits_fn` — 「トークン → 点数」として包んだモデル(`model_logits_fn`、Day 10.1);
  - `tok` — トークナイザー;
  - `prompts` — 書き出しのリスト;
  - `cfg` — **1つの**生成の設定(例:`PRESET_SETTINGS["greedy+limits"]`);
  - `train_set` — 任意:学習用の質問の集合(「新しい」の数のため)。`set[str] | None = None` は「文字列の集合、または何も無し(初期値)」の意味。
- **出力:**4つの数と文章を入れた辞書。

```python
    gens = [generate(logits_fn, tok, p, cfg) for p in prompts]
    n = max(1, len(gens))
    body = [[i for i in g.ids if i != EOS_ID] for g in gens]
```
- `gens` — 与えられた設定で、**すべてのプロンプト**の続きを生成する(`Generation` オブジェクトのリスト。Day 10.1)。
- `n = max(1, len(gens))` — 出力の数。ただし**0にはならない**ので、プロンプトのリストが空でも、下の割り算が「ゼロ割り」で失敗しない。
- `body` — 出力ごとの、**`<eos>` を除いた、生成されたトークンID**(`[i for i in g.ids if i != EOS_ID]`)。**一度だけ**計算して、下の2つの確認で使い回す(Day 11の版は、インラインで2回作り直していた)。

```python
    return {
        "finished": sum(g.finished for g in gens) / n,
        "repetitive": sum(is_repetitive(tok.encode(g.prompt) + b) for g, b in zip(gens, body)) / n,
        "distinct2": distinct_2(body),
        "new": None if train_set is None else sum((g.prompt + g.text) not in train_set for g in gens if g.finished),
        "texts": {g.prompt: g.text + (" ⟨eos⟩" if g.finished else " …") for g in gens},
    }
```
- **`finished`(A2):**`sum(g.finished ...)` は `True` を足し合わせる(それぞれ1と数える);`n` で割って → **割合**(0.0〜1.0)。
- **`repetitive`(A3):**出力ごとに、**プロンプトのトークン + 生成したトークン**に `is_repetitive` を適用する(繰り返しは文全体で判定する)。`zip(gens, body)` が、各出力とそのトークンのリストを組にする。`n` で割って → 割合。
- **`distinct2`(A4):****すべての**出力をまとめた多様さ(`distinct_2(body)`)。
- **`new`(A5、参考用):**`train_set` が渡されていなければ `None`。そうでなければ、**終えた**出力のうち、文全体(`prompt + text`)が学習用の集合に**無い**ものの数。
- **`texts`:****プロンプト → 生成された文**の辞書。モデルが自分で質問を終えたなら ` ⟨eos⟩`、長さの上限で切られたなら ` …` を付ける — そのままMarkdownの報告書に貼れる。

---

## 3. 使われ方(Day 12)

```python
checks[name] = {s: quick_checks(fn, tok, prompts, PRESET_SETTINGS[s], train_set)
                for s in ("greedy+limits", "sample")}
gl = checks[name]["greedy+limits"]
r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
```
- 辞書内包表記が、**2つの設定**で `quick_checks` を実行する。`greedy+limits` の数が、学習のまとめ(Day 11.1)にコピーされる — それが、学習が**作業6の目安に合格するか**を決める。`sample` の結果は報告書のためだけ。

## 4. 知っておくと良い小さなこと

- **Day 11の `generation_check` と同じ式**なので、Day 11とDay 12の数字を比べられる。
- **`finished`/`repetitive` は割合**、`new` は**個数**(学習用の集合が無ければ `None`)。
- **`texts` はプロンプトをキーにしている**:同じプロンプトが重複すると、上書きされる。
- **この回では、`quick_checks` を直接テストする新しいテストは無い**(`generate`、`is_repetitive`、`distinct_2` に頼っていて、それらはDay 10.2でテストされている)。

## 5. 一段落でまとめ

`generation.py` は、最後に新しい関数 `quick_checks` が1つ足された以外、変わっていません。この関数は、プロンプトのリストについて1つの設定で文章を生成し、作業6の自動の項目 — 完結の割合、繰り返しの割合、distinct-2、(任意で)新しい文の数、そして文章 — を返すので、Day 11とDay 12が同じコードを使えます。
