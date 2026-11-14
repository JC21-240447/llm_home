# Day 9.3 — test_training.py — やさしい1行ずつの解説(日本語)

> **Day 9 の5つのファイルの関係**
> - **Day 9.1 `batching.py`** と **Day 9.2 `training.py`** — テストされるコード(先に読んでください)。
> - **Day 9 `day9_train.py`** と **`day9_compare_runs.py`** — スクリプト。
> - **Day 9.3(このファイル)`test_training.py`** — **8個の自動テスト**。すべて**CPUで数秒**。一番大事なのは:*止めて再開した学習が、中断しなかった学習とまったく同じ損失になること*(`day9_compare_runs.py` と同じ証明を、自動で行う)。
>
> pytestの基本(`assert`、`parametrize`、`tmp_path`、`pytest.raises`、フィクスチャ)は、Day 6.2、7.2、8.2にあります。ここでは新しいことだけを説明します。

---

## 0. 全体像(まずここから)

| 約束 | テスト |
|---|---|
| 学習率が直線的に上がり(ウォームアップ)、コサインのように下がり、二度と上がらない | `test_lr_schedule_warmup_then_cosine` |
| 重み減衰は行列だけ。すべてのパラメータがちょうど1つのグループにある | `test_weight_decay_only_on_matrices` |
| `y` は `x` を1つずらしたもの。保存したバッチの生成器の状態が、次のバッチを再現する | `test_random_windows_shift_and_resume_state` |
| 検証用の窓が、隙間も重なりもなく列を覆う | `test_validation_windows_cover_the_stream_without_overlap` |
| **止めて再開 = 中断なし** | `test_stopped_and_resumed_run_equals_uninterrupted_run` |
| 設定を変えた再開は**拒否される** | `test_resume_refuses_changed_settings` |
| 既存のフォルダーでの新規の学習は**拒否される** | `test_new_run_refuses_an_existing_folder` |
| **強制終了**の後、再開すると古いログの行が消える | `test_resume_after_a_hard_kill_removes_lines_written_after_the_checkpoint` |

実行(前と同じ):`.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. 準備

```python
from __future__ import annotations

import csv, json, math, sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.batching import RandomWindows, validation_batches, validation_windows  # noqa: E402
from src.model import ModelA, ModelConfig  # noqa: E402
from src.token_data import FORMAT_VERSION, TokenFileMeta, encode_lines, sha256_bytes, write_token_files  # noqa: E402
from src.training import TrainConfig, lr_at, param_groups, train  # noqa: E402

VOCAB = 40
```
- これまでのテストファイルと同じ準備。**Day 9.1**(バッチ)、**Day 8.1**(モデル)、**Day 7.1**(トークンファイル。偽のデータを作るため)、**Day 9.2**(学習)から読み込む。`VOCAB = 40` — 見立ての語彙。

---

## 2. 小さな部品のテスト

### 2.1 `test_lr_schedule_warmup_then_cosine`

```python
def test_lr_schedule_warmup_then_cosine() -> None:
    lrs = [lr_at(s, 1e-3, 10, 100, 0.1) for s in range(101)]
    assert lrs[0] == pytest.approx(1e-4)          # first warm-up step
    assert lrs[9] == pytest.approx(1e-3)          # end of warm-up = peak
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))  # then never rises
    assert lrs[100] == pytest.approx(1e-4)        # ends at min_lr
    assert lrs[55] == pytest.approx(1e-4 + 0.5 * 9e-4, rel=1e-6)  # halfway through the cosine
```
- 最大 `1e-3`、ウォームアップ10、100ステップ、最後の割合0.1で、ステップ0〜100の学習率を計算する。
- ステップ0 = `1e-4`(ウォームアップの最初のステップ = 最大 ÷ 10)。ステップ9 = `1e-3`(ウォームアップの終わり = 最大)。
- `all(a >= b for a, b in zip(lrs[9:], lrs[10:]))` — 各値を**次の値**と比べる:最大の後は、上がらない。
- ステップ100 = `1e-4`(最小)。ステップ55(コサインの中間)= ちょうど中点 `1e-4 + 0.5 × 9e-4`。(`pytest.approx(…, rel=1e-6)` =「0.0001%以内で等しい」。)

### 2.2 `test_weight_decay_only_on_matrices`

```python
def test_weight_decay_only_on_matrices() -> None:
    m = ModelA(ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32))
    decay, no_decay = param_groups(m, 0.1)
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    ids = [id(p) for g in (decay, no_decay) for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))   # every parameter exactly once (tied once)
```
- 小さなモデルを作る。`param_groups` が2つのグループの辞書を返す(`decay`、`no_decay` に展開される)。
- `decay` の中身はすべて**行列**(2次元以上)。`no_decay` の中身はすべて**ベクトル**(1次元 — 正規化の倍率)。
- `ids` — 両方のグループの、すべてのパラメータの同一性。連鎖した確認が求めるのは:**重複がない**こと(`len(ids) == len(set(ids))`)と、**合計がモデルのパラメータ数と等しい**こと — つまり、すべてのパラメータがちょうど1つのグループにある(共有された埋め込みは1回だけ数える)。

### 2.3 `test_random_windows_shift_and_resume_state`

```python
def test_random_windows_shift_and_resume_state() -> None:
    tokens = np.arange(1000, dtype="<u2")
    w = RandomWindows(tokens, block_size=16, batch_size=4, seed=0)
    x, y = w.next_batch()
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])       # y is x shifted by one
    state = w.state_dict()
    expected = w.next_batch()
    w2 = RandomWindows(tokens, block_size=16, batch_size=4, seed=999)
    w2.load_state_dict(state)
    got = w2.next_batch()
    assert torch.equal(expected[0], got[0]) and torch.equal(expected[1], got[1])
```
- 偽の列 `0, 1, 2, …, 999` で、結果が見やすくなる。
- 形は `(4, 16)`。そして `x[:, 1:] == y[:, :-1]` — **`y` は `x` を1つずらしたもの**(次のトークンの学習の核心)。
- 生成器の状態を保存し(`state`)、**次の**バッチ(`expected`)を取る。**別のシード**(999)の**2つ目の生成器**を作り、保存した状態を読み込む。その次のバッチ(`got`)は**まったく同じ**でなければならない — 状態を保存して戻せば将来のバッチが再現されることの証明(再開が頼っていること)。`torch.equal` = 完全に同一のテンソル。

### 2.4 `test_validation_windows_cover_the_stream_without_overlap`

```python
def test_validation_windows_cover_the_stream_without_overlap() -> None:
    tokens = np.arange(100, dtype="<u2")
    assert validation_windows(tokens, 10) == 9
    xs = torch.cat([x for x, _ in validation_batches(tokens, 10, 4)])
    assert xs.shape == (9, 10)
    assert torch.equal(xs.flatten(), torch.arange(90))
```
- 100トークン、ブロック10 → 窓は9個。4個ずつのバッチ(4 + 4 + 1個の窓)から、すべての `x` を `torch.cat` で集める → 形は `(9, 10)`。
- 平らにすると `0, 1, …, 89` と等しいはず — **隙間なし、重なりなし、順番どおり**。(トークン90〜99は、使われない余り。)

---

## 3. 小さな、端から端までの学習

### 3.1 補助

```python
def _make_repo(root: Path) -> None:
    rng = np.random.default_rng(0)
    tok_dir = root / "data/tokens"
    for split, n in (("train", 300), ("val", 40)):
        lines = [" ".join(map(str, rng.integers(4, VOCAB, size=rng.integers(3, 12)))) for _ in range(n)]
        tokens, offsets = encode_lines(lines, lambda s: [int(t) for t in s.split()], 1, 2)
        meta = TokenFileMeta(FORMAT_VERSION, split, VOCAB, "test", 1, 2, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), "test")
        write_token_files(tok_dir, split, tokens, offsets, meta)
    (root / "configs").mkdir()
    ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32, name="test").save(
        root / "configs/model_test.json")
```
- 本物のデータなしで、一時フォルダーの中に**完全なミニプロジェクト**を作る:
  - 300個の学習用の「質問」と40個の検証用の「質問」。それぞれ、4〜39のランダムな数(偽のトークンID)が**3〜11個**並んだ行で、`"17 5 33"` のような文字;
  - `encode_lines(lines, lambda s: [int(t) for t in s.split()], 1, 2)` — Day 7.1の関数に、文字を整数に変えるだけの小さな**自作の「エンコーダー」**(`lambda` = 1行の関数)を渡す。`1` = `<bos>`、`2` = `<eos>`;
  - メタのラベル(仮の指紋 `"test"`)と、`write_token_files` による分割ごとの3つのファイル;
  - 小さなモデルの設定(2層、幅32、ブロック16)を `configs/model_test.json` に保存。

```python
def _cfg(name: str, **kw) -> TrainConfig:
    base = dict(run_dir=f"runs/{name}", model_config="configs/model_test.json", tokenizer=None, prompts=[],
                batch_size=4, max_steps=12, warmup_steps=3, lr=3e-3, log_every=1, eval_every=4, device="cpu",
                precision="fp32", seed=7)
    return TrainConfig(**{**base, **kw})
```
- **CPU、fp32で12ステップ**の学習の設定:バッチ4、ウォームアップ3、4ステップごとに評価、**毎ステップ記録**(`log_every=1`)、`tokenizer=None`(トークナイザーの確認とサンプルの文章を飛ばす)、`prompts=[]`。上書きは `{**base, **kw}` で合わせる(前と同じ技)。

```python
def _losses(root: Path, name: str) -> tuple[dict[int, str], dict[int, str]]:
    with (root / "runs" / name / "log.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return ({int(r["step"]): r["train_loss"] for r in rows if r["event"] == "train"},
            {int(r["step"]): r["val_loss"] for r in rows if r["event"] == "eval"})
```
- 学習の `log.csv` を読み、**ステップ → 損失の文字**の辞書を2つ(学習用、検証用)返す。損失は、書かれた**文字列**のままなので、比べることは「最後に表示された桁まで等しい」ということ。

### 3.2 `test_stopped_and_resumed_run_equals_uninterrupted_run` — 一番大事なテスト

```python
def test_stopped_and_resumed_run_equals_uninterrupted_run(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    full = train(_cfg("a"), tmp_path)
    assert full["step"] == 12 and not full["stopped_early"]

    part = train(_cfg("b", stop_after=6), tmp_path)          # 6 is not an evaluation step: last.pt saved at stop
    assert part["step"] == 6 and part["stopped_early"]
    saved = json.loads((tmp_path / "runs/b/train_config.json").read_text(encoding="utf-8"))
    resumed_cfg = TrainConfig.from_dict(saved)
    resumed_cfg.resume, resumed_cfg.stop_after = True, None
    done = train(resumed_cfg, tmp_path)
    assert done["step"] == 12
```
1. 一時フォルダー(`tmp_path`)にミニプロジェクトを作る。
2. **学習「a」**:止めずに12ステップすべて;ステップ12で終わったか確認する。
3. **学習「b」**:ステップ6で止める(`stop_after=6`)。ステップ6は評価のステップ**ではない**(評価は4、8、12)ので、「停止時に `last.pt` を保存する」処理も試される。ステップ6で早く止まったか確認する。
4. **b を再開**:b の保存済みの設定を読み、`resume = True` にして `stop_after` を消し、もう一度実行 → ステップ12に到達するはず。

```python
    train_a, val_a = _losses(tmp_path, "a")
    train_b, val_b = _losses(tmp_path, "b")
    assert train_a == train_b                                   # every step's loss, to the last printed digit
    assert val_a == val_b
    assert float(val_a[12]) < float(val_a[0])                   # and it learned something
    assert float(val_a[0]) == pytest.approx(math.log(VOCAB), abs=0.3)
    assert (tmp_path / "runs/b/best.pt").exists() and (tmp_path / "runs/b/last.pt").exists()
```
- **すべての学習の損失とすべての検証の損失が、a と b で同一**でなければならない(文字列の辞書の等価比較)。
- テストが意味を持つための確認:最後の検証の損失が**最初のものより低い**(学習した)、最初の損失が約 `ln(40)`(未学習のモデルは一様な当て推量)、そして両方のチェックポイントファイルが存在する。
- CPUのfp32で `log_every=1` なので、等価は厳密になる。

### 3.3 `test_resume_refuses_changed_settings`

```python
def test_resume_refuses_changed_settings(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("c", stop_after=4), tmp_path)
    changed = _cfg("c", lr=1e-2, resume=True)
    with pytest.raises(SystemExit):
        train(changed, tmp_path)
```
- 学習「c」を始めて、ステップ4で止める。そして**違う学習率**(`1e-2`)で再開しようとする。`train` は拒否しなければならない(`SystemExit`)— Day 9.2の§5.3を参照(`RESUME_FREE_KEYS` にない設定は、一致しなければならない)。

### 3.4 `test_new_run_refuses_an_existing_folder`

```python
def test_new_run_refuses_an_existing_folder(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("d", max_steps=2, eval_every=2), tmp_path)
    with pytest.raises(SystemExit):
        train(_cfg("d", max_steps=2, eval_every=2), tmp_path)
```
- 学習「d」(2ステップ)を行う。**同じ名前で別の新しい学習**を始めるのは拒否されなければならない。古い学習が、うっかり上書きされないように。

### 3.5 `test_resume_after_a_hard_kill_removes_lines_written_after_the_checkpoint`

```python
def test_resume_after_a_hard_kill_removes_lines_written_after_the_checkpoint(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("a"), tmp_path)
    train(_cfg("e", stop_after=6), tmp_path)
    log = tmp_path / "runs/e/log.csv"
    with log.open("a", encoding="utf-8", newline="") as f:      # lines a killed run would have left behind
        f.write("7,train,448,0.001,9.9,9.9,,,1.0,1,,1.0\n8,eval,512,0.001,,,9.9,9.9,,,,1.0\n")
    cfg = TrainConfig.from_dict(json.loads((tmp_path / "runs/e/train_config.json").read_text(encoding="utf-8")))
    cfg.resume, cfg.stop_after = True, None
    train(cfg, tmp_path)
    assert _losses(tmp_path, "a") == _losses(tmp_path, "e")
```
- **強制終了**(ウィンドウを閉じた。Ctrl+Cではない)をシミュレートする:
  - 基準として学習「a」を実行。学習「e」はステップ6で止める(チェックポイントはステップ6);
  - eの `log.csv` に、ステップ7と8の**偽の行**を、でたらめな値(`9.9`)で追記する — 強制終了されたプログラムが、最後のチェックポイントの後のステップを記録していたかのように(各偽の行は `LOG_COLUMNS` の12列を持つ);
  - eを最後まで再開する。
- 再開の後、`_truncate_after`(Day 9.2の§4.8)が**偽の行を消している**はずで、学習し直されたステップは、**中断しなかった学習「a」と同じログ**になるはず。(偽の `9.9` の行が残っていれば、辞書が違ってしまう。)

---

## 4. テストはいくつ?

8個のテスト:学習率のスケジュール(1)、重み減衰のグループ(1)、ランダムな窓(1)、検証用の窓(1)、停止と再開の等価(1)、変えた設定の拒否(1)、既存のフォルダーの拒否(1)、強制終了の後始末(1)。期待される結果:`8 passed`。

## 5. このテストが確かめて*いない*こと

- GPU / bf16 / fp16 / 勾配のスケーラー(テストはCPUのfp32で動く)。
- 勾配の累積(`grad_accum > 1`)と、勾配クリッピング(直接は)。
- Ctrl+Cのハンドラー(`_StopFlag`)— 代わりに `stop_after` で停止をシミュレートしている。
- サンプルの文章の生成と、トークナイザーの確認(ここでは `tokenizer=None`)。
- `diverged.pt` の安全停止と、`--overwrite`。

## 6. 一段落でまとめ

この8個のCPUテストは、学習率のスケジュール、重み減衰が行列だけにかかること、学習用の窓が1つずれていてその乱数の状態を保存・復元できること、検証用の窓が列をぴったり敷き詰めることを確かめ、そして一時フォルダーに作った小さな偽のプロジェクトで、止めて再開した学習が中断しなかった学習とまったく同じ損失になること、設定を変えた再開や既存のフォルダーでの新規の学習が拒否されること、強制終了の後に再開すると古いログの行が消えることを確かめます。
