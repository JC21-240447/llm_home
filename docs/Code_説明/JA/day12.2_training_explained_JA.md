# Day 12.2 — training.py(更新版)— 変更点 + 1行ずつの解説(日本語)

> **これは、Day 9.2 で最初に見た `training.py` の更新版です。**
> - **ファイルの大部分は同じ**です。その1行ずつの解説は **`day9.2_training_explained_JA.md`**(とEN)にあります:設定、学習率のスケジュール、重み減衰のグループ、評価、チェックポイント、Ctrl+Cの処理、再開、ログ。
> - **Day 12の新しいもの:早期終了 + `status.json`。**このファイルは、変更されたすべての行を説明します。
>
> **Day 12 の5つのファイルの関係:**`day12_train_main.py`(早期終了を使う)、`day12_monitor.py`(早期終了のカウンターと状態を表示)、`generation.py`(12.1)、**`training.py`(12.2、このファイル)**、`test_training.py`(12.3、新しい動作のテスト)。

---

## 0. 全体像(まずここから)

### 問題

Day 11では、一番良い検証の損失が、しばしば**早い時期**(ステップ125〜225)に来て、その後、損失が上がりました(**過学習**)。スケジュールを最後まで学習するのは時間の無駄で、*最後の*チェックポイントは*最良の*ものより悪くなります。

### 早期終了(early stopping)

> 検証の損失が **N ステップ**(**patience**)**改善しなければ**、**学習を止める**。`best.pt` が既に最良のモデルを持っているので、何も失われない。

```
検証の損失:  3.7 → 2.9 → 2.4 → 2.2(最良、ステップ120)→ 2.25 → 2.3 → 2.31 → 2.4 ...
                                              ↑ best_step                 └── 100ステップ改善なし → 停止
```
Day 12の設定での例:10ステップごとに検証、patience 100 → **新しい最良が出ない評価が10回続いたら**止まる。

### `status.json` — 「この学習はどう終わったか?」

学習が終わると、小さなファイル `status.json` を書きます:

| `status` | 意味 |
|---|---|
| `finished` | `max_steps` まで普通に到達した |
| `early_stopped` | 早期終了で止まった |
| `stopped` | Ctrl+Cか `stop_after` で止まった(再開できる) |

Day 12のスクリプトは、このファイルを読んで、シードが**完了**したか、**再開**しなければならないかを決める(Day 11のようにログを解析する代わりに)。

---

## 1. 変わっていない部分(参照の表)

| 部分 | 説明がある場所 |
|---|---|
| `lr_at`、`param_groups`、`pick_device`、`pick_precision`、`autocast_ctx`、`sha256_file` | Day 9.2 §4 |
| `evaluate`、`greedy_continue`、`save_checkpoint`、`load_checkpoint`、`_StopFlag`、`_truncate_after`、`_fmt` | Day 9.2 §4 |
| 設定とデータの照合、モデル/オプティマイザの構築、ログの書き手、メインのループの構造 | Day 9.2 §5 |

---

## 2. 何が変わったか、1つずつ

### 2.1 docstring

新しい段落:*「早期終了(Day 12、任意):`early_stop_patience = N` を指定すると、検証の損失が N ステップ改善しなかったときに学習が終わる。`best.pt` が最良のモデルを保つ。`status.json` が、学習がどう終わったかを記録する:finished、early_stopped、または stopped(Ctrl+C。あとで再開)。」*

### 2.2 `TrainConfig` の新しい設定

```python
    early_stop_patience: int | None = None  # Day 12: stop when validation has not improved for this many steps
```
- **`None` = 早期終了は無し**(初期値)なので、Day 9とDay 11は前と同じに動く。数字を入れるとオンになる。
- `stop_after` の隣にある。*注意:*これは `RESUME_FREE_KEYS` に**入っていない**ので、再開した学習は、最初の学習と**同じpatience**を使わなければならない(さもないと「resume refused」)。

### 2.3 新しい学習は、古い `status.json` も消す

```python
        for p in (last_path, best_path, log_path, samples_path, run_dir / "status.json"):
            p.unlink(missing_ok=True)
```
- フォルダーで**新しい**学習を始めるとき、削除する残りのファイルのリストに `status.json` も入った。だから、古い「finished」のラベルが、新しい学習のものと間違えられることは無い。
- (**再開した**学習は消さない:下の「知っておくと良い小さなこと」を参照。)

### 2.4 最良の点が*どこ*だったかを覚える:`best_step`

```python
    step, best_val, elapsed_before, tokens_seen = 0, float("inf"), 0.0, 0
    best_step = 0
```
- 新しいカウンター `best_step` — **検証の損失が最低になったステップ**(「改善なしのステップ数」を数えるのに必要)。

```python
        step, best_val = st["step"], st["best_val"]
        best_step = st.get("best_step", step)
```
- **再開**するときは、チェックポイントから復元する。`st.get("best_step", step)` — `dict.get(キー, 既定値)`:チェックポイントが**古い**(Day 12より前に作られ、中に `best_step` が無い)場合は、今の `step` を使う。だから古いチェックポイントも読み込める。

```python
            "model_config": mcfg.to_dict(), "train_config": cfg.to_dict(), "best_val": best_val, "best_step": best_step,
```
- `state_for_save`(チェックポイントに入るもの)が、**`best_step`** も保存するようになった。これは大事:無ければ、再開した学習は最良の点がいつだったかを忘れ、早期終了が止まらなかった学習と違う動きをしてしまう(Day 12.3がまさにこれをテストする)。

### 2.5 評価のたびに `best_step` を更新する

```python
    def checkpoint_and_sample(lr: float) -> float:
        nonlocal best_val, best_step
        val = evaluate(...)
        improved = val < best_val
        best_val = min(best_val, val)
        if improved:
            best_step = step
```
- `nonlocal best_val, best_step` — 内側の関数が、外側の2つの変数を**変更**できる(Day 9.2と同じ `nonlocal` の考え方)。
- この評価が**新しい最良**(`improved`)なら、今の `step` を `best_step` として覚える。(関数の残り — ログ、サンプル、`last.pt` / `best.pt` の保存 — は変わらない。)

### 2.6 メインのループの2つの旗

```python
    stopped_early = False
    early_stopped = False
```
- `stopped_early`(既存)= Ctrl+C / `stop_after` で止まった(あとで再開)。**`early_stopped`(新)**= 早期終了で止まった(学習は*完了*しており、再開しない)。

### 2.7 早期終了の確認(変更の心臓部)

```python
                if (cfg.early_stop_patience is not None and step % cfg.eval_every == 0 and step < cfg.max_steps
                        and step - best_step >= cfg.early_stop_patience):
                    early_stopped = True
                    print(f"Early stop at step {step}: no improvement for {step - best_step} steps "
                          f"(best validation loss {best_val:.4f} at step {best_step}; best.pt keeps it).")
                    break
```
これは、各ステップの最後、いつもの評価の**後**にある。4つの条件が**すべて**真でなければならない:

| 条件 | 理由 |
|---|---|
| `cfg.early_stop_patience is not None` | 早期終了が**オン**であること |
| `step % cfg.eval_every == 0` | **評価のステップ**でだけ(`best_step` が最新になっているように) |
| `step < cfg.max_steps` | 最後のステップでは、どのみち普通に終わる |
| `step - best_step >= cfg.early_stop_patience` | **最良の点から、少なくとも N ステップ**経った |

- すべて真なら:`early_stopped = True` にし、メッセージ(改善のなかったステップ数、最良の損失とその場所)を表示し、ループを **`break`** で抜ける。
- この確認は `checkpoint_and_sample` の**後**にあるので、このステップで `last.pt` はすでに保存済み。`best.pt` は最良のモデル。
- *ループの中の順番:*Ctrl+C / `stop_after` のブロックが先(それが働けば、ここに来る前に `break` する)。

### 2.8 ループの後:`status.json` を書く

```python
    status = "stopped" if stopped_early else ("early_stopped" if early_stopped else "finished")
    (run_dir / "status.json").write_text(json.dumps({"status": status, "step": step, "best_step": best_step,
                                                      "best_val": best_val}, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
```
- **状態**を入れ子の条件式で決める:Ctrl+C/`stop_after` → `"stopped"`;そうでなく早期終了 → `"early_stopped"`;それ以外は `"finished"`。
- `{status, step, best_step, best_val}` をJSONとして保存する(LFの改行。いつもと同じ)。
- このコードは `try/finally` の**後**にあるので、普通の終わり方のときに動く。学習が**クラッシュ**したとき(例えば「loss became NaN」の `SystemExit`)や、Ctrl+Cを**2回**押したときは、`status.json` は書かれない — そしてDay 12のスクリプトは「`last.pt` はあるが状態が無い」を**未完了**(再開)として扱う。

### 2.9 より豊かな戻り値

```python
    return {"run_dir": str(run_dir), "step": step, "max_steps": cfg.max_steps, "best_val": best_val,
            "best_step": best_step, "early_stopped": early_stopped, "status": status,
            "stopped_early": stopped_early, "device": device_name, "precision": precision, "last_lr": last_lr,
            "elapsed_s": elapsed_before + time.perf_counter() - t_start}
```
- まとめの辞書に、**`best_step`**、**`early_stopped`**、**`status`** が増えた。(古いキーは変わらないので、Day 9とDay 11のスクリプトは動き続ける。)

---

## 3. 知っておくと良い小さなこと

- **再開した学習は、終わるまで古い `status.json` を保つ。**「新規」の枝は消すが、「再開」の枝は消さない。だから、*再開した*学習が学習している間、モニターは前回の停止の `[stopped]` を表示し続けることがある。学習が終われば更新される。
- **早期終了は、評価のステップしか見ない。**`eval_every = 10`、`patience = 100` なら、最良から100ステップ以上経った、最初の10の倍数のステップで止まる。
- **最良のモデルは決して失われない:**`best.pt` は、止まるかどうかに関係なく、改善のたびに保存される。
- **コサインの学習率のスケジュールは、今も `max_steps` に従う:**早期終了した学習は、スケジュールが最小に届く前に終わるだけ。
- **古いチェックポイントも読める**(`best_step` が今のステップに戻る)が、その場合、「改善なしのステップ数」はそこから数え始める。
- **Day 9とDay 11には影響しない:**`early_stop_patience = None` なら、終わりに `status.json` が書かれること以外は、何も変わらない。

## 4. 一段落でまとめ

`training.py` に、任意の早期終了が加わりました:新しい設定 `early_stop_patience`、評価のたびに更新されてチェックポイントに保存される `best_step` のカウンター(だから再開は正確)、評価のステップごとに、検証の損失がそのステップ数改善していなければループを止める確認、そして学習がどう終わったかをスクリプトが分かるように、最後に書かれる `status.json`(`finished` / `early_stopped` / `stopped`)です。それ以外は、Day 9.2のとおりです。
