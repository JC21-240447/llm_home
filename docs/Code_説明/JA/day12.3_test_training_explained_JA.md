# Day 12.3 — test_training.py(更新版)— 変更点 + 1行ずつの解説(日本語)

> **これは、Day 9.3 で最初に見た `test_training.py` の更新版です。**
> - **最初の8個のテストは同じ**です。その解説は **`day9.3_test_training_explained_JA.md`**(とEN)にあります。
> - **早期終了のために、新しいテスト2個と新しい補助1個が足されました。**このファイルは、それらを説明します。
>
> **Day 12 の5つのファイルの関係:**`day12_train_main.py`、`day12_monitor.py`、`generation.py`(12.1)、`training.py`(12.2 — 新しい早期終了を持つコード)、**`test_training.py`(12.3、このファイル — それをテストする)**。

---

## 0. 全体像(まずここから)

早期終了は、2つの約束を満たさなければなりません:

1. 検証が良くならなくなったら、**本当に学習を止め、最良のモデルを残す**。
2. **停止と再開の後でも動く** — 中断しなかった学習と**同じステップ**で止まらなければならない。

(1)をテストするには、検証の損失が**必ず悪くなる**状況が必要です。本物のデータはそれを約束できないので、テストは、わざと**仕掛けを施した小さなデータセット**を作ります。

### 仕掛けを施したデータ(なぜ検証が必ず悪くなるか)

| | 使うトークンID |
|---|---|
| **学習用の文章** | **4 〜 9** だけ(6種類のトークン) |
| **検証用の文章** | **20 〜 39** だけ(学習では一度も見ない) |

モデルは、すぐに「トークン4〜9しか存在しない」と学び、トークン20〜39に、ほぼ**ゼロの確率**を与えます。だから**学習の損失は下がるが、検証の損失は上がります**。検証の最良の点は一番最初で、改善のないまま6ステップ進むと、早期終了が**必ず**働きます。

---

## 1. 変わっていない部分

import、`VOCAB = 40`、補助関数 `_make_repo`、`_cfg`、`_losses`、そして最初の8個のテストは、Day 9.3とまったく同じです(学習率のスケジュール、重み減衰のグループ、ランダムな窓、検証用の窓、停止と再開の等価、変えた設定の拒否、既存のフォルダーの拒否、強制終了の後始末)。

---

## 2. 新しい補助:`_make_diverging_repo`

```python
def _make_diverging_repo(root: Path) -> None:
    """Training text uses only token IDs 4-9, validation only 20-39: learning the training text makes validation
    worse, so the validation loss is sure to rise and early stopping must trigger."""
    rng = np.random.default_rng(1)
    tok_dir = root / "data/tokens"
    for split, lo, hi, n in (("train", 4, 10, 300), ("val", 20, 40, 40)):
        lines = [" ".join(map(str, rng.integers(lo, hi, size=8))) for _ in range(n)]
        tokens, offsets = encode_lines(lines, lambda s: [int(t) for t in s.split()], 1, 2)
        meta = TokenFileMeta(FORMAT_VERSION, split, VOCAB, "test", 1, 2, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), "test")
        write_token_files(tok_dir, split, tokens, offsets, meta)
    (root / "configs").mkdir()
    ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32, name="test").save(
        root / "configs/model_test.json")
```
- `_make_repo`(Day 9.3)と同じ考えで、データの作り方が違う:
  - ループは、分割ごとに、**トークンIDの範囲** `lo` 〜 `hi`(上端は含まない:`rng.integers(4, 10)` は4〜9を出す)と行数 `n` を与える:**train** = ID 4〜9、300行;**val** = ID 20〜39、40行;
  - 各行 = **ランダムなID 8個**を文字として書いたもの(`"7 5 9 4 …"`);小さな自作の「エンコーダー」が文字を整数にする。`1` = `<bos>`、`2` = `<eos>`;
  - いつものメタのラベルとファイルに加えて、小さなモデルの設定(2層、幅32、ブロック16)。
- `np.random.default_rng(1)` — 固定シードなので、偽のデータはいつも同じ。

---

## 3. 新しいテスト1:早期終了が学習を終え、最良を残す

```python
def test_early_stopping_ends_the_run_and_keeps_the_best(tmp_path: Path) -> None:
    """Day 12: with a patience, training stops once validation has not improved for that many steps."""
    _make_diverging_repo(tmp_path)
    res = train(_cfg("f", max_steps=60, eval_every=2, lr=1e-2, early_stop_patience=6), tmp_path)
    status = json.loads((tmp_path / "runs/f/status.json").read_text(encoding="utf-8"))
    assert res["early_stopped"] and status["status"] == "early_stopped"
    assert res["step"] < 60 and res["step"] - res["best_step"] >= 6
    _, val = _losses(tmp_path, "f")
    assert float(val[res["best_step"]]) == pytest.approx(res["best_val"], abs=1e-4)
    assert min(float(v) for v in val.values()) == pytest.approx(res["best_val"], abs=1e-4)
```
- 仕掛けのデータを作る。学習は:60ステップのスケジュール、**2ステップごと**に評価、大きな学習率 `1e-2`(モデルが学習用の文章を速く学ぶように)、**patience 6**。
- `status = …` は、Day 12.2が書いた `status.json` を読む。
- **確認1:**結果が `early_stopped` と言い、`status.json` も同意する(`"early_stopped"`)。
- **確認2:**ステップ60より**前**に止まり、最良のステップから少なくとも**6ステップ後**(`res["step"] - res["best_step"] >= 6`)— まさにルールどおり。
- **確認3(最良は本当に最良):**記録された検証の損失を読む(`_losses` が学習用と検証用の辞書を返す。`_` は学習用を無視)。`best_step` で記録された損失は、**記録された最良の値** `res["best_val"]` と等しく、ログ全体の**最小の**検証の損失も、それと等しい。`pytest.approx(..., abs=1e-4)` は、ログが損失を小数4桁の文字として保存するため、ごく小さな差を許す。

---

## 4. 新しいテスト2:早期終了が、停止と再開を越えて働く

```python
def test_early_stopping_survives_stop_and_resume(tmp_path: Path) -> None:
    """The best step is saved in the checkpoint, so a resumed run stops at the same step as a non-stop run."""
    _make_diverging_repo(tmp_path)
    full = train(_cfg("g", max_steps=60, eval_every=2, lr=1e-2, early_stop_patience=6), tmp_path)
    train(_cfg("h", max_steps=60, eval_every=2, lr=1e-2, early_stop_patience=6, stop_after=full["step"] - 3),
          tmp_path)
    cfg = TrainConfig.from_dict(json.loads((tmp_path / "runs/h/train_config.json").read_text(encoding="utf-8")))
    cfg.resume, cfg.stop_after = True, None
    resumed = train(cfg, tmp_path)
    assert (resumed["step"], resumed["best_step"]) == (full["step"], full["best_step"])
    assert _losses(tmp_path, "g") == _losses(tmp_path, "h")
```
- **学習「g」** — **中断しない**基準:あるステップ `full["step"]` で早期終了が働くまで学習する。
- **学習「h」** — **同じ設定**だが、その点の3ステップ前に**わざと中断**する(`stop_after = full["step"] - 3`)。`full["step"]` は評価のステップ(偶数)なので、3を引くと**奇数**のステップ — 評価のステップ*ではない* — になり、「停止時に `last.pt` を保存する」処理(Day 9.2)も試される。
- **hを再開:**保存済みの設定を読み、`resume = True` にし、`stop_after` を消して、もう一度学習する。
- **確認1:**再開した学習が、中断しなかった学習と**同じステップ**で、**同じ `best_step`** で終わる。
- **確認2:**gとhで、記録された**すべての学習と検証の損失が同一**。
- **なぜこれでバグが見つかるか:**もしチェックポイントが `best_step` を保存**しなかったら**、再開した学習はカウンターをやり直し(Day 12.2の§2.4の予備により、`best_step` = 再開したステップになる)、早期終了が学習gより**遅く**働く — 確認が失敗する。だからこのテストは、`best_step` が正しく保存され、復元されることを証明する。

---

## 5. テストはいくつ?

今は**10個のテスト**(Day 9.3の8個 + 新しい2個)。期待される結果:`10 passed`。すべてCPUで数秒。

## 6. このテストが確かめて*いない*こと

- **`early_stop_patience` が `None`**(初期値)のときの早期終了 — それを設定しない、古い8個のテストが間接的に確かめている。
- `best_step` の無い**古いチェックポイント**の読み込み(`.get` の予備)。
- **早期終了の境界**そのもの(`>=`):テストは、最良の点から少なくとも6ステップ後に止まることだけを確かめる。
- `finished` と `stopped` の `status.json`(`early_stopped` だけが確認される)。
- `day12_monitor.py` と `day12_train_main.py` 自体(本物のデータやモデルが必要)。

## 7. 一段落でまとめ

検証が必ず悪くなるように仕掛けた小さなデータセット(学習はトークンID 4〜9、検証は20〜39)に基づく、新しいCPUテスト2個が、早期終了が学習を止め、`status.json = early_stopped` を書き、本当の最良の検証の損失を残し、最良の点から少なくとも `patience` ステップ後に止まること、そして、わざとの停止と再開の後でも、中断しなかった学習とまったく同じステップ、同じ最良のステップ、同一のログで終わることを証明します。
