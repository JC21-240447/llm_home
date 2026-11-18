# Day 12 — day12_train_main.py — やさしい1行ずつの解説(日本語)

> **Day 12 の5つのファイルの関係**
> - **Day 12(このファイル)`day12_train_main.py`** — **本番の学習**:Day 11の設定、早期終了、**3つのシード**、**最終のモデル1つ**。
> - **Day 12 `day12_monitor.py`** — 2つ目のターミナルから学習を見守る。
> - **Day 12.1 `generation.py`**(`quick_checks` を追加)、**Day 12.2 `training.py`**(早期終了 + `status.json` を追加)、**Day 12.3 `test_training.py`**(早期終了のテスト)。
>
> おすすめの読む順番:**12.2 → 12.1 → 12(このファイル)→ 12(monitor)→ 12.3**。

---

## 0. 全体像(まずここから)

Day 11は「どの大きさとドロップアウト?」に答え、`configs/day11_decision.json` を書きました。Day 12は、その選択での**本番の学習**で、**信頼でき、説明できる**結果を出すように設計されています。

| 設計の選択 | 理由 |
|---|---|
| **決定のファイルを読む** | 大きさとドロップアウトは、記憶ではなく、Day 11の書き下したルールから来る |
| **3つのシード(42、43、44)** | Day 11で、**シード1つでは検証の損失が最大0.06ずれる**と分かった。だから、運の良い1つの数ではなく、**平均とばらつき**を報告する |
| **400ステップのスケジュール** | Day 11の最良の点はステップ125〜225。短いスケジュールなら、その頃には**学習率がすでに下がっている**(800ステップのスケジュールでは、ステップ200でもまだ高い学習率) |
| **10ステップごとの検証** | 最良の点を見逃さないよう、頻繁に確認する |
| **早期終了、patience 100** | 改善のない評価が10回続いたら止める(Day 12.2) |
| **最終のモデル = 検証の損失が最低のシード**(確認に合格したものの中で) | 書き下した、再現できる選び方 |
| **テスト用のデータは使わない** | 最後の、正直な評価(**Day 13**)のために取っておく |

### 書き出されるもの

| 出力 | 中身 |
|---|---|
| `runs/day12/main_s42`、`main_s43`、`main_s44` | シードごとに学習のフォルダー1つ(Gitには入れない) |
| `docs/DAY12_MAIN_TRAINING.md` | 結果、Day 11との比較、サンプルの文 |
| `docs/figures/day12_main.png` | 3つのシードの学習と検証の曲線 |
| `configs/day12_final.json` | **どのチェックポイントが最終のモデルか**(Day 13が読む) |
| `run_log.md` | 項目1つ |

### 再開できる

Ctrl+Cやクラッシュの後は、**同じコマンド**をもう一度実行します:**完了したシードは飛ばされ**、未完了のシードは `last.pt` から続きます。

```
.venv\Scripts\python.exe scripts\day12_train_main.py --machine home
```
2つ目のターミナルから見守る:`.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5`(後日のLinuxサーバー:`.venv/bin/python scripts/day12_train_main.py --machine server`)。

---

## 1. 先に知っておきたい用語

| 用語 | かんたんな意味 |
|---|---|
| **シード** | ランダムな出発点(初期の重みとバッチの順番)。シードが違えば、結果が少し違う。 |
| **平均(mean)** | 平均値。 |
| **ばらつき(spread)** | ここでは:3つの最良の検証の損失の**最大 − 最小**。結果がどれだけ運に左右されるかの、おおまかな尺度。 |
| **早期終了 / patience** | Day 12.2を参照。 |
| **最終のモデル** | これから使う、1つのチェックポイント(1つのシードの `best.pt`)。 |

---

## 2. importと定数

```python
from __future__ import annotations

import argparse, json, math, platform, statistics, sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.generation import PRESET_SETTINGS, load_model, model_logits_fn, quick_checks  # noqa: E402
from src.model import ModelConfig, expected_parameters  # noqa: E402
from src.selection import MAX_REPETITIVE, MIN_FINISHED, RunSummary, summarize_run  # noqa: E402
from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day12-v1"
KEY_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "勤続年数が", "大阪支社で", "課長の", "今日は")
```
- 新しい標準のモジュール:**`statistics`**(`statistics.mean`)。他はすべてプロジェクトから:トークナイザー(**6.1**)、新しい **`quick_checks`** を持つ生成(**12.1**)、モデルの式(**8.1**)、学習のまとめと確認のしきい値 `MIN_FINISHED`/`MAX_REPETITIVE`(**11.1**)、早期終了を持つ学習ループ(**12.2**)。
- `KEY_PROMPTS` — 報告書のサンプルの表に出す6つの書き出し。

---

## 3. ヘルパー関数

### 3.1 `read_prompts`
- Day 10・11と同じ(1行に1プロンプト。空行と `#` のコメントは飛ばす)。

### 3.2 `run_status`

```python
def run_status(run_dir: Path) -> str:
    """'new', 'done' (finished or early-stopped) or 'unfinished' (stopped by Ctrl+C or killed)."""
    status = run_dir / "status.json"
    if status.exists() and json.loads(status.read_text(encoding="utf-8"))["status"] in ("finished", "early_stopped"):
        return "done"
    return "unfinished" if (run_dir / "last.pt").exists() else "new"
```
- シードをどう扱うかを、**`status.json`**(Day 12.2)で決める — Day 11のログの解析より、単純で確実:
  - `status.json` が **finished** か **early_stopped** → **`done`**(飛ばす);
  - そうでなく、`last.pt` があれば → **`unfinished`**(Ctrl+Cで止まった、または強制終了:**再開**);
  - そうでなければ → **`new`**。
- (状態が `stopped` の学習は、わざと中断されて再開できるので、未完了として扱う。)

### 3.3 `draw_figure`

```python
def draw_figure(runs: list[RunSummary], runs_root: Path, path: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for c, r in zip(colours, runs):
        with (runs_root / r.name / "log.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        ev = [(int(x["step"]), float(x["val_loss"])) for x in rows if x["event"] == "eval" and x["val_loss"]
              and int(x["step"]) > 0]
        tr = [(int(x["step"]), float(x["train_loss"])) for x in rows if x["event"] == "train" and x["train_loss"]]
        ax.plot([s for s, _ in tr], [v for _, v in tr], color=c, lw=1.0, alpha=0.45, ls="--")
        ax.plot([s for s, _ in ev], [v for _, v in ev], color=c, lw=1.8, label=f"{r.name} validation")
        ax.plot([r.best_step], [r.best_val], "o", color=c, ms=6)
    ax.set_ylim(0.8, 4.0)
    ...
```
- Day 11と同じ(ファイル専用のバックエンドのmatplotlib。**matplotlibがインストールされている必要がある**)だが、今度は**シードごとに1色**(`zip(colours, runs)`)で、**シードごとに2本の線**:**検証の損失**(実線、太い、凡例のラベル付き)と**学習の損失**(破線、細い、半透明:`alpha=0.45`)、それに**最良の点の丸**。破線と実線の間の隔たりが、丸暗記の差。
- `ax.set_ylim(0.8, 4.0)` — 3つのシードを比べやすいように、y軸の範囲を固定(0.8〜4.0の外の値は切れる)。そして、ラベル、タイトル、グリッド、凡例、保存、閉じる(前と同じ)。

---

## 4. `parse_args`

```python
ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
ap.add_argument("--decision", type=Path, default=REPO_ROOT / "configs/day11_decision.json")
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
ap.add_argument("--max-steps", type=int, default=400)
ap.add_argument("--warmup-steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--patience", type=int, default=100)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--eval-max-windows", type=int, default=None, help="(testing only) fewer validation windows")
ap.add_argument("--runs-root", default="runs/day12")
ap.add_argument("--prompts", ...)  ap.add_argument("--report", ...)  ap.add_argument("--figure", ...)
ap.add_argument("--final", type=Path, default=REPO_ROOT / "configs/day12_final.json")
ap.add_argument("--no-log", action="store_true")
```
- セクション0の設定が初期値。`--seeds` は複数の整数を受け取る(`nargs="+", type=int`)。`--patience` = 早期終了のpatience。`--final` = 最終のモデルのファイルの書き出し先。他はDay 11と同じ。**再開するには、最初のときと同じオプションを使う。**

---

## 5. `main`

### 5.1 決定を読む

```python
    a = parse_args()
    decision = json.loads(a.decision.read_text(encoding="utf-8"))
    size, dropout, model_config = decision["size"], float(decision["dropout"]), decision["model_config"]
    runs_root = REPO_ROOT / a.runs_root
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Day 12 main training | {size}, dropout {dropout} (from {a.decision.name}) | seeds {a.seeds} | "
          f"{device_name} ===")
```
- **Day 11の決定のファイル**を読み、大きさ、ドロップアウト、モデルの設定のパスを取る。デバイスはDay 11と同じ(入れ子の条件式)。設定がどこから来たかを示す見出しを表示する。

### 5.2 手順1 — シードごとに学習

```python
    statuses: dict[str, dict] = {}
    for i, seed in enumerate(a.seeds, 1):
        name = f"main_s{seed}"
        run_dir = runs_root / name
        state = run_status(run_dir)
        print(f"\n[{i}/{len(a.seeds)}] {name}: {state}")
        if state != "done":
            cfg = TrainConfig(
                run_dir=f"{a.runs_root}/{name}", model_config=model_config, batch_size=a.batch_size,
                max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, dropout=dropout,
                log_every=a.eval_every, eval_every=a.eval_every, eval_max_windows=a.eval_max_windows,
                early_stop_patience=a.patience, prompts=[], device=a.device, seed=seed, machine=a.machine,
                resume=(state == "unfinished"),
            )
            result = train(cfg, REPO_ROOT)
            if result["stopped_early"]:
                print("\nStopped. Run the same command again to continue from here.")
                return 0
        statuses[name] = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
```
- シードごとに:`main_s42` などと名前を付け、状態を調べ、**完了**なら飛ばす。
- そうでなければ `TrainConfig` を作る:**`seed` 以外はすべてのシードで同一の設定**;`log_every = eval_every = 10`;**`early_stop_patience=a.patience`**(Day 12.2で新しい);`prompts=[]`(学習中はサンプルの文章を作らない);未完了なら `resume`。
- `train(cfg, REPO_ROOT)` が実行する。**中断された**(`stopped_early`、つまりCtrl+C)なら、ヒントを表示して**終了**する。同じコマンドで、あとで続く。(**早期終了した**学習は `stopped_early = False` なので、ループは次のシードに進む。)
- そして、そのシードの **`status.json`**(どう終わったか、どのステップか)を `statuses` に読み込む — 飛ばされたシードについても行う。

### 5.3 手順2 — まとめと生成の確認(テスト用は使わない)

```python
    tok = ByteBPETokenizer.load(REPO_ROOT / "data/tokenizer/tokenizer.json")
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    params = expected_parameters(ModelConfig.load(REPO_ROOT / model_config))
    runs: list[RunSummary] = []
    checks: dict[str, dict] = {}
    print("\n--- generation check on each best.pt ---")
    for seed in a.seeds:
        name = f"main_s{seed}"
        r = summarize_run(runs_root / name / "log.csv", name, size, dropout, params)
        model, _ = load_model(runs_root / name / "best.pt", device)
        fn = model_logits_fn(model, device)
        checks[name] = {s: quick_checks(fn, tok, prompts, PRESET_SETTINGS[s], train_set)
                        for s in ("greedy+limits", "sample")}
        gl = checks[name]["greedy+limits"]
        r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
        runs.append(r)
        print(f"  {name}: best val {r.best_val:.4f} at step {r.best_step} ({statuses[name]['status']} at step "
              f"{statuses[name]['step']}) | finished {gl['finished']:.0%} | repetitive {gl['repetitive']:.0%} | "
              f"distinct-2 {gl['distinct2']:.2f}")
        del model
```
- トークナイザー、プロンプト、学習用の質問(「new」のため)、そして式からの**パラメータ数**を読み込む。
- シードごとに:学習のまとめ(**Day 11.1**)、その **`best.pt`** の読み込み、そして **Day 12.1の `quick_checks`** で、2つの設定(`greedy+limits`、`sample`)の**作業6の確認**。貪欲の数がまとめに入る。シードごとに1行表示:最良の損失とステップ、**どう終わったか**、3つの確認の数。モデルを解放する。

### 5.4 統計と最終のモデル

```python
    vals = [r.best_val for r in runs]
    mean_val = statistics.mean(vals)
    spread = max(vals) - min(vals)
    passing = [r for r in runs if r.passes_checks()]
    final = min(passing or runs, key=lambda r: r.best_val)
    d11 = [decision.get("best_val_seed42"), decision.get("best_val_seed43")]
    d11 = [v for v in d11 if v is not None]
    d11_mean = statistics.mean(d11) if d11 else None
```
- `vals` — 各シードの最良の検証の損失。`mean_val` — その**平均**。`spread` — **最大 − 最小**。
- `passing` — 一番良いチェックポイントが、作業6の目安に合格したシード(`passes_checks`、Day 11.1:完結 90%以上、繰り返し 5%以下)。
- **`final`** — **合格したものの中で、検証の損失が最低**のシード。`passing or runs` は:**どれも**合格しなければ、**すべて**のシードに戻る(下で警告が出る)、という意味。
- `d11` — 同じ設定の**Day 11の試験の値**を、決定のファイルの2つの**任意のキー** `best_val_seed42` と `best_val_seed43` から読む(`dict.get` は無ければ `None`);あるものだけを残し、`d11_mean` = その平均、または `None`。*(あなたが貼ったDay 11のスクリプトは、この2つのキーを書かない。Day 11の追加のシードの後などに、ファイルに自分で足していなければ、Day 11との比較は `-` になる。)*

```python
    print(f"\nBest validation loss per seed: {', '.join(f'{v:.4f}' for v in vals)}")
    print(f"Mean {mean_val:.4f} (perplexity {math.exp(mean_val):.2f}), spread {spread:.4f}")
    if d11_mean is not None:
        print(f"Day 11 trials of the same configuration (800-step schedule): mean {d11_mean:.4f} -> "
              f"change {mean_val - d11_mean:+.4f}")
    print(f"Final model: {final.name} (best val {final.best_val:.4f} at step {final.best_step})"
          + ("" if final.passes_checks() else "  WARNING: does not pass the task 6 checks"))
```
- シードごとの値、**平均(パープレキシティ付き)とばらつき**、(あれば)**Day 11との変化**(`:+.4f` は符号を必ず表示する)、そして**最終のモデル**を表示する — 選ばれたものでさえ確認に不合格なら、**WARNING** 付きで。

### 5.5 手順3 — 出力

```python
    draw_figure(runs, runs_root, a.figure)
    final_ckpt = f"{a.runs_root}/{final.name}/best.pt"
    a.final.parent.mkdir(parents=True, exist_ok=True)
    a.final.write_text(json.dumps({
        "final_checkpoint": final_ckpt, "run": final.name, "size": size, "dropout": dropout,
        "model_config": model_config, "parameters": params, "best_val": round(final.best_val, 4),
        "best_step": final.best_step, "seeds": a.seeds, "best_val_per_seed": [round(v, 4) for v in vals],
        "mean_best_val": round(mean_val, 4), "spread": round(spread, 4), "test_set_used": False,
        "decided": f"{datetime.now():%Y-%m-%d %H:%M}", "script": SCRIPT_VERSION,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
```
- 図を描く。そして **`configs/day12_final.json`** を書く — **Day 13への引き継ぎのファイル**:**最終のチェックポイントのパス**、学習、大きさ、ドロップアウト、モデルの設定、パラメータ数、最良の損失とステップ、**すべてのシードとその最良の損失**、**平均とばらつき**、**`"test_set_used": False`**(テスト用のデータに触れなかったという、記録された約束)、日時、スクリプトのバージョン。

```python
    write_report(a, runs, statuses, checks, final, mean_val, spread, d11, d11_mean, device_name, size, dropout,
                 params, len(prompts))
    print(f"\nReport: {a.report}\nFigure: {a.figure}\nFinal model file: {a.final}  ->  {final_ckpt}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(...)   # 見出し;スクリプト/デバイス/PyTorch;設定;シードごとの1行;平均/ばらつき;最終のモデル
        print("Appended an entry to run_log.md")
    return 0
```
- 報告書を書き(セクション6)、場所を表示し、実行ログに追記する(設定、各シードの最良の損失とどう終わったか、平均/ばらつき、最終のチェックポイント)。`return 0`。

---

## 6. `_cell` と `write_report`

```python
def _cell(text: str) -> str:
    """Make a generated sentence safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")
```
- 生成された文を、Markdownの表の中で安全にする:`|` をエスケープし、改行を空白にし、`<` を `&lt;` にする。`<e5ae>` や `<eos>` のような断片が、**HTMLのタグと間違えられて**消えないように。

`write_report` が `docs/DAY12_MAIN_TRAINING.md` を作る(図のパスは、前と同じく、報告書からの相対にする)。セクション:

| セクション | 内容 |
|---|---|
| ヘッダー | バージョン、日付、マシン、デバイス、バージョン |
| 設定 | 「Configuration from Day 11: **大きさ、ドロップアウト**(Nパラメータ)」、batch × 256、**400ステップのスケジュール**(ウォームアップ、最大の学習率、10%までのコサイン)、検証の間隔、**早期終了のpatience**、シード;「**テスト用のデータは使わなかった。**」 |
| **Final model** | `best.pt` のパス、最良の検証の損失、パープレキシティ、ステップ;決まった書き出し(greedy + limits)での**finished / repetitive / distinct-2** を、目安(`MIN_FINISHED`、`MAX_REPETITIVE`)の隣に |
| 図 | 曲線 |
| **All seeds** | シードごとに1行:**どう終わったか**(「early_stopped at step 230」)、最良の検証の損失、パープレキシティ、最良のステップ、そこでの学習の損失、**gap**、finished、repetitive、distinct-2、そして**新しい文**を `new/終えた数`(`round(finished_rate × n_prompts)` が終えた出力の数になる)として;続いて**平均とばらつき** |
| **Compared with Day 11** | 表:Day 11(800ステップ、シードの数、平均)とDay 12(Nステップ + 早期終了、3シード、平均);そして「**Change: ±x。上のシードのばらつきと照らして読むこと:ばらつきより小さい変化は、改善の証拠ではない。**」(Day 11の数が無ければ空) |
| サンプルの文 | 最終のモデルの、`KEY_PROMPTS` の文を、greedy + limits とサンプリングで(`_cell` を適用) |

- 「ばらつきより小さい変化は証拠ではない」という文が、正直な統計の要点:シードのばらつきが例えば0.06なら、0.02の違いは、ただの運かもしれない。

---

## 7. 実行の流れ(順番)

1. Day 11の決定(大きさ、ドロップアウト、モデルの設定)を読む。
2. シードごとに:完了なら飛ばし、未完了なら再開し、そうでなければ学習する(同じ設定、早期終了あり)。
3. シードごとに:ログをまとめ、`best.pt` で生成を確かめる(`quick_checks`)。
4. 平均とばらつきを計算し、**最終のシード**(確認に合格したものの中で検証の損失が最低)を選ぶ。
5. 図 → `configs/day12_final.json` → 報告書 → 実行ログ。

## 8. 結果の読み方

- **平均とばらつき:**正直な結果は「平均 ± ばらつき」。例:検証の損失 2.05、ばらつき 0.04(パープレキシティ ≈ 7.8)。
- **どう終わったか:**`early_stopped at step X` は、モデルが早くピークに達して、その後過学習したことを示す。`finished` は、スケジュールの終わりまで良くなり続けたという意味。
- **Gapの列:**大きい = 丸暗記。Day 11と比べる。
- **最終のモデル:**ファイル `configs/day12_final.json` が、Day 13に、**テスト用のデータ**で(一度だけ)評価するチェックポイントを教える。
- **WARNINGの行:**どのシードも生成の確認に合格しなかった — 先に進む前に調べる。

## 9. 知っておくと良い小さなこと

- **検証の損失で3つのシードから一番良いものを選ぶのは、少し楽観的**で、だから**テスト用のデータ**をDay 13のために取っておく。
- **再開には同じオプションが必要。**Day 12.2は、違う設定(違う `--patience` を含む)での再開を拒否する。
- **Day 11との比較には、決定のファイルに2つの任意のキー**(`best_val_seed42`、`best_val_seed43`)**が必要**。無ければ、比較は飛ばされる(`-`)。
- **いくつかの数がコードに書き込まれている:**報告書の文の「800 steps」、モニター/時間の推定の中のブロックサイズ `256`、図のyの範囲 0.8〜4.0。
- **図の色は5色**なので、シードが5つより多いと、余分な学習は描かれない。
- **ばらつき = 最大 − 最小**で、標準偏差ではない。3シードなら十分。
- **`stopped_early`**(Ctrl+C)はスクリプトを抜けるが、**早期終了**は抜けない — そのシードの、普通の、完全な終わり方。
- **matplotlibが必要。**

## 10. 一段落でまとめ

`day12_train_main.py` は、Day 11で選んだ大きさとドロップアウトを、3つのシード、400ステップのスケジュール、10ステップごとの検証、改善なしの100ステップ後の早期終了で学習し(再開できて、シードごとに `status.json` がある)、各シードの一番良いチェックポイントを決まった書き出しで確かめ、最良の検証の損失の平均とばらつき、Day 11との変化を報告し、確認に合格した中で損失が最低のシードを最終のモデルとして選び、曲線の図、Markdownの報告書、`configs/day12_final.json` を書きます — すべて、Day 13のために取っておく、テスト用のデータには触れずに。
