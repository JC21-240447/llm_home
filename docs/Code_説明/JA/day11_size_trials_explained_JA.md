# Day 11 — day11_size_trials.py — やさしい1行ずつの解説(日本語)

> **Day 11 の3つのファイルの関係**
> - **Day 11.1 `selection.py`** — 決めるルール(**先に読んでください**)。
> - **Day 11(このファイル)`day11_size_trials.py`** — **運転手**:(大きさ, ドロップアウト)のすべてを同じ量で学習し、各best checkpointで生成を確かめ、`selection.py` を呼んで決め、報告書、図、決定のファイルを書く。
> - **Day 11.2 `test_selection.py`** — 決めるルールのテスト。

---

## 0. 全体像(まずここから)

Day 11の問い:**本番の学習(Day 12)には、Model Aのどの大きさ(tiny / small / medium)と、どのドロップアウト(0.0 / 0.1)を使うべきか?**

スクリプトは、公平な実験を行います:

```
(大きさ, ドロップアウト)ごとに  →  初期設定で6個の学習:
        まったく同じ設定で学習(違うのは大きさとドロップアウトだけ)
そして学習ごとに:
        一番良いチェックポイント(検証の損失が最低)を取る → 12の決まった書き出しで生成を確かめる
そして:
        書き下したルール(selection.py)を適用 → 1つ選ぶ → 決定を保存
```

### 「同じ予算」= 公平な競走

| 設定 | 値 |
|---|---|
| バッチ | 窓32個 × 256トークン(1ステップ8,192トークン) |
| ステップ | 800(1学習あたり約650万トークン。学習用の文章を約3周) |
| ウォームアップ | 60ステップ |
| 最大の学習率 | 0.001 |
| シード | 42(すべての学習で、同じ最初のバッチの順番) |
| 検証 | **25ステップごと**(Day 9で、一番良い点が最初の数百ステップで来うると分かったので、頻繁に確認する必要がある) |

### 出力

| ファイル | 中身 |
|---|---|
| `runs/day11/<size>_d<dropout>/` | 試験ごとにDay 9の学習のフォルダー1つ(例:`tiny_d00`、`medium_d10`。Gitには入れない) |
| `docs/DAY11_SIZE_TRIALS.md` | 結果の表、決定とその理由、サンプルの文 |
| `docs/figures/day11_val_loss.png` | すべての学習の検証の曲線 |
| `configs/day11_decision.json` | 選ばれた大きさとドロップアウト(**Day 12が読む**) |
| `run_log.md` | 項目1つ |

### 再開できる

Ctrl+Cやクラッシュの後は、同じコマンドをもう一度実行します:**終わった学習は飛ばされ**、**終わっていない学習は `last.pt` から続きます**。

```
.venv\Scripts\python.exe scripts\day11_size_trials.py --machine home
```
(後日のLinuxサーバー:`.venv/bin/python scripts/day11_size_trials.py --machine server`。)

---

## 1. 先に知っておきたい用語

ほとんどの用語は、Day 11.1(検証の損失、最良のステップ、差、許容範囲)とDay 9.2(学習、チェックポイント、再開)のもの。Pythonの新しい考え方:**`nargs="+"`** — `argparse` のオプションが**1つ以上の値**を受け取れるようにする(例:`--sizes tiny small`)。

---

## 2. importと定数

```python
from __future__ import annotations

import argparse, json, math, platform, sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import EOS_ID, ByteBPETokenizer  # noqa: E402
from src.generation import (PRESET_SETTINGS, distinct_2, generate, is_repetitive, load_model, model_logits_fn)  # noqa: E402
from src.model import ModelConfig, expected_parameters  # noqa: E402
from src.selection import TOLERANCE, RunSummary, choose, summarize_run  # noqa: E402
from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day11-v1"
KEY_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "勤続年数が", "今日は")
```
- いつもの準備。これまで作ったものをほぼすべて再利用する:トークナイザー(**Day 6.1**)、生成エンジン(**Day 10.1**)、モデルのパラメータの式(**Day 8.1**)、決めるロジック(**Day 11.1**)、学習ループ(**Day 9.2**)。
- `KEY_PROMPTS` — 報告書のサンプルの文の表に出す4つの書き出し(プロンプトファイルの行にあるべき。無ければ表に `-` が出る)。

---

## 3. ヘルパー関数

### 3.1 `run_name`

```python
def run_name(size: str, dropout: float) -> str:
    return f"{size}_d{int(round(dropout * 100)):02d}"
```
- 学習のフォルダーの名前を付ける:`tiny` + 0.0 → `tiny_d00`;`medium` + 0.1 → `medium_d10`。`round(dropout * 100)` は浮動小数点の驚きを避ける(`0.1 × 100` は `10.000000000000002`)。`:02d` = 0埋めの2桁。

### 3.2 `read_prompts`

```python
def read_prompts(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
```
- Day 10と同じ:1行に1プロンプト。空行と `#` のコメントは飛ばす。

### 3.3 `run_state`

```python
def run_state(run_dir: Path, max_steps: int) -> str:
    """'new', 'finished' or 'unfinished' (judged from the log: the last evaluation must be at max_steps)."""
    log = run_dir / "log.csv"
    if not (run_dir / "last.pt").exists() or not log.exists():
        return "new"
    steps = [int(line.split(",", 1)[0]) for line in log.read_text(encoding="utf-8").splitlines()[1:]
             if ",eval," in line]
    return "finished" if steps and steps[-1] >= max_steps else "unfinished"
```
- 試験をどう扱うかを決める:**`new`**(チェックポイントもログもまだ無い)、**`finished`**、**`unfinished`**。
- ログを単なる文字として読む:ヘッダーを飛ばし(`[1:]`)、**eval** の行(`,eval,` を含む行)ごとに最初の項目(ステップ)を取る — `line.split(",", 1)[0]` は最初のカンマで切る。**最後の評価が `max_steps` に届いていれば** → finished(Day 9.2は最後のステップでいつも評価する)。そうでなければ unfinished。

### 3.4 `generation_check`

```python
def generation_check(ckpt: Path, tok: ByteBPETokenizer, prompts: list[str], device: torch.device,
                     train_set: set[str]) -> dict:
    model, _ = load_model(ckpt, device)
    fn = model_logits_fn(model, device)
    out: dict = {}
    for setting in ("greedy+limits", "sample"):
        gens = [generate(fn, tok, p, PRESET_SETTINGS[setting]) for p in prompts]
        n = len(gens)
        out[setting] = {
            "finished": sum(g.finished for g in gens) / n,
            "repetitive": sum(is_repetitive(tok.encode(g.prompt) + [i for i in g.ids if i != EOS_ID])
                              for g in gens) / n,
            "distinct2": distinct_2([[i for i in g.ids if i != EOS_ID] for g in gens]),
            "new": sum((g.prompt + g.text) not in train_set for g in gens if g.finished),
            "texts": {g.prompt: (g.text + (" ⟨eos⟩" if g.finished else " …")) for g in gens},
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out
```
- 学習の **best checkpoint** を読み込み(Day 10.1)、**2つの設定**(`greedy+limits` と `sample`。Day 10.1のプリセット)で、すべてのプロンプトの続きを生成する。
- 設定ごとに**作業6の自動の項目**を記録する(Day 10の報告書と同じ式):
  - `finished` — `<eos>` で終わった出力の**割合**;
  - `repetitive` — どの本物の質問よりも繰り返しが多い出力の割合(先に `EOS_ID` を除く);
  - `distinct2` — 多様さ;
  - `new` — 終えた出力のうち、学習用の質問ではないものの**個数**(参考用のみ);
  - `texts` — プロンプトごとの生成された文。`⟨eos⟩`(完結)か `…`(切れた)の印付きで、報告書用。
- 最後にモデルとGPUメモリを解放する(`del model`、`empty_cache`)ので、次の学習がきれいな状態で始まる。
- (決定には **greedy+limits** の数が使われる。`sample` は参考表示。)

### 3.5 `draw_figure`

```python
def draw_figure(runs: list[RunSummary], root: Path, path: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"tiny": "#1f77b4", "small": "#2ca02c", "medium": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for r in runs:
        with (root / r.name / "log.csv").open(encoding="utf-8", newline="") as f:
            ev = [(int(x["step"]), float(x["val_loss"])) for x in csv.DictReader(f)
                  if x["event"] == "eval" and x["val_loss"] and int(x["step"]) > 0]
        ax.plot([s for s, _ in ev], [v for _, v in ev], color=colours.get(r.size, "gray"),
                ls="-" if r.dropout == 0 else "--", lw=1.6, label=f"{r.size}, dropout {r.dropout}")
        ax.plot([r.best_step], [r.best_val], "o", color=colours.get(r.size, "gray"), ms=5)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("Day 11 size trials: validation loss (dots = lowest point of each run)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
```
- **すべての検証の曲線を1枚の絵に**描く(Day 9のcompareスクリプトと同じ、ファイル専用の `Agg` バックエンドのmatplotlib — **matplotlibがインストールされている必要がある**)。
- **色 = 大きさ**(tinyは青、smallは緑、mediumは赤。未知の大きさは灰色);**線の種類 = ドロップアウト**(0なら実線、それ以外は破線)。ステップ0より後の評価だけを描く(ステップ0は ≈ `ln(vocab)` で、絵がつぶれてしまうため)。**点**が、各学習の最良の点を示す。
- そしてラベル、タイトル、グリッド、凡例、保存、閉じる。

---

## 4. `parse_args`

```python
ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
ap.add_argument("--sizes", nargs="+", default=["tiny", "small", "medium"])
ap.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.1])
ap.add_argument("--max-steps", type=int, default=800)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--warmup-steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--eval-max-windows", type=int, default=None, help="(testing only) fewer validation windows")
ap.add_argument("--runs-root", default="runs/day11")
ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY11_SIZE_TRIALS.md")
ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day11_val_loss.png")
ap.add_argument("--decision", type=Path, default=REPO_ROOT / "configs/day11_decision.json")
ap.add_argument("--no-log", action="store_true")
```
- 実験の設定で、初期値はセクション0に書いたとおり。`--sizes` と `--dropouts` は**複数の値**を受け取る(`nargs="+"`)。例:`--sizes tiny small --dropouts 0.0`。`--eval-max-windows` は簡単なテスト用。`Path` のオプションと `--no-log` は前と同じ。
- **再開で大事なこと:**終わっていない学習を続けるには、最初のときと**同じオプション**を使う(Day 9.2は、違う設定での再開を拒否する)。

---

## 5. `main`

### 5.1 準備

```python
    a = parse_args()
    runs_root = REPO_ROOT / a.runs_root
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    trials = [(s, d) for s in a.sizes for d in a.dropouts]
    print(f"=== Day 11 size trials | machine={a.machine} | {device_name} | {len(trials)} runs x {a.max_steps} steps ===")
```
- `device` は**入れ子の条件式**:`auto` かつGPUがあれば → `cuda`;`auto` でそれ以外 → `cpu`;デバイスを明示したら、それを使う。
- `trials` — 大きさ × ドロップアウトの**すべての組み合わせ**(2重ループの内包表記:3 × 2 = 6組。順番は tiny 0.0、tiny 0.1、small 0.0、…)。

### 5.2 手順1 — すべての試験を学習する

```python
    for i, (size, dropout) in enumerate(trials, 1):
        name = run_name(size, dropout)
        run_dir = runs_root / name
        state = run_state(run_dir, a.max_steps)
        print(f"\n[{i}/{len(trials)}] {name}: {state}")
        if state == "finished":
            continue
        cfg = TrainConfig(
            run_dir=f"{a.runs_root}/{name}", model_config=f"configs/model_{size}.json", batch_size=a.batch_size,
            max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, dropout=dropout,
            log_every=a.eval_every, eval_every=a.eval_every, eval_max_windows=a.eval_max_windows,
            prompts=[], device=a.device, seed=a.seed, machine=a.machine, resume=(state == "unfinished"),
        )
        summary = train(cfg, REPO_ROOT)
        if summary["stopped_early"]:
            print("\nStopped. Run the same command again to continue from here.")
            return 0
```
- 試験ごとに:名前を付け、状態を見て、`[3/6] small_d00: new` と表示し、**終わっていれば飛ばす**。
- そうでなければ、Day 9.2の `TrainConfig` を作る:
  - モデルの設定は `configs/model_<size>.json`(Day 8);
  - すべての学習で、(大きさと)`dropout` 以外は**同一の設定**(バッチ、ステップ、ウォームアップ、学習率、シード);
  - `log_every = eval_every = 25` → 25ステップごとに、学習の損失の行と評価の行(だから、Day 11.1が最良のステップでの学習の損失を見つけられる);
  - `prompts=[]` → **学習中はサンプルの文章を作らない**(速い。生成は、あとで best checkpoint で確かめる);
  - `resume=(state == "unfinished")` → 終わっていない学習は続け、そうでなければ新しく始める。
- `train(cfg, REPO_ROOT)`(Day 9.2)が学習全体を行う。**早く止まった**(Ctrl+Cを押した)なら、ヒントを表示して**終了**する。同じコマンドをもう一度実行すれば続く。
- すべての学習で**同じシード**なので、どの学習も、**同じ順番の学習用の窓**を見る — 比較を公平にする、もう1つの工夫。

### 5.3 手順2 — まとめ、生成を確かめ、決める

```python
    tok = ByteBPETokenizer.load(REPO_ROOT / "data/tokenizer/tokenizer.json")
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    runs: list[RunSummary] = []
    gen: dict[str, dict] = {}
    print("\n--- generation check on each best.pt ---")
    for size, dropout in trials:
        name = run_name(size, dropout)
        params = expected_parameters(ModelConfig.load(REPO_ROOT / f"configs/model_{size}.json"))
        r = summarize_run(runs_root / name / "log.csv", name, size, dropout, params)
        g = generation_check(runs_root / name / "best.pt", tok, prompts, device, train_set)
        gl = g["greedy+limits"]
        r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
        runs.append(r)
        gen[name] = g
        print(f"  {name:<10} best val {r.best_val:.4f} at step {r.best_step:>4} | finished {gl['finished']:.0%} | "
              f"repetitive {gl['repetitive']:.0%} | distinct-2 {gl['distinct2']:.2f}")
    decision = choose(runs)
```
- トークナイザー、プロンプト、**学習用の質問の集合**(「新しい」の数のため)を読み込む。
- 試験ごとに:**式**からのパラメータ数(Day 8.1)、ログからの学習の**まとめ**(Day 11.1)、その `best.pt` での**生成の確認**。確認の `greedy+limits` の割合を、**まとめにコピー**する(`r.finished_rate = …`)— 最初は `None` だった項目。学習ごとに1行表示する。
- `choose(runs)`(Day 11.1)が、書き下したルールを適用する。

### 5.4 手順3 — 表示、図、報告書、決定のファイル、ログ

```python
    print(f"\n{'run':<10} {'params':>11} {'best val':>9} {'ppl':>6} {'step':>5} {'train@best':>10} "
          f"{'gap':>6} {'final val':>9} {'tok/s':>9} {'peak MB':>8}")
    for r in runs:
        print(...)  # 学習ごとに、そろえた1行。無い数は nan / 0 で表示
    print(f"\nDecision: {decision.chosen.name if decision.chosen else 'NONE'} - {decision.reason}")
```
- 画面に、整った結果の表(書式のコードは前と同じ。`math.exp(best_val)` = パープレキシティ;`float('nan')` / `0` が無い値の代わり)と、理由付きの決定を表示する。

```python
    draw_figure(runs, runs_root, a.figure)
    write_report(a, runs, gen, decision, device_name, prompts)
    if decision.chosen:
        c = decision.chosen
        a.decision.parent.mkdir(parents=True, exist_ok=True)
        a.decision.write_text(json.dumps({
            "chosen_run": c.name, "size": c.size, "dropout": c.dropout, "model_config": f"configs/model_{c.size}.json",
            "parameters": c.parameters, "best_val": round(c.best_val, 4), "best_step": c.best_step,
            "tokens_per_step": a.batch_size * 256, "trial_steps": a.max_steps, "tolerance": TOLERANCE,
            "reason": decision.reason, "decided": f"{datetime.now():%Y-%m-%d %H:%M}", "script": SCRIPT_VERSION,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
```
- 図を描き、報告書を書く(セクション6)。
- **学習が選ばれたら**、`configs/day11_decision.json` を書く — **Day 12への引き継ぎのファイル**:選ばれた学習の名前、大きさ、ドロップアウト、モデルの設定のパス、パラメータ数、最良の検証の損失とステップ、1ステップのトークン数(`batch × 256`。256 = ブロックサイズ)、試験のステップ数、許容範囲、**理由の文**、日時、スクリプトのバージョン。誰も選ばれなければ、ファイルは書かれない。

```python
    print(f"Report: {a.report}\nFigure: {a.figure}\nDecision file: {a.decision}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(...)   # 見出し;スクリプト/デバイス/PyTorch;試験の一覧;学習ごとの1行(最良の損失、ステップ、最後);決定
        print("Appended an entry to run_log.md")
    return 0
```
- 場所を表示し、実行ログに追記する(試験の一覧、各学習の最良と最後の検証の損失、理由付きの決定)。`return 0`。

---

## 6. `write_report` — Markdownの報告書

```python
def write_report(a, runs, gen, decision, device_name, prompts) -> None:
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
    c = decision.chosen
    md = [ ... ]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")
```
- `fig_rel` — 報告書のフォルダーから見た図のパス。画像のリンクが動くように(Day 9のcompareスクリプトと同じ技)。報告書のセクション:

| セクション | 内容 |
|---|---|
| ヘッダー | バージョン、日付、マシン、デバイス、PyTorch/Python;「Every run: batch …、ステップ、ウォームアップ、学習率、シード、検証はNステップごと。**違うのは大きさとドロップアウトだけ。**」 |
| **Decision** | `**Chosen: <run>**` に、大きさ、ドロップアウト、パラメータ、最良の検証の損失、パープレキシティ、ステップを添える — または「No run chosen」(条件式:選ばれた学習の文は、学習がある場合にだけ作られる);**理由**;**書き出されたルール**(決まった N 個の書き出しで greedy + limits が、完結 90%以上、繰り返し 5%以下;許容範囲内の学習は同じくらい良い;パラメータが少ないほうが勝つ);**許容範囲内だった学習**と**却下された学習とその理由** |
| 図 | 検証の曲線の絵 |
| **Results** | 学習ごとに1行:パラメータ、**最良の検証の損失**、パープレキシティ、最良のステップ、そこでの学習の損失、**gap**、最後の検証の損失、**最良の後の上昇**、トークン/秒、ピークMB、完結、繰り返し、distinct-2(無い値は `-`) |
| 説明 | **Gap** = 最良のステップでの、検証の損失 − 学習の損失(「見たことのない文型が、どれだけ難しいか」);**Rise after best** = 最後 − 最良(「その後、どれだけ過学習したか」);生成の3つの項目 |
| **サンプルの文** | 2つの表(**greedy + limits**と、温度0.8 / top-k 50 / top-p 0.95の**サンプリング**):行 = 学習、列 = 4つの `KEY_PROMPTS`、セル = ⟨eos⟩ か … 付きの生成された文(`|` はエスケープ。その書き出しがプロンプトファイルに無ければ `-`) |

- `*( … for r in runs)` の展開が、学習ごとに表の1行を作る(前の日と同じ)。

---

## 7. `__main__`

```python
if __name__ == "__main__":
    sys.exit(main())
```
- いつもと同じ終わり方。(`write_report` はファイルの中で `main` の*後*に定義されているが、`main` が動くときに呼ばれるだけなので問題ない。)

---

## 8. 実行の流れ(順番)

1. オプションを読む → 6つの試験を並べる。
2. 試験ごとに:終わっていれば飛ばし、終わっていなければ再開し、そうでなければ最初から学習する(Day 9.2の `train`)— **すべて同じ設定で**。
3. 試験ごとに:ログをまとめ、`best.pt` で生成を確かめる。
4. `choose(...)` が書き下したルールを適用する。
5. 表と決定を表示 → 図 → 報告書 → `configs/day11_decision.json` → 実行ログ。

## 9. 結果の読み方

- **最良の検証の損失 / パープレキシティ:**低いほど汎化が良い。約0.02以内は「同じ」として比べる。
- **ステップ:**最良のステップが800よりずっと前なら、その後に**過学習**した。**最良の後の上昇**がその大きさを示す。
- **Gap:**大きな差(検証が学習よりずっと高い)= 丸暗記。ドロップアウトは、それを小さくするはず。
- **完結 / 繰り返し:**90%を下回る/5%を上回る学習は、損失に関係なく**却下**される。
- **トークン/秒、ピークMB:**各大きさの、速さとメモリのコスト(選ばれなかった大きなモデルは、Day 12では遅いかもしれない)。
- **決定のファイル:**`configs/day11_decision.json` を開く — Day 12が使うもの。

## 10. 知っておくと良い小さなこと

- **再開には同じ引数が必要。**終わっていない学習は、オプションが最初と一致するときだけ再開する。そうでなければ、Day 9.2が拒否する。
- **作りかけのフォルダー。**最初のチェックポイントの前に学習が強制終了されると、`last.pt` が無いフォルダーだけが残ることがある。状態は `new` になるが、Day 9.2は、既にファイルのあるフォルダーでの「新規」の学習を拒否する — その学習のフォルダーを削除する(または新しい `--runs-root` を使う)してやり直す。
- **設定ごとに1回の学習(シード1つ)。**だから、許容範囲0.02は、ゆれについての仮定であり、このスクリプトが測ったものではない。
- **`generation_check` にはトークナイザーの指紋の確認が無い**(`load_model` を直接使う)。トークナイザーは、学習時に `train`(Day 9.2)が、Day 10では `load_checked` が確認している。ここでは、学習が使ったものと同じトークナイザーのファイル。
- **決定のファイルの `tokens_per_step` は `batch_size × 256`**で、256はブロックサイズとして書き込まれている。
- **選ぶのは検証のデータだけ。**テスト用の分割はここでは触れないので、後で正直な最終スコアを出せる(検証の損失で複数の学習から一番良いものを選ぶのは、少し楽観的になる)。
- **`texts` はプロンプトをキーにしている**ので、プロンプトファイルに同じプロンプトが重複すると、報告書の中で上書きされる。
- **図にはmatplotlibが必要。**

## 11. 一段落でまとめ

`day11_size_trials.py` は、Model Aを3つの大きさ、ドロップアウト0.0と0.1で学習し(同一の設定とシードで6個の再開できる学習、25ステップごとに検証)、各学習の best checkpoint を12の決まった書き出しで生成を確かめ、前もって書かれたルール(完結・繰り返しの確認に合格し、損失が0.02以内なら同じとみなし、パラメータが一番少ないものを取る)を `selection.py` に適用させ、そして検証の曲線の図、決定と理由を載せたMarkdownの報告書、Day 12用の `configs/day11_decision.json` を保存します。
