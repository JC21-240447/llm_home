# Day 9 — day9_train.py — やさしい1行ずつの解説(日本語)

> **Day 9 の5つのファイルの関係**
> - **Day 9.1 `batching.py`** と **Day 9.2 `training.py`** — バッチと学習ループ(先に読んでください)。
> - **Day 9(このファイル)`day9_train.py`** — `train(...)` を呼んで、学習を始める/再開する**コマンドラインのスクリプト**。
> - **Day 9 `day9_compare_runs.py`** — 停止と再開が同じ結果になる証明。
> - **Day 9.3 `test_training.py`** — 自動テスト。

---

## 0. 全体像(まずここから)

`training.py` がすべての処理を持っています。このスクリプトは、**玄関**にすぎません:コマンドラインのオプションを読み、`TrainConfig` にし、`train(...)` を呼び、まとめを表示し、`run_log.md` にメモを追記します。

学習は、それぞれ専用のフォルダー `runs/day9/<run-name>/`(Gitには入れない)にあります。

### できることは3つ

```
# 1. 新しい学習を始める
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_a --max-steps 600

# 2. わざと早く止めて(中断のシミュレーション)、再開する
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --max-steps 600 --stop-after 300
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --resume
```
- **学習中のCtrl+C:**今のステップが終わり、`last.pt` が保存されて、止まる。あとで `--resume` で続ける。
- **再開した**学習は、**すべての設定を、その学習自身の `train_config.json` から**取る。もう一度指定できるのは、`--stop-after`、`--machine`、`--device` だけ。
- bf16の無いGPU(RTX 2060)では、`--precision auto` は、勾配のスケーラー付きのfp16を選ぶ。
- 後日のLinuxサーバー:`.venv/bin/python scripts/day9_train.py --machine server ...`。

2つの「スモーク」学習(中断なしの `smoke_a` と、止めて再開した `smoke_b`)を、`day9_compare_runs.py` が比べます。

---

## 1. 先に知っておきたい用語

用語はすべてDay 9.2のもの(学習、再開、チェックポイント、ログ)。argparseの新しい考え方が1つ:**引数グループ**(`ap.add_argument_group(...)`)— 関連するオプションをまとめる、`--help` の中の見出し。

---

## 2. docstring(説明文)

- 英語 + 日本語。学習のフォルダーの中身(`train_config.json`、`model_config.json`、`log.csv`、`samples.md`、`last.pt`、`best.pt`)、3つの使い方の例、Ctrl+Cの動き、再開のルールを説明している。

---

## 3. importと準備

```python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day9-v1"
```
- 標準のimportと、前の日からの `sys.path` の技。`torch` は、実行ログにバージョンを書くためだけにここでimportしている。**Day 9.2**から:設定のクラス `TrainConfig` と `train` 関数。

---

## 4. `parse_args`

```python
def parse_args() -> argparse.Namespace:
    d = TrainConfig(run_dir="")
    ap = argparse.ArgumentParser(description="Day 9: train or resume Model A.")
    ap.add_argument("--run-name", required=True, help="folder name under --runs-root")
    ap.add_argument("--runs-root", default="runs/day9")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--overwrite", action="store_true", help="start again in an existing run folder")
```
- `d = TrainConfig(run_dir="")` — **初期値を読むためだけ**に作った設定オブジェクト(Day 5の `SplitConfig` と同じ技):初期値は `TrainConfig` に一度だけ書き、ここで使い回す。
- 最初のオプションのグループ(いつも有効):
  - `--run-name`(**必須**)— フォルダー名。`--runs-root` — その親(初期値 `runs/day9`);
  - `--resume` — フラグ:既存の学習を続ける;
  - `--stop-after N` — ステップ N でわざと止める;
  - `--machine`、`--device` — ラベル / デバイスの選択(Day 8と同じ);
  - `--overwrite` — 既にファイルのあるフォルダーで、やり直すことを許す。

```python
    g = ap.add_argument_group("settings of a new run (ignored with --resume)")
    g.add_argument("--model-config", default=d.model_config)
    g.add_argument("--batch-size", type=int, default=d.batch_size)
    g.add_argument("--grad-accum", type=int, default=d.grad_accum)
    g.add_argument("--max-steps", type=int, default=d.max_steps)
    g.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    g.add_argument("--lr", type=float, default=d.lr)
    g.add_argument("--min-lr-ratio", type=float, default=d.min_lr_ratio)
    g.add_argument("--weight-decay", type=float, default=d.weight_decay)
    g.add_argument("--grad-clip", type=float, default=d.grad_clip)
    g.add_argument("--dropout", type=float, default=None)
    g.add_argument("--log-every", type=int, default=d.log_every)
    g.add_argument("--eval-every", type=int, default=d.eval_every)
    g.add_argument("--eval-max-windows", type=int, default=None)
    g.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default=d.precision)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--prompt", action="append", help="prompt for the samples (repeatable)")
    return ap.parse_args()
```
- 2つ目のグループ:**新しい学習の設定**(`--resume` のときは無視される)。すべて `TrainConfig` の項目に対応している(Day 9.2の§3の表を参照)。どの行も同じ形:オプション名、型、そして `d` から取った初期値。(`--dropout` と `--eval-max-windows` の初期値は `None` = 「モデルの設定のものを使う / 全部の窓を使う」。)
- `--prompt` と `action="append"`(Day 7のデモと同じ):何度も指定して、自分のサンプル用の書き出しを決められる。

---

## 5. `build_config`

```python
def build_config(a: argparse.Namespace) -> TrainConfig:
    run_dir = f"{a.runs_root}/{a.run_name}"
    if a.resume:
        saved_path = REPO_ROOT / run_dir / "train_config.json"
        if not saved_path.exists():
            raise SystemExit(f"nothing to resume: {saved_path} does not exist")
        cfg = TrainConfig.from_dict(json.loads(saved_path.read_text(encoding="utf-8")))
        cfg.resume, cfg.stop_after, cfg.machine, cfg.device, cfg.overwrite = True, a.stop_after, a.machine, a.device, False
        return cfg
```
- `run_dir` — 学習のフォルダーを文字列で。
- **再開モード:**その学習の保存済みの `train_config.json` を読み、そこから設定を作り直す(**コマンドラインの設定は無視される**)。そして5つだけを上書きする(タプルの代入):`resume = True`、`stop_after` = 入力したもの(だから **`--stop-after` なしの `--resume` は、古い停止の位置を消して、最後まで走る**)、マシンとデバイスのラベル、`overwrite = False`。

```python
    cfg = TrainConfig(
        run_dir=run_dir, model_config=a.model_config, batch_size=a.batch_size, grad_accum=a.grad_accum,
        max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, min_lr_ratio=a.min_lr_ratio,
        weight_decay=a.weight_decay, grad_clip=a.grad_clip, dropout=a.dropout, log_every=a.log_every,
        eval_every=a.eval_every, eval_max_windows=a.eval_max_windows, precision=a.precision, device=a.device,
        seed=a.seed, stop_after=a.stop_after, machine=a.machine, overwrite=a.overwrite,
    )
    if a.prompt:
        cfg.prompts = a.prompt
    return cfg
```
- **新規モード:**コマンドラインの値から設定を作る。`--prompt` を指定していれば、初期の書き出しを置き換える。

---

## 6. `append_run_log`

```python
def append_run_log(summary: dict, cfg: TrainConfig) -> None:
    status = "stopped early (resume later)" if summary["stopped_early"] else "finished"
    body = [
        f"- Machine label: {cfg.machine}; device: {summary['device']}; precision: {summary['precision']}",
        f"- Script: scripts/day9_train.py ({SCRIPT_VERSION}); PyTorch {torch.__version__}",
        f"- Run: {cfg.run_dir}; model config {cfg.model_config}; {'resumed' if cfg.resume else 'new run'}",
        f"- Step {summary['step']} of {summary['max_steps']} ({status}); best validation loss "
        f"{summary['best_val']:.4f}; total training time {summary['elapsed_s']:.0f} s",
    ]
    with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {cfg.machine} | Day 9 train {Path(cfg.run_dir).name}\n\n"
                + "\n".join(body) + "\n")
```
- 前の日と同じ形:箇条書き(マシン、デバイス、精度、バージョン、どの学習で再開したかどうか、**到達したステップ、状態、最良の検証の損失、合計の時間**)を `run_log.md` に追記する。`Path(cfg.run_dir).name` は、見出しに学習の名前だけを入れる。ここには `--no-log` は無い。

---

## 7. `main`

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    cfg = build_config(a)
    print(f"=== Day 9 train | run {cfg.run_dir} | machine={cfg.machine} | {'resume' if cfg.resume else 'new run'} ===")
    summary = train(cfg, REPO_ROOT)
    print(f"\n{'Stopped early' if summary['stopped_early'] else 'Finished'} at step {summary['step']} of "
          f"{summary['max_steps']}; best validation loss {summary['best_val']:.4f}")
    print(f"Run folder: {summary['run_dir']}")
    append_run_log(summary, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- コンソールの安全対策(前と同じ)、オプションを読み、設定を作り、見出しを表示し、**学習を実行**(`train(cfg, REPO_ROOT)` — 仕事はすべてそこで行われる)、最終的な状態と学習のフォルダーを表示し、実行ログに追記し、0を返す。終わり方はいつもと同じ。

---

## 8. 実行の流れ(順番)

1. オプションを読む → `build_config`(コマンドラインから新規、または保存済みの設定から再開)。
2. `train(...)`(Day 9.2):確認 → 作る →(再開)→ ループ → チェックポイント、ログ、サンプル。
3. まとめを表示 → `run_log.md`。

## 9. 結果の読み方

- **学習中:**`step 100 | loss 4.12 | lr 9.8e-04 | 38,000 tokens/s` のような行と、評価のたびの `validation loss … (perplexity …)`。良くなれば `* best, saved best.pt`。
- **学習の後:**`runs/day9/<run>/log.csv`(曲線)、`samples.md`(評価ごとの生成された文章)を開き、`best.pt` を使う。
- **健全なサイン:**検証の損失が約 `ln(vocab)` から下がり、サンプルが意味不明な文字から、質問らしい文章に変わっていく。

## 10. 知っておくと良い小さなこと

- **再開は、他のオプションを無視する。**`--lr 0.01 --resume` と入力しても、学習率は**変わらない**(保存された設定が勝つ)。設定を変えたければ、新しい学習を始める。
- **新規の学習は、既にファイルのあるフォルダーを拒否する** — 新しい `--run-name`、`--resume`、`--overwrite` のどれかを使う。
- **`--stop-after` はテスト/試しのため。**本番では、普通はCtrl+Cを使う。
- **止めた学習でも、実行ログに項目が書かれる**(状態は「stopped early」)。

## 11. 一段落でまとめ

`day9_train.py` は学習のコマンドラインの玄関です:(新規なら)あなたのオプションから、(再開なら)その学習の保存済みの `train_config.json` から `TrainConfig` を作り、`training.py` の `train(...)` を呼び、結果を表示して、`run_log.md` にメモを追記します。
