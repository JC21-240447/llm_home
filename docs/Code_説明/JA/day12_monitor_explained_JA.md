# Day 12 — day12_monitor.py — やさしい1行ずつの解説(日本語)

> **Day 12 の5つのファイルの関係**
> - **Day 12 `day12_train_main.py`** — 本番の学習(3つのシード、早期終了、最終のモデル)。
> - **Day 12(このファイル)`day12_monitor.py`** — 小さな**読み取り専用のダッシュボード**:動いている(または終わった)学習の様子を見せる。
> - **Day 12.1 `generation.py`**、**Day 12.2 `training.py`**、**Day 12.3 `test_training.py`** — 更新されたライブラリのファイル(それぞれの解説を参照)。
>
> このスクリプトは、学習のコードをimportしません:学習が書くファイルを読むだけです。

---

## 0. 全体像(まずここから)

本番の学習には、時間がかかります。知りたいのは:*「どこまで進んだ? 検証の損失はまだ良くなっている? あとどれくらい?」* — 学習を**邪魔せずに**。

このスクリプトは、**車のダッシュボード**のようなものです:ちらっと見るもので、運転はしません。

- `log.csv` と `train_config.json`(あれば `status.json`)を**読むだけ**。**学習そのものには決して触れない**ので、**学習中に2つ目のターミナルから、安全に実行できます**。

```
.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42
.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5
```
(VS Codeでは **+** ボタンで2つ目のターミナルを開く。`--watch 5` は5秒ごとに更新する。**Ctrl+C** で終わるのはモニターで、学習ではない。)

### 画面の見え方

```
Run runs/day12/main_s42  [running]
  progress   [###########...................] step 150 / 400
  train loss 1.9521 (last)
  val loss   2.1034 at step 150   |   best 2.0877 (perplexity 8.07) at step 130
  early stop 20 / 100 steps without improvement
  speed      38,420 tokens/s   |   time left at most 1,640 s (less if early stopping triggers)
```
(数字は、見た目を示すための作り物。)

---

## 1. 先に知っておきたい用語

| 用語 | かんたんな意味 |
|---|---|
| **モニター** | 観察するだけのプログラム。 |
| **ETA** | 「到着予定時刻」:推定の残り時間。 |
| **patience / 早期終了のカウンター** | 最後の改善から何ステップ経ったか。許される数のうちいくつか(Day 12.2)。 |
| **ANSIエスケープコード** | ターミナルを制御する特別な文字。`\033[2J` = 画面を消す;`\033[H` = カーソルを左上に戻す。 |

---

## 2. importと定数

```python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
```
- 標準のモジュールだけ(`torch` は無い!)なので、モニターは一瞬で起動する。`time` は更新の間の待ちのため、`math` はパープレキシティのため。`REPO_ROOT` は前の日と同じ。

---

## 3. `summary` — 表示する文字を作る

### 3.1 ファイルを読む

```python
def summary(run_dir: Path) -> str:
    log, cfg_path = run_dir / "log.csv", run_dir / "train_config.json"
    if not log.exists() or not cfg_path.exists():
        return f"{run_dir}: no log yet"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    with log.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    evals = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["event"] == "eval" and r["val_loss"]]
    trains = [r for r in rows if r["event"] == "train" and r["train_loss"]]
    if not evals:
        return f"{run_dir}: started, no validation yet"
```
- ログか設定のファイルがまだ無ければ(学習がまだ始まっていない)→ 落ちずに、短いメッセージを返す。
- 保存済みの設定(`cfg`)とログの行を読む。
- `evals` — **eval** の行の `(ステップ, 検証の損失)` の組;`trains` — 損失のある **train** の行(辞書のまま)。
- **検証がまだ無ければ** →「started, no validation yet」(最初の評価はステップ0なので、これはごく短い間)。

### 3.2 数

```python
    step = max(int(r["step"]) for r in rows)
    best_step, best_val = min(evals, key=lambda e: (e[1], e[0]))
    last_val_step, last_val = evals[-1]
    tps = [float(r["tokens_per_sec"]) for r in trains[-5:] if r["tokens_per_sec"]]
    tokens_per_step = cfg["batch_size"] * cfg.get("grad_accum", 1) * 256
    left = cfg["max_steps"] - step
    eta = left * tokens_per_step / (sum(tps) / len(tps)) if tps and left > 0 else 0.0
```
- `step` — **今のステップ** = ログの中で一番大きいステップ番号。
- `best_step, best_val` — 検証の損失の最低値とその場所(同じなら早いステップ。Day 11.1と同じ)。
- `last_val_step, last_val` — 一番新しい検証の結果。
- `tps` — **最後の5つの train の行のトークン/秒**(`trains[-5:]` = 最後の5つ)で、空のセルは無視する。短い窓なので、*最近の*速さになる。
- `tokens_per_step` = バッチ × 累積 × **256**(ブロックサイズ。定数として書き込まれている);`cfg.get("grad_accum", 1)` はキーが無ければ1を使う。
- `left` = 残りのステップ;`eta`(秒)= **まだ処理するトークン ÷ 1秒あたりのトークン数**(`sum(tps)/len(tps)` = 平均の速さ)。速さがまだ無い、または残りが無ければ0。

### 3.3 状態と進行バー

```python
    patience = cfg.get("early_stop_patience")
    status_file = run_dir / "status.json"
    status = json.loads(status_file.read_text(encoding="utf-8"))["status"] if status_file.exists() else "running"
    bar_n = int(30 * step / max(1, cfg["max_steps"]))
```
- `patience` — 早期終了の設定(オフなら `None`)。
- `status` — **`status.json`**(Day 12.2)があればそこから(`finished` / `early_stopped` / `stopped`)、なければ `"running"`。
- `bar_n` — **30個のバーの文字**のうち、いくつを埋めるか:`30 × step ÷ max_steps`(`max(1, …)` はゼロ割りを避ける)。

### 3.4 表示の行

```python
    lines = [
        f"Run {run_dir.relative_to(REPO_ROOT).as_posix() if run_dir.is_relative_to(REPO_ROOT) else run_dir}  [{status}]",
        f"  progress   [{'#' * bar_n}{'.' * (30 - bar_n)}] step {step} / {cfg['max_steps']}",
        f"  train loss {float(trains[-1]['train_loss']):.4f} (last)" if trains else "  train loss -",
        f"  val loss   {last_val:.4f} at step {last_val_step}   |   best {best_val:.4f} "
        f"(perplexity {math.exp(best_val):.2f}) at step {best_step}",
    ]
```
- **1行目:**学習のフォルダー(できればリポジトリからの相対で。個人のフォルダー名が出ないように)と、その **[状態]**。
- **2行目:****進行バー**:`'#' * bar_n` の埋めた文字 + `'.' * (30 - bar_n)` の空の文字、そして「step X / 最大」。
- **3行目:****最後の学習の損失**(`trains[-1]` = 一番新しい train の行)— **条件式**で、まだ無ければ `train loss -` を表示する。
- **4行目:****最新の検証の損失**とそのステップ、そして**最良の**検証の損失(**パープレキシティ** `exp(best_val)` とステップ付き)。

```python
    if patience:
        waited = last_val_step - best_step
        lines.append(f"  early stop {waited} / {patience} steps without improvement"
                     + ("  -> will stop at the next check" if waited >= patience else ""))
    if tps:
        lines.append(f"  speed      {sum(tps) / len(tps):,.0f} tokens/s   |   time left at most {eta:,.0f} s "
                     f"(less if early stopping triggers)")
    return "\n".join(lines)
```
- 早期終了がオンなら:`waited` = 最良の点から**最新の評価**までのステップ数 — **早期終了のカウンター**「waited / patience」。patienceに届くと、追加の文が出る(実際には、すぐ後で状態が `early_stopped` になっているのを見ることが多い)。
- 速さが分かっていれば:**最近の平均の速さ**と、**残り時間の上限**を表示する — 「at most」(最大)なのは、早期終了でもっと早く終わるかもしれないから。
- 行を改行でつなぎ、全体の文字を返す。

---

## 4. `main`

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Day 12: monitor a training run.")
    ap.add_argument("--run", required=True, help="run folder, for example runs/day12/main_s42")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds (0 = print once)")
    a = ap.parse_args()
    run_dir = (REPO_ROOT / a.run).resolve()
```
- コンソールの安全対策は前と同じ。オプションは2つ:`--run`(**必須**:学習のフォルダー)と `--watch N`(更新の間の秒数。`0` = 1回だけ表示)。

```python
    if not a.watch:
        print(summary(run_dir))
        return 0
    try:
        while True:
            print("\033[2J\033[H" + summary(run_dir), flush=True)   # clear the screen, then print
            time.sleep(a.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
```
- **1回モード**(`--watch` が0):まとめを表示して終わり。
- **監視モード:**終わらないループ:`"\033[2J\033[H"`(画面を消してカーソルを左上に戻す)に続けて、新しいまとめを表示し(`flush=True` はすぐ画面に送る)、`time.sleep(N)` する。**Ctrl+C** は `KeyboardInterrupt` を出し、それを捕まえて → モニターは静かに終了コード0で終わる。(ここでCtrl+Cを押して止まるのは、モニターだけ。)

---

## 5. 画面の読み方

- **progress:**スケジュールがどこまで進んだか(早期終了は、100%の前に終わることがある)。
- **val lossとbest:**最新が最良より*高い*なら、今のところ検証は良くなっていない。
- **early stop X / patience:**Xがpatienceに向かって増えているなら、学習がもうすぐ止まるかもしれない。新しい最良が出ると0に戻る。
- **time left at most:**天井であって、約束ではない。
- **[status]:**`running`、または最終的な状態 `finished` / `early_stopped` / `stopped`。

## 6. 知っておくと良い小さなこと

- **`[running]` は間違っていることがある。**状態は、**`status.json` が無い**限り `running` で、学習がクラッシュした、または強制終了された場合もそう。そして**再開した**学習では、*古い* `status.json`(例:`stopped`)が学習の終わりまで残るので、モニターは実際には学習中なのに `[stopped]` と表示することがある。確実にするには、ステップ番号や速さを見る。
- **256は書き込まれている**(ブロックサイズ)— 時間の推定で。
- **`Path.is_relative_to` が必要**(Python 3.9以上)。あなたのvenvなら問題ない。
- **ANSIの画面クリア**は、VS Codeのターミナルや普通のLinuxのターミナルで動く。それを理解しないターミナルでは、変な文字が見える。
- **学習が書いている間に読む**のは、実際には安全:学習は完全な行をflushするから。
- **最後の5つの速さだけ**を平均するので、更新のたびに数字が少し跳ねる。

## 7. 一段落でまとめ

`day12_monitor.py` は読み取り専用のダッシュボードです:学習の `log.csv`、`train_config.json`、`status.json` を読み、進行バー、最新の学習と検証の損失、(パープレキシティとステップ付きの)最良の検証の損失、早期終了のカウンター、最近の速さ、残り時間の上限を、1回、または `--watch` でN秒ごとに更新して表示します。学習には決して触れません。
