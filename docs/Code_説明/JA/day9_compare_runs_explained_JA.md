# Day 9 — day9_compare_runs.py — やさしい1行ずつの解説(日本語)

> **Day 9 の5つのファイルの関係**
> - **Day 9.1 `batching.py`** と **Day 9.2 `training.py`** — バッチと学習ループ。
> - **Day 9 `day9_train.py`** — 学習を始める/再開する。
> - **Day 9(このファイル)`day9_compare_runs.py`** — 停止と再開が動く**証明**:終わった2つの学習を比べる。
> - **Day 9.3 `test_training.py`** — 自動テスト(同じ証明を、単体テストとして含んでいる)。

---

## 0. 全体像(まずここから)

同じモデルを、2回学習しました:

| 学習 | 何が起こったか |
|---|---|
| **a**(`smoke_a`) | 止めずに、最後まで通して学習した |
| **b**(`smoke_b`) | `stop_after`(例えばステップ300)まで学習し、**止めて**、最後まで**再開した** |

再開が正しく実装されていれば、**b は a とまったく同じに見える**はずです:どのステップでも、損失が同じ。もしチェックポイントが何か(オプティマイザの記憶、乱数の状態など)を忘れていれば、2つの曲線はずれていきます。このスクリプトは、両方の `log.csv` を読み、**4つのこと**を確かめます:

1. 両方の学習が、同じ最後のステップに到達した;
2. 記録したすべてのステップで、**学習の損失**が両方で同じ;
3. **検証の損失**が同じ;
4. 学習 b の曲線に、再開したステップで**飛び**が無い。

そして絵(`docs/figures/day9_smoke_loss.png`)を描き、学習の結果と、最初と最後の評価のサンプルの文章を載せた報告書(`docs/DAY9_SMOKE_REPORT.md`)を書きます。

実行:

```
.venv\Scripts\python.exe scripts\day9_compare_runs.py --a smoke_a --b smoke_b
```

---

## 1. 先に知っておきたい用語

| 用語 | かんたんな意味 |
|---|---|
| **中断なしの学習** | 一度も止めなかった学習(基準)。 |
| **再開のステップ** | 学習 b を止めたステップ(その `stop_after`)。 |
| **許容範囲(tolerance)** | どれくらいの差までを「同じ」と認めるか。 |
| **Matplotlib** | 図を描くのに使うPythonのライブラリ。 |
| **バックエンド `Agg`** | ウィンドウを開かず、直接画像ファイルに描く。 |
| **パープレキシティ** | `exp(損失)`(Day 9.2を参照)。 |

---

## 2. docstring、import、定数

```python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

SAME_TOL: float = 1e-6        # "identical"
CONTINUOUS_TOL: float = 0.02  # "the same curve" (GPU arithmetic may differ in the last digits)
```
- 標準のimport(それに、Day 5と同じ、正規表現用の `re`)。`matplotlib` は、後で `main` の中でimportする。
- 許容範囲が2つ:
  - `SAME_TOL = 1e-6` — これより小さい差 = **「同一」**(CPUでは普通);
  - `CONTINUOUS_TOL = 0.02` — これより小さい差 = **「同じ曲線」**(GPUでは、計算の最後の桁が違うことがあり、小さな差が、多くのステップで少し大きくなりうる)。

---

## 3. ヘルパー関数

### 3.1 `read_log`

```python
def read_log(path: Path) -> tuple[dict[int, float], dict[int, float], list[dict]]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    train = {int(r["step"]): float(r["train_loss"]) for r in rows if r["event"] == "train" and r["train_loss"]}
    val = {int(r["step"]): float(r["val_loss"]) for r in rows if r["event"] == "eval" and r["val_loss"]}
    return train, val, rows
```
- 学習の `log.csv`(Day 9.2の12列のファイル)を、辞書のリストとして読む。
- `train` — **ステップ → 学習の損失**の辞書。`event` が `"train"` **かつ**損失のセルが空でない行から作る(`and r["train_loss"]`。空文字は偽として扱われる)。`val` — `"eval"` の行と検証の損失について同じ。
- 2つの辞書と、生の行を返す。

### 3.2 `sample_sections`

```python
def sample_sections(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [s.strip() for s in re.split(r"\n(?=## step )", text) if s.strip().startswith("## step")]
```
- `samples.md` を、評価ごとに1つのセクションに分ける。`re.split(r"\n(?=## step )", text)` は、`## step ` の**直前の**改行ごとに切る(`(?=…)` の「先読み」は、それらの文字を取り除かずに先を見るので、各セクションは見出しを保つ)。本当に `## step` で始まる部分だけを残す。

---

## 4. `main` — 比較

### 4.1 オプションと読み込み

```python
    ap = argparse.ArgumentParser(description="Day 9: compare an uninterrupted and a resumed run.")
    ap.add_argument("--a", required=True, help="run name of the uninterrupted run")
    ap.add_argument("--b", required=True, help="run name of the stopped-and-resumed run")
    ap.add_argument("--runs-root", default="runs/day9")
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day9_smoke_loss.png")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY9_SMOKE_REPORT.md")
    a = ap.parse_args()
```
- `--a`、`--b`(**必須**)— 2つの学習の名前。`--runs-root`、`--figure`、`--report` — 初期値付きの場所。

```python
    dir_a, dir_b = REPO_ROOT / a.runs_root / a.a, REPO_ROOT / a.runs_root / a.b
    train_a, val_a, rows_a = read_log(dir_a / "log.csv")
    train_b, val_b, _ = read_log(dir_b / "log.csv")
    cfg_a = json.loads((dir_a / "train_config.json").read_text(encoding="utf-8"))
    cfg_b = json.loads((dir_b / "train_config.json").read_text(encoding="utf-8"))
```
- 両方のログと、両方の保存済みの設定を読む。(`a.a` は `--a` の学習の名前。`a` はargparseの名前空間。)

```python
    resume_step = cfg_b.get("stop_after")
    if resume_step is None:
        raise SystemExit(f"{a.b} was not stopped early (no stop_after in its train_config.json)")
    keys = [k for k in cfg_a if k not in {"run_dir", "stop_after", "resume", "overwrite", "machine", "device"}]
    differing = [k for k in keys if cfg_a.get(k) != cfg_b.get(k)]
    if differing:
        raise SystemExit(f"the two runs use different settings: {differing}")
```
- `resume_step` — 学習 b を止めたステップ。**b の保存済みの設定**から読む(再開は元の `train_config.json` を保つので、`stop_after` が記録されたまま残る)。無ければ、b は中断した学習ではない → 止まる。
- **公平なテストの確認:**違ってよい設定(`run_dir`、`stop_after`、`resume`、`overwrite`、`machine`、`device`)**以外**のすべての設定が、両方の学習で**同一**でなければならない。そうでなければ、比較は無意味 → 止まって、違うキーを挙げる。

### 4.2 数

```python
    common = sorted(set(train_a) & set(train_b))
    diffs = [abs(train_a[s] - train_b[s]) for s in common]
    vcommon = sorted(set(val_a) & set(val_b))
    vdiffs = [abs(val_a[s] - val_b[s]) for s in vcommon]
    max_diff = max(diffs + vdiffs) if diffs else float("nan")
    final_a, final_b = max(train_a), max(train_b)
    before = max(s for s in train_b if s <= resume_step)
    after = min(s for s in train_b if s > resume_step)
    jump_b = abs(train_b[after] - train_b[before])
    jump_a = abs(train_a.get(after, math.nan) - train_a.get(before, math.nan))
```
- `common` — **両方の**学習にある、記録されたステップ(`&` = 集合の共通部分)。`diffs` — それぞれでの学習の損失の**差の絶対値**。`vcommon` / `vdiffs` — 検証の損失について同じ。
- `max_diff` — すべて(学習と検証)の**最大の差**。共通のものが無ければ `NaN`。
- `final_a`、`final_b` — 各学習の、最後に記録されたステップ。
- **飛びの確認:**`before` = 学習 b で、再開のステップ**以前**の、最後に記録されたステップ。`after` = それより**後**の、最初に記録されたステップ。`jump_b` = その2つのステップの間で、学習 b の損失がどれだけ変わったか(停止/再開をまたぐ)。`jump_a` = 中断なしの学習で、**同じ2つのステップ**の間の変化(`.get(..., math.nan)` は、ステップが無くても落ちないため)。再開がきれいなら、2つの飛びは同じ。

### 4.3 4つの確認と判定

```python
    checks = {
        "same final step": final_a == final_b == cfg_a["max_steps"],
        "training losses match": max(diffs) < CONTINUOUS_TOL,
        "validation losses match": max(vdiffs) < CONTINUOUS_TOL,
        "no jump at the resume step": abs(jump_b - jump_a) < CONTINUOUS_TOL,
    }
    verdict = "identical" if max_diff < SAME_TOL else ("the same curve" if all(checks.values()) else "DIFFERENT")
```
- 4つの確認を、True/Falseの辞書として:
  1. 両方の学習が、設定された **`max_steps`** で終わった(連鎖した比較 `a == b == c`);
  2. 学習の損失の最大の差が `0.02` 未満;
  3. 検証の損失の最大の差が `0.02` 未満;
  4. 再開のステップでの損失の飛びが、両方の学習で同じ(`0.02` 以内)。
- **判定:**差が `1e-6` 未満 → **「identical(同一)」**。そうでなく、4つの確認がすべて合格なら → **「the same curve(同じ曲線)」**(GPUで普通)。それ以外は **「DIFFERENT(違う)」**。

---

## 5. 図

```python
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
```
- matplotlib を**ここで**importする(図のためだけに必要。**インストールされている必要がある** — Day 1のパッケージの一覧には無かった)。`matplotlib.use("Agg")` は**ファイル専用のバックエンド**を選ぶ:ウィンドウは開かない(サーバーでは大事)。出力先のフォルダーがあることを確認し、図(`fig`)と、1組の軸(`ax`)を作る。8 × 4.5インチ、150 dpi。

```python
    ax.plot(list(train_a), list(train_a.values()), lw=2.4, color="#9db7d5", label=f"training loss, {a.a} (no stop)")
    ax.plot(list(train_b), list(train_b.values()), lw=1.2, ls="--", color="#1f4e79",
            label=f"training loss, {a.b} (stopped and resumed)")
    ax.plot(list(val_a), list(val_a.values()), "o", ms=6, color="#e3a8a0", label=f"validation loss, {a.a}")
    ax.plot(list(val_b), list(val_b.values()), "x", ms=6, color="#a33a2c", label=f"validation loss, {a.b}")
```
- 4つの線/印:x = ステップ、y = 損失。
  - 学習 a の学習の損失:**太い薄青の線**;
  - 学習 b の学習の損失:**細い破線の濃い青**を、**上に重ねる** — 2つの学習が同一なら、破線は太い線にぴったり乗る;
  - 検証の損失:丸(a)と、バツ(b)。

```python
    ax.axvline(resume_step, color="gray", lw=1, ls=":")
    ax.text(resume_step, ax.get_ylim()[1] * 0.97, f"  stopped / resumed at step {resume_step}", va="top",
            fontsize=8, color="gray")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title("Day 9 smoke run: uninterrupted vs stopped and resumed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.figure)
    plt.close(fig)
```
- 再開のステップに点線の縦線を引き、上のほうにラベルを付ける(`ax.get_ylim()[1] * 0.97` = y軸の上端の97%)。軸ラベル、タイトル、薄いグリッド、凡例。`tight_layout` ははみ出しを防ぎ、`savefig` がPNGを書き、`plt.close` がメモリを解放する。

---

## 6. 報告書

```python
    train_meta = json.loads((REPO_ROOT / "data/tokens/train_meta.json").read_text(encoding="utf-8"))
    last_row = [r for r in rows_a if r["event"] == "train"][-1]
    tps = [float(r["tokens_per_sec"]) for r in rows_a if r["event"] == "train" and r["tokens_per_sec"]]
    epochs = int(last_row["tokens_seen"]) / int(train_meta["tokens"])
    secs = sample_sections(dir_a / "samples.md")
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
```
- 報告書のためのデータ(学習 a から):
  - Day 7のメタ(学習用のトークン数);
  - `last_row` — **最後の学習の行**(`[-1]` = 最後の項目):ここまでに見たトークン数、GPUメモリなど;
  - `tps` — 1秒あたりのトークン数のすべて(空でないもの);
  - `epochs` = 見たトークン数 ÷ 学習用のトークン数 = **学習用の文章を何周したか**;
  - `secs` — サンプルのセクション;
  - `fig_rel` — **報告書のフォルダーから見た**図のパスを、`/` の区切りで(`as_posix()`)。Markdownの画像のリンクが動くように。

Markdownの `lines` には次が入る:

| セクション | 内容 |
|---|---|
| ヘッダー | 日付;どの学習か;モデルの設定、batch × accum、ステップ数、最大の学習率、ウォームアップ |
| 図 | `![loss curves](…)` |
| **Resume check** | 表:4つの確認それぞれ → PASS/FAIL、**最大の差**、**判定** |
| **Learning** | 「ステップ0と最後」の表:検証の損失、**検証のパープレキシティ**、学習の損失。`val_a[min(val_a)]` = **一番早い**評価(ステップ0、学習前)の損失、`val_a[max(val_a)]` = **一番遅い**もの |
| 学習の事実 | 見たトークン数とデータを何周したか;速さの**中央値**(`sorted(tps)[len(tps)//2]`);GPUメモリの最大値または「- (CPU run)」 |
| **Greedy samples** | 学習 a の、最初と最後の評価のサンプルのセクション(`[secs[0], "", secs[-1]]`。2つ未満なら全部) |

- `*(… for k, v in checks.items())` は、確認ごとに表の1行を展開する(前の日と同じ技)。

```python
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```
- 報告書を保存する(UTF-8、LFの改行)。

---

## 7. 表示と終了コード

```python
    print(f"Final step: {a.a} {final_a}, {a.b} {final_b} (resumed at {resume_step})")
    for k, v in checks.items():
        print(f"  {k:<28} {'PASS' if v else 'FAIL'}")
    print(f"Largest difference between the runs: {max_diff:.2e} -> {verdict}")
    print(f"Validation loss: {val_a[min(val_a)]:.4f} (step 0) -> {val_a[max(val_a)]:.4f} (step {max(val_a)})")
    print(f"Figure: {a.figure}\nReport: {a.report}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
```
- 最後のステップ、確認ごとのPASS/FAILの行(`:<28` = 28文字で左寄せ)、判定付きの最大の差、ステップ0から最後までの検証の損失、ファイルの場所を表示する。
- **4つの確認がすべて合格したときだけ終了コード0**(そうでなければ1)— 自動の確認として使える。

---

## 8. 結果の読み方

- **判定「identical」**(最大の差 < 1e-6):再開した学習が、中断しなかった学習を、そっくり再現している。CPUでは普通。
- **「the same curve」**:差がごく小さく(< 0.02)、4つの確認がすべて合格。GPUでは普通。
- **「DIFFERENT」**:再開が壊れている(または、学習の設定やデータが違った)。どの確認が失敗したかを見る。
- **図:**破線が、太い線の上にぴったり乗り、バツが丸の中に収まっているはず。
- **Learningの表:**検証の損失が、ステップ0の約 `ln(vocab)` から下がり、パープレキシティが急に下がる。
- **サンプル:**最初の評価の文章は意味不明で、最後のものは質問らしく見えるはず。

## 9. 知っておくと良い小さなこと

- **比べるのはログで、チェックポイントではない。**ログは両方の学習が書いたものなので、ログが同じなら、学習が同じということ。
- **`log_every` の倍数で止めること**(Day 9.2の§7を参照)。さもないと、再開後の最初の記録の区間が少ないステップの平均になり、学習は同一でも、いくつかの `train_loss` が学習 a と違うことがある。
- **共通の記録されたステップが無い**と、`max(diffs)` がPythonのエラーで失敗する。普通のスモークの設定では起こらない。
- **matplotlibが必要**(venvに `pip install matplotlib`)。Day 1で確認したパッケージの一覧には入っていない。
- **GPUでの学習**は、最後の桁が違うことがある(GPUの計算が非決定的)ので、許容範囲がある。

## 10. 一段落でまとめ

このスクリプトは、中断しなかった学習と、止めて再開した学習のログを読み、同じ最後のステップに到達したこと、記録したすべてのステップで学習と検証の損失が一致すること、再開のステップに飛びが無いことを確かめ、結果を「identical」/「the same curve」/「DIFFERENT」と判定し、2つの曲線を1つの図に描き、学習の結果と、最初と最後のサンプルの文章を載せたMarkdownの報告書を書き、すべて合格したときだけ0で終了します。
