# Day 8 — day8_model_check.py — やさしい1行ずつの解説(日本語)

> **Day 8 の3つのファイルの関係**
> - **Day 8.1 `model.py`** — モデル本体(**先に読んでください**。このスクリプトは、それを作って動かします)。
> - **Day 8(このファイル)`day8_model_check.py`** — *運転手*:Model Aを3つの大きさで作り、時間を測り、実際のデータで確かめ、設定を保存し、報告する。
> - **Day 8.2 `test_model.py`** — モデルの自動テスト。

---

## 0. 全体像(まずここから)

学習(Day 9)に何時間もかける前に、次の3つを確かめておきたいものです:

1. **各候補のモデルはどれくらい大きく、このGPUでどれくらい速いか?**
2. **モデルは正しくつながっているか?**(間違ってつながったモデルは「学習」しても何も学ばない)
3. **各大きさに、十分なデータがあるか?**

このスクリプトは、飛行機の**飛行前点検**のようなものです:

| 確認 | 何が起こるか | 何の証明になるか |
|---|---|---|
| **大きさ** | tiny / small / medium について、パラメータを数え、*1パラメータあたりの学習トークン数*を計算 | 各モデルが、どれだけデータを「欲しがる」かが分かる |
| **速さとメモリ** | 実際のブロックサイズで、学習1ステップ(順伝播+逆伝播)の時間を測る | モデルが収まるか、学習にどれくらいかかるかが分かる |
| **初期の損失** | 学習前の損失 ≈ `ln(語彙サイズ)` | モデルはまだ何も知らないので、一様に当て推量するはず |
| **2つのAttentionが一致** | 手書きの版 = 速い版 | 速いコードが、読みやすいコードと同じ計算をしている証明 |
| **1バッチを覚える** | 実際の1バッチでの損失が0近くまで下がる | モデル + 損失 + 勾配 + オプティマイザが、一緒に動く証明 |

そして3つの設定(`configs/model_*.json`)をDay 9・Day 11用に保存し、`docs/DAY8_MODEL_REPORT.md` を書き、`run_log.md` にメモを追記します。

実行:

```
.venv\Scripts\python.exe scripts\day8_model_check.py --machine home
```

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **順伝播(forward pass)** | データがモデルを通って、予測と損失ができること。 |
| **逆伝播(backward pass)** | モデルが**勾配**を求めること:各パラメータについて「損失を下げるには、どちらへ変えるべきか?」。 |
| **学習ステップ** | 順伝播 + 逆伝播(+ 実際の学習ではオプティマイザの更新)を1回。 |
| **オプティマイザ(AdamW)** | 勾配を使って、実際にパラメータを変える決まり。 |
| **バッチ** | まとめて処理する、いくつかの列。 |
| **過学習 / 丸暗記(overfit / memorise)** | 少数の例を、そっくり覚えること。ここでは**テスト**としてわざと使う:正しいモデルなら、これが*できなければならない*。 |
| **1パラメータあたりの学習トークン数** | `学習トークン数 ÷ パラメータ数`。学習できる数1つあたりに、どれだけデータがあるかの、おおよその目安。 |
| **1秒あたりのトークン数** | 速さ:1秒に何トークン処理できるか。 |
| **GPUメモリの最大使用量** | ある瞬間に使われたGPUメモリの最大。 |
| **bf16 / fp16 / fp32** | 数の形式(Day 1を参照)。bf16は新しめのGPU(compute capability 8以上)が必要。 |
| **Autocast** | 自動混合精度(Day 1を参照)。 |
| **`torch.device`** | 計算をどこで行うかを表す:`cuda`(GPU)か `cpu`。 |

---

## 2. docstring(説明文)

- 英語 + 日本語、4つの手順、使い方、そして**サーバーの注意**:パラメータ数はどこでも同じで、速さとメモリはGPUによる。**bf16の無いGPU(例:RTX 2060)では、時間測定はfp16、丸暗記の確認はfp32を使う** — スクリプトが自分で選ぶ。

---

## 3. importと準備

```python
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
```
- これまでの日と同じ。加えて:
  - `nullcontext` — 「何もしない」`with` ブロック(下の `autocast` を参照);
  - `numpy as np`(Day 7.1と同じ)と `torch`(Day 8.1と同じ)。

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.model import PRESETS, ModelA, count_parameters, expected_parameters, preset_config  # noqa: E402

SCRIPT_VERSION: str = "day8-v1"
```
- Day 6・7と同じ `sys.path` の技。トークナイザー(語彙サイズのため)と、**Day 8.1**から、モデル、プリセット、数える補助関数を読み込む。

---

## 4. `SizeReport` データクラス

```python
@dataclass
class SizeReport:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    total: int
    embedding: int
    non_embedding: int
    tokens_per_param: float
    ms_per_step: float
    tokens_per_sec: float
    peak_mem_mb: float
```
- モデルの大きさごとの「結果カード」(Day 6・7と同じ考え方):プリセット名、層/ヘッド/幅、パラメータ数(合計 / 埋め込み / 埋め込み以外)、1パラメータあたりの学習トークン数、1ステップのミリ秒、1秒あたりのトークン数、GPUメモリの最大使用量(MB)。

---

## 5. ヘルパー関数

### 5.1 `pick_device`

```python
def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available")
    return torch.device(choice)
```
- `auto` → GPUがあればGPU、なければCPU。`cuda` を**指定した**のにGPUが無ければ、(黙ってCPUを使う代わりに)分かりやすいメッセージで止める。それ以外は、指定したものを使う。

### 5.2 `precision`

```python
def precision(device: torch.device) -> tuple[str, torch.dtype | None]:
    """(label, autocast dtype). bf16 where supported, fp16 on older GPUs, fp32 on the CPU."""
    if device.type != "cuda":
        return "fp32", None
    if torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16
```
- 数の形式を選ぶ:**CPU → fp32**(autocastなし、`None`)、**bf16のあるGPU → bf16**、**古いGPU → fp16**(Day 1の「おすすめの精度」の考え方)。
- *注意:*PyTorchのバージョンによっては、古いGPUでも `is_bf16_supported()` が、bf16を**エミュレーション**して「はい」と答えることがある(遅い。Day 1の `native_bf16` と報告値の違いを参照)。報告書には**実際に使った精度**が出るので確認すること:RTX 2060なら、普通はfp16と出るはず。

### 5.3 `autocast`

```python
def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()
```
- すぐ使える `with` ブロックを返す:dtypeがあれば本物の**autocast**(Day 1)、完全な32ビットなら `nullcontext()`(何もしない)。これで、同じ `with autocast(...):` のコードが、両方で動く。

### 5.4 `real_batch`

```python
def real_batch(tokens: np.ndarray, batch: int, length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """batch windows of length+1 tokens from fixed random positions: x = first length, y = shifted by one."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - length - 1, size=batch)
    win = np.stack([np.asarray(tokens[s:s + length + 1], dtype=np.int64) for s in starts])
    w = torch.from_numpy(win)
    return w[:, :-1], w[:, 1:]
```
- Day 7のトークンの列(`train.bin` のメモリマップ)から、**実際の学習用の窓**を切り出す。
- `np.random.default_rng(seed)` — シード付きの乱数生成器(NumPy版)。`rng.integers(0, N, size=batch)` が、`batch` 個のランダムな**開始位置**を選ぶ。
- 各開始位置 `s` から `length + 1` 個のトークン `tokens[s : s+length+1]` を取り、64ビット整数に変える(`np.asarray(..., dtype=np.int64)` はメモリマップから取り出してコピーもする)。`np.stack` が `(batch, length+1)` の表に積む。
- `torch.from_numpy` でPyTorchのテンソルにする。
- **`x = w[:, :-1]`**(最初の `length` 個のトークン)と **`y = w[:, 1:]`**(同じ窓を1つずらしたもの)。各位置で、モデルは「ここまでのトークンから、次のトークンは?」と聞かれ、答えは `y` の値。(窓は、列の中で、ある質問から次の質問にまたがることがあるが、問題ない。)

### 5.5 `time_training_step`

```python
def time_training_step(model, x, y, device, dtype, repeats) -> tuple[float, float]:
    """Average milliseconds of one forward + backward pass, and the peak GPU memory in MB."""
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def step() -> None:
        with autocast(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
```
- `model.train()` — 学習モード(dropoutに関係する)。GPUの「最大メモリ」のカウンターをリセットするので、読み取る最大値は**このモデルだけ**のもの。
- `step()` — 順伝播を1回(autocastの下で)して損失を得る。`loss.backward()`(逆伝播:各パラメータの `.grad` を埋める)。そして `zero_grad(set_to_none=True)` で勾配を捨てる(メモリが解放され、次のステップがきれいに始まる)。**オプティマイザの更新は無し** — 速さのテストだけ。

```python
    step()  # warm-up (memory allocation, kernel selection)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) * 1000 / repeats
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return ms, peak
```
- まず1ステップ**ウォームアップ**する(最初のステップは遅い:メモリの確保、GPUのカーネルの選択)。Day 1の `bench_matmul` と同じ、ウォームアップ + `synchronize` の考え方(GPUは裏で動くので、時計を読む前に、本当に終わるのを待つ)。
- `repeats` ステップ(初期値10)を測り、**1ステップのミリ秒**にする。
- `peak` — 使われたGPUメモリの最大値(MB。`/ 2**20` がバイトをMBにする)。CPUでは `NaN`(「数ではない」)。

### 5.6 `memorise_one_batch`

```python
def memorise_one_batch(vocab, block, x, y, device, dtype, steps, seed) -> list[float]:
    torch.manual_seed(seed)
    model = ModelA(preset_config("tiny", vocab, block)).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    use = dtype if dtype == torch.bfloat16 else None   # fp16 would need a gradient scaler: use fp32 instead
    losses: list[float] = []
    for _ in range(steps):
        with autocast(device, use):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
```
- **本物の学習ループ**のミニ版を、**固定した1バッチ**で行う(だから、モデルはそれを覚え*られる*):
  - シードを固定 → 新しいtinyモデルをデバイスに載せ、学習モードにする;
  - `AdamW` オプティマイザ:学習率 `3e-3`、weight decayなし;
  - `use = dtype if dtype == torch.bfloat16 else None` — **bf16だけを使う**。fp16は勾配のスケーラー(Day 1)が必要になるので、fp16のGPUでは、この確認は単にfp32で動く(docstringにもある)。
  - 各ステップ:順伝播 → 損失;`zero_grad`(古い勾配を消す);`backward`(勾配を計算);`opt.step()`(**パラメータを更新**);`loss.item()`(損失を普通の数として)を覚える。
- 300個の損失のリスト(初期値)を返す。正しい構成なら、0に向かって下がるはず。

### 5.7 `write_report`、`mem_text`、`_milestones`

```python
def write_report(path, a, device_name, prec, vocab, train_tokens, reports, init_loss, attn_diff, losses) -> None:
    lines = [ ... ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```
- `docs/DAY8_MODEL_REPORT.md` を書く(他の日と同じ形)。セクション:

| セクション | 内容 |
|---|---|
| ヘッダー | バージョン、日付、マシンのラベル、デバイス、**時間測定の精度**、PyTorch/Pythonのバージョン、語彙、学習トークン数、ブロックサイズ、時間測定のバッチ |
| **Sizes** | プリセットごとに1行の表:層、ヘッド、幅、パラメータ(合計 / 埋め込み / 埋め込み以外)、**1パラメータあたりの学習トークン数**、1ステップのms、1秒あたりのトークン数、GPUメモリの最大使用量。注:1ステップ = 順伝播 + 逆伝播(オプティマイザなし)。埋め込みと出力層は1つの行列を共有するので、1回だけ数える。メモリの最大値はCPUでは `-`。「大きさはDay 11で、検証用の損失で選ぶ」 |
| **Checks on real token windows** | 初期の損失と `ln(vocab)`、手書きと速いAttentionの最大の差、1バッチを覚える(最初の損失 → 最後の損失)、5つの節目での損失の小さな表 |

- `*(... for r in reports)` は表の行を展開する(前と同じ技)。

```python
def mem_text(mb: float) -> str:
    return "-" if math.isnan(mb) else f"{mb:,.0f}"
```
- `NaN`(CPU)なら `-`、そうでなければ、3桁区切りのMB。

```python
def _milestones(n: int) -> list[int]:
    return sorted({max(1, n * k // 5) for k in range(1, 6)})
```
- 等間隔の5つのステップ番号:300ステップなら `{60, 120, 180, 240, 300}`(`n*k//5`、最低1。`n` がごく小さいときは、集合が重複を除く)。300個全部ではなく、損失の曲線の数点を表示するために使う。
- (`write_report` は、その下で定義された `_milestones` と `mem_text` を使う。関数は後で*呼ばれる*だけなので、動く。)

### 5.8 `append_run_log`

```python
def append_run_log(log_path, a, device_name, prec, reports, init_loss, attn_diff, losses) -> None:
    body = [ ... ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {a.machine} | Day 8 model check\n\n" + "\n".join(body) + "\n")
```
- Day 7と同じ形:箇条書きの行(マシン、デバイス、精度、スクリプト/PyTorch/Pythonのバージョン、大きさごとのパラメータ数・ms/step・tokens/s・メモリの最大値の1行、3つの確認の結果)を `run_log.md` に追記する。

---

## 6. `parse_args`

```python
ap.add_argument("--machine", choices=["home", "server", "school"], required=True)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--block-size", type=int, default=256)
ap.add_argument("--batch", type=int, default=32, help="batch size for the timing step")
ap.add_argument("--repeats", type=int, default=10, help="timed steps per size")
ap.add_argument("--overfit-steps", type=int, default=300)
ap.add_argument("--overfit-batch", type=int, default=4)
ap.add_argument("--overfit-len", type=int, default=64)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tokenizer", ...)  ap.add_argument("--tokens-dir", ...)  ap.add_argument("--configs-dir", ...)
ap.add_argument("--report", ...)  ap.add_argument("--run-log", ...)  ap.add_argument("--no-log", action="store_true")
```
- 前と同じ `argparse` の考え方。新しい数のオプション:
  - `--device` — `auto` / `cuda` / `cpu`;
  - `--block-size` 256 — 列の長さ(モデルの `block_size` でもある);
  - `--batch` 32 — 時間測定のバッチの大きさ(**GPUのメモリが足りなければ、下げる**。例:6GBのRTX 2060);
  - `--repeats` 10 — 大きさごとに測るステップ数;
  - `--overfit-steps` 300、`--overfit-batch` 4、`--overfit-len` 64 — 丸暗記のテスト(64トークンの窓4つ、300ステップ);
  - `--seed` 42。
- `Path` のオプション(トークナイザー、トークンファイルのフォルダー、設定のフォルダー、報告書、実行ログ)と `--no-log` は、前と同じ働き。

---

## 7. `main`

### 7.1 準備と入力の確認

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("tokenizer", "tokens_dir", "configs_dir", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    meta_path = a.tokens_dir / "train_meta.json"
    if not a.tokenizer.exists():
        raise SystemExit(f"Missing {a.tokenizer}. Run Day 6 first.")
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}. Run Day 7 first.")
```
- 前置きは前と同じ。トークナイザー(Day 6)と `train_meta.json`(Day 7)が必要。

```python
    device = pick_device(a.device)
    prec, dtype = precision(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Day 8 model check | machine={a.machine} | {device_name} | {prec} | PyTorch {torch.__version__} ===")
```
- デバイスと精度を選び、GPUの名前(または「CPU」)を取り、見出しを表示する。

```python
    vocab = ByteBPETokenizer.load(a.tokenizer).vocab_size
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["vocab_size"] != vocab:
        raise SystemExit("train_meta.json was made with a different tokenizer. Rerun Day 7.")
    train_tokens = int(meta["tokens"])
    tokens = np.memmap(a.tokens_dir / "train.bin", dtype=np.dtype("<u2"), mode="r")
    print(f"Vocabulary {vocab:,}; training tokens {train_tokens:,}; block size {a.block_size}")
```
- `vocab` — **本物のトークナイザー**の語彙サイズ。
- Day 7のメタファイルを読む。**整合性の確認:**その `vocab_size` がトークナイザーのものと等しくなければ、トークンファイルは別のトークナイザーで作られたものなので、止まる。(これは、Day 7.1の注意で、*読み込む側*がするべきだと書いた種類の確認。語彙サイズを比べるもので、SHA-256ではない。)
- `train_tokens` — 学習用のトークン数(メタから)。
- `np.memmap(..., dtype=np.dtype("<u2"), mode="r")` — `train.bin` を、リトルエンディアンの `uint16`(Day 7.1の形式)の**読み取り専用のメモリマップ**として直接開く — ファイル全体を読み込まずに。

### 7.2 確認1:3つの大きさ、パラメータと、時間を測る1ステップ

```python
    a.configs_dir.mkdir(parents=True, exist_ok=True)
    x, y = (t.to(device) for t in real_batch(tokens, a.batch, a.block_size, a.seed))
    reports: list[SizeReport] = []
    print(f"\n{'preset':<7} {'params':>11} {'tokens/param':>12} {'ms/step':>9} {'tokens/s':>11} {'peak MB':>8}")
```
- 設定のフォルダーを作る。**実際の1バッチ**(`batch × block_size` トークン)を取り、`x`、`y` をデバイスに移す(`.to(device)`。ジェネレーターを2つの名前に展開)。**同じバッチ**を3つの大きさすべてに使う → 公平な比較。
- 表の見出しを表示する(書式のコードは前と同じ)。

```python
    for name in PRESETS:
        cfg = preset_config(name, vocab, a.block_size)
        cfg.save(a.configs_dir / f"model_{name}.json")
        torch.manual_seed(a.seed)
        model = ModelA(cfg).to(device)
        counts = count_parameters(model)
        if counts["total"] != expected_parameters(cfg):
            raise SystemExit(f"{name}: parameter count does not match the formula")
        ms, peak = time_training_step(model, x, y, device, dtype, a.repeats)
```
- `tiny`、`small`、`medium` の順に:
  - **本物の語彙**で設定を作り、`configs/model_<name>.json` として**保存**する(Day 9・11用);
  - シードを固定し、モデルを作ってGPUに移す;
  - **パラメータ数を、手書きの式と照らし合わせる**(Day 8.1の§7)— 食い違いはバグなので → 止まる;
  - 学習1ステップの時間を測る。

```python
        r = SizeReport(name, cfg.n_layer, cfg.n_head, cfg.d_model, counts["total"], counts["embedding"],
                       counts["non_embedding"], train_tokens / counts["total"], ms,
                       a.batch * a.block_size / (ms / 1000), peak)
        reports.append(r)
        print(f"{name:<7} {r.total:>11,} {r.tokens_per_param:>12.2f} {r.ms_per_step:>9.1f} "
              f"{r.tokens_per_sec:>11,.0f} {mem_text(r.peak_mem_mb):>8}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
```
- 結果カード:
  - **`tokens_per_param = 学習トークン数 / パラメータの合計`** — 学習できる数1つあたりに、学習トークンがいくつあるか;
  - **`tokens_per_sec = batch × block_size / (ms ÷ 1000)`** — 1秒に処理されるトークン数(1ステップのトークン数 ÷ 1ステップの秒数)。
- 行を表示する。そして `del model` と `empty_cache()` で、次の大きさの前に**GPUメモリを解放**する(ある大きさのメモリが、次の大きさをゆがめないように)。

### 7.3 確認2:初期の損失と、2つのAttention

```python
    torch.manual_seed(a.seed)
    tiny = ModelA(preset_config("tiny", vocab, a.block_size)).to(device).eval()
    with torch.no_grad():
        fast_logits, init = tiny(x, y)
        tiny.set_attention("manual")
        manual_logits, _ = tiny(x, y)
    init_loss = init.item()
    attn_diff = (fast_logits - manual_logits).abs().max().item()
```
- 新しい**未学習**のtinyモデルを、評価モードで、普通の**fp32**で(ここではautocastなし)。
- `torch.no_grad()` — 勾配を追跡しない(速く、そのためのメモリも要らない)。
- 実際のバッチを**fast**のAttentionで通して `fast_logits` と損失 `init` を得る。**manual**のAttentionに切り替えて(`set_attention`、同じ重み)、もう一度通して `manual_logits` を得る。
- `init_loss` — 学習前の損失。
- `attn_diff` — 2つのロジットのテンソルの、**絶対値の差の最大**(`.abs().max()`)を、Pythonの数として(`.item()`)。

```python
    print(f"\nLoss before training: {init_loss:.3f}  (ln {vocab} = {math.log(vocab):.3f})")
    print(f"Manual vs fast attention: max difference {attn_diff:.2e}")
    if abs(init_loss - math.log(vocab)) > 0.5 or attn_diff > 1e-3:
        raise SystemExit("Check FAILED: initial loss or attention difference out of range")
    del tiny
```
- 両方を表示する。初期の損失が `ln(vocab)` から0.5より離れている(例:つなぎ間違いや、悪い初期化)か、2つのAttentionの差が0.001より大きければ**失敗**。(`:.2e` は `3.1e-07` のような指数表記。)
- なぜ `ln(vocab)` か? 一様に当て推量するモデルは、`V` 個のトークンのそれぞれに確率 `1/V` を与えるので、損失は `−log(1/V) = ln V`。`V = 1,674` なら約 **7.42**。

### 7.4 確認3:実際の1バッチを覚える

```python
    ox, oy = (t.to(device) for t in real_batch(tokens, a.overfit_batch, a.overfit_len, a.seed + 1))
    losses = memorise_one_batch(vocab, a.block_size, ox, oy, device, dtype, a.overfit_steps, a.seed)
    print(f"Memorise one batch ({a.overfit_batch} x {a.overfit_len}): loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    for s in _milestones(len(losses)):
        print(f"   step {s:>4}: {losses[s - 1]:.4f}")
    if losses[-1] > 0.5:
        raise SystemExit("Check FAILED: the model could not memorise one batch")
```
- **小さな実際のバッチ**(64トークンの窓4つ。`seed + 1` で別の窓)を取り、丸暗記のループ(セクション5.6)を実行し、最初 → 最後の損失と、5つの節目での損失を表示する。
- 最後の損失がまだ0.5より大きければ**失敗**:正しい損失・勾配・オプティマイザを持つ正しいモデルは、小さなバッチを**覚えられなければならない**。失敗は、どこかが間違ってつながっているということ。

### 7.5 報告書、ログ、終了

```python
    write_report(a.report, a, device_name, prec, vocab, train_tokens, reports, init_loss, attn_diff, losses)
    print(f"\nAll checks PASS. Report: {a.report}\nConfigs: {a.configs_dir}")
    if not a.no_log:
        append_run_log(a.run_log, a, device_name, prec, reports, init_loss, attn_diff, losses)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- 報告書を書き、PASSと場所を表示し(すべての確認に合格したときだけ到達)、`--no-log` でなければログを書き、0を返す。終わり方はいつもと同じ。

---

## 8. 実行の流れ(順番)

1. オプション → Day 6・7のファイルを確認 → デバイスと精度。
2. 語彙を読み込み、トークンファイルと合うか確認し、`train.bin` をメモリマップとして開く。
3. 実際の1バッチを取る。
4. tiny / small / medium について:設定を保存し、作り、パラメータ数を確認し、**学習1ステップの時間を測り**、表示する。
5. 初期の損失 ≈ `ln(vocab)` か? 手書きと速いAttentionは同じか?
6. 実際の1バッチを覚える → 損失は0近くになるか?
7. 報告書 → 実行ログ。

## 9. 結果の読み方

- **パラメータ:**3つの大きさで、約100万 / 500万 / 1,500万(あなたの語彙での正確な数は表にある)。
- **1パラメータあたりの学習トークン数:**これが**小さい**(よく言われる、1パラメータあたり約20トークンという大まかな目安を大きく下回る)と、大きいモデルは学習データを**過学習**(丸暗記)するかもしれない。また、あなたのコーパスはひな形から作られているので、本当の多様さは、大きさより低い。ここでは決めない — **Day 11が、検証用の損失で決める。**
- **ステップあたりのms / 1秒あたりのトークン数:**Day 9がどれくらいかかるかの目安。ステップの時間 × ステップ数 ≈ 学習の総時間(実際の学習では、オプティマイザとデータの読み込みが加わる)。
- **GPUメモリの最大使用量:**使うカードに収まる必要がある(RTX 2060 = 6GB、RTX 4060 = 8GB)。`medium` が大きすぎるなら、`--batch` を下げる。
- **初期の損失 ≈ ln(vocab):**モデルは一様な当て推量から始まる(良い)。
- **Attentionの最大の差が約1e-6以下:**速いAttentionと手書きのAttentionが一致している。
- **1バッチを覚える:**損失が、約 `ln(vocab)` から0近くまで下がるはず。

## 10. 知っておくと良い小さなこと

- **時間測定には、オプティマイザの更新もデータの読み込みも無い**ので、実際の学習のステップは、もっと遅い。
- **古いGPUでのbf16:**RTX 2060の報告書に `bf16` と出たら、PyTorchがエミュレーションしているかもしれない(セクション5.2)。本物のbf16には、compute capability 8以上が必要(Day 1の `native_bf16`)。
- **すべての大きさで同じバッチ**だと、比較が公平になるが、時間測定には1バッチは小さな標本。実行ごとの数%のばらつきは普通。
- **ここで保存するのは設定だけで、重みではない。**このスクリプトでは、モデルを学習しない(丸暗記のモデルは捨てられる)。
- **設定は語彙に依存する。**トークナイザーを学習し直したら(Day 6)、トークンファイルと設定が合うように、Day 7とDay 8もやり直す。
- **確認は順番に行われ、最初の失敗で止まる。**報告書は、すべて合格したときだけ保存される(3つの大きさの設定は、実際にはもっと前、手順4で保存される)。

## 11. 一段落でまとめ

このスクリプトは、本物の語彙でModel Aを3つの大きさで作り、パラメータを数え(式で照らし合わせ)、実際のトークンの窓で順伝播+逆伝播の1ステップの時間を測って速さとGPUメモリを求め、未学習のモデルが `ln(語彙)` の損失から始まること、速いAttentionと手書きのAttentionが一致すること、モデルが実際の1バッチを覚えられること(つまりモデル、損失、勾配、オプティマイザが正しくつながっていること)を証明し、そして3つの設定をDay 9・11用に保存し、報告書と実行ログを書きます。
