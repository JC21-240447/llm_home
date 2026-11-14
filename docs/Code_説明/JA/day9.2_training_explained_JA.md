# Day 9.2 — training.py — やさしい1行ずつの解説(日本語)

> **Day 9 の5つのファイルの関係**
> - **Day 9.1 `batching.py`** — バッチを作る(先に読んでください)。
> - **Day 9.2(このファイル)`training.py`** — **学習ループ**:Day 9の心臓部(Day 11・12でも使う)。
> - **Day 9 `day9_train.py`** — `train(...)` を呼ぶコマンドラインのスクリプト。
> - **Day 9 `day9_compare_runs.py`** — 停止と再開が正しく動く証明。
> - **Day 9.3 `test_training.py`** — 自動テスト。
>
> ここまでで一番大きなファイルです。セクション0と1をゆっくり読んでください。残りはそれに続きます。

---

## 0. 全体像(まずここから)

学習 = 同じ小さなサイクルを何千回も繰り返すこと:

```
窓のバッチを取る  →  モデルが次のトークンを予測  →  損失を測る
→  勾配を計算(各パラメータをどちらへ変えるか)  →  パラメータを少し変える
```

サイクル1回を **ステップ(step)** と呼びます。このファイルは、そのサイクルを、*本格的な*学習に必要なものすべてで包みます:

| 必要なもの | `training.py` の解決方法 |
|---|---|
| 穏やかに始まり、穏やかに終わる学習率 | **ウォームアップ + コサイン減衰**(`lr_at`) |
| 小さなGPUで、実質的に大きなバッチ | **勾配の累積**(1ステップに複数の小バッチ) |
| 学習を安定させる | **勾配クリッピング**、**AdamW**、重み減衰は行列だけ |
| 速く、メモリを節約 | **混合精度**(bf16、またはfp16 + スケーラー) |
| 学習の様子を知る | `log_every` ステップごとに **log.csv**;`eval_every` ステップごとに**検証の損失**と**サンプルの文章** |
| 作業を失わない | **チェックポイント** `last.pt` と `best.pt`、安全に保存 |
| 止めた後で続ける | **再開**:*まったく同じ*バッチと学習率で |
| 丁寧に止める | **Ctrl+C** はステップを終えて保存する |
| 大惨事を見つける | 損失がNaN/無限大になったら、止まって `diverged.pt` を保存 |

### 1回の最適化ステップ(docstringより)

1. このステップの学習率(ウォームアップ、その後 `min_lr` までコサイン減衰);
2. `grad_accum` 個の小バッチ:順伝播(autocast)、損失、逆伝播;
3. 勾配の大きさを `grad_clip` までに抑える;
4. AdamWで更新(重み減衰は行列だけ。正規化の倍率にはかけない)。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **学習率(lr)** | 各ステップで、パラメータをどれだけ大きく変えるか。大きすぎると不安定、小さすぎると遅い。 |
| **ウォームアップ** | ごく小さな学習率から始めて直線的に上げ、最初の(ランダムで混沌とした)ステップがモデルを壊さないようにする。 |
| **コサイン減衰** | ウォームアップの後、学習率を、コサインの曲線に沿ってなめらかに `min_lr` まで下げる:最初は大きな歩幅、最後は細かい歩幅。 |
| **勾配の累積** | 小さなバッチをいくつか実行し、その勾配を**足し合わせ**、更新は1回。実質のバッチ = `batch_size × grad_accum`。巨大な1バッチのメモリは要らない。 |
| **勾配クリッピング** | 勾配が巨大(「スパイク」)なとき、全体の大きさが最大 `grad_clip` になるように縮める。爆発を防ぐ。 |
| **AdamW** | オプティマイザ:Adam(パラメータごとに歩幅を適応させる)+ 切り離した重み減衰。`beta1`、`beta2` が記憶の長さを決める(0.9 / 0.95)。 |
| **重み減衰(weight decay)** | 過学習を抑えるため、重みを0へゆるく引っ張る。**行列だけ**にかけ、正規化の倍率(ベクトル)にはかけない。 |
| **GradScaler** | Day 1のfp16の助っ人(損失に大きな数をかけて、小さな勾配が0にならないようにする)。 |
| **パープレキシティ(perplexity)** | `exp(損失)`。おおよそ「モデルが、いくつの同じくらい可能性のあるトークンから選んでいるか」。V個のトークンへの一様な当て推量のパープレキシティはV。低いほど良い。 |
| **チェックポイント** | 続きに必要なものすべて(モデル、オプティマイザなど)を保存した、スナップショット。 |
| **再開(resume)** | 止まった学習を、そのチェックポイントから続けること。 |
| **RNGの状態** | 乱数生成器の内部の状態。戻すと、その後の「ランダム」な数が、そっくり繰り返される。 |
| **貪欲デコード(greedy)** | 一番点数の高い次のトークンを、いつも1つ選ぶ。(Day 10で、本格的なサンプリングを足す。) |
| **シグナル / SIGINT** | **Ctrl+C** を押したとき、OSが送るメッセージ。 |
| **コンテキストマネージャー(`with`)** | `with` ブロックに入るときに準備をし、出るときに後片付けをするオブジェクト。 |
| **クロージャ / `nonlocal`** | 別の関数の中で定義された関数で、外側の関数の変数を読める(`nonlocal` があれば変更もできる)。 |
| **デコレーター `@torch.no_grad()`** | 関数を包んで、勾配を記録せずに動かす(速く、メモリも少ない)。 |

---

## 2. docstring、import、定数

- docstring(英語 + 日本語)は、最適化ステップ、記録と評価、**再開**(`last.pt` に入るもの)、**安全な停止**(Ctrl+C。チェックポイントは一時ファイルに書いてから名前を変える)を説明している。

```python
from __future__ import annotations

import csv, hashlib, json, math, os, platform, signal, time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from src.batching import RandomWindows, open_tokens, validation_batches, validation_windows
from src.model import ModelA, ModelConfig
from src.token_data import read_meta
```
- ほとんどは前の日で見たもの。新しいもの:`os`(ファイル名の変更)、`signal`(Ctrl+Cの処理)、`fields`(データクラスの項目の一覧)、`platform`(Pythonのバージョン)。
- プロジェクトの他のファイルから:バッチの補助(**Day 9.1**)、モデル(**Day 8.1**)、`read_meta`(**Day 7.1**)。

```python
CHECKPOINT_FORMAT: str = "model-a-checkpoint-v1"
LOG_COLUMNS: tuple[str, ...] = ("step", "event", "tokens_seen", "lr", "train_loss", "train_ppl", "val_loss",
                                "val_ppl", "grad_norm", "tokens_per_sec", "peak_vram_mb", "elapsed_s")
DEFAULT_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "大阪支社で", "社員の平均")
RESUME_FREE_KEYS: frozenset[str] = frozenset({"stop_after", "resume", "device", "machine", "overwrite"})
```
- `CHECKPOINT_FORMAT` — すべてのチェックポイントに入れる名札(Day 6.1・7.1と同じ考え方)。
- `LOG_COLUMNS` — `log.csv` の12個の列。
- `DEFAULT_PROMPTS` — サンプルの出力に使う、決まった文の書き出し(「営業部の」、「ボーナスが」、…)。
- `RESUME_FREE_KEYS` — 再開するときに**違ってよい**設定。バッチにも学習の曲線にも影響しないから(いつ止めるか、再開するか、どのデバイスか、ラベル)。

---

## 3. `TrainConfig` — すべての設定

```python
@dataclass
class TrainConfig:
    run_dir: str
    model_config: str = "configs/model_tiny.json"
    tokens_dir: str = "data/tokens"
    tokenizer: str | None = "data/tokenizer/tokenizer.json"
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 600
    warmup_steps: int = 60
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float | None = None
    log_every: int = 10
    eval_every: int = 100
    eval_max_windows: int | None = None
    sample_tokens: int = 40
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    precision: str = "auto"
    device: str = "auto"
    seed: int = 42
    stop_after: int | None = None
    resume: bool = False
    overwrite: bool = False
    machine: str = "home"
```

| 設定 | 意味 |
|---|---|
| `run_dir` | この学習のフォルダー(必須。例:`runs/day9/smoke_a`) |
| `model_config` | どの `configs/model_*.json`(Day 8)を使うか |
| `tokens_dir`、`tokenizer` | Day 7のトークンファイルとDay 6のトークナイザーの場所(`tokenizer=None` ならトークナイザーの確認とサンプルの文章を飛ばす) |
| `batch_size`、`grad_accum` | 小バッチごとの窓の数と、1ステップの小バッチの数 |
| `max_steps` | 最適化ステップの総数 |
| `warmup_steps` | 直線的なウォームアップのステップ数 |
| `lr`、`min_lr_ratio` | 最大の学習率と、最後の学習率(それに対する割合。0.1 → 10%で終わる) |
| `weight_decay` | 重み減衰の強さ(行列だけ) |
| `beta1`、`beta2` | AdamWの記憶の定数(0.9、0.95) |
| `grad_clip` | 勾配の大きさの上限 |
| `dropout` | `None` = モデルの設定のdropoutを使う。数字なら上書き |
| `log_every`、`eval_every` | 記録・評価の頻度(ステップ) |
| `eval_max_windows` | 速くするため検証用の窓を制限(`None` = 全部) |
| `sample_tokens` | 各サンプルの続きの長さ(40トークン) |
| `prompts` | 決まった文の書き出し(`default_factory=lambda: ...` はオブジェクトごとに専用のリストを与える) |
| `precision` | `auto` / `bf16` / `fp16` / `fp32` |
| `device` | `auto` / `cuda` / `cpu` |
| `seed` | 乱数のシード(モデルの初期化とバッチの生成器) |
| `stop_after` | このステップでわざと止める(中断のシミュレーション) |
| `resume`、`overwrite` | 学習を再開する / 既存のフォルダーでやり直す |
| `machine` | ログ用のラベル |

```python
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
```
- `to_dict` — 設定を普通の辞書として(`train_config.json` として保存される)。
- `from_dict` — 辞書から作り直す。**未知のキーは無視する**(`fields(cls)` が本物の項目名を挙げる)。だから、後で設定が増えたり減ったりしても、古い `train_config.json` が読める。

---

## 4. 小さな部品

### 4.1 `lr_at` — 学習率のスケジュール

```python
def lr_at(step: int, max_lr: float, warmup: int, max_steps: int, min_ratio: float) -> float:
    """Learning rate for the step about to be taken (step counts from 0)."""
    min_lr = max_lr * min_ratio
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```
- `min_lr` = 最後の学習率。
- **ウォームアップ**(`step < warmup`):`max_lr × (step+1) / warmup` — `max_lr/warmup` から `max_lr` まで、直線的に上がる。
- 終わりの後(`step >= max_steps`):`min_lr`。
- **コサイン減衰**:`progress` は、残りのステップで0から1へ進む(`max(1, …)` はゼロ割りを避ける)。式 `min_lr + 0.5 × (max_lr − min_lr) × (1 + cos(π × progress))` は、`max_lr` から始まり(progress 0:cos 0 = 1)、`min_lr` で終わり(progress 1:cos π = −1)、間はなめらかなS字で下がる。
- 例(最大1e-3、ウォームアップ10、100ステップ):ステップ0 → 1e-4;ステップ9 → 1e-3;ステップ55(中間)→ 5.5e-4;ステップ100 → 1e-4。(Day 9.3が、まさにこれをテストする。)

### 4.2 `param_groups` — 重み減衰は行列だけ

```python
def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
```
- パラメータを、オプティマイザ用に2つのグループに分ける:
  - **行列**(`p.dim() >= 2`:埋め込み、Attention、MLPの重み)→ `weight_decay` がかかる;
  - **ベクトル**(`p.dim() < 2`:RMSNormの倍率)→ 減衰**なし**(倍率を0へ縮めても害になるだけ)。
- `named_parameters()` は、共有された(重み共有した)テンソルを1回だけ挙げる。`requires_grad` = 「学習できる」。

### 4.3 `pick_device`、`pick_precision`、`autocast_ctx`、`sha256_file`

```python
def pick_device(choice: str) -> torch.device: ...
def pick_precision(choice: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if choice == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if choice == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("this GPU does not support bf16; use --precision fp16")
    return choice

def autocast_ctx(device: torch.device, precision: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()
```
- `pick_device` — Day 8と同じ。
- `pick_precision` — CPU → fp32;`auto` → 対応していればbf16、なければfp16;bf16の無いGPUで bf16を**指定**したら → ヒント付きで止まる。(Day 8と同じ注意:PyTorchのバージョンによっては、古いGPUでも `is_bf16_supported()` がエミュレーションで「はい」と答えることがある。表示される精度が、選ばれたものを教えてくれる。)
- `autocast_ctx` — Day 8の `autocast` と同じ考え方:辞書が精度の名前をdtypeに対応させる(`.get` はfp32で `None` を返す → `nullcontext()`)。
- `sha256_file` — Day 4と同じ。

### 4.4 `evaluate`

```python
@torch.no_grad()
def evaluate(model, tokens, block_size, batch_size, device, precision, max_windows) -> float:
    """Average next-token loss over the fixed validation windows."""
    model.eval()
    total, count = 0.0, 0
    for x, y in validation_batches(tokens, block_size, batch_size, max_windows):
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device, precision):
            _, loss = model(x, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    model.train()
    return total / max(1, count)
```
- `@torch.no_grad()` — 中では勾配を使わない(デコレーター)。
- 評価モード;固定した検証用のバッチごとに:デバイスに移し、順伝播し、バッチの**平均の損失**を得て、バッチの窓の数(`x.size(0)`)をかけて `total` に足す。窓の総数で割れば、最後のバッチが小さくても**正確な平均**になる。
- 学習モードに戻し、平均の損失を返す(`max(1, count)` はゼロ割りを避ける)。

### 4.5 `greedy_continue`

```python
@torch.no_grad()
def greedy_continue(model, ids, max_new, eos_id, device) -> list[int]:
    """Always take the highest-scoring next token (Day 10 adds proper sampling)."""
    model.eval()
    out = list(ids)
    for _ in range(max_new):
        x = torch.tensor([out[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt == eos_id:
            break
    model.train()
    return out[len(ids):]
```
- 書き出しの続きを、トークン1つずつ書く:
  - 最後の `block_size` 個のトークン(`out[-block_size:]`)を入力にする(1つだけのバッチ);
  - `logits[0, -1]` = **最後の位置**の点数。`.argmax()` = **一番点数の高いトークン**。それを足す;
  - `<eos>` のトークンか、`max_new` トークンで止まる。
- **新しい**トークンだけを返す(`out[len(ids):]`)。「貪欲」は決定的なので、別々の学習のサンプルの文章を比べられる。

### 4.6 チェックポイント:`save_checkpoint`、`load_checkpoint`

```python
def save_checkpoint(path: Path, state: dict) -> None:
    """Write to a temporary file, then rename: an interruption never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_checkpoint(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a Model A checkpoint")
    return state
```
- `save_checkpoint`:まず `last.pt.tmp` に書き、`os.replace(tmp, path)` で、1回の**不可分(atomic)**な操作で入れ替える。書き込み中に電源が落ちても、古い `last.pt` は無事。
- `load_checkpoint`:`torch.load(..., weights_only=True)` は、**数・テンソル・単純な入れ物だけ**を読み込む(ファイルに隠された任意のコードの実行を拒む — 安全機能)。`map_location=device` はテンソルを正しいデバイスに置く。形式の名札を確認する。

### 4.7 `_StopFlag` — 丁寧なCtrl+C

```python
class _StopFlag:
    """Ctrl+C once = stop after the current step and save; twice = stop immediately."""

    def __init__(self) -> None:
        self.requested = False
        self._old = None

    def __enter__(self) -> "_StopFlag":
        try:
            self._old = signal.signal(signal.SIGINT, self._handler)
        except ValueError:          # not in the main thread (for example inside some test runners)
            self._old = None
        return self

    def _handler(self, signum, frame) -> None:  # noqa: ARG002
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\nCtrl+C: finishing this step, saving last.pt, then stopping (press again to stop at once).")

    def __exit__(self, *exc) -> None:
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
```
- `with _StopFlag() as stop:` の形で使う、**コンテキストマネージャー**。
- `__enter__`(`with` ブロックの始まりで実行):`signal.signal(SIGINT, handler)` が**自分用のCtrl+Cの処理**を設定し、古いものを覚える。(メインスレッドでなければ、Pythonが `ValueError` を出す。その場合は何も設定しない。)
- `_handler`(Ctrl+Cが押されたとき実行):**1回目**は `requested = True` にしてメッセージを表示するだけ — 学習ループが、今のステップの後でこの旗を確認する。**2回目**は `KeyboardInterrupt` を出して、すぐ止める。
- `__exit__`(ブロックの終わりで実行):前の処理を**元に戻す**。

### 4.8 `_truncate_after` — 強制終了の後始末

```python
def _truncate_after(log_path: Path, samples_path: Path, step: int) -> None:
    if log_path.exists():
        with log_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        keep = [r for r in rows if int(r["step"]) <= step]
        if len(keep) != len(rows):
            with log_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                w.writeheader()
                w.writerows(keep)
            print(f"Removed {len(rows) - len(keep)} log rows after step {step} (they will be trained again)")
    if samples_path.exists():
        text = samples_path.read_text(encoding="utf-8")
        parts = text.split("\n## step ")
        kept = [parts[0]] + [p for p in parts[1:] if int(p.split(" ", 1)[0]) <= step]
        if len(kept) != len(parts):
            samples_path.write_text("\n## step ".join(kept), encoding="utf-8", newline="\n")
```
- **問題:**プログラムがCtrl+Cで止められたのではなく*強制終了*された(ウィンドウを閉じた、停電)場合、`log.csv` と `samples.md` に、最後に保存したチェックポイントの**後**のステップが入っていることがある。それらのステップは再開時にもう一度学習されるので、古い行を消さないと、ログに重複ができる。
- `log.csv`:全行を読み、`step <= チェックポイントのステップ` の行を残す。何か除かれたら、ファイルを(ヘッダー + 残した行で)書き直し、除いた数を表示する。
- `samples.md`:テキストを `"\n## step "` でセクションに分ける。`parts[0]` はヘッダー。他の各部分はステップ番号から始まる(`p.split(" ", 1)[0]`)。チェックポイントのステップまでのセクションを残し、必要ならファイルを書き直す。

### 4.9 `_fmt`

```python
def _fmt(v: float | None, digits: int = 4) -> str:
    return "" if v is None else f"{v:.{digits}f}"
```
- 数を指定の小数の桁数で整形する。`None` なら空文字(入れ子の `{v:.{digits}f}` は、小数の桁数を変数 `digits` から取る)。

---

## 5. `train` — 学習の全体

```python
def train(cfg: TrainConfig, repo_root: Path) -> dict:
    """Train (or resume) one run. Returns a short summary."""
```
設定とリポジトリのフォルダーを受け取り、まとめの辞書を返す。

### 5.1 パス、デバイス、精度

```python
    root = repo_root
    run_dir = (root / cfg.run_dir).resolve()
    tokens_dir = (root / cfg.tokens_dir).resolve()
    device = pick_device(cfg.device)
    precision = pick_precision(cfg.precision, device)
```
- 学習のフォルダーとトークンファイルの絶対パス。デバイスと精度を選ぶ。

### 5.2 設定とデータを、互いに照らし合わせる

```python
    mcfg = ModelConfig.load((root / cfg.model_config).resolve())
    if cfg.dropout is not None:
        mcfg.dropout = cfg.dropout
    train_meta = read_meta(tokens_dir, "train")
    if train_meta.vocab_size != mcfg.vocab_size:
        raise SystemExit("model config and token files use different vocabulary sizes")
```
- モデルの設定(Day 8)を読み、必要ならdropoutを上書きし、学習用のメタのラベル(Day 7.1)を読む。**語彙サイズが一致しなければならない**(Day 8と同じ確認)。

```python
    tok = None
    if cfg.tokenizer:
        from src.bpe_tokenizer import ByteBPETokenizer
        tok_path = (root / cfg.tokenizer).resolve()
        if sha256_file(tok_path) != train_meta.tokenizer_sha256:
            raise SystemExit("data/tokens was made with a different tokenizer; rerun Day 7")
        tok = ByteBPETokenizer.load(tok_path)
    train_tokens = open_tokens(tokens_dir, "train")
    val_tokens = open_tokens(tokens_dir, "val")
```
- トークナイザーが指定されていれば:その**SHA-256**を計算し、トークンファイルのメタに保存された指紋(`tokenizer_sha256`)と比べる。一致しなければ、トークンは別のトークナイザーで作られたもの → 止まる。(これが、Day 7.1の注意で、学習プログラムがするべきだと書いた、*より強い*確認。Day 8は語彙サイズを比べただけだった。)トークナイザーを読み込む(サンプルの文章に必要)。importが `if` の中にあるので、トークナイザーのモジュールは、使うときだけ必要になる。
- 学習用と検証用のトークンの列を開く(Day 9.1)。

### 5.3 学習のフォルダー:新規か再開か

```python
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    log_path, samples_path = run_dir / "log.csv", run_dir / "samples.md"
    if cfg.resume:
        if not last_path.exists():
            raise SystemExit(f"nothing to resume: {last_path} does not exist")
        saved = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        diff = [k for k, v in cfg.to_dict().items() if k not in RESUME_FREE_KEYS and saved.get(k) != v]
        if diff:
            raise SystemExit(f"resume refused: settings differ from the first run: {diff}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()) and not cfg.overwrite:
            raise SystemExit(f"{run_dir} already has files. Use a new --run-name, --resume, or --overwrite.")
        run_dir.mkdir(parents=True, exist_ok=True)
        for p in (last_path, best_path, log_path, samples_path):
            p.unlink(missing_ok=True)
        (run_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
                                                   encoding="utf-8", newline="\n")
        mcfg.save(run_dir / "model_config.json")
```
- **再開:**`last.pt` が必要。今の設定を、保存された `train_config.json` と比べる:**「自由な」キー以外**に違いがあれば → **拒否**(違う学習率で再開すると、静かに別の学習になるため)。
- **新規:**フォルダーに既にファイルがあれば(`--overwrite` でない限り)拒否 — 古い学習を、うっかり上書きしないように。フォルダーを作り、残っていた古いファイルを消し、**設定を保存**する(`train_config.json`)とモデルの設定(`model_config.json`)— だから学習は、いつでも再現できる。

### 5.4 モデル、オプティマイザ、スケーラー、バッチの生成器を作る

```python
    torch.manual_seed(cfg.seed)
    model = ModelA(mcfg).to(device)
    use_fused = device.type == "cuda"
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            fused=use_fused)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))
    windows = RandomWindows(train_tokens, mcfg.block_size, cfg.batch_size, cfg.seed)
```
- PyTorchの乱数生成器にシードを与え、デバイス上にモデルを作る。
- **AdamW**:2つのパラメータグループ、学習率、betaを指定。GPUでは `fused=True` = 速い実装。
- **GradScaler** は **fp16のときだけ有効**(bf16/fp32では何もしない)。*注意:*これは新しい形 `torch.amp.GradScaler(device_type, …)` を使っている。Day 1のスモークテストには古いPyTorch用の予備があったが、これには無い。
- バッチの生成器(Day 9.1)を、同じシードで。

### 5.5 初期のカウンターと、チェックポイントからの復元

```python
    step, best_val, elapsed_before, tokens_seen = 0, float("inf"), 0.0, 0
    if cfg.resume:
        st = load_checkpoint(last_path, device)
        if st["model_config"] != mcfg.to_dict():
            raise SystemExit("resume refused: the model settings differ from the checkpoint")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        windows.load_state_dict(st["windows"])
        torch.set_rng_state(st["torch_rng"].cpu())
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        step, best_val = st["step"], st["best_val"]
        elapsed_before, tokens_seen = st["elapsed_s"], st["tokens_seen"]
        _truncate_after(log_path, samples_path, step)
        print(f"Resumed from {last_path.name} at step {step} (best validation loss so far {best_val:.4f})")
```
- カウンター:`step` = 終えたステップ数、`best_val` = ここまでで一番低い検証の損失(無限大から始まる)、`elapsed_before` = すでに使った時間、`tokens_seen`。
- **再開**では、**すべて**を復元する:
  - モデルの設定がチェックポイントと合うか確認;
  - モデルの重み、**オプティマイザの状態**(Adamの移動平均!)、スケーラーの状態、**バッチの生成器の状態**(Day 9.1)、PyTorchのCPUとCUDAの乱数の状態;
  - `step`、`best_val`、経過時間、`tokens_seen`;
  - そのステップより後のログとサンプルを片付ける(セクション4.8)。
- だから、再開した学習は、**まったく同じバッチと学習率**で続き、損失の曲線は飛ばない。

### 5.6 設定を知らせる

```python
    tokens_per_step = cfg.batch_size * cfg.grad_accum * mcfg.block_size
    n_val = validation_windows(val_tokens, mcfg.block_size)
    if cfg.eval_max_windows is not None:
        n_val = min(n_val, cfg.eval_max_windows)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(...)  # デバイス | 精度 | モデル | パラメータ数;  batch x accum x block = tokens/step | steps | 検証用の窓
```
- `tokens_per_step` = 最適化ステップ1回で処理するトークン数。検証用の窓を数える。2行のまとめを表示する。

### 5.7 クロージャ:`state_for_save`、`write_row`、`checkpoint_and_sample`

```python
    def state_for_save() -> dict:
        return {
            "format": CHECKPOINT_FORMAT, "step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(), "windows": windows.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "model_config": mcfg.to_dict(), "train_config": cfg.to_dict(), "best_val": best_val,
            "elapsed_s": elapsed_before + (time.perf_counter() - t_start), "tokens_seen": tokens_seen,
            "tokenizer_sha256": train_meta.tokenizer_sha256, "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
        }
```
- **チェックポイントに入るもの**:形式の名札、今のステップ、モデルの重み、オプティマイザ、スケーラー、バッチの生成器の状態、PyTorchの乱数の状態(CPUとGPU)、2つの設定、`best_val`、経過時間、見たトークン数、トークナイザーの指紋、PyTorch/Pythonのバージョン(記録のため)。`train` の**中**で定義されているので、呼ばれた瞬間の `step`、`best_val` などの*現在の*値を読む(クロージャ)。

```python
    new_log = not log_path.exists()
    log_f = log_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(log_f, fieldnames=LOG_COLUMNS)
    if new_log:
        writer.writeheader()
```
- `log.csv` を**追記**モードで開く(再開した学習が追記するように)。ヘッダーは、新しいファイルのときだけ書く。

```python
    def write_row(event, lr, train_loss, val_loss, grad_norm, tps) -> None:
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        writer.writerow({
            "step": step, "event": event, "tokens_seen": tokens_seen, "lr": f"{lr:.6g}",
            "train_loss": _fmt(train_loss), "train_ppl": _fmt(math.exp(train_loss) if train_loss is not None else None, 2),
            "val_loss": _fmt(val_loss), "val_ppl": _fmt(math.exp(val_loss) if val_loss is not None else None, 2),
            "grad_norm": _fmt(grad_norm), "tokens_per_sec": _fmt(tps, 0), "peak_vram_mb": _fmt(peak, 0),
            "elapsed_s": f"{elapsed_before + time.perf_counter() - t_start:.1f}",
        })
        log_f.flush()
```
- `log.csv` に**1行**書く。`event` は `"train"` か `"eval"`。パープレキシティ = `exp(損失)`。無い値は空文字になる。`log_f.flush()` は行をすぐディスクに書き出させるので、クラッシュしても何も失われない。

```python
    def checkpoint_and_sample(lr: float) -> float:
        nonlocal best_val
        val = evaluate(model, val_tokens, mcfg.block_size, cfg.batch_size, device, precision, cfg.eval_max_windows)
        improved = val < best_val
        best_val = min(best_val, val)
        write_row("eval", lr, None, val, None, None)
        if tok is not None and cfg.prompts:
            lines = [f"\n## step {step} (validation loss {val:.4f})\n", "| Prompt | Greedy continuation |", "| --- | --- |"]
            for p in cfg.prompts:
                cont = greedy_continue(model, [1] + tok.encode(p), cfg.sample_tokens, 2, device)
                text = tok.decode(cont).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {p} | {text}{' ⟨eos⟩' if cont and cont[-1] == 2 else ''} |")
            with samples_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
        state = state_for_save()
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        print(f"step {step:>6} | validation loss {val:.4f} (perplexity {math.exp(val):.2f})"
              f"{'  * best, saved best.pt' if improved else ''} | saved last.pt")
        return val
```
- **「評価の瞬間」:**
  1. **検証の損失**を計算する(セクション4.4)。`improved` = 今までで最も低いか。`best_val` を更新する(`nonlocal` により、内側の関数が外側の変数を変えられる);
  2. `eval` の行を書く;
  3. トークナイザーがあれば:決まった書き出しごとに、入力 `[1] + tok.encode(p)`(**`1` = `<bos>`**)を作り、貪欲な続きを生成し(`2` = `<eos>` で止まる)、文字に復号し、Markdownの表が壊れないよう **`|` と改行をエスケープ**し、`## step N (validation loss …)` というセクションを `samples.md` に追記する。文章が時間とともに良くなるのを見られる。
  4. `last.pt`(いつも)と `best.pt`(検証が良くなったときだけ)を保存;
  5. 1行表示する。

### 5.8 メインのループ

```python
    t_start = time.perf_counter()
    model.train()
    interval_loss = torch.zeros((), device=device)
    interval_steps, interval_t0 = 0, time.perf_counter()
    stopped_early = False
    last_lr = cfg.lr
    try:
        with _StopFlag() as stop:
            if step == 0 and not cfg.resume:
                checkpoint_and_sample(lr_at(0, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio))
                interval_t0 = time.perf_counter()
            while step < cfg.max_steps:
```
- ストップウォッチを始め、学習モードにする。`interval_loss` — **前回の記録以降のステップの損失を足し込む** GPU上のテンソル(CPUへのコピーは記録のときだけにする。毎ステップコピーするとGPUが遅くなる)。
- Ctrl+Cの処理を設定する(`with _StopFlag() as stop`)。
- **学習の前**(新しい学習のステップ0):評価を1回行う → **基準になる**検証の損失(≈ `ln(vocab)`)と、最初のサンプルの文章、そして最初のチェックポイント。
- `max_steps` ステップが終わるまでループする。

```python
                lr = lr_at(step, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
                last_lr = lr
                for g in opt.param_groups:
                    g["lr"] = lr
```
- このステップの学習率を求め、オプティマイザのすべてのグループに設定する。

```python
                step_loss = torch.zeros((), device=device)
                for _ in range(cfg.grad_accum):
                    x, y = windows.next_batch()
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with autocast_ctx(device, precision):
                        _, loss = model(x, y)
                    step_loss += loss.detach() / cfg.grad_accum
                    scaler.scale(loss / cfg.grad_accum).backward()
```
- **勾配の累積ループ**(`grad_accum` 個の小バッチ。初期値は1):
  - バッチを取り(Day 9.1)、デバイスに移す(`non_blocking=True` で、コピーと計算を重ねられる);
  - autocastの下で順伝播 → 損失;
  - `step_loss += loss.detach() / grad_accum` — 記録のために、このステップの**平均の損失**を保つ(`detach()` = 勾配を追跡しない);
  - `scaler.scale(loss / grad_accum).backward()` — 逆伝播。`grad_accum` で割ることで、**勾配の合計が小バッチの平均と等しくなる**。(`scaler.scale` はfp16のスケール係数をかける。bf16/fp32では何もしない。)勾配は、小バッチをまたいで、パラメータの中で**足し合わされる**。

```python
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                tokens_seen += tokens_per_step
                interval_loss += step_loss
                interval_steps += 1
```
- `scaler.unscale_(opt)` — fp16のスケーリングを元に戻し、勾配が本当の大きさになる。
- `clip_grad_norm_(…, grad_clip)` — 勾配の全体の大きさが `grad_clip` を超えたら、**縮める**。クリッピング**前**の大きさ(`grad_norm`。記録される)を返す。
- `scaler.step(opt)` — **AdamWの更新**(無限大が見つかれば、スケーラーは飛ばす)。`scaler.update()` はスケールを調整し、`zero_grad` は勾配を消す。
- ステップを数える:`step += 1`、`tokens_seen`、区間の合計に足す。

```python
                if step % cfg.log_every == 0 or step == cfg.max_steps:
                    mean_loss = (interval_loss / interval_steps).item()
                    if not math.isfinite(mean_loss):
                        save_checkpoint(run_dir / "diverged.pt", state_for_save())
                        raise SystemExit(f"loss became {mean_loss} at step {step}; saved diverged.pt and stopped")
                    dt = time.perf_counter() - interval_t0
                    tps = interval_steps * tokens_per_step / dt if dt > 0 else None
                    write_row("train", lr, mean_loss, None, float(grad_norm), tps)
                    print(f"step {step:>6} | loss {mean_loss:.4f} | lr {lr:.2e} | {(tps or 0):,.0f} tokens/s")
                    interval_loss.zero_()
                    interval_steps, interval_t0 = 0, time.perf_counter()
```
- `log_every` ステップごと(と最後のステップ)の**記録**:`mean_loss` = 前回の記録以降のステップの平均の損失(`.item()` = GPUからCPUへの、唯一のコピー)。**損失がNaNか無限大なら**、`diverged.pt` を(調査用に)保存して**止まる**。そうでなければ、1秒あたりのトークン数を計算し、`train` の行を書き、1行表示し、区間のカウンターをリセットする。

```python
                stop_now = stop.requested or (cfg.stop_after is not None and step >= cfg.stop_after)
                if stop_now and interval_steps > 0:      # log the unfinished interval before stopping
                    mean_loss = (interval_loss / interval_steps).item()
                    write_row("train", lr, mean_loss, None, float(grad_norm), None)
                    interval_loss.zero_()
                    interval_steps = 0
                if step % cfg.eval_every == 0 or step == cfg.max_steps:
                    checkpoint_and_sample(lr)
                    interval_t0 = time.perf_counter()
                elif stop_now:
                    save_checkpoint(last_path, state_for_save())
                    print(f"step {step:>6} | saved last.pt")
                if stop_now and step < cfg.max_steps:
                    stopped_early = True
                    print(f"Stopped at step {step}. Continue with --resume.")
                    break
    finally:
        log_f.close()
```
- `stop_now` — Ctrl+Cが押された、**または** `stop_after` に達した(Day 9のテスト用の、わざとの「中断」)。
- **終わっていない記録の区間**がある状態で止まるなら、先にその平均を書く(記録されないステップが無いように)。
- **評価**は `eval_every` ステップごと(と最後)→ `checkpoint_and_sample`(これが `last.pt` も保存する)。評価のステップ**ではない**所で止まるなら、`last.pt` を保存するだけ。
- 早く止まる場合:`stopped_early` を立て、再開の方法を表示し、ループを `break` で抜ける。
- `finally: log_f.close()` — エラーの後でも、**必ず**ログファイルを閉じる。

### 5.9 まとめ

```python
    return {"run_dir": str(run_dir), "step": step, "max_steps": cfg.max_steps, "best_val": best_val,
            "stopped_early": stopped_early, "device": device_name, "precision": precision, "last_lr": last_lr,
            "elapsed_s": elapsed_before + time.perf_counter() - t_start}
```
- 呼び出し側(`day9_train.py`。表示して、実行ログを書く)のための小さな辞書。

---

## 6. 学習の後の、学習のフォルダー

| ファイル | 中身 |
|---|---|
| `train_config.json`、`model_config.json` | 設定(再開した学習は、これと同じでなければならない) |
| `log.csv` | 記録/評価のステップごとに1行(12列) |
| `samples.md` | 評価のたびの、決まった書き出しの貪欲な続き |
| `last.pt` | 最新のチェックポイント(再開用) |
| `best.pt` | 検証の損失が最も低いチェックポイント(生成・最終モデル用) |
| `diverged.pt` | 損失が爆発したときだけ |

## 7. 知っておくと良い小さなこと

- **正確に比べるなら、`log_every` の倍数のステップで止めること。**`log_every` の倍数でないステップで止めると、止めたステップに平均の途中の行が余分に書かれ、再開後の最初の区間は、少ないステップの平均になるので、中断しなかった学習と比べて、いくつかの記録された `train_loss` が少し違う(実際の学習は同一のまま)。Day 9のスモークテストは、倍数で止める(log-every 10 で300)。
- **ステップ0の検証の損失**は、学習前の基準(≈ `ln(vocab)` のはず — Day 8の確認)。
- **`best.pt` と `last.pt`:**生成には `best.pt` を使い、学習を続けるには `last.pt` を使う。
- **損失の確認は記録のときだけ**なので、NaNに気づくまでに、最大 `log_every` ステップかかることがある。
- **重み減衰は、埋め込みにもかかる**(行列なので)が、正規化の倍率にはかからない。
- **パープレキシティ** `exp(損失)` は、損失より感覚がつかみやすい:最初は語彙サイズくらいで、学習が効くと急に下がる。
- **このハンドラーは、メインスレッドでしかCtrl+Cを捕まえられない**(一部のテスト実行環境では、ハンドラーは飛ばされる)。
- **`torch.amp.GradScaler(device_type, …)` を持つPyTorchが必要**(5.4を参照)。

## 8. 一段落でまとめ

`training.py` は学習のすべてを動かします:ステップごとに、ウォームアップしてからコサインで下がる学習率を設定し、混合精度で小バッチの勾配を累積し、勾配をクリップしてAdamW(重み減衰は行列だけ)で更新します。`log.csv` に記録し、固定した検証用の窓で評価し、貪欲なサンプルの文章を書き、`last.pt` と `best.pt` を安全に(一時ファイルに書いてから名前を変えて)保存します。チェックポイントには、モデル、オプティマイザ、スケーラー、ステップ、バッチの生成器を含むすべての乱数の状態が入るので、再開した学習は中断しなかった学習とまったく同じに続き、Ctrl+Cは丁寧に止まり、強制終了は再開時に片付けられます。
