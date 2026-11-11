# day1_env_check.py — やさしい1行ずつの解説(日本語)

## 0. 全体像(まずここから)

このスクリプトは、AIモデルの学習を始める前に行う、**コンピューターの健康診断**です。

お医者さんは、身長・体重・心拍・血圧を測り、最後にトレッドミルで少し走らせます。このスクリプトも同じことをします。

| 健康診断 | このスクリプトがやること |
|---|---|
| 基本の身体測定 | Python、OS、CPU、RAM、ディスクの空きを記録する |
| 読み書きは正常か? | 日本語を保存して読み戻す(UTF-8テスト) |
| 外の世界とつながるか? | pypi.org、pytorch.org、github.com に接続してみる |
| どんな筋肉があるか? | PyTorch にGPU名・VRAM・compute capability を聞く |
| トレッドミルの速度テスト | 行列積のベンチマーク(fp32 / fp16 / bf16) |
| どれだけ荷物を持てるか? | GPUメモリが一杯になるまで確保する(VRAMプローブ) |
| ミニ運動 | 混合精度で小さな学習を回す(AMPスモークテスト) |
| 診断書を書く | JSONファイルを保存し、`run_log.md` にメモを追記する |

実行方法:
.
```
.venv\Scripts\python.exe scripts\day1_env_check.py --machine school
```

`--machine school` か `--machine home` は、どのPCの結果かをログに残すための**ラベル**にすぎません。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **venv** | このプロジェクト専用のPythonパッケージの箱。他のプロジェクトと混ざらない。 |
| **GPU** | グラフィックボード。単純な計算を大量に同時にこなす。AI学習に必須。 |
| **VRAM** | GPU専用のメモリ(GPUの机)。モデルが机に乗らないと学習は落ちる。 |
| **CUDA** | NVIDIAが作った、GPUで計算するための仕組み。 |
| **Compute capability** | GPUの能力のバージョン番号。7.5 = RTX 2060(Turing)、8.9 = RTX 4060(Ada)。8.0以上は「Ampere世代以降」。 |
| **fp32 / fp16 / bf16** | 小数の保存方法の3種類。fp32は32ビットで正確だが重い。fp16とbf16は16ビットで半分のサイズ、速い。 |
| **fp16 と bf16 の違い** | fp16は精度は高いが扱える数の範囲が狭い(とても小さい数は0になる)。bf16は範囲が広い(fp32並み)が精度は低め。bf16は新しいGPU(8.0以上)が必要。 |
| **matmul** | 行列積(行列のかけ算)。ニューラルネットの中心的な計算。 |
| **TFLOPS** | 1秒間に何兆回の小数計算ができるか。大きいほど速い。 |
| **AMP / autocast** | 自動混合精度。安全な所は16ビット、必要な所は32ビットで、PyTorchが自動で使い分ける。 |
| **GradScaler** | fp16用の助っ人。勾配(学習の信号)が小さすぎるとfp16では0になってしまうので、先に損失を大きな数倍にして、後で元に戻す。 |
| **WDDM** | Windowsのグラフィックドライバーの動作モード。VRAMが足りないとき、普通のRAMを借りることがある(遅い)。 |
| **JSON** | 構造のあるデータ(キーと値)を保存する、テキスト形式。 |
| **スモークテスト** | 「火を噴かずに動くか?」だけを見る、とても小さなテスト。 |

---

## 2. ファイル先頭のdocstring(説明文)

```python
r"""Day 1 - environment check for Phase 1 (Model A).
...
"""
```

- `r"""` 〜 `"""` — ファイルの一番上にある**三重引用符の文字列**を *docstring* と呼びます。ただの説明文で、Pythonは実行しません。
- 先頭の `r` は **raw文字列**(生の文字列)の意味。バックスラッシュ `\` をそのまま文字として扱います。使い方の例に `.venv\Scripts\...` のようなWindowsのパスがあり、`r` がないと `\S` が特別な記号と誤解される恐れがあるためです。
- 中身は英語と日本語の両方で、同じ内容(スクリプトが行う8つのことと、実行方法)を説明しています。

---

## 3. import(道具を持ってくる)

```python
from __future__ import annotations
```
- 型ヒント(`str | None` など)を、すぐ評価せず「ただの文字」として扱います。新しい書き方が、より多くのPythonバージョンで使えるようになります。

```python
import argparse
import importlib.metadata as md
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
```

| 行 | 何が使えるか |
|---|---|
| `import argparse` | `--machine school` のようなコマンドラインの指定を読み取る。 |
| `import importlib.metadata as md` | 「パッケージXは何版が入っている?」を調べる。`md` と短くしている。 |
| `import json` | Pythonの辞書をJSONテキストに変換する。 |
| `import math` | 数学の道具。ここでは `math.isfinite`(NaNや無限大でないか)だけ使う。 |
| `import os` | OSの道具。ここでは `os.cpu_count()` だけ。 |
| `import platform` | PythonとOSの情報。 |
| `import shutil` | ファイルの道具。ここでは `shutil.disk_usage`(ディスクの空き)。 |
| `import subprocess` | 別のプログラム(`nvidia-smi` など)を実行する。 |
| `import sys` | Python自身の情報(バージョン、コンソール、パス)。 |
| `import time` | ストップウォッチ(`time.time()`)。 |
| `from datetime import datetime` | 現在の日付と時刻。 |
| `from pathlib import Path` | ファイルパスをきれいに扱う。 |
| `from typing import Any` | 型ヒントで「何でもよい」を表す型。 |
| `from urllib.request import Request, urlopen` | 標準のWebアクセス道具(追加インストール不要)。 |

---

## 4. 定数(最初に一度だけ決める設定)

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
```
- `__file__` = このスクリプト自身のパス(`scripts/day1_env_check.py`)。
- `.resolve()` = 完全な絶対パスにする。
- `.parent` = そのファイルがあるフォルダ(`scripts/`)。
- もう一度 `.parent` = さらに一つ上(**リポジトリのルート**)。
- `: Path` は型の注釈で、「この変数はPathです」という意味。

```python
RUN_LOG: Path = REPO_ROOT / "run_log.md"
RUNS_DIR: Path = REPO_ROOT / "runs"
```
- `Path` では `/` が**パスをつなぐ**演算子になります。つまり `<repo>/run_log.md` と `<repo>/runs`。

```python
EXPECTED_GPU_TAG: dict[str, str] = {"school": "2060", "home": "4060"}
```
- 辞書:マシンのラベル → GPU名に含まれるはずの数字。後で、`--machine` を間違えたときに警告するために使います。

```python
NETWORK_TARGETS: dict[str, str] = {
    "pypi": "https://pypi.org/simple/pip/",
    "pytorch_wheels": "https://download.pytorch.org/whl/",
    "github": "https://github.com",
}
```
- 接続できる必要がある3つのサイト:Pythonパッケージの入手先(PyPI)、PyTorchのダウンロード先、GitHub。学校のネットワークではブロックされることがあります。

```python
PACKAGES: list[str] = ["torch", "numpy", "psutil", "tqdm", "pytest"]
```
- バージョンを記録したいパッケージの一覧。

```python
JAPANESE_PROBE: str = "今日は晴れ。売上・給与・勤怠"
```
- 日本語が文字化けしないかを試すためのサンプル文(「今日は晴れ。売上・給与・勤怠」)。

---

## 5. ヘルパー関数(お手伝い役)

### 5.1 `setup_utf8_console`

```python
def setup_utf8_console() -> None:
    """Make printing Japanese safe on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
```
- `def ... -> None` — 何も返さない関数を定義する。
- `for stream in (sys.stdout, sys.stderr)` — 2つの出力口(通常の出力とエラー出力)に同じ処理をする。
- `stream.reconfigure(encoding="utf-8", errors="replace")` — コンソールに「UTF-8を使って」と伝える。`errors="replace"` は、どうしても表示できない文字があっても落ちずに `?` で置き換える、という意味。
- `# type: ignore[attr-defined]` — 型チェッカーへのメモ:「`reconfigure` が見えないのは承知しているので無視して」。
- `except (AttributeError, ValueError): pass` — その出力口が `reconfigure` に対応していなければ、何もせず静かに進む。

### 5.2 `run_cmd`

```python
def run_cmd(cmd: list[str], timeout: float = 15.0) -> str | None:
    """Run a command, return stripped stdout or None on any failure."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "").strip()
    return text if out.returncode == 0 and text else None
```
- コマンドをリスト(例:`["nvidia-smi", "--query-gpu=..."]`)で受け取り、制限時間は15秒。
- `-> str | None` — 文字列**または** `None` を返す。
- `subprocess.run(...)` — コマンドを実行する。
  - `capture_output=True` — 出力を画面に出さず、受け取る。
  - `text=True` — 生のバイトではなく文字列で受け取る。
  - `timeout=timeout` — 遅すぎたら強制終了。
  - `check=False` — コマンドが失敗しても例外を出さない(自分で確認する)。
- `except (OSError, subprocess.SubprocessError): return None` — プログラムが存在しない(`OSError`)か、時間切れ(`SubprocessError`)なら `None` を返す。
- `text = (out.stdout or "").strip()` — 出力(なければ空文字)の前後の空白・改行を取り除く。
- 最後の行 — コマンドが成功(`returncode == 0`)**かつ**文字が空でないときだけ `text` を返し、そうでなければ `None`。

### 5.3 `gb`

```python
def gb(num_bytes: float) -> float:
    return round(num_bytes / 1024**3, 2)
```
- バイトをギガバイトに変換する。`1024**3` = 1,073,741,824。`round(..., 2)` は小数第2位まで。

---

## 6. GPUが要らないチェック

### 6.1 `check_system`

```python
def check_system() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "in_venv": sys.prefix != sys.base_prefix,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine_arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "stdout_encoding": getattr(sys.stdout, "encoding", None),
        "disk_free_gb": gb(shutil.disk_usage(REPO_ROOT).free),
        "repo_root": str(REPO_ROOT),
    }
```
辞書(名前 → 値)を1行ずつ作ります:
- `"python"` — Pythonのバージョン(例:`3.11.9`)。
- `"python_executable"` — 実行中の `python.exe` の場所。venvのものかどうか分かる。
- `"in_venv"` — `sys.prefix != sys.base_prefix`。venvの中ではこの2つが違うので、`True` なら「venvの中で動いている」。
- `"os"` — f文字列(`{}` の部分に値が入る文字列)。例:`Windows 11 (10.0.22631)`。
- `"machine_arch"` — CPUの種類(例:`AMD64`)。
- `"cpu_count"` — CPUのスレッド数。
- `"stdout_encoding"` — コンソールの文字コード。`getattr(..., None)` は、無ければ安全に `None` を返す。
- `"disk_free_gb"` — リポジトリがあるドライブの空き容量。
- `"repo_root"` — リポジトリのパスを文字列に(`str(...)`)。JSONは `Path` を保存できないため。

```python
    try:
        import psutil  # type: ignore

        info["ram_total_gb"] = gb(psutil.virtual_memory().total)
        info["ram_available_gb"] = gb(psutil.virtual_memory().available)
    except ImportError:
        info["ram_total_gb"] = None
    return info
```
- `psutil`(RAMを調べるパッケージ)を読み込む。関数の**中**で `import` するので、無くてもスクリプト全体は落ちない。
- `ram_total_gb` = RAMの総量。`ram_available_gb` = 今空いているRAM(2行目は1行目と同じ考えで、`.total` が `.available` に変わっただけ)。
- `psutil` が入っていなければ(`ImportError`)、RAMは `None` で記録。
- `return info` — 辞書を返す。

### 6.2 `check_packages`

```python
def check_packages() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions
```
- 空の辞書から始める。
- `PACKAGES` の名前を1つずつ順番に処理する。
- `md.version(name)` — インストール済みのバージョン(例:`"2.4.0+cu121"`)を取得。
- 入っていなければ `PackageNotFoundError` になるので、`None` を保存する。

### 6.3 `check_utf8_roundtrip`

```python
def check_utf8_roundtrip() -> bool:
    """Write and re-read Japanese text explicitly as UTF-8."""
    tmp = RUNS_DIR / "_utf8_probe.txt"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_text(JAPANESE_PROBE, encoding="utf-8")
    ok = tmp.read_text(encoding="utf-8") == JAPANESE_PROBE
    tmp.unlink(missing_ok=True)
    return ok
```
「ラウンドトリップ」= 行って戻ってきて、変わっていないか確認すること。
- `tmp = RUNS_DIR / "_utf8_probe.txt"` — 一時ファイルのパス。
- `RUNS_DIR.mkdir(parents=True, exist_ok=True)` — `runs` フォルダを作る。`parents=True` は足りない親フォルダも作る。`exist_ok=True` は「既にあってもエラーにしない」。
- `tmp.write_text(JAPANESE_PROBE, encoding="utf-8")` — 日本語の文をUTF-8で保存。
- `ok = tmp.read_text(encoding="utf-8") == JAPANESE_PROBE` — 読み戻して比較。同じなら `True`。
- `tmp.unlink(missing_ok=True)` — 一時ファイルを削除(既に無くてもエラーにしない)。
- `return ok`。

### 6.4 `check_url`

```python
def check_url(url: str, timeout: float = 8.0) -> dict[str, Any]:
    start = time.time()
    try:
        req = Request(url, headers={"User-Agent": "model-a-env-check"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read(1)
            return {"ok": True, "status": resp.status, "seconds": round(time.time() - start, 2)}
    except Exception as exc:  # noqa: BLE001 - we want every failure reason
        return {"ok": False, "error": repr(exc)[:200], "seconds": round(time.time() - start, 2)}
```
- `start = time.time()` — ストップウォッチ開始。
- `Request(url, headers={"User-Agent": ...})` — Webリクエストを作り、「名札」を付ける。名札が無いリクエストを断るサイトがあるため。
- `with urlopen(req, timeout=timeout) as resp:` — ページを開く。`with` は終わりに自動で接続を閉じる。
- `resp.read(1)` — **1バイトだけ**読む。サイトが応答するか知りたいだけで、ページ全体は要らない。
- `return {"ok": True, "status": resp.status, ...}` — 成功:ステータスコード(200 = OK)と、かかった秒数。
- `except Exception as exc:` — **どんな**エラーでも捕まえる(接続できない、ブロック、タイムアウトなど)。理由を知りたいので全部捕まえる。
- `repr(exc)[:200]` — エラーを文字列にして200文字まで。
- `# noqa: ...` は、コードスタイルのチェッカーに「わざとやっています」と伝えるコメント。

### 6.5 `check_nvidia_smi`

```python
def check_nvidia_smi() -> dict[str, str | None]:
    line = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {"nvidia_smi": line}
```
- `nvidia-smi` はNVIDIA純正のツールで、グラフィックドライバーと一緒に入る。
- `--query-gpu=name,driver_version,memory.total` — GPU名、ドライバーのバージョン、総メモリだけを聞く。
- `--format=csv,noheader` — 見出しなしの、カンマ区切りの1行。
- 上の `run_cmd` を使うので、結果は文字列の1行か `None`。
- なぜ確認するのか? PyTorchを通さず、**ドライバーに直接**聞けるから。`nvidia-smi` は動くのにPyTorchが「CUDAなし」と言うなら、原因はGPUではなくPyTorchのインストールです。

---

## 7. PyTorch + GPU が必要なチェック

### 7.1 `torch_info`

```python
def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": repr(exc)[:300]}
```
- PyTorchを `try` の中で読み込む。失敗しても、スクリプト全体が落ちる代わりに「`importable: False` とエラー内容」の小さな報告を返す。

```python
    info: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not info["cuda_available"]:
        return info
```
- `version` — PyTorchのバージョン。
- `cuda_build` — このPyTorchが**ビルドされた**CUDAのバージョン(`None` ならCPU専用のPyTorchで、よくある間違い)。
- `cuda_available` — 今、PyTorchは実際にGPUを見つけて使えるか?
- 使えなければここで早めに返す。GPUが無いと以下は意味がないので。

```python
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    free_b, total_b = torch.cuda.mem_get_info(idx)
```
- `idx = 0` — 1番目のGPUを使う。
- `props` — GPUの詳細(名前、プロセッサ数など)を持つオブジェクト。
- `major, minor` — compute capabilityを2つの数に分けたもの(8.9なら `8` と `9`)。
- `free_b, total_b` — VRAMの空きと総量(バイト)。

```python
    info.update(
        device_count=torch.cuda.device_count(),
        gpu_name=props.name,
        compute_capability=f"{major}.{minor}",
        sm_count=props.multi_processor_count,
        vram_total_gb=gb(total_b),
        vram_free_gb=gb(free_b),
        native_bf16=major >= 8,  # Ampere (8.x) and newer have real bf16
    )
```
`info.update(...)` で、たくさんの項目を一度に追加します:
- `device_count` — GPUの数。
- `gpu_name` — 例:`NVIDIA GeForce RTX 4060 Laptop GPU`。
- `compute_capability` — 例:`"8.9"`。
- `sm_count` — SM(ストリーミングマルチプロセッサ)の数。GPUの「小さな計算ユニットのグループ」。多いほど高性能。
- `vram_total_gb`、`vram_free_gb` — `gb()` で変換。
- `native_bf16` — `major >= 8` なら `True`。RTX 2060は7.5なので `False`、RTX 4060は8.9なので `True`。

```python
    try:
        info["bf16_reported_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        info["bf16_reported_supported"] = None
    info["recommended_precision"] = (
        "bf16 (no scaler)" if info["native_bf16"] else "fp16 + GradScaler"
    )
    return info
```
- PyTorchにも直接「bf16に対応していますか?」と聞く。(古いGPUでは、PyTorchが遅い**エミュレーション**で「はい」と答えることがあるため、自分の `native_bf16` 判定も残している。)この問い合わせが失敗したら `None`。
- `recommended_precision` — 一言のおすすめ:新しいGPUなら「bf16、スケーラー不要」、古いGPUなら「fp16 + GradScaler」。

### 7.2 `bench_matmul`

```python
def bench_matmul(dtype_name: str, n: int = 4096, iters: int = 30) -> dict[str, Any]:
    import torch

    dtype = getattr(torch, dtype_name)
```
- 引数:数値形式の名前(例:`"float16"`)、行列のサイズ `n`(4096×4096)、繰り返し回数(30)。
- `getattr(torch, "float16")` は、**文字列**を本物のオブジェクト `torch.float16` に変える。

```python
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
```
- 4096×4096の乱数の行列を2つ、GPU上に直接、指定の数値形式で作る。(`b` は `a` と同じ考え方。)

```python
    for _ in range(5):  # warm-up
        a @ b
    torch.cuda.synchronize()
```
- `a @ b` = 行列積。
- まず5回、**ウォームアップ**として実行(最初の数回はGPUが「目覚める」ので遅い)。
- `_` は使い捨ての変数名(カウンターは要らない)。
- `torch.cuda.synchronize()` — **GPUが本当に終わるまで待つ。**GPUは裏で動くので、Pythonはすぐ次へ進んでしまう。待たないと時間の測定が狂う。

```python
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize()
```
- **GPU上のストップウォッチの印**を2つ作る。Pythonで測るより、GPU自身で測るほうが正確。
- `start.record()` で1つ目の印 → 30回かけ算 → `end.record()` で2つ目の印 → 全部終わるのを待つ。

```python
    ms = start.elapsed_time(end) / iters
    tflops = 2 * n**3 / (ms / 1000) / 1e12
```
- `elapsed_time` は2つの印の間の合計ミリ秒。30で割ると**1回あたりのms**。
- TFLOPSの式:n×nの行列2つのかけ算はおよそ `2 × n³` 回の演算(n²個の出力それぞれに、n回のかけ算 + n回の足し算)。これを秒(`ms / 1000`)で割ると1秒あたりの演算数、さらに `1e12` で割ると*テラ*(兆)単位。

```python
    del a, b
    torch.cuda.empty_cache()
    return {"ms_per_matmul": round(ms, 3), "tflops": round(tflops, 2)}
```
- 大きな行列を消してGPUメモリを返し、次のテストをきれいな状態で始める。
- 速度の結果を返す。

### 7.3 `probe_vram`

```python
def probe_vram(chunk_mb: int = 256) -> dict[str, Any]:
    import torch

    free_b, total_b = torch.cuda.mem_get_info(0)
    chunks: list[Any] = []
    allocated = 0
```
- 256MBずつ、VRAMを**実際に**どれだけ確保できるか調べる。
- 何かする**前に**空き/総量を測る(docstringにある通り、Windowsは実際より多く見せることがあるため重要)。
- `chunks` — 確保した塊を入れるリスト(Pythonにすぐ捨てられないため)。`allocated` — 確保した合計バイト数。

```python
    try:
        while allocated < total_b:
            chunks.append(torch.empty(chunk_mb * 1024**2, dtype=torch.uint8, device="cuda"))
            allocated += chunk_mb * 1024**2
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    finally:
        chunks.clear()
        torch.cuda.empty_cache()
```
- `while allocated < total_b` — カードの総メモリ分まで確保するのを続ける。
- `torch.empty(256 * 1024**2, dtype=torch.uint8, ...)` — GPU上に256MBを(中身は埋めずに)確保。`uint8` は1つ1バイト。リストに追加し、合計に足す。
- `except (OutOfMemoryError, RuntimeError): pass` — GPUが「もう無理」と言うのは**想定どおり**。静かに止める。
- `finally:` — **何があっても**実行される。塊を全部手放し(`clear()`)、メモリをGPUに返す(`empty_cache()`)。

```python
    spill = allocated > free_b + 256 * 1024**2
```
- 最初に空いていたVRAM(+誤差256MB)より多く確保できた場合、Windowsがデータを**普通のRAM**に逃がした可能性が高い。それは遅いので `spill`(あふれ)として印を付ける。

```python
    return {
        "usable_vram_gb": gb(min(allocated, free_b)),  # real VRAM-resident budget
        "free_before_probe_gb": gb(free_b),
        "allocated_in_probe_gb": gb(allocated),
        "total_vram_gb": gb(total_b),
        "system_memory_spill_suspected": spill,
    }
```
- `usable_vram_gb` — **正直な**数字:「確保できた量」と「空いていた量」の小さいほう。モデルの設計はこの数字を予算にすべき。
- 他の値は記録用。`gb(...)` は前と同じ変換。

### 7.4 `amp_smoke`

```python
def amp_smoke(dtype_name: str, use_scaler: bool, steps: int = 30) -> dict[str, Any]:
    """Tiny regression with autocast; proves the precision path works."""
    import torch

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)
```
- **ミニ学習**。目的は良いモデルを作ることではなく、このGPUで16ビット学習が落ちたり、でたらめな値にならないかを証明すること。
- `dtype` — さっきと同じ「文字列 → オブジェクト」の変換。
- `torch.manual_seed(0)` — 乱数を固定して、結果を再現できるようにする。

```python
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
```
- `Sequential(...)` — 層を順番に通す小さなニューラルネット:
  - `Linear(256, 512)` — 256個の数を受け取り、512個にする。
  - `GELU()` — なめらかな「活性化関数」。非線形性を加え、直線だけでなく曲がった関係も学べるようにする。
  - `Linear(512, 256)` — 256個に戻す。
- `.cuda()` — モデルをGPUに移す。
- `AdamW` — オプティマイザ:モデルの重みを更新する役。`lr=1e-3` は学習率0.001(一歩の大きさ)。

```python
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
```
- GradScalerを作る。`enabled=False` なら「何もしない」版になる(スケーリングが要らないbf16用)。
- `except` は**古いPyTorch**用で、そちらでは関数の場所が違う。

```python
    x = torch.randn(64, 256, device="cuda")
    y = torch.randn(64, 256, device="cuda")
```
- `x` = 偽の入力(64サンプル × 256個の数)。`y` = 偽の正解(同じ形、`x` と同じ考え方)。仕組みだけを試すので、乱数のデータで十分。

```python
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = torch.nn.functional.mse_loss(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.item()))
```
学習1ステップを30回繰り返します:
1. `opt.zero_grad(set_to_none=True)` — 前のステップの勾配(学習の信号)を消す。`set_to_none=True` は少し速くメモリも節約。
2. `with torch.autocast(...)` — このブロックの中では、PyTorchが安全な所で自動的に16ビット計算を使う。
3. `model(x)` — 順伝播:モデルの予想。`mse_loss(予想, y)` — 平均二乗誤差:「予想は正解からどれだけ離れているか?」。小さいほど良い。
4. `scaler.scale(loss).backward()` — 損失を(スケーラーが有効なら)大きな倍率でかけ、**逆伝播**で勾配(各重みをどちらへ動かすか)を計算。
5. `scaler.unscale_(opt)` — 勾配を元の大きさに割り戻す。次の行が本当の値で動くように。
6. `clip_grad_norm_(..., 1.0)` — 勾配が危険なほど大きければ、全体の大きさが最大1.0になるよう縮める。学習の「暴走」を防ぐ。
7. `scaler.step(opt)` — 重みを更新(NaNや無限大を見つけたら、スケーラーはこのステップを飛ばす)。
8. `scaler.update()` — 次のステップのために倍率を調整。
9. `losses.append(float(loss.item()))` — このステップの損失を、普通のPythonの数として保存。

```python
    finite = all(math.isfinite(v) for v in losses)
    return {
        "dtype": dtype_name,
        "scaler": use_scaler,
        "first_loss": round(losses[0], 4),
        "last_loss": round(losses[-1], 4),
        "passed": finite and losses[-1] < losses[0],
    }
```
- `finite` — 損失が**全部**ちゃんとした数か(NaN・無限大なし)?
- `"passed"` — 全部有限**かつ**最後の損失が最初より小さい(少しは学習できた)ときだけ合格。

---

## 8. レポート作成:`append_run_log`

```python
def append_run_log(machine: str, stamp: str, report: dict[str, Any], json_name: str) -> None:
    t = report["torch"]
    sysinfo = report["system"]
```
- ラベル、時刻、レポート全体、JSONのファイル名を受け取る。`t`(PyTorch情報)と `sysinfo`(システム情報)という短い名前を作り、以降の行を短くする。

```python
    lines = [
        "",
        f"## {stamp} | {machine} | Day 1 environment check",
        f"- Machine label: {machine} (what actually ran: this entry's machine)",
        f"- OS: {sysinfo['os']}",
        f"- Python: {sysinfo['python']} (venv: {sysinfo['in_venv']})",
        f"- RAM: {sysinfo.get('ram_total_gb')} GB, free disk: {sysinfo['disk_free_gb']} GB",
        f"- nvidia-smi: {report['nvidia_smi'].get('nvidia_smi')}",
        f"- PyTorch: {t.get('version')} (CUDA build {t.get('cuda_build')}), "
        f"cuda_available={t.get('cuda_available')}",
    ]
```
- Markdownのログ1件分を、**文字列のリスト**として作る。最初の空文字 `""` は、新しい記録の前の空行。
- `## ...` は時刻とマシン名のMarkdown見出し。
- `sysinfo['os']` は角かっこ(無いと落ちる)。`sysinfo.get('ram_total_gb')` は `.get`(無ければ `None`)。ここではどちらでも動くが、値が無いかもしれない所には `.get` を使っている。
- PyTorchの行は2つのf文字列に分かれているが、Pythonが**自動でつなげて**1行にする。

```python
    if t.get("cuda_available"):
        vp = report.get("vram_probe", {})
        lines += [
            f"- GPU: {t['gpu_name']}, {t['vram_total_gb']} GB total, "
            f"free at start (usable budget): {vp.get('usable_vram_gb')} GB, "
            f"compute capability {t['compute_capability']}",
            f"- VRAM probe: allocated {vp.get('allocated_in_probe_gb')} GB, "
            f"spill to system memory suspected: {vp.get('system_memory_spill_suspected')}",
            f"- Native bf16: {t['native_bf16']} -> plan: {t['recommended_precision']}",
        ]
```
- GPUが使えるときだけ、GPUの行を追加する。
- `report.get("vram_probe", {})` — プローブを飛ばした場合(`--skip-vram-probe`)は空の辞書を使うので、`.get(...)` が落ちずに `None` を返す。
- `lines += [...]` — リストの最後に複数行を追加する。

```python
        for name, res in report.get("matmul", {}).items():
            lines.append(f"- matmul 4096 {name}: {res['tflops']} TFLOPS")
        for res in report.get("amp", []):
            lines.append(
                f"- AMP smoke {res['dtype']} (scaler={res['scaler']}): "
                f"{'PASS' if res['passed'] else 'FAIL'}"
            )
```
- 1つ目のループ:数値形式(float32、float16、あればbfloat16)ごとに速度を1行。
- 2つ目のループ:AMPテストごとに `PASS` か `FAIL` を1行。
- `'PASS' if res['passed'] else 'FAIL'` — 1行のif/else。

```python
    net = ", ".join(f"{k}={'ok' if v['ok'] else 'BLOCKED'}" for k, v in report["network"].items())
    lines += [
        f"- Network: {net}",
        f"- UTF-8 Japanese round trip: {'PASS' if report['checks']['utf8_roundtrip'] else 'FAIL'}",
        f"- Overall: {'PASS' if report['overall_pass'] else 'FAIL'} "
        f"(details: runs/{json_name})",
    ]
```
- ネットワーク結果をループして `", "` でつなぎ、`pypi=ok, pytorch_wheels=ok, github=BLOCKED` のような文字列にする。
- ネットワーク、UTF-8、総合結果の行を追加する。(`PASS/FAIL` の書き方は前と同じ。)

```python
    if not RUN_LOG.exists():
        RUN_LOG.write_text("# Run log\n", encoding="utf-8")
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
```
- `run_log.md` がまだ無ければ、タイトル付きで作る。
- `"a"`(**追記**)モードで開く:末尾に足すだけで、古い記録は消さない。
- `"\n".join(lines)` — 全部の行を改行でつなぎ、最後に改行を1つ足す。

---

## 9. メインプログラム

```python
def main() -> int:
    setup_utf8_console()
```
- `main()` は整数を返す:`0` = 成功、`1` = 失敗(プログラムの終了コードのようなもの)。
- 最初に、コンソールを日本語に対応させる。

```python
    parser = argparse.ArgumentParser(description="Day 1 environment check")
    parser.add_argument("--machine", choices=["school", "home"], required=True)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-vram-probe", action="store_true")
    args = parser.parse_args()
```
- コマンドラインのオプションを定義する:
  - `--machine` は `school` か `home` のどちらかで、**必須**。
  - `--skip-network` — 値なしのフラグ。書けば `True`、書かなければ `False`(`store_true`)。
  - `--skip-vram-probe` — 同じ形式のフラグ。
- `parse_args()` が入力を読み取る。値は `args.machine` のように使う。

```python
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M")
```
- `stamp` — 読みやすい時刻(例:`2026-10-30 14:05`)。ログ用。
- `file_stamp` — ファイル名に安全な時刻(例:`20261030_1405`)。Windowsのファイル名には空白やコロンが使えないため。

```python
    print(f"=== Day 1 environment check | machine={args.machine} | {stamp} ===")
    print(f"Japanese display test: {JAPANESE_PROBE}")
```
- 見出しを表示し、日本語も表示する。コンソールで日本語が正しく見えるか、**自分の目で**確認できる。

```python
    report: dict[str, Any] = {"machine": args.machine, "timestamp": stamp}
    report["system"] = check_system()
    report["packages"] = check_packages()
    report["nvidia_smi"] = check_nvidia_smi()
    report["network"] = (
        {} if args.skip_network else {k: check_url(u) for k, u in NETWORK_TARGETS.items()}
    )
    report["torch"] = torch_info()
```
- レポートを辞書として始め、セクションごとに埋める(system、packages、nvidia-smi)。
- ネットワーク:`--skip-network` なら空の `{}`。そうでなければ**辞書内包表記**で、名前とURLの組ごとに `check_url` を実行し、同じ名前で結果を保存。
- 最後にPyTorchの情報を追加。

```python
    t = report["torch"]
    if t.get("cuda_available"):
        report["matmul"] = {"float32": bench_matmul("float32"), "float16": bench_matmul("float16")}
        if t["native_bf16"]:
            report["matmul"]["bfloat16"] = bench_matmul("bfloat16")
        if not args.skip_vram_probe:
            report["vram_probe"] = probe_vram()
        report["amp"] = [amp_smoke("float16", use_scaler=True)]
        if t["native_bf16"]:
            report["amp"].append(amp_smoke("bfloat16", use_scaler=False))
```
GPUが使えるときだけ実行:
- fp32とfp16の速度テストは必ず。bf16は、GPUが**ネイティブにbf16対応のときだけ**(RTX 4060は対応、RTX 2060は非対応)。
- 飛ばす指定がなければVRAMプローブ。
- fp16のAMPテストを、スケーラー**あり**で。bf16がネイティブ対応なら、bf16もスケーラー**なし**で実行(`.append` でリストに追加)。

```python
    checks: dict[str, bool] = {
        "python_3_10_plus": sys.version_info >= (3, 10),
        "running_in_venv": report["system"]["in_venv"],
        "torch_importable": bool(t.get("importable")),
        "cuda_available": bool(t.get("cuda_available")),
        "fp16_amp_ok": any(r["dtype"] == "float16" and r["passed"] for r in report.get("amp", [])),
        "utf8_roundtrip": check_utf8_roundtrip(),
    }
    report["checks"] = checks
    report["overall_pass"] = all(checks.values())
```
最終的な成績表:6つのYes/No質問です。
- Pythonは3.10以上か?
- venvの中で動いているか?
- PyTorchを読み込めるか?
- CUDAが使えるか?
- fp16のAMPテストに合格したか?(`any(...)` =「1つでも真があるか?」)
- 日本語のUTF-8ラウンドトリップは正常か?(テストはここで実行される)
- `all(checks.values())` — **全部**が `True` のときだけ総合合格。
- `bool(...)` はきれいな `True`/`False` にそろえる。

```python
    tag = EXPECTED_GPU_TAG[args.machine]
    if t.get("gpu_name") and tag not in t["gpu_name"]:
        print(f"NOTE: --machine {args.machine} usually has an RTX {tag}, but found {t['gpu_name']}.")
        print("      Check that you used the right --machine label; the run log must be truthful.")
```
- ラベルから、期待されるGPUの数字(学校なら `"2060"`)を調べる。
- GPU名があり、**かつ**その数字が含まれていなければ警告を表示:`--machine` を間違えた可能性が高い。(2つ目の `print` は警告の2行目で、同じ考え方。)

```python
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    json_name = f"env_check_{args.machine}_{file_stamp}.json"
    (RUNS_DIR / json_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    append_run_log(args.machine, stamp, report, json_name)
```
- `runs/` があることを確認。
- `env_check_school_20261030_1405.json` のようなファイル名を作る。
- `json.dumps(report, indent=2, ensure_ascii=False)` — レポートをインデント付きのきれいなJSONにする。`ensure_ascii=False` で、日本語が `\u4eca...` のようなコードにならず日本語のまま残る。
- UTF-8で保存し、`run_log.md` に要約を追記する。

```python
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== SUMMARY ===")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    blocked = [k for k, v in report["network"].items() if not v["ok"]]
    if blocked:
        print(f"NOTE: network targets not reachable: {blocked} (tell me, this affects installs)")
    print(f"\nSaved: runs/{json_name}  and an entry in run_log.md")
    return 0 if report["overall_pass"] else 1
```
- レポート全体を画面に表示。
- 要約:チェックごとに `[PASS]` か `[FAIL]` を1行。
- `blocked = [...]` — **リスト内包表記**で、失敗したサイトの名前を集める。空でなければ警告を表示。
- 保存先を表示。
- 全部合格なら `0`、そうでなければ `1` を返す。

```python
if __name__ == "__main__":
    raise SystemExit(main())
```
- `__name__ == "__main__"` は、ファイルを直接実行したときだけ真(他のファイルから `import` されたときは偽)。つまり、自分で実行したときだけ診断が始まる。
- `raise SystemExit(main())` — `main()` を実行し、その戻り値(0か1)をプログラムの終了コードにする。スクリプトやCIツールは、これで合格かどうかを知れる。

---

## 10. 実行の流れ(順番)

1. コンソールをUTF-8に → `--machine` を読む。
2. システム情報 → パッケージのバージョン → `nvidia-smi` → ネットワーク → PyTorch情報。
3. GPUが動けば → 速度テスト → VRAMプローブ → ミニ学習テスト。
4. 成績表(`checks`)を作る → 総合の合否。
5. マシンのラベルがおかしければ警告。
6. JSON保存 → `run_log.md` に追記 → 要約を表示。
7. 終了コード `0`(合格)か `1`(不合格)。

## 11. 結果の読み方

- **`native_bf16: False`(RTX 2060)** → 学習は fp16 + GradScaler で。
- **`native_bf16: True`(RTX 4060)** → bf16 を使う。スケーラーは不要。
- **`usable_vram_gb`** → モデル、バッチサイズ、オプティマイザの状態に使える、実際のメモリ予算。
- **`system_memory_spill_suspected: True`** → Windowsが実際より多くのメモリを「使える」ように見せるが、遅い。確保できた生の数字ではなく `usable_vram_gb` を信じる。
- **`BLOCKED` のネットワーク先** → そのネットワークでは pip / PyTorch のインストールが失敗するかもしれない。
- **`fp16_amp_ok` が `FAIL`** → このマシンでは、まだ混合精度学習が安全ではない。学習の前に直す。

## 12. 一段落でまとめ

このスクリプトは、コンピューターの情報を集め、日本語が安全に扱えるか、インターネットにつながるか、GPUの速度とメモリはどうかを試し、小さな16ビット学習テストを回し、すべてをPASS/FAILで採点して、結果をJSONファイルと `run_log.md` のメモの両方に保存します。これで、学校のPCと自宅ノートPCを正直に比べられます。
