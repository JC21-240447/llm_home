# Day 3 — day3_generate_questions.py — やさしい1行ずつの解説(日本語)

> **Day 3 の2つのファイルの関係**
> - **Day 3.1 `make_questions.py`** — チームの質問生成プログラム(機械)。
> - **Day 3(このファイル)`day3_generate_questions.py`** — *実行役*:チームの設定で生成プログラムを安全に動かし、結果を検査して保存する(動画で記録する実行)。
>
> *何を*動かすのかを知りたければ、**先に Day 3.1 を読んでください**。このファイルは、*どう動かすか*を説明します。
> Day 4(説明済み)は、このスクリプトの関数をいくつか再利用しています — その部分は「Day 4と同じ」と書きます。

---

## 0. 全体像(まずここから)

**工場の監督**(このスクリプト)と**機械**(`make_questions.py`)を想像してください。

監督は:
1. 原料と道具がすべてそろっているか確認する;
2. 原料を**新しい作業台**に移す(元のファイルには触れない);
3. **固定した設定**(文型ごとに最大2,000問、シード42)で機械を動かし、時間を測る;
4. 製品を**検査**する(重複なし、埋まっていないスロットなし、質問タイプが60);
5. 製品、**別のPCに持っていくzip**、指紋付きの**レシート**(`generation_info.json`)を保存する;
6. まとめと**例の質問10個**を表示し、`run_log.md` にメモを書く。

Day 4では、あとで(別のマシンで)コーパスを作り直して、この結果と比べます。

実行:

```
.venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home
```

---

## 1. 先に知っておきたい用語

用語はDay 4・5と同じ:**シード**、**SHA-256**、**作業フォルダー**、**往復**、**zip**。新しいもの:

| 用語 | かんたんな意味 |
|---|---|
| **`generation_info.json`** | 実行の「レシート」:設定、入力と出力の指紋、件数、時間。 |
| **別のPC用のzip** | `questions.csv` を圧縮したzip。別のマシンに持っていって比べられる。 |
| **`random.sample`** | リストから、異なる項目をランダムにいくつか選ぶ(重複なし)。 |

---

## 2. docstring(説明文)

- 英語と日本語、6つの手順、使い方のコマンド、そして**学校PCの注意**(大事なルールが2つ):
  - `pandas` と `openpyxl` は、**そのPC自身の `.venv` の中だけ**に入れる(学校のPythonには入れない);
  - 学校のリポジトリは**公開**されている:`data/source/` や生成したファイルは**コミットしない**(チームの表と内部用のラベルが入っているため)。
- Day 4が別のマシンで作り直して、zipと比べる、とも書いてある。

---

## 3. importと定数

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
```
- すべてDay 4と同じ(それぞれはDay 4を参照)。違いは2つ:`platform`(Pythonのバージョン。Day 6・7と同じ)と、`random`(10個の例のため)。`Counter` と `dataclass` は、ここでは要らない。

```python
SCRIPT_VERSION: str = "day3-v1"
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GENERATOR: str = "make_questions.py"
REQUIRED_INPUTS: tuple[str, ...] = (GENERATOR, "用語集.csv", "同義語辞書.csv", "DB.xlsx")
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
BROKEN_MARKS: tuple[str, ...] = ("{", "}", "のの", "社員の社員", "平均平均")
```
- Day 4と同じ(バージョン名、リポジトリのルート、生成プログラムのファイル名、必須の4ファイル、ぴったりのヘッダー、壊れた文の印)。ここには `SUMMARY_FILES` は無い。

---

## 4. ヘルパー関数

### 4.1 `sha256_bytes`

```python
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```
- Day 4と同じ:バイト列の指紋。

### 4.2 `check_inputs`(Day 4と同じ)

```python
def check_inputs(source_dir: Path) -> list[Path]:
    missing = [n for n in REQUIRED_INPUTS if not (source_dir / n).exists()]
    src_files = sorted(source_dir.glob("src_*.txt"))
    if not src_files:
        missing.append("src_*.txt")
    if missing:
        raise SystemExit(f"Missing in {source_dir}: {', '.join(missing)}")
    libs = [m for m in ("pandas", "openpyxl") if importlib.util.find_spec(m) is None]
    if libs:
        raise SystemExit(...)
    return src_files
```
- Day 4の `check_inputs` と同じ考え方:足りないファイル(必須の4つ + `src_*.txt` が最低1つ)を集め、あれば止まる。`pandas` と `openpyxl` が見つかるか確認し、無ければ止まる。文型ファイルを返す。
- ライブラリのメッセージは、**この `.venv` の中だけ**に入れるよう指示する:`.venv\Scripts\python.exe -m pip install pandas openpyxl`。

### 4.3 `copy_inputs`(Day 4の `prepare_work_dir` と同じ考え方)

```python
def copy_inputs(source_dir: Path, out_dir: Path, src_files: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_INPUTS:
        shutil.copy2(source_dir / name, out_dir / name)
    for f in src_files:
        shutil.copy2(f, out_dir / f.name)
```
- **新しい**出力フォルダーを作り(`exist_ok=False` → 既にあればエラー。だから本当に新しい)、生成プログラム、入力、文型ファイルをコピーする。Day 4との違い:この関数はフォルダーのパスを**受け取る**(時刻付きの名前は `main` が決める)。

### 4.4 `run_generator`(Day 4と同じ)

```python
def run_generator(out_dir: Path, per_pattern: int, seed: int) -> tuple[str, float]:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, GENERATOR, "--corpus", str(per_pattern), "--seed", str(seed)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"The generator stopped with exit code {proc.returncode}.")
    return proc.stdout, seconds
```
- Day 4とまったく同じ考え方:コマンド `python make_questions.py --corpus 2000 --seed 42` を作り、**新しいフォルダーの中で**実行し(`cwd=out_dir`。生成プログラムの相対的なファイル名が動き、出力もそこに作られる)、UTF-8を強制し、時間を測り、失敗したら出力を表示して止める。コンソールの出力と秒数を返す。(だからDay 3.1は「今いるフォルダーから」ファイルを読む。)

### 4.5 `check_questions`

```python
def check_questions(raw: bytes, expected_types: int) -> tuple[list[list[str]], dict[str, object]]:
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    if next(reader) != QUESTION_HEADER:
        raise SystemExit("questions.csv has an unexpected header.")
    rows = [r for r in reader]
    texts = [r[0] for r in rows]
    problems: list[str] = []
    if len(set(texts)) != len(texts):
        problems.append("duplicate questions")
    if any(any(m in t for m in BROKEN_MARKS) for t in texts):
        problems.append("broken sentences")
    types = {r[2] for r in rows}
    if len(types) != expected_types:
        problems.append(f"{len(types)} question types instead of {expected_types}")
    if problems:
        raise SystemExit("Check FAILED: " + ", ".join(problems))
```
- **Day 4の `check_corpus` のもっと単純な版。**CSVをバイト列から解析し(BOMは `utf-8-sig` が取る)、`next(reader)` でヘッダーを確認し、3つの確認をする:
  1. **重複**(集合がリストより短い);
  2. **壊れた文**(どれかの文に、どれかの `BROKEN_MARK`。`any` が入れ子);
  3. **質問タイプの数**が、期待する60と等しい。
- 問題があれば、全部を並べて止まる。(ここには、ラベルの確認や4列の確認は無い。Day 4が足した。)

```python
    lengths = [len(t) for t in texts]
    return rows, {
        "questions": len(rows),
        "patterns_with_questions": len({r[1] for r in rows}),
        "types": len(types),
        "characters": sum(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
    }
```
- 解析した行(後で例の表示に使う)と、基本の統計を返す:質問数、質問がある文型の数、タイプ数、総文字数、平均の長さ。

### 4.6 `write_zip`

```python
def write_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="questions.csv")
```
- zipを作る(`"w"` = 書き込み、`ZIP_DEFLATED` = 実際に圧縮する)。`zf.write(ファイル, arcname=...)` は、ファイルを**zipの中で `questions.csv` という名前**で保存する(フルパスではなく)。Day 4の `read_corpus_bytes` は、zipの中にちょうど1つの `questions.csv` を探す — これに合う形式。

### 4.7 `append_run_log`

```python
def append_run_log(log, machine, stats, per_pattern, seed, seconds, csv_sha, out_rel, zip_name) -> None:
    entry = [ "", f"## ... | {machine} | Day 3 question generation", "", ... "- Checks: PASS", "" ]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))
```
- Day 4のログ関数と同じ形:Markdownの行(マシン、生成プログラム、設定、結果の件数、**`questions.csv` のSHA-256**、出力フォルダーとzipの名前、「Checks: PASS」)を `run_log.md` に追記する。失敗していればスクリプトが先に止まるので、「PASS」は常に書かれる。

---

## 5. `parse_args`(Day 4と同じ形)

```python
ap.add_argument("--machine", choices=["school", "home"], required=True)
ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "data/source")
ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs/day3_generation")
ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
ap.add_argument("--per-pattern", type=int, default=2000, help="max questions per pattern (team setting: 2000)")
ap.add_argument("--seed", type=int, default=42, help="random seed (team setting: 42)")
ap.add_argument("--expected-types", type=int, default=60)
ap.add_argument("--examples", type=int, default=10, help="example questions to print")
ap.add_argument("--no-log", action="store_true")
```
- Day 4と同じオプション(マシン、入力フォルダー、作業の場所、実行ログ、文型ごとの上限、シード、期待するタイプ数、no-log)。新しいのは **`--examples`**(表示する例の質問の数、初期値10)。Day 4の報告書やチームのコーパスのオプションは、ここには無い。

---

## 6. `main`

### 6.1 準備とバナー

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    a.source_dir, a.out_root, a.run_log = a.source_dir.resolve(), a.out_root.resolve(), a.run_log.resolve()
```
- コンソールの安全対策は前と同じ。ここでは、3つのパスを**タプルの代入1行**で絶対パスにしている(Day 4の `setattr` のループの代わり)。

```python
    started = datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print(f" Day 3 question generation | machine={a.machine} | {started:%Y-%m-%d %H:%M:%S}")
    print(f" Python {platform.python_version()} | max {a.per_pattern} per pattern | seed {a.seed}")
    print("=" * 70)
```
- 開始時刻(`started`)を覚え、`20261111_143012` のようなフォルダー名に使える印を作る。
- `"=" * 70` は `=` を70回繰り返す(線)。バナーには、マシン、時刻、Pythonのバージョン、上限、シードが出る — **動画**で役に立つ。

### 6.2 入力をコピーして指紋を取る

```python
    src_files = check_inputs(a.source_dir)
    out_dir = a.out_root / stamp
    copy_inputs(a.source_dir, out_dir, src_files)
    print(f"Output folder: {out_dir}")
    print("Inputs (SHA-256, first 12 characters):")
    inputs = {p.name: sha256_bytes(p.read_bytes()) for p in [*(a.source_dir / n for n in REQUIRED_INPUTS), *src_files]}
    for name, h in inputs.items():
        print(f"  {name:<22} {h[:12]}")
```
- 入力を確認し、時刻付きの出力フォルダーを決め、ファイルをコピーする。
- `inputs` — ファイル名 → 指紋の辞書(Day 4の報告書の入力と同じ内包表記)。各名前(`<22` = 22文字で左寄せ)と、指紋の**最初の12文字**を表示する — 動画で、どのファイルを使ったかの証拠になる。

### 6.3 生成プログラムを実行

```python
    print("\nRunning the team's generator: python make_questions.py "
          f"--corpus {a.per_pattern} --seed {a.seed}")
    generator_out, seconds = run_generator(out_dir, a.per_pattern, a.seed)
    print(generator_out.rstrip())
```
- 正確なコマンドを表示し、実行し、生成プログラムが表示したもの(Day 3.1の合計の表)を表示する。`rstrip()` は末尾の空白を取る。

### 6.4 検査、指紋、zip

```python
    csv_path = out_dir / "questions.csv"
    raw = csv_path.read_bytes()
    rows, stats = check_questions(raw, a.expected_types)
    csv_sha = sha256_bytes(raw)
    zip_name = f"questions_corpus_{a.machine}_{stamp}.zip"
    write_zip(csv_path, out_dir / zip_name)
```
- 生成された `questions.csv` をバイト列で読み、検査を実行し、指紋(`csv_sha`)を取り、zipに `questions_corpus_<machine>_<時刻>.zip` という名前を付けて、同じフォルダーに作る。

### 6.5 レシート `generation_info.json`

```python
    info = {
        "script": "scripts/day3_generate_questions.py",
        "script_version": SCRIPT_VERSION,
        "machine": a.machine,
        "started": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "settings": {"per_pattern": a.per_pattern, "seed": a.seed},
        "inputs_sha256": inputs,
        "generation_seconds": round(seconds, 2),
        "stats": stats,
        "questions_csv_sha256": csv_sha,
        "zip": zip_name,
    }
    (out_dir / "generation_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
```
- Day 5のマニフェストと同じ種類の「レシート」:スクリプトとバージョン、マシン、開始時刻、Pythonのバージョン、**設定**、**入力**と**出力**の指紋、時間、統計、zipの名前。読みやすいJSON(日本語はそのまま)で保存する。

### 6.6 まとめと例

```python
    print(f"\nGeneration time: {seconds:.1f} s")
    print(f"Questions: {stats['questions']:,} | types: {stats['types']} | "
          f"patterns with questions: {stats['patterns_with_questions']} | avg chars: {stats['avg_chars']}")
    print("Check: PASS (no duplicates, no unfilled slots, 60 question types)")
    print(f"questions.csv SHA-256: {csv_sha}")
    print(f"Zip for the other PC: {out_dir / zip_name}")
```
- 時間、件数、PASSのメッセージ(検査に合格したときだけ到達)、`questions.csv` の**完全な指紋**、zipの場所を表示する。(「60 question types」の文字は固定の文。本当の確認は `--expected-types` を使った。)

```python
    print(f"\n{a.examples} example questions (random, seed {a.seed}):")
    for text, pid, tid, _ in random.Random(a.seed).sample(rows, min(a.examples, len(rows))):
        print(f"  [{tid} {pid}] {text}")
```
- `random.Random(a.seed).sample(rows, k)` は、**異なる**行を `k` 個ランダムに選ぶ。シード付きなので、毎回同じ例になる。`min(a.examples, len(rows))` は、あるより多くを求めないため。
- 各行は4つの列に展開され(`_` は意味のラベルを無視)、`[T05 P012] 質問の文` の形で表示される — 動画にちょうどよい。

### 6.7 ログと終了

```python
    if not a.no_log:
        out_rel = str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/")
        append_run_log(a.run_log, a.machine, stats, a.per_pattern, a.seed, seconds, csv_sha, out_rel, zip_name)
        print(f"\nAppended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- `--no-log` でなければ:リポジトリからの相対パスを `/` で書き(Day 4・5と同じ技)、`run_log.md` に追記する。0を返す。終わり方は他のスクリプトと同じ。

---

## 7. 実行の流れ(順番)

1. オプション → 絶対パス → バナー。
2. `check_inputs` → 新しい時刻付きフォルダー → `copy_inputs` → 入力の指紋。
3. `run_generator`(`cwd` = 新しいフォルダー)。
4. `check_questions` → 指紋 → zip。
5. `generation_info.json` → まとめ → ランダムな例10個 → 実行ログ。

**そのあと出力フォルダーにあるもの:**入力のコピー、生成プログラムのファイル(`patterns.csv`、`question_types.csv`、`columns.csv`、`samples.csv`、`questions.csv`)、`questions_corpus_<machine>_<時刻>.zip`、`generation_info.json`。

## 8. 結果の読み方

- **PASS** = 重複なし、埋まっていないスロット・壊れたつながりなし、質問タイプの数が期待どおり。
- **`questions.csv SHA-256`** — コーパスの指紋。別のPC(同じ入力、同じシード、同じ上限)が**同じ**SHA-256を作れば、生成は再現できる(Day 4が比べるのはこれ)。
- **入力のSHA-256(最初の12文字)** — チームのファイルのどの版を使ったかの証明。
- **例の質問** — 文が自然に見えるかの、人の目による簡単な確認。

## 9. 知っておくと良い小さなこと

- **`data/source/` や生成したファイルを、公開された学校のリポジトリに、コミットしないこと**(docstringの警告)。
- **Day 4より弱い検査。**`check_questions` は、すべての行が4列か、各ラベルが文型・タイプと合うかを確認しない。生成プログラムはそういう行を作らないが、もしあれば、4列に展開する例の表示が、Pythonのエラーで落ちる。
- **「PASS (…60 question types)」は固定の文。**本当に期待する数は `--expected-types` から来る。
- **`relative_to(REPO_ROOT)`** は、`--out-root` をリポジトリの外にするとエラーになる(Day 4・5と同じ注意)。
- **同じ入力 + 同じシード + 同じ上限 = 同じ出力。**別のマシンでも、同じ `questions.csv` のSHA-256になるはず(生成プログラムは、シード付きの乱数と並べ替えたリストを使う。Day 3.1を参照)。

## 10. 一段落でまとめ

このスクリプトは、チームの入力を新しい時刻付きフォルダーにコピーして指紋を取り、そこでチームの生成プログラムを固定した設定(文型ごとに2,000問、シード42)で動かし、結果の重複、壊れた文、質問タイプの数を検査し、別のPCに持っていくzipと、すべての指紋を入れたJSONのレシートを保存し、ランダムな例の質問10個付きのまとめを表示し、`run_log.md` にメモを追記します。
