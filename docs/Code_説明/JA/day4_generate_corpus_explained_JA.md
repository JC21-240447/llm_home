# day4_generate_corpus.py — やさしい1行ずつの解説(日本語)

## 0. 全体像(まずここから)

このスクリプトは、**工場の品質管理担当者**だと考えてください。

工場の機械はすでにあります。それがチームの `make_questions.py` です。この機械は、文型と単語リストから、日本語の質問を何千個も作ります。このスクリプトは質問を**自分では作りません**。**機械を安全に動かし、製品を検査し、チームの見本と比べて、検査報告書を書く**のが仕事です。

| 工場の仕事 | このスクリプトがやること |
|---|---|
| 倉庫に材料がそろっているか確認 | `check_inputs` — 入力ファイルとライブラリがあるか |
| きれいな作業台に材料を移す(倉庫には触らない) | `prepare_work_dir` — 時刻付きの新しいフォルダーにコピー |
| 設定を固定して機械を動かす | `run_generator` — 同じシード+同じ上限=毎回同じ出力 |
| 製品を1つずつ検査 | `check_corpus` — 重複なし、壊れた文なし、ラベルOK、60タイプ |
| 見本と比べる | `compare_with_team` / `compare_summary_files` |
| 検査報告書を書く | `write_report` + `append_run_log` |

実行方法:

```
.venv\Scripts\python.exe scripts\day4_generate_corpus.py --machine home
```

**なぜ作業フォルダーを使うのか?** チームの生成プログラムは、実行したフォルダーに出力ファイルを書きます。`data/source` の中で動かすと、チームのファイルを上書きしてしまいます。そこで、先に全部を新しいフォルダーにコピーして、そこで動かします。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **コーパス** | 大量の文章の集まり。ここでは、生成された何千もの質問。 |
| **文型(パターン)** | 空欄のある文のひな形。例:「{社員}の{給与}を教えて」。生成プログラムが用語集の言葉で空欄を埋める。 |
| **質問タイプ** | 同じ種類のことを聞く文型のグループ(このプロジェクトでは60タイプを想定)。 |
| **シード** | 乱数の出発点になる数。同じシードなら、毎回同じ「ランダム」な選び方になる。結果を**再現できる**。 |
| **文型ごとの上限** | 1つの文型から残す質問の最大数(2000)。組み合わせが何百万もある文型がコーパスを占領しないようにする(*バランス調整*)。 |
| **SHA-256ハッシュ** | ファイルの指紋。バイト列から計算した長い文字列。1バイト変わるだけで全く別の指紋になる。どの入力ファイルを使ったかの証明に使う。 |
| **byte-identical(バイト単位で同一)** | 2つのファイルが1バイトも違わないこと。最も厳しい「同じ」。 |
| **CSV** | 表をテキストで表したもの。1行が1レコード、列はカンマ区切り。 |
| **BOM / `utf-8-sig`** | Excelなどが、UTF-8ファイルの先頭に付ける、見えない小さな印。`utf-8-sig` は、その印を取り除いて読む。 |
| **ZIP** | 1つ以上のファイルを圧縮してまとめたもの。 |
| **`dataclass`** | データを入れるだけのクラスを、短く作る方法。 |
| **終了コード** | プログラムがOSに返す数。0 = 正常、0以外 = 問題あり。 |
| **`SystemExit`** | Pythonがわざとプログラムを止めるための例外(メッセージ付きにできる)。 |
| **cwd** | 「カレントワーキングディレクトリ」:プログラムが今いると思っているフォルダー。 |

---

## 2. ファイル先頭のdocstring(説明文)

```python
r"""Day 4 - generate the question corpus ...
...
"""
```
- Day 1と同じ考え方:raw(`r`)の三重引用符の説明文で、英語の部分と日本語の部分があり、実行はされない。
- スクリプトの6つの手順(確認、コピー、実行、検査、比較、報告)、使い方のコマンド、そして注意書きを載せている:学校PCでは `--machine school` を使い、そのPCにも `pandas` と `openpyxl` を入れておくこと。
- 「CPU only」— Day 1と違い、ここではGPUを使わない。

---

## 3. import(道具を持ってくる)

```python
from __future__ import annotations
```
- Day 1と同じ(新しい型ヒントの書き方がどこでも使える)。

```python
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
```

`argparse`、`json`、`os`、`shutil`、`subprocess`、`sys`、`time`、`datetime`、`Path` は **Day 1と同じ**(コマンドライン指定、JSON、ファイルコピー、他のプログラムの実行、ストップウォッチ、日時、パス)。新しいものは次のとおり:

| 行 | 何が使えるか |
|---|---|
| `import csv` | CSVを正しく読む(文字列の中のカンマや引用符も扱える)。 |
| `import hashlib` | 指紋(SHA-256)を計算する。 |
| `import importlib.util` | ライブラリを読み込まずに「入っているか?」を調べる。 |
| `import io` | メモリ上の文字列を、ファイルのように扱う(`StringIO`)。 |
| `import zipfile` | `.zip` を開く。 |
| `from collections import Counter, defaultdict` | `Counter` は数を数える。`defaultdict` は、無いキーを自動で作る辞書。 |
| `from dataclasses import dataclass` | データ用クラスの近道。 |

---

## 4. 定数

```python
SCRIPT_VERSION: str = "day4-v1"
```
- レポートに書き込むバージョン名。どの版のスクリプトが作ったか分かる。

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
```
- Day 1と同じ:このスクリプトの場所から求めた、リポジトリのルート。

```python
GENERATOR: str = "make_questions.py"
REQUIRED_INPUTS: tuple[str, ...] = (GENERATOR, "用語集.csv", "同義語辞書.csv", "DB.xlsx")
```
- `GENERATOR` — チームの生成プログラムのファイル名。
- `REQUIRED_INPUTS` — 必ず必要な4つのファイルの**タプル**(変更できないリスト):生成プログラム、用語集、同義語辞書、サンプル表 `DB.xlsx`。`tuple[str, ...]` は「文字列がいくつでも」の意味。

```python
SUMMARY_FILES: tuple[str, ...] = ("patterns.csv", "question_types.csv", "columns.csv")
```
- 生成プログラムが作る3つの要約ファイル。後で、チームのコピーと比べる。

```python
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
```
- `questions.csv` の1行目(列の名前)がこれと**ぴったり**同じでなければならない:質問、パターンID、質問タイプID、意味(ラベル)。

```python
BROKEN_MARKS: tuple[str, ...] = ("{", "}", "のの", "社員の社員", "平均平均")
```
- **壊れた文**のサイン:
  - `{` や `}` — 文型の空欄が埋まっていない。
  - `のの`、`社員の社員`、`平均平均` — 言葉が間違ってつながっている(「の」の重なりなど)。

---

## 5. データクラス

```python
@dataclass
class GeneratedCorpus:
    """The generator's questions.csv, parsed."""

    raw_bytes: bytes
    rows: list[list[str]]  # 質問, パターンID, 質問タイプID, 意味
```
- `@dataclass` は**デコレーター**で、クラスの面倒な部分(値を保存する `__init__`)を自動で書いてくれる。
- つまり `GeneratedCorpus(raw, rows)` は2つのものを入れる箱になる:
  - `raw_bytes` — ファイルの元のバイト列(後でバイト単位の比較や指紋の計算に使うので残す)。
  - `rows` — 表を行のリストにしたもの。各行は4つの文字列のリスト。

---

## 6. ヘルパー関数

### 6.1 指紋

```python
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())
```
- `sha256_bytes` — バイト列のSHA-256を計算し、16進数の文字列(英数字)で返す。
- `sha256_file` — ファイル全体をバイトで読み(`read_bytes`)、`sha256_bytes` に渡す。(同じ仕事のファイル版。)

### 6.2 `check_inputs`

```python
def check_inputs(source_dir: Path) -> list[Path]:
    """Stop with a clear message if an input file or library is missing. Returns the src_*.txt files."""
    missing = [n for n in REQUIRED_INPUTS if not (source_dir / n).exists()]
    src_files = sorted(source_dir.glob("src_*.txt"))
    if not src_files:
        missing.append("src_*.txt")
```
- `missing = [...]` — **リスト内包表記**:必要な4つの名前を順に見て、`source_dir` に**無い**ものだけを集める。
- `source_dir.glob("src_*.txt")` — 名前が `src_` で始まり `.txt` で終わるファイル(文型ファイル)を全部探す。`sorted(...)` で順番を固定し、結果が再現できるようにする。
- 1つも無ければ、`"src_*.txt"` を不足リストに追加する。

```python
    if missing:
        raise SystemExit(
            f"Missing in {source_dir}: {', '.join(missing)}\n"
            "Put the team's files there first (see Step 1 of the Day 4 walkthrough)."
        )
```
- 何か足りなければ、足りないものを示すメッセージを出してプログラム全体を止める。`', '.join(missing)` はリストを `a, b, c` の形にする。

```python
    libs = [m for m in ("pandas", "openpyxl") if importlib.util.find_spec(m) is None]
    if libs:
        raise SystemExit(
            f"Missing Python libraries: {', '.join(libs)}\n"
            r"Install them: .venv\Scripts\python.exe -m pip install -r requirements.txt"
        )
    return src_files
```
- `find_spec(m)` は「ライブラリ `m` は見つかる?」とPythonに聞く。無ければ `None`。つまり `libs` = 入っていないライブラリ。
- `pandas` と `openpyxl` は、このスクリプトではなく**生成プログラム**(`DB.xlsx` を読むため)が必要とする。早めに分かりやすく失敗させるため、今ここで確認する。
- `r"..."` のraw文字列は、Windowsのパスのバックスラッシュをそのまま残す。
- 最後に、文型ファイルのリストを返す。

### 6.3 `prepare_work_dir`

```python
def prepare_work_dir(source_dir: Path, work_root: Path, src_files: list[Path]) -> Path:
    """Copy the generator and its inputs into a fresh folder, so data/source is never written to."""
    work = work_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    work.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_INPUTS:
        shutil.copy2(source_dir / name, work / name)
    for f in src_files:
        shutil.copy2(f, work / f.name)
    return work
```
- `work = work_root / <時刻>` — `20261106_143012` のような(日付+秒までの時刻の)新しいフォルダー名。実行ごとに別のフォルダーになる。
- `work.mkdir(parents=True, exist_ok=False)` — フォルダーを作る。`exist_ok=False` は**わざと**で、もしフォルダーが既にあれば、再利用せず**エラーにする**。「新しい」を本当に新しくするため。
- 1つ目のループ — 必須の4ファイルを作業フォルダーにコピー。`shutil.copy2` はファイルと一緒に更新日時などの情報(メタデータ)もコピーする。
- 2つ目のループ — `src_*.txt` の文型ファイルを全部コピー。`f.name` はフォルダーを除いたファイル名。
- 作業フォルダーのパスを返す。

### 6.4 `run_generator`

```python
def run_generator(work: Path, per_pattern: int, seed: int) -> tuple[str, float]:
    """Run the team's generator in the work folder with this venv's Python. Returns (console output, seconds)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
```
- `os.environ` は環境変数の集まり。`dict(os.environ, A=..., B=...)` は、それを**コピー**して2つの設定を追加・上書きする。これで子プログラムのPythonが全面的にUTF-8を使い、Windowsでも日本語が壊れない。

```python
    cmd = [sys.executable, GENERATOR, "--corpus", str(per_pattern), "--seed", str(seed)]
```
- コマンドをリストで作る:`python make_questions.py --corpus 2000 --seed 42`。
- `sys.executable` = このスクリプトを動かしているPython(venvのPython)そのもの。生成プログラムも同じライブラリを使える。
- `str(...)` は数を文字列にする。コマンドライン引数は文字列でなければならないため。

```python
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - start
```
- `perf_counter()` — 高精度のストップウォッチ。
- `subprocess.run(...)` — コマンドを実行する:
  - `cwd=work` — 作業フォルダー**の中で**実行する(だから出力ファイルがそこに作られる)。
  - `env=env` — 上のUTF-8設定を使う。
  - `capture_output=True`、`text=True` — Day 1と同じ:出力を文字列として受け取る。
  - `encoding="utf-8", errors="replace"` — 出力をUTF-8として読み、読めないバイトは落ちずに `?` にする。
- `seconds` = 終了時刻 − 開始時刻。

```python
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"The generator stopped with exit code {proc.returncode}. See the error above.")
    return proc.stdout, seconds
```
- 生成プログラムが失敗(終了コードが0以外)したら、それが出した内容(通常出力とエラー出力)を全部表示して理由が分かるようにし、止める。
- 成功なら、コンソール出力と所要時間を返す。

### 6.5 `read_corpus_bytes`

```python
def read_corpus_bytes(path: Path) -> bytes:
    """questions.csv bytes, from a .csv or from the questions.csv inside a .zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith("questions.csv")]
            if len(members) != 1:
                raise SystemExit(f"{path}: expected one questions.csv inside, found {members}")
            return zf.read(members[0])
    return path.read_bytes()
```
- `path.suffix.lower() == ".zip"` — ファイルはZIPか?(大文字小文字は無視)
- `zipfile.ZipFile(path)` で開く(`with` が終わりに閉じる)。
- `zf.namelist()` で中のファイル名の一覧を得て、`questions.csv` で終わるものを残す。
- **ちょうど1つ**でなければならない。0個や複数なら、見つかったものを示して止める。
- `zf.read(members[0])` — ディスクに展開せずに、そのファイルのバイト列を読む。
- ZIPでなければ、ファイルのバイト列をそのまま読む。

### 6.6 `parse_corpus`

```python
def parse_corpus(raw: bytes) -> GeneratedCorpus:
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader)
    if header != QUESTION_HEADER:
        raise SystemExit(f"Unexpected header {header}")
    return GeneratedCorpus(raw, [row for row in reader])
```
- `raw.decode("utf-8-sig")` — バイト列を文字列にし、BOMがあれば取り除く。
- `io.StringIO(...)` — 文字列を、開いたファイルのように使えるようにする。
- `csv.reader(...)` — 1行ずつ、文字列のリストとして読む。
- `header = next(reader)` — **最初の**行(列名)を取る。
- ヘッダーが `QUESTION_HEADER` とぴったり同じでなければ、想定した形式ではないので止める。
- `[row for row in reader]` — 残りの行をリストに集める。
- 元のバイト列と解析した行の両方を入れたデータクラスを返す。

---

## 7. 品質チェック:`check_corpus`

```python
def check_corpus(corpus: GeneratedCorpus, expected_types: int) -> dict[str, object]:
    """Quality checks on the generated questions. Raises SystemExit if any check fails."""
    texts = [r[0] for r in corpus.rows]
    problems: list[str] = []
```
- `texts` — 質問文だけ(全行の0列目)。
- `problems` — 見つけた問題の説明を集める空のリスト。(先に全部集めて、まとめて報告する。)

```python
    if any(len(r) != 4 for r in corpus.rows):
        problems.append("a row does not have 4 columns")
```
- **チェック1:形。** `any(...)` =「ちょうど4列でない行が1つでもあるか?」

```python
    if len(set(texts)) != len(texts):
        problems.append(f"{len(texts) - len(set(texts))} duplicate questions")
```
- **チェック2:重複。** `set`(集合)は重複を取り除く。集合がリストより短ければ、繰り返しがある。その差が重複の数。

```python
    broken = [t for t in texts if any(m in t for m in BROKEN_MARKS)]
    if broken:
        problems.append(f"{len(broken)} broken sentences, e.g. {broken[0]}")
```
- **チェック3:壊れた文。** 悪いサインを**どれか1つでも**含む質問を集める。あれば、件数と例を1つ(`broken[0]`)報告する。

```python
    bad_labels = 0
    for text, pid, tid, meaning in corpus.rows:
        try:
            label = json.loads(meaning)
            if label.get("パターン") != pid or label.get("質問タイプ") != tid:
                bad_labels += 1
        except json.JSONDecodeError:
            bad_labels += 1
    if bad_labels:
        problems.append(f"{bad_labels} labels unreadable or not matching their row")
```
- **チェック4:ラベル。** 各行の4列目(`meaning`)は、小さなJSONテキスト。各行について:
  - `for text, pid, tid, meaning in ...` — 4つの列を4つの名前に分けて受け取る。
  - `json.loads(meaning)` — JSONテキストを辞書にする。
  - ラベルの**中**にあるパターンとタイプを、その行の列にあるパターンID・タイプIDと比べる。違えば不良として数える。
  - JSON自体が壊れていれば(`JSONDecodeError`)、それも不良。
- 0より大きければ、件数を報告する。

```python
    types = {r[2] for r in corpus.rows}
    if len(types) != expected_types:
        problems.append(f"{len(types)} question types, expected {expected_types}")
```
- **チェック5:質問タイプ。** ループ付きの `{ ... }` は**集合内包表記**:タイプIDの重複なしの集まり。ちょうど `expected_types`(60)個でなければならない。

```python
    if problems:
        raise SystemExit("Corpus check FAILED: " + "; ".join(problems))
```
- どれか1つでも問題があれば、プログラムを止め、全部の問題を `; ` でつないで表示する。

```python
    lengths = [len(t) for t in texts]
    return {
        "questions": len(texts),
        "patterns_with_questions": len({r[1] for r in corpus.rows}),
        "types": len(types),
        "characters": sum(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
```
- すべて合格なら、統計を計算する。`lengths` = 各質問の文字数。
- 辞書で返す:質問の総数、質問が出ている文型の数、タイプ数、総文字数、平均・最短・最長の文字数。

---

## 8. チームとの比較

### 8.1 `compare_with_team`

```python
def compare_with_team(new: GeneratedCorpus, team_raw: bytes | None) -> dict[str, object]:
    """Byte comparison plus a per-pattern comparison of the question sets."""
    if team_raw is None:
        return {"available": False}
    team = parse_corpus(team_raw)
```
- チームのコーパスが見つからなければ `team_raw` は `None`。その場合は「比較できない」と報告するだけ。
- あれば、自分たちのものと同じ方法で、チームのコーパスを解析する。

```python
    def by_pattern(c: GeneratedCorpus) -> dict[str, set[str]]:
        d: dict[str, set[str]] = defaultdict(set)
        for text, pid, _, _ in c.rows:
            d[pid].add(text)
        return d
```
- **関数の中の関数**(ここでしか使わない)。
- `defaultdict(set)` — 無いキーが自動で空の集合から始まる辞書。
- 各行について、質問文をそのパターンIDの集合に追加する。結果は `{パターンID: {そのパターンの全質問}}`。
- `_` は「この列は要らない」の意味。

```python
    a, b = by_pattern(new), by_pattern(team)
    all_pids = sorted(set(a) | set(b))
    differing = [p for p in all_pids if a.get(p) != b.get(p)]
```
- `a` = 自分たちの分、`b` = チームの分。
- `set(a) | set(b)` — 両方のパターンIDの**和集合**(片方にしか無いパターンも確認できる)。`sorted` で順番を固定。
- `differing` — 自分たちの質問の集合が、チームのものと**等しくない**パターン。集合なので順番は関係なく、中身だけを比べる。

```python
    return {
        "available": True,
        "byte_identical": new.raw_bytes == team_raw,
        "team_questions": len(team.rows),
        "patterns_identical": len(all_pids) - len(differing),
        "patterns_differing": differing,
    }
```
- `byte_identical` — 2つのファイルは1バイトも違わないか?(最も厳しいテスト)
- `team_questions` — チームのコーパスの質問数。
- `patterns_identical` — 全パターン数 − 違うパターン数。
- `patterns_differing` — 違うパターンIDのリスト。

### 8.2 `compare_summary_files`

```python
def compare_summary_files(work: Path, source_dir: Path) -> dict[str, str]:
    """Is each regenerated summary file identical to the team's copy in data/source?"""
    out: dict[str, str] = {}
    for name in SUMMARY_FILES:
        team = source_dir / name
        if not team.exists():
            out[name] = "no team copy"
        else:
            out[name] = "identical" if (work / name).read_bytes() == team.read_bytes() else "DIFFERENT"
    return out
```
- `patterns.csv`、`question_types.csv`、`columns.csv` それぞれについて:
  - チームのコピーが無ければ `"no team copy"`。
  - あれば、作業フォルダーで新しく作られたファイルとチームのファイルを、バイト単位で比べ、`"identical"` か `"DIFFERENT"`。

---

## 9. 報告書

### 9.1 `load_pattern_table`

```python
def load_pattern_table(work: Path) -> list[dict[str, str]]:
    with (work / "patterns.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
```
- 生成された `patterns.csv` を開く(BOMがあれば取り除く)。
- `newline=""` は `csv` モジュールが求める指定で、引用符の中の改行を正しく扱える。
- `csv.DictReader` は各行を、列名をキーにした辞書(例:`p["パターンID"]`)にする。`list(...)` で全部を集める。

### 9.2 `write_report` — データを準備する部分

```python
def write_report(path, stats, cmp_team, cmp_files, patterns, corpus, inputs, per_pattern, seed, seconds, generator_out) -> None:
```
- ここまでに分かったことを全部受け取り、Markdownファイルを1つ書く。(実際のコードでは、読みやすいよう引数を1行に1つずつ書いてある。)

```python
    gen_count = Counter(r[1] for r in corpus.rows)
```
- `Counter` は、各パターンIDの生成された質問が何個あるかを数える。

```python
    type_rows: dict[str, list] = {}
    for p in patterns:
        t = type_rows.setdefault(p["質問タイプID"], [p["質問タイプ"], 0, 0, 0])
        t[1] += 1
        t[2] += int(p["作れる質問の数"])
        t[3] += gen_count.get(p["パターンID"], 0)
```
- **質問タイプごと**の要約を作る。各項目はリスト:`[タイプ名, 文型の数, 作れる質問の数, 実際に書いた質問の数]`。
- `setdefault(キー, 初期値)` —「このキーの項目をちょうだい。まだ無ければ、この初期値で作って」。初期値はタイプ名と3つの0。
- `t[1] += 1` — このタイプに文型がもう1つ属する。
- `t[2] += int(...)` — この文型が**作れる**質問の数を足す(CSVでは数も文字列なので `int()` を使う)。
- `t[3] += gen_count.get(..., 0)` — この文型で**実際に書かれた**質問の数を足す(無ければ0)。

```python
    first_question: dict[str, str] = {}
    for text, _, tid, _ in corpus.rows:
        first_question.setdefault(tid, text)
```
- タイプごとに、最初に出会った質問を覚えておく(`setdefault` はまだ何も無いときだけ保存する)。「例」の列に使う。

```python
    styles = Counter(p["文体"] for p in patterns)
    empty = [p["パターンID"] for p in patterns if gen_count.get(p["パターンID"], 0) == 0]
    possible = sum(int(p["作れる質問の数"]) for p in patterns)
```
- `styles` — 文体(例:丁寧、くだけた)ごとの文型の数。
- `empty` — 質問が**0個**になった文型(例:全部が重複として除かれた)。
- `possible` — 全文型が合計で作れる、重複のない質問の総数。

### 9.3 `write_report` — Markdownを組み立てる部分

```python
    lines = [
        "# Day 4 report: question generation",
        "",
        f"Generated by `scripts/day4_generate_corpus.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}.",
        ...
```
- `lines` は、報告書になる文字列の行のリスト。`""` は空行。
- `{datetime.now():%Y-%m-%d %H:%M}` — f文字列の中で、そのまま日付を整形している(`:` の後ろが書式)。
- `` `...` `` のバッククォートは、Markdownのコード風表示。

```python
        "| File | SHA-256 |",
        "| --- | --- |",
        *(f"| `{n}` | `{h}` |" for n, h in inputs.items()),
```
- 最初の2行は、Markdown表のヘッダー。
- `*( ... )` はジェネレーターを**展開**してリストに入れる:入力ファイル1つにつき、指紋付きの表の1行。これが「どの入力ファイルでこのコーパスができたか」の証明になる。

```python
        f"- At most {per_pattern:,} questions per pattern (balancing), seed {seed}.",
        f"- Generation time: {seconds:.1f} seconds.",
```
- `{per_pattern:,}` — 3桁区切りの数(`2,000`)。`{seconds:.1f}` — 小数第1位まで。

```python
        f"| Patterns | {len(patterns)} ({', '.join(f'{k} {v}' for k, v in styles.items())}) |",
        f"| Question types | {stats['types']} |",
        f"| All distinct questions the patterns can make | {possible:,} |",
        f"| Questions written (after the per-pattern limit and duplicate removal) | {stats['questions']:,} |",
        f"| Patterns with at least one question | {stats['patterns_with_questions']} |",
        f"| Patterns with no question left after duplicate removal | {', '.join(empty) or 'none'} |",
        f"| Characters | {stats['characters']:,} |",
        f"| Length per question (min / average / max) | {stats['min_chars']} / {stats['avg_chars']} / {stats['max_chars']} |",
```
- **Result(結果)**の表の行。各行は「ラベル | 値」。
- 1行目:文型の数と、ネストしたf文字列(中の引用符の種類が違う点に注意)で作った `polite 40, casual 20` のような一覧。
- 他の行は `stats` から数を取り、`:,` で整形する。
- `', '.join(empty) or 'none'` — 質問が0個の文型IDをつなぐ。結果が空文字なら `none` を表示する(`or` は、最初の値が空のとき2つ目を選ぶ)。

```python
        "Checks passed: no duplicates, no unfilled slots or broken joins, every row has a readable label",
        "that matches its pattern and type.",
```
- 固定の文。失敗していればもっと前でプログラムが止まるので、常に「passed」と書いても安全。

```python
    if cmp_team.get("available"):
        lines += [
            f"- Compared with the team's corpus (`questions_corpus.zip`, {cmp_team['team_questions']:,} questions):",
            f"  byte-identical: **{'yes' if cmp_team['byte_identical'] else 'no'}**;"
            f" identical patterns: {cmp_team['patterns_identical']};"
            f" differing patterns: {', '.join(cmp_team['patterns_differing']) or 'none'}.",
        ]
    else:
        lines.append("- The team's corpus was not found, so no comparison was made.")
```
- チームのコーパスがあれば、比較結果の2行を追加(`**...**` はMarkdownの太字で、分けて書かれた文字列は自動で1行につながる)。無ければ、その旨を1行追加。

```python
    lines += [f"- `{n}` compared with the team's copy: {v}." for n, v in cmp_files.items()]
```
- 要約ファイルごとに1行:`identical`、`DIFFERENT`、`no team copy` のどれか。

```python
        *(
            f"| {tid} | {v[0]} | {v[1]} | {v[2]:,} | {v[3]:,} | {first_question.get(tid, '')} |"
            for tid, v in sorted(type_rows.items())
        ),
```
- **タイプごとの表**の行(その表のヘッダーの後):タイプID、名前、文型の数、作れる数、書いた数、例。`sorted(...)` でタイプID順に並べる。

```python
        "```text",
        generator_out.strip(),
        "```",
```
- 生成プログラム自身のコンソール出力を、Markdownのコードブロックに入れる。

```python
    path.write_text("\n".join(lines), encoding="utf-8")
```
- 全部の行を改行でつなぎ、UTF-8で保存する。

### 9.4 `append_run_log`

```python
def append_run_log(log: Path, machine: str, stats, cmp_team, per_pattern: int, seed: int, seconds: float, work_rel: str) -> None:
    same = "n/a" if not cmp_team.get("available") else ("yes" if cmp_team["byte_identical"] else
                                                        f"no ({len(cmp_team['patterns_differing'])} patterns differ)")
```
- 短い文字列 `same` を作る:
  - チームのコーパスが無い → `n/a`
  - バイト単位で同一 → `yes`
  - それ以外 → `no (N patterns differ)`。
- これは**入れ子の条件式**(if/else の中にもう一つ if/else)。

```python
    entry = [
        "",
        f"## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 4 question generation",
        "",
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Generator: data/source/make_questions.py (team), run by scripts/day4_generate_corpus.py ({SCRIPT_VERSION})",
        f"- Settings: max {per_pattern} per pattern, seed {seed}; time {seconds:.1f} s",
        f"- Result: {stats['questions']:,} questions, {stats['types']} types, "
        f"{stats['patterns_with_questions']} patterns with questions, avg {stats['avg_chars']} chars",
        f"- Identical to the team's corpus: {same}",
        f"- Output folder: {work_rel}",
        "- Checks: PASS",
        "",
    ]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))
```
- 短いMarkdownのログ1件分を作る(Day 1の `append_run_log` と同じ作り)。
- `"- Checks: PASS"` は常に書かれる。失敗していればプログラムがすでに止まっているため。
- `run_log.md` を追記モード `"a"` で開き、末尾に追加する。

---

## 10. メイン部分

### 10.1 `parse_args`

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 4: run the team's question generator, check and report.")
    ap.add_argument("--machine", choices=["school", "home"], required=True)
    ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "data/source")
    ap.add_argument("--work-root", type=Path, default=REPO_ROOT / "runs/day4_generation")
    ap.add_argument("--team-corpus", type=Path, default=REPO_ROOT / "data/source/questions_corpus.zip")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY4_GENERATION_REPORT.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--per-pattern", type=int, default=2000, help="max questions per pattern (team setting: 2000)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (team setting: 42)")
    ap.add_argument("--expected-types", type=int, default=60)
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()
```
- Day 1と同じ `argparse` の考え方。`ap` はパーサーの短い名前。
- `--machine` — 必須で、`school` か `home`(Day 1と同じ)。
- `--source-dir`、`--work-root`、`--team-corpus`、`--report`、`--run-log` — 5つの**パス**のオプション。`type=Path` で、入力した文字を `Path` に変える。すべて初期値があるので、普通は何も入力しない。(この5行は全部同じ形で、名前と初期の場所だけが違う。)
- `--per-pattern` — 整数(`type=int`)、初期値は2000(チームの設定)。`help=` は `--help` で表示される説明。
- `--seed` — 初期値は42(チームの設定)。
- `--expected-types` — チェックで想定する質問タイプの数(60)。
- `--no-log` — フラグ(Day 1と同じ `store_true`):付ければ `run_log.md` に**追記しない**。
- `ap.parse_args()` が入力を読み取って返す。

### 10.2 `main` — 前半

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
```
- 出力ストリームが `reconfigure` に対応していれば、表示できない文字を落ちずに置き換えるよう設定する。(Day 1の `setup_utf8_console` の軽い版で、ここでは `errors` だけ設定。)

```python
    a = parse_args()
    for name in ("source_dir", "work_root", "team_corpus", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
```
- `a` に全部のオプションが入っている(`a.machine`、`a.seed` など)。
- ループは、5つのパスのオプションを `.resolve()` で**完全な絶対パス**にする。`getattr(a, name)` は「名前が文字列で書かれた属性を読む」、`setattr(a, name, 値)` は「それを書き戻す」。同じ行を5回書かずに済む。

```python
    print(f"=== Day 4 question generation | machine={a.machine} | max {a.per_pattern}/pattern | seed {a.seed} ===")
    src_files = check_inputs(a.source_dir)
    work = prepare_work_dir(a.source_dir, a.work_root, src_files)
    print(f"Work folder: {work}")
```
- 見出し → 手順1(入力の確認)→ 手順2(新しい作業フォルダーにコピー)→ 作業フォルダーの場所を表示。

```python
    generator_out, seconds = run_generator(work, a.per_pattern, a.seed)
    print(generator_out.strip())
    print(f"Generation took {seconds:.1f} s")
```
- 手順3:チームの生成プログラムを実行。出力と時間を受け取り、両方を表示する。

```python
    corpus = parse_corpus((work / "questions.csv").read_bytes())
    stats = check_corpus(corpus, a.expected_types)
    team_raw = read_corpus_bytes(a.team_corpus) if a.team_corpus.exists() else None
    cmp_team = compare_with_team(corpus, team_raw)
    cmp_files = compare_summary_files(work, a.source_dir)
```
- 手順4:生成された `questions.csv` を読んで解析 → 5つの品質チェックを実行。
- 手順5:チームのコーパスは、**あるときだけ**読み(無ければ `None`)→ 質問を比較 → 要約ファイルを比較。

```python
    inputs = {f.name: sha256_file(f) for f in [*(a.source_dir / n for n in REQUIRED_INPUTS), *src_files]}
```
- **辞書内包表記**:ファイル名 → 指紋。
- ループ対象のリストは、`*` で2つをつないで作る:必須の4ファイル(`a.source_dir / n` で作った完全パス)と、`src_*.txt` のファイル。

```python
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(a.report, stats, cmp_team, cmp_files, load_pattern_table(work), corpus, inputs,
                 a.per_pattern, a.seed, seconds, generator_out)
```
- 報告書のフォルダー(`docs/`)があることを確認してから、報告書を書く。`load_pattern_table(work)` が文型の表を渡す。

```python
    (work / "check_result.json").write_text(
        json.dumps({"stats": stats, "compare_team": cmp_team, "compare_files": cmp_files,
                    "questions_csv_sha256": sha256_bytes(corpus.raw_bytes)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
```
- 統計、比較結果、生成された `questions.csv` の指紋を入れた、機械が読める `check_result.json` を作業フォルダーに保存する。`ensure_ascii=False` は日本語をそのまま読める形で残し(Day 1と同じ)、`indent=2` で見やすくする。

### 10.3 `main` — 表示と終了

```python
    print()
    print(f"Questions: {stats['questions']:,} | types: {stats['types']} | "
          f"patterns with questions: {stats['patterns_with_questions']} | avg chars: {stats['avg_chars']}")
    print("Corpus check: PASS (no duplicates, no broken sentences, labels readable)")
```
- 空行、そして数字を1行にまとめた要約と、PASSのメッセージ(ここに来るのは全チェックに合格したときだけ)。

```python
    if cmp_team.get("available"):
        print(f"Same as the team's corpus: byte-identical={cmp_team['byte_identical']}, "
              f"identical patterns={cmp_team['patterns_identical']}, "
              f"differing patterns={cmp_team['patterns_differing'] or 'none'}")
    for n, v in cmp_files.items():
        print(f"{n}: {v}")
    print(f"Report: {a.report}")
```
- (チームのコーパスがあれば)比較結果、要約ファイルごとに1行、そして報告書の保存先を表示する。

```python
    if not a.no_log:
        work_rel = str(work.relative_to(REPO_ROOT)).replace("\\", "/")
        append_run_log(a.run_log, a.machine, stats, cmp_team, a.per_pattern, a.seed, seconds, work_rel)
        print(f"Appended an entry to {a.run_log.name}")
    return 0
```
- `--no-log` が無い限り:作業フォルダーのパスを**リポジトリからの相対パス**(`relative_to`)にし、Windowsの `\` を `/` に変えて、どのマシンでも同じ見た目のログにする。ログを追記し、確認を表示する。
- `return 0` — 成功。

```python
if __name__ == "__main__":
    sys.exit(main())
```
- Day 1(`SystemExit(main())`)と同じ考え方:ファイルを直接実行したときだけ `main()` を動かし、その戻り値を終了コードにする。`sys.exit(...)` と `raise SystemExit(...)` は同じ働き。
- 前に出てきた `raise SystemExit("メッセージ")` もすべて、プログラムを止めてメッセージを表示し、終了コード1を返す。

---

## 11. 実行の流れ(順番)

1. オプションを読む → パスを絶対パスにする。
2. `check_inputs` — ファイルとライブラリはあるか?
3. `prepare_work_dir` — `runs/day4_generation/<時刻>/` にコピー。
4. `run_generator` — そこで `make_questions.py` を、シード42・上限2000で実行。
5. `parse_corpus` → `check_corpus` — 5つの品質チェック。
6. チームのコーパスと要約ファイルを比較。
7. 入力の指紋を計算 → `docs/DAY4_GENERATION_REPORT.md` と `check_result.json` を書く。
8. 要約を表示 → `run_log.md` に追記(`--no-log` の場合を除く)。

## 12. 結果の読み方

- **`byte-identical: yes`** — チームのコーパスを完全に再現できた。最高の結果。
- **`byte-identical: no` と違う文型のリスト** — どこかが違う。まずそのパターンIDを見る(入力ファイル、ライブラリのバージョン、シードや上限の違いなど)。
- **要約ファイルが `identical`** — 生成プログラムは、チームの `patterns.csv`、`question_types.csv`、`columns.csv` も再現できた。
- **「Patterns with no question left after duplicate removal」** — 質問が全部、他の文型の質問と重複していた文型。
- **Corpus check FAILED** — メッセージを読む。見つかった問題(重複、壊れた文、不良ラベル、タイプ数の違い)が全部並んでいる。

## 13. 知っておくと良い小さなこと(通常の使い方ではバグではありません)

- **失敗は必ず大きな音で。** すべての失敗は `SystemExit` を使うので、失敗した実行がログに「PASS」と書くことはない。
- **4列チェックの順番。** 4列でない行があると、後のラベルのループ(4つに分けて受け取る)が、親切なメッセージを出す前にPythonの `ValueError` で落ちる。チームの生成プログラムでは起こらないはずで、CSVが壊れたときだけの話。
- **`--work-root` をリポジトリの外にした場合。** 最後の `relative_to(REPO_ROOT)` がエラーになる。初期値のままなら問題ない。
- **同じシード ≠ 常に同じ出力。** 再現性は、同じ入力・同じシード・同じ上限に加えて、同じ生成プログラムのコードとライブラリの動作にも依存する。だから報告書に入力の指紋を残している。

## 14. 一段落でまとめ

このスクリプトは、入力をきれいにコピーした場所で、チームの質問生成プログラムを固定したシードで動かし、重複・壊れた文・読めないラベル・質問タイプの数が正しいかを検査し、チームのコーパスとバイト単位・文型単位で比較して、Markdownの報告書(すべての入力の指紋付き)、JSONの結果、`run_log.md` のメモを書きます。これで、生成が安全で、再現でき、証明できるものになります。
