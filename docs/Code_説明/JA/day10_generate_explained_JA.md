# Day 10 — day10_generate.py — やさしい1行ずつの解説(日本語)

> **Day 10 の3つのファイルの関係**
> - **Day 10.1 `generation.py`** — エンジン(**先に読んでください**)。
> - **Day 10(このファイル)`day10_generate.py`** — 3つのコマンド `candidates`、`generate`、`report` を持つ**コマンドラインの運転手**。**学校の作業計画の作業10**を完了する。
> - **Day 10.2 `test_generation.py`** — エンジンの自動テスト。

---

## 0. 全体像(まずここから)

Day 9の後には、学習したチェックポイント(`best.pt`、`last.pt`)があります。Day 10では、**モデルが何を学んだか**を、3つの方法で見られます:

| コマンド | することと使い方 | 典型的な使い方 |
|---|---|---|
| **`candidates`** | 書き出しに対する、**次のトークンの上位10個と、その確率**を見せる | 「`今日は` の後に、モデルは何が来ると思っている?」 |
| **`generate`** | プロンプトファイルの**各行を、1つの設定で続ける** | 手軽な実験 |
| **`report`** | 学習の `best.pt` **と** `last.pt` について、**Day 10の証拠一式**:4つの書き出しの候補、プロンプトファイルでの**4つの設定すべて**、自動の確認、繰り返しの上限のデモ | 提出する報告書 |

```
# 1. candidates
.venv\Scripts\python.exe scripts\day10_generate.py candidates --text 今日は --text 営業部の
# 2. generate
.venv\Scripts\python.exe scripts\day10_generate.py generate --setting sample
# 3. report
.venv\Scripts\python.exe scripts\day10_generate.py report --run smoke_a --machine home
```
4つの設定(Day 10.1より):`greedy`、`greedy+limits`、`sample`(温度0.8、top-k 50、top-p 0.95)、`sample-hot`(温度1.5。ランダムすぎるとどうなるかを見せる)。

生成は軽いので、`--device cpu` でもどのPCでも動きます。同じシード → 同じ文。(後日のLinuxサーバー:`.venv/bin/python scripts/day10_generate.py report --run smoke_a --machine server`。)

---

## 1. 先に知っておきたい用語

用語はすべてDay 10.1のもの(ロジット、貪欲、サンプリング、温度、top-k/top-p、`<eos>`、distinct-2)。Pythonの新しい考え方:
- **サブコマンド**(`argparse` の `add_subparsers`):1つのスクリプトに複数の「コマンド」があり、それぞれ専用のオプションを持つ(`git commit`、`git push` のように)。
- **`dataclasses.replace`**:データクラスのオブジェクトの、いくつかの項目を変えた**コピー**を作る。

---

## 2. docstring、import、定数

- 英語 + 日本語:3つのコマンド、4つの設定、サーバーの注意。(プロンプトファイル `prompts/day10_prompts.txt` は、`generate` と `report` の初期の入力で、リポジトリに存在している必要がある。)

```python
from __future__ import annotations

import argparse, csv, json, platform, sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.model import ModelA, ModelConfig  # noqa: E402
from src.generation import (PRESET_SETTINGS, Generation, SamplingConfig, distinct_2, generate, is_repetitive,
                            load_model, model_logits_fn, next_token_candidates, token_text)  # noqa: E402
from src.training import sha256_file  # noqa: E402

SCRIPT_VERSION: str = "day10-v2"
CANDIDATE_PROMPTS: tuple[str, ...] = ("今日は", "営業部の", "ボーナスが", "大阪支社で働く社員の")
```
- いつもの準備(前の日と同じ `sys.path` の技)。他のファイルから:トークナイザー(**Day 6.1**)、モデル(**Day 8.1**)、生成エンジン全体(**Day 10.1**)、`sha256_file`(**Day 9.2**)。
- `CANDIDATE_PROMPTS` — 報告書の候補の表に使う4つの書き出し:一般的なもの(`今日は`)と、会社風のもの。

---

## 3. ヘルパー関数

### 3.1 `read_prompts`

```python
def read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
```
- プロンプトファイルを読む:1行に1プロンプト。**空行と、`#` で始まる行(コメント)は飛ばす。**

### 3.2 `rel`

```python
def rel(path: Path) -> str:
    """A path relative to the repository for reports (no personal folder names in committed files)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name
```
- パスを、`/` の区切りで**リポジトリからの相対**で表示する。リポジトリの外のファイルなら**ファイル名だけ**。だから、報告書にあなた個人のフォルダー名(`C:\Users\...` など)が入ることは無い — 公開されたリポジトリがあるので、大事。

### 3.3 `val_loss_at`

```python
def val_loss_at(run_dir: Path, step: int) -> str:
    log = run_dir / "log.csv"
    if log.exists():
        with log.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["event"] == "eval" and int(row["step"]) == step and row["val_loss"]:
                    return f"{float(row['val_loss']):.4f}"
    return "-"
```
- 学習の `log.csv`(Day 9.2)から、**指定のステップで記録された検証の損失**を探す。報告書が「このチェックポイントはステップ N で保存され、そのときの検証の損失は X」と言えるように。見つからなければ `"-"`。

### 3.4 `pick_device`、`show`、`candidate_rows`、`load_checked`

```python
def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)
```
- Day 8と同じ(「CUDAが無い」メッセージは無し)。

```python
def show(g: Generation) -> str:
    """Prompt + continuation for tables: ⟨eos⟩ = finished, … = cut at the length limit."""
    end = " ⟨eos⟩" if g.finished else " …"
    return (g.text.replace("|", "\\|").replace("\n", " ") + end).strip()
```
- 生成結果をMarkdownの表用に整える:**続きの文**(`g.text`。docstringとは違い、プロンプトは別の列/見出しに表示される)の `|` をエスケープし(Day 9.2と同じ)、改行を置き換え、印を付ける:**⟨eos⟩** = モデルが質問を終えた;**…** = 長さの上限で切られた。

```python
def candidate_rows(logits_fn, tok, prompt, k) -> list[tuple[str, float]]:
    ids = [1] + tok.encode(prompt)
    return [(token_text(tok, i), p) for i, p in next_token_candidates(logits_fn(ids), k)]
```
- プロンプトについて:モデルへの入力を作り(`[1]` = `<bos>` + プロンプトのトークン)、点数を得て、上位 `k` 個の候補を取り、各トークンに**読めるラベル**(`token_text`)を付ける。`(文字, 確率)` の組を返す。

```python
def load_checked(path: Path, tok_path: Path, device: torch.device):
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run Day 9 first.")
    model, info = load_model(path, device)
    if info["tokenizer_sha256"] != sha256_file(tok_path):
        raise SystemExit(f"{path.name} was trained with a different tokenizer")
    return model, info
```
- チェックポイントを読み込み、**トークナイザーを確認する**:チェックポイントに保存された指紋が、今ディスクにあるトークナイザーのファイルのSHA-256と等しくなければならない。さもないと、番号が別の言葉を意味してしまう → 止まる。(Day 9.2の§5.2と同じ種類の確認。)

---

## 4. コマンド1:`candidates`

```python
def cmd_candidates(a, tok, device) -> int:
    model, info = load_checked(a.checkpoint, a.tokenizer, device)
    fn = model_logits_fn(model, device)
    md = [f"# Next-token candidates ({rel(a.checkpoint)}, step {info['step']})", ""]
```
- チェックポイントを読み込んで確認し、モデルを点数の関数として包み(Day 10.1の§4.3)、Markdownの行を始める。

```python
    for prompt in a.text or list(CANDIDATE_PROMPTS):
        rows = candidate_rows(fn, tok, prompt, a.k)
        print(f"\n{prompt}  ->  pieces: {' / '.join(tok.pieces(prompt))}")
        for r, (t, p) in enumerate(rows, 1):
            print(f"  {r:>2}. {t:<12} {p:7.2%}")
        print(f"      top {a.k} total: {sum(p for _, p in rows):.2%}")
        md += [f"## {prompt}", "", "| Rank | Next token | Probability |", "| --- | --- | --- |",
               *(f"| {r} | {t} | {p:.2%} |" for r, (t, p) in enumerate(rows, 1)),
               f"| | top {a.k} total | {sum(p for _, p in rows):.2%} |", ""]
```
- 各書き出し(`--text` で指定したもの、または初期の4つ — `a.text or …`)について:
  - 上位 `k` 個の候補を取る;
  - プロンプトを**トークナイザーの断片**と一緒に表示し、順位付きのリストを表示する:順位、トークン、**確率のパーセント**(`:7.2%` = 幅7で小数2桁のパーセント;`:<12` = 12文字で左寄せ)、そして**上位 `k` の確率の合計**;
  - 同じ内容をMarkdownの表にも加える。
- 合計は、モデルがどれだけ「自信がある」かを示す:上位10個で95%を覆うなら、モデルは集中している。30%しか覆わないなら、迷っている。

```python
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(md), encoding="utf-8", newline="\n")
        print(f"\nSaved: {a.out}")
    return 0
```
- `--out` があれば、Markdownの表も保存する。0を返す。

---

## 5. コマンド2:`generate`

```python
def settings_from_args(a: argparse.Namespace) -> SamplingConfig:
    cfg = PRESET_SETTINGS[a.setting]
    over = {k: v for k, v in (("temperature", a.temperature), ("top_k", a.top_k), ("top_p", a.top_p),
                              ("repetition_penalty", a.repetition_penalty), ("seed", a.seed),
                              ("max_new_tokens", a.max_new_tokens)) if v is not None}
    return replace(cfg, **over, name=cfg.name + ("*" if over else ""))
```
- 選んだ**プリセット**(`--setting`)から始める。`over` — **入力した上書き**(`--temperature`、`--top-k` など)の辞書で、`(名前, 値)` の組のうち、**`None` でないもの**を残して作る。
- `replace(cfg, **over, name=…)` — 上書きを適用した、プリセットの**コピー**。何か上書きしたら、名前に `*` が付く(表で、変更された設定だと分かるように)。`replace` は検証をもう一度行うので、悪い値(`--temperature 0`)は弾かれる。

```python
def cmd_generate(a, tok, device) -> int:
    model, info = load_checked(a.checkpoint, a.tokenizer, device)
    fn = model_logits_fn(model, device)
    cfg = settings_from_args(a)
    prompts = a.text or read_prompts(a.prompts)
    print(f"Checkpoint {rel(a.checkpoint)} (step {info['step']}) | setting {cfg.to_dict()}")
    md = [f"# Generation ({rel(a.checkpoint)}, step {info['step']}, setting {cfg.name})", "",
          f"Settings: `{cfg.to_dict()}`", "", "| Prompt | Continuation |", "| --- | --- |"]
    for prompt in prompts:
        g = generate(fn, tok, prompt, cfg)
        print(f"  {prompt} | {show(g)}")
        md.append(f"| {prompt} | {show(g)} |")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
        print(f"Saved: {a.out}")
    return 0
```
- 読み込んで確認し、設定を作り、プロンプトを取り(`--text` で指定したもの、または**プロンプトファイル**の行)、設定を表示し、プロンプトごとに:**生成**し(Day 10.1の `generate`)、`プロンプト | 続き` を表示し、表の行を足す。任意で保存。

---

## 6. コマンド3:`report`

### 6.1 準備

```python
def cmd_report(a, tok, device) -> int:
    run_dir = REPO_ROOT / a.runs_root / a.run
    ckpts = {"best": run_dir / "best.pt", "last": run_dir / "last.pt"}
    loaded = {name: load_checked(p, a.tokenizer, device) for name, p in ckpts.items()}
    fns = {name: model_logits_fn(m, device) for name, (m, _) in loaded.items()}
    infos = {name: info for name, (_, info) in loaded.items()}
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
```
- 両方のチェックポイント(`best` と `last`)を読み込み、それぞれを点数の関数として包み、情報(ステップなど)を保つ。3つの**辞書内包表記**が、`"best"` / `"last"` をキーにした `loaded`、`fns`、`infos` を作る。
- `prompts` — プロンプトファイルから。`train_set` — **すべての学習用の質問の集合**(速く検索できる)。あとの「新しい?」の確認(参考用のみ)に使う。`device_name` はDay 8と同じ。
- そして見出しを表示し、各チェックポイントについて、そのステップと、そのステップでの検証の損失を表示する。報告書は、2つのチェックポイントの表で始まる(**best = 検証の損失が最も低い**)。

### 6.2 セクション1:次のトークンの候補(作業10)

```python
    for prompt in CANDIDATE_PROMPTS:
        rb = candidate_rows(fns["best"], tok, prompt, 10)
        rl = candidate_rows(fns["last"], tok, prompt, 10)
        ...
        for r, ((tb, pb), (tl, pl)) in enumerate(zip(rb, rl), 1):
            print(f"  {r:>2}. {tb:<10} {pb:7.2%}   | {tl:<10} {pl:7.2%}")
        md += [... "Rank | best.pt token | Probability | last.pt token | Probability" の表 ...]
```
- 4つの書き出しそれぞれについて:**両方の**チェックポイントの上位10個の候補を、**横に並べて**(`zip(rb, rl)` が、bestの1位とlastの1位、…を組にする)、確率と上位10の合計付きで見せる。bestとlastで、モデルの考えがどう違うかが分かる。

### 6.3 セクション2:両方のチェックポイントで、すべての設定で生成

```python
    results: dict[tuple[str, str], list[Generation]] = {}
    for sname, cfg in PRESET_SETTINGS.items():
        for cname in ckpts:
            results[(cname, sname)] = [generate(fns[cname], tok, p, cfg) for p in prompts]
```
- **すべてのプリセット × 両方のチェックポイント**について、**すべてのプロンプト**の続きを生成し、`(チェックポイント名, 設定名)` をキーにした辞書に保存する。
- 報告書は次を書く:**4つの設定の表**(方法、温度、top-kまたは`-`、top-pまたは(1なら)`-`、繰り返しの上限の `on`/`off` — `c.max_run` は設定されていれば真)と、**プロンプトごとの表**:行 = 設定、列 = `best.pt` と `last.pt` の続き(`show(...)`)。`⟨eos⟩` = 終えた、`…` = 40トークンで切られた。

### 6.4 セクション3:自動の確認(作業6の項目 A2〜A5)

```python
    for (cname, sname), gens in results.items():
        n = len(gens)
        fin = sum(g.finished for g in gens)
        rep = sum(is_repetitive(tok.encode(g.prompt) + [i for i in g.ids if i != 2]) for g in gens)
        d2 = distinct_2([[i for i in g.ids if i != 2] for g in gens])
        new = sum((g.prompt + g.text) not in train_set for g in gens if g.finished)
        md.append(f"| {cname} | {sname} | {fin}/{n} | {rep}/{n} | {d2:.2f} | {new}/{fin} |")
        print(f"{cname:<5} {sname:<14} {fin:>6}/{n:<2} {rep:>8}/{n:<2} {d2:>11.2f} {new:>3}/{fin:<2}")
```
- (チェックポイント、設定)ごとに:
  - **Finished(A2)** — `<eos>` で終わった出力がいくつか(`sum` が True/False のTrueを数える);
  - **Repetitive(A3)** — どの本物の質問よりも繰り返しが多い出力がいくつか(`is_repetitive` を、**プロンプト + 生成したトークン(`<eos>` を除く)**に対して。`2` は `<eos>` のID);
  - **distinct-2(A4)** — すべての出力にわたる多様さ;
  - **New(A5、参考用のみ)** — **終えた**出力のうち、学習用の質問**ではない**ものがいくつか(`prompt + text` を `train_set` と比べる)。新しい = 良い、ではない。
- 整った表として表示し(書式のコードは前と同じ)、報告書に `finished/n`、`repetitive/n`、`distinct-2`、`new/finished` として加える。

### 6.5 セクション4:**未学習**のモデルでの繰り返しの上限

```python
    md += untrained_limits_section(run_dir, tok, device)
```
- デモのセクションを加える(次の関数)。

### 6.6 保存とログ

```python
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"\nReport: {a.report}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(...)   # 見出し + スクリプト/デバイス/PyTorch;学習とbest/lastのステップ;プロンプトファイル、設定、報告書の名前
        print("Appended an entry to run_log.md")
    return 0
```
- `docs/DAY10_GENERATION_REPORT.md` を保存し、(`--no-log` でなければ)`run_log.md` に短い項目を追記する。

### 6.7 `untrained_limits_section` — 「上限が働く」デモ

```python
def untrained_limits_section(run_dir, tok, device) -> list[str]:
    cfg_path, train_cfg_path = run_dir / "model_config.json", run_dir / "train_config.json"
    if not cfg_path.exists() or not train_cfg_path.exists():
        return []
    seed = json.loads(train_cfg_path.read_text(encoding="utf-8"))["seed"]
    torch.manual_seed(seed)
    untrained = ModelA(ModelConfig.load(cfg_path)).to(device).eval()
    fn = model_logits_fn(untrained, device)
```
- **問題:***学習した*チェックポイントは、めったに繰り返しに落ちないので、上限が目に見えて何かをすることは無い。**工夫:**学習の**未学習(ステップ0)のモデル**を作り直す:学習の保存済みのモデルの設定と**シード**を読み、同じシードを設定してモデルを作る — **Day 9はまさにこれをした**(`torch.manual_seed(seed)` の後に `ModelA(...)`)ので、重みは**本物の学習の最初の重みと同一**になる。(学習の設定ファイルが無ければ、このセクションは飛ばされる:`return []`。)

```python
    lines = ["## 4. Repetition limits on the untrained model (step 0)", "", f"The step-0 model is rebuilt with the run's seed ({seed}), ... ", "",
             "| Prompt | greedy | greedy+limits |", "| --- | --- | --- |"]
    for prompt in ("営業部の", "大阪支社で"):
        g0 = generate(fn, tok, prompt, PRESET_SETTINGS["greedy"])
        g1 = generate(fn, tok, prompt, PRESET_SETTINGS["greedy+limits"])
        print(f"  {prompt} | {show(g0)[:30]} | {show(g1)[:30]}")
        lines.append(f"| {prompt} | {show(g0)} | {show(g1)} |")
    return lines + [""]
```
- プロンプト2つについて、上限**なし**の貪欲法(未学習のモデルは、1つの断片を何度も繰り返す)と、上限**あり**(同じトークンは続けて4個まで、3トークンの並びは2回まで — 繰り返しから出るよう強制される)を比べる。表示(最初の30文字)し、表として書く。これが、**上限が働くことの目に見える証拠**になる。

---

## 7. 引数(サブコマンド)

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 10: next-word candidates and generation.")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    sub = ap.add_subparsers(dest="command", required=True)
    default_ckpt = REPO_ROOT / "runs/day9/smoke_a/best.pt"
```
- すべてのコマンドに共通の2つのオプション(`--tokenizer`、`--device`)と、`add_subparsers(dest="command", required=True)` — コマンドの名前を**必ず**書く必要があり、その名前が `a.command` に入る。`default_ckpt` — `candidates`/`generate` の初期のチェックポイント。

```python
    c = sub.add_parser("candidates", help="top-k next tokens for beginnings")
    c.add_argument("--checkpoint", type=Path, default=default_ckpt)
    c.add_argument("--text", action="append", help="beginning (repeatable); default: four examples")
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--out", type=Path, default=None, help="also save a Markdown table")
```
- **`candidates`** コマンド:`--checkpoint`、`--text`(`action="append"` で何度も指定できる)、`--k`(候補の数、10)、`--out`。

```python
    g = sub.add_parser("generate", help="continue prompts with one setting")
    g.add_argument("--checkpoint", ...)  g.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
    g.add_argument("--text", action="append", help="beginning (repeatable) instead of the prompt file")
    g.add_argument("--setting", choices=list(PRESET_SETTINGS), default="sample")
    g.add_argument("--temperature", type=float)  g.add_argument("--top-k", type=int)  g.add_argument("--top-p", type=float)
    g.add_argument("--repetition-penalty", type=float)  g.add_argument("--seed", type=int)  g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--out", type=Path, default=None)
```
- **`generate`** コマンド:チェックポイント、プロンプトファイル(または `--text`)、プリセット(`--setting`、初期値 `sample`)、そして初期値のない**任意の上書き**(だから `None` = 「指定なし」。`settings_from_args` がこれに頼る)。

```python
    r = sub.add_parser("report", help="full Day 10 report for best.pt and last.pt of a run")
    r.add_argument("--run", default="smoke_a")
    r.add_argument("--runs-root", default="runs/day9")
    r.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
    r.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY10_GENERATION_REPORT.md")
    r.add_argument("--machine", choices=["home", "server", "school"], default="home")
    r.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    for name in ("tokenizer", "checkpoint", "prompts", "report", "out"):
        if getattr(a, name, None) is not None:
            setattr(a, name, getattr(a, name).resolve())
    return a
```
- **`report`** コマンド:どの学習か(`--run`、フォルダー `runs/day9/<run>`)、プロンプトファイル、報告書のパス、マシンのラベル、`--no-log`。
- 最後に、`getattr`/`setattr` のループ(前の日と同じ)でパスのオプションを**絶対パス**にする。`getattr(a, name, None)` は、そのコマンドに無いオプションでは `None` を返すので、存在するものだけが処理される。

---

## 8. `main`

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    tok = ByteBPETokenizer.load(a.tokenizer)
    device = pick_device(a.device)
    return {"candidates": cmd_candidates, "generate": cmd_generate, "report": cmd_report}[a.command](a, tok, device)


if __name__ == "__main__":
    sys.exit(main())
```
- コンソールの安全対策、解析、トークナイザーの読み込み、デバイスの選択、そして**振り分け**:辞書がコマンド名を関数に対応させ、`[a.command]` が1つ選び、`(a, tok, device)` がそれを呼ぶ。その戻り値(0)が終了コード。

---

## 9. 結果の読み方

- **候補:**学習したモデルは、もっともらしい次のトークンに高い確率を置くはず。**上位10の合計**が、どれだけ集中しているかを教える。完結して見える質問の後で `<eos>` が高く出る = 終わり方を学んだ。
- **貪欲法とサンプリング:**貪欲は決定的で、繰り返すことがある。`sample` はもっと多様。`sample-hot` はわざとランダムすぎる(意味不明、壊れた文字)— 温度のよい実例。
- **自動の確認:****Finished** は高く、**Repetitive** は低く、**distinct-2** はそこそこ高く。**New** は参考用だけ。
- **セクション4**は、未学習のモデルで、上限が繰り返しを断ち切るのを見せる。
- **best.pt と last.pt:**短いスモークの学習の後では、たいてい似ている。bestのステップの後で検証が悪くなったときは、違ってくる。

## 10. 知っておくと良い小さなこと

- **プロンプトファイルが存在する必要がある**(初期値は `prompts/day10_prompts.txt`)。
- **`show()` が見せるのは続きだけ**で、docstringには「プロンプト + 続き」とあるが、プロンプトは表の中で別に表示される。
- **`report` コマンドには、`best.pt` と `last.pt` の両方が必要**(Day 9の学習は、最初の評価で両方を作る)。
- **ログの項目は `a.prompts.relative_to(REPO_ROOT)` を使う**ので、プロンプトファイルがリポジトリの外にあると失敗する(報告書そのものは動く)。
- **ハードコードされた `2`** は `<eos>` のID(Day 9.2と同じ)。
- **「New」は参考用のみ:**学習用のセットに無い文でも、意味不明なことがある。

## 11. 一段落でまとめ

`day10_generate.py` は生成のコマンドラインの入口です:`candidates` は次のトークンの上位とその確率を見せ、`generate` は選んだ(または上書きした)設定でプロンプトを続け、`report` は、最良と最後のチェックポイントの証拠一式 — 横に並べた候補、すべてのプロンプトでの4つの設定、自動の確認(finished / repetitive / distinct-2 / new)、そして未学習のステップ0のモデルで繰り返しの上限がループを止めることのデモ — を、各チェックポイントのトークナイザーの指紋がトークナイザーのファイルと一致することを確認したうえで作ります。
