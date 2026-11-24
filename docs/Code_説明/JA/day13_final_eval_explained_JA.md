# Day 13 — day13_final_eval.py — やさしい1行ずつの解説(日本語)

> **Day 13 の4つのファイルの関係**
> - **Day 13.1 `evaluation.py`** — 道具箱(先に読む)。
> - **Day 13(このファイル)`day13_final_eval.py`** — **前半**:最終のモデルの、プログラムによる最終評価 + 人が採点するシート。
> - **Day 13 `day13_score_summary.py`** — 後半:人の採点を読み、最終の表を書く。
> - **Day 13.2 `test_evaluation.py`** — 道具箱のテスト。
>
> おすすめの読む順番:**13.1 → 13 final_eval(これ)→ 13 score_summary → 13.2**。

---

## 0. 全体像(まずここから)

Day 12は、**検証用**の文章だけで最終のモデル(`configs/day12_final.json`)を選びました。Day 13は**最終試験**です:

| 部分 | 内容 | 理由 |
|---|---|---|
| **A1 パープレキシティ** | 検証用**と、テスト用の文章**で(テスト用を使うのは最初で最後) | 正直な数字:選択に一度も使っていない |
| **比較の基準** | 一様な当てずっぽう、学習前のモデル、バイグラム(数えるだけ)の基準 | 数字だけでは意味がない。「何と比べて?」 |
| **A2〜A5** | **固定の50の書き出し**、2つの生成方法、**best.pt と last.pt** で、完結/繰り返し/distinct-2/コピー | 作業6のチェック項目 |
| **追加の確認** | 単位の誤り、壊れた数字、学習にない値 | Day 12で見えた誤り(例:勤続年数が300万円) |
| **人の採点シート** | 最終のモデルの100個の出力。順番をばらばらにし、生成方法を隠す | 日本語の質は人が判断する(後半) |

```
.venv\Scripts\python.exe scripts\day13_final_eval.py --machine home
```

### 書き出されるもの

| ファイル | 中身 |
|---|---|
| `docs/DAY13_AUTO_EVALUATION.md` | プログラムによる結果(報告書) |
| `docs/figures/day13_perplexity.png` | A1を基準と比べた図 |
| `docs/day13/auto_results.json` | すべての数(後半が読む) |
| `docs/day13/generations.csv` | **200個**すべての出力と確認の結果(Excelで開ける) |
| `docs/day13/scoring_sheet.xlsx` | 採点者に渡すシート |
| `docs/day13/scoring_key.json` | 各出力がどの方法で作られたか(**採点者には見せない**) |

---

## 1. 先に知っておきたい用語

| 用語 | かんたんな意味 |
|---|---|
| **テスト用の文章** | どんな決定にも使っていないデータ。最後に一度だけ使うと、正直な点が出る。 |
| **パープレキシティ** | `exp(損失)`:「モデルが何個のトークンで迷っているか」。低いほど良い。 |
| **ブラインド採点** | 採点者はどの方法の出力かを知らないので、偏りが入らない。 |
| **シャッフル** | ランダムに並べ替えること(ここでは固定の種13なので再現できる)。 |
| **A2〜A5** | 作業6の自動の項目:A2 完結、A3 繰り返し、A4 distinct-2、A5 学習用の質問のコピー。 |

---

## 2. importと定数

```python
import argparse, csv, json, math, platform, random, sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from src.batching import open_tokens
from src.bpe_tokenizer import ByteBPETokenizer
from src.evaluation import (PROMPT_KINDS, BigramBaseline, ScoringRow, UnitChecker, has_broken_number,
                            load_eval_prompts, max_digit_run, term_phrases, write_scoring_workbook)   # Day 13.1
from src.generation import (EOS_ID, PRESET_SETTINGS, distinct_2, generate, is_repetitive, load_model, model_logits_fn)
from src.model import ModelA, ModelConfig
from src.selection import MAX_REPETITIVE, MIN_FINISHED
from src.training import evaluate, pick_precision

SCRIPT_VERSION = "day13-v1"
METHODS = ("greedy+limits", "sample")
SHUFFLE_SEED = 13
```
- 新しい標準のモジュール:**`random`**(シャッフルのため)。プロジェクトから:トークンのファイル(7.1/9.1)、トークナイザー(6.1)、**Day 13.1の道具箱**、生成(10.1)、モデル(8.1)、目安のしきい値(11.1)、そして学習のライブラリ(9.2)の `evaluate` と `pick_precision` — 学習と**同じ**損失の関数なので、数字を比べられる。
- `METHODS` — 比べる2つの生成方法(Day 10.1のプリセット)。`SHUFFLE_SEED = 13` — シャッフルを固定する。

---

## 3. `parse_args` と `read_lines`

- オプション:`--machine`;`--final`(初期値 `configs/day12_final.json`、Day 12からの引き継ぎのファイル);`--prompts`(50の書き出しのCSV);`--device`;`--batch-size 32`;`--eval-max-windows`(テスト用);`--out-dir docs/day13`;`--report`;`--figure`;**`--force`**(採点がすでにあっても採点シートを置き換える);`--no-log`。
- `read_lines(path)` — ファイルの行(**空の行を除く**)。

---

## 4. `main`

### 4.1 最終のモデルを見つける

```python
    final = json.loads(a.final.read_text(encoding="utf-8"))
    best_path = REPO_ROOT / final["final_checkpoint"]
    run_dir = best_path.parent
    last_path = run_dir / "last.pt"
    for p in (best_path, last_path):
        if not p.exists():
            raise SystemExit(f"{p} not found. Day 13 needs the Day 12 run folder ({run_dir}); do not delete runs/day12.")
```
- Day 12の引き継ぎのファイルを読む → `best.pt` のパス;`last.pt` は同じフォルダー。どちらかが無ければ、分かりやすいメッセージで止まる(`runs/day12` はGit管理外なので、うっかり消しやすい)。

```python
    device = ...                       # 前と同じ入れ子の条件式
    precision = pick_precision("auto", device)
    mcfg = ModelConfig.load(REPO_ROOT / final["model_config"])
    vocab = mcfg.vocab_size
```
- デバイス、精度(学習と同じ bf16/fp16/fp32)、モデルの設定;`vocab` = 1,674。

### 4.2 データ

```python
    tok = ByteBPETokenizer.load(...)
    train_tok, val_tok, test_tok = (open_tokens(tokens_dir, s) for s in ("train", "val", "test"))
    train_lines = read_lines(REPO_ROOT / "data/corpus/train.txt")
    test_lines = read_lines(REPO_ROOT / "data/corpus/test.txt")
    train_set = set(train_lines)
    prompts = load_eval_prompts(a.prompts)
    counts = {k: sum(p.kind == k for p in prompts) for k in PROMPT_KINDS}
```
- 3つの分割のトークンの列(**`test` を初めて開くのはここ**)、学習用とテスト用の質問(文字のまま)、`train_set`(「コピー」の判定を速くする)、50の書き出し、種類ごとの数(表示される)。

### 4.3 `loss_of`

```python
    def loss_of(model, tokens) -> float:
        v = evaluate(model, tokens, mcfg.block_size, a.batch_size, device, precision, a.eval_max_windows)
        model.eval()
        return v
```
- Day 9.2の `evaluate` で、モデルがトークンの列に対して出す平均の損失。この関数は終わりに、モデルを**学習モードに戻す**ので、`model.eval()` でドロップアウトをまた切る — **生成は評価モードで動かす必要がある**。

### 4.4 A1 — パープレキシティと基準

```python
    a1["uniform"]   = {"label": ..., "val": math.log(vocab), "test": math.log(vocab)}
```
- **一様な当てずっぽう:**どのトークンも同じ確率 → 損失 = ln(1,674) ≈ 7.42(パープレキシティ = 1,674)。

```python
    torch.manual_seed(0)
    untrained = ModelA(mcfg).to(device).eval()
    a1["untrained"] = {... "val": loss_of(untrained, val_tok), "test": loss_of(untrained, test_tok)}
    del untrained
```
- **学習前のモデル:**同じ大きさでランダムな重み(固定の種0)→ 学習がどこから始まったかを示す。`del` でメモリを空ける。

```python
    bigram = BigramBaseline.fit(train_tok, vocab)
    lam = bigram.tune(val_tok)
    a1["bigram"] = {..., "val": bigram.loss(val_tok), "test": bigram.loss(test_tok)}
```
- **数えるだけの基準(Day 13.1):****学習用**のトークンで作り、重み `lam` は**検証用**で選び、両方で報告する。

```python
    models = {}
    for name, path in (("best.pt", best_path), ("last.pt", last_path)):
        model, info = load_model(path, device)
        models[name] = (model, info)
        a1[name] = {"label": f"Final model {name} (step {info['step']})", "val": ..., "test": ..., "step": info["step"]}
```
- 最終の学習の**両方の**チェックポイントを読み込み、それぞれ測る。`best.pt` = 検証の損失が最低;`last.pt` = 学習が終わった所(作業6の 6-2 が両方を求める)。

```python
    others = {}
    for seed in final.get("seeds", []):
        p = run_dir.parent / f"main_s{seed}" / "best.pt"
        if p.exists() and p.resolve() != best_path.resolve():
            m, info = load_model(p, device)
            others[f"main_s{seed}"] = {...}
            del m
```
- **他の2つのシード**の `best.pt` にも検証とテストの数を付ける — **参考としてだけ**(テストの結果がシードにどれだけ左右されるか)。最終のモデルが決まった**後**に評価するので、どの選択にも関わらない。
- そして、検証とテストの `損失(パープレキシティ)` をすべて表示する。

### 4.5 検出器 — まず本物のデータで確かめる

```python
    unit_checker = UnitChecker(term_phrases(REPO_ROOT / "data/source/columns.csv")).learn(train_lines)
    digit_limit = max_digit_run(train_lines)
    calib = {"questions": len(test_lines),
             "wrong_unit": sum(bool(unit_checker.mismatches(q)) for q in test_lines),
             "broken_number": sum(has_broken_number(q, digit_limit) for q in test_lines),
             "unseen_value": sum(bool(unit_checker.unseen_values(q)) for q in test_lines),
             "digit_limit": digit_limit, "items_learned": len(unit_checker.allowed)}
```
- 単位の検査器(学習用の質問から学ぶ)を作り、数字の長さの上限を測る。**較正:**各検出器を**本物のテスト用の質問すべて**に使う — 本物の質問では、ほぼ0件になるはず。そうなら、生成された文の警告は**誤警報ではなく本物の問題**(この理屈は報告書に書かれる)。

```python
    unseen = [p.prompt for p in prompts if p.kind == "unseen"]
    unseen_new = sum(not any(q.startswith(u) for q in train_lines) for u in unseen)
```
- 「学習にない言い方」の書き出しのうち、学習用の質問の始まりに**本当に一度も無い**ものの数(10個の「unseen」が本当に未知か確かめる)。

### 4.6 50の書き出しでの A2〜A5

```python
    for mname, (model, _) in models.items():          # best.pt, last.pt
        fn = model_logits_fn(model, device)
        for method in METHODS:                          # greedy+limits, sample
            body_ids, rows = [], []
            for p in prompts:                           # 50の書き出し
                g = generate(fn, tok, p.prompt, PRESET_SETTINGS[method])
                body = [i for i in g.ids if i != EOS_ID]
                body_ids.append(body)
                full = p.prompt + g.text
                rows.append({"model": mname, "method": method, "kind": p.kind, "prompt": p.prompt, "text": g.text,
                    "finished": g.finished, "repetitive": is_repetitive(tok.encode(p.prompt) + body),
                    "copy": g.finished and full in train_set,
                    "wrong_unit": "; ".join(f"{t}+{u}" for t, u in unit_checker.mismatches(full)),
                    "broken_number": has_broken_number(full, digit_limit),
                    "unseen_value": "; ".join(unit_checker.unseen_values(full))})
```
- **2つのチェックポイント × 2つの方法 × 50の書き出し = 200個の出力。**それぞれ:生成(最大40トークン、プリセットの種42);`body` = `<eos>` を除いた生成のID;`full` = 書き出し+続き(検出器は文全体を見る)。
- 出力ごとに:**finished**(`<eos>` で終わった)、**repetitive**(どの学習用の質問より繰り返しが多い)、**copy**(終えた**かつ**学習用の質問とまったく同じ — A5、参考:丸暗記)、**wrong_unit** / **unseen_value**(見つかったものを並べた文字。空 = 無し)、**broken_number**。

```python
            n = len(rows)
            agg[f"{mname}|{method}"] = {"finished": sum(...)/n, "repetitive": ..., "distinct2": distinct_2(body_ids),
                "copy": ..., "wrong_unit": sum(bool(r["wrong_unit"]) ...)/n, "broken_number": ..., "unseen_value": ...}
            outputs.extend(rows)
```
- (チェックポイント, 方法)ごとに**割合**でまとめる(distinct-2はすべての出力をまとめて計算)、すべての行を残し、組み合わせごとに1行表示する。

### 4.7 人の採点シート

```python
    scored = [r for r in outputs if r["model"] == "best.pt"]          # 100個の出力
    random.Random(SHUFFLE_SEED).shuffle(scored)
    key = [{"id": f"S{i:03d}", "method": r["method"], "kind": r["kind"], "prompt": r["prompt"],
            "text": r["text"] + ("" if r["finished"] else " …"), "out_of_scope": r["kind"] == "out_of_scope"}
           for i, r in enumerate(scored, 1)]
```
- **最終のモデル(best.pt)**だけ、両方の方法 → **100個の出力**(採点者は200ではなく100個を読む)。`random.Random(13).shuffle` が、greedyとsampleの出力を**再現できるランダムな順番**に混ぜるので、採点者は方法を推測できない。
- `key` = **答えの表**:ID `S001…S100`(ゼロ埋め)、方法、種類、書き出し、文(終わらなかった出力には「 …」が付く)、対象外かどうか。

```python
    key_path, sheet_path = a.out_dir / "scoring_key.json", a.out_dir / "scoring_sheet.xlsx"
    scores_dir = a.out_dir / "scores"
    has_scores = scores_dir.exists() and any(scores_dir.glob("scores_*.xlsx"))
    old_key = json.loads(key_path.read_text(encoding="utf-8"))["rows"] if key_path.exists() else None
    if has_scores and old_key is not None and old_key != key and not a.force:
        raise SystemExit("The new outputs differ from the sheet the scorers already used ... Use --force only if you intend to score again.")
```
- **安全のための見張り:**採点者が**すでに採点ファイルを** `docs/day13/scores` に置いていて、新しい出力が古い答えの表と**違う**(例えば、モデルやGPUが少し違う文を出した)なら、**上書きせずに止まる** — さもないと、採点が答えの表と合わなくなる。`--force` は意図的に上書きする。

```python
    key_path.write_text(json.dumps({...,"shuffle_seed": SHUFFLE_SEED, "rows": key}, ...))
    write_scoring_workbook(sheet_path, [ScoringRow(k["id"], k["prompt"], k["text"], k["out_of_scope"]) for k in key])
    scores_dir.mkdir(exist_ok=True)
```
- 答えの表のJSONを書き、Excelのシートを作る(**生成方法の列は入っていない**)、採点者がファイルを置く空の `scores/` フォルダーを作る。

### 4.8 ファイル、図、報告書、ログ

- `generations.csv` — 200行すべて。**`utf-8-sig`** = Excelが日本語を正しく表示するのに必要な印付きのUTF-8。
- `auto_results.json` — スクリプト、日付、マシン、デバイス、最終のモデルの情報、A1、他のシード、まとめた生成の数、較正、書き出しの数、unseenの数、方法の設定、採点する出力の数、目安。**後半がこのファイルを読む。**
- `draw_figure`、`write_report`、そして `run_log.md` への1項目(テストの損失/パープレキシティ、バイグラムの基準、チェックポイントと方法ごとの1行、シードの情報)。`return 0`。

---

## 5. `draw_figure`

- **テストのパープレキシティ**の棒グラフ:uniform、untrained、bigram、last.pt、best.pt を**対数軸**で(さもないと1,674の棒が大きすぎて、他が見えなくなる)。灰色の棒 = 基準、青 = 最終のモデル。各棒の上に値を表示(`v * 1.12` で少し上に)。`set_ylim(1, max × 3)` はラベルの余白。**matplotlibが必要。**

## 6. `_cell` と `write_report`

`_cell` は文をMarkdownの表で安全にする(Day 12と同じ)。`write_report` が `docs/DAY13_AUTO_EVALUATION.md` を作る:

| セクション | 内容 |
|---|---|
| ヘッダー | バージョン、日付、マシン;最終のモデル;「**テスト用の文章を使うのはここが初めて**」 |
| **A1 パープレキシティ** | 図 + 表(各行の検証/テストの損失とパープレキシティ。他のシードは「(reference)」)、学習前のモデルとバイグラムとの比較(「書き出し全体を読むことでTransformerが加えるものを示す」) |
| **A2〜A5** | 生成の設定と目安(完結 ≥ 90%、繰り返し ≤ 5%);チェックポイント × 方法ごとに1行 |
| 較正のメモ | 本物のテスト用の質問で検出器が見つけたもの;数字の長さの上限;「学習にない値」の説明 |
| 種類ごと | 種類ごとに:数、greedy対sampleの 完結 / 単位の誤り / コピー(最終のモデル) |
| **すべての出力** | 各書き出しに対する、最終のモデルのgreedyとsamplingの続き;記号 **U** 単位の誤り、**B** 壊れた数字、**V** 学習にない値、**R** 繰り返し、**C** 完全なコピー、**…** 終わらなかった |
| 次へ | 採点シートの場所、採点者がコピーを保存する方法(`docs/day13/scores/scores_<イニシャル>.xlsx`)と後半の役割;**last.pt はプログラムでの確認だけ** |

---

## 7. 実行の流れ(順番)

1. Day 12の引き継ぎのファイルを読み、`best.pt` と `last.pt` があるか確かめる。
2. データを読み、基準、best.pt、last.pt、他のシードのA1パープレキシティ。
3. 検出器を、本物のテスト用の質問で較正する。
4. 200個の出力を生成し、A2〜A5と追加の確認を計算する。
5. 最終のモデルの100個の出力をシャッフル → 答えの表 + Excelのシート(上書きの見張りつき)。
6. CSV、JSON、図、報告書、実行ログ。

## 8. 結果の読み方

- **A1:**`best.pt` のテストのパープレキシティは、学習前(≈ 1,674)より**ずっと低く**、**バイグラムより低い**はず。テストが検証に近い = 検証での選択に過剰に合わせていない。
- **bestとlast:**`last.pt` がずっと悪ければ、学習の後半で過学習したということ(だからDay 12は `best.pt` を残した)。
- **完結 / 繰り返し:**目安(90%以上 / 5%以下)と比べる。
- **単位の誤り / 壊れた数字:**Day 12で見えた問題。0%に近いのが理想。
- **コピー:**高ければ、学習用の質問を丸暗記している。

## 9. 知っておくと良い小さなこと

- **A1の表のすべての行でテスト用の文章を評価する**(学習前のモデルや他のシードも)。どの結果も決定を変えないので問題ないが、今はテストの数字で**何も選んではいけない**。
- **再実行**すると、同じマシンでは同じ文になる(greedyと種つきのサンプリングは決定的)ので、採点の答えの表は同じまま。別のGPUではごくわずかな違いがあり得て、まさにそのための見張り。
- **単位の誤りの検出は、学習で単位と一緒に出てきた項目だけを判定する。**
- **「seed 42」** と「max 40 tokens」は、設定から読むのではなく、報告書の文に書き込まれている。
- **学習にない値**は、書き出しそのものに書かれた数字も見る。
- **CPUでの200回の生成は遅いことがある**(新しいトークンごとに順伝播1回)。
- **matplotlibとopenpyxlが必要。**
- **`scoring_key.json` を採点者に見せてはいけない** — どの出力がどの方法かが分かってしまう。

## 10. 一段落でまとめ

`day13_final_eval.py` は最終試験の前半です:Day 12の最終のモデル(best.pt と last.pt)を、一様・学習前・バイグラムの基準と並べて、検証用と、**初めてテスト用の文章**で測り、50の固定の書き出しから2つの方法で200個の出力を生成して、完結、繰り返し、distinct-2、コピー、そして(本物のテスト用の質問で較正した)3つの追加の検出器を計算します。さらに、最終のモデルの100個の出力をシャッフルして、答えの表を別にしたブラインドのExcel採点シートにし(採点者がすでに使ったシートを上書きしない見張りつき)、報告書、図、CSV、JSON、実行ログを書きます。
