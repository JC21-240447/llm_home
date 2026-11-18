# Day 11.1 — selection.py — やさしい1行ずつの解説(日本語)

> **Day 11 の3つのファイルの関係**
> - **Day 11.1(このファイル)`selection.py`** — **決めるためのロジック**:学習のまとめを作り、**書き下したルール**でモデルを選ぶ。
> - **Day 11 `day11_size_trials.py`** — *運転手*:(大きさ, ドロップアウト)のすべてを学習し、生成を確かめ、このファイルで決め、報告書を書く。
> - **Day 11.2 `test_selection.py`** — 決めるルールの6個の自動テスト。
>
> おすすめの読む順番:**11.1 → 11 → 11.2**。

---

## 0. 全体像(まずここから)

Day 8で**3つの大きさのモデル**(tiny / small / medium)を作りました。Day 11では、そのすべてを(ドロップアウト0.0と0.1も入れて **6個の学習**)、**同じ量**だけ学習させ、本番の学習のために**1つを選ばなければ**なりません。*公平に*選ぶには、どうすればよいか?

危ない方法:全部の数字を見てから、好きなものを選び、理由を後付けする。正直な方法:**結果を見る前にルールを書き**、プログラムにそれを適用させる。このファイルが、それです。

### ルール(docstringより)

| 手順 | ルール | 理由 |
|---|---|---|
| 1 | 一番良いチェックポイントが、決まった書き出しで**作業6の自動の目安**を満たすものだけが**候補**:**完結 90%以上**、**繰り返し 5%以下** | 損失がすばらしくても、文を終えられない、繰り返すモデルは使えない |
| 2 | 候補の中で、**検証の損失の最低値**を見つける | 検証 = モデルが学習していない文章 |
| 3 | それとの差が **0.02** 以内の候補は「**同じくらい良い**」 | 0.02未満の差(パープレキシティで約**2%**)は、実行ごとのゆれの範囲。大きいモデルが良いという証拠にならない |
| 4 | その中で、**パラメータが一番少ない**ものを選ぶ(同じなら検証の損失が低い方) | 小さい = 速く、安く、丸暗記しにくい |

### 小さな例

| 学習 | パラメータ | 最良の検証の損失 | 確認に合格? |
|---|---|---|---|
| tiny | 100万 | 2.615 | はい |
| small | 515万 | 2.600 | はい |
| medium | 1,480万 | 2.640 | はい |

最低値 = 2.600(small)。「同じくらい良い」= 0.02以内 → tiny(0.015悪い)と small。**パラメータが一番少ない → tiny が選ばれる**。small の損失が最低でも。

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **検証の損失** | 取り分けた文章での損失(Day 5)。低いほど、よく汎化している。 |
| **最良のステップ** | 検証の損失が一番低かったステップ。 |
| **過学習(overfitting)** | モデルが学習用の文章を丸暗記し始めること:学習の損失は下がり続けるが、検証の損失は良くならない、または上がる。 |
| **差(gap)** | 最良のステップでの、検証の損失 − 学習の損失。差が大きい = 練習したものに比べて、見たことのない言い回しがずっと下手。 |
| **最良の後の上昇** | 最後の検証の損失 − 最良の検証の損失:最良の点の後に、どれだけ悪くなったか(0 = 上がらなかった)。 |
| **ドロップアウト** | 丸暗記を減らすため、学習中に一部の数をランダムに切ること(Day 8.1)。 |
| **パープレキシティ** | `exp(損失)`(Day 9.2)。損失の差0.02 ≈ パープレキシティで2%。 |
| **許容範囲(tolerance)** | 「本当の違いはない」とみなす差の大きさ(0.02)。 |
| **中央値** | 並べ替えたリストの真ん中の値(平均より頑丈)。 |

---

## 2. docstringと定数

```python
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOLERANCE: float = 0.02
MIN_FINISHED: float = 0.90
MAX_REPETITIVE: float = 0.05
```
- 標準のimport。**ルールの3つの数**は、名前付きの定数として先頭にあるので、ルールが1か所で見える:許容範囲0.02、完結が90%以上、繰り返しが5%以下。

---

## 3. `RunSummary` — 1つの学習についての事実

```python
@dataclass
class RunSummary:
    name: str
    size: str
    dropout: float
    parameters: int
    best_val: float
    best_step: int
    train_at_best: float | None
    final_val: float
    final_step: int
    tokens_per_sec: float | None       # median over the run
    peak_vram_mb: float | None
    elapsed_s: float
    finished_rate: float | None = None  # filled in after the generation check
    repetitive_rate: float | None = None
    distinct2: float | None = None
    notes: list[str] = field(default_factory=list)
```

| 項目 | 意味 |
|---|---|
| `name`、`size`、`dropout` | どの学習か(`tiny_d00` = tiny、ドロップアウト0.0) |
| `parameters` | 学習できる数の個数(式から。Day 8.1) |
| `best_val`、`best_step` | 検証の損失の最低値と、それが起きたステップ |
| `train_at_best` | そのステップで記録された学習の損失(無ければ `None`) |
| `final_val`、`final_step` | 最後の検証の結果 |
| `tokens_per_sec` | 典型的な速さ(学習全体の中央値) |
| `peak_vram_mb` | GPUメモリの最大使用量 |
| `elapsed_s` | 学習にかかった合計時間 |
| `finished_rate`、`repetitive_rate`、`distinct2` | 生成の確認の結果(作業6の項目 A2〜A4)。最初は `None` で、あとでDay 11の運転手が**埋める** |
| `notes` | 自由なメモ(`default_factory=list`:オブジェクトごとに専用のリスト。Day 5と同じ) |

```python
    @property
    def gap(self) -> float | None:
        """Validation minus training loss at the best step: how much worse unseen wording is."""
        return None if self.train_at_best is None else self.best_val - self.train_at_best

    @property
    def rise_after_best(self) -> float:
        """How much the validation loss rose from its best to the end (0 = never rose)."""
        return self.final_val - self.best_val
```
- 2つの**計算される値**(`@property`:かっこなしで使う。Day 8.1と同じ):
  - `gap` = 最良のステップでの、検証の損失 − 学習の損失(学習の損失が無ければ `None`);
  - `rise_after_best` = 最後 − 最良の検証の損失。
- 例:最良 2.60、学習 1.50 → gap 1.10;最後 2.90 → 上昇 0.30(最良の点の後で過学習した)。

```python
    def passes_checks(self) -> bool:
        if self.finished_rate is None or self.repetitive_rate is None:
            return False
        return self.finished_rate >= MIN_FINISHED and self.repetitive_rate <= MAX_REPETITIVE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gap"], d["rise_after_best"] = self.gap, self.rise_after_best
        return d
```
- `passes_checks` — **ルールの手順1**:生成の確認がまだなら(割合が `None`)→ **不合格**(確認なし = 候補ではない)。そうでなければ:完結 90%以上、**かつ**繰り返し 5%以下。
- `to_dict` — まとめを辞書として、**さらに**2つの計算された値を足す(プロパティなので、`asdict` には入らない)。

---

## 4. `summarize_run`

```python
def _f(x: str) -> float | None:
    return float(x) if x not in ("", None) else None
```
- 小さな補助:文字 → 数。空のセル(CSVのセルは文字で、空は「値なし」)は `None`。

```python
def summarize_run(log_path: Path, name: str, size: str, dropout: float, parameters: int) -> RunSummary:
    with log_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    evals = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["event"] == "eval" and r["val_loss"]]
    if not evals:
        raise ValueError(f"{log_path}: no validation results")
    trains = {int(r["step"]): float(r["train_loss"]) for r in rows if r["event"] == "train" and r["train_loss"]}
```
- 学習の `log.csv`(Day 9.2の形式)を読む。
- `evals` — すべての **eval** の行の `(ステップ, 検証の損失)` のリスト。評価が1つも無ければ `ValueError`。
- `trains` — **train** の行から作った、**ステップ → 学習の損失**の辞書(Day 9のcompareスクリプトと同じ)。

```python
    best_step, best_val = min(evals, key=lambda e: (e[1], e[0]))
    final_step, final_val = evals[-1]
```
- **最良の点:**損失(`e[1]`)が**最低**の評価。同じなら**早いステップ**(`e[0]`)。キー `(損失, ステップ)` は組として比べられる。
- **最後の点:**最後の評価(`evals[-1]`)。

```python
    tps = sorted(v for v in (_f(r["tokens_per_sec"]) for r in rows if r["event"] == "train") if v)
    vram = [v for v in (_f(r["peak_vram_mb"]) for r in rows) if v is not None]
    elapsed = max((float(r["elapsed_s"]) for r in rows if r["elapsed_s"]), default=0.0)
```
- `tps` — train の行の速さ(トークン/秒)で、**空や0を除き**(`if v`)並べ替えたもの — **中央値**を取れるように。
- `vram` — 存在するGPUメモリの値すべて(CPUの学習には無い)。
- `elapsed` — ログの中で一番大きい経過時間(= 合計時間)。無ければ `0.0`。

```python
    return RunSummary(
        name=name, size=size, dropout=dropout, parameters=parameters, best_val=best_val, best_step=best_step,
        train_at_best=trains.get(best_step), final_val=final_val, final_step=final_step,
        tokens_per_sec=tps[len(tps) // 2] if tps else None, peak_vram_mb=max(vram) if vram else None,
        elapsed_s=elapsed,
    )
```
- まとめを作る:**最良のステップ**での学習の損失(`trains.get(best_step)`、無ければ `None`)、**速さの中央値**(`tps[len(tps)//2]` = 真ん中の項目)、メモリの**最大値**(`max`)、経過時間。

---

## 5. `Decision` と `choose` — コードになったルール

```python
@dataclass
class Decision:
    chosen: RunSummary | None
    lowest_val: float | None
    as_good: list[str]
    rejected: dict[str, str]
    reason: str
```
- 結果:**選ばれた学習**(または `None`)、候補の中の検証の損失の最低値、「同じくらい良い」学習の名前、**却下された**学習とその理由、そして**決定を説明する1文**。

```python
def choose(runs: list[RunSummary], tolerance: float = TOLERANCE) -> Decision:
    rejected = {r.name: (f"finished {r.finished_rate:.0%}, repetitive {r.repetitive_rate:.0%}"
                         if r.finished_rate is not None else "no generation check")
                for r in runs if not r.passes_checks()}
    candidates = [r for r in runs if r.passes_checks()]
    if not candidates:
        return Decision(None, None, [], rejected, "no run passed the task 6 automatic guidelines")
```
- **手順1(候補):**`rejected` — 内包表記で作る辞書:確認に不合格のすべての学習について、その名前と短い理由(`:.0%` は割合をパーセントにする。例:`finished 80%, repetitive 10%`、または「no generation check」)。`candidates` — 合格した学習。
- **誰も**合格しなければ → 選ばない(`chosen = None`)。説明の理由付き。(運転手は「No run chosen」と書き、決定のファイルは書かない。)

```python
    lowest = min(r.best_val for r in candidates)
    as_good = [r for r in candidates if r.best_val - lowest <= tolerance]
    chosen = min(as_good, key=lambda r: (r.parameters, r.best_val, r.dropout))
```
- **手順2:**候補の中の、検証の損失の最低値。
- **手順3:**`as_good` = 損失が最低値より**高くても `tolerance` 以内**の候補(`<=` なので、ちょうど0.02も入る)。
- **手順4:**`min(..., key=(parameters, best_val, dropout))` — **パラメータを最初に**比べ(少ないほうが勝つ)、次に**検証の損失の低い方**、次に**ドロップアウトの低い方**(損失まで厳密に同じなら、ドロップアウトなしの、単純なほう)。

```python
    best_overall = min(candidates, key=lambda r: r.best_val)
    if chosen is best_overall:
        reason = f"{chosen.name} has the lowest validation loss ({chosen.best_val:.4f})"
        if len(as_good) > 1:
            reason += " and is also the smallest of the runs within the tolerance"
    else:
        reason = (f"{best_overall.name} has the lowest validation loss ({best_overall.best_val:.4f}), but "
                  f"{chosen.name} ({chosen.best_val:.4f}) is within {tolerance} of it with fewer parameters "
                  f"({chosen.parameters:,} vs {best_overall.parameters:,}), so the smaller model is chosen")
    return Decision(chosen, lowest, [r.name for r in as_good], rejected, reason)
```
- **説明の文**を作る:選ばれた学習が、損失の一番低いものでもあれば、そう言う(そして、ほぼ同じ学習が複数あったなら、その中で一番小さいとも)。そうでなければ **「X の損失が最低だが、Y は0.02以内で、パラメータが少ないので、小さいモデルを選ぶ」** と説明する(`chosen is best_overall` は*同じオブジェクト*かを比べる)。
- `Decision` を返す。この文はそのまま報告書に入る — 教員のレビューのための、できあいの根拠。

---

## 6. 知っておくと良い小さなこと

- **ルールそのものが大事。**結果の*前に*書かれるので、好みに合わせて曲げられない。報告書は、ルールを結果の隣に表示する。
- **損失が一番良い学習でも、却下されることがある**(手順1)。例:損失2.50だが、文を80%しか終えない。
- **0.02の許容範囲は仮定**で、docstringに書かれている(「実行ごとのゆれの範囲」)。運転手は、**各設定を1回だけ(シード1つ)**学習するので、ゆれはここでは測られない。0.02は、妥当で、控えめな推測。
- **`gap` は、学習の損失の*区間平均*を使う**(Day 9.2は、直近の `log_every` ステップの平均を記録し、dropout > 0ならdropoutがかかった状態)ので、丸暗記のおおよその尺度であり、学習どうしを比べるには良いが、厳密な数ではない。
- **ドロップアウトは、損失を通してだけ効く**:ルールは、最後の同点の決め方を除いて、ドロップアウトを好んだり避けたりしない。
- **テスト用の分割はどこでも使われない** — 選ぶのは検証の損失だけ。
- **割合が `None` のとき `passes_checks` が `False` を返す**のはわざと:生成の確認を忘れても、学習が通り抜けることは決してない。

## 7. 一段落でまとめ

`selection.py` は、学習のログをまとめ(最良の検証の損失とステップ、そのときの学習の損失、差、最良の後の上昇、速さ、メモリ)、前もって書いたルールを適用します:一番良いチェックポイントが、文の90%以上を完結し、繰り返しが5%以下の学習だけを残し、検証の損失が最低値から0.02以内のものを同じくらい良いとみなし、パラメータが一番少ないものを選びます。選んだものを、分かりやすい言葉の理由と一緒に返します。
