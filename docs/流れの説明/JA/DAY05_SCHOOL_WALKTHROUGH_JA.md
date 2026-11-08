# Day 5(学校PC): 質問コーパスを文型ごとに分割し、データカードを書く

**マシン:** 学校PC(ログイン `240447`)。**リポジトリ:** `JC21-240447/2026_llm_jk3a`(**公開**)。**フォルダー:** `C:\Users\240447\Desktop\LLM\phase1_day1`。
**学校の課題:** 5。

コードは自宅と **同じ** です: `scripts\day5_build_corpus.py` を `--machine school` で実行します。学校で違う点:
- リポジトリが公開されているので、生成したコーパス(`data/corpus/`、内部用のラベルも入っている)を、先に `.gitignore` に追加する。
- 学校の `LICENSES_AND_SOURCES.md` はまだ古い版(本と法令)なので、置き換える。
- メンバーもこのリポジトリにプッシュするので、`git push` の前に `git pull` する。

分割の理由(なぜ文型ごとか、40%のルール、上限、オーバーサンプリング)は、自宅用の手順書 `DAY05_HOME_WALKTHROUGH` を読んでください。そこに書いたことは学校でもすべて当てはまります。

**先に学校で Day 3 と Day 4 を済ませてください。** Day 5 は、Day 3 と Day 4 のためにコピーした `data\source\questions_corpus.zip` と `data\source\patterns.csv` を読みます。

---

## Step 1. SSD からファイルをコピーする

自宅で SSD(`to_school` フォルダー)に入れるもの:
- `scripts\day5_build_corpus.py`
- `LICENSES_AND_SOURCES.md`
- `docs\流れの説明\`(任意。手順書)

学校で(VS Codeターミナル、`phase1_day1`。SSD の実際のドライブ文字を使う):

```powershell
$SSD = "E:\to_school"
Copy-Item "$SSD\scripts\day5_build_corpus.py" scripts\
Copy-Item "$SSD\LICENSES_AND_SOURCES.md" .
dir data\source\questions_corpus.zip, data\source\patterns.csv
```

最後のコマンドで2つのファイルが表示されなければなりません。表示されなければ、先に学校で Day 3 と Day 4 を行います。

## Step 2. 生成したコーパスを公開リポジトリに入れないようにする(何かを実行する前に)

`.gitignore` を開き、最後に(Day 4 で追加した `data/source/` の行の下に)次を追加します:

```text
# Day 5: generated corpus files with internal labels (rebuild with scripts/day5_build_corpus.py)
data/corpus/
```

確認します:

```powershell
git check-ignore -v data\corpus\train.txt
```

`data/corpus/` を含む1行が表示されなければなりません。

## Step 3. 分割を実行する(VS Codeターミナル、`phase1_day1`)

```powershell
.venv\Scripts\python.exe scripts\day5_build_corpus.py --machine school
```

期待する出力(自宅とまったく同じ数値):

```text
Loaded 169,727 questions, 300 patterns with questions, 60 types.
Patterns with no questions (...): ['P171']
split      lines    unique       chars              token bounds  patterns  types
train    148,860   147,317   3,482,192    1,646,022 - 10,126,458       258     60
val        4,517     4,517      95,987          42,388 - 279,399        21     21
test       4,553     4,553     109,720          51,015 - 320,290        21     21
Near-duplicates removed: {'val': 0, 'test': 0}; over eval cap: {'val': 5700, 'test': 7640}
Split check: PASS (no shared patterns, no near-duplicate leaks)
```

次に `data\splits\split_manifest.json` を開き、`outputs_sha256` を確認します:

| ファイル | SHA-256 の先頭 |
| --- | --- |
| `data/corpus/train.txt` | `21d5d2900c71` |
| `data/corpus/val.txt` | `bc5a13145d1b` |
| `data/corpus/test.txt` | `c0c99ebdd755` |

自宅と同じハッシュなら、学校PCがまったく同じ学習用・検証用・テスト用のファイルを作ったことの証明になります。出力と、この3行のスクリーンショットを撮ります。

何かが違ったら、コミットしないでください。出力を私に送ってください。

## Step 4. 決めたことを記録する

自宅用の手順書(Step 4)と同じ **D4** と **D5** の文を `docs\DECISIONS.md` の最後に、同じ1行を `README.md` の「Data and licenses」の章に追加します。

## Step 5. コミットしてプッシュする(VS Codeターミナル、`phase1_day1`)

```powershell
git status --short
```

確認: **`data/corpus/`、`data/source/`、`runs/` の中のものは、1つも出てはいけません。** 出たら止めて、一覧を私に送ってください。

```powershell
git add .gitignore LICENSES_AND_SOURCES.md README.md run_log.md
git add scripts/day5_build_corpus.py data/splits docs/DATA_CARD.md docs/DECISIONS.md
git status --short
git commit -m "Day 5 school: split by pattern, data card"
git pull
git push
```

公開されるものと、それで問題ない理由: スクリプト、マニフェスト(文型ID、設定、件数、ハッシュ)、データカード(件数と設計。ラベルは無い)、決めたこと。質問文、ラベル、チームの表は、学校PCと SSD にだけ残ります。

## Day 5(学校)の終わりのチェックリスト

- [ ] 学校で Day 3 と Day 4 を済ませた
- [ ] `.gitignore` に `data/corpus/` を追加し、`git check-ignore` で確認した
- [ ] 出力の数値が一致し、`Split check: PASS`。3つのハッシュが表と一致した
- [ ] D4、D5 と README の1行を追加した
- [ ] コミットとプッシュをした。`data/corpus/`、`data/source/`、`runs/` のものは Git に入っていない

## 送り返すもの

コンソール出力すべて(テキストで)と、`outputs_sha256` の行。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題5】文章データ(コーパス)の準備
状況: 予定通り終わった
内容: 学校PCで、生成した大量の質問文を文型ごとに学習用・検証用・テスト用に分けた。
      同じ文型の文がテストに混ざらないようにし、一度も見ていない言い回しで評価できる
      ようにした。データカードを作成した。
証拠: docs/DATA_CARD.md、data/splits/split_manifest.json、run_log.md
次: 課題8(文章を細かく分けるプログラム = トークナイザー)
```
