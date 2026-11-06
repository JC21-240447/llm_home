# Day 4(学校PC): 質問を作り直し、結果が同一であることを証明する

**マシン:** 学校PC(ログイン `240447`)。**リポジトリ:** `JC21-240447/2026_llm_jk3a`(**公開**)。**フォルダー:** `C:\Users\240447\Desktop\LLM\phase1_day1`。
**学校の課題:** 5。

## 学校での Day 4 の内容と、その理由

Day 3 で、自宅で 169,727 の質問を生成しました。今日は学校PCで、**同じ** プログラムを **同じ** 入力ファイルで実行し、その結果を Day 3 の zip と比べます。1バイトも違わなければ、質問がこのプログラムとこのファイルから作られたことを、学校PCの上で、誰の前でも証明できます。

CPU だけを使い、数秒で終わるので、学校PCで問題ありません。

**先生方に約束したルールは今も有効です:** インストールは `.venv` の中だけ。`py -m pip install` は使わない。管理者権限は使わない。新しい Python は入れない。

---

## Step 1. USB メモリからファイルをコピーする

`C:\Users\240447\Desktop\LLM\phase1_day1` の中に置きます:

| USB メモリから | 置く場所 |
| --- | --- |
| `source` フォルダー全体(チームのファイルと `DB.xlsx`) | `data\source\` |
| `questions_corpus_home_<時刻>.zip` | `data\source\` |
| `day3_generate_questions.py`、`day4_generate_corpus.py` | `scripts\` |
| `requirements.txt` | リポジトリのルート(古いものを置き換える) |

## Step 2. チームのデータを公開リポジトリに入れないようにする(`git add` の前に必ず行う)

VS Code で `.gitignore` を開き、最後に次の行を追加します:

```text
# Team data files: shared by USB only, never pushed to this public repository
data/source/
```

`runs/*/` はもう `.gitignore` にあるので、生成したファイルも無視されます。確認します(VS Codeターミナル、`phase1_day1`):

```powershell
git check-ignore -v data\source\DB.xlsx
```

`.gitignore` と `data/source/` を含む1行が表示されなければなりません。何も出なければ、行が保存されていません。先に直してください。

## Step 3. 学校の `.venv` に pandas と openpyxl を入れる(VS Codeターミナル、`phase1_day1`)

```powershell
.venv\Scripts\python.exe -m pip install pandas openpyxl
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding ascii requirements_lock_school.txt
```

- `.venv` の中だけに入れます。吉澤先生と阿保先生に伝えたとおりです。
- `pip check` は `No broken requirements found.` と出なければなりません。
- 最後の行でロックファイルを更新し、新しいバージョンを記録します(pandas は Python 3.11 以上に対応しているので、学校の 3.12 で問題ありません)。

学校のネットワークが遅くても、ダウンロードは小さいです(一番大きい pandas でも数十 MB で、GB ではありません)。

## Step 4. 自宅の zip と比べて Day 4 を実行する(VS Codeターミナル、`phase1_day1`)

`<時刻>` は、実際の zip の名前に置き換えます:

```powershell
.venv\Scripts\python.exe scripts\day4_generate_corpus.py --machine school --reference data\source\questions_corpus_home_<時刻>.zip
```

期待する出力(大事な行):

```text
Questions: 169,727 | types: 60 | patterns with questions: 300 | avg chars: 23.51
Corpus check: PASS (no duplicates, no broken sentences, labels readable)
Same as the reference corpus (questions_corpus_home_<時刻>.zip): byte-identical=True, identical patterns=300, differing patterns=none
patterns.csv: identical
question_types.csv: identical
columns.csv: identical
```

`byte-identical=False` と出たら、何も変えずに、出力をすべて私に送ってください。USB メモリの入力ファイルのどれかが、自宅で使ったものと違うということです。

出力のスクリーンショットを撮ります。これが教授に見せる証拠です。

**任意:** 学校PCで生成そのものも教授に見せたい場合は、学校で Day 3 も実行できます(これも数秒です):

```powershell
.venv\Scripts\python.exe scripts\day3_generate_questions.py --machine school
```

## Step 5. 学校でコミットしてプッシュする(VS Codeターミナル、`phase1_day1`)

```powershell
git status --short
```

追加する前によく確認します: **`data/source/` の中のものと `runs/` の中のものは、1つも出てはいけません。** 出たら止めて、一覧を私に送ってください。

```powershell
git add .gitignore requirements.txt requirements_lock_school.txt run_log.md
git add scripts/day3_generate_questions.py scripts/day4_generate_corpus.py docs/DAY4_GENERATION_REPORT.md
git status --short
git commit -m "Day 4 school: regenerated the question corpus, identical to the home run"
git pull
git push
```

このリポジトリにはチームのメンバーもプッシュするので、`git push` の前に `git pull` が必要です。

報告 `docs/DAY4_GENERATION_REPORT.md` には、チームが考えた架空の名前を含む質問の例、入力のハッシュ、合計が入っていますが、内部用のラベルは入っていないので、公開リポジトリに入れて問題ありません。

## レビューで話すポイント

- **同じプログラム、同じ入力、同じ出力が2台のマシンで:** SHA-256 の比較で、コーパスが再現できることを証明する。これが正直な報告の土台になる。
- **データを公開リポジトリに入れない理由:** チームの表と内部用のラベルはプロジェクトのデータで、公開するものではない。公開するのはコードと報告だけ。
- **もう一度 `pip check` とロックファイルを行う理由:** 学校の環境が変わった(ライブラリが2つ増えた)ので、先生方が受け入れた証拠を新しくする。

## Day 4(学校)の終わりのチェックリスト

- [ ] USB メモリからファイルをコピーした
- [ ] `.gitignore` に `data/source/` を追加し、`git check-ignore` で確認した
- [ ] pandas と openpyxl を `.venv` の中だけに入れた。`pip check` OK。ロックファイルを更新した
- [ ] Day 4 の出力が `byte-identical=True`。スクリーンショットを撮った
- [ ] コミットとプッシュをした。`data/source/` や `runs/` のものは Git に入っていない

## 送り返すもの

1. Step 4 のコンソール出力すべて(テキストで)またはスクリーンショット。
2. Step 5 の `git status --short` の一覧(予想外のものがあった場合)。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題5】生成した質問文の確認(学校PCでの再現)
状況: 予定通り終わった
内容: 自宅で生成した質問文と同じプログラム・同じ入力を学校PCで実行し、生成結果が
      1バイトも違わず一致することを確認した。追加のライブラリは仮想環境の中だけに入れた。
証拠: docs/DAY4_GENERATION_REPORT.md、run_log.md、実行画面のスクリーンショット
次: 質問文を学習用・検証用・テスト用に分ける(Day 5)
```
