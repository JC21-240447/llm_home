# Day 6(学校PC): 自作のトークナイザー(バイト単位の BPE)を学習する

**マシン:** 学校PC(ログイン `240447`)。**リポジトリ:** `JC21-240447/2026_llm_jk3a`(**公開**)。**フォルダー:** `C:\Users\240447\Desktop\LLM\phase1_day1`。
**学校の課題:** 8。

コードは自宅と **同じ** です。トークナイザーの仕組み、語彙が16,000ではなく1,674である理由、レビューで話すポイントは、`DAY06_HOME_WALKTHROUGH` を読んでください。学校で違う点:
- ファイルは SSD で運ぶ。
- **先に学校で Day 5 を済ませる。** トークナイザーは `data\corpus\train.txt` で学習し、このファイルは学校PCの Day 5 で作られるため。
- `git push` の前に `git pull` する。

CPU だけを使い、学習は数秒です。

---

## Step 1. SSD からファイルをコピーする

自宅で SSD(`to_school` フォルダー)に入れるもの: `src\bpe_tokenizer.py`、`src\__init__.py`、`scripts\day6_train_bpe.py`、`tests\test_bpe_tokenizer.py`(手順書がほしければ `docs\流れの説明\` も)。

学校で(VS Codeターミナル、`phase1_day1`。SSD の実際のドライブ文字を使う):

```powershell
$SSD = "E:\to_school"
New-Item -ItemType Directory -Force src, tests | Out-Null
Copy-Item "$SSD\src\bpe_tokenizer.py", "$SSD\src\__init__.py" src\
Copy-Item "$SSD\scripts\day6_train_bpe.py" scripts\
Copy-Item "$SSD\tests\test_bpe_tokenizer.py" tests\
dir data\corpus\train.txt, data\corpus\val.txt
```

最後のコマンドで2つのファイルが表示されなければなりません。表示されなければ、先に学校で Day 5 を行います。

## Step 2. テスト、学習、もう一度テスト(VS Codeターミナル、`phase1_day1`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine school
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果:
- 1回目の pytest: `29 passed, 1 skipped`
- 学習: 743種類の部分、1,314回の結合、1,674を選択、`tokenizer.json SHA-256: 53b03761cd1b614972e1b9bfd63d0a66287d6fd069cd0a9fd91419ec249ca94b`(**自宅と同じ**)
- 2回目の pytest: `30 passed`

SHA-256 が同じなら、学校PCがまったく同じトークナイザーを作ったことの証明になります。スクリーンショットを撮ります。違ったら、コミットせずに出力を私に送ってください。

## Step 3. 決めたことを記録する

自宅用の手順書(Step 6)と同じ **D6** の文を `docs\DECISIONS.md` に追加します。

## Step 4. コミットしてプッシュする(VS Codeターミナル、`phase1_day1`)

```powershell
git status --short
```

確認: `data/corpus/`、`data/source/`、`runs/` の中のものは、1つも出てはいけません。

```powershell
git add src scripts/day6_train_bpe.py tests data/tokenizer docs/DAY6_TOKENIZER_REPORT.md docs/DECISIONS.md run_log.md
git status --short
git commit -m "Day 6 school: own byte-level BPE tokenizer"
git pull
git push
```

公開されるもの: トークナイザーのコード、テスト、報告、トークナイザーのファイル。トークナイザーのファイルには、質問文から学んだ部分(言葉の断片と、チームが考えた架空の名前)が入っていますが、質問文そのものや内部用のラベルは入っていません。

## Day 6(学校)の終わりのチェックリスト

- [ ] 学校で Day 5 を済ませ、SSD からファイルをコピーした
- [ ] pytest `29 passed, 1 skipped`、その後 `30 passed`
- [ ] SHA-256 `53b03761cd1b…` が自宅と同じ
- [ ] D6 を追加し、コミットとプッシュをした。`data/corpus/`、`data/source/`、`runs/` のものは Git に入っていない

## 送り返すもの

学習のコンソール出力(テキストで)と、2つの pytest の結果の行。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題8】文章を細かく分けるプログラム(トークナイザー)
状況: 予定通り終わった
内容: 学校PCで、自作のバイト単位BPEトークナイザーを学習用の質問文で学習した。語彙の大きさは
      候補ごとに検証用データで測り、決めたルールで選んだ(約1,700)。保存と読み込み、
      文→番号→文の往復が完全に一致することをテストで確認した。
証拠: data/tokenizer/tokenizer.json、docs/DAY6_TOKENIZER_REPORT.md、run_log.md、tests/
次: 課題8・9のデモ(文をトークンと番号に分けて戻す)とコーパスの符号化(Day 7)
```
