# Day 3(自宅ノートPC): 質問コーパスを生成し、動画で記録する

**マシン:** 自宅ノートPC。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**学校の課題:** 5(学習用の文章を作る)。

## 今の Day 3 の内容と、その理由

最初の Day 3(本と法令を集める)は、コーパスを質問文だけにしたので取りやめました。今の Day 3 は **生成そのもの** です。チームの生成プログラム `make_questions.py` を実行し、169,727 の質問を作ります。自分で生成した証拠として、その様子を動画で記録します。

| Day | 場所 | 仕事 | 証拠 |
| --- | --- | --- | --- |
| **Day 3** | 自宅(または学校) | 質問を生成する | 動画、実行ログ、`questions.csv` の SHA-256、zip |
| **Day 4** | 学校 | 学校PCで作り直し、Day 3 の zip と比べる | `byte-identical=True`、実行ログ、報告 |

2つを合わせると、質問がこのプログラムとこの入力ファイルから作られたこと、そして2台の別々のマシンがまったく同じ結果を出すことを示せます。

**自宅で実行する理由について:** 生成は数秒しかかかりません(ノートPCで 2.7 秒)。だから速さは理由になりません。聞かれたら本当の理由を答えてください: チームのファイルを使ったデータ作業を自宅で行い、Day 4 で学校PCでも同じ結果になることを確かめた、ということです。実行ログには実際の時間が記録されるので、説明もそれに合っている必要があります。

### 学校PCで使う場合の注意

コードは同じです。学校で Day 3 を実行するなら `--machine school` を使い、学校の `.venv` に pandas と openpyxl が必要です(Day 4 学校編の Step 1〜3 を参照)。`data/source/` や `runs/` を、公開されている学校のリポジトリにコミットしないでください。

---

## Step 1. 新しいファイルを置く

1. このチャットから `day3_home.zip` をダウンロードして展開する。
2. 展開した `day3_home` フォルダーの **中身をすべて** `llm_home` にコピーし、聞かれたら **置き換える / Replace** を選ぶ。中身:
   - `scripts/day3_generate_questions.py`(新規)
   - `scripts/day4_generate_corpus.py`(更新: `--reference` で Day 3 の zip と比べられるようになり、「チームのコーパス」ではなく「基準のコーパス」と表示する)
   - `docs/流れの説明/EN` と `JA` の、Day 3 と Day 4(学校)の手順書
3. 生成に必要なものは、Day 4 のときから `data\source\` にすべてあります(`DB.xlsx` も含む)。pandas と openpyxl も、自宅の `.venv` にもう入っています。

## Step 2. 録画せずに一度リハーサルする(VS Codeターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home --no-log
```

`--no-log` を付けると、リハーサルは `run_log.md` に記録されません。最後が次のようになるか確認します:

```text
Questions: 169,727 | types: 60 | patterns with questions: 300 | avg chars: 23.51
Check: PASS (no duplicates, no unfilled slots, 60 question types)
questions.csv SHA-256: d9ff2a6a6708f4af81ba3cf8c7db17d88d4112c4e823906f4853ebd1d6d62586
```

この SHA-256 は、チームの `questions.csv` のハッシュです。Day 4 で、あなたのノートPCがそれを1バイトも違わずに作ることが分かっているので、完全に一致するはずです。一致しなければ、止めて出力を私に送ってください。

リハーサルで `runs\day3_generation\` にフォルダーができます。消しても残しても構いません(Git は無視します)。

## Step 3. 本番を録画する

1. **画面の準備:** ブラウザー、チャットのアプリ、個人的なものを閉じる。ポップアップが出ないように **応答不可 / Do not disturb**(Windows の通知の設定)をオンにする。
2. **録画を始める:** **Snipping Tool** を開き、**録画 / Record** を選び、VS Code のウィンドウを選んで **スタート / Start** を押す。Xbox Game Bar(**Win + Alt + R**)でもよく、作業中のウィンドウを録画します。
3. **VS Codeターミナルで、次を1つずつ実行する。** それぞれの出力が読めるくらい、ゆっくり進めます:
   ```powershell
   Get-Date
   hostname
   git log -1 --oneline
   dir data\source
   .venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home
   ```
   - `Get-Date` と `hostname` は、いつ、どのPCで実行したかを示します。
   - `git log -1 --oneline` は、フォルダーにあるコードの版を示します。
   - `dir data\source` は、入力ファイル(チームの用語集・同義語辞書・文型・表)を示します。
   - 最後の行が本番の実行です。`Appended an entry to run_log.md` が出るまで待ちます。
4. **結果を見せる:**
   - **質問の例10個** と `Check: PASS` の行を、ゆっくり上にスクロールして見せる。
   - エクスプローラーで `runs\day3_generation\<時刻>\questions.csv` を開き(VS Code ではテキストで表示される)、数百行スクロールする。
   - `run_log.md` を開き、一番下の新しい `Day 3 question generation` のエントリーを見せる。
5. **録画を止める。** 動画は `C:\Users\Merrick\Videos\Captures`(または `ビデオ\キャプチャ`)に保存されます。
6. **動画はリポジトリの外に保存する**(SSD やクラウドなど)。動画は大きいので Git には入れません。

3〜5分あれば十分です。早送りしたり、途中を切ったりしないでください。切っていない録画のほうが良い証拠になります。

## Step 4. 学校に持っていくものを準備する

`runs\day3_generation\<時刻>\` に次ができています:

| ファイル | 使い道 |
| --- | --- |
| `questions_corpus_home_<時刻>.zip` | **USB で学校に持っていく**。Day 4 で学校の結果と比べる |
| `questions.csv` | 生成した質問 |
| `generation_info.json` | 設定、すべての入力と `questions.csv` の SHA-256、時間 |

USB メモリにコピーするもの:
- `runs\day3_generation\<時刻>\questions_corpus_home_<時刻>.zip`
- `data\source\` フォルダー全体(チームのファイルと `DB.xlsx`)
- `scripts\day3_generate_questions.py` と `scripts\day4_generate_corpus.py`
- `requirements.txt`(pandas と openpyxl が書いてある)

**これらを公開されている学校のリポジトリ経由で送らないでください。** `data\source\` と zip には、チームの表と内部用のラベルが入っています。USB だけで運びます。

## Step 5. 自宅でコミットしてプッシュする(VS Codeターミナル、`llm_home`)

```powershell
git status --short
git add scripts run_log.md docs
git status --short
git commit -m "Day 3 home: question generation script, run and walkthroughs"
git push
```

一覧に `runs/` が出てはいけません(生成したファイルは無視されている)。

## レビューで話すポイント

- **Day 3 が証明すること:** 169,727 の質問は、チームのファイルからチームのプログラムで作られ、それを自分で実行した。動画が実行の様子を示し、実行ログと `generation_info.json` が入力、設定、出力のハッシュを記録している。
- **出力のハッシュが大事な理由:** SHA-256 のハッシュは、1文字でも変わるとまったく違う値になります。Day 4 で学校PCが同じハッシュを出せば、2つのコーパスは「似ている」のではなく「同一」です。
- **実行が速い理由:** 生成プログラムは、文型に言葉を当てはめて重複を取り除くだけです。CPU には軽い仕事で、GPU は要りません。
- **入力を別のフォルダーにコピーする理由:** 生成プログラムは、実行した場所に出力を書きます。別のフォルダーにすれば、チームのファイルは変わらず、実行ごとの結果もまとまります。
- **動画を切らない理由:** 日付、PC名、コードの版が見える切れ目のない録画は、疑いにくい証拠です。

## Day 3 の終わりのチェックリスト

- [ ] リハーサルが成功し、SHA-256 `d9ff2a6a6708…` が出た
- [ ] 本番を切らずに録画した: `Get-Date`、`hostname`、`git log`、`dir`、実行、質問、実行ログ
- [ ] 動画をリポジトリの外に保存した
- [ ] USB メモリに zip、`data\source\`、2つのスクリプト、`requirements.txt` を入れた
- [ ] コミットとプッシュをした。`runs/` は Git に入っていない

## 送り返すもの

1. 本番の実行のコンソール出力(テキストで)。
2. 学校で Day 4 を実行したら知らせ、その出力も送る。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題5】学習用の文章の作成(質問文の生成)
状況: 予定通り終わった
内容: チームの用語集・同義語辞書・文型・社員データの表をもとに、質問文を自動で作る
      プログラムを自宅のPCで実行し、大量の質問文を生成した。実行の様子は動画で記録し、
      入力と出力のハッシュ値を実行ログに残した。
証拠: 実行の動画、run_log.md、runs/day3_generation/ の generation_info.json
次: 学校PCで同じプログラムを実行し、同じ結果になることを確認する(Day 4)
```
