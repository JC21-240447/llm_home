# Day 5(自宅ノートPC): 質問コーパスを文型ごとに分割し、データカードを書く

**マシン:** 自宅ノートPC。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**学校の課題:** 5(文章データとコーパス)。

## 何が変わったか、なぜか

チームが別のチャットでデータの作業を終えたので、計画が変わりました(引き継ぎファイルを参照):

| 手順書の Day | 現在 |
| --- | --- |
| Day 2: 用語集と同義語のファイル | **チームが完成**: 551語、1,705の言い換え |
| Day 3: 本と法令 | **取りやめ**: コーパスは質問文だけ |
| Day 4: 合成テキストの生成 | **チームが完成**: 60の質問タイプ、301の文型、169,727の質問 |
| **Day 5: 分割とデータカード** | **今日** |

今日の仕事は、チームのコーパスを、自分をだますことなく **学習用(train)・検証用(val)・テスト用(test)** のファイルに分けることです。

一番大事なルール: **行ではなく、文型ごとに分ける。** 文型とは「{属性}の社員を一覧で見せてください。」のような文の型です。同じ文型の質問は、入る言葉が違うだけで、ほとんど同じ文です。行をランダムに分けると、ほぼ同じ文が学習用とテスト用の両方に入り、テストの点数が「間違った理由で」良く見えてしまいます。そこで **文型を丸ごと** 取り分けます。モデルは、知っている質問タイプについて、一度も見たことのない言い回しでテストされます。

### スクリプトが決めていること、その理由

| ルール | 値 | 理由 |
| --- | --- | --- |
| 取り分ける文型 | 検証用 21、テスト用 21(300の約7% + 7%) | 測るのに十分な言い回しを確保しつつ、学習用にデータの大半を残す |
| 1タイプから1文型。検証用とテスト用のタイプは別 | 60タイプ中42タイプに取り分けた言い回しがある | 多くのタイプを評価できる |
| 文型を出せるのは、質問が100以上かつ文型が3つ以上あるタイプだけ | | 小さなタイプは、すべての言い回しを学習に使う |
| 取り分ける文型は、そのタイプの質問の40%以下 | | このルールが無いと、T50 は学習用16行・テスト用256行になっていた(テスト中に私が見つけた) |
| 取り分けた文型ごとの上限 | 500問 | 1つの大きな文型が点数を支配しないように。上限を超えた行は捨て、学習用には移さない(移すと漏れになる) |
| ほぼ重複の除去 | 句読点と空白を無視して同じ文 | 「…ですか。」と「…ですか？」が別の分割にまたがるのを防ぐ(結果: 0件) |
| オーバーサンプリング | 小さなタイプを約200行まで、最大5回、**学習用だけ** | 引き継ぎファイルの「数回まで、何百回ではない」に従う |
| シード | 42 | 同じ入力なら、いつもまったく同じ出力になる |

### チームのファイルを確認して分かった2つのこと

1. **P171 には質問が1つもありません。** P171 が作る文(例:「部門ごとの月額給与の最高額を教えてください。」)は、すべて **P026**(T06、集計 = 最高額)も作ります。生成スクリプトは最初のものだけを残すので、P171 の質問はすべて P026 として記録されています。学習に失うものはありませんが、同じ文に2つの質問タイプがあり得ることになります。どちらのタイプにするかはチームが決めることです(後での質問で、今日の作業は止めません)。
2. **169,727 の質問は、すべて自分の文型と一致しました**(「課長の社員」を「課長」にする規則を考慮した上で)。コーパスと `patterns.csv` は食い違っていません。

### 学校PC

Day 5 は CPU と普通の Python だけを使うので、学校PCでもコードは同じです。学校のリポジトリは公開されているので手順が違います。`DAY05_SCHOOL_WALKTHROUGH` に従ってください。

---

## Step 1. ファイルを置く

Day 5 が読むものは、Day 3 と Day 4 からすべて `data\source\` にあります。`.gitignore` と `LICENSES_AND_SOURCES.md` も Day 4 で更新済みです。新しいのはスクリプトと手順書だけです。

1. このチャットから `day5_v2.zip` をダウンロードして展開する。
2. 展開した `day5_v2` フォルダーの **中身をすべて** `llm_home` にコピーする:
   - `scripts\day5_build_corpus.py`
   - `docs\流れの説明\EN\` と `JA\`: Day 5 の自宅用と学校用の手順書
3. 確認する(VS Codeターミナル、`llm_home`):
   ```powershell
   dir data\source\questions_corpus.zip, data\source\patterns.csv
   git check-ignore -v data\corpus\train.txt
   ```
   1つ目のコマンドで2つのファイルが表示されること。2つ目で `data/corpus/` を含む1行が表示されること(生成したコーパスが Git に入らないという意味)。

## Step 2. 分割を実行する(VS Codeターミナル、フォルダーは `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day5_build_corpus.py --machine home
```

10秒ほどで終わります。次とまったく同じ数値が出るはずです(同じファイルで同じスクリプトを私が実行した結果です):

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

次に `data\splits\split_manifest.json` を開き、`outputs_sha256` を見ます。私のテスト(Python 3.11 と 3.13 で同じ結果)では、3つのテキストファイルのハッシュは次のとおりでした:

| ファイル | SHA-256 の先頭 |
| --- | --- |
| `data/corpus/train.txt` | `21d5d2900c71` |
| `data/corpus/val.txt` | `bc5a13145d1b` |
| `data/corpus/test.txt` | `c0c99ebdd755` |

一致すれば、あなたのファイルは私のものと1バイトも違わないということで、分割がデータとシードだけで決まることを示します。スクリプト自身も確認をしていて、取り分けた行が学習用に漏れていれば `Split check FAILED` で止まります。

上の数値と違ったら、先に進まず、出力を私に送ってください。

作られたもの:

| ファイル | 内容 | Git に入れるか |
| --- | --- | --- |
| `data/corpus/train.txt`、`val.txt`、`test.txt` | 1行に1問、文だけ。学習の入力 | 入れない(スクリプトで作り直せる) |
| `data/corpus/*_labels.csv` | 同じ行に文型、タイプ、意味 を付けたもの。内部用 | 入れない |
| `data/splits/split_manifest.json` | どの文型をどこに分けたか、設定、入力と出力の SHA-256 | **入れる** |
| `docs/DATA_CARD.md` | データカード。数値はすべてスクリプトが書く | **入れる** |
| `run_log.md` | 新しい `Day 5 corpus split` のエントリー | **入れる** |

## Step 3. 結果を読む

1. VS Code で `docs/DATA_CARD.md` を開く(右クリック > **プレビューを開く / Open Preview** で表が見える)。
2. 6章(質問タイプごと)で、文型を取り分けたすべてのタイプについて、**学習用の行数が検証用とテスト用の合計より多い**ことを確認する。
3. `data\corpus\val.txt` を開き、20行ほど読む。自然な質問で、マニフェストの `patterns_per_split.val` にある文型IDの言い回しが `data\source\patterns.csv` で見つかるはずです。
4. トークン数の意味を理解する。まだトークナイザーが無い(Day 6)ので、スクリプトは正直な **範囲** を出します:
   - **下限(学習用で約165万):** 予定している前処理は、文を同じ文字の種類(ひらがな、カタカナ、漢字、ラテン文字、数字、記号)の連続に分けます。BPE はこの区切りをまたいで結合しないので、トークンは少なくともこの数になります。
   - **上限(約1,010万):** UTF-8 のバイト数。BPE が何も学ばなかった場合の数です。
   - 本当の数は Day 7 で測ります。下限に近くなるはずです。

## Step 4. 決めたことを記録する

`docs/DECISIONS.md` の最後に、次を追加します:

```markdown
## D4. Training corpus (Day 5)

- The corpus is short Japanese questions about employee data only, made by the team's own generator
  from the team's glossary, synonym dictionary, patterns and a 50-row sample table.
- Books and laws (old Day 3) are dropped. General text may be reconsidered only if evaluation shows a need.
- Only the question text is used for training; labels are kept in separate files.

## D5. Split (Day 5)

- Split by pattern, not by line: 21 validation and 21 test patterns, one per question type, from different types.
- A type donates a pattern only if it has 100+ questions and 3+ patterns; the donated pattern holds at most 40% of its type.
- Held-out patterns are capped at 500 questions; lines over the cap are dropped, not moved to train.
- Small types are repeated in train only (up to about 200 lines, at most 5 copies). Seed 42.
- P171 has no questions: all its sentences are identical to P026 sentences. Open question for the team.
```

`README.md` の「Data and licenses」の章にも、次の1行を追加します:

```markdown
All training text is made by the team's own generator from the team's own files; see `docs/DATA_CARD.md`.
```

## Step 5. コミットしてプッシュする(VS Codeターミナル、`llm_home`)

```powershell
git status --short
```

追加する前に一覧を確認します。`data/corpus/` は **出てはいけません**(無視されている)。`.venv` の中身も出てはいけません。

```powershell
git add README.md run_log.md
git add scripts/day5_build_corpus.py data/splits docs
git status --short
git commit -m "Day 5 home: team data, split by pattern, data card"
git push
```

## レビューで話すポイント

- **なぜ文型ごとに分けるか:** 同じ文型の行は、ほぼコピーです。ランダムに分けると、一般化ではなく記憶をテストすることになります。言い回しを丸ごと取り分ければ、見たことのない言い方にモデルが対応できるかを試せます。
- **なぜ1タイプ1文型で、40%のルールがあるか:** 多くのタイプで評価したいが、学習でどのタイプも飢えさせたくない。スクリプトの最初の版は、あるタイプでこれを破っていました(学習用16行に対してテスト用256行)。このルールはそのテストから生まれたもので、実データで設計を確かめた良い例です。
- **なぜ上限を超えた行を学習用に移さず捨てるか:** 取り分けた言い回しを学習用に入れると、まさにテストしたいものが漏れてしまうから。
- **なぜオーバーサンプリングするか、なぜ少しだけか:** 5〜70問しか作れないタイプがあります。数回繰り返せば、モデルがそれらを無視しなくなります。何百回も繰り返すと丸暗記になります。
- **再現性:** 分割は入力ファイルとシードだけで決まります。マニフェストに SHA-256 を記録するので、学校PCとノートPCがまったく同じデータを使ったことを誰でも確かめられます。
- **ライセンス:** すべての行は、チーム自身のファイルから、チーム自身の生成スクリプトで作りました。学習に外部の文章は使っていません。
- **トークンの範囲:** 今は証明できる範囲を報告し、Day 7 で実測値に置き換えます。数値をでっち上げません。

## Day 5 の終わりのチェックリスト

- [ ] Day 5 のスクリプトをコピーし、`data/corpus/` が無視されることを確認した
- [ ] スクリプトの出力が Step 2 の数値と一致し、`Split check: PASS` が出た
- [ ] 3つの `.txt` のハッシュが Step 2 の表と一致した
- [ ] `docs/DATA_CARD.md` を読み、取り分けたすべてのタイプで学習用が検証用+テスト用より多い
- [ ] `docs/DECISIONS.md`(D4、D5)と README の1行を追加した
- [ ] コミットとプッシュをした。`data/corpus/` は Git に入っていない

## 送り返すもの

1. スクリプトのコンソール出力すべて(テキストで)。
2. `data/splits/split_manifest.json` の `outputs_sha256` の部分。
3. 私の未解決の質問への答え(チャットのメッセージで)。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

すでに提出した報告に合わせ、コーパスの大きさは正確な数ではなく、おおまかに書きます。

```
【課題5】文章データ(コーパス)の準備
状況: 予定通り終わった
内容: チームが作った用語集・同義語辞書・文型から作成した大量の質問文を、文型ごとに
      学習用・検証用・テスト用に分けた。同じ文型の文がテストに混ざらないようにし、
      一度も見ていない言い回しで評価できるようにした。データカードを作成した。
証拠: docs/DATA_CARD.md、data/splits/split_manifest.json、run_log.md
次: 課題8(文章を細かく分けるプログラム = トークナイザー)
```
