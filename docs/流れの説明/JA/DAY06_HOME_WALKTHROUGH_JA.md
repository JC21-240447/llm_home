# Day 6(自宅ノートPC): 自作のトークナイザー(バイト単位の BPE)を学習する

**マシン:** 自宅ノートPC。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**学校の課題:** 8(文章を細かく分けるプログラム: 文を入れると部分に分かれる)。課題8・9のデモは Day 7 で行います。

## Day 6 で行うこと、その理由

言語モデルは文字を読めません。読むのは **トークンID**、つまり文の部分を表す番号です。トークナイザーは、文をどう部分に切るか、各部分にどの番号を付けるかを決めます。今日はそれを自分で書き、Day 5 の学習用の文で学習させます。

仕組みは3段階です:

1. **前処理(プレトークナイザー)。** 文字の種類(ひらがな、カタカナ、漢字、ラテン文字、空白)が変わるところで文を切ります。数字と記号は常に1文字ずつです。
   `営業部の社員は何人ですか。` は `営業部 | の | 社員 | は | 何人 | ですか | 。` になります。
2. **バイト。** 各部分を UTF-8 のバイトにします(日本語の1文字は3バイト)。256通りのバイトすべてにトークンがあるので、見たことのない言葉、英語、記号、絵文字を含め、**どんな** 文でも符号化できます。「未知語」はありません。
3. **BPE の結合。** 学習では、隣り合うトークンの組をすべて数え、一番多い組を新しいトークンにまとめることを繰り返します。よく出る言葉は1つのトークンになり、まれな言葉はいくつかの小さなトークンのままです。

ID の並び(計画で決めたもので、変えていない):

| ID | 意味 |
| --- | --- |
| 0, 1, 2, 3 | `<pad>`、`<bos>`(始まり)、`<eos>`(終わり)、`<unk>` |
| 4〜103 | `<unused_0>`〜`<unused_99>`: 後で拡張するための予備のIDのまとまり |
| 104〜359 | 256個の1バイト |
| 360〜 | 学習した結合(学習した順) |

### 大きな発見: 語彙の大きさ

計画では約16,000トークンでした。データは、ここではそれが不可能だと示しています。169,727の質問は、たった **743種類の部分** (`営業部`、`の`、`社員`、`えてください` など)に分かれます。BPE は部分をまたいで結合できないので、学習は **1,314回結合したところで自動的に止まります**。それ以上くり返し出てくる組が無いからです。そのため、あり得る一番大きな語彙は **1,674** です。

結合は順番に学ぶので、**最初の N 個の結合が、そのまま大きさ N のトークナイザー** です。1回の学習で、すべての大きさを測れます:

| 語彙 | 検証用のトークン数 | 1〜3トークンの重要な用語 | 一番長い質問 |
| --- | --- | --- | --- |
| 400 | 168,640 | 16/232 | 79 |
| 800 | 72,159 | 168/232 | 45 |
| 1,200 | 51,888 | 200/232 | 26 |
| 1,600 | 47,129 | 225/232 | 25 |
| **1,674(選択)** | **47,123** | **232/232** | **25** |

**選ぶ前に書いておいたルール:** 1〜3トークンになり得る重要な用語(列名、値、検索語)がすべて1〜3トークンで、検証用の文に必要なトークン数が一番大きいトークナイザーより5%以上は増えない、一番小さい大きさ。1つ目の条件を満たすのは 1,674 だけです。

覚えておく数字があと2つ:
- **一番長い質問は25トークン** です(始まりと終わりを含めて27)。後で文脈の長さを決めるときの根拠になります。
- **学習は約2秒** です。自作の純粋な Python の実装で十分速いので、計画の予備案(既製のトークナイザーのライブラリ)は不要です。

### 「今日は晴れ」が示すこと

学校の課題の紙では「今日は晴れ」が例です。実際の出力はこうです:

```text
今日は晴れ -> <e4bb> / <8a> / 日 / は / <e699> / <b4> / <e382> / <8c>
```

今日 と 晴れ は、私たちの質問に **一度も出てきません**。だからトークナイザーにはそのトークンが無く、バイトのトークン(`<e4bb>` のように表示)で書き、それでも正確に元に戻します。これは正直に見せてください。トークナイザーが **私たちの** 語彙を学んだこと、そしてどんな文でも扱えることの両方を示しています。業務の文なら、学校の例と同じような分かれ方になります:

```text
営業部の社員の平均年収を教えてください。 -> 営業部 / の / 社員 / の / 平均年収 / を / 教 / えてください / 。
```

### 学校PC

コードは同じで、CPU のみです。学校での手順は `DAY06_SCHOOL_WALKTHROUGH` にあります。学校PCでも、`tokenizer.json` の SHA-256 が同じにならなければなりません。

---

## Step 1. ファイルを置く

1. このチャットから `day6.zip` をダウンロードして展開する。
2. 展開した `day6` フォルダーの **中身をすべて** `llm_home` にコピーする:
   - `src\bpe_tokenizer.py`: トークナイザー本体(これから先の Day すべてで使う)
   - `src\__init__.py`: スクリプトが `src` から読み込めるようにする空のファイル
   - `scripts\day6_train_bpe.py`: 学習、測定、選択、保存、報告
   - `tests\test_bpe_tokenizer.py`: 単体テスト
   - `docs\流れの説明\EN` と `JA`: Day 6 の手順書

## Step 2. 先に単体テストを実行する(VS Codeターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: `29 passed, 1 skipped`。スキップされる1つは本物のトークナイザーが必要なテストで、まだ存在しないためです。テストでは、ID の並び、前処理で何も失われないこと、学習が毎回同じになること、難しい文(絵文字、英語、タブ、改行、半角カタカナ)が完全に往復すること、保存と読み込みで同じ結果になることを確かめます。

## Step 3. 学習して選ぶ(VS Codeターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home
```

数秒で終わります。期待する出力(大事な行):

```text
Unique pre-token pieces: 743 (total 1,730,834)
Merges learned: 1,314 -> largest vocabulary 1,674 (most frequent pair occurs fewer than 2 times)
Key terms found in the training text: 279; achievable as 1-3 tokens: 232
Rule: smallest size with all achievable key terms at 1-3 tokens and validation within 5% of the largest -> chosen 1,674
Reload check: PASS (identical merges and encodings); round trip: PASS on 4,523 texts
tokenizer.json SHA-256: 53b03761cd1b614972e1b9bfd63d0a66287d6fd069cd0a9fd91419ec249ca94b
```

学習は毎回同じ結果になり、あなたの Day 5 のファイルは私のものと1バイトも違わないので、SHA-256 はこのとおりでなければなりません。違ったら、出力を私に送ってください。

作られるもの:

| ファイル | 内容 | Git に入れるか |
| --- | --- | --- |
| `data/tokenizer/tokenizer.json` | 選んだトークナイザー(1,674) | **入れる** |
| `data/tokenizer/tokenizer_full.json` | すべての結合(今回は同じ大きさ。別の大きさで保存し直すために残す) | **入れる** |
| `docs/DAY6_TOKENIZER_REPORT.md` | 表全体、ルール、デモ、ハッシュ | **入れる** |
| `run_log.md` | `Day 6 tokenizer` のエントリー | **入れる** |

## Step 4. もう一度テストを実行する

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

今度の期待する結果: `30 passed`。最後のテストは、本物のトークナイザーで検証用の4,517問すべてを符号化して復号し、1つずつ完全に元に戻ることを確かめます。

## Step 5. 報告を読む

`docs\DAY6_TOKENIZER_REPORT.md` をプレビューで開き、3つ確認します:
1. 大きさの表: トークン数は約1,400まで急に減り、その後はほとんど変わらない。これが「小さな語彙」の根拠。
2. 「Achievable key terms longer than 3 tokens at the chosen size: none」。
3. 3トークンより長いままの重要な用語6つと、その理由: いくつかの部分からなる句(数が多い順)か、長い言葉の中にしか出てこない言葉(勤続 は 勤続年数 の中にしか出ない)。

## Step 6. 決めたことを記録する

`docs\DECISIONS.md` の最後に追加します:

```markdown
## D6. Tokenizer (Day 6)

- Own byte-level BPE with a script-class pre-tokenizer (digits and punctuation always single pieces). No tokenizer library.
- Fixed IDs: <pad>=0, <bos>=1, <eos>=2, <unk>=3, <unused_0..99>=4-103, bytes=104-359, merges from 360.
- Vocabulary 1,674 instead of the planned ~16,000: the training text has only 743 unique pieces, so training stops
  after 1,314 merges. Chosen by a written rule (all achievable key terms 1-3 tokens, validation within 5% of the
  largest size); see docs/DAY6_TOKENIZER_REPORT.md.
- Evidence for later: the longest question is 25 tokens (27 with <bos>/<eos>). The context length is decided later.
- Trained only on train.txt; validation used only for measuring.
```

## Step 7. コミットしてプッシュする(VS Codeターミナル、`llm_home`)

```powershell
git status --short
git add src scripts tests data/tokenizer docs run_log.md
git status --short
git commit -m "Day 6 home: own byte-level BPE tokenizer, size chosen from evidence"
git push
```

一覧に `data/corpus/` が出てはいけません。

## レビューで話すポイント

- **なぜバイトか:** 256個のバイトのトークンがあれば、あり得るすべての文を符号化できる。英語、記号、絵文字でも「未知語」の問題が起きない。
- **なぜ前処理をするか:** 文字の種類が変わるところで切ると、トークンが意味のある単位になり(漢字の言葉とその後の助詞が1つのトークンに混ざらない)、学習も速くなる。
- **なぜ数字を1文字ずつにするか:** どんな数でも1桁ずつ書けるので、モデルがトークンの無い数に出会うことがない。
- **なぜ予備のID(`<unused_0..99>`)があるか:** 後で拡張するための予備のIDのまとまり。埋め込みの表の大きさを変えずに済む。
- **なぜ16,000ではなく1,674か:** 測ったから。文には743種類の部分しかないので、それより大きな語彙は存在し得ない。書いておいたルールで選び、表を報告に載せるほうが、きりのいい数を選ぶより説明しやすい。
- **なぜ train.txt だけで学習するか:** 検証用の文が語彙を決めると、検証の点数が良く見えすぎる。検証用は測るためだけに使う。
- **今日は晴れ が示すこと:** トークナイザーは業務の言葉を知っている。知らない言葉はバイトで書かれ、それでも完全に元に戻る。課題の紙の例ではなく、実際の出力を見せる。
- **毎回同じ:** 同じ学習用の文からは、どの PC でも同じファイル(同じ SHA-256)ができる。

## Day 6 の終わりのチェックリスト

- [ ] ファイルをコピーした(`src`、`scripts`、`tests`、手順書)
- [ ] 学習前のテスト: `29 passed, 1 skipped`
- [ ] 学習: 743種類の部分、1,314回の結合、1,674を選択、SHA-256 `53b03761cd1b…`
- [ ] 学習後のテスト: `30 passed`
- [ ] 報告を読み、`docs/DECISIONS.md` に D6 を追加した
- [ ] コミットとプッシュをした

## 送り返すもの

1. Step 3 のコンソール出力(テキストで)。
2. Step 4 の pytest の結果の行。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題8】文章を細かく分けるプログラム(トークナイザー)
状況: 予定通り終わった
内容: バイト単位のBPEトークナイザーを自作し、学習用の質問文で学習した。語彙の大きさは、
      候補ごとに検証用データで測り、決めたルールで選んだ(約1,700)。保存と読み込み、
      文→番号→文の往復が完全に一致することをテストで確認した。
証拠: data/tokenizer/tokenizer.json、docs/DAY6_TOKENIZER_REPORT.md、run_log.md、tests/
次: 課題8・9のデモ(文をトークンと番号に分けて戻す)とコーパスの符号化(Day 7)
```
