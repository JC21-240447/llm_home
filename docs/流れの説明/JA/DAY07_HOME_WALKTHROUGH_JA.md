# Day 7(自宅ノートPC): トークナイザーのテスト、文→番号→文のデモ、コーパスの符号化

**マシン:** 自宅ノートPC。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**作業計画:** **作業8**(デモ)と **作業9**(トークンを番号にして元に戻す)を完了する。今日が終わると作業9は完了なので、その資料を書けます。

## Day 7 で行うこと、その理由

Day 6 でトークナイザーを作りました。今日は、すべてのデータで正しく動くことを証明し、学習用にデータを準備します:

1. **デモ**(`day7_tokenizer_demo.py`): 文ごとに、切り分けた部分、各部分の番号、番号から戻した文を表示する。学校の作業8と作業9の証拠になる。
2. **全件の確認**(`day7_encode_corpus.py`): 学習用・検証用・テスト用の **すべての** 質問を符号化して復号し、完全に同じに戻ることを確かめる。難しい文9つ(絵文字、英語、記号、改行、4バイトの漢字)も同様に確かめる。
3. **コーパスの符号化**: 各分割を `<bos> 質問 <eos> <bos> 質問 <eos> …` という1本の長い番号の列にし、16ビットの整数として `data\tokens\` に保存する。学習プログラム(Day 9)はこのファイルを、全体をメモリに読み込まずに、少しずつ直接読む。書き込んだ後、ディスクからファイルを読み直して復号し、ファイルそのものが正しいことを証明する。
4. **実際のトークン数**: Day 5 では範囲(学習用で165万〜1,010万)しか分からなかった。今回は測る: **学習用は2,028,601トークン**。この数で、Day 11 のモデルの大きさを決める。

新しいコード:

| ファイル | 内容 |
| --- | --- |
| `src\token_data.py` | トークンファイルの書き込みと読み込み(Day 9 でも使う) |
| `scripts\day7_encode_corpus.py` | 全件の往復確認、用語の表、符号化、読み直し、報告 |
| `scripts\day7_tokenizer_demo.py` | 文→トークン→番号→文のデモ |
| `tests\test_token_data.py` | 新しいテスト4つ(列の形、ファイルの往復、大きさの確認、本物のファイル) |

### サーバーで使う場合の注意(後日)

Day 7 のコードは普通の Python と NumPy だけなので、Linux でもそのまま動きます。サーバーでは `\` の代わりに `/` を使い、Linux の venv のパスを使います。例: `.venv/bin/python scripts/day7_encode_corpus.py --machine server`。トークンファイルはバイトの並び順を決めて書くので、サーバーでもノートPCと同じ SHA-256 にならなければなりません。

---

## Step 1. ファイルを置く

1. このチャットから `day7.zip` をダウンロードして展開する。
2. 展開した `day7` フォルダーの **中身をすべて** `llm_home` にコピーする。今回は置き場所に注意: `src\token_data.py` は **`llm_home\src`** に入れる(`data\source` ではない)。
3. 確認する(VS Codeターミナル、`llm_home`):
   ```powershell
   dir src, scripts\day7_*.py, tests
   ```
   `src` に `bpe_tokenizer.py`、`token_data.py`、`__init__.py`、`tests` に2つのテストファイルが見えること。

## Step 2. 符号化の前にテストする(VS Codeターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: `33 passed, 1 skipped`。スキップされる1つは本物のトークンファイルを確かめるテストで、まだファイルが無いためです。

## Step 3. コーパスを符号化する(VS Codeターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day7_encode_corpus.py --machine home
```

10秒ほどで終わります。期待する出力(大事な行):

```text
Round trip on 9 tricky texts: PASS
train   148,860 questions   2,028,601 tokens  avg 13.6  max  28  partial-char    3 in 1 questions  -> round trip + read-back PASS
val       4,517 questions      56,157 tokens  avg 12.4  max  22  partial-char  503 in 231 questions  -> round trip + read-back PASS
test      4,553 questions      63,877 tokens  avg 14.0  max  25  partial-char    0 in 0 questions  -> round trip + read-back PASS
Key terms: 772 (in training text: 279); 1-3 tokens: 464 of all, 273 of those in the training text
Total tokens: 2,148,635 (train 2,028,601)
```

`docs\DAY7_ENCODING_REPORT.md` を開き、最後のハッシュを確認します:

| ファイル | SHA-256 の先頭 |
| --- | --- |
| `data/tokens/train.bin` | `2c95990940ea` |
| `data/tokens/val.bin` | `e39257300d28` |
| `data/tokens/test.bin` | `ec593af0f94a` |

私のものと同じハッシュなら、ファイルが1バイトも違わないことの証明になります。違ったら出力を送ってください。

## Step 4. 符号化の後にテストする

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: `34 passed`。最後のテストは本物の検証用トークンファイルを読み、すべての質問が完全に元に戻ることを確かめます。

## Step 5. 作業8・9のデモ

```powershell
.venv\Scripts\python.exe scripts\day7_tokenizer_demo.py
```

文ごとに、部分、番号、戻した文を表示します。例:

```text
Input   : 営業部の社員の平均年収を教えてください。
Pieces  : 営業部 / の / 社員 / の / 平均年収 / を / 教 / えてください / 。
Numbers : [982, 362, 379, 362, 1239, 373, 402, 403, 364]
Decoded : 営業部の社員の平均年収を教えてください。
Same as input: YES  (9 tokens)
```

デモには「今日は晴れ」も入っています。バイトのトークンになりますが、文は完全に元に戻ります。結果はすべて `docs\DAY7_TOKENIZER_DEMO.md` に保存されます。自分の文でも試せます:

```powershell
.venv\Scripts\python.exe scripts\day7_tokenizer_demo.py --text "開発部で年収が一番高い人は誰ですか。"
```

`--text` を付けると既定のデモファイルが上書きされるので、既定の例を保存したい場合は、最後に何も付けないコマンドを実行してください。

## Step 6. 報告を読み、3つの発見を理解する

1. **実際の大きさ:** 学習用は2,028,601トークン。質問は平均13.6トークン(`<bos>` と `<eos>` を含む)、一番長いもので28トークン。この数は、Day 11(モデルの大きさ)と文脈の長さを決めるときに使う。
2. **「部分文字のトークン」** は、トークナイザーが学んでいない文字を、バイトごとに書いたものです。学習用は3つだけ、テスト用は0。検証用は231問で503個あり、原因はたった3つの部分です: **満**(学習用では「未満」の中にしか出ない)、**昇順** と **という**(学習用の言い回しに出てこない組み合わせ)。新しい文字は1つも無く、新しいのは組み合わせだけです。これは、取り分けた分割が目的どおりに働いている証拠です: 検証用には、モデルが見たことのない言い回しが入っている。これらの質問も完全に符号化・復号できますが、モデルにとっては難しくなります。
3. **重要な用語:** 学習用の質問に出てくる重要な用語279個のうち、273個が1〜3トークン。残りの6個は、いくつかの部分からなる句(数が多い順)か、長い言葉の中にしか出ない言葉(勤続)。Day 6 と同じ結果です。

## Step 7. 記録してコミットする

`docs\DECISIONS.md` に追加します:

```markdown
## D7. Token files (Day 7)

- Each split is one uint16 stream (<bos> question <eos> ...), little-endian, in data/tokens/, with offsets and a meta file
  holding the vocabulary size and the tokenizer SHA-256 (training refuses a mismatched file).
- Real size: train 2,028,601 tokens; average question 13.6 tokens, longest 28 (with <bos>/<eos>).
- Round trip verified on every question and on the files read back from disk.
```

そしてコミットします(VS Codeターミナル、`llm_home`)。`data/tokens/` は作り直せるので、Git では無視されます:

```powershell
git status --short
git add src scripts tests docs run_log.md
git status --short
git commit -m "Day 7 home: tokenizer tests, demo, encoded corpus"
git push
```

一覧に `data/tokens/` が出てはいけません。

## レビューで話すポイント

- **なぜ一部ではなく全件を確かめるか:** 1文字でも失うトークナイザーは、学習データを気づかないうちに壊す。全件と、ディスクから読み直したファイルを確かめれば、それを防げる。
- **なぜ `<bos>`/`<eos>` 付きの1本の番号の列にするか:** 学習プログラムは列から決まった長さの区間を切り出す。始まりと終わりの印で、モデルは質問の始まりと終わりを知る。Model A は質問を最後まで作ることを学ぶ必要があるので、これが大事。
- **なぜ16ビットの数か:** 語彙(1,674)は十分に収まり、ファイルは32ビットの半分の大きさになる。
- **なぜ情報ファイルにトークナイザーのハッシュを入れるか:** トークナイザーが変わったら、合わない番号で学習するのではなく、学習が古いトークンファイルを拒否するため。
- **検証用の部分文字のトークンの意味:** 取り分けた言い回しは、学習用に無い組み合わせ(満、昇順、という)を少し使っている。検証用が本当に見たことのない言い回しを試していることを示す。
- **作業9を一文で:** 各部分には決まった番号があり、番号は完全に元の文に戻る。デモとテストがそれを証明している。

## Day 7 の終わりのチェックリスト

- [ ] ファイルをコピーした(`src\token_data.py` は `src` に、2つのスクリプト、新しいテスト)
- [ ] 符号化前の pytest: `33 passed, 1 skipped`
- [ ] 符号化: 3つの分割すべてで往復と読み直しが PASS。学習用 2,028,601 トークン。ハッシュが一致
- [ ] 符号化後の pytest: `34 passed`
- [ ] デモを実行し、`docs\DAY7_TOKENIZER_DEMO.md` が保存された
- [ ] D7 を追加し、コミットとプッシュをした。`data/tokens/` は Git に入っていない

## 送り返すもの

1. Step 3 のコンソール出力(テキストで)。
2. Step 4 の pytest の結果の行。
3. 必要なときに **作業9の資料** を頼んでください。今日で作業9は完了です。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題9】番号にして元に戻す仕組み
状況: 予定通り終わった
内容: トークンを番号に変換し、番号から元の文に戻す仕組みを確認した。学習用・検証用・テスト用の
      すべての質問文と、絵文字・英語・記号を含む文で、完全に元に戻ることを確認した。質問文を
      番号の列に変換し、学習用のファイルとして保存した。文→トークン→番号→文のデモを作成した。
証拠: docs/DAY7_TOKENIZER_DEMO.md、docs/DAY7_ENCODING_REPORT.md、run_log.md、tests/
次: 課題10(次の単語を予測する仕組み)に向けて、モデルを作る(Day 8)
```
