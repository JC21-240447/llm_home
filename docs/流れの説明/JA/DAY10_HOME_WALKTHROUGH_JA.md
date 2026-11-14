# Day 10(自宅ノートPC): 次の単語の候補、サンプリング、繰り返しの制御

**マシン:** 自宅ノートPC(RTX 4060 Laptop)。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**作業計画:** **作業10**(次の単語を予測する仕組み)が **完了** する。作業10の3つの証拠がそろう: テストに合格したモデルのコード(Day 8)、最初の学習の損失の曲線(Day 9)、**次の単語の候補トップ10の表**(今日)。

## Day 10 で行うこと、その理由

Day 9 でモデルを学習し、`samples.md` には、いつも一番確率の高いトークンを選んだときの文が出ていました。今日は、モデルの確率から文を正しく作る部分を作ります。また、確率そのものも見せます。これが作業10の目標「「今日」「は」の次に何が来るとモデルは考えるか」です。

**次の単語の確率が文になるまで(1トークンずつ):**
1. モデルがそれまでのトークンを読み、1,674個のトークンすべてに点数を付ける。softmax で合計1の確率にする。**候補の表は、この上位10個を示す。**
2. 点数を調整する: 特別なトークン(`<pad>`、`<bos>`、`<unk>`、`<unused_*>`)を除き、繰り返しの上限をかける(下を参照)。
3. 1つ選ぶ: **貪欲法** = いつも一番上(同じ入力なら同じ出力)。**サンプリング** = 温度 → top-k → top-p → 固定した乱数の種で選ぶ。
4. `<eos>`(質問の終わり)か40トークンまで繰り返す。

**繰り返しの上限は、学習データから測って決めた。** 上限を決める前に、学習用の質問文を測りました: 同じトークンは **続けて最大4回**(「社員番号100003」)、同じ3トークンの並びは1つの質問に **最大2回**(「1991年から1997年」)。そこで上限をちょうどこの数にしました: 同じトークンの5回目、同じ3トークンの並びの3回目は選べません。これで「5000000…」のような繰り返しは止まりますが、本当の質問が止められることはありません。よく使われる「同じ3トークンの並びを繰り返さない」という設定では、「1991年から1997年に入社した社員」のような本当の質問まで止めてしまいます。先に測ったことで、この誤りを避けられました。

新しいファイル:

| ファイル | 内容 |
| --- | --- |
| `src\generation.py` | 候補、点数の調整、貪欲法とサンプリング、`generate()`、作業6の自動の確認(完結・繰り返し・distinct-2) |
| `scripts\day10_generate.py` | 3つのコマンド: `candidates`、`generate`、`report` |
| `prompts\day10_prompts.txt` | 固定した12の書き出し: 社内の9つ、少し難しい1つ(「社員の平均」)、範囲外の2つ(「今日は」「天気が」) |
| `tests\test_generation.py` | 新しいテスト12項目 |

比べる4つの設定(報告を同じに作り直せるように固定):

| 設定 | 内容 | 比べる理由 |
| --- | --- | --- |
| `greedy` | 一番上のトークン、制御なし | Day 9 の `samples.md` と同じ |
| `greedy+limits` | 貪欲法 + 繰り返しの上限 | 安全で、毎回同じ結果の設定 |
| `sample` | 温度0.8、top-k 50、top-p 0.95、上限あり | ばらつきがあるが慎重。使う予定の設定 |
| `sample-hot` | 温度1.5、上限あり | ランダムすぎるとどうなるかを見せるため |

**作りながら直した点が1つあります。** 最初の版では、どの書き出しでも乱数を同じ種42から始めていたので、違う書き出しが同じ乱数を使っていました。そのため「営業部の」「開発部の」「東京本社の」がまったく同じ文になりました。今は、種と書き出しから、書き出しごとに決まった種を作ります。何度実行しても結果は同じで、書き出しどうしは独立しています。テストで確かめています。

### サーバーで使う場合の注意(後日)

コードは同じ: `.venv/bin/python scripts/day10_generate.py report --run smoke_a --machine server`。生成は軽いので、`--device cpu` でもどのPCでも動きます。

---

## Step 1. ファイルを置く

1. `day10.zip` をダウンロードして展開する。
2. `day10` フォルダーの **中身すべて** を `llm_home` にコピーする(`prompts` フォルダーは新しい)。
3. 確認(VS Code ターミナル、`llm_home`):
   ```powershell
   dir src\generation.py, scripts\day10_generate.py, tests\test_generation.py, prompts
   ```

## Step 2. すべてのテストを実行する(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: **`66 passed`**(54+新しい12項目)。

| 新しいテスト | 証明すること |
| --- | --- |
| `test_candidates_are_sorted_probabilities` | 上位10個は本当の softmax の確率で、大きい順 |
| `test_special_tokens_are_never_written_but_eos_is_allowed` | `<bos>` や `<pad>` などは書けない。`<eos>` は書ける |
| `test_same_token_run_limit_matches_training_data` | 続けて4回は許す(社員番号100003 と同じ)。5回目は許さない |
| `test_trigram_limit_matches_training_data` | 3トークンの並びは2回まで(1991年から1997年 と同じ)。3回目は許さない |
| `test_repetition_penalty_lowers_only_used_tokens` | 任意のペナルティは、使ったトークンだけを変える |
| `test_greedy_takes_the_top_token` | 貪欲法 = 一番高い点数 |
| `test_top_k_one_and_tiny_top_p_behave_like_greedy` | top-k と top-p が正しいトークンを除く |
| `test_temperature_changes_how_often_the_second_token_is_chosen` | 2,000回選んで測る: 温度1では2番目のトークンが softmax のとおりの割合で選ばれ、温度が低いと少なく、高いと多くなる |
| `test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts` | 上で説明した種の修正 |
| `test_generate_stops_at_eos_and_is_reproducible` | `<eos>` で終わり、長さの上限で止まり、2回やっても同じ結果になる |
| `test_repetition_check_and_distinct_2` | 作業6の A3 と A4 が正しく計算される |
| `test_invalid_settings_are_refused` | ありえない設定には分かりやすいエラーを出す |

## Step 3. 次の単語の候補: 作業計画の例(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day10_generate.py candidates --text 今日は --text 営業部の --out docs\DAY10_NEXT_TOKEN_CANDIDATES.md
```

初期設定で `runs\day9\smoke_a\best.pt` を使います。送ってもらったチェックポイントで、同じコードを実行した結果:

```text
今日は  ->  pieces: <e4bb> / <8a> / 日 / は
   1. 誰             71.97%
   2. 何人             6.51%
   3. 教              2.14%
   ...
      top 10 total: 86.92%
```

GPU では最後の桁が少し違うことがあります。分かること:
- 「今」は学習用の質問に無いので、2つのバイトの部分で書かれる(Day 7 のデモと同じ)。
- モデルは社内の質問しか学んでいないので、「…は」の後には質問の言葉を予想する: 「誰」、「何人」。
- 天気を知らないので、「今日は」の後に「晴れ」とは続けられない。**候補は、モデルが学んだことをそのまま示している。**

## Step 4. Day 10 の報告一式(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day10_generate.py report --run smoke_a --machine home
```

Day 9 の学習の2つのチェックポイント、`best.pt`(ステップ200、検証の損失が最低)と `last.pt`(ステップ600)の両方で実行し、`docs\DAY10_GENERATION_REPORT.md` を書きます。内容は次の4つです:
1. 4つの書き出しの候補トップ10(`best.pt` と `last.pt` を並べる)
2. すべての書き出しを、4つの設定と2つのチェックポイントで生成
3. 作業6の自動の確認
4. 学習前(ステップ0)のモデルでの繰り返しの上限

1分かかりません(私の CPU で45秒)。期待する自動の確認(あなたのチェックポイントを私の CPU で実行した結果):

```text
ckpt  setting         finished  repetitive  distinct-2    new
best  greedy             12/12        0/12        0.56  12/12
last  greedy             12/12        0/12        0.47   8/12
best  greedy+limits      12/12        0/12        0.56  12/12
last  greedy+limits      12/12        0/12        0.47   8/12
best  sample             12/12        0/12        0.76  11/12
last  sample             12/12        0/12        0.80   8/12
best  sample-hot         10/12        0/12        0.98  10/10
last  sample-hot         12/12        0/12        0.92   9/12
```

greedy の行は同じにならなければなりません。Day 9 の `samples.md` とも一致します。例えば「大阪支社で」は、`best.pt` では「働く社員の2ですか。」、`last.pt` では「働く社員の時間外手当の中央値を出してください。」です。サンプリングの行は乱数で選ぶので、GPU では所々の言葉が変わることがあります。

## Step 5. 報告を読む: 5つの発見

`docs\DAY10_GENERATION_REPORT.md` を開き、**Ctrl+Shift+V** を押します。

1. **`last.pt` の方が、社内の言葉を正しく使う。** 「勤続年数が」に対して、`best.pt` は「30万円以上の社員…」(年数にお金。作業6の H4 の誤り)と書き、`last.pt` は「30年以下の社員を教えてください。」と正しく書く。Day 9 の発見の確認です: 検証の損失が一番低いモデルが、知っている書き出しで一番良い文を書くとは限らない。
2. **サンプリングで、文を崩さずにばらつきが出る。** `sample` は両方のチェックポイントで12の質問すべてを終え、distinct-2 は約0.5(貪欲法)から約0.8に上がる。例: 「ボーナスが」→「500万円以上の社員は何人いますか。」
3. **ランダムすぎると文が壊れる。** `sample-hot`(温度1.5)は壊れた文字(�)や意味のない言葉の並びを書く。温度で確率が平らになり、まれなバイトの部分や関係のない言葉が選ばれるためです。だから `sample` は温度0.8に top-k と top-p を合わせて使う。
4. **繰り返しの上限は働き、安全のための仕組みになっている。** 学習したチェックポイントは、これらの書き出しではもう繰り返さないので、`greedy` と `greedy+limits` は同じになる。4つ目の項目で、Day 9 の種を使って学習前のモデルを作り直すと、`samples.md` の「のののの…」がそのまま再現される。貪欲法はずっと繰り返し、上限があると「の」を4回で止めて先に進む。
5. **範囲外の書き出しは、社内の質問に変わる。** 「今日は」→「誰ですか。」、「天気が」→「入っている社員を教えてください。」。モデルは社内の質問しか見ていないので、何でも社内の質問にする。範囲外の入力を見分けることは、このモデルの役割ではない。作業6の評価では、「範囲外」の書き出しとして記録する。

## Step 6(任意). 自分の書き出しを試す

```powershell
.venv\Scripts\python.exe scripts\day10_generate.py generate --setting sample --checkpoint runs\day9\smoke_a\last.pt --text 人事部で --text 年齢が
```

`--temperature 1.0` や `--seed 7` のように、設定を1つだけ変えることもできます。

## Step 7. 記録してコミットする

`docs\DECISIONS.md` に追加:

```markdown
## D10. Generation (Day 10)

- Repetition limits measured from the training questions: a token at most 4 times in a row (社員番号100003),
  a 3-token sequence at most twice (1991年から1997年). A generic "no repeated 3-gram" rule would block real questions.
- Settings compared: greedy, greedy+limits, sample (T 0.8, top-k 50, top-p 0.95), sample-hot (T 1.5).
  Default for evaluation: greedy+limits (deterministic) and sample (varied).
- Sampling seed is derived from (seed, prompt): reproducible, and different prompts get independent draws.
- Findings: last.pt (step 600) uses company terms better than best.pt (step 200); T 1.5 breaks text;
  out-of-scope beginnings are turned into company questions.
- Task 6 item A3 corrected to "5+ same tokens in a row" (4 in a row occurs in real questions).
```

その後コミットする(VS Code ターミナル、`llm_home`):

```powershell
git status --short
git add src scripts tests prompts docs run_log.md
git status --short
git commit -m "Day 10 home: next-token candidates, sampling, repetition limits, generation report"
git push
```

## 審査で説明するポイント

- **確率が文になる仕組み:** モデルは文を直接書かない。各ステップで、すべてのトークンに確率を付け、1つ選び、入力に加え、もう一度モデルに聞く。文は、これを `<eos>` まで繰り返したもの。
- **貪欲法とサンプリング:** 貪欲法はいつも一番上を選ぶので、予測できるが単調(同じ書き出しからはいつも同じ質問)。サンプリングは確率に従って選ぶので、ばらつきが出る。種を固定すれば、それでも完全に同じ結果を作り直せる。
- **温度:** softmax の前に点数を温度で割る。1より小さいと上位がもっと選ばれやすくなり、1より大きいと確率が平らになる。報告では、1.5 で文が壊れることが分かる。
- **top-k と top-p:** どちらも、可能性の低いトークン(壊れた部分や関係のない言葉がある所)を切り捨てる。top-k は決まった個数、top-p は合計が95%になるまでの個数を残す。
- **繰り返しの上限が安全な理由:** 学習用の質問文を測って決めたので、本当の質問が決してしないことだけを止める。
- **候補が作業10について証明すること:** 仕組みが最初から最後まで動く。「今日は」に対して「誰」を72%で予想するのは、モデルが社内の質問の形を学び、それ以外は学んでいないことを示している。

## Day 10 の終わりのチェックリスト

- [ ] ファイルをコピーした(`src\generation.py`、`scripts\day10_generate.py`、`prompts\day10_prompts.txt`、`tests\test_generation.py`、説明ファイル2つ)
- [ ] pytest: `66 passed`
- [ ] 今日は と 営業部の の候補を表示し、`docs\DAY10_NEXT_TOKEN_CANDIDATES.md` に保存した
- [ ] 報告を作った。greedy の行が Day 9 の `samples.md` と一致する
- [ ] D10 を追加し、コミットとプッシュをした
- [ ] **作業10が完了**

## 送り返すもの

1. pytest の結果の行。
2. Step 3 の出力。
3. Step 4 の「automatic checks」の表と、「repetition limits on the untrained model」の行。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題10】次に続く単語を予測する仕組み
状況: 予定通り終わった(完了)
内容: 学習したモデルが、書き出し(例:「今日は」)に対して次の単語の候補と確率を上位10個まで出す仕組みと、
      その確率から文を作る仕組み(貪欲法と確率による選び方、温度・top-k・top-p、繰り返しの制御)を作成した。
      繰り返しの上限は学習用の質問文から測って決めた。自動テスト12項目を追加し、合計66項目すべて合格。
証拠: docs/DAY10_GENERATION_REPORT.md(候補の表、生成の比較)、src/generation.py、tests/test_generation.py
次: 試す大きさのモデルを学習して比べ、本番のモデルの大きさを決める(作業11)
```
