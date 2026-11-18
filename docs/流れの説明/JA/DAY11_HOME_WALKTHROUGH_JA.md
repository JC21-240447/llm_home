# Day 11(自宅ノートPC): 大きさの試験と、大きさの決定

**マシン:** 自宅ノートPC(RTX 4060 Laptop)。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**作業計画:** **作業11**「LLMの学習、動作確認、生成結果の評価」を **開始** する。作業11は Day 11〜13 で行う(Day 11 = 大きさを決める、Day 12 = 本番の学習、Day 13 = 評価の表)。作業11は Day 13 の後に完了する。

## Day 11 で行うこと、その理由

Day 8 で3つの大きさを用意しました: tiny(約100万パラメータ)、small(約520万)、medium(約1,480万)。今日は、3つすべてを **まったく同じ条件** で学習させ、**証拠** で1つを選びます。選ぶルールは、結果を見る **前に** 書いておきます。

Day 9 の結果から、試験のやり方を変えました。あなたの `smoke_c` では、検証の損失は **200ステップ目**(学習用の文章を1回も読み終わらない所)で一番低くなり、その後は上がりました。そのため:

1. **検証は100ステップごとではなく、25ステップごと。** 一番良い点は早く来るので、確認の間隔が広いと見逃してしまう。
2. **ドロップアウトを2つ目の条件にする。** ドロップアウトは、学習中に信号の10%をランダムに止めて、決まった並びの丸暗記に頼れなくする方法で、Day 9 で見つかった覚えすぎ(過学習)への標準的な対策です。各大きさを、ドロップアウト0.0と0.1で学習させるので、3 × 2 = **6回**。
3. **生成も確かめる。** Day 9・10 で、検証の損失が一番低いモデルが、一番良い文を書くとは限らないことが分かりました。そこで、各学習の一番良いチェックポイントを、Day 10 の12の書き出しで、作業6の自動の確認にかけます。

**6回すべてに同じ条件:** バッチ 32 × 256トークン、800ステップ(学習用の文章を約3回読む量)、ウォームアップ60、最大の学習率0.001、乱数の種42。違うのは大きさとドロップアウトだけです。

**試験の前に書いたルール**(`src\selection.py`):
1. 一番良いチェックポイントが、12の書き出し(貪欲法+上限)で作業6の目安 **完結90%以上**、**繰り返し5%以下** を満たすものだけを候補にする
2. 候補の中で、検証の損失の一番低い値を見つける
3. それとの差が **0.02以内** の候補は、同じくらい良いとみなす
4. その中で **パラメータが一番少ない** ものを選ぶ(同じなら損失が低い方、さらに同じならドロップアウトなし)

0.02以内なら小さい方を選ぶ理由: 損失の差0.02はパープレキシティで約2%で、実行ごとのゆれの範囲であり、証拠にならないためです。小さいモデルの方が速く、メモリも少なく、覚えすぎも少なくなります。

新しいファイル:

| ファイル | 内容 |
| --- | --- |
| `src\selection.py` | 学習の `log.csv` をまとめ(一番良い点、差、その後の上がり方、速さ、メモリ)、選ぶルールを当てはめる |
| `scripts\day11_size_trials.py` | 6回の学習を行い、各 `best.pt` で生成を確かめ、選び、報告・図・決定のファイルを書く |
| `tests\test_selection.py` | まとめとルールの新しいテスト6項目 |

出力:

| 出力 | 内容 |
| --- | --- |
| `runs\day11\<大きさ>_d<ドロップアウト>\` | 試験ごとの学習のフォルダー(例: `tiny_d00`、`medium_d10`。Git には入れない) |
| `docs\DAY11_SIZE_TRIALS.md` | 決定、結果の表、書いた文の例 |
| `docs\figures\day11_val_loss.png` | 6回すべての検証の損失を1つの図に |
| `configs\day11_decision.json` | 選んだ大きさとドロップアウト(**Day 12 がこれを読む**) |

**途中で止まっても再開できます。** Ctrl+C、スリープ、ウィンドウを閉じた場合など、何で止まっても **同じコマンドをもう一度** 実行してください。終わった学習は飛ばし、途中の学習は `last.pt` から続けます。学習の途中で強制終了させて、確かめてあります。

### サーバーで使う場合の注意(後日)

コードは同じ: `.venv/bin/python scripts/day11_size_trials.py --machine server`。サーバーでは、接続が切れても動き続けるように `tmux` の中で始めます。

---

## Step 1. ファイルを置く

1. `day11.zip` をダウンロードして展開する。
2. `day11` フォルダーの **中身すべて** を `llm_home` にコピーする。
3. 確認(VS Code ターミナル、`llm_home`):
   ```powershell
   dir src\selection.py, scripts\day11_size_trials.py, tests\test_selection.py
   ```

## Step 2. すべてのテストを実行する(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: **`72 passed`**(66+新しい6項目)。

| 番号 | 新しいテスト | 証明すること |
| --- | --- | --- |
| 67 | `test_summary_reads_best_point_gap_and_rise` | 記録から、検証の損失の最低値とそのステップ、そのときの学習の損失、差、最後の損失、上がった量、速さの中央値、メモリの最大値を正しく読む |
| 68 | `test_lowest_loss_wins_when_clearly_better` | はっきり低い損失は、大きいモデルでも選ばれる |
| 69 | `test_smaller_model_wins_within_tolerance` | 0.015 だけ悪いが5分の1の大きさ: 小さい方が選ばれる |
| 70 | `test_runs_failing_the_task6_checks_are_never_chosen` | 90%完結しない、または繰り返しが多い学習は、損失が一番低くても選ばれない |
| 71 | `test_no_candidate_gives_no_choice` | 確認に合格するものが無ければ、何も選ばない(勝手に推測しない) |
| 72 | `test_same_size_tie_prefers_lower_loss_then_no_dropout` | 同じ大きさで同じ損失なら、ドロップアウトなしを選ぶ |

## Step 3. ノートPCの準備

試験は **5〜10分** ほどかかります。始める前に:
- 電源アダプターをつなぐ
- Windows の設定 → システム → 電源 で、「電源に接続時、次の時間が経過した後にデバイスをスリープ状態にする」を、少なくとも今日は **なし** にする
- GPU を使う重いプログラム(ゲーム、動画編集など)を閉じる

## Step 4. 試験を実行する(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day11_size_trials.py --machine home
```

表示されるもの:
1. `[1/6] tiny_d00: new` の後に学習の行(25ステップごと)が続き、`[6/6] medium_d10` まで進む
2. `--- generation check on each best.pt ---` と、学習ごとに1行
3. 結果の表と、**`Decision: <学習の名前> - <理由>`** の行

何かで止まったら、同じコマンドをもう一度実行してください。

## Step 5. 報告を読む

`docs\DAY11_SIZE_TRIALS.md` を開き、**Ctrl+Shift+V** を押します。見るところ:

1. **図:** 学習ごとに1本の線。実線はドロップアウトなし、点線はあり。点は各線の一番低い所。下がってから上がる線は、覚えすぎ(過学習)です。
2. **Best val loss と At step:** どの大きさが一番低くなるか、どれくらい早いか。
3. **Gap**(一番良いステップでの、検証の損失 − 学習の損失): 大きいほど、知っている文型に比べて、見たことのない文型が苦手。
4. **Rise after best:** 一番良い点の後に、どれだけ悪くなったか。ドロップアウトで小さくなるはずです。
5. 一番良いチェックポイントでの **Finished / Repetitive / distinct-2** と、**書いた文の例**。目でも確かめます。例えば「勤続年数が」の後に、年数(正しい)が来るか、金額(誤り)が来るか。

どの大きさが勝つかは、私には予想できません。それを試験で確かめます。Day 9 からは、どの大きさも早く一番良い点に達してから上がり、大きいモデルほど早く上がると思われます。結果がどうであっても、ルールが選び、報告がその理由を示します。

## Step 6. 記録してコミットする

`docs\DECISIONS.md` に、報告の数字を入れて追加:

```markdown
## D11. Model size (Day 11)

- Trials: tiny / small / medium x dropout 0.0 / 0.1, same budget (800 steps, batch 32 x 256, lr 1e-3, seed 42),
  validation every 25 steps; best checkpoints checked on the 12 Day 10 prompts.
- Rule (fixed before the trials): pass task 6 checks (finished >= 90 %, repetitive <= 5 %); within 0.02 of the
  lowest validation loss, choose the fewest parameters.
- Result: chosen <run> (<size>, dropout <d>), best validation loss <x.xxxx> at step <n>.
  Lowest overall: <run> <x.xxxx>. Observations: <e.g. larger models reached their best earlier and rose more;
  dropout reduced the rise after best by ...>.
- Day 12 trains the chosen configuration (configs/day11_decision.json).
```

その後コミットする(VS Code ターミナル、`llm_home`):

```powershell
git status --short
git add src scripts tests configs docs run_log.md
git status --short
git commit -m "Day 11 home: size trials and size decision"
git push
```

`configs\day11_decision.json` は Git に入れます。一覧に `runs/` の中のものや `.pt` ファイルが出てはいけません。

## 審査で説明するポイント

- **データとパラメータ:** モデルはデータの形を学ぶが、文章の量に対してパラメータが多すぎると、学習用の質問を丸暗記する。質問は文型から作ったので、丸暗記すると見たことのない文型が苦手になる。それを測るのが検証の損失である。
- **「大きいほど良い」ではない理由:** 大量の文章で学ぶ汎用の大きなモデルでは、大きい方が勝つことが多い。ここでは文章の量が決まっていて少ないので、正しい大きさは思い込みではなく、測って決める。
- **ルールを先に書いた理由:** 数字を見てから選ぶと、どの選択にも理由を見つけられてしまう。前もって決めたルール(確認、許容の幅、小さい方を優先)なら、決定は正直で、同じ手順で再現できる。
- **許容の幅が0.02の理由:** これより小さい差はパープレキシティで約2%で、実行ごとのゆれの大きさ。ゆれの中の差のために大きいモデルを選ぶと、証拠の無い改善のために速さとメモリを失う。
- **ドロップアウトの理由:** 学習中にランダムな信号を止めると、モデルは覚えた決まった並びに頼れなくなる。覚えすぎへの標準的な対策で、ここで効くかどうかを試験で測る。
- **25ステップごとに検証する理由:** Day 9 で、一番良い点は最初の数百ステップの中にあった。100ステップごとでは、本当の一番良い点が2回の確認の間に入ってしまうかもしれない。
- **文も確かめる理由:** Day 9・10 で、検証の損失が一番低くても、一番良い文を書くとは限らないことが分かった。そのため、作業6の自動の確認にも合格しなければ選ばない。

## Day 11 の終わりのチェックリスト

- [ ] ファイルをコピーした(`src\selection.py`、`scripts\day11_size_trials.py`、`tests\test_selection.py`、説明ファイル2つ)
- [ ] pytest: `72 passed`
- [ ] ノートPCは電源につなぎ、スリープを切った
- [ ] 6回の試験が終わり、`Decision:` の行が出た
- [ ] `docs\DAY11_SIZE_TRIALS.md`、`docs\figures\day11_val_loss.png`、`configs\day11_decision.json` がある
- [ ] 実際の数字で D11 を追加し、コミットとプッシュをした

## 送り返すもの

1. pytest の結果の行。
2. `--- generation check on each best.pt ---` から出力の最後まで(確認、表、決定)。
3. `docs\DAY11_SIZE_TRIALS.md`(ファイル全体、または添付)。
4. `docs\figures\day11_val_loss.png`。

これで決定を確かめ、Day 12 を用意します。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題11】LLMの学習、動作確認、生成結果の評価(開始)
状況: 予定通り終わった(今週の予定: 課題11の開始。モデルの大きさを決める)
内容: 3つの大きさのモデル(約100万・500万・1,500万パラメータ)を、ドロップアウトあり・なしで、
      同じ条件で学習させて比べた(6回)。前もって決めたルール(作業6の自動の確認に合格し、
      検証の損失が一番低いものとの差が0.02以内なら、パラメータが少ない方を選ぶ)で、
      本番のモデルの大きさを決めた。自動テスト6項目を追加し、合計72項目すべて合格。
証拠: docs/DAY11_SIZE_TRIALS.md、docs/figures/day11_val_loss.png、configs/day11_decision.json
次: 決めた大きさで本番の学習を行う(Day 12)
```
