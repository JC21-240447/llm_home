# Day 12(自宅ノートPC): 本番の学習

**マシン:** 自宅ノートPC(RTX 4060 Laptop)。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**作業計画:** 作業11「LLMの学習、動作確認、生成結果の評価」の **続き**(Day 11 = 大きさを決める、**Day 12 = 本番の学習**、Day 13 = 評価の表)。作業11は Day 13 の後に完了する。

## Day 12 で行うこと、その理由

Day 11 で **tiny、ドロップアウト0.1** に決めました(`configs\day11_decision.json`)。今日はこの設定を **本番のモデル**(Day 13 で評価するモデル)として学習させます。Day 11 の試験から、次の3点を変えます。どれも Day 11 で分かったことが理由です。

1. **乱数の種を1つではなく3つ(42、43、44)。** 種43での確認で、同じ設定でも種だけで検証の損失が最大 **0.062** 変わると分かりました。1回の数字は信頼できる結果ではないので、3回の **平均とばらつき** で示します。最終のモデルは、検証の損失が一番低い種のものです。
2. **学習の予定を短く: 800ステップではなく400ステップ。** Day 11 では一番良い点が125〜225ステップに来ましたが、その時点では学習率がまだ高いままでした(学習率は予定の終わりに向かって下がるため)。400ステップにすると、一番良い点の頃にはもう学習率が下がり始めていて、モデルがより小さく慎重に進みます。
3. **早期終了。** 10ステップごとに検証し、検証の損失が **100ステップ** 改善しなければ学習を終えます。一番良い点を過ぎた後の学習は、覚えすぎを増やすだけです。一番良い点は `best.pt` に保存されているので、続けるのは時間の無駄です。

**今日はテスト用のデータに触れません。** 最終のモデルは検証用のデータだけで選びます。テスト用のデータは Day 13 のために取っておき、選ぶときに一度も見ていない文で最終の評価をします。

3回すべてに同じ条件:
- tiny、ドロップアウト0.1(`configs\day11_decision.json` から読む。手で入力し直さない)
- バッチ 32 × 256トークン、400ステップ、ウォームアップ60、最大の学習率0.001、コサインで10%まで下げる
- 10ステップごとに検証、早期終了の待ち(patience)100

違うのは乱数の種だけです。

### 変更したファイル

| ファイル | 変更 |
| --- | --- |
| `src\training.py` | **早期終了**(`early_stop_patience`)。一番良いステップをチェックポイントに保存するので、Ctrl+C と再開の後も残る。最後に各学習が `status.json`(`finished`、`early_stopped`、`stopped`)を書く |
| `src\generation.py` | `quick_checks()`: 固定の書き出しで、作業6の自動の確認(完結、繰り返し、distinct-2、新しい文)を1つの関数で行う |
| `tests\test_training.py` | 早期終了の新しいテスト2項目 |

### 新しいファイル

| ファイル | 内容 |
| --- | --- |
| `scripts\day12_train_main.py` | 3つの種で学習し、各 `best.pt` で生成を確かめ、最終のモデルを選び、Day 11 と比べ、報告・図・最終のモデルのファイルを書く |
| `scripts\day12_monitor.py` | 学習の進み具合、一番良い点、早期終了までの残り、残り時間を表示する(任意。2つ目のターミナルから) |

### 出力

| 出力 | 内容 |
| --- | --- |
| `runs\day12\main_s42\`、`main_s43\`、`main_s44\` | 種ごとの学習のフォルダー(Git には入れない) |
| `docs\DAY12_MAIN_TRAINING.md` | 最終のモデル、すべての種、Day 11 との比較、書いた文の例 |
| `docs\figures\day12_main.png` | 3つの種の学習(点線)と検証(実線)の損失 |
| `configs\day12_final.json` | どのチェックポイントが最終のモデルか。**Day 13 がこれを読む** |

**Day 11 と同じく、途中から再開できます。** 中断されたら **同じコマンドをもう一度** 実行してください。終わった種(または早期終了した種)は飛ばし、途中の種は `last.pt` から、早期終了の数も含めて続けます。3つ目の種の途中でスクリプトを強制終了して確かめました。

### サーバーのメモ(後で)

同じコード: `.venv/bin/python scripts/day12_train_main.py --machine server`。`tmux` の中で始める。

---

## 手順1. ファイルを置く

1. `day12.zip` をダウンロードして展開する。
2. `day12` フォルダーの **中身すべて** を `llm_home` にコピーする。`src\training.py`、`src\generation.py`、`tests\test_training.py` は、Windows に聞かれたら置き換える。
3. 確認する(VS Code のターミナル、`llm_home`):
   ```powershell
   dir scripts\day12_train_main.py, scripts\day12_monitor.py
   ```

## 手順2. すべてのテストを実行する(VS Code のターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: **`74 passed`**(72 + 新しい2項目)。

| 番号 | 新しいテスト | 証明すること |
| --- | --- | --- |
| 73 | `test_early_stopping_ends_the_run_and_keeps_the_best` | 検証の損失が改善しなくなるデータで、学習が `max_steps` の前に終わり、`status.json` が `early_stopped` になり、`best.pt` が(最後ではなく)一番良い点を持っている |
| 74 | `test_early_stopping_survives_stop_and_resume` | Ctrl+C で止めて再開した学習が、一度も止めなかった学習と、まったく同じステップで、同じ一番良い点で止まる |

今までのテスト(Day 9 の「再開しても結果が同じ」テストを含む)もすべて合格するので、早期終了を加えても学習の他の部分は何も変わっていません。

## 手順3. ノートPCの準備

3回の学習は合わせて **約3〜6分** です(tiny は速い。10ステップごとの検証で少し増える)。Day 11 と同じく、電源アダプターをつなぎ、スリープを **なし** にし、GPU を使う重いプログラムを閉じてください。

## 手順4. 本番の学習を実行する(VS Code のターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day12_train_main.py --machine home
```

表示されるもの:
1. `=== Day 12 main training | tiny, dropout 0.1 (from day11_decision.json) | seeds [42, 43, 44] | ... ===`。**tiny, dropout 0.1** と書いてあるか確かめる。
2. `[1/3] main_s42: new`、その後10ステップごとの学習の行。`* best, saved best.pt` が付いた行は、新しい一番良い点。
3. たぶん `Early stop at step 3xx: no improvement for 100 steps (best validation loss ... at step ...; best.pt keeps it).` という行。これは予定通りで良いことで、エラーではありません。
4. `main_s43`、`main_s44` も同じ。
5. `--- generation check on each best.pt ---`、その後にまとめ: 種ごとの損失、平均とばらつき、Day 11 との比較、**`Final model: main_sXX`**。

何かの理由で止まったら、同じコマンドをもう一度実行してください。

**任意: 学習の様子を見る。** 2つ目のターミナル(VS Code のターミナル欄の **+**)を `llm_home` で開いて実行:

```powershell
.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5
```

5秒ごとに更新して、進み具合のバー、一番良い点、`early stop N / 100 steps without improvement`(早期終了までの残り)、残り時間を表示します。記録を読むだけなので、学習の邪魔にはなりません。終えるときは、そのターミナルで **Ctrl+C**(そこで Ctrl+C を押しても学習は止まりません)。学習の後に `--watch 5` なしで実行すると、最後の状態を1回表示します。

## 手順5. 報告を読む

`docs\DAY12_MAIN_TRAINING.md` を開き、**Ctrl+Shift+V** を押します。見るところ:

1. **図。** 3つの色 = 3つの種。実線 = 検証、点線 = 学習。点は `best.pt` の位置。線は互いに近いはずで、離れている所が種によるゆれです。実線が平らになり点線だけが下がり続ける所で、モデルが覚えすぎを始めています。早期終了が学習を終えたのはそこです。
2. **すべての種の表。** *How it ended*(`early_stopped at step ...` か `finished at step 400`)、*Best val loss*、*At step*、*Gap*(一番良いステップでの検証と学習の差)。その下の **Mean ... spread ...** の行。
3. **Day 11 の試験との比較。** Day 11 の tiny_d10 は、800ステップの予定、種2つで平均2.6637でした。**変化は、ばらつきと一緒に読んでください。** 変化が種のばらつきより小さければ、短い予定に効果があったとは言えません。それも正しい結果なので、そのまま報告します。
4. 最終のモデルの **書いた文の例** と、Finished / Repetitive の列。3つとも作業6の目安(完結90%以上、繰り返し5%以下)を満たすはずです。

正確な数字は予測できません。私の予想では、平均は Day 11 に近く(約2.64〜2.70)、ばらつきは0.05前後です。Day 11 で、結果の上限を決めるのは設定ではなくデータだと分かったからです。どんな結果でも、正直に記録します。

## 手順6. 記録してコミットする

`docs\DECISIONS.md` に、報告の数字を入れて追加:

```markdown
## D12. Main model (Day 12)

- Configuration from D11 (tiny, dropout 0.1), 400-step schedule (warm-up 60, lr 1e-3, cosine to 10 %),
  validation every 10 steps, early stopping after 100 steps without improvement, seeds 42 / 43 / 44.
- Best validation loss per seed: <x.xxxx> / <x.xxxx> / <x.xxxx> (mean <x.xxxx>, spread <x.xxxx>).
  Each run ended: <early_stopped at step n / finished>.
- Compared with the Day 11 trials (800 steps, mean 2.6637): change <+/-x.xxxx>, <inside / outside> the seed spread.
- Final model: runs/day12/main_s<seed>/best.pt (lowest validation loss; passes the task 6 checks).
  Recorded in configs/day12_final.json. The test set was not used; it is kept for Day 13.
```

その後コミット(VS Code のターミナル、`llm_home`):

```powershell
git status --short
git add src scripts tests configs docs run_log.md
git status --short
git commit -m "Day 12 home: main training with early stopping, 3 seeds, final model"
git push
```

`configs\day12_final.json` は Git に入れます(小さいファイル。指している `.pt` ファイルは入れない)。一覧に `runs/` の中のものや `.pt` ファイルが出てはいけません。

**学習のフォルダーは残してください。** `runs\day12\main_s<種>\best.pt` が最終のモデルで、Day 13 で使います。`runs\day12\` を消さないでください。

## 発表で話すこと

- **種を3つにした理由:** 種は、最初の重みと、学習用の区切りを読む順番を決めます。Day 11 で、種だけで検証の損失が最大0.06変わると測りました。1回だけ報告するとこのゆれが隠れるので、平均とばらつきを示し、差のどこまでが本物かを分かるようにしました。
- **早期終了の理由:** 一番良い点の後は、モデルは学習用の質問を覚えるだけで、検証の損失は上がります。100ステップ改善しなければ止めることで時間を節約し、残すモデルは `best.pt` に保存された一番良い点のままです。
- **待ちを100ステップにした理由:** 検証の損失は、確認ごとに少し上下します。最初に上がった所で止めると、偶然で早く止まりすぎます。100ステップは確認10回分で、一番良い点が本当に過ぎたと確かめるのに十分な長さです。
- **予定を短くした理由:** 学習率は、予定全体にわたってコサインの曲線で下がります。800ステップでは一番良い点(125〜225ステップ)の時もまだ高く、400ステップならそこで既に下がっているので、モデルがより穏やかに落ち着きます。
- **今日テスト用のデータを使わない理由:** 大きさ・ドロップアウト・種など、すべての選択を検証用のデータで行います。テスト用のデータも選ぶのに使うと、その点数は「見たことのない文での力」の正直な値ではなくなります。Day 13 で最後に1回だけ使います。
- **結果が Day 11 と同じに見えるかもしれない理由:** Day 11 で、上限を決めるのは設定ではなく、学習用の文章の量と種類だと分かりました。3つの種でそれを確かめる本番の学習も、必要な証拠です。

## Day 12 の終わりのチェックリスト

- [ ] ファイルをコピーした(`src\training.py`、`src\generation.py`、`tests\test_training.py`、`scripts\day12_train_main.py`、`scripts\day12_monitor.py`、説明2つ)
- [ ] pytest: `74 passed`
- [ ] 最初の行が `tiny, dropout 0.1`
- [ ] 3つの種すべてが終わった(早期終了または最後まで)。`Final model:` の行が出た
- [ ] `docs\DAY12_MAIN_TRAINING.md`、`docs\figures\day12_main.png`、`configs\day12_final.json` がある
- [ ] D12 を実際の数字で追加し、コミットしてプッシュした。`runs\day12\` を残した

## 送り返すもの

1. pytest の結果の行
2. `--- generation check on each best.pt ---` から出力の最後まで
3. `docs\DAY12_MAIN_TRAINING.md`(ファイル全体、または添付)
4. `docs\figures\day12_main.png`
5. `configs\day12_final.json`

これを見て最終のモデルを確かめ、作業11の最後の日である Day 13 を準備します。

## 報告の例(`docs/weekly_reports.md` 用)

```
【課題11】LLMの学習、動作確認、生成結果の評価(続き)
状況: 予定通り終わった(今週の予定: 決めた大きさで本番の学習を行う)
内容: Day 11 で決めた設定(tiny、ドロップアウト0.1)で本番の学習を行った。乱数の種を3つ
      (42・43・44)使って、結果を平均とばらつきで示した。検証の損失が100ステップ改善しなければ
      学習を止める「早期終了」を学習の仕組みに加え、学習の予定を800ステップから400ステップに
      短くした。検証の損失が一番低い種のモデルを最終のモデルとした。テスト用のデータは
      最後の評価のために使っていない。自動テスト2項目を追加し、合計74項目すべて合格。
証拠: docs/DAY12_MAIN_TRAINING.md、docs/figures/day12_main.png、configs/day12_final.json
次: テスト用のデータで最終の評価を行い、評価の表を作る(Day 13、課題11の完了)
```
