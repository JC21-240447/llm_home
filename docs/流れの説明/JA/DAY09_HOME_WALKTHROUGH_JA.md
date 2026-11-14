# Day 9(自宅ノートPC): 学習プログラム、最初の小さな学習、止めて再開する確認

**マシン:** 自宅ノートPC(RTX 4060 Laptop)。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**作業計画:** **作業10**(次の単語を予測する仕組み)の続き。今日の損失の曲線が、この仕組みが学べることの証拠になる。作業10は Day 10 で、次の単語の候補トップ10を示して完了する。

## Day 9 で行うこと、その理由

Day 8 でモデルを作り、正しくつながっていることを証明しました。今日、モデルが **初めて学習します**。Day 11 と Day 12 でも使う学習プログラムを作り、途中で止めて再開しても結果が変わらないことを証明します。

学習プログラムが1ステップごとに行うこと:

1. `train.bin` のランダムな位置から、256トークンの **区間** を32個取り出してバッチにする。各区間で、モデルは1〜256番目のトークンを読み、2〜257番目を予測する。つまり1つの区間から、256個の「次のトークン」の例が得られる。
2. 損失を **bf16**(4060 が得意な半分の大きさの数)で計算し、勾配を求める。
3. 勾配の大きさを制限し(**クリッピング**)、**AdamW** で重みを更新する。
4. **学習率** を決める: 最初の60ステップで上げ(ウォームアップ)、その後コサインの曲線で下げる。

10ステップごとに `log.csv` に1行書きます。100ステップごとに、検証用データ全体で **検証の損失** を測り、決まった4つの書き出しの続きを `samples.md` に書き、**チェックポイント** を保存します。

新しいファイルと変更したファイル:

| ファイル | 内容 |
| --- | --- |
| `src\batching.py` | ランダムな学習用の区間(乱数の状態を保存できる)と、決まった検証用の区間。メモリマップでディスクから読む |
| `src\training.py` | 学習ループ: 学習率の予定、重み減衰のグループ分けをした AdamW、クリッピング、勾配の累積、bf16/fp16、検証、サンプル、チェックポイント、再開、安全な Ctrl+C |
| `scripts\day9_train.py` | 新しい学習、または `--resume` で再開するためのコマンド |
| `scripts\day9_compare_runs.py` | 止めなかった学習と、止めて再開した学習を比べ、損失の図と報告を作る |
| `tests\test_training.py` | 新しいテスト8項目(「止めて再開 = 止めなかった場合、最後の桁まで同じ」を含む) |
| `requirements.txt` | `matplotlib`(図のため)を追加 |

1つの学習は `runs\day9\<名前>\` に保存されます(Git には入れません):

| ファイル | 内容 |
| --- | --- |
| `train_config.json`、`model_config.json` | その学習の設定 |
| `log.csv` | ステップ、損失、パープレキシティ、学習率、勾配の大きさ、トークン/秒、GPU メモリ、時間 |
| `samples.md` | 評価のたびに、「営業部の」「ボーナスが」「大阪支社で」「社員の平均」の後にモデルが書く続き |
| `last.pt` | 最新のチェックポイント(再開に使う) |
| `best.pt` | これまでで検証の損失が最も低いチェックポイント |

### 再開が正確な理由

`last.pt` には、重み、最適化の状態(AdamW の移動平均)、ステップ数、乱数の状態、学習用の区間を選ぶ乱数の状態を保存します。そのため、再開した学習は、止めなかった学習と同じ区間・同じ学習率で続きます。CPU では損失が最後の桁まで同じになることを、テストで証明しています。

安全のための工夫が2つあります:
- チェックポイントは、いったん一時ファイルに書いてから名前を変えます。保存中に止まっても、壊れた `last.pt` が残ることはありません。
- プログラムが強制終了された場合(ウィンドウを閉じた、電源が切れた)、`log.csv` に最後のチェックポイントより後の行が残ることがあります。再開するときにそれらの行を消して学習し直すので、記録はきれいなままです。

### サーバーで使う場合の注意(後日)

同じコードが Linux で動きます: `.venv/bin/python scripts/day9_train.py --machine server --run-name ...`。bf16 が使えない GPU(RTX 2060)では、`--precision auto` が自動で fp16 と勾配のスケーラーを選びます。サーバーでは、接続を切っても学習が続くように `tmux` の中で長い学習を始めるとよいです。サーバーの準備のときに説明します。

---

## Step 1. ファイルを置き、matplotlib をインストールする

1. `day9.zip` をダウンロードして展開する。
2. `day9` フォルダーの **中身すべて** を `llm_home` にコピーする。Windows に聞かれたら `requirements.txt` は **置き換える**(「ファイルを置き換える」)。新しいファイルは最後に1行を足しただけです。
3. matplotlib を **venv の中に** インストールする(VS Code ターミナル、`llm_home`):
   ```powershell
   .venv\Scripts\python.exe -m pip install "matplotlib>=3.8"
   .venv\Scripts\python.exe -c "import matplotlib; print(matplotlib.__version__)"
   ```
   2行目でバージョン(例: 3.11.x)が表示されれば成功です。
4. 確認:
   ```powershell
   dir src, tests, scripts\day9_*.py
   ```
   `src` に `batching.py` と `training.py` も入っていること。`tests` にテストファイルが4つあること。

## Step 2. すべてのテストを実行する(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: **`54 passed`**(46+新しい8項目)。新しいテストは、とても小さなモデルを CPU で数ステップ学習させるので、10秒ほどかかります。

| 新しいテスト | 証明すること |
| --- | --- |
| `test_lr_schedule_warmup_then_cosine` | ウォームアップで最大の学習率に達し、その後は下がるだけで、最後は最大の10%になる |
| `test_weight_decay_only_on_matrices` | 重み減衰は行列にかけ、正規化の係数にはかけない。すべてのパラメータがちょうど1つのグループに入る |
| `test_random_windows_shift_and_resume_state` | 正解は入力を1トークンずらしたもの。保存した乱数の状態から、同じ次のバッチが出る |
| `test_validation_windows_cover_the_stream_without_overlap` | 検証では、すべてのトークンを1回ずつ、毎回同じ方法で測る |
| `test_stopped_and_resumed_run_equals_uninterrupted_run` | **6ステップで止めて再開し、最後まで学習: すべての損失が、止めなかった学習と同じ** |
| `test_resume_refuses_changed_settings` | 設定を変えて再開しようとすると拒否する(もう同じ学習ではないため) |
| `test_new_run_refuses_an_existing_folder` | 新しい学習が、前の学習をうっかり上書きしない |
| `test_resume_after_a_hard_kill_…` | 強制終了で残った行を、再開のときに消す |

## Step 3. 最初の学習(止めずに最後まで)(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_a
```

初期設定: tiny モデル、バッチ 32 × 256トークン、600ステップ、ウォームアップ60、最大の学習率 0.001。約490万トークンで、学習用の文章を約2.4回読む量です。あなたの GPU なら **1分かからない** はずです。

見るところ:
- **step 0:** 検証の損失が約 **7.4**。Day 8 と同じく、学習前のモデルが均等に予想している状態です。
- 学習の損失は最初の100〜200ステップで速く下がり、その後ゆっくり下がります。
- 検証の損失も下がりますが、学習の損失より **高いまま** です。これは予想どおりです: 検証用には、学習で一度も見ない文型が入っています(Day 5)。この差の大きさを、Day 11 で調べます。

私の CPU では短い版(バッチ8、300ステップ、トークン数は4分の1)を実行し、検証の損失は 7.45 から約 2.72 に、学習の損失は約 1.28 になりました。あなたの学習はもっと多くの文章を見るので、数字はいくらか低くなるはずです。結果を送ってください。

次に `runs\day9\smoke_a\samples.md` を見てください。step 0 では、モデルは1つの部分を繰り返すだけです(「ののののの…」)。最後には、本当の質問の終わり方を書きます。私の CPU での例:

```text
| 大阪支社で | 働く社員の月収の中央値を出してください。 ⟨eos⟩ |
| 営業部の   | 社員を教えてください。 ⟨eos⟩ |
| ボーナスが | 3000000000000000000000000000000000000000 |
```

最後の行は、いつも一番点数の高いトークンを選ぶ **貪欲法(greedy)** によくある弱点です。一度繰り返しに入ると抜け出せません。Day 10 で、サンプリングと繰り返しの制御で直します。

## Step 4. 同じ学習を途中で止めて再開する(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --stop-after 300
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --resume
```

1つ目のコマンドは step 300 で止まり、`Stopped at step 300. Continue with --resume.` と表示します。2つ目は `Resumed from last.pt at step 300` と表示して、600まで続けます。`--resume` のときは、設定を学習のフォルダーから読むので、もう一度書く必要はありません。

## Step 5. 2つの学習を比べる(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day9_compare_runs.py --a smoke_a --b smoke_b
```

期待する結果:

```text
Final step: smoke_a 600, smoke_b 600 (resumed at 300)
  same final step              PASS
  training losses match        PASS
  validation losses match      PASS
  no jump at the resume step   PASS
Largest difference between the runs: ... -> identical   (または: the same curve)
```

私の CPU では差はちょうど0でした(identical)。GPU では、高速な注意の逆伝播の一部などで、足し算の順番が実行のたびに少し変わることがあります。そのため GPU では、ごく小さな0でない差になり、「the same curve」と表示されることがあります。どちらも、再開が正しく動いているという意味です。FAIL または DIFFERENT と出たら、出力を送ってください。

VS Code で `docs\DAY9_SMOKE_REPORT.md` を開き、**Ctrl+Shift+V** でプレビューを表示してください。図 `docs\figures\day9_smoke_loss.png` では、2本の学習の曲線が、step 300 の点線をまたいでもぴったり重なっています。**この図が、今日の作業10の主な証拠です。**

## Step 6(任意、1分). 自分で Ctrl+C で止める

学習中にノートPCを使いたくなった、という実際の場面を試します。

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_c --max-steps 3000
```

数秒後に **Ctrl+C を1回** 押します。今のステップを終え、`last.pt` を保存して止まります。その後:

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_c --resume
```

止めたところから続きます。Ctrl+C を **2回** 押すのは、すぐに止める必要があるときだけにしてください。その場合は、最後に保存したチェックポイントから続きます。

## Step 7. 記録してコミットする

`docs\DECISIONS.md` に追加:

```markdown
## D9. Trainer (Day 9)

- Training examples: random 256-token windows from the train stream (x = window, y = window shifted by one).
  Validation: fixed non-overlapping windows covering the whole validation stream.
- AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), linear warm-up then cosine decay to 10% of the peak,
  gradient clipping at 1.0, bf16 autocast on the RTX 4060 (fp16 + gradient scaler on GPUs without bf16).
- Checkpoints hold the model, optimizer, scaler, step and all random states; a stopped and resumed run gives the
  same losses as an uninterrupted run (test on CPU: identical; smoke run on GPU: see docs/DAY9_SMOKE_REPORT.md).
- Smoke run settings: tiny model, batch 32 x 256, 600 steps, warm-up 60, peak lr 1e-3.
```

その後コミットする(VS Code ターミナル、`llm_home`)。`runs` フォルダーは Git に入りません。報告と図は入れます:

```powershell
git status --short
git add requirements.txt src scripts tests docs run_log.md
git status --short
git commit -m "Day 9 home: trainer, smoke run, stop/resume check"
git push
```

一覧に `.pt` ファイルや `runs/` の中のものが出てはいけません。

## 審査で説明するポイント

- **損失とは:** 各位置で、モデルは本当の次のトークンに確率を付ける。損失は −log(その確率) の平均。正しいトークンに確率1を付ければ損失は0。損失7.42は、1,674個すべてに均等に予想を分けた状態。
- **パープレキシティとは:** e^損失。「モデルが平均でまだ何個のトークンの中で迷っているか」。最初は約1,674で、私の CPU の試験では検証で約15になった。低いほど、確かで、正しいことが多い。
- **ウォームアップの理由:** 最初は重みがランダムで、勾配が大きく不安定。小さな学習率から始めて60ステップで上げると、最初の更新が暴れない。最後のコサインの減少で、モデルが良い位置に落ち着く。
- **AdamW と、正規化の係数に重み減衰をかけない理由:** AdamW は重みごとに歩幅を調整する。重み減衰は大きな行列を少しずつ0に近づけて丸暗記を減らすが、正規化の係数は1の近くにあるべき倍率なので、対象から外す。
- **勾配クリッピングの理由:** 1つの変わったバッチが大きな勾配を出し、重みを壊すことがある。クリッピングで、1回の更新の大きさに上限を付ける。
- **bf16 の理由:** 半分の大きさの数で、学習が速くなり、メモリも減る。bf16 は float32 と同じ範囲を持つので、損失のスケーリングが要らない。bf16 の無い RTX 2060 では、fp16 と勾配のスケーラーを使う。
- **検証の損失が学習の損失より高い理由:** 検証用は文型ごと分けている(Day 5)ので、覚えているかではなく、見たことのない言い方をどれだけ扱えるかを測る。これが正直な測り方で、Day 11 でモデルの大きさを選ぶのに使う。
- **再開が正確でなければならない理由:** 長い学習(Day 12)は途中で止まることがある。再開でバッチや学習率が変わると、同じ実験ではなくなる。すべての乱数の状態を保存して同じにし、テストと図でそれを証明している。

## Day 9 の終わりのチェックリスト

- [ ] ファイルをコピーし、`requirements.txt` を置き換え、matplotlib を `.venv` にインストールした
- [ ] pytest: `54 passed`
- [ ] `smoke_a` が600ステップで終わり、検証の損失が約7.4から下がった
- [ ] `smoke_b` が300で止まり、再開して600まで進んだ
- [ ] 比較: PASS が4つ、結果は「identical」または「the same curve」
- [ ] `docs\DAY9_SMOKE_REPORT.md` と `docs\figures\day9_smoke_loss.png` がある
- [ ] (任意)Ctrl+C で止めて再開した
- [ ] D9 を追加し、コミットとプッシュをした。`.pt` ファイルは Git に入っていない

## 送り返すもの

1. pytest の結果の行。
2. Step 3 の最後の10行(最後の損失)。
3. Step 5 の出力の全部(比較)。
4. `runs\day9\smoke_a\samples.md` の最後の部分(step 600 のサンプル)。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題10】次に続く単語を予測する仕組み(途中)
状況: 予定通り終わった(今週の予定: 学習プログラムの作成と小さな学習)
内容: 学習プログラムを作成し、自宅ノートPCの GPU で小さなモデルの学習を行った。損失が学習前の約7.4から
      下がり、次の単語を予測できるようになっていくことを確認した。途中で止めて再開しても、止めなかった
      場合と同じ損失の曲線になることを確認した(自動テスト8項目を追加、合計54項目すべて合格)。
証拠: docs/DAY9_SMOKE_REPORT.md、docs/figures/day9_smoke_loss.png、run_log.md、tests/
次: 次の単語の候補トップ10を表示する仕組みを作り、課題10を完了する(Day 10)
```
