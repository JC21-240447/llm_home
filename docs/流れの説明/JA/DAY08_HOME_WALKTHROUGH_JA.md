# Day 8(自宅ノートPC): モデル(小さなデコーダーのみの Transformer)とそのテスト

**マシン:** 自宅ノートPC(RTX 4060 Laptop)。**リポジトリ:** `JC21-240447/llm_home`。**フォルダー:** `C:\Users\Merrick\Desktop\llm_home`。
**作業計画:** **作業10**(次の単語を予測する仕組み)を **開始** する。作業10は Day 10 で完了する(Day 9 で最初の小さな学習を行い、Day 10 で次の単語の候補トップ10を示す)。

## Day 8 で行うこと、その理由

Day 6 と Day 7 で、すべての質問を番号にしました。今日は、その番号から学ぶ部分、つまり **Model A 本体** を作ります。モデルはトークン番号の列を読み、各位置で、1,674個のトークンそれぞれに「次に来る可能性の高さ」の点数を付けます。今日はまだ **学習しません**(学習は Day 9)。今日は、モデルが正しく作られていることを証明します:

1. **モデルのコード**(`src\model.py`): デコーダーのみの Transformer(GPT 型)を、PyTorch で自分たちで書いたもの。
2. **新しい自動テスト12項目**(`tests\test_model.py`): 計画で求められている2つを含む。**因果性のテスト**(後ろのトークンが前の位置の点数を変えないこと)と、**1バッチを覚えるテスト**(小さな1つのバッチを覚えられること)。
3. **実際のデータと GPU での確認**(`scripts\day8_model_check.py`): 3つの試す大きさのパラメータ数・速さ・GPU メモリと、`train.bin` の実際のトークンでの3つの確認。

新しいコード:

| ファイル | 内容 |
| --- | --- |
| `src\model.py` | モデル本体: 設定(`ModelConfig`)、3つの大きさ、RMSNorm、回転位置埋め込み、因果的な注意(手書き版と高速版)、MLP、ブロック、重み共有、パラメータ数の計算 |
| `scripts\day8_model_check.py` | 3つの大きさを作り、GPU で学習1ステップの時間を測り、実データで確認し、`configs\model_*.json` を保存し、報告と実行ログを書く |
| `tests\test_model.py` | テスト12項目(形、最初の損失、因果性×2、手書き=高速、重み共有、パラメータの式、大きさの範囲、回転位置、長さの上限、設定の保存、1バッチを覚える) |

計画ファイルではモデルのファイル名を `day8_model.py` としていますが、これまでの Day と同じ形にそろえます: 何度も使うコードは `src\`、その Day のスクリプトは `scripts\`、テストは `tests\`。Day 9〜11 は `src\model.py` を読み込んで使います。

### 設計を1つの表で

| 部分 | 選んだもの | 理由 |
| --- | --- | --- |
| 全体の形 | デコーダーのみの Transformer(GPT 型)、プレノルム、残差接続 | 「次のトークンを予測する」ための、標準的でよく知られた設計 |
| 正規化 | RMSNorm(float32 で計算) | LayerNorm より簡単で、bf16 でも安定する |
| 位置 | RoPE(回転位置埋め込み) | クエリとキーのベクトルを回転させて位置を表すので、注意は2つのトークンが「どれだけ離れているか」で決まる。学習する位置の表が無いので、1つの長さに縛られない |
| 注意 | 因果的(前だけを見る)。読みやすい手書き版と、PyTorch の高速な `scaled_dot_product_attention` | 手書き版は読んで説明するため、高速版は学習のため。同じ結果になることをテストで証明する |
| MLP | 幅の4倍、GELU、バイアスなし | 標準的 |
| 出力層 | トークン埋め込みと重みを共有(重み共有) | 「トークン→ベクトル」と「ベクトル→トークンの点数」に1つの行列を使う。パラメータが減り、2つの見方がそろう |
| 初期値 | 正規分布(0, 0.02)。各ブロックの2つの出力の射影は 1/√(2×層の数) 倍 | ブロックを重ねても信号が大きくなりすぎないようにする |
| ブロックサイズ | 256トークン(重みではなく設定) | 一番長い質問は28トークンなので、1つの区間に質問がいくつも丸ごと入る。Day 9 で確かめる |

### 3つの試す大きさ(元の計画からの変更と、その理由)

計画では約2,500万・6,000万・1億パラメータを想定していました。これは語彙16,000と、もっと多い文章の量を前提にした数字です。実際は **語彙1,674**(Day 6)、**学習用2,028,601トークン**(Day 7)です。2,500万のモデルでは、学習用のトークン1つあたり12個以上のパラメータになり、質問をただ丸暗記してしまいます。そこで、試す大きさをデータに合わせました:

| 名前 | 層 | ヘッド | 幅 | パラメータ数 | 1パラメータあたりの学習トークン数 |
| --- | --- | --- | --- | --- | --- |
| tiny | 4 | 4 | 128 | 1,001,856 | 2.02 |
| small | 6 | 4 | 256 | 5,150,464 | 0.39 |
| medium | 8 | 6 | 384 | 14,805,120 | 0.14 |

Day 11 では、計画どおり **検証用の損失** で大きさを選びます。変わったのは候補だけです。

### サーバーで使う場合の注意(後日)

モデルは PyTorch だけで書いているので、Linux でもそのまま動きます。サーバーでは `/` と Linux の venv のパスを使います: `.venv/bin/python scripts/day8_model_check.py --machine server`。パラメータ数とテストの結果は同じにならなければなりません。速さとメモリは GPU によって変わります。bf16 が使えない GPU では、スクリプトが自動で、時間の測定に fp16、覚えるテストに fp32 を使います。

---

## Step 1. ファイルを置く

1. このチャットから `day8.zip` をダウンロードして展開する。
2. 展開した `day8` フォルダーの **中身すべて** を `llm_home` にコピーする: `src\model.py` は `llm_home\src` へ、`scripts\day8_model_check.py` は `llm_home\scripts` へ、`tests\test_model.py` は `llm_home\tests` へ、2つの説明ファイルは `docs\流れの説明\EN` と `docs\流れの説明\JA` へ。
3. 確認(VS Code ターミナル、`llm_home`):
   ```powershell
   dir src, tests, scripts\day8_*.py
   ```
   `src` に `__init__.py`、`bpe_tokenizer.py`、`token_data.py`、`model.py` があること。`tests` にテストファイルが3つあること。

## Step 2. PyTorch が GPU を使えるか確かめる(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

期待する結果(Day 1 と同じ): `2.15.0+cu132 True NVIDIA GeForce RTX 4060 Laptop GPU True`。`False` と出たら止めて、出力を送ってください。

## Step 3. すべてのテストを実行する(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

期待する結果: **`46 passed`**(Day 6・7 の34項目+新しい12項目)。新しいテストは CPU で動き、数秒で終わります。テストを名前ごとに見たいときは `-v` を付けます。

新しい12項目が証明すること:

| テスト | 証明すること |
| --- | --- |
| `test_output_shapes_and_loss` | 各位置で語彙のトークンごとに1つの点数を出し、損失を1つの数で出す |
| `test_initial_loss_is_close_to_uniform_guess` | 学習前の損失が ln(語彙の大きさ) に近い。モデルは何も好みが無い状態から始まる |
| `test_causality_…`(×2、手書き版と高速版) | 12番目のトークンを変えても、0〜11番目の位置の点数は変わらない。モデルは先を見られない |
| `test_manual_and_fast_attention_agree` | 読みやすい注意と高速な注意が、同じ計算をしている |
| `test_weight_tying` | 出力層と埋め込みが同じ行列である |
| `test_parameter_count_matches_formula` | PyTorch で数えたパラメータ数が、手で式を書いた数と一致する |
| `test_presets_are_in_the_planned_size_range` | 3つの大きさが約100万・500万・1,500万である |
| `test_rope_depends_only_on_relative_position` | 回転位置埋め込みでは、クエリとキーの内積が2つのトークンの距離だけで決まる |
| `test_too_long_input_is_rejected` | ブロックサイズより長い列には、分かりやすいエラーを出す |
| `test_config_round_trip` | ファイルに保存した設定が同じに読み戻せる。ありえない設定は拒否される |
| `test_overfit_one_batch` | 小さなモデルが小さな1バッチを覚える(損失が0.1未満になる)。モデル・損失・勾配が正しくつながっている |

## Step 4. GPU と実際のデータでモデルを確かめる(VS Code ターミナル、`llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day8_model_check.py --machine home
```

1分もかかりません。出力は次のようになります。パラメータ数と最初の損失は **まったく同じ** にならなければなりません。速さとメモリはあなたの GPU の数字なので、私には予想できません:

```text
=== Day 8 model check | machine=home | NVIDIA GeForce RTX 4060 Laptop GPU | bf16 | PyTorch 2.15.0+cu132 ===
Vocabulary 1,674; training tokens 2,028,601; block size 256

preset       params tokens/param   ms/step    tokens/s  peak MB
tiny      1,001,856         2.02       ...         ...      ...
small     5,150,464         0.39       ...         ...      ...
medium   14,805,120         0.14       ...         ...      ...

Loss before training: 7.422  (ln 1674 = 7.423)
Manual vs fast attention: max difference ...e-07   (1e-3 未満なら合格)
Memorise one batch (4 x 64): loss 7.42… -> 0.0…
...
All checks PASS.
```

私の CPU では、覚えるテストは300ステップで 7.424 → 0.0031 でした。あなたの GPU は bf16 で計算するので、最後の桁は違います。0.5 未満なら合格で、0.01 前後なら普通です。

このスクリプトは `configs\model_tiny.json`、`model_small.json`、`model_medium.json` も保存し(Day 9 と Day 11 が読む)、`docs\DAY8_MODEL_REPORT.md` を書き、`run_log.md` に追記します。

## Step 5. 報告を読み、3つのことを理解する

1. **最初の損失 = ln(1,674) ≈ 7.42。** 学習前のモデルは、1,674個のトークンに均等に予想を分けます。均等な予想の損失は ln(1,674) です。ちょうどこの数になることは、初期値が正しい証拠です。Day 9 からは、7.42 より下がった分がモデルの学んだことです。
2. **1バッチを覚えられる。** 1つの決まったバッチで、損失が約7.42から0近くまで下がります。モデル・マスク・損失に誤りがあれば、これはできません。本当の学習の前に行う、標準的な最初のテストです。
3. **1パラメータあたりのトークン数。** tiny は約2、small は約0.4、medium は約0.14です。パラメータに対して文章がこれだけ少ないと、大きいモデルは一般化せずに丸暗記しやすくなります。Day 11 で検証用の損失を使って確かめます。

## Step 6. 記録してコミットする

`docs\DECISIONS.md` に追加:

```markdown
## D8. Model architecture and trial sizes (Day 8)

- Decoder-only Transformer, own code (src/model.py): pre-norm RMSNorm, rotary positions (RoPE), causal attention
  (manual + fast path, proven equal by a test), GELU MLP 4x, no biases, weight tying, scaled residual init.
- Block size 256 tokens (longest question 28 tokens); a setting, not part of the weights.
- Trial sizes changed from ~25M/60M/100M to ~1.0M/5.2M/14.8M (tiny/small/medium): vocabulary is 1,674 (not 16k)
  and train is 2,028,601 tokens, so the planned sizes would have 12+ parameters per training token.
- Day 11 still chooses the size by validation loss.
```

その後コミットする(VS Code ターミナル、`llm_home`)。新しい `configs` フォルダーは小さいので Git に入れます:

```powershell
git status --short
git add src scripts tests configs docs run_log.md
git status --short
git commit -m "Day 8 home: Model A (decoder-only Transformer), tests, model check"
git push
```

一覧に `data/` が出てはいけません。

## 審査で説明するポイント

- **注意(Attention)を簡単な言葉で:** 各位置で、モデルは質問(クエリ)を出し、前のトークンはそれぞれラベル(キー)と中身(バリュー)を出す。クエリと各キーを比べ、その点数を合計1の重みにし(ソフトマックス)、中身を重みで混ぜる。だから「年収」は、質問の前にある「営業部」の情報を取り込める。
- **因果マスクの理由:** モデルは「次の」トークンを予測するように学ぶ。位置 t が t+1 のトークンを見られたら、学ばずに答えを写してしまう。マスクで後ろの位置をすべて隠し、因果性のテストでそれを証明する。
- **重み共有の理由:** 埋め込みはトークンをベクトルにし、出力層はベクトルをトークンの点数に戻す。1つの行列を両方に使うと 214,272 パラメータ(tiny の21%)を節約でき、各トークンの2つの見方がそろう。
- **回転位置埋め込みの理由:** 注意だけでは語順が分からない。RoPE はクエリとキーを位置とともに大きくなる角度で回すので、2つの一致の度合いは距離で決まる。両方の位置を同じだけずらしても点数が同じになることを、テストで示している。
- **注意を2つ書いた理由:** 手書き版は短く、審査で1行ずつ説明できる。学習には高速版を使う。同じ出力になること(差は約10⁻⁷)をテストで証明している。
- **大きさを計画より小さくした理由:** モデルの大きさはデータの量に合わせる必要がある。学習用は200万トークンと測ったので、候補を約100万・500万・1,500万にし、Day 11 で「大きいほど良い」ではなく検証用の損失で選ぶ。
- **学習の前にテストする理由:** 最初の損失の確認と1バッチを覚える確認で、つなぎ方の誤りを数秒で見つけられる。これが無いと、誤りは何時間も後に「学習がうまくいかない」という形でしか見えない。

## Day 8 の終わりのチェックリスト

- [ ] ファイルをコピーした(`src\model.py`、`scripts\day8_model_check.py`、`tests\test_model.py`、説明ファイル2つ)
- [ ] GPU の確認: `True … RTX 4060 … True`
- [ ] pytest: `46 passed`
- [ ] モデルの確認: パラメータ数 1,001,856 / 5,150,464 / 14,805,120、最初の損失 7.422、注意の差 1e-3 未満、覚えるテスト 0.5 未満、`All checks PASS`
- [ ] `configs\model_*.json` と `docs\DAY8_MODEL_REPORT.md` がある
- [ ] D8 を追加し、コミットとプッシュをした

## 送り返すもの

1. Step 3 の pytest の結果の行。
2. Step 4 のコンソール出力の全部(テキストで)。Day 9 のバッチサイズを決めるのに、あなたの実際の ms/step、tokens/s、peak MB が必要です。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【課題10】次に続く単語を予測する仕組み(開始)
状況: 予定通り終わった(今週の予定: 課題10の開始)
内容: 次の単語を予測するモデル(デコーダーのみの Transformer)を自分たちのコードで作成した。
      12項目の自動テスト(未来の単語を見ないこと、1つのバッチを覚えられること など)がすべて合格し、
      自宅ノートPCの GPU で3つの大きさのモデルの動作と速さを確認した。
証拠: src/model.py、tests/test_model.py、docs/DAY8_MODEL_REPORT.md、configs/、run_log.md
次: 学習プログラムを作り、小さな学習を実行する(Day 9)。課題10は Day 10 で完了する。
```
