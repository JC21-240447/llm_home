# Day 1(自宅ノートPC): 環境・リポジトリ・GPU確認

**マシン:** 自宅ノートPC、RTX 4060 8 GB。**リポジトリ:** `JC21-240447/llm_home`(Private、新規、空)。**ローカルのフォルダー:** `llm_home`。
コマンドはすべて Windows PowerShell 用です。どのウィンドウを使うかは、見出しに書いてあります。

## 学校PCとの違いと、その理由

| 項目 | 学校PC | 自宅ノートPC | 理由 |
| --- | --- | --- | --- |
| `--machine` / `-Machine` ラベル | `school` | `home` | どのマシンで実行したかを実行ログに残すため。 |
| リポジトリ | `2026_llm_jk3a`(Public) | `llm_home`(Private) | マシンごとに証拠を分けて持つため。 |
| 精度の方針 | fp16 + GradScaler | bf16(RTX 40シリーズは対応) | bf16 は fp32 と同じ数値の範囲を持つので、ロススケーリングが要らない。 |
| 使えるVRAM | 約5 GB(実測。残りはWindowsなどが使用) | プローブで実測する。8 GB と決めつけない | 同じ理由で、開始時の空きVRAMが予算になる。 |
| コード | `day1_env_check.py`、`setup_env.ps1` | **同じファイル**。ラベルだけが違う | コードを1つにして、結果を比較できるようにする。 |

## Step 0. 触る前に確認する(VS Codeターミナル: ターミナル > 新しいターミナル)

1. 「親フォルダーに git リポジトリが見つかりました」というポップアップは、**X** で閉じます(はい / 常に / しない は、まだ押さない)。`llm_home` より上のどこかのフォルダーに、隠しフォルダー `.git` があるという意味です。どこか調べます:
   ```powershell
   git rev-parse --show-toplevel
   ```
   - `fatal: not a git repository` と出る: 上には何もないので、問題ありません。(`git` 自体が「認識されません」と出る場合は、下の表のとおり先に Git を入れます。)
   - `C:/Users/you` のようなパスが出る: 親フォルダーがすでにリポジトリです。`git init` の前に、そのパスを私に送ってください。リポジトリの中にリポジトリがあると、状態の表示が混乱します。
2. セットアップに必要な情報を集めます:
   ```powershell
   git --version
   py -0p
   nvidia-smi
   ```
3. 次の表で判断します:

| 表示 | すること |
| --- | --- |
| `py -0p` に `3.12` の行がある | OK。`-PythonCmd "py -3.12"`(既定)を使う。 |
| 別のバージョンだけ(例: 3.13) | 3.11 以上ならどれでも使える。`-PythonCmd "py -3.13"` のように指定し、バージョンを私に知らせる。 |
| `py` が認識されない | python.org から Python 3.12 を入れる(「Add python.exe to PATH」と「py launcher」にチェック)。その後 VS Code を開き直す。 |
| `nvidia-smi` のドライバーが 580 以上 | 学校と同じ PyTorch の index(`https://download.pytorch.org/whl/cu132`)を使う。 |
| ドライバーが 580 未満、または `nvidia-smi` が無い | 先にドライバーを更新する(NVIDIA アプリ、またはPCメーカーのサイト)。PyTorch はまだ入れない。 |
| `git` が認識されない | git-scm.com から Git for Windows を、既定のオプションのまま入れる。その後 VS Code を開き直す。 |

学校PCで CUDA 13.2 の index が使えたのは、CUDA 13.x のビルドがドライバー 580 以上を必要とし、学校のドライバーがそれを満たしていたからです。自宅のドライバーが足りているかどうかは、`nvidia-smi` を実行するまで分かりません。pytorch.org のセレクターが別の CUDA バージョンを示した場合は、そのURLを使い、私に知らせてください。

## Step 1. スターターファイルを `llm_home` に入れる

1. このチャットから `phase1_home.zip` をダウンロードして展開する。
2. 展開した `phase1_home` フォルダーの**中身をすべて** `llm_home` にコピーする。`llm_home\setup_env.ps1` と `llm_home\scripts\day1_env_check.py` がある状態にする(`llm_home\phase1_home\setup_env.ps1` ではない)。
3. VS Code のエクスプローラーで、`scripts`、`docs`、`setup_env.ps1`、`README.md` などが `llm_home` の直下に見えることを確認する。隠しファイルの `.gitignore` もここにあります。

## Step 2. ワンショットのセットアップを実行する(VS Codeターミナルではなく、別の Windows PowerShell)

1. VS Code のエクスプローラーで `llm_home` フォルダーを右クリック > **パスのコピー**。
2. スタートメニューから Windows PowerShell を開き、次を実行する(引用符の中に自分のパスを貼る):
   ```powershell
   cd "ここにパスを貼る"
   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Machine home -TorchIndexUrl https://download.pytorch.org/whl/cu132
   ```
3. PyTorch のダウンロードは大きいですが、自宅の回線なら学校よりずっと速いはずです。`Done. Send me: run_log.md ...` と出るまで待ちます。
4. 別の Python を選んだ場合は、末尾に `-PythonCmd "py -3.13"`(自分のバージョン)を付けます。

スクリプトは、フォルダーの作成、`.venv` の作成、PyTorch とその他のパッケージのインストール、`requirements_lock_home.txt` の作成、`scripts\day1_env_check.py --machine home` の実行までを行います。

## Step 3. 結果を読む

`run_log.md`(新しいエントリー)と、最新の `runs\env_check_home_*.json` を開きます。学校と比べます:

| 確認項目 | 良い結果 |
| --- | --- |
| UTF-8 の読み書き | 成功(日本語がそのまま戻る) |
| ネットワーク | PyPI、PyTorch、GitHub に接続できる |
| GPU | 名前に 4060 を含む。compute capability は 8.9 |
| bf16 のネイティブ対応 | `True`、方針は `bf16 (no scaler)` |
| ベンチマーク | fp16 と bf16 が fp32 より速い(数値は自分のもの。学校の数値を写さない) |
| VRAM プローブ | `usable_vram_gb` が計画に使える値。`system_memory_spill_suspected` が `True` なら、8 GB ではなく空きの値で計画する |
| AMP スモークテスト | 成功 |

GPU名についての NOTE が出たら、ラベルを確認します。ログは事実どおりでなければなりません。

## Step 4. 新しい Private リポジトリへプッシュする(VS Codeターミナル、フォルダーは `llm_home`)

1. 正しいフォルダーにいるか確認する: `dir` に `setup_env.ps1` が見える。Step 0 で親リポジトリが無かったことも、もう一度確認する。
2. 名前と no-reply メールを置き換えて実行する(GitHub > Settings > Emails。学校のリポジトリと同じもので構いません):
   ```powershell
   git init -b main
   git config --local user.name "YOUR NAME"
   git config --local user.email "YOUR-NOREPLY-EMAIL"
   git config --local core.quotepath false
   git status --short
   ```
   一覧を読みます。スターターファイル、`data/` や `src/` など(`.gitkeep` 入りのフォルダー)、`requirements_lock_home.txt`、`runs/`(小さな `env_check_home_*.json` だけ)が見え、`.venv` は見えないはずです。`.venv` や `.pt` ファイルが出たら、止めて私に知らせてください。
3. スターターファイルを、名前を指定して追加し、コミットする:
   ```powershell
   git add .gitignore README.md LICENSES_AND_SOURCES.md requirements.txt setup_env.ps1 scripts docs data src configs tests
   git commit -m "Day 1 home: project skeleton and environment scripts"
   ```
4. 証拠のファイルを、2回目のコミットで追加する:
   ```powershell
   git add requirements_lock_home.txt run_log.md runs
   git status --short
   git commit -m "Day 1 home: environment check evidence"
   ```
5. 接続してプッシュする(リポジトリは空なので、普通の最初のプッシュです):
   ```powershell
   git remote add origin https://github.com/JC21-240447/llm_home.git
   git push -u origin main
   ```
   最初のプッシュでブラウザーのサインインが開くので、自分のアカウントでサインインします。成功すると `main -> main` で終わります。
6. ブラウザーでリポジトリのページを再読み込みし、ファイルと `runs/env_check_home_*.json` があることを確認する。

## レビューで話すポイント

- コードは1つで、マシンは2台。違うのはラベルだけなので、結果の違いはコードではなくハードウェアから来る。
- bf16 と fp16: bf16 は fp32 と同じ数値の範囲を持つのでロススケーリングが要らず、fp16 は要る。どちらを使うかはGPUの世代で決まり、スクリプトが自動で判定する。
- VRAM の予算は、箱の数字ではなく、開始時に実測する。ノートPCの画面表示や他のプログラムが一部を使う。
- 証拠のルール: 実行ログとJSONをコミットするので、報告書のすべての数値をファイルまでたどれる。
- 自宅のマシンは Private リポジトリ: ログや実験は、実行した本人のもとに残る。

## Day 1(自宅)の終わりのチェックリスト

- [ ] Step 0 を実行した。親リポジトリは無かった(または私に報告した)
- [ ] Python 3.11 以上と、ドライバー 580 以上を確認した
- [ ] `setup_env.ps1 -Machine home` がエラーなく終わった
- [ ] `run_log.md` に `home` の新しいエントリーがあり、GPUは RTX 4060、bf16 のネイティブ対応は True
- [ ] `requirements_lock_home.txt` と `runs\env_check_home_*.json` がある
- [ ] 2回のコミットをプッシュし、GitHub でファイルが見える
- [ ] `.venv`、重み、大きなデータはプッシュしていない

## 送り返すもの

1. セットアップの出力テキスト(PowerShell からテキストでコピー)。
2. 新しい `run_log.md` エントリーのスクリーンショット。
3. Step 0 の `git rev-parse --show-toplevel`、`git --version`、`py -0p`、`nvidia-smi` の出力。
4. プッシュ成功の画面と、GitHub のページのスクリーンショット。

## 報告用の文(日本語、`docs/weekly_reports.md` 用)

```
【Day 1(自宅ノートPC)】環境構築とGPU確認
状況: (予定通り終わった / 予定まで進まなかった / 手が付かなかった のどれか)
内容: 自宅ノートPC(RTX 4060)に Python と PyTorch の環境を作り、GPUが使えることを確認した。
結果: (run_log.md の数値を書き写す: GPU名、bf16対応、使えるVRAM、行列積の速度)
証拠: run_log.md、runs/env_check_home_*.json、requirements_lock_home.txt
次: Day 2(用語集と同義語の形式を決める)
```
