# Day 1 ウォークスルー(日本語): 環境構築・クリーンなリポジトリ・GPU確認

今回の実行マシン: **学校PC(RTX 2060, 6 GB)**
GitHubリポジトリ: `https://github.com/JC21-240447/2026_llm_jk3a`(オーナー `JC21-240447`)
ローカルフォルダー: `LLM\phase1_day1`(このフォルダーがリポジトリのルートになります)
エディター: Visual Studio Code(VS Code)
対応する学校タスク: 2(言語・ライブラリの記録)と 7(PC・プログラム環境整備)、報告日は 11/5

この文書の表記ルール:
- 「VS Codeターミナル」は VS Code 内のターミナルです(メニュー: ターミナル > 新しいターミナル / Terminal > New Terminal)。Windows では既定で PowerShell です。別の Windows PowerShell ウィンドウでも、ここに書いたコマンドはすべて動きます。
- コマンドのブロックごとに、どこで実行するかを書いています。
- Windows の画面名は 日本語 / English の順で書きます。

---

## 0. 終わったときの状態

```
phase1_day1/                      <- リポジトリのルート("2026_llm_jk3a")
  .gitignore
  .vscode/settings.json           (手順6で作成)
  README.md
  LICENSES_AND_SOURCES.md
  run_log.md                      (スクリプトが追記します)
  requirements.txt
  requirements_lock_school.txt    (セットアップスクリプトが作成)
  setup_env.ps1
  data/  src/  configs/  tests/   (セットアップスクリプトが作成)
  runs/env_check_school_<時刻>.json
  scripts/day1_env_check.py
  docs/DECISIONS.md
  docs/weekly_reports.md
  docs/evidence_day1_school.png   (あなたが撮るスクリーンショット)
  docs/walkthrough/DAY01_WALKTHROUGH_EN.md / _JA.md
  .venv/                          (GitHubにはアップロードしません)
```

---

## 1. 手順(ステップごと)

### 手順1. インストール状況を確認する(約5分)

VS Code を開き、フォルダーを開きます。
1. メニュー **ファイル > フォルダーを開く / File > Open Folder...**
2. `LLM\phase1_day1` を選びます(スクリーンショットにあった、`docs`、`scripts`、`.gitignore`、`setup_env` などが入っているフォルダーです)。
3. 「このフォルダー内のファイルの作成者を信頼しますか? / Do you trust the authors...」と聞かれたら **はい、作成者を信頼します / Yes, I trust the authors** を選びます(自分のフォルダーなので問題ありません)。
4. ターミナルを開きます: **ターミナル > 新しいターミナル / Terminal > New Terminal**。

次の3つを実行します(**VS Codeターミナル**):

```powershell
git --version
py -0p
nvidia-smi
```

それぞれの意味:

| コマンド | 期待する結果 | 失敗した場合 |
| --- | --- | --- |
| `git --version` | バージョン番号 | Git が入っていません。学校の管理者に確認するか、権限があれば「Git for Windows」を入れてください。私に知らせてください。 |
| `py -0p` | Python の一覧。**3.12** があるのが理想(最低でも 3.11) | 3.11 以上が無ければ、一覧を私に送ってください。 |
| `nvidia-smi` | 「RTX 2060」、**Driver Version**、**CUDA Version** が書かれた表 | ドライバーが無い場合は知らせてください。直るまでこのPCではGPUを使えません。 |

メモしておくもの: GPU名、VRAM、Driver Version、CUDA Version(`nvidia-smi` の表の右上)。CUDA Version は手順3で使います。

インストールが禁止されている場合(管理者権限なし、pip が通らないなど)は、ここで止めて私に知らせてください。計画を変更します(ポータブル版 Python、または先生への許可の相談など)。学校のセキュリティを回避しようとしないでください。

### 手順2. VS Code の拡張機能を入れる

拡張機能を開きます: `Ctrl+Shift+X`。次を入れます:
- **Python**(発行元: Microsoft)
- **Pylance**(Microsoft)
- 任意: **Markdown All in One**(この手順書を読みやすく表示するため)

Markdown のプレビュー: ファイルを開いて `Ctrl+Shift+V`。

### 手順3. PyTorch のダウンロード先(index URL)を決める

ここだけは私が埋められませんでした。現在の PyTorch のインストールコマンドを公式ページで確認できなかったためです。あなたのルールでは、バージョンに依存する情報を記憶で答えてはいけません。そのため、公式ページから自分で読み取ります。

1. ブラウザーで https://pytorch.org/get-started/locally/ を開きます。
2. **Stable**、**Windows**、**Pip**、**Python**、そして **CUDA** の項目を選びます。
3. `pip3 install torch ... --index-url https://download.pytorch.org/whl/cuXXX` のようなコマンドが表示されます。
4. `--index-url` の後ろのアドレスだけをコピーします(例: `https://download.pytorch.org/whl/cuXXX`)。これが `TorchIndexUrl` です。

CUDA の選び方: **ドライバーが対応している**ものを選びます。`nvidia-smi` に出る `CUDA Version` は、そのドライバーが動かせる最大のバージョンです。それ以下のバージョンを選んでください。選択肢が新しすぎる場合は私に聞いてください。学校PCのドライバーが古いことが、「cuda_available = False」の一番ありそうな原因です。

同じページの Python に関する注意書きも読んでください。最新の PyTorch は Python 3.10 以上(ページによっては 3.11 以上)を求めることがあります。使えるなら Python 3.12 が無難です。

### 手順4. セットアップスクリプトを実行する

**VS Codeターミナル**で実行します(ターミナルが `phase1_day1` フォルダーで開いていること。プロンプトが `phase1_day1>` で終わっているはずです):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Machine school -TorchIndexUrl <手順3のURLを貼り付け>
```

スクリプトが順に行うこと(各ステップが表示されます):
1. `nvidia-smi` を表示します。
2. 使う Python のバージョンを表示します(既定は `py -3.12`。3.11 しか無い場合は `-PythonCmd "py -3.11"` を追加)。
3. `data`、`src`、`configs`、`scripts`、`runs`、`docs`、`tests` フォルダーを作ります。
4. 仮想環境 `.venv` を作ります。
5. 指定した index URL から PyTorch をインストールします。
6. `requirements.txt` から `numpy`、`psutil`、`tqdm`、`pytest` を入れます。
7. 入った正確なバージョンを `requirements_lock_school.txt` に保存し、環境チェックを実行します。

PyTorch のダウンロードは大きいです(約2 GB 以上)。待ってください。ターミナルを閉じないでください。

`-ExecutionPolicy Bypass` はこの1回のコマンドだけに効きます。Windows の設定は変更しません。それでも学校PCがブロックする場合は、手動版を実行してください:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url <手順3のURL>
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding ascii requirements_lock_school.txt
.\.venv\Scripts\python.exe scripts\day1_env_check.py --machine school
```

### 手順5. チェック結果を読む

すべて正常なら、最後に次のように表示されます:

```
[PASS] python_3_10_plus
[PASS] running_in_venv
[PASS] torch_importable
[PASS] cuda_available
[PASS] fp16_amp_ok
[PASS] utf8_roundtrip
```

| 行 | 意味 | FAIL の場合 |
| --- | --- | --- |
| `python_3_10_plus` | Python が十分新しい | 新しい Python を入れる |
| `running_in_venv` | プロジェクトの `.venv` の Python で実行できた | `.\.venv\Scripts\python.exe scripts\day1_env_check.py --machine school` で実行する |
| `torch_importable` | PyTorch が正しく入った | エラー文を私に送る |
| `cuda_available` | PyTorch がGPUを認識できる | ドライバーに合わない CUDA 版、またはドライバーが古い。`nvidia-smi` と使った index URL を送る |
| `fp16_amp_ok` | ロススケーリング付きの fp16 学習が実際に動いた。2060ではこれが精度の計画 | エラー文を私に送る |
| `utf8_roundtrip` | 日本語を保存して読み戻せた | 以降のすべての日に影響するので知らせる |

あわせて次も確認します:
- `NOTE: --machine school usually has an RTX 2060...`: GPUが違うときに出ます。ラベルが事実と合っているか確認してください。run_log は事実だけを書く必要があります。
- `NOTE: network targets not reachable`: 学校のネットワークが PyPI、PyTorch のサーバー、GitHub のいずれかをブロックしています。どれか教えてください。
- ログの精度の行: RTX 2060 では `Native bf16: False -> plan: fp16 + GradScaler` が正常です。
- `free at start (usable budget)`: プローブ前に実際に空いているVRAMです(6 GB のうち一部は Windows や開いているアプリが使っています)。Day 8〜11 でバッチサイズを決めるのに使います。`spill to system memory suspected: True` は、ドライバーがプローブを実VRAMの外(遅い共有メモリ)にまで逃がしたことを意味します。その場合プローブの「allocated」の数字は本当の予算ではないので、「free at start」の数字を信じてください。

### 手順6. VS Code でインタープリターを選ぶ

1. `Ctrl+Shift+P` を押し、`Python: インタープリターを選択`(英語UIでは `Python: Select Interpreter`)と入力して Enter。
2. `.venv\Scripts\python.exe` で終わる項目を選びます(「推奨 / Recommended」や「ワークスペース / Workspace」と表示されます)。
3. ファイル `.vscode/settings.json` を作ります(エクスプローラーの空白を右クリック > 新しいフォルダー `.vscode`、その中に新しいファイル `settings.json`)。内容はそのまま:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "files.encoding": "utf8",
  "files.eol": "\n"
}
```

4. ターミナルを閉じて(ゴミ箱アイコン)、新しく開きます。プロンプトが `(.venv)` で始まるはずです。
5. 確認(**VS Codeターミナル**):

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`.venv` 内のパス、PyTorch のバージョン、`True`、GPU名が表示されれば成功です。

Windows のUIが日本語なので、`files.encoding: utf8` が重要です。これが無いと、ツールによっては日本語を別の文字コードで保存し、後の日で問題が起きます。

### 手順7. 学校PCでの証拠を残す

1. VS Code ターミナルにチェックの結果を表示します(画面が消えていたら `.\.venv\Scripts\python.exe scripts\day1_env_check.py --machine school` をもう一度実行します。`run_log.md` に2件目が追記されますが、事実の記録なので問題ありません)。
2. `Win+Shift+S`(スクリーンショット / Snipping)で、GPU名、Python、PyTorch の行が写る範囲を選びます。
3. `docs\evidence_day1_school.png` として保存します。
4. `run_log.md` に `school` と今日の日付のエントリがあることを確認します。VS Code で開いて読んでください。

### 手順8. GitHub リポジトリを作る(あなたのスクリーンショットの設定)

スクリーンショットでは、オーナー `JC21-240447`、名前 `2026_llm_jk3a`、公開設定 Public、README オフになっています。既存フォルダーを push する用途として正しい設定です。

そのページを下にスクロールして確認します:
- **Add README: Off**(自分たちの README を使うため)
- **Add .gitignore: No .gitignore**(自分たちの .gitignore を使うため)
- **Choose a license: None**(ライセンスは後で決め、`LICENSES_AND_SOURCES.md` に書きます)

そして **Create repository** を押します。

Public と Private: Public だと、push した内容(`run_log.md` や文書を含む)を誰でも読めます。このプロジェクトには秘密にすべきものが入らないので問題ありません。先生に見せる必要があるなら Public が一番簡単です。後からリポジトリの Settings で Private に変更できます。パスワード、トークン、個人情報は絶対にコミットしないでください。

### 手順9. フォルダーを GitHub につないで push する

**VS Codeターミナル**で、`phase1_day1` の中で実行します:

```powershell
git init -b main
git config --local user.name "あなたの名前"
git config --local user.email "GitHubのメールアドレス"
git add .
git status
```

注意:
- `--global` ではなく `--local` を使います。学校PCは共用なので、あなたの名前をこのリポジトリの中だけに留めるためです。
- `git status` の一覧に `.venv/` や `.pt` ファイルが**出てはいけません**。出ていたら止めて、出力を私に送ってください。

コミットして push します:

```powershell
git commit -m "Day 1: environment, repo skeleton, school PC check"
git remote add origin https://github.com/JC21-240447/2026_llm_jk3a.git
git push -u origin main
```

最初の push では、ブラウザーまたは「GitHub に接続」ウィンドウが開いてサインインを求められます。自分のアカウントでサインインします。パスワードは、その公式の GitHub サインイン画面以外には入力しないでください。共用の学校PCなので、終わったら Windows の **資格情報マネージャー / Credential Manager** を開き、`git:https://github.com` の項目を削除するか、ブラウザーで GitHub からサインアウトしてください。次の人があなたとして push できないようにするためです。

ブラウザーで `https://github.com/JC21-240447/2026_llm_jk3a` を開きます。ファイルが表示され、`.venv` が無ければ成功です。

push が失敗した場合:

| メッセージ | 意味 | 対処 |
| --- | --- | --- |
| `fatal: not a git repository` | フォルダーが違う | `phase1_day1` に `cd` してやり直す |
| `remote origin already exists` | `remote add` を2回実行した | `git remote set-url origin https://github.com/JC21-240447/2026_llm_jk3a.git` |
| `Authentication failed` | サインインの問題 | ブラウザーのウィンドウで再サインインする。ターミナルにパスワードを貼らない |
| `Could not resolve host: github.com` | 学校のネットワークが GitHub をブロック | 知らせてください。USBでコピーして、家から push します |

### 手順10. 報告用スニペットを記入する

`docs/weekly_reports.md` を開き、第4節の日本語スニペットを、実際の値を入れて貼り付けます。そして再度コミットします:

```powershell
git add .
git commit -m "Day 1: evidence and weekly report snippet"
git push
```

### 手順11. 家で(後で)

家のノートPC用に2つ目のリポジトリ(名前は自由。例: `2026_llm_jk3a_home`)を作り、同じファイルをコピーして、`-Machine home` で手順3〜9をやり直します。ドライバーが違う可能性があるので、手順3はノートPCでもう一度行ってください。コードは同じです。違いが出るのはラベルと PyTorch の index URL だけです。

---

## 2. 各ファイルの役割(ディフェンス用)

| ファイル | 役割 |
| --- | --- |
| `scripts/day1_env_check.py` | このマシンでGPU学習ができることを証明し、証拠を書き出す |
| `setup_env.ps1` | 同じ環境を繰り返し構築できるようにする |
| `requirements.txt` | PyTorch 以外のパッケージ(バージョンは緩め) |
| `requirements_lock_school.txt` | このPCに実際に入った正確なバージョン |
| `run_log.md` | 事実の記録: 日付、マシン、GPU、結果 |
| `.gitignore` | 重みや大きなデータを Git に入れない(SSDにバックアップする) |
| `LICENSES_AND_SOURCES.md` | 出典とライセンスの記録(Day 3 から記入) |
| `docs/DECISIONS.md` | 言語・ライブラリの決定(学校タスク2) |

## 3. 教員レビューで話すポイント

1. **なぜクリーンなリポジトリと固定のフォルダー構成にするのか。** 毎日1つのスクリプトを `scripts/` に、出力を `data/` か `runs/` に追加します。どの結果も、それを作ったコードまでたどれます。
2. **なぜ run log を付けるのか。** どのマシンで何を実行したかを記録します。長時間の学習は家で行い、学校PCはそれを再現します。そのため、家での作業を学校での作業として報告してはいけません。
3. **なぜバージョンを固定するのか。** PyTorch のバージョンが違うと数値結果が変わることがあります。ロックファイルがあれば、もう一方のPCで同じ環境を再構築できます。
4. **なぜ精度の経路が2つあるのか。** RTX 2060(compute capability 7.5)は bf16 をネイティブでサポートしないため、小さな勾配が 0 に丸められるのを防ぐロススケーラー付きの fp16 を使います。RTX 4060 は bf16 をネイティブでサポートするので、スケーラーは不要です。スクリプトは仮定せず、実際の学習ステップで該当する経路をテストします。
5. **なぜネットワークをテストするのか。** インストールや後のダウンロードがネットワークに依存し、学校のネットワークはブロックが起きやすい場所だからです。
6. **なぜ VRAM を測るのか。** 6 GB と 8 GB の違いが、後のバッチサイズとモデルサイズを決めます。仕様書を読むより測るほうが確実です。
7. **なぜリポジトリが2つなのか。** それぞれのログが自分のハードウェアだけを記録します。代償は、コードの変更を両方にコピーする必要があることです。だからコードは同一に保ち、設定だけを変えます。

## 4. Day 1 終了チェックリスト

- [ ] GitHub リポジトリ `JC21-240447/2026_llm_jk3a` を作成(README オフ、.gitignore なし、ライセンスなし)
- [ ] `git --version`、`py -0p`、`nvidia-smi` の出力を保存した
- [ ] PyTorch の index URL を公式ページからコピーした(推測していない)
- [ ] `setup_env.ps1` がエラーなく終わった
- [ ] 環境チェックの6行がすべて PASS(または失敗内容を私に送った)
- [ ] VS Code のインタープリターが `.venv` で、プロンプトに `(.venv)` が出る
- [ ] `.vscode/settings.json` を作成した
- [ ] `run_log.md` に `school` のエントリがある
- [ ] `requirements_lock_school.txt` がある
- [ ] スクリーンショットを `docs/evidence_day1_school.png` として保存した
- [ ] 最初のコミットを push し、GitHub にファイルが表示され `.venv` が無い
- [ ] 共用PCから GitHub の認証情報を削除した
- [ ] 日本語の報告スニペットを `docs/weekly_reports.md` に貼った

## 5. 私に送ってほしいもの

1. 環境チェックのターミナル出力(テキストをコピー、または `run_log.md` のエントリ)。
2. `runs/env_check_school_*.json`。
3. `git --version`、`py -0p`、`nvidia-smi` の出力。
4. 使った PyTorch の index URL。
5. エラーがあれば、そのまま(写し直した要約ではなくテキストで)。
6. 次の質問への回答: ノートPCの Windows と Python のバージョン、ネイティブ Windows か Ubuntu/WSL2 か、自宅で学習する件について先生の回答があるか、週に使える集中時間。

## 6. 日本語の報告スニペット(学校への報告、第1週)

学校PCで実際に実行してから使ってください。かっこの中は実際の値に置き換えます。

```
【タスク7: PC・プログラム環境整備】状況: 予定通り終わった
成果物: 学校PC(RTX 2060)にPython [バージョン]とPyTorch [バージョン]の環境を構築し、day1_env_check.pyでGPU認識とfp16学習の動作を確認した。言語・ライブラリ選定はdocs/DECISIONS.mdに記載。
証拠: run_log.md、runs/env_check_school_*.json、docs/evidence_day1_school.png
次: 用語集と同義語ファイルの形式を決める(Day 2)。
```

何かがブロックされた場合は、状況を 予定まで進まなかった にして、本当の理由を書いてください。

## 7. トラブルシューティング早見表

| 症状 | 考えられる原因 | 対処 |
| --- | --- | --- |
| `py` が認識されない | Python ランチャーが未インストール | `python --version` を試し、結果を私に知らせる |
| `pip` がサーバーに届かない | 学校のプロキシやファイアウォール | 知らせてください。別の方法を使います |
| `nvidia-smi` は動くのに `cuda_available False` | ドライバーに合わない CUDA 版 | PyTorch のページで古い CUDA を選び、手順4をやり直す(先に `.venv` を削除) |
| ターミナルで日本語が文字化けする | コンソールの文字コード | スクリプトは自分の出力を UTF-8 に固定しています。それでも化けたら知らせてください |
| `.venv` がインタープリターに出ない | VS Code が再スキャンしていない | `Ctrl+Shift+P` > `Developer: Reload Window` |
| 「スクリプトの実行が無効」と出る | PowerShell のポリシー | 手順4の手動コマンドを使う |

環境を最初からやり直す場合: VS Code のターミナルをすべて閉じ、`.venv` フォルダーを削除して、手順4をやり直します。
