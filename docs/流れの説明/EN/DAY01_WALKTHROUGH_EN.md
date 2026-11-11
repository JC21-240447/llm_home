# Day 1 Walkthrough (English): Environment, clean repo, GPU check

Machine for this run: **school PC (RTX 2060, 6 GB)**
GitHub repository: `https://github.com/JC21-240447/2026_llm_jk3a` (owner `JC21-240447`)
Local folder: `LLM\phase1_day1` (this folder becomes the repository root)
Editor: Visual Studio Code (VS Code)
School task covered: 2 (language and library note) and 7 (PC and program setup), report on 11/5

Conventions in this file:
- "VS Code terminal" means the terminal inside VS Code (menu: Terminal > New Terminal / ターミナル > 新しいターミナル). On Windows it is PowerShell by default. A separate Windows PowerShell window also works for every command here.
- Every command block says where to run it.
- Windows UI names are given as English / Japanese.

---

## 0. What you will have at the end

```
phase1_day1/                      <- repository root (the "2026_llm_jk3a" repo)
  .gitignore
  .vscode/settings.json           (you create it in Step 6)
  README.md
  LICENSES_AND_SOURCES.md
  run_log.md                      (the script appends entries)
  requirements.txt
  requirements_lock_school.txt    (created by the setup script)
  setup_env.ps1
  data/  src/  configs/  tests/   (created by the setup script)
  runs/env_check_school_<time>.json
  scripts/day1_env_check.py
  docs/DECISIONS.md
  docs/weekly_reports.md
  docs/evidence_day1_school.png   (your screenshot)
  docs/walkthrough/DAY01_WALKTHROUGH_EN.md / _JA.md
  .venv/                          (not uploaded to GitHub)
```

---

## 1. Step-by-step

### Step 1. Check what is installed (5 minutes)

Open VS Code, then open the folder:
1. Menu **File > Open Folder... / ファイル > フォルダーを開く**
2. Choose `LLM\phase1_day1` (the folder you showed in the screenshot, containing `docs`, `scripts`, `.gitignore`, `setup_env`, and so on).
3. If VS Code asks "Do you trust the authors of the files in this folder? / このフォルダー内のファイルの作成者を信頼しますか?", choose **Yes, I trust the authors / はい、作成者を信頼します**. It is your own folder.
4. Open the terminal: **Terminal > New Terminal / ターミナル > 新しいターミナル**.

Now run these three checks (**VS Code terminal**):

```powershell
git --version
py -0p
nvidia-smi
```

What each one tells you:

| Command | What you want to see | If it fails |
| --- | --- | --- |
| `git --version` | a version number | Git is not installed. Ask the school admin, or install "Git for Windows" if you have rights. Tell me. |
| `py -0p` | a list of Pythons. You want **3.12** (3.11 is the minimum I will accept) | If there is no Python 3.11 or newer, tell me the list. |
| `nvidia-smi` | a table with "RTX 2060", a **Driver Version**, and a **CUDA Version** | Driver missing: tell me; GPU work cannot happen on this PC until the driver is fixed. |

Write down: GPU name, VRAM, Driver Version, CUDA Version (top right of the `nvidia-smi` table). You need the CUDA Version in Step 3.

If installs are blocked (no admin rights, no internet for pip), stop here and tell me. The plan then changes (portable Python or asking the teacher for permission). Do not try to work around school security.

### Step 2. Install the VS Code extensions

Open Extensions: `Ctrl+Shift+X`. Install:
- **Python** (publisher: Microsoft)
- **Pylance** (Microsoft)
- optional: **Markdown All in One** (to read these walkthrough files nicely)

To preview a Markdown file: open it, press `Ctrl+Shift+V`.

### Step 3. Choose the PyTorch download address (the "index URL")

This is the one step I could not fill in for you. I was unable to verify the current PyTorch install command against the official page, and your file says never to rely on memory for this. So you read it from the official page:

1. Open https://pytorch.org/get-started/locally/ in a browser.
2. Select: **Stable**, **Windows**, **Pip**, **Python**, and a **CUDA** option.
3. The page shows a command like `pip3 install torch ... --index-url https://download.pytorch.org/whl/cuXXX`.
4. Copy only the address after `--index-url` (for example `https://download.pytorch.org/whl/cuXXX`). This is your `TorchIndexUrl`.

How to choose the CUDA option: pick one that your driver supports. The `CUDA Version` shown by `nvidia-smi` is the highest the driver can run. Choose that version or an older one. If the selector offers only a newer CUDA than your driver supports, ask me. An old school driver is the most likely cause of "cuda_available = False".

Also read the Python note on that page. Current PyTorch may require Python 3.10 or newer (some pages say 3.11). Python 3.12 is a safe choice if it is available.

### Step 4. Run the setup script

Run in **VS Code terminal** (the terminal must be open in the `phase1_day1` folder; the prompt should end with `phase1_day1>`):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Machine school -TorchIndexUrl <PASTE THE URL FROM STEP 3>
```

What the script does, in order (it prints each step):
1. Shows `nvidia-smi`.
2. Prints the Python version it will use (default `py -3.12`; if you only have 3.11, add `-PythonCmd "py -3.11"`).
3. Creates the folders `data`, `src`, `configs`, `scripts`, `runs`, `docs`, `tests`.
4. Creates the virtual environment `.venv`.
5. Installs PyTorch from your index URL.
6. Installs `numpy`, `psutil`, `tqdm`, `pytest` from `requirements.txt`.
7. Saves the exact installed versions to `requirements_lock_school.txt` and runs the environment check.

The PyTorch download is large (about 2 GB or more). Wait for it. Do not close the terminal.

`-ExecutionPolicy Bypass` affects only this one command. It does not change a Windows setting. If the school PC blocks it anyway, run the manual version:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url <URL FROM STEP 3>
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding ascii requirements_lock_school.txt
.\.venv\Scripts\python.exe scripts\day1_env_check.py --machine school
```

### Step 5. Read the check result

The last lines look like this when everything is fine:

```
[PASS] python_3_10_plus
[PASS] running_in_venv
[PASS] torch_importable
[PASS] cuda_available
[PASS] fp16_amp_ok
[PASS] utf8_roundtrip
```

| Line | Meaning | If FAIL |
| --- | --- | --- |
| `python_3_10_plus` | Python is new enough | install a newer Python |
| `running_in_venv` | the script ran with the project's `.venv` Python | run it as `.\.venv\Scripts\python.exe scripts\day1_env_check.py --machine school` |
| `torch_importable` | PyTorch installed correctly | send me the error text |
| `cuda_available` | PyTorch can see the GPU | wrong CUDA wheel for the driver, or old driver. Send me `nvidia-smi` and the index URL you used |
| `fp16_amp_ok` | a real fp16 training step with loss scaling works. This is the precision plan for the 2060 | send me the error text |
| `utf8_roundtrip` | Japanese text can be saved and read back | tell me, it affects every later day |

Also look at:
- `NOTE: --machine school usually has an RTX 2060...`: it appears if the GPU is different. Check that the label is true. The run log must say the truth.
- `NOTE: network targets not reachable`: the school network blocks PyPI, the PyTorch server or GitHub. Tell me which.
- The precision line in the log: `Native bf16: False -> plan: fp16 + GradScaler` is expected on the RTX 2060.
- `free at start (usable budget)` is the VRAM that is really free before the probe (Windows and open apps already use some of the 6 GB). Days 8 to 11 use this to set batch sizes. `spill to system memory suspected: True` means the driver let the probe go beyond real VRAM into slow shared memory; the probe's "allocated" number is then not a real budget, so trust the "free at start" number.

### Step 6. Select the interpreter in VS Code

1. Press `Ctrl+Shift+P`, type `Python: Select Interpreter` (Japanese UI: `Python: インタープリターを選択`) and press Enter.
2. Choose the entry ending in `.venv\Scripts\python.exe` (marked "Recommended / 推奨" or "Workspace / ワークスペース").
3. Create the file `.vscode/settings.json` (right-click empty space in the Explorer > New Folder `.vscode`, then New File `settings.json`) with exactly:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "files.encoding": "utf8",
  "files.eol": "\n"
}
```

4. Close the terminal (trash icon) and open a new one. The prompt should now start with `(.venv)`.
5. Confirm (**VS Code terminal**):

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see a path inside `.venv`, a PyTorch version, `True`, and the GPU name.

`files.encoding: utf8` matters because your Windows UI is Japanese: without it, some tools save Japanese text in a different encoding and later days break.

### Step 7. Take the school-PC evidence

1. Make the VS Code terminal show the check summary (run `.\.venv\Scripts\python.exe scripts\day1_env_check.py --machine school` again if the screen was cleared; it appends a second entry to `run_log.md`, which is fine and honest).
2. Press `Win+Shift+S` (screenshot / スクリーンショット), select the area showing the GPU name, Python and PyTorch lines.
3. Save it as `docs\evidence_day1_school.png`.
4. Check that `run_log.md` has an entry with `school` and today's date. Open it in VS Code and read it.

### Step 8. Create the GitHub repository (from your screenshot)

Your screenshot shows: owner `JC21-240447`, name `2026_llm_jk3a`, visibility Public, README off. That is the right setup for pushing an existing folder.

Scroll down on that page and check:
- **Add README: Off** (we have our own).
- **Add .gitignore: No .gitignore** (we have our own).
- **Choose a license: None** for now. The license is decided later and written in `LICENSES_AND_SOURCES.md`.

Then press **Create repository**.

Public or private: with Public, anyone can read everything you push, including `run_log.md` and your documents. That is fine for this project because nothing secret goes in it. If your teacher needs to see it, Public is simplest; you can switch to Private later in the repository Settings. Never commit passwords, tokens or personal data.

### Step 9. Connect the folder to GitHub and push

Run in **VS Code terminal**, inside `phase1_day1`:

```powershell
git init -b main
git config --local user.name "YOUR NAME"
git config --local user.email "YOUR GITHUB EMAIL"
git add .
git status
```

Notes:
- Use `--local`, not `--global`. The school PC is shared, so your name must stay inside this repo only.
- `git status` should list the files and must NOT list `.venv/` or any `.pt` file. If it does, stop and send me the output.

Commit and push:

```powershell
git commit -m "Day 1: environment, repo skeleton, school PC check"
git remote add origin https://github.com/JC21-240447/2026_llm_jk3a.git
git push -u origin main
```

On the first push a browser window or a "Connect to GitHub" window opens for sign-in. Sign in with your own account. Do not type a password anywhere except that official GitHub sign-in page. Because this is a shared school PC, after you finish open Windows **Credential Manager / 資格情報マネージャー** and remove the `git:https://github.com` entry, or sign out of GitHub in the browser, so the next person cannot push as you.

Check in the browser: open `https://github.com/JC21-240447/2026_llm_jk3a`. You should see the files and no `.venv`.

If the push fails:

| Message | Meaning | Fix |
| --- | --- | --- |
| `fatal: not a git repository` | wrong folder | `cd` into `phase1_day1` and repeat |
| `remote origin already exists` | you ran `remote add` twice | `git remote set-url origin https://github.com/JC21-240447/2026_llm_jk3a.git` |
| `Authentication failed` | sign-in problem | sign in again in the browser window; do not paste passwords into the terminal |
| `Could not resolve host: github.com` | school network blocks GitHub | tell me; we use a USB copy and push from home |

### Step 10. Fill in the report snippet

Open `docs/weekly_reports.md` and paste the Japanese snippet from Section 4 with real values. Then commit again:

```powershell
git add .
git commit -m "Day 1: evidence and weekly report snippet"
git push
```

### Step 11. At home (later)

Make a second repository for the home laptop (name of your choice, for example `2026_llm_jk3a_home`), copy the same files in, and repeat Steps 3 to 9 with `-Machine home`. Do Step 3 again on the laptop because the driver may differ. The code is the same; only the label and the PyTorch index URL can differ.

---

## 2. Explanation of what each file is for (defense material)

| File | Purpose |
| --- | --- |
| `scripts/day1_env_check.py` | Proves the machine can run GPU training; writes evidence |
| `setup_env.ps1` | Builds the same environment repeatably |
| `requirements.txt` | Packages other than PyTorch (loose versions) |
| `requirements_lock_school.txt` | The exact versions that were installed on this PC |
| `run_log.md` | Honest record: date, machine, GPU, result |
| `.gitignore` | Keeps weights and big data out of Git (back them up to the SSD) |
| `LICENSES_AND_SOURCES.md` | Source and license log, filled from Day 3 |
| `docs/DECISIONS.md` | Language and library decision (school task 2) |

## 3. Talking points for the faculty review

1. **Why a clean repository with a fixed layout.** Each day adds one script to `scripts/` and outputs to `data/` or `runs/`. Any result can be traced to the code that made it.
2. **Why a run log.** It records which machine ran what. The long training runs at home and the school PC reproduces it, so reports must never claim home work as school work.
3. **Why pin versions.** A different PyTorch version can change numerical results. The lock file lets the other PC rebuild the same environment.
4. **Why two precision paths.** The RTX 2060 (compute capability 7.5) has no native bf16, so it uses fp16 with a loss scaler to avoid tiny gradients rounding to zero. The RTX 4060 has native bf16, which needs no scaler. The script tests the real path with actual training steps instead of assuming.
5. **Why test the network.** Installs and later downloads depend on it, and a school network is the most likely place for a block.
6. **Why a VRAM probe.** 6 GB versus 8 GB decides batch size and model size later. Measuring is better than reading the spec sheet.
7. **Why two repositories.** Each log describes only its own hardware. The cost is that code changes have to be copied across, which is why code stays identical and only settings differ.

## 4. End of Day 1 checklist

- [ ] GitHub repo `JC21-240447/2026_llm_jk3a` created (README off, .gitignore none, license none)
- [ ] `git --version`, `py -0p`, `nvidia-smi` outputs saved
- [ ] PyTorch index URL copied from the official page (not guessed)
- [ ] `setup_env.ps1` finished without errors
- [ ] All six lines of the env check show PASS (or the failures are sent to me)
- [ ] VS Code interpreter is `.venv`, prompt shows `(.venv)`
- [ ] `.vscode/settings.json` created
- [ ] `run_log.md` has a `school` entry
- [ ] `requirements_lock_school.txt` exists
- [ ] Screenshot saved as `docs/evidence_day1_school.png`
- [ ] First commit pushed; GitHub page shows files and no `.venv`
- [ ] GitHub credentials removed from the shared PC
- [ ] Japanese report snippet pasted into `docs/weekly_reports.md`

## 5. What to send back to me

1. The terminal output of the env check (copy the text, or the `run_log.md` entry).
2. `runs/env_check_school_*.json`.
3. Output of `git --version`, `py -0p`, `nvidia-smi`.
4. Which PyTorch index URL you used.
5. Any error, copied exactly (text, not a retyped summary).
6. Answers to: laptop Windows and Python versions, native Windows or Ubuntu/WSL2, whether the professor has answered about training at home, and focused hours per week.

## 6. Japanese report snippet (school report, week 1)

Use it only after the run really happened on the school PC. Fill in the brackets.

```
【タスク7: PC・プログラム環境整備】状況: 予定通り終わった
成果物: 学校PC(RTX 2060)にPython [バージョン]とPyTorch [バージョン]の環境を構築し、day1_env_check.pyでGPU認識とfp16学習の動作を確認した。言語・ライブラリ選定はdocs/DECISIONS.mdに記載。
証拠: run_log.md、runs/env_check_school_*.json、docs/evidence_day1_school.png
次: 用語集と同義語ファイルの形式を決める(Day 2)。
```

If something was blocked, use 予定まで進まなかった and state the real reason.

## 7. Troubleshooting quick table

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `py` is not recognized | Python launcher not installed | try `python --version`; tell me the result |
| `pip` cannot reach the server | school proxy or firewall | tell me; we use another route |
| `cuda_available False` but `nvidia-smi` works | wrong CUDA wheel for the driver | pick an older CUDA option on the PyTorch page and rerun Step 4 (delete `.venv` first) |
| Garbled Japanese in the terminal | console encoding | the script forces UTF-8 for its own output; tell me if it still garbles |
| `.venv` not shown as an interpreter | VS Code has not rescanned | `Ctrl+Shift+P` > `Developer: Reload Window` |
| Script blocked by "running scripts is disabled" | PowerShell policy | use the manual commands in Step 4 |

Redo the environment from scratch: close VS Code terminals, delete the `.venv` folder, and rerun Step 4.
