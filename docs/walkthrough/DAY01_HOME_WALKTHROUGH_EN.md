# Day 1 (home laptop): environment, repository, GPU check

**Machine:** home laptop, RTX 4060 8 GB. **Repository:** `JC21-240447/llm_home` (private, new, empty). **Local folder:** `llm_home`.
All commands are for Windows PowerShell. The label says which window to use.

## What changes compared with the school PC, and why

| Item | School PC | Home laptop | Why |
| --- | --- | --- | --- |
| `--machine` / `-Machine` label | `school` | `home` | The run log must say which machine ran. |
| Repository | `2026_llm_jk3a` (public) | `llm_home` (private) | Each machine keeps its own evidence. |
| Precision plan | fp16 + GradScaler | bf16 (native on RTX 40-series) | bf16 has the range of fp32, so no loss scaling is needed. |
| Usable VRAM | about 5 GB (measured; Windows and other programs use the rest) | measured by the probe, do not assume 8 GB | Same reason: free VRAM at start is the budget. |
| Code | `day1_env_check.py`, `setup_env.ps1` | **identical files**; only the label differs | One code base keeps results comparable. |

## Step 0. Look before you touch (VS Code terminal: Terminal > New Terminal)

1. Close the popup "A git repository was found in the parent folders" with the **X** (do not press Yes, Always or Never yet). It means some folder above `llm_home` already contains a hidden `.git` folder. Find out which one:
   ```powershell
   git rev-parse --show-toplevel
   ```
   - It prints `fatal: not a git repository`: good, nothing above you. (If `git` itself is "not recognized", install Git first, see below.)
   - It prints a path such as `C:/Users/you`: a parent folder is already a repository. Send me the path before you run `git init`, because a repository inside a repository causes confusing status output.
2. Collect the facts the setup needs:
   ```powershell
   git --version
   py -0p
   nvidia-smi
   ```
3. Decide with this table:

| What you see | What to do |
| --- | --- |
| `py -0p` lists a `3.12` line | Good. Use `-PythonCmd "py -3.12"` (the default). |
| It lists only another version, e.g. 3.13 | Any 3.11 or newer works. Use that, e.g. `-PythonCmd "py -3.13"`. Tell me which. |
| `py` is not recognized | Install Python 3.12 from python.org (tick "Add python.exe to PATH" and "py launcher"), then reopen VS Code. |
| `nvidia-smi` shows a driver of 580 or higher | Use the same PyTorch index as at school (`https://download.pytorch.org/whl/cu132`). |
| Driver below 580, or `nvidia-smi` not found | Update the driver first (NVIDIA App, or the laptop maker's site). Do not install PyTorch yet. |
| `git` is not recognized | Install Git for Windows from git-scm.com with the default options, then reopen VS Code. |

The CUDA 13.2 index worked on the school PC because CUDA 13.x builds need a driver of 580 or higher. Whether the home driver is high enough is not known until you run `nvidia-smi`. If the pytorch.org selector offers a different CUDA version, use the URL it shows and tell me.

## Step 1. Put the starter files into `llm_home`

1. Download `phase1_home.zip` from this chat and unzip it.
2. Copy **everything inside** the unzipped `phase1_home` folder into `llm_home`, so that `llm_home\setup_env.ps1` and `llm_home\scripts\day1_env_check.py` exist (not `llm_home\phase1_home\setup_env.ps1`).
3. In VS Code check that the Explorer shows `scripts`, `docs`, `setup_env.ps1`, `README.md`, and so on directly under `llm_home`. Hidden `.gitignore` is also there.

## Step 2. Run the one-shot setup (a SEPARATE Windows PowerShell window, not the VS Code terminal)

1. In VS Code, right-click the `llm_home` folder in the Explorer > **Copy Path**.
2. Open Windows PowerShell from the Start menu. Run (paste your path inside the quotes):
   ```powershell
   cd "PASTE-THE-PATH-HERE"
   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Machine home -TorchIndexUrl https://download.pytorch.org/whl/cu132
   ```
3. The PyTorch download is large; on home internet it should be much faster than at school. Wait until you see `Done. Send me: run_log.md ...`.
4. If you chose another Python, add `-PythonCmd "py -3.13"` (your version) at the end.

The script creates the folders, builds `.venv`, installs PyTorch and the other packages, writes `requirements_lock_home.txt`, and runs `scripts\day1_env_check.py --machine home`.

## Step 3. Read the result

Open `run_log.md` (new entry) and the newest `runs\env_check_home_*.json`. Compare with school:

| Check | What a good result looks like |
| --- | --- |
| UTF-8 round trip | passes (Japanese text comes back unchanged) |
| Network | PyPI, PyTorch and GitHub reachable |
| GPU | the name contains 4060; compute capability 8.9 |
| Native bf16 | `True`, plan: `bf16 (no scaler)` |
| Benchmark | fp16 and bf16 faster than fp32 (the numbers are yours; do not copy school numbers) |
| VRAM probe | `usable_vram_gb` is what you can plan with; if `system_memory_spill_suspected` is `True`, plan with the free number, not 8 GB |
| AMP smoke test | passes |

If the script prints a NOTE about the GPU name, check the label. The log must be truthful.

## Step 4. Push to the new private repository (VS Code terminal, folder `llm_home`)

1. Make sure you are in the right folder: `dir` shows `setup_env.ps1`. Check again that Step 0 found no parent repository.
2. Replace the name and the no-reply email (GitHub > Settings > Emails; the same one as for the school repo is fine):
   ```powershell
   git init -b main
   git config --local user.name "YOUR NAME"
   git config --local user.email "YOUR-NOREPLY-EMAIL"
   git config --local core.quotepath false
   git status --short
   ```
   Read the list. You should see the starter files, `data/`, `src/` etc. (folders with `.gitkeep`), `requirements_lock_home.txt`, `runs/` (only the small `env_check_home_*.json`), and no `.venv`. If `.venv` or any `.pt` file appears, stop and tell me.
3. Add and commit the starter files by name:
   ```powershell
   git add .gitignore README.md LICENSES_AND_SOURCES.md requirements.txt setup_env.ps1 scripts docs data src configs tests
   git commit -m "Day 1 home: project skeleton and environment scripts"
   ```
4. Add the evidence in a second commit:
   ```powershell
   git add requirements_lock_home.txt run_log.md runs
   git status --short
   git commit -m "Day 1 home: environment check evidence"
   ```
5. Connect and push (the repository is empty, so this is a plain first push):
   ```powershell
   git remote add origin https://github.com/JC21-240447/llm_home.git
   git push -u origin main
   ```
   The first push opens a browser sign-in; use your own account. Success ends with `main -> main`.
6. In the browser, reload the repository page and check that the files and `runs/env_check_home_*.json` are there.

## Talking points for the review

- One code base, two machines. Only a label differs, so any difference in results comes from the hardware, not from the code.
- bf16 vs fp16: bf16 keeps fp32's number range, so training does not need loss scaling; fp16 does. Which one we use follows from the GPU generation, and the script detects it.
- The VRAM budget is measured at start, not read from the box. The laptop's display and other programs use part of it.
- Evidence rule: run logs and JSON files are committed, so every number in a report can be traced to a file.
- Private repository for the home machine: logs and experiments stay with the person who ran them.

## End of Day 1 (home) checklist

- [ ] Step 0 done; no parent repository found (or reported to me)
- [ ] Python 3.11+ and driver 580+ confirmed
- [ ] `setup_env.ps1 -Machine home` finished without errors
- [ ] `run_log.md` has a new `home` entry; GPU shows RTX 4060; native bf16 True
- [ ] `requirements_lock_home.txt` and `runs\env_check_home_*.json` exist
- [ ] Two commits pushed; files visible on GitHub
- [ ] No `.venv`, weights or big data pushed

## Send back

1. The text output of the setup (copy from PowerShell, as text).
2. A screenshot of the new `run_log.md` entry.
3. The output of `git rev-parse --show-toplevel`, `git --version`, `py -0p`, `nvidia-smi` from Step 0.
4. A screenshot of the successful push and of the GitHub page.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【Day 1(自宅ノートPC)】環境構築とGPU確認
状況: (予定通り終わった / 予定まで進まなかった / 手が付かなかった のどれか)
内容: 自宅ノートPC(RTX 4060)に Python と PyTorch の環境を作り、GPUが使えることを確認した。
結果: (run_log.md の数値を書き写す: GPU名、bf16対応、使えるVRAM、行列積の速度)
証拠: run_log.md、runs/env_check_home_*.json、requirements_lock_home.txt
次: Day 2(用語集と同義語の形式を決める)
```
