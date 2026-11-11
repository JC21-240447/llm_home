# Day 8 (home laptop): the model (a small decoder-only Transformer) and its tests

**Machine:** home laptop (RTX 4060 Laptop). **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School plan:** **starts task 10** (the next-word mechanism). Task 10 is finished on Day 10 (Day 9 trains a first small run, Day 10 shows the top-10 next-word candidates).

## What Day 8 does, and why

Days 6 and 7 turned every question into numbers. Today we build the part that learns from those numbers: **Model A itself**. It reads a sequence of token numbers and, at every position, gives a score to each of the 1,674 tokens: "how likely is this token to come next?". Today the model is **not trained** yet (that is Day 9). Today we prove that it is built correctly:

1. **The model code** (`src\model.py`): a decoder-only Transformer (the GPT design), written by us with PyTorch.
2. **12 new automatic tests** (`tests\test_model.py`), including the two the plan asks for: the **causality test** (a later token must not change the scores of earlier positions) and the **overfit-one-batch test** (the model must be able to memorise one small batch).
3. **A check on the real data and the GPU** (`scripts\day8_model_check.py`): the three trial sizes with their parameter counts, speed and GPU memory, and three checks on real token windows from `train.bin`.

New code:

| File | What |
| --- | --- |
| `src\model.py` | the model: settings (`ModelConfig`), the three trial sizes, RMSNorm, rotary positions, causal attention (manual and fast versions), MLP, blocks, weight tying, parameter counter |
| `scripts\day8_model_check.py` | builds the three sizes, times one training step on the GPU, runs the real-data checks, saves `configs\model_*.json`, writes the report and the run log |
| `tests\test_model.py` | 12 tests (shapes, starting loss, causality ×2, manual = fast, weight tying, parameter formula, size ranges, rotary positions, length limit, saving settings, memorising one batch) |

The plan file names the model `day8_model.py`. We keep the repository pattern of the earlier days instead: the reusable code in `src\`, the Day script in `scripts\`, the tests in `tests\`. Days 9 to 11 import `src\model.py`.

### The design, in one table

| Part | Choice | Why |
| --- | --- | --- |
| Overall shape | decoder-only Transformer (GPT style), pre-norm, residual connections | the standard, well-understood design for "predict the next token" |
| Normalisation | RMSNorm, computed in float32 | simpler than LayerNorm, stable with bf16 |
| Positions | RoPE (rotary position embedding) | position is given by rotating query and key vectors, so attention depends on *how far apart* two tokens are. There is no learned position table, so nothing is tied to one fixed length |
| Attention | causal; a readable manual version and PyTorch's fast `scaled_dot_product_attention` | the manual one is for reading and defending; the fast one is for training; a test proves they give the same result |
| MLP | 4 × width, GELU, no biases | standard |
| Output layer | shares its weights with the token embedding (weight tying) | one matrix for "token → vector" and "vector → token score": fewer parameters and the two views stay consistent |
| Initialisation | normal(0, 0.02); the two output projections of each block scaled by 1/√(2 × layers) | keeps the signal from growing as blocks are added |
| Block size | 256 tokens (a setting, not fixed in the weights) | the longest question is 28 tokens, so one window holds many whole questions; Day 9 confirms |

### The three trial sizes (changed from the original plan, and why)

The plan suggested about 25M, 60M and 100M parameters. Those numbers assumed a 16,000-token vocabulary and a much larger text collection. Our real numbers are a **1,674-token vocabulary** (Day 6) and **2,028,601 training tokens** (Day 7). A 25M model would have more than 12 parameters for every training token and would simply memorise the questions. So the trial sizes are scaled to our data:

| Preset | Layers | Heads | Width | Parameters | Training tokens per parameter |
| --- | --- | --- | --- | --- | --- |
| tiny | 4 | 4 | 128 | 1,001,856 | 2.02 |
| small | 6 | 4 | 256 | 5,150,464 | 0.39 |
| medium | 8 | 6 | 384 | 14,805,120 | 0.14 |

Day 11 still chooses by **validation loss**, exactly as planned. Only the candidates changed.

### SERVER NOTE (later)

The model is plain PyTorch and runs unchanged on Linux. On the server, use `/` and the Linux venv path: `.venv/bin/python scripts/day8_model_check.py --machine server`. Parameter counts and the test results must be the same. Speed and memory depend on the GPU. On a GPU without bf16, the script times with fp16 and runs the memorisation check in fp32 by itself.

---

## Step 1. Put the files in place

1. Download `day8.zip` from this chat and unzip it.
2. Copy **everything inside** the unzipped `day8` folder into `llm_home`: `src\model.py` into `llm_home\src`, `scripts\day8_model_check.py` into `llm_home\scripts`, `tests\test_model.py` into `llm_home\tests`, and the two walkthroughs into `docs\流れの説明\EN` and `docs\流れの説明\JA`.
3. Check (VS Code terminal, `llm_home`):
   ```powershell
   dir src, tests, scripts\day8_*.py
   ```
   `src` must show `__init__.py`, `bpe_tokenizer.py`, `token_data.py` and `model.py`. `tests` must show three test files.

## Step 2. Check that PyTorch sees the GPU (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Expected (as on Day 1): `2.15.0+cu132 True NVIDIA GeForce RTX 4060 Laptop GPU True`. If it says `False`, stop and send me the output.

## Step 3. Run all tests (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: **`46 passed`** (the 34 from Days 6 and 7, plus 12 new). The new tests run on the CPU and take a few seconds. To see every test by name, add `-v`.

What the 12 new tests prove:

| Test | What it proves |
| --- | --- |
| `test_output_shapes_and_loss` | the model gives one score per vocabulary token at every position, and a single loss number |
| `test_initial_loss_is_close_to_uniform_guess` | before training, the loss is about ln(vocabulary size): the model starts with no preference |
| `test_causality_…` (×2, manual and fast) | changing token 12 does not change the scores at positions 0 to 11: the model cannot look ahead |
| `test_manual_and_fast_attention_agree` | the readable attention and the fast attention compute the same thing |
| `test_weight_tying` | the output layer and the embedding are the same matrix |
| `test_parameter_count_matches_formula` | the parameter count from PyTorch equals the count written out by hand |
| `test_presets_are_in_the_planned_size_range` | the three sizes are about 1M, 5M and 15M |
| `test_rope_depends_only_on_relative_position` | with rotary positions, query·key depends only on the distance between the two tokens |
| `test_too_long_input_is_rejected` | a sequence longer than the block size gives a clear error |
| `test_config_round_trip` | the settings saved to a file load back identical; impossible settings are refused |
| `test_overfit_one_batch` | a tiny model memorises one small batch (loss falls below 0.1): the model, loss and gradients are wired correctly |

## Step 4. Check the model on the GPU and the real data (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day8_model_check.py --machine home
```

It takes well under a minute. The output looks like this. The parameter counts and the starting loss must be **exactly** these. Speed and memory are your own GPU's numbers, so I cannot predict them:

```text
=== Day 8 model check | machine=home | NVIDIA GeForce RTX 4060 Laptop GPU | bf16 | PyTorch 2.15.0+cu132 ===
Vocabulary 1,674; training tokens 2,028,601; block size 256

preset       params tokens/param   ms/step    tokens/s  peak MB
tiny      1,001,856         2.02       ...         ...      ...
small     5,150,464         0.39       ...         ...      ...
medium   14,805,120         0.14       ...         ...      ...

Loss before training: 7.422  (ln 1674 = 7.423)
Manual vs fast attention: max difference ...e-07   (anything below 1e-3 passes)
Memorise one batch (4 x 64): loss 7.42… -> 0.0…
...
All checks PASS.
```

On my CPU the memorisation went 7.424 → 0.0031 in 300 steps. Your GPU computes in bf16, so your last digits will differ. The check passes below 0.5, and a value near 0.01 is normal.

It also saves `configs\model_tiny.json`, `model_small.json` and `model_medium.json` (Day 9 and Day 11 read them), writes `docs\DAY8_MODEL_REPORT.md`, and adds an entry to `run_log.md`.

## Step 5. Read the report and understand three findings

1. **Starting loss = ln(1,674) ≈ 7.42.** An untrained model spreads its guess evenly over 1,674 tokens, and the loss of an even guess is ln(1,674). Seeing exactly this number shows the initialisation is right. From Day 9 on, every drop below 7.42 is something the model has learned.
2. **Memorising one batch works.** The loss falls from about 7.42 to near 0 on one fixed batch. If the model, the mask or the loss had a bug, it could not do this. This is the standard first test before any real training.
3. **Tokens per parameter.** tiny has about 2 training tokens per parameter, small about 0.4 and medium about 0.14. With so little text per parameter, the larger models are likely to memorise rather than generalise. Day 11 checks this with validation loss.

## Step 6. Record and commit

Add to `docs\DECISIONS.md`:

```markdown
## D8. Model architecture and trial sizes (Day 8)

- Decoder-only Transformer, own code (src/model.py): pre-norm RMSNorm, rotary positions (RoPE), causal attention
  (manual + fast path, proven equal by a test), GELU MLP 4x, no biases, weight tying, scaled residual init.
- Block size 256 tokens (longest question 28 tokens); a setting, not part of the weights.
- Trial sizes changed from ~25M/60M/100M to ~1.0M/5.2M/14.8M (tiny/small/medium): vocabulary is 1,674 (not 16k)
  and train is 2,028,601 tokens, so the planned sizes would have 12+ parameters per training token.
- Day 11 still chooses the size by validation loss.
```

Then commit (VS Code terminal, `llm_home`). The new `configs` folder is small and belongs in Git:

```powershell
git status --short
git add src scripts tests configs docs run_log.md
git status --short
git commit -m "Day 8 home: Model A (decoder-only Transformer), tests, model check"
git push
```

`data/` must not appear in the list.

## Talking points for the review

- **Attention in plain words:** at each position the model asks a question (query), every earlier token offers a label (key) and a content (value). The model compares the query with each key, turns the scores into weights that add up to 1 (softmax), and takes the weighted mix of the contents. So 「年収」 can draw information from 「営業部」 earlier in the question.
- **Why the causal mask:** the model is trained to predict the *next* token. If position t could see token t+1, it would copy the answer instead of learning. The mask hides every later position, and the causality test proves it.
- **Why weight tying:** the embedding turns a token into a vector, and the output layer turns a vector back into token scores. Using one matrix for both saves 214,272 parameters (21% of the tiny model) and ties the two views of each token together.
- **Why rotary positions:** attention by itself does not know word order. RoPE rotates query and key by an angle that grows with the position, so their match depends on the distance between them. The test shows that shifting both positions by the same amount gives the same score.
- **Why two attention versions:** the manual one is short enough to explain line by line in a review. The fast one is what training uses. A test proves they give the same output (difference about 10⁻⁷).
- **Why the sizes are smaller than the plan:** model size must match the amount of data. We measured 2.0M training tokens, so the trials are about 1M, 5M and 15M, and the choice is made by validation loss on Day 11, not by "bigger is better".
- **Why test before training:** the starting-loss check and the memorise-one-batch check catch wiring mistakes in seconds. Without them, a bug would only show up as "training does not work" hours later.

## End of Day 8 checklist

- [ ] Files copied (`src\model.py`, `scripts\day8_model_check.py`, `tests\test_model.py`, both walkthroughs)
- [ ] GPU check: `True … RTX 4060 … True`
- [ ] pytest: `46 passed`
- [ ] Model check: parameter counts 1,001,856 / 5,150,464 / 14,805,120; starting loss 7.422; attention difference below 1e-3; memorisation below 0.5; `All checks PASS`
- [ ] `configs\model_*.json` and `docs\DAY8_MODEL_REPORT.md` exist
- [ ] D8 added; committed and pushed

## Send back

1. The pytest result line of Step 3.
2. The full console output of Step 4 (as text). I need your real ms/step, tokens/s and peak MB to plan Day 9's batch size.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題10】次に続く単語を予測する仕組み(開始)
状況: 予定通り終わった(今週の予定: 課題10の開始)
内容: 次の単語を予測するモデル(デコーダーのみの Transformer)を自分たちのコードで作成した。
      12項目の自動テスト(未来の単語を見ないこと、1つのバッチを覚えられること など)がすべて合格し、
      自宅ノートPCの GPU で3つの大きさのモデルの動作と速さを確認した。
証拠: src/model.py、tests/test_model.py、docs/DAY8_MODEL_REPORT.md、configs/、run_log.md
次: 学習プログラムを作り、小さな学習を実行する(Day 9)。課題10は Day 10 で完了する。
```
