# Day 6.2 — test_bpe_tokenizer.py — Simple Line-by-Line Explanation (English)

> **How the three Day 6 files fit together**
> - **Day 6.1 `bpe_tokenizer.py`** — the *engine*.
> - **Day 6 `day6_train_bpe.py`** — the *driver*.
> - **Day 6.2 (this file) `test_bpe_tokenizer.py`** — the *inspector*: automatic tests that check the engine (Day 6.1) is correct.
>
> It helps to have read **Day 6.1** first, because every test calls its functions.

---

## 0. The big picture (read this first)

A **test file** is like an **automatic checklist** for a machine. Instead of you checking by hand ("does it still decode correctly?"), you write small functions that check one promise each. Run them any time (after every change) and the computer tells you which promises still hold: **passed** or **failed**.

This file uses **pytest**, a popular Python testing tool. Its rules are simple:
- A function whose name starts with `test_` is a test.
- Inside, an `assert` line says "this must be true". If it is false, the test **fails**.

The promises being checked:

| Promise | Test |
|---|---|
| The ID numbers never change (0–3 special, 4–103 reserved, 104–359 bytes, 360+ merges) | `test_fixed_id_layout` |
| Cutting text into pieces loses nothing | `test_pretokenize_is_lossless` |
| Cutting works as expected on examples | `test_pretokenize_examples` |
| Training gives the same result every time | `test_training_is_deterministic` |
| `decode(encode(text)) == text` for any text | `test_round_trip` |
| Merges really make text shorter and produce whole words | `test_merges_shorten_known_text` |
| `<bos>` / `<eos>` / `<pad>` behave correctly | `test_bos_eos_and_skip_special` |
| A smaller tokenizer is exactly a prefix of the bigger one | `test_truncated_is_prefix` |
| Saving and loading change nothing | `test_save_and_load_identical` |
| The **real** tokenizer round-trips on the **real** validation text | `test_real_tokenizer_round_trip_on_validation` |

Run them like this:

```
.venv\Scripts\python.exe -m pytest tests -q
```
(`-m pytest` runs pytest with the venv's Python; `tests` is the folder; `-q` = quiet, short output.)

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Unit test** | A small automatic check of one small piece of code. |
| **pytest** | The tool that finds and runs test functions and reports the results. |
| **assert** | "This must be true, otherwise fail." |
| **Fixture** | A helper that prepares something once (here: a small trained tokenizer) and gives it to tests that ask for it. |
| **`scope="module"`** | The fixture is created **once per file**, not once per test (faster). |
| **Parametrize** | Run the same test several times with different inputs. |
| **`tmp_path`** | A built-in pytest fixture: a fresh temporary folder that is cleaned up automatically. |
| **Skip** | Tell pytest "do not run this test now" (without failing). |
| **Lossless / round trip / deterministic** | See Day 6.1. |

---

## 2. The docstring

```python
"""Unit tests for src/bpe_tokenizer.py (Day 6).
src/bpe_tokenizer.py の単体テスト(Day 6)

Run (VS Code terminal, repo root):
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
```
- A short description in English and Japanese, and the command to run.
- This docstring is **not** a raw string (no `r` in front), so the backslashes are written **doubled** (`\\`): in a normal string, `\\` means one real backslash. (Earlier files used `r"""` instead.)

---

## 3. Imports and setup

```python
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
```
- `from __future__ ...`, `sys`, `Counter`, `Path` — same as before. New: `pytest`, the testing tool (used for fixtures, parametrize and skip).

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- Same idea as Day 6: find the repo root from this file's location (this file lives in `tests/`, so `.parent.parent` is the repo root) and add it to Python's import path so `from src.bpe_tokenizer import ...` works.

```python
from src.bpe_tokenizer import (  # noqa: E402
    BOS_ID, BYTE_OFFSET, EOS_ID, FIRST_MERGE_ID, NUM_RESERVED, PAD_ID, RESERVED_FIRST_ID, UNK_ID,
    ByteBPETokenizer, count_pieces, pretokenize, train_bpe,
)
```
- Imports everything the tests need from the engine: the ID constants, the tokenizer class and the training helpers. (`noqa: E402` as in Day 6.)

---

## 4. Test data

```python
TINY_CORPUS = [
    "営業部の社員の平均年収を教えてください。",
    "開発部の社員の平均月収を教えてください。",
    "営業部で年収が600万円以上の人は何人ですか。",
    "大阪支社の社員を一覧で見せて。",
] * 5
```
- A tiny pretend training set: 4 realistic sentences. `* 5` repeats the list 5 times (20 lines), so that pairs occur **often enough** (at least 2 times) to be merged.

```python
TRICKY_TEXTS = [
    "",
    "今日は晴れ",
    "ABC abc 123",
    "絵文字😀と記号①②",
    "\t改行\nを含む\r\n",
    "<bos>や<unused_5>は普通の文字",
    "ﾊﾝｶｸｶﾅとＺＥＮＫＡＫＵ",
]
```
- Nasty texts to try to break the tokenizer:
  1. the **empty string**;
  2. a simple Japanese phrase;
  3. English letters and digits;
  4. an **emoji** and circled numbers (rare characters; emoji take 4 bytes);
  5. **tab, newline and carriage return** characters;
  6. text that **looks like** special tokens (`<bos>`, `<unused_5>`) but must be treated as ordinary text;
  7. **half-width katakana** and **full-width Latin letters** (different Unicode forms; no normalisation is applied, so they must survive unchanged).

---

## 5. The fixture

```python
@pytest.fixture(scope="module")
def tiny_tokenizer() -> ByteBPETokenizer:
    merges, _ = train_bpe(count_pieces(TINY_CORPUS), max_merges=10_000, min_frequency=2)
    return ByteBPETokenizer(merges)
```
- `@pytest.fixture` marks this as a helper that pytest calls when a test lists `tiny_tokenizer` as a parameter.
- `scope="module"` → built **once** for the whole file and shared (training takes a moment; no need to repeat).
- `train_bpe(...)` returns `(merges, log)`; `merges, _ = ...` keeps the merges and throws the log away (`_` = "don't need it").
- Return a tokenizer built from those merges. So all tests below use a tokenizer trained on the tiny corpus.

---

## 6. The tests

### 6.1 `test_fixed_id_layout`

```python
def test_fixed_id_layout() -> None:
    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)
    assert RESERVED_FIRST_ID == 4 and NUM_RESERVED == 100
    assert BYTE_OFFSET == 104 and FIRST_MERGE_ID == 360
    tok = ByteBPETokenizer([])
    assert tok.vocab_size == 360
    assert tok.token_bytes[4] == b"<unused_0>" and tok.token_bytes[103] == b"<unused_99>"
    assert tok.token_bytes[BYTE_OFFSET] == b"\x00" and tok.token_bytes[359] == b"\xff"
```
- Checks the **ID map never changes** — the model will depend on these numbers later, so an accidental change must be caught.
- Line 1: the four special IDs are 0, 1, 2, 3.
- Line 2: the reserved block starts at 4 and has 100 entries.
- Line 3: bytes start at 104; merges start at 360.
- `tok = ByteBPETokenizer([])` — a tokenizer with **no merges**. Its `vocab_size` must be exactly 360 (specials + reserved + 256 bytes).
- Its `token_bytes` list must have `b"<unused_0>"` at index 4 and `b"<unused_99>"` at 103 (first and last reserved).
- And the byte block must run from `b"\x00"` (byte 0) at index 104 to `b"\xff"` (byte 255) at index 359. (`b"..."` is a bytes literal; `\x00` is hexadecimal for byte 0.)

### 6.2 `test_pretokenize_is_lossless`

```python
@pytest.mark.parametrize("text", TRICKY_TEXTS + TINY_CORPUS[:4])
def test_pretokenize_is_lossless(text: str) -> None:
    assert "".join(pretokenize(text)) == text
```
- `@pytest.mark.parametrize("text", [...])` runs this test **once per value** in the list. Here: the 7 tricky texts + the first 4 corpus sentences = **11 runs**.
- Joining the pieces with `""` must give back the exact original. This proves cutting text into pieces **loses nothing**.

### 6.3 `test_pretokenize_examples`

```python
def test_pretokenize_examples() -> None:
    assert pretokenize("営業部の社員は何人ですか。") == ["営業部", "の", "社員", "は", "何人", "ですか", "。"]
    assert pretokenize("600万円") == ["6", "0", "0", "万円"]  # digits are always single pieces
    assert pretokenize("ボーナス、") == ["ボーナス", "、"]
```
- Three exact expectations:
  1. The docstring example: kanji, hiragana and punctuation runs are cut where the character type changes.
  2. **Digits are always single pieces**: `600` → `6`, `0`, `0`.
  3. Punctuation `、` does not stick to the katakana word.

### 6.4 `test_training_is_deterministic`

```python
def test_training_is_deterministic() -> None:
    counts = count_pieces(TINY_CORPUS)
    assert train_bpe(counts, 10_000, 2)[0] == train_bpe(Counter(dict(reversed(list(counts.items())))), 10_000, 2)[0]
```
- Training the same data must give the **same merges**, even if the input is in a different order.
- `counts` — the piece counts.
- Left side: train normally; `[0]` takes the first item of the result (the merges).
- Right side: build a counter with the **same items in reverse order** (`list(counts.items())` → `reversed(...)` → `dict(...)` → `Counter(...)`) and train on that.
- They must be equal. This works because `train_bpe` sorts the pieces and breaks ties by smaller IDs (Day 6.1). This is what lets the **school PC and home laptop produce identical tokenizers**.

### 6.5 `test_round_trip`

```python
@pytest.mark.parametrize("text", TRICKY_TEXTS + TINY_CORPUS[:4])
def test_round_trip(tiny_tokenizer: ByteBPETokenizer, text: str) -> None:
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text
```
- Same 11 texts. This time with the trained tokenizer (the fixture is passed in by naming it as a parameter).
- `decode(encode(text))` must equal the original. This is the **lossless promise** — including emoji, tabs and special-looking text.

### 6.6 `test_merges_shorten_known_text`

```python
def test_merges_shorten_known_text(tiny_tokenizer: ByteBPETokenizer) -> None:
    text = TINY_CORPUS[0]
    assert len(tiny_tokenizer.encode(text)) < len(text.encode("utf-8"))
    assert "営業部" in tiny_tokenizer.pieces(text)
```
- Checks that learning actually **helps**.
- The number of tokens must be **fewer than the number of UTF-8 bytes** (without merges it would be exactly one token per byte).
- And the word `営業部` must appear as **one whole token** in `pieces(text)` (it occurs often in the tiny corpus, so training should have merged it fully).

### 6.7 `test_bos_eos_and_skip_special`

```python
def test_bos_eos_and_skip_special(tiny_tokenizer: ByteBPETokenizer) -> None:
    ids = tiny_tokenizer.encode("社員", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert tiny_tokenizer.decode(ids) == "社員"
    assert tiny_tokenizer.decode([PAD_ID, 4, *tiny_tokenizer.encode("社員")]) == "社員"
```
- With `add_bos` and `add_eos`, the first ID must be `<bos>` and the last `<eos>` (`ids[-1]` = last item).
- Decoding that list must give back just `社員` — special IDs are **skipped**.
- Third check: put `<pad>` (ID 0) and the first reserved ID (4) in front of the real tokens. `[a, b, *list]` **unpacks** the list inside a new list. Decode must still give `社員` — padding and reserved IDs are ignored.

### 6.8 `test_truncated_is_prefix`

```python
def test_truncated_is_prefix(tiny_tokenizer: ByteBPETokenizer) -> None:
    small = tiny_tokenizer.truncated(FIRST_MERGE_ID + 5)
    assert small.vocab_size == FIRST_MERGE_ID + 5
    assert small.merges == tiny_tokenizer.merges[:5]
    assert small.decode(small.encode(TINY_CORPUS[0])) == TINY_CORPUS[0]
```
- Ask for a tokenizer with room for exactly 5 merges (vocab size 365).
- Its size must be 365, its merges must be **exactly the first 5** of the big tokenizer's merges (the "prefix" idea Day 6 relies on), and it must still round-trip text correctly.

### 6.9 `test_save_and_load_identical`

```python
def test_save_and_load_identical(tiny_tokenizer: ByteBPETokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tiny_tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.merges == tiny_tokenizer.merges
    for text in TRICKY_TEXTS + TINY_CORPUS[:4]:
        assert loaded.encode(text) == tiny_tokenizer.encode(text)
```
- `tmp_path` is a **built-in** pytest fixture: a fresh temporary folder, deleted afterwards. So the test never touches your real files.
- Save the tokenizer to a temp file, load it back, and check: same merges, and **identical encodings** for all 11 texts.

### 6.10 `test_real_tokenizer_round_trip_on_validation`

```python
def test_real_tokenizer_round_trip_on_validation() -> None:
    """Runs only after Day 6 has produced data/tokenizer/tokenizer.json and Day 5 the validation file."""
    tok_path = REPO_ROOT / "data/tokenizer/tokenizer.json"
    val_path = REPO_ROOT / "data/corpus/val.txt"
    if not (tok_path.exists() and val_path.exists()):
        pytest.skip("run Day 5 and Day 6 first")
    tok = ByteBPETokenizer.load(tok_path)
    lines = [ln for ln in val_path.read_text(encoding="utf-8").split("\n") if ln]
    assert all(tok.decode(tok.encode(ln)) == ln for ln in lines)
```
- Unlike the others, this test uses the **real** files produced by Day 5 and Day 6.
- If either file is missing → `pytest.skip(...)`: the test is reported as **skipped**, not failed (so the test suite also works on a fresh checkout).
- Otherwise: load the real tokenizer, read the real validation lines (non-empty only), and check that **every line round-trips**. `all(...)` is true only if every single line passes.

---

## 7. How many tests are there?

| Test | Runs |
|---|---|
| `test_fixed_id_layout` | 1 |
| `test_pretokenize_is_lossless` | 11 (parametrized) |
| `test_pretokenize_examples` | 1 |
| `test_training_is_deterministic` | 1 |
| `test_round_trip` | 11 (parametrized) |
| `test_merges_shorten_known_text` | 1 |
| `test_bos_eos_and_skip_special` | 1 |
| `test_truncated_is_prefix` | 1 |
| `test_save_and_load_identical` | 1 |
| `test_real_tokenizer_round_trip_on_validation` | 1 (may be skipped) |
| **Total** | **30** |

## 8. How to read the result

- `30 passed` — everything holds.
- `29 passed, 1 skipped` — fine: the real-file test was skipped because Day 5/6 outputs are not there yet.
- `FAILED test_xxx` — a promise is broken. Read the lines pytest prints under `assert` to see which values differed.
- Run the tests **after every change** to the tokenizer (and on both PCs). They are fast.

## 9. What these tests do *not* cover (honest limits)

- `load()` error handling (wrong format, wrong ID layout, wrong `vocab_size`) is not tested.
- The training stop conditions (`min_frequency`, `max_merges`) are not tested directly.
- `decode` with an **out-of-range ID** is not tested (it would raise `IndexError`; see Day 6.1 notes).
- The tiny-corpus tokenizer is only a stand-in; the last test is what checks the **real** tokenizer, and only on the validation lines.

If you want, these could be added later as extra tests.

## 10. Quick summary in one paragraph

This file is an automatic checklist (pytest) for the tokenizer: it checks that the ID layout is fixed, that cutting and encoding never lose any text (even emoji, tabs and look-alike special tokens), that training is deterministic, that merges really shorten text, that special IDs behave, that a smaller tokenizer is exactly a prefix of the bigger one, that saving and loading change nothing, and finally that the real tokenizer round-trips all real validation lines.
