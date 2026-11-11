# Day 7.2 — test_token_data.py — Simple Line-by-Line Explanation (English)

> **How the four Day 7 files fit together**
> - **Day 7.1 `token_data.py`** — the file-format library (**read it first**).
> - **Day 7 `day7_encode_corpus.py`** and **`day7_tokenizer_demo.py`** — the scripts.
> - **Day 7.2 (this file) `test_token_data.py`** — the *inspector*: automatic tests that check Day 7.1 is correct.
>
> pytest basics (test functions, `assert`, fixtures, `tmp_path`, `skip`) were explained in **Day 6.2**; this file only explains what is new.

---

## 0. The big picture (read this first)

If the token files were written wrongly (wrong byte order, missing questions, mismatched counts), Day 9's training would silently learn from **garbage**. These tests make sure the file format library does exactly what it promises:

| Promise | Test |
|---|---|
| The stream is laid out as `<bos> question <eos>` for every question, and can be cut back into questions | `test_stream_layout` |
| Writing files and reading them back (with or without memory map) gives **identical** data, and the file really is **little-endian** | `test_files_round_trip` |
| A file whose size does not match its meta label is **rejected** | `test_size_mismatch_is_rejected` |
| The **real** validation token file matches the real corpus | `test_real_token_files_match_corpus` |

Run (same as Day 6.2):

```
.venv\Scripts\python.exe -m pytest tests -q
```

---

## 1. New words

| Word | Simple meaning |
|---|---|
| **`pytest.raises(ValueError)`** | A test tool: "this code **must** raise a `ValueError`." If it does not raise, the test **fails**. Used to test that bad input is rejected. |
| **`np.array_equal(a, b)`** | `True` if two NumPy arrays have the same shape and the same values. |
| **`np.asarray(x)`** | Treat `x` as a normal NumPy array (turns a memory map into a plain array for comparing). |
| **`bytes[:2]`** | The first two bytes of a bytes value. |

(Fixtures, `scope="module"`, `tmp_path` and `skip` are the same as Day 6.2.)

---

## 2. The docstring, imports and setup

```python
"""Unit tests for src/token_data.py (Day 7).
...
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- Same style as Day 6.2: a non-raw docstring (so the backslashes are doubled `\\`), `sys`, `Path`, `pytest`, and the `REPO_ROOT` / `sys.path` trick. New: `numpy as np` (as in Day 7.1).

```python
from src.bpe_tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer, count_pieces, train_bpe  # noqa: E402
from src.token_data import (  # noqa: E402
    FORMAT_VERSION, TOKEN_DTYPE, TokenFileMeta, encode_lines, read_token_files, sha256_bytes, split_stream, write_token_files,
)
```
- From the tokenizer (Day 6.1): marker IDs, the tokenizer class and training helpers (to build a tiny tokenizer). From the library (Day 7.1): everything the tests exercise.

---

## 3. Test data and fixture

```python
LINES = ["営業部の社員は何人ですか。", "開発部の平均年収を教えてください。", "今日は晴れ😀", "A1"]
```
- Four test sentences: two normal questions, one with an **emoji**, and a very short one with a letter and a digit.

```python
@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    merges, _ = train_bpe(count_pieces(LINES * 5), 10_000, 2)
    return ByteBPETokenizer(merges)
```
- A fixture (as in Day 6.2) that builds a tiny tokenizer **once**: the four lines repeated 5 times are the training text; `train_bpe(counts, 10_000, 2)` passes `max_merges` and `min_frequency` by position. All tests below receive it by naming `tok` as a parameter.

---

## 4. The tests

### 4.1 `test_stream_layout`

```python
def test_stream_layout(tok: ByteBPETokenizer) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    assert tokens.dtype == TOKEN_DTYPE and len(offsets) == len(LINES)
    parts = split_stream(tokens, offsets)
    for line, ids in zip(LINES, parts):
        assert ids[0] == BOS_ID and ids[-1] == EOS_ID
        assert tok.decode(ids.tolist()) == line
    assert sum(len(p) for p in parts) == len(tokens)
```
- Encode the four lines into a stream plus offsets (Day 7.1).
- The tokens array must have the **exact little-endian uint16 type**, and there must be **one offset per line**.
- Cut the stream back into questions (`split_stream`). For every original line and its piece:
  - it starts with `<bos>` and ends with `<eos>`;
  - it **decodes to the original line**.
- Last check: the lengths of all pieces **add up to the total length** — nothing is missing, nothing is counted twice (no gaps or overlaps).

### 4.2 `test_files_round_trip`

```python
def test_files_round_trip(tok: ByteBPETokenizer, tmp_path: Path) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    meta = TokenFileMeta(FORMAT_VERSION, "train", tok.vocab_size, "x", BOS_ID, EOS_ID, len(LINES), len(tokens),
                         sha256_bytes(tokens.tobytes()), "y")
    write_token_files(tmp_path, "train", tokens, offsets, meta)
```
- Prepare a meta label. `"x"` and `"y"` are dummy fingerprints (the library does not check them), and the real fingerprint of the token bytes is computed with `sha256_bytes(tokens.tobytes())`.
- `tmp_path` (built-in pytest fixture) is a fresh temporary folder, so the test never touches your real `data/tokens`.
- Write the three files there.

```python
    for memmap in (True, False):
        t2, o2, m2 = read_token_files(tmp_path, "train", memmap=memmap)
        assert np.array_equal(np.asarray(t2), tokens) and np.array_equal(o2, offsets)
        assert m2 == meta
```
- Read the files back **both ways** (memory map and normal load). For each:
  - the tokens and offsets must **equal** the originals (`np.asarray` turns the memory map into a plain array before comparing);
  - the meta read from JSON must **equal** the meta we wrote (dataclasses compare field by field).

```python
    # little-endian on disk, whatever the machine
    raw = (tmp_path / "train.bin").read_bytes()
    assert int.from_bytes(raw[:2], "little") == BOS_ID
```
- Check the **byte order on disk**. Read the file as raw bytes and interpret its first 2 bytes as a **little-endian** number: it must equal `BOS_ID` (1). In little-endian the bytes are `01 00`; a big-endian file would give `00 01` = 256 and fail. This guards the promise "files are identical on Windows and Linux".

### 4.3 `test_size_mismatch_is_rejected`

```python
def test_size_mismatch_is_rejected(tok: ByteBPETokenizer, tmp_path: Path) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    meta = TokenFileMeta(FORMAT_VERSION, "val", tok.vocab_size, "x", BOS_ID, EOS_ID, len(LINES), len(tokens) + 1,
                         "z", "y")
    write_token_files(tmp_path, "val", tokens, offsets, meta)
    with pytest.raises(ValueError):
        read_token_files(tmp_path, "val")
```
- Create a **deliberately wrong** label: it claims `len(tokens) + 1` tokens (one too many).
- Write the files, then try to read them. The reader must notice that the real token count differs from the label and raise `ValueError`.
- `with pytest.raises(ValueError):` — the test **passes** only if the code inside raises that exception; if it reads happily, the test fails. (This tests the **safety net** from Day 7.1, section 6.6.)

### 4.4 `test_real_token_files_match_corpus`

```python
def test_real_token_files_match_corpus() -> None:
    """Runs only after Day 7 has written data/tokens/ (checks the real validation file)."""
    tok_path = REPO_ROOT / "data/tokenizer/tokenizer.json"
    out_dir = REPO_ROOT / "data/tokens"
    val_txt = REPO_ROOT / "data/corpus/val.txt"
    if not ((out_dir / "val.bin").exists() and tok_path.exists() and val_txt.exists()):
        pytest.skip("run Day 7 first")
    tok = ByteBPETokenizer.load(tok_path)
    tokens, offsets, meta = read_token_files(out_dir, "val")
    lines = [ln for ln in val_txt.read_text(encoding="utf-8").split("\n") if ln]
    assert meta.vocab_size == tok.vocab_size and meta.questions == len(lines)
    assert all(tok.decode(ids.tolist()) == ln for ln, ids in zip(lines, split_stream(tokens, offsets)))
```
- A test on the **real** files (like Day 6.2's last test). If the real validation token file, the tokenizer or the validation text is missing → `pytest.skip` (reported as **skipped**, not failed).
- Otherwise: load the real tokenizer, read the real validation token file, read the real validation lines, and check:
  - the **meta label** agrees with the tokenizer's vocabulary size and the number of lines;
  - **every question** cut from the token file decodes to the corresponding text line.

---

## 5. How many tests?

| Test | Runs |
|---|---|
| `test_stream_layout` | 1 |
| `test_files_round_trip` | 1 (checks both read modes inside) |
| `test_size_mismatch_is_rejected` | 1 |
| `test_real_token_files_match_corpus` | 1 (may be skipped) |
| **Total** | **4** |

Expected result: `4 passed`, or `3 passed, 1 skipped` before Day 7 has produced the real files. (Pytest also runs Day 6.2's tests if you run the whole `tests` folder.)

## 6. What these tests do *not* cover

- The **uint16 overflow** error in `encode_lines` (an ID above 65,535).
- `read_meta` rejecting a **wrong format name**.
- A meta file with **missing or extra keys**.
- The `tokens_sha256` / `tokenizer_sha256` values (the library stores them but never verifies them; Day 9 should compare them).

These could be added as extra tests later.

## 7. Quick summary in one paragraph

The tests check that the token stream is laid out as `<bos> … <eos>` per question and can be split back and decoded exactly, that writing and reading the three files (memory-mapped or not) returns identical data in true little-endian order, that a file whose size disagrees with its meta label is rejected, and — when the real files exist — that the real validation token file matches the real validation text.
