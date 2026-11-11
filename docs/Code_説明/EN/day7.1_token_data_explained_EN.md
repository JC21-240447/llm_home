# Day 7.1 — token_data.py — Simple Line-by-Line Explanation (English)

> **How the four Day 7 files fit together**
> - **Day 7.1 (this file) `token_data.py`** — the *file format library*: how token IDs are written to and read from disk.
> - **Day 7 `day7_encode_corpus.py`** — the *main script*: tests the tokenizer on every question and uses this library to write the token files.
> - **Day 7 `day7_tokenizer_demo.py`** — a small *demo* script (text → pieces → numbers → text).
> - **Day 7.2 `test_token_data.py`** — the *inspector*: automatic tests for this library.
>
> Suggested reading order: **7.1 → 7 (encode_corpus) → 7 (demo) → 7.2**.

---

## 0. The big picture (read this first)

Yesterday's tokenizer (Day 6) turns text into **numbers**. Tomorrow's trainer (Day 9) will read **millions** of those numbers. So we need a **fast, safe way to store them on disk**. This file defines that storage format.

Think of a **train timetable printed as one long roll of paper**:
- Every question is written as `<bos> token token token ... <eos>` and the questions are written **one after another** into **one long stream** (the roll).
- A second small file remembers **where each question starts** (the "offsets") — like bookmarks.
- A third small file (JSON) is the **label on the roll**: how many questions, how many tokens, which tokenizer made it, the fingerprint — so nobody uses the wrong roll by mistake.

A tiny example (made-up IDs; `B` = `<bos>`, `E` = `<eos>`):

```
question 1 = "A"  -> tokens  [B, 140, E]
question 2 = "BC" -> tokens  [B, 150, 160, E]

train.bin          : B 140 E B 150 160 E          (one stream, 7 numbers)
train_offsets.bin  : 0, 3                          (question 1 starts at 0, question 2 at 3)
train_meta.json    : { questions: 2, tokens: 7, ... }
```

| File (per split) | What it holds | Type |
|---|---|---|
| `train.bin` | The stream of token IDs | `uint16` (little-endian) |
| `train_offsets.bin` | Start position of each question | `int64` (little-endian) |
| `train_meta.json` | Counts, vocabulary size, fingerprints | JSON text |

Why **one stream**? The trainer will cut training "windows" from the stream using a **memory map**, without loading the whole file into RAM.

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **NumPy** | A Python library for fast arrays of numbers. (`np` is its usual short name.) |
| **Array / `ndarray`** | A fixed-type list of numbers, stored compactly and processed fast. |
| **dtype** | The number type of an array. |
| **uint16** | Unsigned 16-bit integer: whole numbers 0 to 65,535, using **2 bytes** each. |
| **int64** | Signed 64-bit integer, using 8 bytes each. |
| **Little-endian** | The byte order used to store a multi-byte number: smallest byte first. Example: 258 (= `0x0102`) is stored as bytes `02 01`. Fixing the order makes files identical on every computer. |
| **Offset** | The position where something starts in a stream. |
| **Stream** | A long sequence of items one after another. |
| **Memory map (`memmap`)** | An array that is backed directly by a file: the operating system loads only the parts you touch. |
| **Window** | A short piece cut from the stream (what the trainer will feed the model). |
| **Meta file** | A small file describing a bigger file. |
| **Slice / view** | `a[s:e]` is the part of an array from `s` to `e`. In NumPy it is a *view* (no copy). |

---

## 2. The docstring

- Same style as before. It describes the **format** (uint16 little-endian stream, `<bos> ... <eos>` per question, offsets file, meta JSON) and **why one stream**.
- The "SERVER NOTE" says: plain Python + NumPy, identical on Windows and Linux, because the files are written **little-endian explicitly**.

---

## 3. Imports

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
```
- `from __future__ ...`, `hashlib`, `json`, `dataclass`, `Path`, `Iterable` — same as earlier days.
- **New:** `asdict` (turns a dataclass object into a plain dictionary) and `numpy as np` (the first **third-party** library in the tokenizer code; it must be installed in the venv).

---

## 4. Constants

```python
TOKEN_DTYPE = np.dtype("<u2")    # uint16, little-endian
OFFSET_DTYPE = np.dtype("<i8")   # int64, little-endian
FORMAT_VERSION = "model-a-tokens-v1"
```
- `np.dtype("<u2")` — the type code reads: `<` = little-endian, `u` = unsigned integer, `2` = 2 bytes. So uint16, always little-endian.
- `np.dtype("<i8")` — little-endian, signed integer, 8 bytes = int64.
- `FORMAT_VERSION` — a name tag stored in the meta file (like Day 6.1's `FORMAT_VERSION`), so `read_meta` can reject files of another format.

---

## 5. The `TokenFileMeta` dataclass

```python
@dataclass
class TokenFileMeta:
    """Everything needed to trust and use one token file."""

    format: str
    split: str
    vocab_size: int
    tokenizer_sha256: str
    bos_id: int
    eos_id: int
    questions: int
    tokens: int
    tokens_sha256: str
    source_sha256: str
```
- The "label on the roll":
  - `format` — the format name tag.
  - `split` — `train`, `val` or `test`.
  - `vocab_size` — vocabulary size of the tokenizer that made it.
  - `tokenizer_sha256` — fingerprint of that tokenizer file.
  - `bos_id`, `eos_id` — the IDs used for begin/end markers.
  - `questions`, `tokens` — counts.
  - `tokens_sha256` — fingerprint of the token stream bytes.
  - `source_sha256` — fingerprint of the text file it was made from (e.g. `train.txt`).
- Together these let a later program check "is this file really made from this text by this tokenizer?".

---

## 6. Functions

### 6.1 `sha256_bytes`

```python
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```
- Same as Day 4: SHA-256 fingerprint of some bytes.

### 6.2 `encode_lines`

```python
def encode_lines(lines: Iterable[str], encode, bos_id: int, eos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode lines into one stream (<bos> ... <eos> per line) and the start offset of each line."""
    stream: list[int] = []
    offsets: list[int] = []
    for line in lines:
        offsets.append(len(stream))
        stream.append(bos_id)
        stream.extend(encode(line))
        stream.append(eos_id)
```
- Inputs: the lines, the **`encode` function itself** (you pass `tok.encode`; functions can be passed around like values), and the two marker IDs.
- `stream` — the growing list of all token IDs. `offsets` — the start position of each question.
- For each line:
  1. `offsets.append(len(stream))` — **bookmark**: the next question starts at the current length.
  2. add `<bos>`;
  3. add the line's token IDs (`extend` adds a whole list);
  4. add `<eos>`.

```python
    if stream and max(stream) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("a token ID does not fit in uint16")
    return np.asarray(stream, dtype=TOKEN_DTYPE), np.asarray(offsets, dtype=OFFSET_DTYPE)
```
- Safety check: `np.iinfo(TOKEN_DTYPE).max` is the largest number a uint16 can hold (65,535). If any ID is larger, stop (it would be silently corrupted otherwise). `stream and ...` avoids calling `max` on an empty list.
- `np.asarray(list, dtype=...)` converts the Python lists to compact NumPy arrays of the right types and returns both.

### 6.3 `split_stream`

```python
def split_stream(tokens: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """The token IDs of each line, including its <bos> and <eos>."""
    ends = list(offsets[1:]) + [len(tokens)]
    return [tokens[s:e] for s, e in zip(offsets, ends)]
```
- The reverse of the bookmarks: cut the stream back into questions.
- Each question ends where the **next one starts**, and the last one ends at the end of the stream: `ends = offsets[1:] + [len(tokens)]`.
- `zip(offsets, ends)` pairs each start with its end; `tokens[s:e]` is the slice for that question (a *view*, no copying).

### 6.4 `write_token_files`

```python
def write_token_files(out_dir: Path, split: str, tokens: np.ndarray, offsets: np.ndarray,
                      meta: TokenFileMeta) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tokens": out_dir / f"{split}.bin",
        "offsets": out_dir / f"{split}_offsets.bin",
        "meta": out_dir / f"{split}_meta.json",
    }
    tokens.astype(TOKEN_DTYPE).tofile(paths["tokens"])
    offsets.astype(OFFSET_DTYPE).tofile(paths["offsets"])
    paths["meta"].write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return paths
```
- Make sure the folder exists; decide the three file paths (`train.bin`, `train_offsets.bin`, `train_meta.json`).
- `tokens.astype(TOKEN_DTYPE)` makes sure the array has exactly the right type (little-endian uint16). `.tofile(path)` writes the **raw bytes** of the array — no header, nothing else. Because the dtype fixes the byte order, the file is the same on every computer.
- Same for the offsets (int64).
- The meta: `asdict(meta)` → dictionary → `json.dumps` (readable, Japanese kept, 2-space indent) → saved with LF line endings (same idea as Day 5/6.1 for identical SHA-256).
- Return the paths dictionary.

### 6.5 `read_meta`

```python
def read_meta(out_dir: Path, split: str) -> TokenFileMeta:
    data = json.loads((out_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
    if data.get("format") != FORMAT_VERSION:
        raise ValueError(f"{split}_meta.json: unknown format {data.get('format')}")
    return TokenFileMeta(**data)
```
- Read the JSON into a dictionary; reject it if the format tag is not ours.
- `TokenFileMeta(**data)` — the `**` **unpacks the dictionary into keyword arguments**: `{"split": "train", ...}` becomes `split="train", ...`. (If the file has missing or extra keys, Python raises a `TypeError`.)

### 6.6 `read_token_files`

```python
def read_token_files(out_dir: Path, split: str, memmap: bool = True) -> tuple[np.ndarray, np.ndarray, TokenFileMeta]:
    """Open a split. With memmap=True the tokens stay on disk until used (what the trainer needs)."""
    meta = read_meta(out_dir, split)
    path = out_dir / f"{split}.bin"
    tokens = np.memmap(path, dtype=TOKEN_DTYPE, mode="r") if memmap else np.fromfile(path, dtype=TOKEN_DTYPE)
    offsets = np.fromfile(out_dir / f"{split}_offsets.bin", dtype=OFFSET_DTYPE)
    if len(tokens) != meta.tokens or len(offsets) != meta.questions:
        raise ValueError(f"{split}: file sizes do not match {split}_meta.json")
    return tokens, offsets, meta
```
- Read the meta first.
- `memmap=True` (default): `np.memmap(path, dtype=..., mode="r")` — a **read-only memory map**: the array behaves like normal but the data stays on disk; only the parts you touch are loaded. This is what the trainer needs for huge files.
- `memmap=False`: `np.fromfile(...)` — read the whole file into memory.
- The offsets are small, so they are always loaded fully.
- **Consistency check:** the number of tokens in the file must equal `meta.tokens`, and the number of offsets must equal `meta.questions`. Otherwise → `ValueError` (the file is truncated, or the meta belongs to another file).
- Return the tokens, offsets and meta.

---

## 7. Small things worth knowing

- **uint16 limit.** IDs above 65,535 cannot be stored. Our vocabulary is small (hundreds to a few thousand), so this is safe, and `encode_lines` raises an error if it ever happens.
- **The reader checks sizes, not fingerprints.** `read_token_files` verifies the counts, but does not recompute `tokens_sha256` or compare `tokenizer_sha256` / `vocab_size`. Checking that the file matches the current tokenizer is the **trainer's job** (Day 9) — the meta file contains the information for it.
- **Memory use while writing.** `encode_lines` builds a normal Python list first; fine for this corpus size, heavier for a gigantic one.
- **Views, not copies.** Slices from `split_stream` point into the original array. Don't modify them if you need the original unchanged.

## 8. Quick summary in one paragraph

`token_data.py` defines a simple, fixed file format for encoded text: all questions as `<bos> … <eos>` in one little-endian `uint16` stream, a separate `int64` file with each question's start position, and a JSON "label" with counts and fingerprints; it can write these files, read them back (optionally as a memory map), cut the stream back into questions, and reject files whose sizes do not match their label.
