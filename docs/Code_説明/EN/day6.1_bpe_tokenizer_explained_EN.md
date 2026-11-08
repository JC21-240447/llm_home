# Day 6.1 — bpe_tokenizer.py — Simple Line-by-Line Explanation (English)

> **How the three Day 6 files fit together**
> - **Day 6.1 (this file) `bpe_tokenizer.py`** — the *engine*: the tokenizer itself (cut text, turn into numbers, learn merges).
> - **Day 6 `day6_train_bpe.py`** — the *driver*: uses this engine to train, try several sizes, pick one, save it, and write a report.
> - **Day 6.2 `test_bpe_tokenizer.py`** — the *inspector*: automatic tests that check the engine.
>
> Suggested reading order: **6.1 → 6 → 6.2**.

---

## 0. The big picture (read this first)

A neural network cannot read text. It can only read **numbers**. A **tokenizer** is the translator between the two:

```
"営業部の社員"  --encode-->  [ 412, 380, 951 ]  --decode-->  "営業部の社員"
```

Think of **LEGO**:
- You start with **256 tiny bricks**: one for every possible byte (0–255). With them you can build *anything* (any text, even emoji), so there are never "unknown words".
- Then you look at the training text and notice that some two bricks are very often side by side. You **glue** them into one bigger brick. Repeat many times. Each glue step is called a **merge**.
- The **ordered list of merges** is the "recipe book". To encode new text: start with single bricks, then replay the recipes **in the order they were learned**.

This file contains three parts:

| Part | What it does |
|---|---|
| **Pre-tokenizer** (`char_class`, `pretokenize`) | First cuts text into "words" by character type (kanji run, hiragana run, ...) |
| **`ByteBPETokenizer` class** | Encode, decode, save, load, make a smaller version |
| **Training** (`count_pieces`, `train_bpe`) | Learn the merges from the training text |

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Token** | One piece the model reads. Could be a byte, a character, or a whole word like 営業部. |
| **Token ID** | The number of a token. The model only sees IDs. |
| **Vocabulary / vocab size** | The full list of tokens / how many there are. |
| **Byte / UTF-8** | Computers store text as bytes (0–255). In UTF-8, one Japanese character usually takes **3 bytes**. |
| **BPE (Byte Pair Encoding)** | Method: repeatedly merge the most frequent pair of neighbouring tokens. "Byte-level" = start from bytes. |
| **Merge** | One learned rule: "token A followed by token B becomes new token C". |
| **Pair** | Two neighbouring tokens `(A, B)`. |
| **Rank** | The position of a merge in the list. Rank 0 was learned first. Lower rank = applied earlier. |
| **Pre-tokenizer** | A first cutting step before BPE. BPE merges can never cross its cuts. |
| **Special tokens** | Tokens with a job: `<pad>` (padding), `<bos>` (begin), `<eos>` (end), `<unk>` (unknown). |
| **Reserved IDs** | 100 spare IDs kept empty for future needs. |
| **Lossless** | Nothing is lost: `decode(encode(text)) == text`. |
| **Round trip** | Encode, then decode, and compare with the original. |
| **Deterministic** | Same input → always same output (no hidden randomness). |
| **Cache** | A notebook that remembers answers already computed, to avoid repeating work. |
| **`frozenset`** | A set that can never be changed. |

---

## 2. The docstring (top of the file)

```python
r"""Byte-level BPE tokenizer for Model A, written from scratch (no tokenizer library).
...
"""
```
- Same style as before (raw string, English + Japanese, never executed).
- It explains the 3 steps (pre-tokenizer → bytes → merges), the **fixed ID layout** (see section 4), and the promise that the tokenizer is **lossless** because no normalisation is used and every byte can be represented.
- Example from the docstring: `営業部の社員は何人ですか。` → `営業部 | の | 社員 | は | 何人 | ですか | 。`.

---

## 3. Imports

```python
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
```
- `from __future__ ...`, `json`, `Counter`, `defaultdict`, `dataclass`, `Path`, `Iterable` — all **same as Days 4 and 5**. (No `csv`, `zipfile`, etc.: this file only needs JSON and counting.)

---

## 4. Constants and the ID layout

```python
FORMAT_VERSION: str = "model-a-byte-bpe-v1"
PRETOKENIZER_NAME: str = "script_class_v1"
```
- Two name tags written into the saved file. `load` checks them, so you cannot accidentally load a file made by a different format.

```python
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
NUM_RESERVED: int = 100
RESERVED_FIRST_ID: int = len(SPECIAL_TOKENS)                  # 4
BYTE_OFFSET: int = RESERVED_FIRST_ID + NUM_RESERVED           # 104
FIRST_MERGE_ID: int = BYTE_OFFSET + 256                       # 360
```
- `SPECIAL_TOKENS` — the four special names, in order.
- `PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3` — **tuple unpacking**: four names given four numbers in one line.
- `NUM_RESERVED = 100` — number of spare IDs.
- The next three lines calculate where each block starts:

| ID range | What lives there |
|---|---|
| 0 – 3 | `<pad>`, `<bos>`, `<eos>`, `<unk>` |
| 4 – 103 | `<unused_0>` … `<unused_99>` (100 reserved spare IDs) |
| 104 – 359 | the 256 single bytes (`BYTE_OFFSET` = 104 → byte `b` has ID `104 + b`) |
| 360 … | learned merges, in training order (`FIRST_MERGE_ID` = 360) |

```python
Pair = tuple[int, int]
```
- A **type alias**: `Pair` is just a nicer name for "a tuple of two integers".

---

## 5. The pre-tokenizer

### 5.1 `char_class`

```python
def char_class(ch: str) -> str:
    """Script class of one character. Digits and 'other' never form runs."""
    o = ord(ch)
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    ...
    return "other"
```
- **Same logic as Day 5's `script_class`** (hiragana / katakana / kanji / latin / digit / space / other, using Unicode number ranges). The only visual difference is that the Latin test has extra parentheses `(ch.isascii() and ch.isalpha())`; the meaning is the same.
- Day 5's version was written to "match the planned Day 6 pre-tokenizer"; this is the real one.

### 5.2 `_RUN_CLASSES`

```python
_RUN_CLASSES: frozenset[str] = frozenset({"hiragana", "katakana", "kanji", "latin", "space"})
```
- The character types that **are allowed to form runs** (several in a row stay together). `digit` and `other` are **not** in this set, so each digit and each punctuation mark is always a piece by itself.
- Why? Numbers can be written digit by digit (so any number is possible), and `。` never sticks to a word.

### 5.3 `pretokenize`

```python
def pretokenize(text: str) -> list[str]:
    """Cut text into script-class runs. Concatenating the result gives the text back exactly."""
    pieces: list[str] = []
    current = ""
    previous = ""
    for ch in text:
        cls = char_class(ch)
        if current and cls == previous and cls in _RUN_CLASSES:
            current += ch
        else:
            if current:
                pieces.append(current)
            current = ch
        previous = cls
    if current:
        pieces.append(current)
    return pieces
```
- `pieces` — the finished pieces. `current` — the piece being built. `previous` — the class of the last character.
- For each character: find its class (`cls`).
  - **If** we are building a piece **and** the class is the same as before **and** that class is allowed to form runs → add the character to `current`.
  - **Else** → the piece ends: save `current` (if not empty) and start a new piece with this character.
- `previous = cls` — remember for the next character.
- After the loop, save the last piece.
- **Lossless:** nothing is removed or changed, so joining the pieces gives the original text.
- Walk-through: `600万円` → `6` (digit, own piece) | `0` | `0` | `万円` (kanji run).

---

## 6. The tokenizer class

### 6.1 Creating it: `__post_init__`

```python
@dataclass
class ByteBPETokenizer:
    """A trained tokenizer: an ordered list of merges plus the fixed ID layout."""

    merges: list[Pair]
```
- A dataclass (see Day 4) with a single field: the **ordered list of merges**. Everything else is derived from it.

```python
    def __post_init__(self) -> None:
        self.ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [s.encode("utf-8") for s in SPECIAL_TOKENS]
        self.token_bytes += [f"<unused_{i}>".encode("utf-8") for i in range(NUM_RESERVED)]
        self.token_bytes += [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self._cache: dict[str, list[int]] = {}
```
- `__post_init__` runs **automatically right after** the dataclass fills `merges`. Here it builds helper data:
- `self.ranks` — a lookup table: pair → rank (its position in the merge list). `enumerate` gives position `i` and the pair. Used to quickly ask "is this pair mergeable, and how early was it learned?".
- `self.token_bytes` — the list "ID → the bytes this token stands for". Built **in ID order** so that list index = token ID:
  1. The 4 special names as bytes (IDs 0–3).
  2. 100 reserved names like `b"<unused_0>"` (IDs 4–103).
  3. 256 single bytes: `bytes([b])` makes a 1-byte value for b = 0…255 (IDs 104–359).
  4. For each merge `(a, b)`: the new token's bytes are the bytes of token `a` **plus** the bytes of token `b`. (`self.token_bytes[a] + self.token_bytes[b]`). So merge number k automatically gets ID `360 + k`.
- `self._cache` — the notebook for `_encode_piece` (piece text → its token IDs). The leading `_` means "private, for internal use".

### 6.2 Sizes

```python
    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)
```
- `@property` lets you write `tok.vocab_size` **without parentheses**, like a variable. Value = 360 + number of merges.

```python
    def truncated(self, vocab_size: int) -> "ByteBPETokenizer":
        n = max(0, min(len(self.merges), vocab_size - FIRST_MERGE_ID))
        return ByteBPETokenizer(self.merges[:n])
```
- Returns a **smaller version** that keeps only the first `n` merges.
- `n = max(0, min(len(self.merges), vocab_size - 360))` — how many merges are needed to reach the requested size, but never negative and never more than we have.
- `self.merges[:n]` — slice: the first `n` merges.
- The docstring's key idea: merges are learned in order, so the first N merges **are exactly** the tokenizer that training would have produced if it had stopped at that size. So **one training run gives every smaller size for free** (Day 6 uses this).
- The type is written in quotes `"ByteBPETokenizer"` because the class is not finished being defined at that point.

### 6.3 Encoding one piece: `_encode_piece`

```python
    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
```
- First look in the notebook. If this piece was already encoded, return the saved answer.

```python
        ids = [BYTE_OFFSET + b for b in piece.encode("utf-8")]
```
- Turn the piece into bytes, then each byte into its token ID (`104 + byte`). Example: `万` (3 bytes) → 3 IDs. This is the "single bricks" starting state.

```python
        while len(ids) > 1:
            best_rank = None
            best_at = -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_at = r, i
            if best_rank is None:
                break
            ids[best_at:best_at + 2] = [FIRST_MERGE_ID + best_rank]
```
- Repeat while there are at least 2 tokens:
  - `best_rank` / `best_at` — the earliest-learned mergeable pair found so far, and where it is.
  - Look at every neighbouring pair `(ids[i], ids[i+1])`. `ranks.get(...)` returns its rank, or `None` if it is not a learned merge.
  - Keep the pair with the **lowest rank** (`r < best_rank`) — "apply the oldest recipe first". On a tie the first (left-most) one is kept, because the comparison is strict `<`.
  - If no pair is mergeable (`best_rank is None`) → finished, `break`.
  - Otherwise replace the two tokens at that position by **one** token with ID `360 + rank`. `ids[best_at:best_at+2] = [x]` is **slice assignment**: two items replaced by one.
- This replays training in the same order, which is why the result equals what training "expected".

```python
        self._cache[piece] = ids
        return ids
```
- Save the answer in the notebook and return it.

### 6.4 `encode`

```python
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text to token IDs. Special-token strings in the text are treated as ordinary text."""
        ids: list[int] = [BOS_ID] if add_bos else []
        for piece in pretokenize(text):
            ids.extend(self._encode_piece(piece))
        if add_eos:
            ids.append(EOS_ID)
        return ids
```
- Start the list with `<bos>` if requested, else empty.
- Cut the text with `pretokenize`, encode each piece, and glue the IDs together (`extend`).
- Add `<eos>` at the end if requested.
- Important detail from the docstring: if the text itself contains the characters `<bos>`, they are encoded as ordinary characters, **not** as the special token. (So users cannot inject special tokens by typing them.)

### 6.5 `decode`

```python
    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Token IDs to text. Special and reserved IDs are skipped by default."""
        out = bytearray()
        for i in ids:
            if i < BYTE_OFFSET:
                if skip_special:
                    continue
            out += self.token_bytes[i]
        return out.decode("utf-8", errors="replace")
```
- `out = bytearray()` — a growable bytes container.
- For each ID: if it is below 104 (special or reserved) **and** `skip_special` is true → skip it (`continue`).
- Otherwise append that token's bytes (`token_bytes[i]`). If `skip_special` is false, even special tokens add their name bytes.
- At the end convert all bytes back to text with UTF-8. `errors="replace"` means: if the bytes are not valid UTF-8 (for example a half character), insert `�` instead of crashing.

### 6.6 `pieces` — for human eyes

```python
    def pieces(self, text: str) -> list[str]:
        result: list[str] = []
        for i in self.encode(text):
            raw = self.token_bytes[i]
            try:
                result.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                result.append("<" + raw.hex() + ">")
        return result
```
- Shows the tokens as readable text, for demos.
- If a token is a complete UTF-8 string, show it. If it is only part of a character (invalid alone), show its bytes in hex like `<e5ae>`.

### 6.7 Saving and loading

```python
    def to_dict(self) -> dict[str, object]:
        return {
            "format": FORMAT_VERSION,
            "pre_tokenizer": PRETOKENIZER_NAME,
            "special_tokens": {name: i for i, name in enumerate(SPECIAL_TOKENS)},
            "reserved": {"prefix": "<unused_", "first_id": RESERVED_FIRST_ID, "count": NUM_RESERVED},
            "byte_offset": BYTE_OFFSET,
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for a, b in self.merges],
            "vocab_preview": {
                str(i): self.token_bytes[i].decode("utf-8", errors="replace")
                for i in range(FIRST_MERGE_ID, self.vocab_size)
            },
        }
```
- Builds the dictionary that will become the JSON file:
  - the two name tags, the special token IDs, the reserved info, `byte_offset`, `vocab_size`;
  - `merges` — the real data (pairs as 2-element lists, because JSON has no tuples);
  - `vocab_preview` — a **human-readable** table "ID → text" for the learned tokens (JSON keys must be strings, hence `str(i)`). It is only for people to read; `load` ignores it.

```python
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": the same bytes on Windows and Linux, so the SHA-256 is the same on every PC
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
```
- Make sure the folder exists, then write the JSON.
- `ensure_ascii=False` keeps Japanese readable (as in Days 1, 4, 5). `indent=1` = pretty with 1-space indent.
- `newline="\n"` — always LF line endings (same idea as Day 5's `write_lines`), so the file's SHA-256 is **identical on the school PC and home laptop**.

```python
    @classmethod
    def load(cls, path: Path) -> "ByteBPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_VERSION or data.get("pre_tokenizer") != PRETOKENIZER_NAME:
            raise ValueError(f"{path}: not a {FORMAT_VERSION} / {PRETOKENIZER_NAME} tokenizer file")
        if data.get("byte_offset") != BYTE_OFFSET:
            raise ValueError(f"{path}: unexpected ID layout")
        tok = cls([(int(a), int(b)) for a, b in data["merges"]])
        if tok.vocab_size != data.get("vocab_size"):
            raise ValueError(f"{path}: vocab_size does not match the merges")
        return tok
```
- `@classmethod` — a function that belongs to the class itself (you call `ByteBPETokenizer.load(path)`), and receives the class as `cls`.
- Read the JSON, then **three safety checks**: right format and pre-tokenizer names; right ID layout; the saved `vocab_size` matches the number of merges. Any failure → `ValueError` with a clear message.
- Rebuild the tokenizer from the merges, turning each `[a, b]` list back into a `(a, b)` tuple of ints.

---

## 7. Training

### 7.1 `TrainingLog`

```python
@dataclass
class TrainingLog:
    unique_pieces: int
    total_pieces: int
    possible_merges_upper_bound: int
    merges_learned: int
    stop_reason: str
    last_merge_count: int
```
- A report card of the training run: how many different pieces, how many pieces in total, the theoretical maximum number of merges, how many were really learned, why training stopped, and how often the last merged pair occurred.

### 7.2 `count_pieces`

```python
def count_pieces(lines: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        counts.update(pretokenize(line))
    return counts
```
- For each training line, cut it into pieces and add them to the counter. `Counter.update(list)` adds 1 for every item. Result: "how often each piece occurs", e.g. `{"営業部": 530, "の": 4120, ...}`.
- Training works on these **unique pieces with weights**, not on the whole text → much faster.

### 7.3 `train_bpe` — the heart

```python
def train_bpe(piece_counts: Counter[str], max_merges: int, min_frequency: int = 2) -> tuple[list[Pair], TrainingLog]:
```
- Input: the piece counts, a safety cap on merges (`max_merges`), and the minimum number of times a pair must occur to be merged (`min_frequency`). Output: the merge list and the log.

```python
    pieces = sorted(piece_counts)  # sorted: deterministic order
    words: list[list[int]] = [[BYTE_OFFSET + b for b in p.encode("utf-8")] for p in pieces]
    weights: list[int] = [piece_counts[p] for p in pieces]
    upper_bound = sum(max(0, len(w) - 1) for w in words)
```
- `pieces` — unique pieces in sorted order (so training is deterministic).
- `words` — each piece as a list of byte-IDs (the "single bricks" state).
- `weights` — how often each piece occurs.
- `upper_bound` — the maximum number of merges possible: each merge shortens the total length by at least 1, so it can never exceed `sum(length − 1)`.

```python
    pair_counts: Counter[Pair] = Counter()
    where: dict[Pair, set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += weights[wi]
            where[pair].add(wi)
```
- Count every neighbouring pair, **weighted** by how often its piece occurs. `zip(w, w[1:])` pairs each item with the next one.
- `where[pair]` — the set of piece numbers that contain this pair. This lets us later update **only** the affected pieces instead of recounting everything.

```python
    merges: list[Pair] = []
    stop_reason = f"reached max_merges={max_merges}"
    last_count = 0
    while len(merges) < max_merges:
```
- The merge list starts empty. `stop_reason` has a default (changed if we stop earlier). The loop runs until the safety cap.

```python
        if not pair_counts:
            stop_reason = "no pairs left (every piece is a single token)"
            break
        best, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if count < min_frequency:
            stop_reason = f"most frequent pair occurs fewer than {min_frequency} times"
            break
```
- No pairs left → stop.
- Pick the **best pair**: `max(...)` with a `key`. For each item `kv = (pair, count)` the key is `(count, -first_id, -second_id)`:
  - highest **count** wins;
  - on a tie, the pair with the **smaller token IDs** wins (negating makes smaller IDs rank higher). So ties are broken the same way every time → **deterministic**.
- If even the best pair occurs fewer than `min_frequency` times → stop. (Rare pairs would only memorise the training data.)

```python
        new_id = FIRST_MERGE_ID + len(merges)
        merges.append(best)
        last_count = count
```
- The new token gets the next free ID (360, 361, ...). Record the merge and the count.

```python
        for wi in sorted(where.pop(best, set())):
            w = words[wi]
            for pair in zip(w, w[1:]):          # remove this word's old pairs
                pair_counts[pair] -= weights[wi]
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair != best:
                    where[pair].discard(wi)
```
- For each piece that contains the best pair (`where.pop` takes the set and removes the entry; `sorted` for determinism):
  - **Remove its old pair counts**: subtract the weight for each pair in the piece; delete entries that reach 0.
  - Also remove this piece from the `where` sets of its other pairs (`discard` = remove without error if absent).

```python
            merged: list[int] = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(w[i])
                    i += 1
            words[wi] = merged
```
- Build the merged version of the piece, scanning left to right: where the best pair appears, put the new ID and skip 2; otherwise copy the single token. (Matches `encode`'s left-most-first behaviour.)

```python
            for pair in zip(merged, merged[1:]):  # add its new pairs
                pair_counts[pair] += weights[wi]
                where[pair].add(wi)
```
- **Add the new pair counts** of the merged piece (including pairs that involve the new token).
- This "remove old, add new" update is **incremental**: only pieces containing the merged pair are recounted, not the whole corpus.

```python
    log = TrainingLog(len(pieces), sum(weights), upper_bound, len(merges), stop_reason, last_count)
    return merges, log
```
- Create the report card and return the merges together with it.

### 7.4 A tiny worked example (letters instead of bytes)

Pieces: `hug` ×10, `pug` ×5, `pun` ×12. (In the real code letters are byte IDs.)

| Round | Pair counts (highest) | Merge | Result |
|---|---|---|---|
| 1 | (p,u)=17, (u,g)=15, (u,n)=12, (h,u)=10 | **p+u → "pu"** | hug, [pu]g, [pu]n |
| 2 | (pu,n)=12, (h,u)=10, (u,g)=10, (pu,g)=5 | **pu+n → "pun"** | hug, [pu]g, [pun] |
| 3 | (h,u)=10 and (u,g)=10 tie → smaller first ID wins ("h" < "u") | **h+u → "hu"** | [hu]g, … |
| 4 | (hu,g)=10 | **hu+g → "hug"** | [hug], … |

The merge list is the ordered recipe `[pu, pun, hu, hug, …]`. A new word is encoded by replaying these in order.

---

## 8. Small things worth knowing

- **Day 5 vs Day 6 digits.** Day 5's `pretoken_chunks` counts a run of digits like `600` as **one** chunk, but this real pre-tokenizer splits it into **three** pieces. So Day 5's "token lower bound" is still a valid lower bound, but a bit looser than the real minimum for text with numbers.
- **`decode` does not check IDs.** An ID that is not in the vocabulary (for example 99999) raises a Python `IndexError`. Normal use never produces this.
- **One cache per tokenizer object.** `truncated()` makes a new object with an empty cache. That is correct (a different tokenizer must not reuse answers) but means each size is re-warmed.
- **Training speed.** The incremental update is fast, but `max(pair_counts.items(), ...)` scans every pair for every merge. Fine for a corpus of this size.
- **Why `min_frequency = 2` matters.** Pairs seen only once would give the tokenizer memorised, useless tokens.

## 9. Quick summary in one paragraph

`bpe_tokenizer.py` cuts text into same-script runs, turns each run into UTF-8 bytes (so any text is representable), and then replays an ordered list of learned merges to build bigger tokens; it can decode back exactly, save and load itself as JSON, produce smaller versions by keeping only the first merges, and learn those merges from piece counts with a fully deterministic, incremental algorithm.
