# Day 7 — day7_tokenizer_demo.py — Simple Line-by-Line Explanation (English)

> **How the four Day 7 files fit together**
> - **Day 7.1 `token_data.py`** — file-format library.
> - **Day 7 `day7_encode_corpus.py`** — the big main script.
> - **Day 7 (this file) `day7_tokenizer_demo.py`** — a small **demo**: show, step by step, how a sentence becomes pieces, numbers, and comes back. It only needs the tokenizer from Day 6 (not the token files).
> - **Day 7.2 `test_token_data.py`** — automatic tests for the library.

---

## 0. The big picture (read this first)

This small script exists to produce **evidence** for tasks 8 and 9 of your school plan:

| Task | What the demo shows |
|---|---|
| **Task 8: splitting** | The **pieces** (tokens) a sentence is cut into |
| **Task 9: numbering** | The **number (ID)** of each piece |
| **Task 9: restoring** | The text **decoded back** from the numbers, and whether it is **identical** to the input |

It prints the result on screen **and** saves it as `docs/DAY7_TOKENIZER_DEMO.md` ("only real output is shown" — nothing is typed by hand).

Think of a **translation receipt**: sentence → pieces → numbers → sentence again, with a tick if nothing changed.

Run it:

```
.venv\Scripts\python.exe scripts\day7_tokenizer_demo.py
.venv\Scripts\python.exe scripts\day7_tokenizer_demo.py --text "開発部の平均年収は？" --text "今日は晴れ"
```
- The first command uses 5 built-in example sentences. The second shows **your own** sentences (`--text` can be repeated).
- Unlike the other scripts, it has **no `--machine` option** and does **not** write to `run_log.md`; it is a demo, not a measured run.

---

## 1. Words you need to know

All words are the same as Day 6.1 (token, piece, ID, encode, decode, round trip). One new Python idea: **`action="append"`** — an `argparse` setting that lets the same option be written several times and collects every value into a list.

---

## 2. The docstring

- Same style as before. It explains the three things shown (pieces / numbers / decoded-back), the output file, the two usage commands, and the server note (`.venv/bin/python ...` on Linux).

---

## 3. Imports and setup

```python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
```
- Everything is the same as Days 6–7: `argparse`, `sys`, `datetime`, `Path`, `REPO_ROOT`, the `sys.path` trick, and the tokenizer class from Day 6.1. Only these few imports are needed — it is a small script.

```python
DEFAULT_SENTENCES: tuple[str, ...] = (
    "営業部の社員の平均年収を教えてください。",
    "ボーナスが100万円以上の人は何人ですか。",
    "大阪支社で勤続年数が一番長い人は誰？",
    "開発部の課長の月収はいくら？",
    "今日は晴れ",
)
```
- Five example sentences used when you do not give any `--text`.

---

## 4. `demo_rows`

```python
def demo_rows(tok: ByteBPETokenizer, text: str) -> tuple[list[tuple[str, int]], str, bool]:
    ids = tok.encode(text)
    pieces = tok.pieces(text)
    decoded = tok.decode(ids)
    return list(zip(pieces, ids)), decoded, decoded == text
```
- Does the three steps for **one sentence**:
  1. `ids = tok.encode(text)` — the numbers.
  2. `pieces = tok.pieces(text)` — the readable pieces (Day 6.1; incomplete characters appear as `<xx..>` hex).
  3. `decoded = tok.decode(ids)` — the text rebuilt from the numbers.
- Returns three things:
  - `list(zip(pieces, ids))` — pairs `(piece, ID)`, like `[("営業部", 412), ("の", 380), ...]`. `zip` joins the two lists position by position; `pieces` and `ids` have the same length because `pieces()` itself goes through `encode`.
  - `decoded` — the restored text.
  - `decoded == text` — `True` if it is identical to the input.
- The return type `tuple[list[tuple[str, int]], str, bool]` just says "a list of (text, number) pairs, a string and a boolean".

---

## 5. `main`

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Day 7 demo: text -> pieces -> numbers -> text.")
    ap.add_argument("--text", action="append", help="sentence to show (repeatable); default: built-in examples")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs/DAY7_TOKENIZER_DEMO.md")
    a = ap.parse_args()
    tok = ByteBPETokenizer.load(a.tokenizer.resolve())
    sentences = a.text or list(DEFAULT_SENTENCES)
```
- Same console safety line as before. The argument parser is written inside `main` here (a short script).
- `--text` with `action="append"`: each `--text "..."` adds one sentence to a list. If you never give it, `a.text` is `None`.
- `--tokenizer` (Day 6 file) and `--out` (the Markdown file to write) are `Path` options with defaults.
- Load the tokenizer.
- `sentences = a.text or list(DEFAULT_SENTENCES)` — use your sentences if you gave any; otherwise (`None`) use the built-in five. (`or` picks the second value when the first is empty/None.)

```python
    md = [
        "# Day 7 demo: text -> pieces -> numbers -> text",
        "",
        f"Generated by `scripts/day7_tokenizer_demo.py` on {datetime.now():%Y-%m-%d %H:%M} "
        f"with `data/tokenizer/tokenizer.json` (vocabulary {tok.vocab_size:,}). Real output only.",
        "",
        "Pieces shown as `<xx..>` are single bytes of a character that never occurs in the training questions.",
        "",
    ]
    all_ok = True
```
- `md` — the list of Markdown lines for the saved file (same pattern as other reports): a title, a "generated by" line with the date and vocabulary size, and a note explaining `<xx..>` pieces.
- `all_ok` — starts as `True`; becomes `False` if any sentence fails to restore.

```python
    for text in sentences:
        rows, decoded, ok = demo_rows(tok, text)
        all_ok &= ok
```
- For each sentence run `demo_rows`. `all_ok &= ok` is `all_ok = all_ok & ok` — logical AND: once one `ok` is `False`, `all_ok` stays `False`.

```python
        print(f"\nInput   : {text}")
        print(f"Pieces  : {' / '.join(p for p, _ in rows)}")
        print(f"Numbers : {[i for _, i in rows]}")
        print(f"Decoded : {decoded}")
        print(f"Same as input: {'YES' if ok else 'NO'}  ({len(rows)} tokens)")
```
- Show the result on screen in four lines:
  - the input;
  - the pieces joined with ` / ` (`p for p, _ in rows` takes the piece from each pair; `_` ignores the number);
  - the list of numbers (`[i for _, i in rows]`);
  - the decoded text; and a `YES/NO` for "same as input" with the token count.

```python
        md += [
            f"## {text}",
            "",
            "| # | Piece | Number (ID) |",
            "| --- | --- | --- |",
            *(f"| {k} | {p} | {i} |" for k, (p, i) in enumerate(rows, 1)),
            "",
            f"Numbers: `{[i for _, i in rows]}`",
            "",
            f"Decoded back: {decoded} (same as input: **{'yes' if ok else 'no'}**)",
            "",
        ]
```
- Add the same information to the Markdown file:
  - a heading with the sentence;
  - a table with the header row and one row per token. `enumerate(rows, 1)` numbers the rows starting at **1**, and `(p, i)` unpacks each pair;
  - the numbers as a list; and the decoded text with a bold `yes/no`.
- `*( ... )` unpacks the generator into list items (same trick as in Day 4–7).

```python
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"\nAll sentences restored exactly: {'YES' if all_ok else 'NO'}")
    print(f"Saved: {a.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
```
- Make sure the output folder exists, save the Markdown (UTF-8, LF line endings), print the overall verdict and the file path.
- `return 0 if all_ok else 1` — exit code **0** when everything restored, **1** otherwise (so the script also works as an automatic check).
- Same ending as the other scripts.

---

## 6. What the output looks like

For `今日は晴れ` you will see something like (numbers differ on your machine):

```
Input   : 今日は晴れ
Pieces  : 今日 / は / 晴れ
Numbers : [431, 380, 502]
Decoded : 今日は晴れ
Same as input: YES  (3 tokens)
```
and the same content as a table in `docs/DAY7_TOKENIZER_DEMO.md`.

## 7. Small things worth knowing

- **Special characters in the Markdown table.** If a sentence contains `|` (a pipe) or a line break, the saved Markdown table can look broken. The built-in examples do not.
- **Only the tokenizer is needed.** It does not use the token files; it works as soon as Day 6 is done.
- **"Real output only."** The report contains exactly what the tokenizer produced, which makes it good evidence for your school plan.
- **No `--machine` / no log** — it does not record SHA-256 or timings; use `day7_encode_corpus.py` for that.

## 8. Quick summary in one paragraph

The demo takes a few sentences (built-in or your own), shows how the tokenizer cuts each one into pieces, the number of every piece, and the text decoded back from those numbers with a yes/no for "identical to the input", prints it, and saves it as a Markdown file as evidence; it exits with 0 only if every sentence is restored exactly.
