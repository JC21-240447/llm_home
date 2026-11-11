r"""Day 7 - test the tokenizer on every question and encode the corpus into token files.
Day 7 - すべての質問でトークナイザーを確かめ、コーパスをトークンファイルに符号化する

==========================  ENGLISH  ==========================

1. Loads data/tokenizer/tokenizer.json (Day 6) and the three splits (Day 5).
2. Round trip on EVERY question of train, validation and test, plus tricky texts
   (emoji, English, symbols, line breaks): decode(encode(text)) must equal text.
   The script stops if even one fails.
3. Key glossary terms (column names, values, query words and their variants):
   how many tokens each one needs. Written to docs/DAY7_TERM_TOKENS.csv.
4. Encodes each split as one stream  <bos> question <eos> <bos> question <eos> ...
   into data/tokens/{train,val,test}.bin (uint16) with offsets and meta files.
5. Reads the files back from disk and decodes them: every question must come
   back exactly (proves the binary files themselves are correct).
6. Reports the REAL token counts (Day 5 could only give bounds), tokens per
   character, question lengths in tokens, and tokens for characters never learned.
   Writes docs/DAY7_ENCODING_REPORT.md and a run_log.md entry.

Usage (VS Code terminal, repo root):
    .venv\Scripts\python.exe scripts\day7_encode_corpus.py --machine home

SERVER NOTE (Linux, later): the code is the same. Use forward slashes and the
Linux venv path, and the label server:
    .venv/bin/python scripts/day7_encode_corpus.py --machine server
The token files are identical on both machines (compare the SHA-256 in the report).

==========================  日本語  ==========================

1. data/tokenizer/tokenizer.json(Day 6)と、3つの分割(Day 5)を読み込む。
2. 学習用・検証用・テスト用の **すべての** 質問と、難しい文(絵文字、英語、記号、改行)で、
   decode(encode(text)) == text になることを確かめる。1つでも違えば止まる。
3. 重要な用語(列名、値、検索語とその言い換え)が、それぞれ何トークンになるかを調べ、
   docs/DAY7_TERM_TOKENS.csv に書き出す。
4. 各分割を <bos> 質問 <eos> <bos> 質問 <eos> ... の1本の列にして、
   data/tokens/{train,val,test}.bin(uint16)と、位置のファイル、情報のファイルに保存する。
5. ディスクからファイルを読み直して復号し、すべての質問が完全に元に戻ることを確かめる
   (バイナリファイルそのものが正しいことの証明)。
6. **実際の** トークン数(Day 5 では範囲しか分からなかった)、1文字あたりのトークン数、
   質問のトークン数、学習していない文字のトークンを報告する。
   docs/DAY7_ENCODING_REPORT.md を書き、run_log.md に追記する。

使い方(VS Codeターミナル、リポジトリのルートで):
    .venv\Scripts\python.exe scripts\day7_encode_corpus.py --machine home

サーバーで使う場合の注意(Linux、後日): コードは同じです。スラッシュと Linux の venv のパス、
server というラベルを使います:
    .venv/bin/python scripts/day7_encode_corpus.py --machine server
トークンファイルは両方のマシンで同じになります(報告の SHA-256 で比べる)。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer  # noqa: E402
from src.token_data import (  # noqa: E402
    FORMAT_VERSION,
    TokenFileMeta,
    encode_lines,
    read_token_files,
    sha256_bytes,
    split_stream,
    write_token_files,
)

SCRIPT_VERSION: str = "day7-v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")
KEY_DB_GROUPS: frozenset[str] = frozenset({"列名", "値", "検索語"})
TRICKY_TEXTS: tuple[str, ...] = (
    "", "今日は晴れ", "Hello, world! 123", "絵文字😀👍と記号①②③※", "改行\nとタブ\tを含む文",
    "ＡＢＣ全角とﾊﾝｶｸ", "<bos><eos><unused_7> は普通の文字", "「」『』【】・〜…", "𠮷野家(4バイトの漢字)",
)


@dataclass
class SplitReport:
    split: str
    questions: int
    characters: int
    tokens: int                # including <bos>/<eos>
    content_tokens: int        # without <bos>/<eos>
    avg_tokens: float          # per question, including <bos>/<eos>
    max_tokens: int
    partial_tokens: int        # tokens that are only part of a character (a character never learned)
    questions_with_partial: int
    tokens_sha256: str


def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]


def load_key_terms(glossary: Path, synonyms: Path) -> list[tuple[str, str, str]]:
    """(term, group, source) for column names, values and query words, standard terms and variants."""
    out: dict[str, tuple[str, str, str]] = {}
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                g = r["DB区分"].strip()
                t = r["標準用語"].strip()
                if g in KEY_DB_GROUPS and t:
                    out.setdefault(t, (t, g, "用語集"))
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                g = r["DB区分"].strip()
                t = r["変形・同義語"].strip()
                if g in KEY_DB_GROUPS and t:
                    out.setdefault(t, (t, g, "同義語辞書"))
    return sorted(out.values())


def partial_token_ids(tok: ByteBPETokenizer) -> set[int]:
    """IDs whose bytes are not complete characters: used only for characters training never learned.
    (Single ASCII bytes such as digits are complete characters and are NOT counted.)"""
    out: set[int] = set()
    for i in range(BYTE_OFFSET, tok.vocab_size):
        try:
            tok.token_bytes[i].decode("utf-8")
        except UnicodeDecodeError:
            out.add(i)
    return out


def write_report(path: Path, reports: list[SplitReport], tok: ByteBPETokenizer, tok_sha: str,
                 term_rows: list[list[str]], n_tricky: int) -> None:
    in_train = [r for r in term_rows if r[5] == "yes"]
    dist_all: Counter = Counter(min(int(r[3]), 6) for r in term_rows)
    dist_in: Counter = Counter(min(int(r[3]), 6) for r in in_train)
    long_in = [f"{r[0]} ({r[3]})" for r in in_train if int(r[3]) > 3]
    lines = [
        "# Day 7 report: tokenizer tests and encoded corpus",
        "",
        f"Generated by `scripts/day7_encode_corpus.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}.",
        f"Tokenizer: `data/tokenizer/tokenizer.json`, vocabulary {tok.vocab_size:,}, SHA-256 `{tok_sha}`.",
        "",
        "## Round trip",
        "",
        f"Every question in train, validation and test, plus {n_tricky} tricky texts (emoji, English, symbols,",
        "line breaks, a 4-byte kanji), was encoded and decoded: **all identical to the original**.",
        "The token files were then read back from disk and decoded: **all questions identical**.",
        "",
        "## Real token counts",
        "",
        "| Split | Questions | Characters | Tokens (with `<bos>`/`<eos>`) | Tokens per character | Avg tokens per question | Longest question | Partial-character tokens | Questions using them |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {r.split} | {r.questions:,} | {r.characters:,} | {r.tokens:,} | {r.content_tokens / r.characters:.3f} "
            f"| {r.avg_tokens:.1f} | {r.max_tokens} | {r.partial_tokens:,} | {r.questions_with_partial:,} |"
            for r in reports
        ),
        "",
        "Partial-character tokens are pieces of a character that training never learned (written byte by byte).",
        "Digits and ASCII symbols are single bytes but complete characters, so they are not counted.",
        "Tokens per character counts the question text only (without `<bos>`/`<eos>`).",
        "",
        "## Key terms",
        "",
        f"{len(term_rows)} key terms (column names, values, query words and their variants from the glossary and the",
        f"synonym dictionary); {len(in_train)} of them occur in the training questions. Tokens needed per term:",
        "",
        "| Tokens | Terms in the training questions | All key terms |",
        "| --- | --- | --- |",
        *(f"| {k if k < 6 else '6 or more'} | {dist_in[k]} | {dist_all[k]} |" for k in sorted(dist_all)),
        "",
        f"Terms in the training questions that need more than 3 tokens: {', '.join(long_in) or 'none'}.",
        "They are phrases of several pieces, or words that only occur inside longer words.",
        "Terms that never occur in the questions are written with smaller tokens; they still encode and decode",
        "exactly. Every term with its pieces is listed in `docs/DAY7_TERM_TOKENS.csv`.",
        "",
        "## Files (SHA-256)",
        "",
        *(f"- `data/tokens/{r.split}.bin`: `{r.tokens_sha256}`" for r in reports),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def append_run_log(log_path: Path, machine: str, reports: list[SplitReport], tok_sha: str) -> None:
    body = [
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Script: scripts/day7_encode_corpus.py ({SCRIPT_VERSION}); Python {platform.python_version()}",
        f"- Tokenizer SHA-256: {tok_sha}",
        *(f"- {r.split}: {r.questions:,} questions, {r.tokens:,} tokens (avg {r.avg_tokens:.1f}, max {r.max_tokens}), "
          f"SHA-256 {r.tokens_sha256}" for r in reports),
        "- Checks: round trip PASS on every question and tricky text; disk read-back PASS",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 7 encode corpus\n\n" + "\n".join(body) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 7: test the tokenizer on every question and encode the corpus.")
    ap.add_argument("--machine", choices=["home", "server", "school"], required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "data/corpus")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/tokens")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY7_ENCODING_REPORT.md")
    ap.add_argument("--terms-csv", type=Path, default=REPO_ROOT / "docs/DAY7_TERM_TOKENS.csv")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("tokenizer", "corpus_dir", "out_dir", "glossary", "synonyms", "report", "terms_csv", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    if not a.tokenizer.exists():
        raise SystemExit(f"Missing {a.tokenizer}. Run Day 6 first.")

    print(f"=== Day 7 encode corpus | machine={a.machine} | Python {platform.python_version()} ===")
    tok = ByteBPETokenizer.load(a.tokenizer)
    tok_sha = sha256_bytes(a.tokenizer.read_bytes())
    print(f"Tokenizer: vocabulary {tok.vocab_size:,}, SHA-256 {tok_sha[:12]}...")

    # ---- 1. round trip on tricky texts
    bad = [t for t in TRICKY_TEXTS if tok.decode(tok.encode(t)) != t]
    if bad:
        raise SystemExit(f"Round trip FAILED on tricky texts: {bad}")
    print(f"Round trip on {len(TRICKY_TEXTS)} tricky texts: PASS")

    # ---- 2-5. every split: round trip, encode, write, read back
    partial_ids = partial_token_ids(tok)
    reports: list[SplitReport] = []
    for split in SPLITS:
        path = a.corpus_dir / f"{split}.txt"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run Day 5 first.")
        lines = read_lines(path)
        unique = list(dict.fromkeys(lines))
        failures = [ln for ln in unique if tok.decode(tok.encode(ln)) != ln]
        if failures:
            raise SystemExit(f"Round trip FAILED on {len(failures)} {split} questions, e.g. {failures[0]!r}")

        tokens, offsets = encode_lines(lines, tok.encode, BOS_ID, EOS_ID)
        meta = TokenFileMeta(FORMAT_VERSION, split, tok.vocab_size, tok_sha, BOS_ID, EOS_ID, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), sha256_bytes(path.read_bytes()))
        write_token_files(a.out_dir, split, tokens, offsets, meta)

        disk_tokens, disk_offsets, _ = read_token_files(a.out_dir, split, memmap=True)
        for ln, ids in zip(lines, split_stream(disk_tokens, disk_offsets)):
            if ids[0] != BOS_ID or ids[-1] != EOS_ID or tok.decode(ids.tolist()) != ln:
                raise SystemExit(f"Disk read-back FAILED in {split}: {ln!r}")

        per_q = [int(e - s) for s, e in zip(offsets, list(offsets[1:]) + [len(tokens)])]
        is_partial = [int(x) in partial_ids for x in tokens.tolist()]
        q_partial = sum(1 for ids in split_stream(tokens, offsets) if any(int(x) in partial_ids for x in ids.tolist()))
        r = SplitReport(split, len(lines), sum(len(ln) for ln in lines), len(tokens), len(tokens) - 2 * len(lines),
                        sum(per_q) / len(per_q), max(per_q), sum(is_partial), q_partial,
                        hashlib.sha256((a.out_dir / f"{split}.bin").read_bytes()).hexdigest())
        reports.append(r)
        print(f"{split:<6} {r.questions:>8,} questions  {r.tokens:>10,} tokens  avg {r.avg_tokens:4.1f}  max {r.max_tokens:>3}  "
              f"partial-char {r.partial_tokens:>4} in {r.questions_with_partial} questions  -> round trip + read-back PASS")

    # ---- 3. key terms
    terms = load_key_terms(a.glossary, a.synonyms)
    train_blob = "\n".join(read_lines(a.corpus_dir / "train.txt"))
    term_rows: list[list[str]] = []
    for term, group, source in terms:
        n = len(tok.encode(term))
        term_rows.append([term, group, source, str(n), " / ".join(tok.pieces(term)), "yes" if term in train_blob else "no"])
    a.terms_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.terms_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "DB区分", "source", "tokens", "pieces", "in_train_text"])
        w.writerows(term_rows)
    in_train = [r for r in term_rows if r[5] == "yes"]
    print(f"Key terms: {len(term_rows)} (in training text: {len(in_train)}); "
          f"1-3 tokens: {sum(1 for r in term_rows if int(r[3]) <= 3)} of all, "
          f"{sum(1 for r in in_train if int(r[3]) <= 3)} of those in the training text")

    write_report(a.report, reports, tok, tok_sha, term_rows, len(TRICKY_TEXTS))
    total = sum(r.tokens for r in reports)
    print(f"\nTotal tokens: {total:,} (train {reports[0].tokens:,})")
    print(f"Report: {a.report}\nTerm table: {a.terms_csv}")
    if not a.no_log:
        append_run_log(a.run_log, a.machine, reports, tok_sha)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
