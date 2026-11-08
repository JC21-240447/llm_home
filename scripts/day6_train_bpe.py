r"""Day 6 - train our own byte-level BPE tokenizer and choose its size from evidence.
Day 6 - 自作のバイトレベル BPE トークナイザーを学習し、大きさを根拠に基づいて決める

==========================  ENGLISH  ==========================

1. Reads data/corpus/train.txt (Day 5). Validation and test text are NOT used
   for training; validation is only used to measure the candidates.
2. Learns every merge the training text allows (it stops by itself when no
   pair occurs at least --min-frequency times).
3. Because merges are learned in order, the first N merges ARE the tokenizer
   of size N. So one training run lets us measure several sizes:
     - tokens needed for the validation text (smaller = better compression),
     - key glossary terms (column names, values, query words) that encode to
       1 to 3 tokens,
     - the longest question in tokens (input for the context-length decision).
4. Chooses the size with a fixed, written rule (see choose_size()):
   the smallest candidate where every key term that CAN be 1-3 tokens is 1-3
   tokens, and validation needs at most --tolerance more tokens than the
   largest tokenizer.
5. Saves data/tokenizer/tokenizer.json (chosen size) and
   data/tokenizer/tokenizer_full.json (all merges), reloads them and checks
   they behave identically, checks round trips on all validation lines,
   prints demo splits, writes docs/DAY6_TOKENIZER_REPORT.md and a run_log.md entry.

Usage (VS Code terminal, repo root, venv python):
    .venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home
To save another size later without training again:
    .venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home --resave-size 4000

SCHOOL-PC SWAP NOTE:
    Same code, CPU only. On the school PC use --machine school. The result is
    deterministic: the school PC must produce the same tokenizer.json SHA-256.

==========================  日本語  ==========================

1. data/corpus/train.txt(Day 5)を読む。検証用とテスト用の文は学習に使わない。
   検証用は候補の大きさを測るためだけに使う。
2. 学習用の文で可能なすべての結合を学ぶ(どの組も --min-frequency 回以上
   出てこなくなったら自動で止まる)。
3. 結合は順番に学ぶので、最初の N 個の結合が、そのまま大きさ N のトークナイザーになる。
   だから1回の学習で、いくつかの大きさを測れる:
     - 検証用の文に必要なトークン数(少ないほど圧縮が良い)
     - 重要な用語(列名、値、検索語)のうち、1〜3トークンになるもの
     - 一番長い質問のトークン数(文脈の長さを決める材料)
4. 決まったルール(choose_size() を参照)で大きさを選ぶ:
   1〜3トークンになり得る重要な用語がすべて1〜3トークンで、検証用のトークン数が
   一番大きいトークナイザーより --tolerance 以上は増えない、一番小さい候補。
5. data/tokenizer/tokenizer.json(選んだ大きさ)と data/tokenizer/tokenizer_full.json
   (すべての結合)を保存し、読み込み直して同じ動きをすることを確かめ、検証用の
   全行で往復(符号化→復号)を確かめ、分割の例を表示し、
   docs/DAY6_TOKENIZER_REPORT.md を書き、run_log.md に追記する。

使い方(VS Codeターミナル、リポジトリのルートで、venv の python):
    .venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home
あとで学習し直さずに別の大きさを保存する場合:
    .venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home --resave-size 4000

学校PCで使う場合の注意:
    コードは同じで、CPU のみ。学校PCでは --machine school を使う。結果は毎回同じなので、
    学校PCでも tokenizer.json の SHA-256 が同じにならなければならない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import (  # noqa: E402
    FIRST_MERGE_ID,
    ByteBPETokenizer,
    TrainingLog,
    count_pieces,
    pretokenize,
    train_bpe,
)

SCRIPT_VERSION: str = "day6-v1"
KEY_DB_GROUPS: frozenset[str] = frozenset({"列名", "値", "検索語"})
CANDIDATE_SIZES: tuple[int, ...] = (400, 500, 600, 800, 1000, 1200, 1400, 1500, 1600, 2000, 4000, 8000, 16000)
DEMO_SENTENCES: tuple[str, ...] = (
    "今日は晴れ",
    "営業部の社員の平均年収を教えてください。",
    "ボーナスが100万円以上の人は何人ですか。",
    "大阪支社で勤続年数が一番長い人は誰？",
    "Model A tokenizer test 2026",
)


@dataclass
class SizeResult:
    """Measurements for one candidate vocabulary size."""

    vocab_size: int
    val_tokens: int
    tokens_per_char: float
    key_ok: int          # achievable key terms that are 1-3 tokens
    key_achievable: int  # key terms that training can make 1-3 tokens (see achievable_terms())
    max_val_tokens: int
    max_train_tokens: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]


def load_key_terms(glossary: Path, synonyms: Path, train_blob: str) -> list[str]:
    """Key terms (列名, 値, 検索語 and their variants) that actually occur in the training text."""
    terms: set[str] = set()
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            terms |= {r["標準用語"].strip() for r in csv.DictReader(f) if r["DB区分"].strip() in KEY_DB_GROUPS}
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            terms |= {r["変形・同義語"].strip() for r in csv.DictReader(f) if r["DB区分"].strip() in KEY_DB_GROUPS}
    return sorted(t for t in terms if t and t in train_blob)


def achievable_terms(key_terms: list[str], piece_counts: dict[str, int], min_frequency: int) -> list[str]:
    """Key terms that training CAN turn into 1-3 tokens: at most 3 pre-token pieces, and every piece
    occurs on its own in the training text at least min_frequency times. A term that only ever appears
    inside a longer word (for example 勤続 inside 勤続年数) cannot become its own token."""
    out: list[str] = []
    for t in key_terms:
        parts = pretokenize(t)
        if len(parts) <= 3 and all(piece_counts.get(p, 0) >= min_frequency for p in parts):
            out.append(t)
    return out


def measure(tok: ByteBPETokenizer, val: list[str], train_sample: list[str], key_terms: list[str],
            achievable: list[str]) -> SizeResult:
    val_lengths = [len(tok.encode(ln)) for ln in val]
    key_ok = sum(1 for t in achievable if len(tok.encode(t)) <= 3)
    chars = sum(len(ln) for ln in val)
    return SizeResult(
        vocab_size=tok.vocab_size,
        val_tokens=sum(val_lengths),
        tokens_per_char=sum(val_lengths) / chars,
        key_ok=key_ok,
        key_achievable=len(achievable),
        max_val_tokens=max(val_lengths),
        max_train_tokens=max(len(tok.encode(ln)) for ln in train_sample),
    )


def choose_size(results: list[SizeResult], tolerance: float) -> SizeResult:
    """The written rule: the smallest size where all achievable key terms are 1-3 tokens and the
    validation token count is within `tolerance` of the largest tokenizer's count."""
    best_val = results[-1].val_tokens
    for r in results:
        if r.key_ok == r.key_achievable and r.val_tokens <= (1 + tolerance) * best_val:
            return r
    return results[-1]


def roundtrip_failures(tok: ByteBPETokenizer, lines: list[str]) -> int:
    return sum(1 for ln in lines if tok.decode(tok.encode(ln)) != ln)


def write_report(path: Path, log: TrainingLog | None, results: list[SizeResult], chosen: SizeResult,
                 tolerance: float, min_frequency: int, seconds: float, demos: list[tuple[str, list[str]]],
                 key_terms: list[str], achievable: list[str], tok: ByteBPETokenizer, hashes: dict[str, str],
                 n_train: int, n_val: int, new_val_pieces: list[str]) -> None:
    long_terms = [t for t in achievable if len(tok.encode(t)) > 3]
    short_all = sum(1 for t in key_terms if len(tok.encode(t)) <= 3)
    long_other = [f"{t} ({len(tok.encode(t))})" for t in key_terms if t not in achievable and len(tok.encode(t)) > 3]
    lines = [
        "# Day 6 report: tokenizer",
        "",
        f"Generated by `scripts/day6_train_bpe.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Training",
        "",
        f"- Training text: `data/corpus/train.txt`, {n_train:,} lines. Validation ({n_val:,} lines) is used only to measure.",
    ]
    if log is not None:
        lines += [
            f"- Pre-token pieces: {log.total_pieces:,} in total, **{log.unique_pieces:,} unique**.",
            f"- Upper bound on merges for this text: {log.possible_merges_upper_bound:,}"
            " (a merge can never cross a piece boundary).",
            f"- Merges learned: **{log.merges_learned:,}**, so the largest possible vocabulary is "
            f"**{FIRST_MERGE_ID + log.merges_learned:,}**. Stopped because: {log.stop_reason} "
            f"(minimum frequency {min_frequency}).",
            f"- Training time: {seconds:.1f} s.",
        ]
    lines += [
        "",
        "## Size candidates (one training run; the first N merges are the size-N tokenizer)",
        "",
        "| Vocabulary | Validation tokens | Tokens per character | Key terms at 1-3 tokens | Longest val question | Longest train question |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {r.vocab_size:,}{' **(chosen)**' if r is chosen else ''} | {r.val_tokens:,} | {r.tokens_per_char:.3f} "
            f"| {r.key_ok}/{r.key_achievable} | {r.max_val_tokens} | {r.max_train_tokens} |"
            for r in results
        ),
        "",
        f"**Rule:** the smallest size where every achievable key term is 1-3 tokens, and validation",
        f"needs at most {tolerance:.0%} more tokens than the largest tokenizer. **Chosen: {chosen.vocab_size:,}.**",
        "",
        f"Key terms: column names, values and query words from the glossary and synonym dictionary that occur in",
        f"the training text ({len(key_terms)} terms). {len(achievable)} of them are achievable: at most 3 pre-token",
        f"pieces, each occurring on its own in the training text at least {min_frequency} times.",
        f"Achievable key terms longer than 3 tokens at the chosen size: {', '.join(long_terms) or 'none'}.",
        f"All key terms at 1-3 tokens at the chosen size: {short_all}/{len(key_terms)}. The rest, with their token",
        f"counts: {', '.join(long_other) or 'none'}. They are phrases of several pieces, or words that only occur",
        "inside longer words (for example 勤続 only inside 勤続年数), so training cannot make them one token.",
        "",
        f"Validation contains {len(new_val_pieces)} pieces that never occur in training: "
        f"{', '.join(new_val_pieces) or 'none'}. They are still encoded, from smaller tokens.",
        "",
        "## Demo (real output of the chosen tokenizer)",
        "",
        "| Text | Pieces |",
        "| --- | --- |",
        *(f"| {t} | {' / '.join(p)} |" for t, p in demos),
        "",
        "Pieces shown as `<xx..>` are single bytes of a character the training text never used.",
        "Any text can still be written, byte by byte, and decodes back exactly.",
        "",
        "## Files (SHA-256)",
        "",
        *(f"- `{n}`: `{h}`" for n, h in hashes.items()),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_run_log(log_path: Path, machine: str, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 6 tokenizer\n\n{text}\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 6: train the byte-level BPE tokenizer and choose its size.")
    ap.add_argument("--machine", choices=["school", "home"], required=True)
    ap.add_argument("--train", type=Path, default=REPO_ROOT / "data/corpus/train.txt")
    ap.add_argument("--val", type=Path, default=REPO_ROOT / "data/corpus/val.txt")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/tokenizer")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY6_TOKENIZER_REPORT.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--max-merges", type=int, default=100_000, help="safety cap; training normally stops earlier")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--resave-size", type=int, default=None,
                    help="skip training: cut tokenizer_full.json to this size and save it as tokenizer.json")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("train", "val", "glossary", "synonyms", "out_dir", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    for p in (a.train, a.val):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run Day 5 first.")

    print(f"=== Day 6 tokenizer | machine={a.machine} | Python {platform.python_version()} ===")
    train = read_lines(a.train)
    val = read_lines(a.val)
    full_path = a.out_dir / "tokenizer_full.json"
    chosen_path = a.out_dir / "tokenizer.json"

    log: TrainingLog | None = None
    seconds = 0.0
    if a.resave_size is None:
        start = time.perf_counter()
        merges, log = train_bpe(count_pieces(train), a.max_merges, a.min_frequency)
        seconds = time.perf_counter() - start
        full = ByteBPETokenizer(merges)
        full.save(full_path)
        print(f"Unique pre-token pieces: {log.unique_pieces:,} (total {log.total_pieces:,})")
        print(f"Merges learned: {log.merges_learned:,} -> largest vocabulary {full.vocab_size:,} "
              f"({log.stop_reason}); {seconds:.1f} s")
    else:
        if not full_path.exists():
            raise SystemExit(f"--resave-size needs {full_path}. Run without it first.")
        full = ByteBPETokenizer.load(full_path)
        print(f"Loaded {full_path.name}: largest vocabulary {full.vocab_size:,}")

    train_blob = "\n".join(train)
    piece_counts = count_pieces(train)
    key_terms = load_key_terms(a.glossary, a.synonyms, train_blob)
    achievable = achievable_terms(key_terms, piece_counts, a.min_frequency)
    new_val_pieces = sorted({p for ln in val for p in pretokenize(ln)} - set(piece_counts))
    train_sample = train[::10]  # every 10th line: enough to find the longest questions quickly
    sizes = [s for s in CANDIDATE_SIZES if s < full.vocab_size] + [full.vocab_size]
    if a.resave_size is not None and a.resave_size not in sizes:
        sizes = sorted(set(sizes) | {min(a.resave_size, full.vocab_size)})
    print(f"Key terms found in the training text: {len(key_terms)}; achievable as 1-3 tokens: {len(achievable)}")
    print(f"\n{'vocab':>7} {'val tokens':>11} {'tok/char':>9} {'key terms 1-3':>14} {'max val':>8} {'max train':>10}")
    results: list[SizeResult] = []
    for s in sizes:
        r = measure(full.truncated(s), val, train_sample, key_terms, achievable)
        results.append(r)
        print(f"{r.vocab_size:>7,} {r.val_tokens:>11,} {r.tokens_per_char:>9.3f} "
              f"{f'{r.key_ok}/{r.key_achievable}':>14} {r.max_val_tokens:>8} {r.max_train_tokens:>10}")

    if a.resave_size is None:
        chosen = choose_size(results, a.tolerance)
        print(f"\nRule: smallest size with all achievable key terms at 1-3 tokens and validation within "
              f"{a.tolerance:.0%} of the largest -> chosen {chosen.vocab_size:,}")
    else:
        chosen = next(r for r in results if r.vocab_size == min(a.resave_size, full.vocab_size))
        print(f"\nManual choice (--resave-size): {chosen.vocab_size:,}")

    tok = full.truncated(chosen.vocab_size)
    tok.save(chosen_path)

    # ---- reload and verify
    reloaded = ByteBPETokenizer.load(chosen_path)
    reloaded_full = ByteBPETokenizer.load(full_path)
    same = reloaded.merges == tok.merges and reloaded_full.merges == full.merges
    same = same and all(reloaded.encode(ln) == tok.encode(ln) for ln in val)
    extra = ["", "今日は晴れ", "ABC abc 123", "絵文字😀と記号①②", "\t改行\nを含む", "<bos>は普通の文字"]
    failures = roundtrip_failures(reloaded, val + extra)
    if not same or failures:
        raise SystemExit(f"Reload check FAILED: identical={same}, round-trip failures={failures}")
    print(f"Reload check: PASS (identical merges and encodings); round trip: PASS on {len(val) + len(extra):,} texts")

    demos = [(t, reloaded.pieces(t)) for t in DEMO_SENTENCES]
    print("\nDemo (chosen tokenizer):")
    for t, p in demos:
        print(f"  {t}\n    -> {' / '.join(p)}  ({len(p)} tokens)")

    hashes = {
        "data/tokenizer/tokenizer.json": hashlib.sha256(chosen_path.read_bytes()).hexdigest(),
        "data/tokenizer/tokenizer_full.json": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "data/corpus/train.txt (input)": sha256_file(a.train),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(a.report, log, results, chosen, a.tolerance, a.min_frequency, seconds, demos, key_terms,
                 achievable, reloaded, hashes, len(train), len(val), new_val_pieces)
    print(f"\ntokenizer.json SHA-256: {hashes['data/tokenizer/tokenizer.json']}")
    print(f"Report: {a.report}")

    if not a.no_log:
        text = "\n".join([
            "- Machine label: " + a.machine + " (CPU only; no GPU used)",
            f"- Script: scripts/day6_train_bpe.py ({SCRIPT_VERSION}); Python {platform.python_version()}",
            (f"- Training: {log.unique_pieces:,} unique pieces, {log.merges_learned:,} merges, {seconds:.1f} s, "
             f"largest vocabulary {full.vocab_size:,}" if log else "- Re-saved from tokenizer_full.json (no training)"),
            f"- Chosen vocabulary: {chosen.vocab_size:,}; validation {chosen.val_tokens:,} tokens "
            f"({chosen.tokens_per_char:.3f} per char); key terms 1-3 tokens {chosen.key_ok}/{chosen.key_achievable}; "
            f"longest question {max(chosen.max_val_tokens, chosen.max_train_tokens)} tokens",
            f"- tokenizer.json SHA-256: {hashes['data/tokenizer/tokenizer.json']}",
            "- Checks: reload PASS, round trip PASS",
        ])
        append_run_log(a.run_log, a.machine, text)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
