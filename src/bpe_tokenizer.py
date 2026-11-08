r"""Byte-level BPE tokenizer for Model A, written from scratch (no tokenizer library).
Model A 用のバイトレベル BPE トークナイザー(ライブラリを使わず自作)

==========================  ENGLISH  ==========================

How text becomes token IDs:
1. Pre-tokenizer (script classes): the text is cut into runs of the same kind
   of character: hiragana, katakana, kanji, Latin letters, spaces. Digits and
   punctuation/symbols are always single pieces, so any number can be written
   digit by digit and a punctuation mark never sticks to a word.
   Example: 営業部の社員は何人ですか。 -> 営業部 | の | 社員 | は | 何人 | ですか | 。
2. Bytes: each piece is turned into its UTF-8 bytes. Every byte (0-255) has
   its own token, so ANY text can be encoded: there are no unknown words.
3. Merges: training repeatedly finds the most frequent pair of neighbouring
   tokens and merges it into a new token. Encoding replays the learned merges
   in the order they were learned.

Fixed ID layout:
    0 <pad>, 1 <bos>, 2 <eos>, 3 <unk>
    4 .. 103      <unused_0> .. <unused_99>   reserved spare IDs for later extension
    104 .. 359    the 256 single bytes
    360 ..        learned merges, in training order

The tokenizer is lossless: decode(encode(text)) == text for any text, because
no normalisation is applied and every byte is representable.

SCHOOL-PC SWAP NOTE: pure Python, CPU only. Identical on both PCs.

==========================  日本語  ==========================

文字列がトークンIDになるまで:
1. 前処理(文字の種類で区切る): ひらがな、カタカナ、漢字、ラテン文字、空白の
   それぞれの連続で区切る。数字と記号は常に1文字ずつに分けるので、どんな数でも
   1桁ずつ書け、句読点が単語にくっつくことはない。
   例: 営業部の社員は何人ですか。 -> 営業部 | の | 社員 | は | 何人 | ですか | 。
2. バイト: 各部分を UTF-8 のバイト列にする。0〜255 のすべてのバイトにトークンが
   あるので、どんな文字列でも表せる(未知語が無い)。
3. 結合: 学習では、隣り合うトークンの組のうち一番多いものを探し、新しい
   トークンにまとめることを繰り返す。符号化では、学んだ結合を学んだ順に当てはめる。

ID の並び(固定):
    0 <pad>, 1 <bos>, 2 <eos>, 3 <unk>
    4 .. 103      <unused_0> .. <unused_99>   後で拡張するための予備のID
    104 .. 359    256個の1バイト
    360 ..        学習した結合(学習した順)

正規化をせず、すべてのバイトを表せるので、どんな文字列でも
decode(encode(text)) == text になる(情報が失われない)。

学校PCで使う場合の注意: 純粋な Python で CPU のみ。両方のPCで同じです。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FORMAT_VERSION: str = "model-a-byte-bpe-v1"
PRETOKENIZER_NAME: str = "script_class_v1"
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
NUM_RESERVED: int = 100
RESERVED_FIRST_ID: int = len(SPECIAL_TOKENS)                  # 4
BYTE_OFFSET: int = RESERVED_FIRST_ID + NUM_RESERVED           # 104
FIRST_MERGE_ID: int = BYTE_OFFSET + 256                       # 360

Pair = tuple[int, int]


# ============================================================ pre-tokenizer
def char_class(ch: str) -> str:
    """Script class of one character. Digits and 'other' never form runs."""
    o = ord(ch)
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    if (0x30A1 <= o <= 0x30FA) or o in (0x30FC, 0x30FD, 0x30FE) or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "katakana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or o in (0x3005, 0x3006, 0x30F6):
        return "kanji"
    if (ch.isascii() and ch.isalpha()) or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
        return "latin"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return "other"


_RUN_CLASSES: frozenset[str] = frozenset({"hiragana", "katakana", "kanji", "latin", "space"})


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


# ================================================================ tokenizer
@dataclass
class ByteBPETokenizer:
    """A trained tokenizer: an ordered list of merges plus the fixed ID layout."""

    merges: list[Pair]

    def __post_init__(self) -> None:
        self.ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [s.encode("utf-8") for s in SPECIAL_TOKENS]
        self.token_bytes += [f"<unused_{i}>".encode("utf-8") for i in range(NUM_RESERVED)]
        self.token_bytes += [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self._cache: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- sizes
    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    def truncated(self, vocab_size: int) -> "ByteBPETokenizer":
        """The same tokenizer with only the first merges, so its vocabulary has `vocab_size` entries.
        Because merges are learned in order, this is exactly the tokenizer that training would
        have produced if it had stopped at that size."""
        n = max(0, min(len(self.merges), vocab_size - FIRST_MERGE_ID))
        return ByteBPETokenizer(self.merges[:n])

    # ------------------------------------------------------------- encoding
    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = [BYTE_OFFSET + b for b in piece.encode("utf-8")]
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
        self._cache[piece] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text to token IDs. Special-token strings in the text are treated as ordinary text."""
        ids: list[int] = [BOS_ID] if add_bos else []
        for piece in pretokenize(text):
            ids.extend(self._encode_piece(piece))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Token IDs to text. Special and reserved IDs are skipped by default."""
        out = bytearray()
        for i in ids:
            if i < BYTE_OFFSET:
                if skip_special:
                    continue
            out += self.token_bytes[i]
        return out.decode("utf-8", errors="replace")

    def pieces(self, text: str) -> list[str]:
        """Human-readable token pieces (for demos). A piece that is not complete UTF-8 is shown as <xx..>."""
        result: list[str] = []
        for i in self.encode(text):
            raw = self.token_bytes[i]
            try:
                result.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                result.append("<" + raw.hex() + ">")
        return result

    # ----------------------------------------------------------- save/load
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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": the same bytes on Windows and Linux, so the SHA-256 is the same on every PC
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

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


# ================================================================= training
@dataclass
class TrainingLog:
    """What happened during training, for the report."""

    unique_pieces: int
    total_pieces: int
    possible_merges_upper_bound: int
    merges_learned: int
    stop_reason: str
    last_merge_count: int


def count_pieces(lines: Iterable[str]) -> Counter[str]:
    """How often each pre-token piece occurs in the training lines."""
    counts: Counter[str] = Counter()
    for line in lines:
        counts.update(pretokenize(line))
    return counts


def train_bpe(piece_counts: Counter[str], max_merges: int, min_frequency: int = 2) -> tuple[list[Pair], TrainingLog]:
    """Learn merges from piece counts.

    Each unique piece is a list of token IDs with a weight (how often it occurs).
    Pair counts are kept up to date incrementally: only the pieces that contain the
    merged pair are recounted. Ties are broken by the smaller token IDs, so training
    is fully deterministic.
    """
    pieces = sorted(piece_counts)  # sorted: deterministic order
    words: list[list[int]] = [[BYTE_OFFSET + b for b in p.encode("utf-8")] for p in pieces]
    weights: list[int] = [piece_counts[p] for p in pieces]
    upper_bound = sum(max(0, len(w) - 1) for w in words)

    pair_counts: Counter[Pair] = Counter()
    where: dict[Pair, set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += weights[wi]
            where[pair].add(wi)

    merges: list[Pair] = []
    stop_reason = f"reached max_merges={max_merges}"
    last_count = 0
    while len(merges) < max_merges:
        if not pair_counts:
            stop_reason = "no pairs left (every piece is a single token)"
            break
        best, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if count < min_frequency:
            stop_reason = f"most frequent pair occurs fewer than {min_frequency} times"
            break
        new_id = FIRST_MERGE_ID + len(merges)
        merges.append(best)
        last_count = count
        for wi in sorted(where.pop(best, set())):
            w = words[wi]
            for pair in zip(w, w[1:]):          # remove this word's old pairs
                pair_counts[pair] -= weights[wi]
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair != best:
                    where[pair].discard(wi)
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
            for pair in zip(merged, merged[1:]):  # add its new pairs
                pair_counts[pair] += weights[wi]
                where[pair].add(wi)

    log = TrainingLog(len(pieces), sum(weights), upper_bound, len(merges), stop_reason, last_count)
    return merges, log