r"""Token files for Model A: write, read and check encoded text (used from Day 7 on; Day 9 reads them).
Model A のトークンファイル: 符号化した文の書き込み・読み込み・確認(Day 7 から使い、Day 9 が読む)

==========================  ENGLISH  ==========================

Format of one split (for example data/tokens/train.bin):
    - one long stream of token IDs, stored as unsigned 16-bit integers (uint16),
      little-endian. uint16 holds IDs up to 65,535; our vocabulary is far smaller.
    - every question is stored as  <bos> question tokens <eos>, one after another.
    - a second file (train_offsets.bin, int64) stores where each question starts,
      so a single question can be found again without searching.
    - a JSON file (train_meta.json) stores the vocabulary size, the tokenizer's
      SHA-256 and the counts, so a training run can refuse a mismatched file.

Why one stream: the trainer (Day 9) cuts training windows from the stream with a
memory map, without loading the whole file into memory.

SERVER NOTE: plain Python + NumPy, identical on Windows and Linux. Files are
written little-endian explicitly, so a file made on one machine reads the same
on the other.

==========================  日本語  ==========================

1つの分割のファイル形式(例: data/tokens/train.bin):
    - トークンIDを1本の長い列にして、符号なし16ビット整数(uint16、リトルエンディアン)で保存する。
      uint16 は 65,535 までのIDを表せる。私たちの語彙はそれよりずっと小さい。
    - 各質問は <bos> 質問のトークン <eos> の形で、順番につなげて保存する。
    - 2つ目のファイル(train_offsets.bin、int64)に各質問の始まりの位置を保存するので、
      探さなくても1つの質問を取り出せる。
    - JSON ファイル(train_meta.json)に語彙の大きさ、トークナイザーの SHA-256、件数を保存するので、
      学習のときに合わないファイルを使うことを防げる。

1本の列にする理由: 学習プログラム(Day 9)はメモリマップを使って列から学習用の区間を切り出すので、
ファイル全体をメモリに読み込まずに済む。

サーバーで使う場合の注意: 普通の Python と NumPy だけで、Windows と Linux で同じです。
リトルエンディアンを明示して書くので、どちらのマシンで作ったファイルも同じように読めます。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

TOKEN_DTYPE = np.dtype("<u2")    # uint16, little-endian
OFFSET_DTYPE = np.dtype("<i8")   # int64, little-endian
FORMAT_VERSION = "model-a-tokens-v1"


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_lines(lines: Iterable[str], encode, bos_id: int, eos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode lines into one stream (<bos> ... <eos> per line) and the start offset of each line."""
    stream: list[int] = []
    offsets: list[int] = []
    for line in lines:
        offsets.append(len(stream))
        stream.append(bos_id)
        stream.extend(encode(line))
        stream.append(eos_id)
    if stream and max(stream) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("a token ID does not fit in uint16")
    return np.asarray(stream, dtype=TOKEN_DTYPE), np.asarray(offsets, dtype=OFFSET_DTYPE)


def split_stream(tokens: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """The token IDs of each line, including its <bos> and <eos>."""
    ends = list(offsets[1:]) + [len(tokens)]
    return [tokens[s:e] for s, e in zip(offsets, ends)]


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


def read_meta(out_dir: Path, split: str) -> TokenFileMeta:
    data = json.loads((out_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
    if data.get("format") != FORMAT_VERSION:
        raise ValueError(f"{split}_meta.json: unknown format {data.get('format')}")
    return TokenFileMeta(**data)


def read_token_files(out_dir: Path, split: str, memmap: bool = True) -> tuple[np.ndarray, np.ndarray, TokenFileMeta]:
    """Open a split. With memmap=True the tokens stay on disk until used (what the trainer needs)."""
    meta = read_meta(out_dir, split)
    path = out_dir / f"{split}.bin"
    tokens = np.memmap(path, dtype=TOKEN_DTYPE, mode="r") if memmap else np.fromfile(path, dtype=TOKEN_DTYPE)
    offsets = np.fromfile(out_dir / f"{split}_offsets.bin", dtype=OFFSET_DTYPE)
    if len(tokens) != meta.tokens or len(offsets) != meta.questions:
        raise ValueError(f"{split}: file sizes do not match {split}_meta.json")
    return tokens, offsets, meta
