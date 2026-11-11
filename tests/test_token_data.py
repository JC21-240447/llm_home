"""Unit tests for src/token_data.py (Day 7).
src/token_data.py の単体テスト(Day 7)

Run (VS Code terminal, repo root):
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer, count_pieces, train_bpe  # noqa: E402
from src.token_data import (  # noqa: E402
    FORMAT_VERSION,
    TOKEN_DTYPE,
    TokenFileMeta,
    encode_lines,
    read_token_files,
    sha256_bytes,
    split_stream,
    write_token_files,
)

LINES = ["営業部の社員は何人ですか。", "開発部の平均年収を教えてください。", "今日は晴れ😀", "A1"]


@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    merges, _ = train_bpe(count_pieces(LINES * 5), 10_000, 2)
    return ByteBPETokenizer(merges)


def test_stream_layout(tok: ByteBPETokenizer) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    assert tokens.dtype == TOKEN_DTYPE and len(offsets) == len(LINES)
    parts = split_stream(tokens, offsets)
    for line, ids in zip(LINES, parts):
        assert ids[0] == BOS_ID and ids[-1] == EOS_ID
        assert tok.decode(ids.tolist()) == line
    assert sum(len(p) for p in parts) == len(tokens)


def test_files_round_trip(tok: ByteBPETokenizer, tmp_path: Path) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    meta = TokenFileMeta(FORMAT_VERSION, "train", tok.vocab_size, "x", BOS_ID, EOS_ID, len(LINES), len(tokens),
                         sha256_bytes(tokens.tobytes()), "y")
    write_token_files(tmp_path, "train", tokens, offsets, meta)
    for memmap in (True, False):
        t2, o2, m2 = read_token_files(tmp_path, "train", memmap=memmap)
        assert np.array_equal(np.asarray(t2), tokens) and np.array_equal(o2, offsets)
        assert m2 == meta
    # little-endian on disk, whatever the machine
    raw = (tmp_path / "train.bin").read_bytes()
    assert int.from_bytes(raw[:2], "little") == BOS_ID


def test_size_mismatch_is_rejected(tok: ByteBPETokenizer, tmp_path: Path) -> None:
    tokens, offsets = encode_lines(LINES, tok.encode, BOS_ID, EOS_ID)
    meta = TokenFileMeta(FORMAT_VERSION, "val", tok.vocab_size, "x", BOS_ID, EOS_ID, len(LINES), len(tokens) + 1,
                         "z", "y")
    write_token_files(tmp_path, "val", tokens, offsets, meta)
    with pytest.raises(ValueError):
        read_token_files(tmp_path, "val")


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
