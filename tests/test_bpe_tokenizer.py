"""Unit tests for src/bpe_tokenizer.py (Day 6).
src/bpe_tokenizer.py の単体テスト(Day 6)

Run (VS Code terminal, repo root):
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import (  # noqa: E402
    BOS_ID,
    BYTE_OFFSET,
    EOS_ID,
    FIRST_MERGE_ID,
    NUM_RESERVED,
    PAD_ID,
    RESERVED_FIRST_ID,
    UNK_ID,
    ByteBPETokenizer,
    count_pieces,
    pretokenize,
    train_bpe,
)

TINY_CORPUS = [
    "営業部の社員の平均年収を教えてください。",
    "開発部の社員の平均月収を教えてください。",
    "営業部で年収が600万円以上の人は何人ですか。",
    "大阪支社の社員を一覧で見せて。",
] * 5

TRICKY_TEXTS = [
    "",
    "今日は晴れ",
    "ABC abc 123",
    "絵文字😀と記号①②",
    "\t改行\nを含む\r\n",
    "<bos>や<unused_5>は普通の文字",
    "ﾊﾝｶｸｶﾅとＺＥＮＫＡＫＵ",
]


@pytest.fixture(scope="module")
def tiny_tokenizer() -> ByteBPETokenizer:
    merges, _ = train_bpe(count_pieces(TINY_CORPUS), max_merges=10_000, min_frequency=2)
    return ByteBPETokenizer(merges)


def test_fixed_id_layout() -> None:
    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)
    assert RESERVED_FIRST_ID == 4 and NUM_RESERVED == 100
    assert BYTE_OFFSET == 104 and FIRST_MERGE_ID == 360
    tok = ByteBPETokenizer([])
    assert tok.vocab_size == 360
    assert tok.token_bytes[4] == b"<unused_0>" and tok.token_bytes[103] == b"<unused_99>"
    assert tok.token_bytes[BYTE_OFFSET] == b"\x00" and tok.token_bytes[359] == b"\xff"


@pytest.mark.parametrize("text", TRICKY_TEXTS + TINY_CORPUS[:4])
def test_pretokenize_is_lossless(text: str) -> None:
    assert "".join(pretokenize(text)) == text


def test_pretokenize_examples() -> None:
    assert pretokenize("営業部の社員は何人ですか。") == ["営業部", "の", "社員", "は", "何人", "ですか", "。"]
    assert pretokenize("600万円") == ["6", "0", "0", "万円"]  # digits are always single pieces
    assert pretokenize("ボーナス、") == ["ボーナス", "、"]


def test_training_is_deterministic() -> None:
    counts = count_pieces(TINY_CORPUS)
    assert train_bpe(counts, 10_000, 2)[0] == train_bpe(Counter(dict(reversed(list(counts.items())))), 10_000, 2)[0]


@pytest.mark.parametrize("text", TRICKY_TEXTS + TINY_CORPUS[:4])
def test_round_trip(tiny_tokenizer: ByteBPETokenizer, text: str) -> None:
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_merges_shorten_known_text(tiny_tokenizer: ByteBPETokenizer) -> None:
    text = TINY_CORPUS[0]
    assert len(tiny_tokenizer.encode(text)) < len(text.encode("utf-8"))
    assert "営業部" in tiny_tokenizer.pieces(text)


def test_bos_eos_and_skip_special(tiny_tokenizer: ByteBPETokenizer) -> None:
    ids = tiny_tokenizer.encode("社員", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert tiny_tokenizer.decode(ids) == "社員"
    assert tiny_tokenizer.decode([PAD_ID, 4, *tiny_tokenizer.encode("社員")]) == "社員"


def test_truncated_is_prefix(tiny_tokenizer: ByteBPETokenizer) -> None:
    small = tiny_tokenizer.truncated(FIRST_MERGE_ID + 5)
    assert small.vocab_size == FIRST_MERGE_ID + 5
    assert small.merges == tiny_tokenizer.merges[:5]
    assert small.decode(small.encode(TINY_CORPUS[0])) == TINY_CORPUS[0]


def test_save_and_load_identical(tiny_tokenizer: ByteBPETokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tiny_tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.merges == tiny_tokenizer.merges
    for text in TRICKY_TEXTS + TINY_CORPUS[:4]:
        assert loaded.encode(text) == tiny_tokenizer.encode(text)


def test_real_tokenizer_round_trip_on_validation() -> None:
    """Runs only after Day 6 has produced data/tokenizer/tokenizer.json and Day 5 the validation file."""
    tok_path = REPO_ROOT / "data/tokenizer/tokenizer.json"
    val_path = REPO_ROOT / "data/corpus/val.txt"
    if not (tok_path.exists() and val_path.exists()):
        pytest.skip("run Day 5 and Day 6 first")
    tok = ByteBPETokenizer.load(tok_path)
    lines = [ln for ln in val_path.read_text(encoding="utf-8").split("\n") if ln]
    assert all(tok.decode(tok.encode(ln)) == ln for ln in lines)
