r"""Training and validation batches for Model A, read straight from the Day 7 token files.
Model A の学習用・検証用バッチ。Day 7 のトークンファイルから直接読む。

==========================  ENGLISH  ==========================

A training example is a WINDOW of block_size + 1 consecutive tokens cut from the
token stream (<bos> question <eos> <bos> question <eos> ...):
    x = window[:-1]   (what the model reads)
    y = window[1:]    (the next token at every position: what it must predict)
So one window of 256 tokens gives 256 next-token examples at once.

Training windows start at random positions. The random generator's state is
saved in every checkpoint, so a resumed run continues with exactly the batches
the uninterrupted run would have used.

Validation uses FIXED, non-overlapping windows that cover the whole validation
stream, so every evaluation measures the same thing.

The stream is opened with a memory map: only the windows being used are read
from disk, never the whole file.

SERVER NOTE: plain NumPy + PyTorch; identical on Windows and Linux.

==========================  日本語  ==========================

学習の1つの例は、トークンの列から切り出した block_size + 1 個の連続したトークンの「区間」である:
    x = 区間の最後以外(モデルが読むもの)
    y = 区間の最初以外(各位置の次のトークン = 予測すべきもの)
256トークンの区間1つで、256個の「次のトークン」の例が一度に得られる。

学習用の区間はランダムな位置から始まる。乱数の状態をチェックポイントに保存するので、
再開した学習は、中断しなかった場合とまったく同じバッチで続く。

検証用は、検証用の列全体を重ならずに覆う「決まった区間」を使うので、毎回同じものを測れる。
列はメモリマップで開くので、使う区間だけをディスクから読む。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.token_data import TOKEN_DTYPE, read_meta


def open_tokens(tokens_dir: Path, split: str) -> np.ndarray:
    """The token stream of one split as a read-only memory map (checked against its meta file)."""
    meta = read_meta(tokens_dir, split)
    tokens = np.memmap(tokens_dir / f"{split}.bin", dtype=TOKEN_DTYPE, mode="r")
    if len(tokens) != meta.tokens:
        raise ValueError(f"{split}.bin does not match {split}_meta.json")
    return tokens


class RandomWindows:
    """Random training windows with a saveable random state."""

    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int, seed: int) -> None:
        if len(tokens) <= block_size + 1:
            raise ValueError("token stream is shorter than one window")
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.block_size - 1, size=self.batch_size)
        win = np.stack([np.asarray(self.tokens[s:s + self.block_size + 1], dtype=np.int64) for s in starts])
        w = torch.from_numpy(win)
        return w[:, :-1], w[:, 1:]

    def state_dict(self) -> dict:
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["bit_generator"]


def validation_windows(tokens: np.ndarray, block_size: int) -> int:
    """Number of fixed validation windows (the last incomplete piece is not used)."""
    return (len(tokens) - 1) // block_size


def validation_batches(tokens: np.ndarray, block_size: int, batch_size: int,
                       max_windows: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping windows covering the stream from the start, in batches."""
    n = validation_windows(tokens, block_size)
    if max_windows is not None:
        n = min(n, max_windows)
    for first in range(0, n, batch_size):
        idx = range(first, min(first + batch_size, n))
        win = np.stack([np.asarray(tokens[i * block_size:i * block_size + block_size + 1], dtype=np.int64)
                        for i in idx])
        w = torch.from_numpy(win)
        yield w[:, :-1], w[:, 1:]
