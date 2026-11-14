"""Unit tests for src/batching.py and src/training.py (Day 9). CPU only, a few seconds.
src/batching.py と src/training.py の単体テスト(Day 9)。CPU だけで数秒。

The most important one: a run that is stopped and resumed produces exactly the same
losses as a run that was never stopped.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.batching import RandomWindows, validation_batches, validation_windows  # noqa: E402
from src.model import ModelA, ModelConfig  # noqa: E402
from src.token_data import FORMAT_VERSION, TokenFileMeta, encode_lines, sha256_bytes, write_token_files  # noqa: E402
from src.training import TrainConfig, lr_at, param_groups, train  # noqa: E402

VOCAB = 40


def test_lr_schedule_warmup_then_cosine() -> None:
    lrs = [lr_at(s, 1e-3, 10, 100, 0.1) for s in range(101)]
    assert lrs[0] == pytest.approx(1e-4)          # first warm-up step
    assert lrs[9] == pytest.approx(1e-3)          # end of warm-up = peak
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))  # then never rises
    assert lrs[100] == pytest.approx(1e-4)        # ends at min_lr
    assert lrs[55] == pytest.approx(1e-4 + 0.5 * 9e-4, rel=1e-6)  # halfway through the cosine


def test_weight_decay_only_on_matrices() -> None:
    m = ModelA(ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32))
    decay, no_decay = param_groups(m, 0.1)
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    ids = [id(p) for g in (decay, no_decay) for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))   # every parameter exactly once (tied once)


def test_random_windows_shift_and_resume_state() -> None:
    tokens = np.arange(1000, dtype="<u2")
    w = RandomWindows(tokens, block_size=16, batch_size=4, seed=0)
    x, y = w.next_batch()
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])       # y is x shifted by one
    state = w.state_dict()
    expected = w.next_batch()
    w2 = RandomWindows(tokens, block_size=16, batch_size=4, seed=999)
    w2.load_state_dict(state)
    got = w2.next_batch()
    assert torch.equal(expected[0], got[0]) and torch.equal(expected[1], got[1])


def test_validation_windows_cover_the_stream_without_overlap() -> None:
    tokens = np.arange(100, dtype="<u2")
    assert validation_windows(tokens, 10) == 9
    xs = torch.cat([x for x, _ in validation_batches(tokens, 10, 4)])
    assert xs.shape == (9, 10)
    assert torch.equal(xs.flatten(), torch.arange(90))


# --------------------------------------------------------------------------- a tiny end-to-end run


def _make_repo(root: Path) -> None:
    rng = np.random.default_rng(0)
    tok_dir = root / "data/tokens"
    for split, n in (("train", 300), ("val", 40)):
        lines = [" ".join(map(str, rng.integers(4, VOCAB, size=rng.integers(3, 12)))) for _ in range(n)]
        tokens, offsets = encode_lines(lines, lambda s: [int(t) for t in s.split()], 1, 2)
        meta = TokenFileMeta(FORMAT_VERSION, split, VOCAB, "test", 1, 2, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), "test")
        write_token_files(tok_dir, split, tokens, offsets, meta)
    (root / "configs").mkdir()
    ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32, name="test").save(
        root / "configs/model_test.json")


def _cfg(name: str, **kw) -> TrainConfig:
    base = dict(run_dir=f"runs/{name}", model_config="configs/model_test.json", tokenizer=None, prompts=[],
                batch_size=4, max_steps=12, warmup_steps=3, lr=3e-3, log_every=1, eval_every=4, device="cpu",
                precision="fp32", seed=7)
    return TrainConfig(**{**base, **kw})


def _losses(root: Path, name: str) -> tuple[dict[int, str], dict[int, str]]:
    with (root / "runs" / name / "log.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return ({int(r["step"]): r["train_loss"] for r in rows if r["event"] == "train"},
            {int(r["step"]): r["val_loss"] for r in rows if r["event"] == "eval"})


def test_stopped_and_resumed_run_equals_uninterrupted_run(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    full = train(_cfg("a"), tmp_path)
    assert full["step"] == 12 and not full["stopped_early"]

    part = train(_cfg("b", stop_after=6), tmp_path)          # 6 is not an evaluation step: last.pt saved at stop
    assert part["step"] == 6 and part["stopped_early"]
    saved = json.loads((tmp_path / "runs/b/train_config.json").read_text(encoding="utf-8"))
    resumed_cfg = TrainConfig.from_dict(saved)
    resumed_cfg.resume, resumed_cfg.stop_after = True, None
    done = train(resumed_cfg, tmp_path)
    assert done["step"] == 12

    train_a, val_a = _losses(tmp_path, "a")
    train_b, val_b = _losses(tmp_path, "b")
    assert train_a == train_b                                   # every step's loss, to the last printed digit
    assert val_a == val_b
    assert float(val_a[12]) < float(val_a[0])                   # and it learned something
    assert float(val_a[0]) == pytest.approx(math.log(VOCAB), abs=0.3)
    assert (tmp_path / "runs/b/best.pt").exists() and (tmp_path / "runs/b/last.pt").exists()


def test_resume_refuses_changed_settings(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("c", stop_after=4), tmp_path)
    changed = _cfg("c", lr=1e-2, resume=True)
    with pytest.raises(SystemExit):
        train(changed, tmp_path)


def test_new_run_refuses_an_existing_folder(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("d", max_steps=2, eval_every=2), tmp_path)
    with pytest.raises(SystemExit):
        train(_cfg("d", max_steps=2, eval_every=2), tmp_path)


def test_resume_after_a_hard_kill_removes_lines_written_after_the_checkpoint(tmp_path: Path) -> None:
    """If the program is killed (window closed), log.csv can contain steps after last.pt. Resuming must
    remove them, so the final log is the same as an uninterrupted run."""
    _make_repo(tmp_path)
    train(_cfg("a"), tmp_path)
    train(_cfg("e", stop_after=6), tmp_path)
    log = tmp_path / "runs/e/log.csv"
    with log.open("a", encoding="utf-8", newline="") as f:      # lines a killed run would have left behind
        f.write("7,train,448,0.001,9.9,9.9,,,1.0,1,,1.0\n8,eval,512,0.001,,,9.9,9.9,,,,1.0\n")
    cfg = TrainConfig.from_dict(json.loads((tmp_path / "runs/e/train_config.json").read_text(encoding="utf-8")))
    cfg.resume, cfg.stop_after = True, None
    train(cfg, tmp_path)
    assert _losses(tmp_path, "a") == _losses(tmp_path, "e")
