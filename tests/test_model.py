"""Unit tests for src/model.py (Day 8). All run on the CPU in a few seconds.
src/model.py の単体テスト(Day 8)。すべて CPU で数秒で終わる。

Run (VS Code terminal, repo root):
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import (  # noqa: E402
    ATTENTION_MODES,
    ModelA,
    ModelConfig,
    apply_rope,
    count_parameters,
    expected_parameters,
    preset_config,
    rope_tables,
)

VOCAB = 50


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4, d_model=64)
    return ModelConfig(**{**base, **kw})


def make_model(seed: int = 0, **kw) -> ModelA:
    torch.manual_seed(seed)
    return ModelA(tiny_cfg(**kw)).eval()


def test_output_shapes_and_loss() -> None:
    m = make_model()
    idx = torch.randint(0, VOCAB, (3, 10))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 10, VOCAB)
    assert loss is not None and loss.ndim == 0
    assert m(idx)[1] is None


def test_initial_loss_is_close_to_uniform_guess() -> None:
    """Before training the model knows nothing: loss should be about ln(vocab size)."""
    m = make_model()
    idx = torch.randint(0, VOCAB, (8, 32))
    _, loss = m(idx, torch.randint(0, VOCAB, (8, 32)))
    assert abs(loss.item() - math.log(VOCAB)) < 0.3


@pytest.mark.parametrize("mode", ATTENTION_MODES)
def test_causality_future_token_does_not_change_the_past(mode: str) -> None:
    m = make_model(attention=mode)
    idx = torch.randint(0, VOCAB, (2, 20))
    changed = idx.clone()
    k = 12
    changed[:, k] = (changed[:, k] + 1) % VOCAB
    with torch.no_grad():
        a, _ = m(idx)
        b, _ = m(changed)
    assert torch.allclose(a[:, :k], b[:, :k], atol=1e-6)        # earlier positions: identical
    assert not torch.allclose(a[:, k:], b[:, k:], atol=1e-4)    # position k and later: changed


def test_manual_and_fast_attention_agree() -> None:
    m = make_model()
    idx = torch.randint(0, VOCAB, (2, 32))
    with torch.no_grad():
        m.set_attention("fast")
        fast, _ = m(idx)
        m.set_attention("manual")
        manual, _ = m(idx)
    assert torch.allclose(fast, manual, atol=1e-5)


def test_weight_tying() -> None:
    m = make_model()
    assert m.lm_head.weight is m.tok_emb.weight
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_parameter_count_matches_formula() -> None:
    for cfg in (tiny_cfg(), preset_config("tiny", 1674), preset_config("small", 1674)):
        m = ModelA(cfg)
        counts = count_parameters(m)
        assert counts["total"] == expected_parameters(cfg)
        assert counts["embedding"] == cfg.vocab_size * cfg.d_model


def test_presets_are_in_the_planned_size_range() -> None:
    sizes = {name: expected_parameters(preset_config(name, 1674)) for name in ("tiny", "small", "medium")}
    assert 0.5e6 < sizes["tiny"] < 2e6
    assert 3e6 < sizes["small"] < 8e6
    assert 10e6 < sizes["medium"] < 20e6


def test_rope_depends_only_on_relative_position() -> None:
    """Rotary embedding: q·k is the same when both positions shift by the same amount."""
    torch.manual_seed(1)
    cos, sin = rope_tables(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(pq: int, pk: int) -> float:
        rq = apply_rope(q, cos[pq:pq + 1], sin[pq:pq + 1])
        rk = apply_rope(k, cos[pk:pk + 1], sin[pk:pk + 1])
        return float((rq * rk).sum())

    assert score(10, 3) == pytest.approx(score(40, 33), abs=1e-4)
    assert score(10, 3) != pytest.approx(score(10, 9), abs=1e-3)


def test_too_long_input_is_rejected() -> None:
    m = make_model()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))


def test_config_round_trip(tmp_path: Path) -> None:
    cfg = preset_config("small", 1674, dropout=0.1)
    path = tmp_path / "config.json"
    cfg.save(path)
    assert ModelConfig.load(path) == cfg
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_head=3)


def test_overfit_one_batch() -> None:
    """A correct model and loss must be able to memorise one small batch almost perfectly."""
    torch.manual_seed(0)
    m = ModelA(tiny_cfg()).train()
    idx = torch.randint(0, VOCAB, (4, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(300):
        _, loss = m(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert first > 3.0
    assert loss.item() < 0.1
