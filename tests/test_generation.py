"""Unit tests for src/generation.py (Day 10). CPU only, a few seconds. No trained checkpoint needed.
src/generation.py の単体テスト(Day 10)。CPU だけで数秒。学習済みのチェックポイントは不要。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer  # noqa: E402
from src.generation import (  # noqa: E402
    TRAIN_MAX_SAME_TOKEN_RUN,
    TRAIN_MAX_TRIGRAM_REPEATS,
    SamplingConfig,
    banned_special_ids,
    choose_token,
    distinct_2,
    generate,
    is_repetitive,
    next_token_candidates,
    process_logits,
    prompt_seed,
)

TOKENIZER = REPO_ROOT / "data/tokenizer/tokenizer.json"
V = 400
A, B, C = 300, 301, 302      # three ordinary byte-range token IDs used in the tests


def flat(**scores: float) -> torch.Tensor:
    t = torch.zeros(V)
    for k, v in scores.items():
        t[int(k[1:])] = v
    return t


def test_candidates_are_sorted_probabilities() -> None:
    logits = torch.randn(V, generator=torch.Generator().manual_seed(0))
    cands = next_token_candidates(logits, 10)
    probs = [p for _, p in cands]
    assert len(cands) == 10 and probs == sorted(probs, reverse=True)
    assert probs[0] == pytest.approx(torch.softmax(logits, 0).max().item())
    assert 0 < sum(probs) <= 1


def test_special_tokens_are_never_written_but_eos_is_allowed() -> None:
    logits = torch.zeros(V)
    logits[BOS_ID] = 100.0
    out = process_logits(logits, [A], [], SamplingConfig())
    assert all(out[i] == float("-inf") for i in banned_special_ids(V))
    assert EOS_ID not in banned_special_ids(V) and torch.isfinite(out[EOS_ID])
    assert torch.isfinite(out[BYTE_OFFSET:]).all()


def test_same_token_run_limit_matches_training_data() -> None:
    cfg = SamplingConfig(max_run=TRAIN_MAX_SAME_TOKEN_RUN)
    logits = flat(t300=10.0)
    ok = process_logits(logits, [A] * (TRAIN_MAX_SAME_TOKEN_RUN - 1), [], cfg)   # 3 in a row: a 4th is allowed
    assert torch.isfinite(ok[A])
    stop = process_logits(logits, [A] * TRAIN_MAX_SAME_TOKEN_RUN, [], cfg)        # 4 in a row: a 5th is not
    assert stop[A] == float("-inf")


def test_trigram_limit_matches_training_data() -> None:
    cfg = SamplingConfig(max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS)
    once = [A, B, C, A, B]                    # A B C seen once: A B C again is allowed (like 1991年から1997年)
    assert torch.isfinite(process_logits(flat(t302=5.0), once, [], cfg)[C])
    twice = [A, B, C, A, B, C, A, B]          # seen twice: a third time is not
    assert process_logits(flat(t302=5.0), twice, [], cfg)[C] == float("-inf")


def test_repetition_penalty_lowers_only_used_tokens() -> None:
    logits = flat(t300=2.0, t301=-1.0, t302=2.0)
    out = process_logits(logits, [A, B], [A, B], SamplingConfig(repetition_penalty=2.0))
    assert out[A] == pytest.approx(1.0)       # positive score divided
    assert out[B] == pytest.approx(-2.0)      # negative score multiplied
    assert out[C] == pytest.approx(2.0)       # unused token unchanged


def test_greedy_takes_the_top_token() -> None:
    assert choose_token(flat(t300=1.0, t301=3.0), SamplingConfig(), None) == B


def test_top_k_one_and_tiny_top_p_behave_like_greedy() -> None:
    logits = flat(t300=1.0, t301=3.0, t302=2.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert choose_token(logits, SamplingConfig(method="sample", top_k=1), g) == B
        assert choose_token(logits, SamplingConfig(method="sample", top_p=0.01), g) == B


def test_temperature_changes_how_often_the_second_token_is_chosen() -> None:
    logits = torch.full((V,), float("-inf"))
    logits[A], logits[B] = 2.0, 1.0

    def share_b(t: float) -> float:
        g = torch.Generator().manual_seed(1)
        cfg = SamplingConfig(method="sample", temperature=t)
        return sum(choose_token(logits, cfg, g) == B for _ in range(2000)) / 2000

    expected = 1 / (1 + math.e)               # softmax probability of B at temperature 1
    assert share_b(1.0) == pytest.approx(expected, abs=0.04)
    assert share_b(0.5) < share_b(1.0) < share_b(2.0)


def test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts() -> None:
    assert prompt_seed(42, "営業部の") == prompt_seed(42, "営業部の")
    assert prompt_seed(42, "営業部の") != prompt_seed(42, "開発部の")
    assert prompt_seed(42, "営業部の") != prompt_seed(7, "営業部の")


@pytest.mark.skipif(not TOKENIZER.exists(), reason="needs data/tokenizer/tokenizer.json (Day 6)")
def test_generate_stops_at_eos_and_is_reproducible() -> None:
    tok = ByteBPETokenizer.load(TOKENIZER)
    target = tok.encode("社員を教えてください。")

    def fake_model(ids: list[int]) -> torch.Tensor:      # writes the target, then <eos>
        n = len(ids) - 1 - len(tok.encode("営業部の"))
        logits = torch.zeros(tok.vocab_size)
        logits[target[n] if n < len(target) else EOS_ID] = 20.0
        return logits

    g = generate(fake_model, tok, "営業部の", SamplingConfig())
    assert g.finished and g.text == "社員を教えてください。" and g.ids[-1] == EOS_ID
    s1 = generate(fake_model, tok, "営業部の", SamplingConfig(method="sample", temperature=1.0))
    s2 = generate(fake_model, tok, "営業部の", SamplingConfig(method="sample", temperature=1.0))
    assert s1.ids == s2.ids

    def never_ends(ids: list[int]) -> torch.Tensor:
        logits = torch.zeros(tok.vocab_size)
        logits[EOS_ID] = -50.0
        return logits

    cut = generate(never_ends, tok, "営業部の", SamplingConfig(max_new_tokens=7))
    assert cut.stop_reason == "max_new_tokens" and len(cut.ids) == 7


def test_repetition_check_and_distinct_2() -> None:
    assert not is_repetitive([A, A, A, A])            # 4 in a row occurs in training questions
    assert is_repetitive([A] * 5)
    assert is_repetitive([A, B, C] * 3)
    assert distinct_2([[A, B, A, B]]) == pytest.approx(2 / 3)
    assert distinct_2([]) == 0.0


def test_invalid_settings_are_refused() -> None:
    for bad in ({"method": "beam"}, {"temperature": 0}, {"top_p": 0}, {"repetition_penalty": 0.5}):
        with pytest.raises(ValueError):
            SamplingConfig(**bad)
