r"""Text generation for Model A: next-token candidates, greedy and sampled generation, repetition control.
Model A の文の生成: 次のトークンの候補、貪欲法と確率による生成、繰り返しの制御

==========================  ENGLISH  ==========================

How next-token probabilities become text, one token at a time:
    1. The model reads the tokens so far and gives a score (logit) to each of the
       1,674 tokens. softmax turns the scores into probabilities that add up to 1.
       -> next_token_candidates() shows the top 10 of them (school plan task 10).
    2. The scores are adjusted (process_logits):
         a. special tokens that must never be written (<pad>, <bos>, <unk>, <unused_*>) are removed
         b. repetition penalty (optional): tokens already used become a little less likely
         c. repetition limits measured from the training questions:
              - the same token may appear at most 4 times in a row  ("社員番号100003")
              - the same 3-token sequence may appear at most 2 times ("1991年から1997年")
            A token that would break a limit is removed, so loops like 「5000000…」 stop,
            but no real question is ever blocked.
    3. One token is chosen:
         greedy   : always the most likely token (same input -> same output)
         sampling : temperature (sharper < 1 < flatter), then top-k (keep the k best),
                    then top-p (keep the smallest set whose probability adds up to p),
                    then a random draw with a fixed seed (reproducible)
    4. Repeat until <eos> (end of question) or max_new_tokens.

SERVER NOTE: pure PyTorch; identical on Windows and Linux, CPU and CUDA.

==========================  日本語  ==========================

次のトークンの確率が文になるまで(1トークンずつ):
    1. モデルがそれまでのトークンを読み、1,674個のトークンそれぞれに点数(ロジット)を付ける。
       softmax で合計1の確率にする。→ next_token_candidates() が上位10個を示す(作業10)。
    2. 点数を調整する(process_logits):
         a. 書いてはいけない特別なトークン(<pad>, <bos>, <unk>, <unused_*>)を除く
         b. 繰り返しのペナルティ(任意): すでに使ったトークンを少し選ばれにくくする
         c. 学習用の質問文から測った繰り返しの上限:
              - 同じトークンは続けて4回まで(「社員番号100003」)
              - 同じ3トークンの並びは1つの文に2回まで(「1991年から1997年」)
            上限を超えるトークンを除くので、「5000000…」のような繰り返しは止まり、
            本当の質問文が止められることはない。
    3. 1つのトークンを選ぶ:
         貪欲法   : いつも一番確率の高いトークン(同じ入力なら同じ出力)
         サンプリング: 温度(1より小さいと鋭く、大きいと平らに)→ top-k(上位k個だけ残す)
                    → top-p(確率の合計が p になる最小の集まりだけ残す)→ 固定した乱数の種で選ぶ
    4. <eos>(質問の終わり)か、最大の長さまで繰り返す。
"""
from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer
from src.model import ModelA, ModelConfig
from src.training import load_checkpoint

# Measured from the training questions on Day 10 (see the walkthrough): the largest values that occur in real questions.
TRAIN_MAX_SAME_TOKEN_RUN: int = 4
TRAIN_MAX_TRIGRAM_REPEATS: int = 2

LogitsFn = Callable[[list[int]], torch.Tensor]


@dataclass
class SamplingConfig:
    method: str = "greedy"              # "greedy" or "sample"
    temperature: float = 1.0
    top_k: int = 0                      # 0 = off
    top_p: float = 1.0                  # 1.0 = off
    repetition_penalty: float = 1.0     # 1.0 = off
    max_run: int | None = None          # None = off; TRAIN_MAX_SAME_TOKEN_RUN to switch on
    max_trigram_repeats: int | None = None  # None = off; TRAIN_MAX_TRIGRAM_REPEATS to switch on
    max_new_tokens: int = 40
    seed: int = 42
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.method not in ("greedy", "sample"):
            raise ValueError("method must be 'greedy' or 'sample'")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0 (use method='greedy' for the most likely token)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


# The settings compared on Day 10 (fixed, so the report can be reproduced).
PRESET_SETTINGS: dict[str, SamplingConfig] = {
    "greedy": SamplingConfig(name="greedy"),
    "greedy+limits": SamplingConfig(name="greedy+limits", max_run=TRAIN_MAX_SAME_TOKEN_RUN,
                                    max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
    "sample": SamplingConfig(name="sample", method="sample", temperature=0.8, top_k=50, top_p=0.95,
                             max_run=TRAIN_MAX_SAME_TOKEN_RUN, max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
    "sample-hot": SamplingConfig(name="sample-hot", method="sample", temperature=1.5,
                                 max_run=TRAIN_MAX_SAME_TOKEN_RUN, max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
}


@dataclass
class Generation:
    prompt: str
    ids: list[int]          # generated tokens only (without the prompt; <eos> included if reached)
    text: str               # decoded generated text (without <eos>)
    stop_reason: str        # "eos" or "max_new_tokens"

    @property
    def finished(self) -> bool:
        return self.stop_reason == "eos"


# --------------------------------------------------------------------------- loading


def token_text(tok: ByteBPETokenizer, token_id: int) -> str:
    """A readable form of one token: the text, <eos>, or <xx..> for a piece of a character."""
    if token_id == EOS_ID:
        return "<eos>"
    raw = tok.token_bytes[token_id]
    if token_id < BYTE_OFFSET:
        return raw.decode("utf-8")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + raw.hex() + ">"


def load_model(checkpoint: Path, device: torch.device) -> tuple[ModelA, dict]:
    """A trained model ready for generation, plus information about the checkpoint."""
    state = load_checkpoint(checkpoint, device)
    model = ModelA(ModelConfig.from_dict(state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    info = {"path": str(checkpoint), "step": state["step"], "best_val": state["best_val"],
            "tokenizer_sha256": state["tokenizer_sha256"], "model_name": state["model_config"].get("name", "")}
    return model, info


def model_logits_fn(model: ModelA, device: torch.device) -> LogitsFn:
    """Wrap the model as 'token IDs so far -> scores for the next token' (crops to the block size)."""
    block = model.cfg.block_size

    @torch.no_grad()
    def fn(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        return logits[0, -1].float().cpu()

    return fn


# --------------------------------------------------------------------------- the pieces


def next_token_candidates(logits: torch.Tensor, k: int = 10) -> list[tuple[int, float]]:
    """The k most likely next tokens with their probabilities (the raw model, no adjustments)."""
    probs = F.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)]


def banned_special_ids(vocab_size: int) -> list[int]:
    """Every ID below the byte range except <eos>: never written in a question."""
    return [i for i in range(min(BYTE_OFFSET, vocab_size)) if i != EOS_ID]


def run_length_at_end(ids: Sequence[int]) -> int:
    if not ids:
        return 0
    n = 1
    while n < len(ids) and ids[-1 - n] == ids[-1]:
        n += 1
    return n


def process_logits(logits: torch.Tensor, context: Sequence[int], generated: Sequence[int],
                   cfg: SamplingConfig) -> torch.Tensor:
    """Apply bans and repetition control. `context` = prompt + generated (without <bos>);
    `generated` = only the new tokens. Returns new logits (removed tokens are -inf)."""
    out = logits.clone().float()
    out[banned_special_ids(out.numel())] = float("-inf")

    if cfg.repetition_penalty > 1.0 and generated:
        # CTRL-style penalty on tokens the model has already written in this continuation
        used = torch.tensor(sorted(set(generated)), dtype=torch.long)
        vals = out[used]
        out[used] = torch.where(vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty)

    if cfg.max_run is not None and run_length_at_end(context) >= cfg.max_run:
        out[context[-1]] = float("-inf")         # one more would make the run too long

    if cfg.max_trigram_repeats is not None and len(context) >= 2:
        a, b = context[-2], context[-1]
        counts: dict[int, int] = {}
        for x, y, z in zip(context, context[1:], context[2:]):
            if x == a and y == b:
                counts[z] = counts.get(z, 0) + 1
        for z, n in counts.items():
            if n >= cfg.max_trigram_repeats:
                out[z] = float("-inf")
    return out


def choose_token(logits: torch.Tensor, cfg: SamplingConfig, generator: torch.Generator | None) -> int:
    if not torch.isfinite(logits).any():
        return EOS_ID                            # everything was removed: end the question
    if cfg.method == "greedy":
        return int(logits.argmax())
    scores = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(scores, min(cfg.top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    if cfg.top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        before = torch.cumsum(sorted_p, dim=0) - sorted_p          # probability of the tokens ranked above
        keep = before < cfg.top_p                                   # always keeps the top token
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[order[keep]] = True
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


def prompt_seed(seed: int, prompt: str) -> int:
    """A fixed seed for each (seed, prompt) pair. Every prompt gets its own random sequence (with one shared
    seed, different prompts would get identical random draws), and the same prompt always gets the same one."""
    return (seed * 1_000_003 + zlib.crc32(prompt.encode("utf-8"))) % (2**63)


def generate(logits_fn: LogitsFn, tok: ByteBPETokenizer, prompt: str, cfg: SamplingConfig) -> Generation:
    """Continue `prompt` until <eos> or cfg.max_new_tokens. Reproducible: same inputs + seed -> same output."""
    prompt_ids = tok.encode(prompt)
    ids = [BOS_ID] + prompt_ids
    generated: list[int] = []
    generator = torch.Generator().manual_seed(prompt_seed(cfg.seed, prompt)) if cfg.method == "sample" else None
    stop = "max_new_tokens"
    for _ in range(cfg.max_new_tokens):
        logits = process_logits(logits_fn(ids), prompt_ids + generated, generated, cfg)
        nxt = choose_token(logits, cfg, generator)
        ids.append(nxt)
        generated.append(nxt)
        if nxt == EOS_ID:
            stop = "eos"
            break
    text_ids = [i for i in generated if i != EOS_ID]
    return Generation(prompt, generated, tok.decode(text_ids), stop)


# --------------------------------------------------------------------------- quick automatic checks (task 6 items)


def max_same_token_run(ids: Sequence[int]) -> int:
    best = run = 0
    for i, t in enumerate(ids):
        run = run + 1 if i and ids[i - 1] == t else 1
        best = max(best, run)
    return best


def max_trigram_count(ids: Sequence[int]) -> int:
    counts: dict[tuple[int, int, int], int] = {}
    for tri in zip(ids, ids[1:], ids[2:]):
        counts[tri] = counts.get(tri, 0) + 1
    return max(counts.values(), default=0)


def is_repetitive(ids: Sequence[int]) -> bool:
    """Task 6, item A3: more repetition than any training question has
    (5+ same tokens in a row, or one 3-token sequence 3+ times)."""
    return max_same_token_run(ids) > TRAIN_MAX_SAME_TOKEN_RUN or max_trigram_count(ids) > TRAIN_MAX_TRIGRAM_REPEATS


def distinct_2(all_ids: Sequence[Sequence[int]]) -> float:
    """Task 6, item A4: different 2-token sequences / all 2-token sequences, over all outputs."""
    pairs = [p for ids in all_ids for p in zip(ids, ids[1:])]
    return len(set(pairs)) / len(pairs) if pairs else 0.0


def perplexity_of(loss: float) -> float:
    return math.exp(loss)


def quick_checks(logits_fn: LogitsFn, tok: ByteBPETokenizer, prompts: Sequence[str], cfg: SamplingConfig,
                 train_set: set[str] | None = None) -> dict:
    """Task 6 automatic items on a list of prompts (Day 11/12): finished (A2), repetitive (A3), distinct-2 (A4),
    new sentences (A5, reference), and the generated texts."""
    gens = [generate(logits_fn, tok, p, cfg) for p in prompts]
    n = max(1, len(gens))
    body = [[i for i in g.ids if i != EOS_ID] for g in gens]
    return {
        "finished": sum(g.finished for g in gens) / n,
        "repetitive": sum(is_repetitive(tok.encode(g.prompt) + b) for g, b in zip(gens, body)) / n,
        "distinct2": distinct_2(body),
        "new": None if train_set is None else sum((g.prompt + g.text) not in train_set for g in gens if g.finished),
        "texts": {g.prompt: g.text + (" ⟨eos⟩" if g.finished else " …") for g in gens},
    }
