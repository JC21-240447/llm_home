r"""Final evaluation tools for Model A (Day 13): the task 6 items that need more than generation.
Model A の最終評価の道具(Day 13): 作業6の項目のうち、生成だけでは測れないもの

==========================  ENGLISH  ==========================

1. The fixed list of ~50 beginnings, by kind (task 6, Table 7).
2. Two extra automatic problem detectors, added after Day 12 showed these errors:
     - WRONG UNIT (problem type "wrong term use"): an item followed by a value in a unit that
       never follows that item in the training questions, e.g. 勤続年数が300万円 (years of
       service in yen). The allowed units are LEARNED from the training questions, not typed
       by hand, and the detector is checked on the real test questions (it should flag ~0).
     - BROKEN NUMBER: a run of digits longer than any in the training questions
       (e.g. 30000100万円). The limit is also measured from the training questions.
     - UNSEEN VALUE (reference): a number + unit that no training question contains
       (e.g. 30000万円 = 300 million yen). Training values come from fixed lists, so a value
       outside them is invented. Reference only: a new value is not always wrong.
3. A counting baseline for A1: a bigram model (next token predicted from the previous token
   only, by counting the training text). It shows how much the Transformer adds over plain
   counting, which is a fairer reference than "untrained".
4. Human scoring (task 6, 6-4): the scoring workbook (Excel) and reading it back with checks,
   plus the agreement rate (within 1 point) and Cohen's kappa between two scorers.

SERVER NOTE: plain Python + NumPy (+ openpyxl for the workbook); identical on Windows and Linux.

==========================  日本語  ==========================

1. 固定の約50の書き出し(種類つき、作業6の表7)
2. Day 12 で見つかった誤りを数える、自動の確認2つ:
     - 単位の誤り(問題の種類「用語の誤り」): 項目のあとに、学習用の質問でその項目に一度も
       続かない単位の値が来る(例: 勤続年数が300万円)。使える単位は手で書かず学習用の質問から
       学ぶ。本物のテスト用の質問で確かめる(ほぼ0件になるはず)
     - 壊れた数字: 学習用の質問のどれよりも長い数字の並び(例: 30000100万円)
     - 学習にない値(参考): どの学習用の質問にもない「数字+単位」(例: 30000万円)
3. A1 の比較の基準: 1つ前のトークンだけから次を数えて予測する「バイグラム」モデル
4. 人による採点(作業6の 6-4): 採点用の Excel ファイル、読み戻しと確認、一致率(差1点以内)、
   コーエンのカッパ係数
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------------- 1. the fixed beginnings

PROMPT_KINDS: dict[str, tuple[str, str, int]] = {
    # kind: (English name, Japanese name, count in task 6 Table 7)
    "dept": ("Department or branch", "部署・勤務地で始まる", 10),
    "item": ("Item", "項目で始まる", 10),
    "count_rank": ("Count or rank", "人数・順位を聞く", 10),
    "unseen": ("Wording not in training", "学習にない言い方", 10),
    "key_term": ("Key term variant", "用語とその言い換え", 5),
    "out_of_scope": ("Out of scope", "対象外", 5),
}


@dataclass(frozen=True)
class EvalPrompt:
    kind: str
    prompt: str


def load_eval_prompts(path: Path) -> list[EvalPrompt]:
    """Read the kind,prompt CSV and check the kinds."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [EvalPrompt(r["kind"].strip(), r["prompt"].strip()) for r in csv.DictReader(f) if r["prompt"].strip()]
    unknown = {r.kind for r in rows} - set(PROMPT_KINDS)
    if unknown:
        raise ValueError(f"unknown prompt kinds: {sorted(unknown)}")
    if len({r.prompt for r in rows}) != len(rows):
        raise ValueError("a beginning appears twice")
    return rows


# ----------------------------------------------------------------------------- 2. problem detectors

_NUMBER_UNIT = r"([0-9][0-9,]*(?:万[0-9]*)?)\s*(円|歳|年|時間|つ|%|％)"


def term_phrases(columns_csv: Path) -> list[str]:
    """Every way the training questions name a numeric item (columns.csv), longest first."""
    phrases: set[str] = set()
    with columns_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            phrases.add(r["列名"].strip())
            for col in ("質問で使う言い方", "平均と一緒に使う言い方"):
                phrases.update(p.strip() for p in r.get(col, "").split("、") if p.strip())
    return sorted(phrases, key=lambda p: (-len(p), p))


class UnitChecker:
    """Learns which units follow which item in the training questions, then flags new pairs."""

    def __init__(self, phrases: Sequence[str]) -> None:
        alts = "|".join(re.escape(p) for p in sorted(phrases, key=lambda p: (-len(p), p)))
        self.pattern = re.compile(rf"({alts})(?:が|は)?{_NUMBER_UNIT}")
        self.allowed: dict[str, set[str]] = {}
        self.values: set[str] = set()
        self._value = re.compile(_NUMBER_UNIT)

    @staticmethod
    def _unit(u: str) -> str:
        return "%" if u == "％" else u

    def pairs(self, text: str) -> list[tuple[str, str]]:
        return [(m.group(1), self._unit(m.group(3))) for m in self.pattern.finditer(text)]

    def learn(self, lines: Iterable[str]) -> "UnitChecker":
        for line in lines:
            for phrase, unit in self.pairs(line):
                self.allowed.setdefault(phrase, set()).add(unit)
            self.values.update(self.value_strings(line))
        return self

    def value_strings(self, text: str) -> list[str]:
        return [m.group(1) + self._unit(m.group(2)) for m in self._value.finditer(text)]

    def unseen_values(self, text: str) -> list[str]:
        """Number + unit strings that no training question contains (reference check)."""
        return [v for v in self.value_strings(text) if v not in self.values]

    def mismatches(self, text: str) -> list[tuple[str, str]]:
        """(item, unit) pairs never seen in training; items never seen with any unit are not judged."""
        return [(p, u) for p, u in self.pairs(text) if p in self.allowed and u not in self.allowed[p]]


def max_digit_run(lines: Iterable[str]) -> int:
    """The longest run of digits in any of the lines."""
    return max((len(m) for line in lines for m in re.findall(r"[0-9]+", line)), default=0)


def has_broken_number(text: str, limit: int) -> bool:
    """True if the text has a run of digits longer than `limit` (the longest in the training questions)."""
    return any(len(m) > limit for m in re.findall(r"[0-9]+", text))


# ----------------------------------------------------------------------------- 3. bigram counting baseline


@dataclass
class BigramBaseline:
    """P(next | previous) = lam * count(previous, next) / count(previous) + (1 - lam) * P_unigram(next).
    The unigram part uses add-one counts, so no token ever gets probability 0."""

    pair_counts: np.ndarray      # (V, V) float64
    row_totals: np.ndarray       # (V,)
    unigram: np.ndarray          # (V,) probabilities
    lam: float = 0.9

    @classmethod
    def fit(cls, tokens: np.ndarray, vocab_size: int) -> "BigramBaseline":
        t = np.asarray(tokens, dtype=np.int64)
        pairs = np.bincount(t[:-1] * vocab_size + t[1:], minlength=vocab_size * vocab_size)
        pair_counts = pairs.reshape(vocab_size, vocab_size).astype(np.float64)
        uni = np.bincount(t, minlength=vocab_size).astype(np.float64) + 1.0
        return cls(pair_counts, pair_counts.sum(axis=1), uni / uni.sum())

    def loss(self, tokens: np.ndarray, lam: float | None = None) -> float:
        """Average -log P over every next token of the stream (natural log, like the model's loss)."""
        lam = self.lam if lam is None else lam
        t = np.asarray(tokens, dtype=np.int64)
        prev, nxt = t[:-1], t[1:]
        totals = self.row_totals[prev]
        p_bi = np.divide(self.pair_counts[prev, nxt], totals, out=np.zeros(len(prev)), where=totals > 0)
        w = np.where(totals > 0, lam, 0.0)
        p = w * p_bi + (1.0 - w) * self.unigram[nxt]
        return float(-np.log(p).mean())

    def tune(self, val_tokens: np.ndarray, grid: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)) -> float:
        """Choose lam on the VALIDATION stream (never the test stream)."""
        self.lam = min(grid, key=lambda g: self.loss(val_tokens, g))
        return self.lam


# ----------------------------------------------------------------------------- 4. agreement between scorers


def agreement_within_one(a: Sequence[int], b: Sequence[int]) -> float:
    """Share of outputs where the two scores differ by at most 1 point (task 6, 6-4)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty score lists")
    return sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = (1, 2, 3, 4, 5)) -> float:
    """Cohen's kappa: exact agreement corrected for agreement expected by chance.
    1 = perfect, 0 = no better than chance."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty score lists")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ----------------------------------------------------------------------------- 5. the scoring workbook

H_ITEMS: tuple[str, ...] = ("H1", "H2", "H3", "H4")
PROBLEM_TYPES: dict[str, tuple[str, str]] = {
    "G": ("Grammar error", "文法の誤り"),
    "R": ("Repetition", "繰り返し"),
    "T": ("Wrong term use", "用語の誤り"),
    "O": ("Off topic", "話題がずれる"),
    "U": ("Unfinished", "途中で終わる"),
    "N": ("Invented name", "存在しない名前"),
}
_HEADERS: tuple[str, ...] = ("ID", "書き出し\nbeginning", "続き(生成された文)\ncontinuation",
                             "H1 正しい日本語\ncorrect Japanese", "H2 自然さ\nnatural", "H3 意味が分かる\nmakes sense",
                             "H4 社内用語\ncompany terms", "問題の種類\nproblem types", "メモ\nnote")
_CRITERIA: tuple[tuple[str, str, str, str, str], ...] = (
    ("5", "誤りなし / no errors", "問題なく読める / reads fine", "何を聞いているか明確 / clearly what is asked",
     "すべて正しい / all correct"),
    ("4", "小さな誤り1つ(意味は同じ) / one small error", "少し不自然な所が1つ / one slightly unnatural spot",
     "明確だが一部あいまい / clear but partly vague", "1つ少しずれている / one slightly off"),
    ("3", "誤り2つ、または意味が少し変わる誤り1つ / two errors, or one that changes the meaning a little",
     "不自然な所が複数 / several unnatural spots", "なんとか推測できる / can just be guessed",
     "はっきりした誤り1つ / one clear error"),
    ("2", "誤りが多く読みにくい / many errors, hard to read", "言葉のつながりが悪い / words join badly",
     "一部だけ分かる / partly clear only", "誤りが複数 / several errors"),
    ("1", "日本語の文になっていない / not a Japanese sentence", "言葉の順番がめちゃくちゃ / words in no order",
     "意味が分からない / no clear meaning", "用語がまったく合わない / terms do not fit at all"),
)


@dataclass
class ScoringRow:
    id: str
    prompt: str
    text: str
    out_of_scope: bool


@dataclass
class Score:
    h: dict[str, int | None]
    types: set[str] = field(default_factory=set)
    note: str = ""


class ScoreFileError(ValueError):
    pass


def write_scoring_workbook(path: Path, rows: Sequence[ScoringRow]) -> None:
    """An Excel file with an instructions sheet and a scoring sheet (1-5 enforced by data validation)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    info = wb.active
    info.title = "説明 Instructions"
    lines = [
        ("採点の説明 / How to score", True),
        ("・一人で採点してください。他の人と相談しない。 / Score alone; do not discuss with others.", False),
        ("・「採点 Scoring」シートの H1〜H4 に 1〜5 の整数を入れる。 / Enter a whole number 1-5 in H1-H4.", False),
        ("・H4 が「-」の行(対象外の書き出し)は H4 を採点しない。 / Rows where H4 shows '-' (out of scope): "
         "do not score H4.", False),
        ("・問題があれば「問題の種類」に下の記号を入れる(複数可、例: T,O)。 / If there is a problem, enter the "
         "codes below (several allowed, e.g. T,O).", False),
        ("・メモに、何が悪かったかを短く書く(任意)。 / Optionally note briefly what was wrong.", False),
        ("・「…」で終わる文は、最後まで書き終わらなかった文。 / Text ending in '…' did not finish.", False),
        ("・行の順番は、ばらばらにしてあります。 / The rows are shuffled on purpose.", False),
        ("", False),
        ("採点の基準 / Criteria (task 6, Table 8)", True),
    ]
    for text, bold in lines:
        info.append([text])
        if bold:
            info.cell(info.max_row, 1).font = Font(bold=True, size=12)
    info.append(["点 / score", "H1 正しい日本語", "H2 自然さ", "H3 意味が分かる", "H4 社内用語"])
    for c in info[info.max_row]:
        c.font = Font(bold=True)
    first_criteria_row = info.max_row
    for crit in _CRITERIA:
        info.append(list(crit))
    last_criteria_row = info.max_row
    info.append([""])
    info.append(["問題の種類の記号 / Problem type codes (task 6, Table 9)"])
    info.cell(info.max_row, 1).font = Font(bold=True, size=12)
    for code, (en, ja) in PROBLEM_TYPES.items():
        info.append([code, f"{ja} / {en}"])
    info.column_dimensions["A"].width = 14
    for col in "BCDE":
        info.column_dimensions[col].width = 34
    for row in info.iter_rows(min_row=first_criteria_row, max_row=last_criteria_row):   # only the table wraps;
        for c in row:                                                                       # other lines run across
            c.alignment = Alignment(wrap_text=True, vertical="top")

    ws = wb.create_sheet("採点 Scoring")
    ws.append(list(_HEADERS))
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="center")
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    grey = PatternFill("solid", fgColor="D9D9D9")
    for r in rows:
        ws.append([r.id, r.prompt, r.text, None, None, None, "-" if r.out_of_scope else None, None, None])
        if r.out_of_scope:
            ws.cell(ws.max_row, 7).fill = grey
    last = ws.max_row
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                        showErrorMessage=True, errorTitle="1〜5", error="1〜5 の整数を入れてください / enter 1-5")
    ws.add_data_validation(dv)
    dv.add(f"D2:F{last}")
    for i, r in enumerate(rows, start=2):
        if not r.out_of_scope:
            dv.add(f"G{i}")
    for col, width in zip("ABCDEFGHI", (7, 18, 52, 11, 11, 11, 11, 13, 30)):
        ws.column_dimensions[col].width = width
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "D2"
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.active = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_scores(path: Path, expected: Sequence[ScoringRow]) -> dict[str, Score]:
    """Read one scorer's workbook. Every problem (missing score, value not 1-5, unknown code, text that differs
    from the sheet that was handed out) is collected, and all of them are reported at once."""
    from openpyxl import load_workbook

    ws = load_workbook(path, data_only=True)["採点 Scoring"]
    want = {r.id: r for r in expected}
    out: dict[str, Score] = {}
    problems: list[str] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        rid = str(row[0]).strip()
        if rid not in want:
            problems.append(f"{rid}: not an ID of the scoring sheet")
            continue
        exp = want[rid]
        if str(row[1] or "") != exp.prompt or str(row[2] or "") != exp.text:
            problems.append(f"{rid}: the beginning or text differs from the sheet that was handed out")
        h: dict[str, int | None] = {}
        for name, value in zip(H_ITEMS, row[3:7]):
            if name == "H4" and exp.out_of_scope:
                h[name] = None
                continue
            try:
                v = int(value) if value is not None and str(value).strip() != "" else None
            except (TypeError, ValueError):
                v = -1
            if v is None:
                problems.append(f"{rid}: {name} is empty")
            elif not 1 <= v <= 5 or (isinstance(value, float) and value != v):
                problems.append(f"{rid}: {name} must be a whole number 1-5 (found {value!r})")
                v = None
            h[name] = v
        codes = {c.upper() for c in re.split(r"[,、\s]+", str(row[7] or "")) if c and c != "-"}
        bad = codes - set(PROBLEM_TYPES)
        if bad:
            problems.append(f"{rid}: unknown problem type code(s) {sorted(bad)} (use {', '.join(PROBLEM_TYPES)})")
        out[rid] = Score(h, codes & set(PROBLEM_TYPES), str(row[8] or "").strip())
    missing = sorted(set(want) - set(out))
    if missing:
        problems.append(f"rows missing: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    if problems:
        raise ScoreFileError(f"{path.name}: {len(problems)} problem(s)\n  " + "\n  ".join(problems[:40])
                             + ("\n  ..." if len(problems) > 40 else ""))
    return out


def mean(values: Iterable[float]) -> float:
    v = list(values)
    return sum(v) / len(v) if v else math.nan
