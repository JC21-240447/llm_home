"""Model A question generator (Phase 1 data), with the meaning of every question saved for later use.

Inputs (same folder):
  用語集.csv, 同義語辞書.csv, DB.xlsx   the team's glossary, synonym dictionary and employee table
  src_*.txt                             pattern sources (#T.. header lines, then 「文型|列グループ|列グループ2」)
Outputs:
  patterns.csv        every pattern with its question type and how many distinct questions it can make
  question_types.csv  totals per question type
  columns.csv         rules per numeric column
  samples.csv         5 example questions per pattern, with their meaning, for team review
  questions.csv       (only with --corpus N) up to N questions per pattern, balanced, with meanings
Usage:
  python make_questions.py                  # patterns, counts and samples
  python make_questions.py --corpus 2000    # also the balanced question corpus
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import random
import re
from dataclasses import dataclass, field

import pandas as pd

GLOSSARY, SYNONYMS, DB = "用語集.csv", "同義語辞書.csv", "DB.xlsx"

# ======================================================================= columns
@dataclass
class Col:
    name: str
    kind: str
    words: list[str]
    vals: list[str]
    big: str = ""
    small: str = ""
    ask: str = ""
    aggs: list[str] = field(default_factory=list)
    avg_words: list[str] = field(default_factory=list)
    ops: tuple = ("以上", "以下", "未満", "超")

MONEY = ["平均", "合計", "最高額", "最低額"]
NUM = ["平均", "最大値", "最小値"]
COLS = [
    Col("年収", "金額", ["年収", "年俸", "年間給与", "年間収入"], ["400万円", "500万円", "600万円", "700万円", "800万円"], "高い", "低い", "いくら", MONEY),
    Col("月収", "金額", ["月収", "月給", "月額給与"], ["30万円", "35万円", "40万円", "45万円", "50万円"], "高い", "低い", "いくら", MONEY),
    Col("基本給", "金額", ["基本給", "本給", "基本賃金"], ["20万円", "25万円", "30万円", "35万円", "40万円"], "高い", "低い", "いくら", MONEY),
    Col("役職手当", "金額", ["役職手当", "管理職手当"], ["1万円", "3万円", "5万円", "10万円"], "高い", "低い", "いくら", MONEY),
    Col("住宅手当", "金額", ["住宅手当", "住居手当", "家賃補助"], ["1万円", "2万円", "3万円"], "高い", "低い", "いくら", MONEY),
    Col("通勤手当", "金額", ["通勤手当", "通勤費"], ["5000円", "1万円", "1万5000円"], "高い", "低い", "いくら", MONEY),
    Col("残業手当", "金額", ["残業手当", "残業代", "時間外手当"], ["2万円", "5万円", "10万円"], "高い", "低い", "いくら", MONEY),
    Col("賞与", "金額", ["賞与", "ボーナス"], ["50万円", "100万円", "150万円"], "高い", "低い", "いくら", MONEY),
    Col("昇給", "金額", ["昇給", "昇給額"], ["3000円", "5000円", "1万円"], "高い", "低い", "いくら", MONEY),
    Col("年齢", "数値", ["年齢"], ["25歳", "30歳", "40歳", "50歳", "60歳"], "高い", "低い", "何歳", NUM),
    Col("勤続年数", "数値", ["勤続年数", "勤務年数", "在籍年数", "社歴"], ["3年", "5年", "10年", "20年", "30年"], "長い", "短い", "何年", NUM, ["勤続年数", "勤務年数", "在籍年数"]),
    Col("月平均残業時間", "数値", ["月平均残業時間", "平均残業時間", "残業時間"], ["10時間", "20時間", "30時間", "40時間"], "多い", "少ない", "何時間", NUM, ["残業時間"]),
    Col("保有資格数", "数値", ["保有資格数", "資格数", "資格の数"], ["1つ", "2つ", "3つ", "5つ"], "多い", "少ない", "いくつ", NUM, ["保有資格数", "資格数"], ("以上", "以下", "未満")),
    Col("有給取得率", "数値", ["有給取得率", "有休取得率", "有給消化率"], ["30%", "50%", "70%", "90%"], "高い", "低い", "何パーセント", NUM),
    Col("直近総合評価", "評価", ["直近総合評価", "総合評価", "評価"], ["2", "3", "4", "5"], "高い", "低い", "いくつ", NUM, ["総合評価", "評価"]),
    Col("直近健康診断結果", "評価", ["直近健康診断結果", "健康診断結果", "健診結果"], ["1", "2", "3"], "", "", "いくつ"),
    Col("直近メンタル診断結果", "評価", ["直近メンタル診断結果", "メンタル診断結果"], ["1", "2", "3"], "", "", "いくつ"),
    # values calculated from two or more columns
    Col("手当合計", "計算", ["手当の合計", "手当合計", "手当の総額"], ["3万円", "5万円", "10万円"], "高い", "低い", "いくら", ["平均"]),
    Col("賞与を除いた年収", "計算", ["賞与を除いた年収"], ["400万円", "500万円", "600万円"], "高い", "低い", "いくら", ["平均"]),
    Col("賞与の年収比", "比率", ["賞与が年収に占める割合"], ["10%", "15%", "20%"], "高い", "低い", "何パーセント", ["平均"]),
    Col("残業手当の月収比", "比率", ["残業手当が月収に占める割合"], ["5%", "10%", "20%"], "高い", "低い", "何パーセント", ["平均"]),
    Col("基本給の月収比", "比率", ["基本給が月収に占める割合"], ["70%", "80%", "90%"], "高い", "低い", "何パーセント", ["平均"]),
]
BASE = [c for c in COLS if c.kind in ("金額", "数値", "評価")]
GROUPS = {
    "金額": [c for c in BASE if c.kind == "金額"],
    "大小": [c for c in BASE if c.big],
    "平均": [c for c in BASE if c.aggs],
    "比較": [c for c in BASE if c.kind in ("金額", "数値")],
    "評価": [c for c in BASE if c.kind == "評価"],
    "計算": [c for c in COLS if c.kind == "計算"],
    "比率": [c for c in COLS if c.kind == "比率"],
}

def to_number(v: str) -> float:
    s = v.replace("%", "").replace("円", "").replace("歳", "").replace("年", "").replace("時間", "").replace("つ", "")
    total, m = 0, re.match(r"(?:(\d+)万)?(\d+)?$", s)
    if m and (m.group(1) or m.group(2)):
        total = int(m.group(1) or 0) * 10000 + int(m.group(2) or 0)
        return total
    return float(s)

# ======================================================================= words from the team's files
gl = list(csv.reader(open(GLOSSARY, encoding="utf-8-sig")))[1:]
syn: dict[str, list[str]] = {}
for r in list(csv.reader(open(SYNONYMS, encoding="utf-8-sig")))[1:]:
    syn.setdefault(r[2], []).append(r[3])
db = pd.read_excel(DB)

BAD_END = ("課", "チーム", "グループ", "組織", "担当部署", "本部", "拠点", "支店", "部署", "者", "社員")
def value_words(term: str) -> list[str]:
    """Standard value + synonyms that read naturally in 「{X}の社員」."""
    out = [term]
    for v in syn.get(term, []):
        if "の" not in v and not v.endswith(BAD_END) and not re.search(r"[たるい]$", v):
            out.append(v)
    return out

DEPTS = sorted(db["部署"].unique())
RANKS = ["一般", "主任", "係長", "課長", "次長", "部長"]
PLACES = sorted(db["勤務地"].unique())
DEPT_W = {d: value_words(d) for d in DEPTS}
PLACE_W = {p: value_words(p) for p in PLACES}
NAMES = db["氏名"].tolist()
IDS = [str(i) for i in db["社員番号"]]
SURNAMES = sorted({next((s for s in ("佐々木", "原", "林", "森") if n.startswith(s)), n[:2]) for n in NAMES})
REASONS = {"私傷病": ["私傷病", "病気", "傷病"], "育児": ["育児", "子育て"], "産前産後": ["産前産後", "出産"], "介護": ["介護"], "労災": ["労災", "労働災害"]}
MARRIAGE = {"既婚": ["既婚", "配偶者あり"], "未婚": ["未婚", "独身", "配偶者なし"]}
MISSING = {"直近総合評価": ["直近総合評価", "総合評価"], "昇給": ["昇給"], "最終昇給年月日": ["最終昇給年月日", "最終昇給日"],
           "直近健康診断結果": ["直近健康診断結果", "健康診断結果"], "直近メンタル診断結果": ["直近メンタル診断結果"]}
DISPLAY = {"氏名": ["氏名", "名前"], "部署": ["部署"], "役職": ["役職"], "勤務地": ["勤務地"], "性別": ["性別"], "年齢": ["年齢"],
           "入社年月日": ["入社年月日", "入社日"], "生年月日": ["生年月日"], "勤続年数": ["勤続年数"], "年収": ["年収"],
           "月収": ["月収"], "基本給": ["基本給"], "賞与": ["賞与", "ボーナス"], "残業手当": ["残業手当"],
           "月平均残業時間": ["残業時間"], "保有資格数": ["資格数"], "有給取得率": ["有給取得率"], "直近総合評価": ["総合評価"]}

O = tuple  # option = (surface, meta)
def opts(pairs) -> list[O]:
    return [(s, m) for s, m in pairs]

def attr_options(kinds: list[str], person: bool = False) -> list[O]:
    out = []
    for k in kinds:
        if k == "部署":
            out += [(w, {"条件": "部署", "条件値": d}) for d, ws in DEPT_W.items() for w in ws]
        elif k == "勤務地":
            out += [(w, {"条件": "勤務地", "条件値": p}) for p, ws in PLACE_W.items() for w in ws]
        elif k == "役職":
            for r in RANKS:
                if person and r == "一般":
                    continue
                if not person:
                    out.append((f"役職が{r}", {"条件": "役職", "条件値": r}))
                if r != "一般":
                    out.append((r, {"条件": "役職", "条件値": r}))
        elif k == "性別":
            for g, w in (("男", "男性"), ("女", "女性")):
                out.append((w, {"条件": "性別", "条件値": g}))
                if not person:
                    out.append((f"性別が{g}", {"条件": "性別", "条件値": g}))
    return out

def pair_options(word_map: dict[str, list[str]], key: str) -> list[O]:
    return [((a, b), {key: x, key + "2": y}) for x, y in itertools.permutations(word_map, 2) for a in word_map[x] for b in word_map[y]]

SLOT_DOMAINS: dict[str, list[O]] = {
    "{属性}": attr_options(["部署", "役職", "勤務地", "性別"]),
    "{所属}": attr_options(["部署", "勤務地"]),
    "{人の属性}": attr_options(["性別", "役職"], person=True),
    "{性別語}": [("男性", {"性別": "男"}), ("女性", {"性別": "女"})],
    "{役職語}": [(r, {"役職": r}) for r in RANKS if r != "一般"],
    "{グループ}": [(w, {"グループ": c}) for c, ws in {"部署": ["部署", "部門"], "役職": ["役職"], "勤務地": ["勤務地", "勤務先"], "性別": ["性別", "男女"]}.items() for w in ws],
    "{グループ所}": [(w, {"グループ": c}) for c, ws in {"部署": ["部署", "部門"], "勤務地": ["勤務地", "勤務先"]}.items() for w in ws],
    "{グループ人}": [("役職", {"グループ": "役職"}), ("性別", {"グループ": "性別"})],
    "{区分列}": [(w, {"列": c}) for c, ws in {"部署": ["部署", "部門"], "役職": ["役職"], "勤務地": ["勤務地", "勤務先"]}.items() for w in ws],
    "{理由}": [(w, {"休職事由": k}) for k, ws in REASONS.items() for w in ws],
    "{休職事由}": [(k, {"休職事由": k}) for k in list(REASONS) + ["その他"]],
    "{婚姻}": [(w, {"配偶者": k}) for k, ws in MARRIAGE.items() for w in ws],
    "{欠損列}": [(w, {"列": k}) for k, ws in MISSING.items() for w in ws],
    "{氏名}": [(n, {"氏名": n}) for n in NAMES],
    "{姓}": [(s, {"名字": s}) for s in SURNAMES],
    "{社員番号}": [(i, {"社員番号": int(i)}) for i in IDS],
    "{年}": [(str(y), {"年": y}) for y in range(1990, 2026)],
    "{年2}": [(str(y), {"年": y}) for y in (2021, 2022, 2023, 2024, 2025)],
    "{月}": [(str(m), {"月": m}) for m in range(1, 13)],
    "{年代}": [(f"{a}代", {"年代": a}) for a in (20, 30, 40, 50, 60)],
    "{N}": [(n, {"人数": int(n)}) for n in ("3", "5", "10")],
    "{人数}": [(f"{n}人", {"人数": n}) for n in (2, 3, 5, 10)],
    "{経過}": [(f"{n}年", {"経過年数": n}) for n in (1, 2, 3, 5)],
}
PAIR_DOMAINS = {
    ("{部署}", "{部署2}"): pair_options(DEPT_W, "部署"),
    ("{勤務地}", "{勤務地2}"): pair_options(PLACE_W, "勤務地"),
    ("{役職}", "{役職2}"): pair_options({r: [r] for r in RANKS}, "役職"),
    ("{氏名}", "{氏名2}"): pair_options({n: [n] for n in NAMES}, "氏名"),
    ("{表示列}", "{表示列2}"): pair_options(DISPLAY, "表示列"),
    ("{年}", "{年3}"): [((str(a), str(b)), {"年から": a, "年まで": b}) for a in range(1990, 2026) for b in range(a + 1, min(a + 11, 2027))],
}

# ======================================================================= column-bound options
def col_options(col: Col, pat: str, second: bool = False) -> list[O]:
    sfx = "2" if second else ""
    out = []
    avg_ctx_pat = "平均" in pat or "中央値" in pat
    aggs = col.aggs if "{集計}" in pat else [""]
    for agg in aggs:
        words = col.avg_words if ((avg_ctx_pat or agg == "平均") and col.avg_words) else col.words
        for w in words:
            base = {"{列" + sfx + "}": w, "{大" + sfx + "}": col.big, "{小" + sfx + "}": col.small, "{疑問詞}": col.ask}
            meta = {"列" + sfx: col.name}
            if agg:
                base["{集計}"] = agg; meta["集計"] = agg
            variants = [(base, meta)]
            if "{値" + sfx + "}" in pat:
                variants = [({**b, "{値" + sfx + "}": v, "{比較" + sfx + "}": c}, {**m, "値" + sfx: to_number(v), "比較" + sfx: c})
                            for b, m in variants for v in col.vals for c in col.ops]
            if "{下}" in pat and not second:
                variants = [({**b, "{下}": col.vals[i], "{上}": col.vals[j]}, {**m, "下限": to_number(col.vals[i]), "上限": to_number(col.vals[j])})
                            for b, m in variants for i, j in itertools.combinations(range(len(col.vals)), 2)]
            if "{評価列}" in pat:
                vals = col.vals[1:-1] if ("以上" in pat or "以下" in pat) else col.vals
                variants = [({**b, "{評価列}": w, "{評価値}": v}, {**m, "値": int(v)}) for b, m in variants for v in vals]
            out += variants
    return out

# ======================================================================= patterns
@dataclass
class Pattern:
    pid: str
    tid: str
    tname: str
    text: str
    group: str
    group2: str

    @property
    def style(self) -> str:
        t = self.text
        if t.endswith("？") and len(t) <= 22 and "ください" not in t:
            return "短い" if not re.search(r"(る|た|の)？$", t) else "くだけた"
        if re.search(r"(ください|ですか|ますか|です)。$", t):
            return "丁寧"
        return "くだけた"

def load_patterns() -> list[Pattern]:
    out, tid, tname, n = [], "", "", 0
    for f in sorted(glob.glob("src_*.txt")):
        for line in open(f, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                tid, tname = line[1:].split("|")
                continue
            text, g1, g2 = (line.split("|") + ["", ""])[:3]
            n += 1
            out.append(Pattern(f"P{n:03d}", tid, tname, text, g1, g2))
    return out

def factors(p: Pattern) -> tuple[list[list], bool]:
    """Independent choice lists for a pattern; bool = two columns that must differ."""
    fs: list[list] = []
    if p.group:
        fs.append([(c, o) for c in GROUPS[p.group] for o in col_options(c, p.text)])
    if p.group2:
        fs.append([(c, o) for c in GROUPS[p.group2] for o in col_options(c, p.text, second=True)])
    for (a, b), dom in PAIR_DOMAINS.items():
        if a in p.text and b in p.text:
            dom2 = dom
            if a == "{表示列}" or "{表示列}" in p.text and a == "{表示列}":
                pass
            if a == "{表示列}" and "{氏名}" in p.text:
                dom2 = [o for o in dom if "氏名" not in (o[1]["表示列"], o[1]["表示列2"])]
            fs.append([(None, ({a: x, b: y}, m)) for (x, y), m in dom2])
    used_pairs = {s for (a, b) in PAIR_DOMAINS for s in (a, b) if a in p.text and b in p.text}
    for slot, dom in SLOT_DOMAINS.items():
        if slot in p.text and slot not in used_pairs:
            fs.append([(None, ({slot: s}, m)) for s, m in dom])
    if "{表示列}" in p.text and "{表示列}" not in used_pairs:
        excl = {"氏名"} if ("氏名" in p.text or "{氏名}" in p.text) else set()
        fs.append([(None, ({"{表示列}": w}, {"表示列": c})) for c, ws in DISPLAY.items() if c not in excl for w in ws])
    return fs, bool(p.group and p.group2)

def render(p: Pattern, combo) -> tuple[str, dict] | None:
    cols = [c for c, _ in combo if c is not None]
    if len(cols) == 2 and cols[0].name == cols[1].name:
        return None
    text, meta = p.text, {"パターン": p.pid, "質問タイプ": p.tid}
    for _, (subs, m) in combo:
        for k, v in subs.items():
            text = text.replace(k, v)
        meta.update(m)
    # display columns must not repeat a word already in the sentence
    if "表示列" in meta and "表示列2" not in meta and ("氏名" in p.text or "{氏名}" in p.text) and meta["表示列"] == "氏名":
        return None
    text = text.replace("性別別", "男女別").replace("男女ごと", "性別ごと")
    text = re.sub(r"(?<!が)(主任|係長|課長|次長|部長)の社員(?!数)", r"\1", text)   # 課長の社員 -> 課長
    text = re.sub(r"性別が(男|女)(ではない|以外)", r"\1性\2", text)            # 性別が女ではない -> 女性ではない
    if "{" in text or "のの" in text or "社員の社員" in text or "平均平均" in text:
        raise ValueError(f"bad output {p.pid}: {text}")
    return text, meta

def count(p: Pattern) -> int:
    fs, two = factors(p)
    total = 1
    for f in fs:
        total *= len(f)
    if two:  # remove combinations where both column slots use the same column
        from collections import Counter
        c1, c2 = Counter(c.name for c, _ in fs[0]), Counter(c.name for c, _ in fs[1])
        same = sum(c1[k] * c2[k] for k in c1)
        total = total // (len(fs[0]) * len(fs[1])) * (len(fs[0]) * len(fs[1]) - same)
    return total

def generate(p: Pattern, limit: int, rng: random.Random) -> list[tuple[str, dict]]:
    fs, _ = factors(p)
    n = count(p)
    seen: dict[str, dict] = {}
    if n <= limit * 3 and n <= 200000:
        for combo in itertools.product(*fs) if fs else [()]:
            r = render(p, combo)
            if r and r[0] not in seen:
                seen[r[0]] = r[1]
        items = list(seen.items())
        rng.shuffle(items)
        return items[:limit]
    tries = 0
    while len(seen) < limit and tries < limit * 20:
        tries += 1
        r = render(p, [rng.choice(f) for f in fs])
        if r and r[0] not in seen:
            seen[r[0]] = r[1]
    return list(seen.items())

# ======================================================================= main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=0, help="max questions per pattern for questions.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pats = load_patterns()
    counts = {p.pid: count(p) for p in pats}

    with open("patterns.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["パターンID", "質問タイプID", "質問タイプ", "文体", "文型", "列グループ", "列グループ2", "作れる質問の数"])
        for p in pats:
            w.writerow([p.pid, p.tid, p.tname, p.style, p.text, p.group, p.group2, counts[p.pid]])
    types: dict[str, list] = {}
    for p in pats:
        t = types.setdefault(p.tid, [p.tname, 0, 0])
        t[1] += 1; t[2] += counts[p.pid]
    with open("question_types.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["質問タイプID", "質問タイプ", "パターン数", "作れる質問の数"])
        for k, (n, c, q) in types.items():
            w.writerow([k, n, c, q])
    with open("columns.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["列名", "種類", "質問で使う言い方", "平均と一緒に使う言い方", "比較に使う値", "大きいとき", "小さいとき", "値を聞く疑問詞", "使える集計"])
        for c in COLS:
            w.writerow([c.name, c.kind, "、".join(c.words), "、".join(c.avg_words or c.words), "、".join(c.vals), c.big, c.small, c.ask, "、".join(c.aggs)])
    with open("samples.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["パターンID", "質問タイプID", "質問タイプ", "文型", "生成された質問", "意味"])
        for p in pats:
            for q, m in generate(p, 5, rng):
                w.writerow([p.pid, p.tid, p.tname, p.text, q, json.dumps(m, ensure_ascii=False)])
    if a.corpus:
        total = 0
        with open("questions.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(["質問", "パターンID", "質問タイプID", "意味"])
            seen: set[str] = set()
            for p in pats:
                for q, m in generate(p, a.corpus, rng):
                    if q in seen:      # the same sentence can come from two patterns; keep the first
                        continue
                    seen.add(q)
                    w.writerow([q, p.pid, p.tid, json.dumps(m, ensure_ascii=False)]); total += 1
        print("questions.csv:", total, "questions")
    print("patterns:", len(pats), "| types:", len(types), "| all distinct questions:", sum(counts.values()))
    for cap in (500, 1000, 2000, 3000):
        print(f"  max {cap} per pattern -> {sum(min(v, cap) for v in counts.values())}")

if __name__ == "__main__":
    main()
