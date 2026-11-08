# Day 6.2 — test_bpe_tokenizer.py — やさしい1行ずつの解説(日本語)

> **Day 6 の3つのファイルの関係**
> - **Day 6.1 `bpe_tokenizer.py`** — *エンジン*。
> - **Day 6 `day6_train_bpe.py`** — *運転手*。
> - **Day 6.2(このファイル)`test_bpe_tokenizer.py`** — *検査員*:エンジン(Day 6.1)が正しいことを確かめる自動テスト。
>
> どのテストもDay 6.1の関数を呼ぶので、先に **Day 6.1** を読んでおくと分かりやすいです。

---

## 0. 全体像(まずここから)

**テストファイル**は、機械のための**自動チェックリスト**のようなものです。人が手で確認する(「ちゃんと復号できる?」)代わりに、約束を1つずつ確かめる小さな関数を書いておきます。いつでも(変更するたびに)実行すれば、どの約束がまだ守られているか、コンピューターが **passed(合格)** か **failed(不合格)** で教えてくれます。

このファイルは、人気のPythonのテスト道具 **pytest** を使います。ルールは簡単です:
- 名前が `test_` で始まる関数が、テスト。
- 中の `assert` の行は「これは真でなければならない」の意味。偽なら、そのテストは**失敗**する。

確かめている約束:

| 約束 | テスト |
|---|---|
| IDの番号は絶対に変わらない(0〜3 特殊、4〜103 予約、104〜359 バイト、360〜 結合) | `test_fixed_id_layout` |
| 文を断片に切っても何も失われない | `test_pretokenize_is_lossless` |
| 切り方が、例のとおりになる | `test_pretokenize_examples` |
| 学習が毎回同じ結果になる | `test_training_is_deterministic` |
| どんな文でも `decode(encode(text)) == text` | `test_round_trip` |
| 結合が実際に文を短くし、単語をまるごとトークンにする | `test_merges_shorten_known_text` |
| `<bos>` / `<eos>` / `<pad>` が正しく動く | `test_bos_eos_and_skip_special` |
| 小さいトークナイザーは、大きいものの先頭部分とまったく同じ | `test_truncated_is_prefix` |
| 保存と読み込みで何も変わらない | `test_save_and_load_identical` |
| **本物の**トークナイザーが、**本物の**検証用の文で往復できる | `test_real_tokenizer_round_trip_on_validation` |

実行方法:

```
.venv\Scripts\python.exe -m pytest tests -q
```
(`-m pytest` は venv のPythonでpytestを動かす。`tests` はフォルダー。`-q` = 静かに、短い出力。)

---

## 1. 先に知っておきたい用語(ミニ辞書)

| 用語 | かんたんな意味 |
|---|---|
| **単体テスト(unit test)** | コードの小さな部分1つを、自動で確かめる小さなチェック。 |
| **pytest** | テスト関数を見つけて実行し、結果を報告する道具。 |
| **assert** | 「これは真でなければならない。そうでなければ失敗」。 |
| **フィクスチャ(fixture)** | 何か(ここでは、小さく学習したトークナイザー)を一度だけ用意して、求めるテストに渡す補助。 |
| **`scope="module"`** | フィクスチャは、テストごとではなく、**ファイルごとに1回**作られる(速い)。 |
| **パラメータ化(parametrize)** | 同じテストを、違う入力で何回も実行する。 |
| **`tmp_path`** | pytest標準のフィクスチャ:新しい一時フォルダーで、自動で片付けられる。 |
| **スキップ(skip)** | pytestに「今はこのテストを実行しない」と伝える(失敗にはならない)。 |
| **ロスレス / 往復 / 決定的** | Day 6.1を参照。 |

---

## 2. docstring(説明文)

```python
"""Unit tests for src/bpe_tokenizer.py (Day 6).
src/bpe_tokenizer.py の単体テスト(Day 6)

Run (VS Code terminal, repo root):
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
```
- 英語と日本語の短い説明と、実行コマンド。
- このdocstringは raw文字列では**ない**(前に `r` が無い)ので、バックスラッシュは**2つ重ねて**(`\\`)書く:普通の文字列では、`\\` が本物のバックスラッシュ1つの意味になる。(これまでのファイルは `r"""` を使っていた。)

---

## 3. importと準備

```python
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
```
- `from __future__ ...`、`sys`、`Counter`、`Path` — 前と同じ。新しいのは `pytest`(フィクスチャ、パラメータ化、スキップに使うテスト道具)。

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- Day 6と同じ考え方:このファイルの場所からリポジトリのルートを求め(このファイルは `tests/` にあるので、`.parent.parent` がルート)、Pythonのimportの探索パスに足して、`from src.bpe_tokenizer import ...` が動くようにする。

```python
from src.bpe_tokenizer import (  # noqa: E402
    BOS_ID, BYTE_OFFSET, EOS_ID, FIRST_MERGE_ID, NUM_RESERVED, PAD_ID, RESERVED_FIRST_ID, UNK_ID,
    ByteBPETokenizer, count_pieces, pretokenize, train_bpe,
)
```
- テストに必要なものを、エンジンから全部読み込む:IDの定数、トークナイザーのクラス、学習の補助関数。(`noqa: E402` はDay 6と同じ。)

---

## 4. テスト用のデータ

```python
TINY_CORPUS = [
    "営業部の社員の平均年収を教えてください。",
    "開発部の社員の平均月収を教えてください。",
    "営業部で年収が600万円以上の人は何人ですか。",
    "大阪支社の社員を一覧で見せて。",
] * 5
```
- ごく小さな、見立ての学習用データ:現実的な文が4つ。`* 5` でリストを5回繰り返して(20行)、組が結合されるのに**十分な回数**(2回以上)出るようにする。

```python
TRICKY_TEXTS = [
    "",
    "今日は晴れ",
    "ABC abc 123",
    "絵文字😀と記号①②",
    "\t改行\nを含む\r\n",
    "<bos>や<unused_5>は普通の文字",
    "ﾊﾝｶｸｶﾅとＺＥＮＫＡＫＵ",
]
```
- トークナイザーを壊そうとする、意地悪な文:
  1. **空文字**;
  2. 単純な日本語の句;
  3. 英字と数字;
  4. **絵文字**と丸数字(まれな文字。絵文字は4バイト);
  5. **タブ、改行、キャリッジリターン**の文字;
  6. 特殊トークンに**見える**(`<bos>`、`<unused_5>`)が、普通の文字として扱われなければならない文;
  7. **半角カタカナ**と**全角のラテン文字**(Unicodeの別の形。正規化はしないので、変わらずに残らなければならない)。

---

## 5. フィクスチャ

```python
@pytest.fixture(scope="module")
def tiny_tokenizer() -> ByteBPETokenizer:
    merges, _ = train_bpe(count_pieces(TINY_CORPUS), max_merges=10_000, min_frequency=2)
    return ByteBPETokenizer(merges)
```
- `@pytest.fixture` は、これが、テストが引数に `tiny_tokenizer` と書いたときにpytestが呼ぶ補助だと示す。
- `scope="module"` → ファイル全体で**1回だけ**作って共有する(学習には少し時間がかかるので、繰り返さない)。
- `train_bpe(...)` は `(merges, log)` を返す。`merges, _ = ...` は結合を残し、ログを捨てる(`_` =「要らない」)。
- その結合から作ったトークナイザーを返す。つまり、以下のテストはすべて、小さなコーパスで学習したトークナイザーを使う。

---

## 6. テスト

### 6.1 `test_fixed_id_layout`

```python
def test_fixed_id_layout() -> None:
    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)
    assert RESERVED_FIRST_ID == 4 and NUM_RESERVED == 100
    assert BYTE_OFFSET == 104 and FIRST_MERGE_ID == 360
    tok = ByteBPETokenizer([])
    assert tok.vocab_size == 360
    assert tok.token_bytes[4] == b"<unused_0>" and tok.token_bytes[103] == b"<unused_99>"
    assert tok.token_bytes[BYTE_OFFSET] == b"\x00" and tok.token_bytes[359] == b"\xff"
```
- **IDの地図が変わらないこと**を確かめる — モデルは後でこの番号に頼るので、うっかりした変更は見つけなければならない。
- 1行目:4つの特殊IDは 0、1、2、3。
- 2行目:予約のブロックは4から始まり、100個ある。
- 3行目:バイトは104から、結合は360から始まる。
- `tok = ByteBPETokenizer([])` — **結合が1つもない**トークナイザー。その `vocab_size` は、ちょうど360(特殊 + 予約 + 256バイト)でなければならない。
- その `token_bytes` リストの添字4には `b"<unused_0>"`、103には `b"<unused_99>"`(予約の最初と最後)がなければならない。
- そして、バイトのブロックは、添字104の `b"\x00"`(バイト0)から、添字359の `b"\xff"`(バイト255)まで続かなければならない。(`b"..."` はバイト列のリテラルで、`\x00` はバイト0の16進数表記。)

### 6.2 `test_pretokenize_is_lossless`

```python
@pytest.mark.parametrize("text", TRICKY_TEXTS + TINY_CORPUS[:4])
def test_pretokenize_is_lossless(text: str) -> None:
    assert "".join(pretokenize(text)) == text
```
- `@pytest.mark.parametrize("text", [...])` は、このテストをリストの**値ごとに1回**実行する。ここでは:意地悪な文7つ + コーパスの最初の4文 = **11回**。
- 断片を `""` でつなぐと、元の文とぴったり同じになるはず。これで、文を断片に切っても**何も失われない**ことを証明する。

### 6.3 `test_pretokenize_examples`

```python
def test_pretokenize_examples() -> None:
    assert pretokenize("営業部の社員は何人ですか。") == ["営業部", "の", "社員", "は", "何人", "ですか", "。"]
    assert pretokenize("600万円") == ["6", "0", "0", "万円"]  # digits are always single pieces
    assert pretokenize("ボーナス、") == ["ボーナス", "、"]
```
- 3つの厳密な期待:
  1. docstringの例:漢字・ひらがな・句読点の連続が、文字の種類が変わるところで切られる。
  2. **数字はいつも単独の断片**:`600` → `6`、`0`、`0`。
  3. 句読点 `、` はカタカナの単語にくっつかない。

### 6.4 `test_training_is_deterministic`

```python
def test_training_is_deterministic() -> None:
    counts = count_pieces(TINY_CORPUS)
    assert train_bpe(counts, 10_000, 2)[0] == train_bpe(Counter(dict(reversed(list(counts.items())))), 10_000, 2)[0]
```
- 同じデータを学習すれば、入力の順番が違っても、**同じ結合**になるはず。
- `counts` — 断片の数。
- 左側:普通に学習する。`[0]` は結果の最初の項目(結合)を取る。
- 右側:**同じ項目を逆の順番**にしたカウンターを作り(`list(counts.items())` → `reversed(...)` → `dict(...)` → `Counter(...)`)、それで学習する。
- 2つは等しくなければならない。これは、`train_bpe` が断片を並べ替え、同点をIDの小さいほうで決めるから動く(Day 6.1)。これが、**学校PCと自宅ノートPCが同一のトークナイザーを作れる**理由。

### 6.5 `test_round_trip`

```python
@pytest.mark.parametrize("text", TRICKY_TEXTS + TINY_CORPUS[:4])
def test_round_trip(tiny_tokenizer: ByteBPETokenizer, text: str) -> None:
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text
```
- 同じ11個の文。今回は、学習したトークナイザーを使う(フィクスチャは、引数に名前を書くことで渡される)。
- `decode(encode(text))` は元と等しくなければならない。これが**ロスレスの約束** — 絵文字、タブ、特殊トークンに見える文も含めて。

### 6.6 `test_merges_shorten_known_text`

```python
def test_merges_shorten_known_text(tiny_tokenizer: ByteBPETokenizer) -> None:
    text = TINY_CORPUS[0]
    assert len(tiny_tokenizer.encode(text)) < len(text.encode("utf-8"))
    assert "営業部" in tiny_tokenizer.pieces(text)
```
- 学習が実際に**役に立つ**ことを確かめる。
- トークン数は、**UTF-8のバイト数より少ない**はず(結合がなければ、ちょうど1バイト1トークン)。
- そして、単語 `営業部` が `pieces(text)` の中に**1つのまるごとのトークン**として現れるはず(小さなコーパスの中で何度も出てくるので、学習が完全に結合しているはず)。

### 6.7 `test_bos_eos_and_skip_special`

```python
def test_bos_eos_and_skip_special(tiny_tokenizer: ByteBPETokenizer) -> None:
    ids = tiny_tokenizer.encode("社員", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert tiny_tokenizer.decode(ids) == "社員"
    assert tiny_tokenizer.decode([PAD_ID, 4, *tiny_tokenizer.encode("社員")]) == "社員"
```
- `add_bos` と `add_eos` を付けると、最初のIDは `<bos>`、最後は `<eos>` のはず(`ids[-1]` = 最後の項目)。
- そのリストを復号すると、`社員` だけが返るはず — 特殊IDは**飛ばされる**。
- 3つ目の確認:本物のトークンの前に、`<pad>`(ID 0)と最初の予約ID(4)を置く。`[a, b, *リスト]` は、新しいリストの中でリストを**展開する**。復号は、それでも `社員` を返すはず — 埋め草と予約のIDは無視される。

### 6.8 `test_truncated_is_prefix`

```python
def test_truncated_is_prefix(tiny_tokenizer: ByteBPETokenizer) -> None:
    small = tiny_tokenizer.truncated(FIRST_MERGE_ID + 5)
    assert small.vocab_size == FIRST_MERGE_ID + 5
    assert small.merges == tiny_tokenizer.merges[:5]
    assert small.decode(small.encode(TINY_CORPUS[0])) == TINY_CORPUS[0]
```
- ちょうど5個の結合の分の余地を持つトークナイザー(語彙サイズ365)を求める。
- その大きさは365で、結合は大きいトークナイザーの結合の**最初の5個そのもの**(Day 6が頼っている「先頭部分」の考え方)で、文を正しく往復できるはず。

### 6.9 `test_save_and_load_identical`

```python
def test_save_and_load_identical(tiny_tokenizer: ByteBPETokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tiny_tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.merges == tiny_tokenizer.merges
    for text in TRICKY_TEXTS + TINY_CORPUS[:4]:
        assert loaded.encode(text) == tiny_tokenizer.encode(text)
```
- `tmp_path` は、pytestの**標準の**フィクスチャ:新しい一時フォルダーで、後で削除される。だからこのテストが、本物のファイルに触れることはない。
- トークナイザーを一時ファイルに保存し、読み込み直して確認する:結合が同じで、11個の文すべてで**符号化が同一**。

### 6.10 `test_real_tokenizer_round_trip_on_validation`

```python
def test_real_tokenizer_round_trip_on_validation() -> None:
    """Runs only after Day 6 has produced data/tokenizer/tokenizer.json and Day 5 the validation file."""
    tok_path = REPO_ROOT / "data/tokenizer/tokenizer.json"
    val_path = REPO_ROOT / "data/corpus/val.txt"
    if not (tok_path.exists() and val_path.exists()):
        pytest.skip("run Day 5 and Day 6 first")
    tok = ByteBPETokenizer.load(tok_path)
    lines = [ln for ln in val_path.read_text(encoding="utf-8").split("\n") if ln]
    assert all(tok.decode(tok.encode(ln)) == ln for ln in lines)
```
- 他と違い、このテストは、Day 5とDay 6が作った**本物の**ファイルを使う。
- どちらかのファイルが無ければ → `pytest.skip(...)`:テストは失敗ではなく**スキップ**として報告される(だから、新しくチェックアウトした状態でもテスト全体が動く)。
- あれば:本物のトークナイザーを読み込み、本物の検証用の行(空でないものだけ)を読み、**すべての行が往復できる**ことを確かめる。`all(...)` は、1行残らず合格したときだけ真。

---

## 7. テストはいくつあるか?

| テスト | 実行回数 |
|---|---|
| `test_fixed_id_layout` | 1 |
| `test_pretokenize_is_lossless` | 11(パラメータ化) |
| `test_pretokenize_examples` | 1 |
| `test_training_is_deterministic` | 1 |
| `test_round_trip` | 11(パラメータ化) |
| `test_merges_shorten_known_text` | 1 |
| `test_bos_eos_and_skip_special` | 1 |
| `test_truncated_is_prefix` | 1 |
| `test_save_and_load_identical` | 1 |
| `test_real_tokenizer_round_trip_on_validation` | 1(スキップされることがある) |
| **合計** | **30** |

## 8. 結果の読み方

- `30 passed` — すべて守られている。
- `29 passed, 1 skipped` — 問題なし:Day 5・6の出力がまだ無いので、本物のファイルのテストがスキップされた。
- `FAILED test_xxx` — 約束が破られている。pytestが `assert` の下に表示する行を読んで、どの値が違ったかを確かめる。
- トークナイザーを変更する**たびに**(そして両方のPCで)テストを実行する。速い。

## 9. このテストが確かめて*いない*こと(正直な限界)

- `load()` のエラー処理(形式の違い、IDの並びの違い、`vocab_size` の違い)は、テストされていない。
- 学習が止まる条件(`min_frequency`、`max_merges`)は、直接はテストされていない。
- **範囲外のID**での `decode` は、テストされていない(`IndexError` になる。Day 6.1の注意を参照)。
- 小さなコーパスのトークナイザーは、代役にすぎない。**本物の**トークナイザーを確かめるのは最後のテストで、それも検証用の行だけ。

必要なら、これらは後から追加のテストとして足せます。

## 10. 一段落でまとめ

このファイルは、トークナイザーのための自動チェックリスト(pytest)です。IDの並びが固定されていること、切り分けと符号化がどんな文(絵文字、タブ、特殊トークンに似た文も)も失わないこと、学習が決定的であること、結合が実際に文を短くすること、特殊IDが正しく動くこと、小さいトークナイザーが大きいものの先頭部分とまったく同じであること、保存と読み込みで何も変わらないこと、そして最後に、本物のトークナイザーが本物の検証用の全行を往復できることを確かめます。
