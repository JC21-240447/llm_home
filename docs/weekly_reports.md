# Weekly reports

Paste the Japanese snippet from each Day here, with the evidence file names.
Status words: 予定通り終わった / 予定まで進まなかった / 手が付かなかった

【課題5】文章データ(コーパス)の準備
状況: 予定通り終わった
内容: チームが作った用語集・同義語辞書・文型から作成した大量の質問文を、文型ごとに
      学習用・検証用・テスト用に分けた。同じ文型の文がテストに混ざらないようにし、
      一度も見ていない言い回しで評価できるようにした。データカードを作成した。
証拠: docs/DATA_CARD.md、data/splits/split_manifest.json、run_log.md
次: 課題8(文章を細かく分けるプログラム = トークナイザー)

【課題9】番号にして元に戻す仕組み
状況: 予定通り終わった
内容: トークンを番号に変換し、番号から元の文に戻す仕組みを確認した。学習用・検証用・テスト用の
      すべての質問文と、絵文字・英語・記号を含む文で、完全に元に戻ることを確認した。質問文を
      番号の列に変換し、学習用のファイルとして保存した。文→トークン→番号→文のデモを作成した。
証拠: docs/DAY7_TOKENIZER_DEMO.md、docs/DAY7_ENCODING_REPORT.md、run_log.md、tests/
次: 課題10(次の単語を予測する仕組み)に向けて、モデルを作る(Day 8).
