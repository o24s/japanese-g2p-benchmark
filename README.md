# Japanese G2P Benchmark

日本語 G2P (Grapheme-to-Phoneme) の性能を評価するためのベンチマークです。
複数のオープンなデータセットを統合し、音素エラー率 (PER) およびカタカナエラー率 (KER) を測定します。

現在、評価している G2P は以下の通りです。
- [Haqumei](https://github.com/o24s/haqumei)
- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)

## ライセンス

本プロジェクトは、プログラムとデータセットで異なるライセンスを適用しています。

- ソースコード: Apache License, Version 2.0 のもとで利用可能です。
- 評価データセット: CC BY-SA 4.0 (表示 - 継承) が適用されます。このデータセットを改変・拡張して配布する場合は、同じく CC BY-SA 4.0 ライセンスで公開する必要があります。

元データの著作権表記や詳細なライセンス条項については、[NOTICE](NOTICE) ファイルをご確認ください。

## 謝辞

本プロジェクトでは、以下のデータセットを利用しています。

- [prj-beatrice/jsut-label](https://github.com/prj-beatrice/jsut-label) (Wikipedia: CC-BY-SA 3.0 / TANAKA corpus: CC-BY 2.0 / JSUT: CC-BY-SA 4.0)
- [JVS-nonpara-kana dataset](https://github.com/CyberAgentAILab/jvs_nonpara_kana) (CC BY-SA 4.0) - © 2026 Tomoki Koriyama / JVS corpus by Shinnosuke Takamichi et al.
- [Joyo Kanji Yomi Benchmark](https://huggingface.co/datasets/sbintuitions/joyo-kanji-yomi-benchmark) (MIT) - © 2026 SB Intuitions
- [AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench) (CC-BY-SA 3.0)
- [日本語Wikipedia入力誤りデータセット (v2)](https://nlp.ist.i.kyoto-u.ac.jp/?日本語Wikipedia入力誤りデータセット) (CC-BY-SA 3.0)
- [ROHAN](https://github.com/mmorise/rohan4600) (CC0 1.0 Universal)
