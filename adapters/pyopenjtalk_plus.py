import inspect

import pyopenjtalk

if "use_sudachi_kanji_yomi" not in inspect.signature(pyopenjtalk.g2p).parameters:
    raise ImportError(
        "vanilla pyopenjtalk が検出されました。"
        "pyopenjtalk-plus の評価には専用環境を使用してください。"
    )

from .base import G2PAdapter
from ._util import suppress_fd_stderr


class PyOpenJTalkPlusAdapter(G2PAdapter):
    def __init__(
        self,
        use_sudachi_kanji_yomi: bool = True,
        use_tsqyomi: bool = True,
        revert_long_vowels: bool = False,
        revert_yotsugana: bool = False,
    ) -> None:
        pyopenjtalk.tsqyomi.load_model()

        self._opts = {
            "join": True,
            "use_sudachi_kanji_yomi": use_sudachi_kanji_yomi,
            "use_tsqyomi": use_tsqyomi,
            "predict_nani": True,
            "revert_long_vowels": revert_long_vowels,
            "revert_yotsugana": revert_yotsugana,
        }

    def g2p(self, texts: list[str]) -> list[str]:
        with suppress_fd_stderr():
            return [pyopenjtalk.g2p(t, kana=False, **self._opts) for t in texts]

    def g2k(self, texts: list[str]) -> list[str]:
        with suppress_fd_stderr():
            return [pyopenjtalk.g2p(t, kana=True, **self._opts) for t in texts]
