import inspect

import pyopenjtalk

if "use_sudachi_kanji_yomi" in inspect.signature(pyopenjtalk.g2p).parameters:
    raise ImportError(
        "pyopenjtalk-plus が検出されました。"
        "vanilla pyopenjtalk の評価には専用環境を使用してください。"
    )

from .base import G2PAdapter
from ._util import suppress_fd_stderr


def _njd_to_kana(njd_features: list[dict]) -> str:
    prons = []
    for n in njd_features:
        p = n["string"] if n["pos"] == "記号" else n["pron"]
        prons.append(p.replace("’", ""))
    return "".join(prons)


class PyOpenJTalkAdapter(G2PAdapter):
    def g2p(self, texts: list[str]) -> list[str]:
        with suppress_fd_stderr():
            return [pyopenjtalk.g2p(t, join=True) for t in texts]

    def g2k(self, texts: list[str]) -> list[str]:
        with suppress_fd_stderr():
            return [_njd_to_kana(pyopenjtalk.run_frontend(t)) for t in texts]
