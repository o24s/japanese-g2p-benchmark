from haqumei import Haqumei, IuPronunciation

from .base import G2PAdapter


class HaqumeiAdapter(G2PAdapter):
    def __init__(
        self,
        normalize_iu: IuPronunciation = IuPronunciation.None_,
        revert_long_vowels: bool = False,
        revert_yotsugana: bool = False,
    ) -> None:
        self._h = Haqumei(
            normalize_iu=normalize_iu,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
        )

    def g2p(self, texts: list[str]) -> list[str]:
        results = self._h.g2p_batch(texts)
        return [" ".join(r) if isinstance(r, list) else r for r in results]

    def g2k(self, texts: list[str]) -> list[str]:
        return self._h.g2k_batch(texts)
