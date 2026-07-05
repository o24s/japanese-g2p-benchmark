"""
G2Pアダプターのための抽象基底クラス
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


class G2PAdapter(ABC):
    @abstractmethod
    def g2p(self, texts: list[str]) -> list[str]:
        """text のリスト -> 空白区切り音素列のリスト"""
        ...

    @abstractmethod
    def g2k(self, texts: list[str]) -> list[str]:
        """text のリスト -> カタカナ読みのリスト"""
        ...


@dataclass(frozen=True)
class Variant:
    id: str
    adapter: str
    options: dict[str, Any]
    factory: Callable[[], G2PAdapter]
    datasets: list[str]


def make_variant(
    adapter: str,
    options: dict[str, Any],
    factory: Callable[[], G2PAdapter],
    datasets: list[str],
) -> Variant:
    parts = [f"{k}={v}" for k, v in sorted(options.items())]
    opt_suffix = f".{'.'.join(parts)}" if parts else ""

    return Variant(
        id=f"{adapter}{opt_suffix}",
        adapter=adapter,
        options=options,
        factory=factory,
        datasets=datasets,
    )
