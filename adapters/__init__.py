from functools import partial

from .base import G2PAdapter, Variant, make_variant

_LVS_TASKS = ["phoneme", "lvs"]
_NO_LVS_TASKS = ["no_lvs", "ctxt"]


def _build_variants() -> list[Variant]:
    variants: list[Variant] = []

    # pyopenjtalk (vanilla)
    try:
        from .pyopenjtalk import PyOpenJTalkAdapter

        variants.append(
            make_variant(
                adapter="pyopenjtalk",
                options={},
                factory=PyOpenJTalkAdapter,
                datasets=_LVS_TASKS + _NO_LVS_TASKS,
            )
        )
    except ImportError:
        pass

    # pyopenjtalk-plus
    try:
        from .pyopenjtalk_plus import PyOpenJTalkPlusAdapter

        for tsqyomi in [True, False]:
            for sudachi in [True, False]:
                for revert in [True, False]:
                    variants.append(
                        make_variant(
                            adapter="pyopenjtalk_plus",
                            options={
                                "use_sudachi_kanji_yomi": sudachi,
                                "use_tsqyomi": tsqyomi,
                                "revert_long_vowels": revert,
                                "revert_yotsugana": revert,
                            },
                            factory=partial(
                                PyOpenJTalkPlusAdapter,
                                use_sudachi_kanji_yomi=sudachi,
                                use_tsqyomi=tsqyomi,
                                revert_long_vowels=revert,
                                revert_yotsugana=revert,
                            ),
                            datasets=_NO_LVS_TASKS if revert else _LVS_TASKS + _NO_LVS_TASKS,
                        )
                    )
    except ImportError:
        pass

    # haqumei
    try:
        from haqumei import IuPronunciation  # type: ignore

        from .haqumei import HaqumeiAdapter

        # yuu-base は「言う」の基本形だけを ユー にする (活用形は イ 段のまま)。
        _IU_VARIANTS: list[tuple[IuPronunciation, str]] = [
            (IuPronunciation.None_, "none"),
            (IuPronunciation.Yuu, "yuu"),
            (IuPronunciation.YuuBase, "yuu-base"),
        ]

        for iu, iu_label in _IU_VARIANTS:
            for revert in [True, False]:
                variants.append(
                    make_variant(
                        adapter="haqumei",
                        options={
                            "normalize_iu": iu_label,
                            "revert_long_vowels": revert,
                            "revert_yotsugana": revert,
                        },
                        factory=partial(
                            HaqumeiAdapter,
                            normalize_iu=iu,
                            revert_long_vowels=revert,
                            revert_yotsugana=revert,
                        ),
                        datasets=_NO_LVS_TASKS
                        if revert
                        else _LVS_TASKS + _NO_LVS_TASKS,
                    )
                )
    except ImportError:
        pass

    return variants


VARIANTS: list[Variant] = _build_variants()

__all__ = ["G2PAdapter", "Variant", "VARIANTS"]
