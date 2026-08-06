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

        for sudachi in [True, False]:
            for revert in [True, False]:
                variants.append(
                    make_variant(
                        adapter="pyopenjtalk_plus",
                        options={
                            "use_sudachi_kanji_yomi": sudachi,
                            "revert_long_vowels": revert,
                            "revert_yotsugana": revert,
                        },
                        factory=partial(
                            PyOpenJTalkPlusAdapter,
                            use_sudachi_kanji_yomi=sudachi,
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

        _IU_VARIANTS: list[tuple[IuPronunciation, str]] = [
            (IuPronunciation.None_, "none"),
            (IuPronunciation.Yuu, "yuu"),
        ]

        for unidic in [False, True]:
            for iu, iu_label in _IU_VARIANTS:
                for revert in [True, False]:
                    variants.append(
                        make_variant(
                            adapter="haqumei",
                            options={
                                "use_unidic_yomi": unidic,
                                "normalize_iu": iu_label,
                                "revert_long_vowels": revert,
                                "revert_yotsugana": revert,
                            },
                            factory=partial(
                                HaqumeiAdapter,
                                use_unidic_yomi=unidic,
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
