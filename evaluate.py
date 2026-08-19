#!/usr/bin/env python3
"""
現在の環境にインストールされているアダプタでベンチマークを実行する。

Usage:
    uv run --with pyopenjtalk python evaluate.py --adapters pyopenjtalk
    uv run --with pyopenjtalk-plus[onnxruntime] python evaluate.py --adapters pyopenjtalk_plus
    uv run --with haqumei python evaluate.py --adapters haqumei

データセットは長音の畳み方で切ってあり、出典とは別の軸である
(`lvs` は 3 つの出典の混合)。出典で絞るときは `--sources` を使う。

Options:
    --adapters    カンマ区切り前方一致フィルタ
    --datasets    カンマ区切り (phoneme, lvs, no_lvs, ctxt)
    --sources     カンマ区切り (jsut-label, jvs_nonpara_kana,
                  joyo-kanji-yomi-benchmark, rohan4600, ajimee-bench)
    --batch-size  バッチサイズ (default: 256)
    --out         結果 JSON の出力先
    --no-tsv      TSV を出力しない
    --no-table    コンソールテーブルを出力しない
"""

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from rapidfuzz.distance import Levenshtein

BENCH = Path("benchmarks")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def count_edit_ops(
    reference: list[str] | str, hypothesis: list[str] | str
) -> tuple[int, int, int]:
    """置換・削除・挿入の数を返す。"""
    tags = Counter(op.tag for op in Levenshtein.editops(reference, hypothesis))
    return tags["replace"], tags["delete"], tags["insert"]


def _get_diff_string(a: list[str] | str, b: list[str] | str) -> str:
    """差分を `[-削除][+挿入]` の形で 1 行に並べる。

    音素列 (list) は空白区切り、カナ (str) は区切りなしで連結する。
    """
    sep = "" if isinstance(a, str) else " "

    def render(seq: list[str] | str, start: int, end: int) -> str:
        return sep.join(seq[start:end])

    parts: list[str] = []
    for op in Levenshtein.opcodes(a, b):
        src = render(a, op.src_start, op.src_end)
        dest = render(b, op.dest_start, op.dest_end)
        if op.tag == "equal":
            parts.append(src)
        elif op.tag == "replace":
            parts.append(f"[-{src}][+{dest}]")
        elif op.tag == "delete":
            parts.append(f"[-{src}]")
        elif op.tag == "insert":
            parts.append(f"[+{dest}]")

    return sep.join(parts)


_UNVOICED_VOWELS = {"A", "E", "I", "O", "U"}


def _normalize_phone(p: str) -> str:
    return p.lower() if p in _UNVOICED_VOWELS else p


def _prepare_phones(phones: list[str], filter_pau: bool) -> list[str]:
    result = [_normalize_phone(p) for p in phones]
    return [p for p in result if p != "pau"] if filter_pau else result


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _batch_call(method, texts: list[str], batch_size: int) -> list[str]:
    out = []
    for chunk in batched(texts, batch_size):
        out.extend(method(list(chunk)))
    return out


def _dump_phoneme_report(
    path: Path,
    records: list[dict],
    preds: Sequence[list[str] | str],
) -> None:
    lines: list[str] = []
    total_s = total_d = total_i = total_n = 0
    sentence_errors = 0

    # このレポートは pau を除いた側だけを載せる
    for record, pred in zip(records, preds):
        raw = pred if isinstance(pred, list) else pred.strip().split()
        pred_filt = _prepare_phones(raw, True)
        ref_filt = _prepare_phones(record["phonemes"], True)

        s, d, ins = count_edit_ops(ref_filt, pred_filt)
        total_s += s
        total_d += d
        total_i += ins
        total_n += len(ref_filt)

        if s == 0 and d == 0 and ins == 0:
            continue
        sentence_errors += 1

        per = (s + d + ins) / len(ref_filt) * 100 if ref_filt else 0.0
        diff_str = _get_diff_string(ref_filt, pred_filt)
        lines += [
            "=" * 50,
            f"[{record['id']}]",
            f"Text: {record['text']}",
            "-" * 50,
            f"Diff: {diff_str}",
            f"Sentence stats: S={s} D={d} I={ins}  "
            f"N_expected={len(ref_filt)}  PER={per:.2f}%",
            "",
        ]

    overall = (total_s + total_d + total_i) / total_n * 100 if total_n else 0.0
    header = [
        "Phoneme Error Rate report (pau_filtered)",
        f"Total sentences     : {len(records)}",
        f"Sentences with errors: {sentence_errors}",
        f"Overall PER (S+D+I / N): {overall:.2f}%  "
        f"(S={total_s} D={total_d} I={total_i} N={total_n})",
        "",
    ]
    path.write_text("\n".join(header + lines), encoding="utf-8")


def _dump_kana_report(path: Path, records: list[dict], preds: list[str]) -> None:
    lines: list[str] = []
    total_s = total_d = total_i = total_n = 0
    sentence_errors = 0

    for record, pred in zip(records, preds):
        gold = record["kana"]
        s, d, ins = count_edit_ops(gold, pred)

        total_s += s
        total_d += d
        total_i += ins
        total_n += len(gold)

        if s == 0 and d == 0 and ins == 0:
            continue
        sentence_errors += 1

        ker = (s + d + ins) / len(gold) * 100 if gold else 0.0
        diff_str = _get_diff_string(gold, pred)
        lines += [
            "=" * 50,
            f"[{record['id']}]",
            f"Text: {record.get('text', record.get('original_text', ''))}",
            f"Gold: {gold}",
            f"Pred: {pred}",
            f"Diff: {diff_str}",
            f"Sentence stats: S={s} D={d} I={ins}  "
            f"N_expected={len(gold)}  KER={ker:.2f}%",
            "",
        ]

    overall = (total_s + total_d + total_i) / total_n * 100 if total_n else 0.0
    header = [
        "Katakana Error Rate report",
        f"Total sentences     : {len(records)}",
        f"Sentences with errors: {sentence_errors}",
        f"Overall KER (S+D+I / N): {overall:.2f}%  "
        f"(S={total_s} D={total_d} I={total_i} N={total_n})",
        "",
    ]
    path.write_text("\n".join(header + lines), encoding="utf-8")


@dataclass
class Tally:
    """編集操作の累計。`add` を呼ぶたびに 1 文ぶん足す。"""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    n_expected: int = 0

    def add(self, reference: list[str] | str, hypothesis: list[str] | str) -> None:
        s, d, ins = count_edit_ops(reference, hypothesis)
        self.substitutions += s
        self.deletions += d
        self.insertions += ins
        self.n_expected += len(reference)

    @property
    def edit_ops(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def as_dict(self, key: str, n_sentences: int) -> dict:
        """誤り率と内訳をまとめる。`key` は "per" か "ker"。"""
        return {
            key: round(self.edit_ops / self.n_expected, 6) if self.n_expected else 0.0,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "edit_ops": self.edit_ops,
            "n_expected": self.n_expected,
            "n_sentences": n_sentences,
        }


def _eval_phoneme(
    adapter,
    records: list[dict],
    batch_size: int,
    report_path: Path | None = None,
) -> dict:
    preds = _batch_call(adapter.g2p, [r["text"] for r in records], batch_size)

    # pau を含めた場合と除いた場合を同時に数える
    included, filtered = Tally(), Tally()

    for pred, record in zip(preds, records):
        raw = pred if isinstance(pred, list) else pred.strip().split()
        for filter_pau, tally in ((False, included), (True, filtered)):
            tally.add(
                _prepare_phones(record["phonemes"], filter_pau),
                _prepare_phones(raw, filter_pau),
            )

    if report_path is not None:
        _dump_phoneme_report(report_path, records, preds)

    return {
        "pau_included": included.as_dict("per", len(records)),
        "pau_filtered": filtered.as_dict("per", len(records)),
    }


def _eval_kana(
    adapter,
    records: list[dict],
    batch_size: int,
    report_path: Path | None = None,
) -> dict:
    preds = _batch_call(adapter.g2k, [r["text"] for r in records], batch_size)
    tally = Tally()

    for pred, record in zip(preds, records):
        tally.add(record["kana"], pred)

    if report_path is not None:
        _dump_kana_report(report_path, records, preds)

    return tally.as_dict("ker", len(records))


DATASET_DEF: dict[str, tuple[Path, str]] = {
    "phoneme": (BENCH / "jsut_phoneme.jsonl", "phoneme"),
    "lvs": (BENCH / "kana_lvs.jsonl", "kana"),
    "no_lvs": (BENCH / "kana_no_lvs.jsonl", "kana"),
    "ctxt": (BENCH / "kana_ctxt.jsonl", "kana"),
}

_COLUMNS = [
    ("phoneme", "PER"),
    ("lvs", "KER-lvs"),
    ("no_lvs", "KER-no_lvs"),
    ("ctxt", "KER-ctxt"),
]

# 列名だけでは出典が分からないので、表の下に凡例として出す。
DATASET_SOURCES: dict[str, str] = {
    "phoneme": "jsut-label (JSUT Basic5000) 5,000",
    "lvs": "jsut-label 5,000 + jvs_nonpara_kana 3,000 + joyo-kanji-yomi-benchmark 13,095",
    "no_lvs": "rohan4600 4,600",
    "ctxt": "ajimee-bench (JWTD_v2, IME タスクの逆) 422",
}


def check_datasets(names: list[str]) -> list[str]:
    """`--datasets` の値を検査して返す。

    データセットは長音の畳み方で分けてあり、出典とは別の軸である
    (`lvs` は 3 つの出典の混合)。出典で絞るには `--sources` を使う。
    """
    out = []
    for raw in names:
        name = raw.strip()
        if name not in DATASET_DEF:
            known = ", ".join(DATASET_DEF)
            raise SystemExit(
                f"不明なデータセットです: {raw}  (指定できるのは {known})\n"
                f"出典で絞りたい場合は --sources を使ってください。"
            )
        out.append(name)
    return out


def sources_of(records: list[dict]) -> list[str]:
    """レコードに含まれる出典を並べる。"""
    return sorted({str(r["source"]) for r in records if r.get("source")})


def filter_by_source(records: list[dict], sources: list[str] | None) -> list[dict]:
    """`--sources` で出典を絞る。指定が無ければそのまま返す。"""
    if not sources:
        return records
    return [r for r in records if r.get("source") in sources]


def _fmt_cell(results_dict: dict, ds: str) -> str:
    r = results_dict.get(ds)
    if r is None:
        return "—"
    if "error" in r:
        return "ERR"
    if ds == "phoneme":
        sub = r.get("pau_filtered", r)
        return f"{sub['per'] * 100:.2f}%*"
    key = "ker"
    return f"{r[key] * 100:.2f}%"


def _fmt_options(options: dict) -> str:
    if not options:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in options.items())


def print_table(all_results: dict) -> None:
    if not all_results:
        print("(結果なし)")
        return

    adapter_w = 18
    options_w = 110

    header_left = f"{'Adapter':<{adapter_w}}  {'Options':<{options_w}}"
    header_scores = "  ".join(f"{label:>9}" for _, label in _COLUMNS)
    print(f"\n{header_left}  {header_scores}")
    print("-" * (adapter_w + 2 + options_w + 2 + 11 * len(_COLUMNS)))

    for data in all_results.values():
        adapter_str = data["adapter"]
        options_str = _fmt_options(data["options"])
        cells = "  ".join(f"{_fmt_cell(data['results'], ds):>9}" for ds, _ in _COLUMNS)
        print(f"{adapter_str:<{adapter_w}}  {options_str:<{options_w}}  {cells}")

    print()
    for ds, label in _COLUMNS:
        print(f"  {label:<10} {DATASET_SOURCES[ds]}")


def write_tsv(all_results: dict, path: Path) -> None:
    headers = ["adapter", "options"] + [label for _, label in _COLUMNS]
    rows = [headers]

    for data in all_results.values():
        row = [
            data["adapter"],
            _fmt_options(data["options"]),
        ] + [_fmt_cell(data["results"], ds) for ds, _ in _COLUMNS]
        rows.append(row)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"  TSV -> {path}")


def run_evaluation(
    filter_adapters: list[str] | None,
    filter_datasets: list[str] | None,
    batch_size: int,
    dump_dir: Path | None = None,
    filter_sources: list[str] | None = None,
) -> dict:
    from adapters import VARIANTS

    variants = VARIANTS
    if filter_adapters:
        variants = [
            v for v in variants if any(v.id.startswith(f) for f in filter_adapters)
        ]
    if not variants:
        print(
            "[WARN] 一致するバリアントがありません。パッケージがインストールされているか確認してください。"
        )
        return {}

    target_datasets = (
        [d for d in DATASET_DEF if d in filter_datasets]
        if filter_datasets
        else list(DATASET_DEF)
    )

    dataset_cache: dict[str, list[dict]] = {}
    for ds in target_datasets:
        path, _ = DATASET_DEF[ds]
        if not path.exists():
            print(
                f"[WARN] {path} が見つかりません。先に uv run init.py を実行してください。"
            )
            continue
        records = load_jsonl(path)
        if filter_sources:
            kept = filter_by_source(records, filter_sources)
            if not kept:
                have = ", ".join(sources_of(records))
                print(f"  [SKIP] {ds}: 指定の出典を含みません (あるのは {have})")
                continue
            records = kept
        dataset_cache[ds] = records
        note = f"  ({'+'.join(filter_sources)} のみ)" if filter_sources else ""
        print(f"  Loaded {ds}: {len(records)} records{note}")

    # 出典名を打ち間違えると全部飛ばされて 0 件で終わるので、そこで止める
    if filter_sources and not dataset_cache:
        raise SystemExit(
            f"--sources {','.join(filter_sources)} に一致するレコードがありません。"
        )

    all_results: dict = {}

    for v in variants:
        opt_str = _fmt_options(v.options)
        disp_opts = f"  ({opt_str})" if opt_str != "-" else ""

        print(f"\n{'-' * 60}\n  Variant: {v.adapter}{disp_opts}\n{'-' * 60}")

        try:
            adapter = v.factory()
        except Exception as e:
            print(f"  [SKIP] アダプタの初期化に失敗: {e}")
            continue

        variant_results: dict = {}

        for ds in v.datasets:
            if ds not in target_datasets or ds not in dataset_cache:
                continue

            records = dataset_cache[ds]
            _, eval_type = DATASET_DEF[ds]

            print(f"  [{ds}] {len(records)} items ... ", end="", flush=True)
            t0 = time.perf_counter()

            try:
                safe_name = v.id.replace("=", "_")
                rp = (dump_dir / f"{safe_name}.{ds}.txt") if dump_dir else None

                result = (
                    _eval_phoneme(adapter, records, batch_size, report_path=rp)
                    if eval_type == "phoneme"
                    else _eval_kana(adapter, records, batch_size, report_path=rp)
                )
            except Exception as e:
                print(f"ERROR: {e}")
                variant_results[ds] = {"error": str(e)}
                continue

            elapsed = time.perf_counter() - t0
            result["elapsed_s"] = round(elapsed, 2)
            if eval_type == "phoneme":
                filt = result["pau_filtered"]
                incl = result["pau_included"]
                print(
                    f"PER(pau_filt)={filt['per'] * 100:.2f}%  "
                    f"PER(pau_incl)={incl['per'] * 100:.2f}%  "
                    f"({elapsed:.1f}s)"
                )
            else:
                print(f"KER={result['ker'] * 100:.2f}%  ({elapsed:.1f}s)")
            variant_results[ds] = result

        all_results[v.id] = {
            "adapter": v.adapter,
            "options": v.options,
            "results": variant_results,
        }

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="japanese-g2p-benchmark evaluator")
    parser.add_argument(
        "--adapters",
        default=None,
        help="カンマ区切り前方一致フィルタ (例: pyopenjtalk_plus,haqumei)",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="カンマ区切り。出典名でも指定できる "
        "(phoneme=jsut, lvs, no_lvs=rohan, ctxt=ajimee)",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="カンマ区切りで出典を絞る "
        "(jsut-label, jvs_nonpara_kana, joyo-kanji-yomi-benchmark, rohan4600, ajimee-bench)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="結果 JSON の出力先 (省略時: results/results_{adapters}.json)",
    )
    parser.add_argument("--no-tsv", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument(
        "--dump-errors",
        type=Path,
        default=None,
        metavar="DIR",
        help="エラーをダンプするディレクトリ (例: --dump-errors errors/)",
    )
    args = parser.parse_args()

    filter_adapters = (
        [s.strip() for s in args.adapters.split(",")] if args.adapters else None
    )
    filter_datasets = (
        check_datasets(args.datasets.split(",")) if args.datasets else None
    )
    filter_sources = (
        [x.strip() for x in args.sources.split(",")] if args.sources else None
    )

    dump_dir = args.dump_errors
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    print("japanese-g2p-benchmark:")
    all_results = run_evaluation(
        filter_adapters, filter_datasets, args.batch_size, dump_dir, filter_sources
    )

    if not all_results:
        sys.exit(1)

    slug = args.adapters.replace(",", "_") if args.adapters else "all"
    out_path = args.out or (RESULTS / f"results_{slug}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n  JSON -> {out_path}")

    if not args.no_tsv:
        write_tsv(all_results, out_path.with_suffix(".tsv"))
    if not args.no_table:
        print_table(all_results)


if __name__ == "__main__":
    main()
