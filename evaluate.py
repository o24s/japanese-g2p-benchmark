#!/usr/bin/env python3
"""
現在の環境にインストールされているアダプタでベンチマークを実行する。

Usage:
    uv run --with pyopenjtalk python evaluate.py --adapters pyopenjtalk
    uv run --with pyopenjtalk-plus[onnxruntime] python evaluate.py --adapters pyopenjtalk_plus
    uv run --with haqumei python evaluate.py --adapters haqumei

Options:
    --adapters    カンマ区切り前方一致フィルタ
    --datasets    カンマ区切り (phoneme, lvs, no_lvs, ctxt)
    --batch-size  バッチサイズ (default: 256)
    --out         結果 JSON の出力先
    --no-tsv      TSV を出力しない
    --no-table    コンソールテーブルを出力しない
"""

from typing import Sequence

import argparse
import json
import sys
import time
from itertools import batched
from pathlib import Path

from rapidfuzz.distance import Levenshtein

BENCH = Path("benchmarks")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def _get_diff_string(a: list[str] | str, b: list[str] | str) -> str:
    parts: list[str] = []
    is_str = isinstance(a, str)

    for op in Levenshtein.opcodes(a, b):
        tag = op.tag
        src_start = op.src_start
        src_end = op.src_end
        dest_start = op.dest_start
        dest_end = op.dest_end

        if tag == "equal":
            if isinstance(a, str):
                parts.append(a[src_start:src_end])
            else:
                parts.extend(a[src_start:src_end])
        elif tag == "replace":
            old = (
                "".join(a[src_start:src_end])
                if is_str
                else " ".join(a[src_start:src_end])
            )
            new = (
                "".join(b[dest_start:dest_end])
                if is_str
                else " ".join(b[dest_start:dest_end])
            )
            parts.append(f"[-{old}][+{new}]")
        elif tag == "delete":
            old = (
                "".join(a[src_start:src_end])
                if is_str
                else " ".join(a[src_start:src_end])
            )
            parts.append(f"[-{old}]")
        elif tag == "insert":
            new = (
                "".join(b[dest_start:dest_end])
                if is_str
                else " ".join(b[dest_start:dest_end])
            )
            parts.append(f"[+{new}]")

    return "".join(parts) if is_str else " ".join(parts)


def _fmt_diff(ops: list[tuple[str, str, str]], sep: str = " ") -> str:
    parts = []
    for tag, av, bv in ops:
        if tag == "equal":
            parts.append(av)
        elif tag == "substitute":
            parts.append(f"[-{av}]")
            parts.append(f"[+{bv}]")
        elif tag == "delete":
            parts.append(f"[-{av}]")
        else:
            parts.append(f"[+{bv}]")
    return sep.join(p for p in parts if p)


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
    total_s = total_d = total_i = total_n_filt = total_n_incl = 0
    sentence_errors = 0

    for record, pred in zip(records, preds):
        raw = pred if isinstance(pred, list) else pred.strip().split()
        ref_raw = record["phonemes"]

        pred_filt = _prepare_phones(raw, True)
        ref_filt = _prepare_phones(ref_raw, True)
        _pred_incl = _prepare_phones(raw, False)
        ref_incl = _prepare_phones(ref_raw, False)

        ops_filt = Levenshtein.editops(ref_filt, pred_filt)
        s = sum(1 for op in ops_filt if op.tag == "replace")
        d = sum(1 for op in ops_filt if op.tag == "delete")
        ins = sum(1 for op in ops_filt if op.tag == "insert")

        total_s += s
        total_d += d
        total_i += ins
        total_n_filt += len(ref_filt)
        total_n_incl += len(ref_incl)

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

    n_sentences = len(records)
    overall_filt = (
        (total_s + total_d + total_i) / total_n_filt * 100 if total_n_filt else 0.0
    )
    header = [
        "Phoneme Error Rate report (pau_filtered)",
        f"Total sentences     : {n_sentences}",
        f"Sentences with errors: {sentence_errors}",
        f"Overall PER (S+D+I / N): {overall_filt:.2f}%  "
        f"(S={total_s} D={total_d} I={total_i} N={total_n_filt})",
        "",
    ]
    path.write_text("\n".join(header + lines), encoding="utf-8")


def _dump_kana_report(
    path: Path,
    records: list[dict],
    preds: list[str],
    dataset: str,
) -> None:
    lines: list[str] = []
    total_s = total_d = total_i = total_n = 0
    sentence_errors = 0

    for record, pred in zip(records, preds):
        gold = record["kana"]
        ops = Levenshtein.editops(gold, pred)
        s = sum(1 for op in ops if op.tag == "replace")
        d = sum(1 for op in ops if op.tag == "delete")
        ins = sum(1 for op in ops if op.tag == "insert")

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

    n_sentences = len(records)
    overall = (total_s + total_d + total_i) / total_n * 100 if total_n else 0.0
    header = [
        f"Katakana Error Rate report [{dataset}]",
        f"Total sentences     : {n_sentences}",
        f"Sentences with errors: {sentence_errors}",
        f"Overall KER (S+D+I / N): {overall:.2f}%  "
        f"(S={total_s} D={total_d} I={total_i} N={total_n})",
        "",
    ]
    path.write_text("\n".join(header + lines), encoding="utf-8")


def _eval_phoneme(
    adapter,
    records: list[dict],
    batch_size: int,
    report_path: Path | None = None,
) -> dict:
    preds = _batch_call(adapter.g2p, [r["text"] for r in records], batch_size)

    # pauありとpauなしのカウント用
    s_incl = d_incl = i_incl = n_incl = 0
    s_filt = d_filt = i_filt = n_filt = 0

    for pred, record in zip(preds, records):
        raw = pred if isinstance(pred, list) else pred.strip().split()
        ref_raw = record["phonemes"]

        # pauあり
        pred_incl = _prepare_phones(raw, False)
        ref_incl = _prepare_phones(ref_raw, False)
        ops_i = Levenshtein.editops(ref_incl, pred_incl)
        s_incl += sum(1 for op in ops_i if op.tag == "replace")
        d_incl += sum(1 for op in ops_i if op.tag == "delete")
        i_incl += sum(1 for op in ops_i if op.tag == "insert")
        n_incl += len(ref_incl)

        # pauなし
        pred_filt = _prepare_phones(raw, True)
        ref_filt = _prepare_phones(ref_raw, True)
        ops_f = Levenshtein.editops(ref_filt, pred_filt)
        s_filt += sum(1 for op in ops_f if op.tag == "replace")
        d_filt += sum(1 for op in ops_f if op.tag == "delete")
        i_filt += sum(1 for op in ops_f if op.tag == "insert")
        n_filt += len(ref_filt)

    def _per_block(s: int, d: int, ins: int, n_expected: int) -> dict:
        total_ops = s + d + ins
        return {
            "per": round(total_ops / n_expected, 6) if n_expected else 0.0,
            "substitutions": s,
            "deletions": d,
            "insertions": ins,
            "edit_ops": total_ops,
            "n_expected": n_expected,
            "n_sentences": len(records),
        }

    if report_path is not None:
        _dump_phoneme_report(report_path, records, preds)

    return {
        "pau_included": _per_block(s_incl, d_incl, i_incl, n_incl),
        "pau_filtered": _per_block(s_filt, d_filt, i_filt, n_filt),
    }


def _eval_kana(
    adapter,
    records: list[dict],
    batch_size: int,
    report_path: Path | None = None,
) -> dict:
    preds = _batch_call(adapter.g2k, [r["text"] for r in records], batch_size)
    total_s = total_d = total_i = total_n = 0

    for pred, record in zip(preds, records):
        ref = record["kana"]
        ops = Levenshtein.editops(ref, pred)
        total_s += sum(1 for op in ops if op.tag == "replace")
        total_d += sum(1 for op in ops if op.tag == "delete")
        total_i += sum(1 for op in ops if op.tag == "insert")
        total_n += len(ref)

    if report_path is not None:
        _dump_kana_report(report_path, records, preds, dataset="kana")

    total_ops = total_s + total_d + total_i
    return {
        "ker": round(total_ops / total_n, 6) if total_n else 0.0,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
        "edit_ops": total_ops,
        "n_expected": total_n,
        "n_sentences": len(records),
    }


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
        dataset_cache[ds] = load_jsonl(path)
        print(f"  Loaded {ds}: {len(dataset_cache[ds])} records")

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
        help="カンマ区切り (phoneme, lvs, no_lvs, ctxt)",
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
        [s.strip() for s in args.datasets.split(",")] if args.datasets else None
    )

    dump_dir = args.dump_errors
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    print("japanese-g2p-benchmark:")
    all_results = run_evaluation(
        filter_adapters, filter_datasets, args.batch_size, dump_dir
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
