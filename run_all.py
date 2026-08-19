#!/usr/bin/env python3
"""
各アダプタを独立した uv 環境で評価し、results/results.json にマージする。

pyopenjtalk と pyopenjtalk-plus はパッケージ名が衝突するため、
それぞれ別の uv run --with 環境で評価する。

Usage:
    uv run python run_all.py
    uv run python run_all.py --skip haqumei
    uv run python run_all.py --datasets phoneme,no_lvs
    uv run python run_all.py --sources rohan4600
    uv run python run_all.py --dry-run

リリース前の候補を測るときは、`uv run --with` に渡すものを差し替えられる。

    G2P_BENCH_PKG_HAQUMEI=/path/to/haqumei-0.9.0-*.whl uv run python run_all.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from evaluate import print_table, write_tsv

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# (pip パッケージ名, adapter フィルタ前方一致, 中間出力ファイル)
_RUNS = [
    ("pyopenjtalk", "pyopenjtalk", RESULTS / "results_pyopenjtalk.json"),
    (
        "pyopenjtalk-plus[onnxruntime,tsqyomi]",
        "pyopenjtalk_plus",
        RESULTS / "results_pyopenjtalk_plus.json",
    ),
    ("haqumei", "haqumei", RESULTS / "results_haqumei.json"),
]

_COMMON_PKGS = ["datasets", "rapidfuzz"]


def _resolve_pkg(pkg: str, adapter_filter: str) -> str:
    """`uv run --with` に渡すものを環境変数で差し替える。

    未公開のビルドを測るために使う。既定は `_RUNS` の pip パッケージ名。

        G2P_BENCH_PKG_HAQUMEI=/path/to/haqumei-0.9.0-*.whl uv run python run_all.py
    """
    return os.environ.get(f"G2P_BENCH_PKG_{adapter_filter.upper()}", pkg)


def _build_cmd(
    pkg: str,
    adapter_filter: str,
    out: Path,
    datasets: str | None,
    dump_dir: Path | None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        *[arg for p in _COMMON_PKGS for arg in ("--with", p)],
        "--with",
        _resolve_pkg(pkg, adapter_filter),
        "python",
        "evaluate.py",
        "--adapters",
        adapter_filter,
        "--out",
        str(out),
        "--no-tsv",
        "--no-table",
    ]
    if datasets:
        cmd += ["--datasets", datasets]
    if dump_dir:
        cmd += ["--dump-errors", str(dump_dir)]
    return cmd


def _run(
    pkg: str,
    adapter_filter: str,
    out: Path,
    datasets: str | None,
    dump_dir: Path | None,
    dry_run: bool,
) -> bool:
    cmd = _build_cmd(pkg, adapter_filter, out, datasets, dump_dir)

    if dry_run:
        print("  " + " ".join(cmd))
        return True
    print(f"\n{'#' * 60}\n# {pkg}\n")

    return subprocess.run(cmd).returncode == 0


def _merge(paths: list[Path]) -> dict:
    merged: dict = {}
    for p in paths:
        if not p.exists():
            print(f"  [WARN] {p} が見つかりません (評価が失敗した可能性)")
            continue
        merged.update(json.loads(p.read_text(encoding="utf-8")))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="全アダプタ評価オーケストレータ")
    parser.add_argument(
        "--skip",
        default=None,
        help="カンマ区切りでスキップする adapter フィルタ (例: haqumei,pyopenjtalk_plus)",
    )
    parser.add_argument(
        "--datasets", default=None, help="カンマ区切りで絞り込むデータセット"
    )
    parser.add_argument(
        "--sources", default=None, help="カンマ区切りで絞り込む出典 (例: rohan4600)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="コマンドを表示するだけで実行しない"
    )
    parser.add_argument(
        "--dump-errors",
        type=Path,
        default=Path("errors"),
        metavar="DIR",
        help="エラーダンプ先ディレクトリ (default: errors/)",
    )
    parser.add_argument(
        "--no-dump",
        action="store_true",
        help="エラーダンプを無効にする",
    )
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",")} if args.skip else set()
    runs = [(pkg, af, out) for pkg, af, out in _RUNS if af not in skip]

    if not runs:
        sys.exit("[ERROR] 全アダプタがスキップされました。")

    if args.dry_run:
        print("Dry run: 以下のコマンドを実行します")

    dump_dir = None if args.no_dump else args.dump_errors
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    for pkg, adapter_filter, out in runs:
        _run(pkg, adapter_filter, out, args.datasets, dump_dir, args.dry_run)

    if args.dry_run:
        return

    print(f"\n{'-' * 60}\n  マージ中 ...\n")
    merged = _merge([out for _, _, out in _RUNS])

    if not merged:
        sys.exit("[ERROR] マージできる結果がありませんでした。")

    combined_json = RESULTS / "results.json"
    combined_json.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  JSON -> {combined_json}")

    write_tsv(merged, RESULTS / "results.tsv")
    print_table(merged)


if __name__ == "__main__":
    main()
