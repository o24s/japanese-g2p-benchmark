#!/usr/bin/env python3
"""
Downloads source data and converts it into a unified benchmark format.

Run:
    uv run init.py           # first run or update check (uses locked URLs)
    uv run init.py --update  # fetches latest commit SHAs, updates lock, and downloads
    uv run init.py --force   # re-download and re-convert everything using locked URLs

Outputs:
    benchmarks/jsut_phoneme.jsonl   -- PER benchmark (phoneme sequences, manual annotation)
    benchmarks/kana_lvs.jsonl  -- KER benchmark, ou->オー, ei->エー
                                        sources: jsut-label kana, jvs_nonpara_kana, joyo-kanji-yomi
    benchmarks/kana_no_lvs.jsonl  -- KER benchmark, ou/ei preserved
                                        sources: rohan4600
    benchmarks/kana_ctxt.jsonl       -- KER benchmark derived from AJIMEE-Bench (JWTD_v2)
                                        (IME task inverted)

Note on kana_long_vowel field:
    "lvs" = o u ->オー, ei -> エー (uses ー)
    "no_lvs" = kana spelling preserved (no ー substitution)
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

RAW = Path("data/raw")
BENCH = Path("benchmarks")
LOCK_FILE = Path("data/sources.lock.json")
RAW.mkdir(parents=True, exist_ok=True)
BENCH.mkdir(parents=True, exist_ok=True)

_HIRA2KATA_TRANS = str.maketrans(
    "".join(chr(0x3041 + i) for i in range(86)),
    "".join(chr(0x30A1 + i) for i in range(86)),
)


def hira2kata(text: str) -> str:
    """Fast Hiragana to Katakana conversion"""
    return text.translate(_HIRA2KATA_TRANS)


def load_lock() -> dict:
    if LOCK_FILE.exists():
        return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    return {}


def save_lock(lock: dict) -> None:
    LOCK_FILE.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def resolve_github_raw_url(url: str) -> str:
    m = re.match(
        r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)", url
    )
    if not m:
        return url

    owner, repo, branch_or_sha, filepath = m.groups()
    if re.match(r"^[0-9a-f]{40}$", branch_or_sha):
        return url

    repo_url = f"https://github.com/{owner}/{repo}.git"
    try:
        branch = branch_or_sha.split("/")[-1]
        result = subprocess.run(
            ["git", "ls-remote", repo_url, branch],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            sha = result.stdout.split()[0]
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{filepath}"
    except FileNotFoundError:
        print(f"  [WARN] git command not found. Falling back to {url}")
    except subprocess.TimeoutExpired:
        print(f"  [WARN] git ls-remote timed out. Falling back to {url}")
    except Exception as e:
        print(f"  [WARN] Failed to resolve SHA: {e}")

    return url


def sync_file_source(
    name: str, default_url: str, dest: Path, lock: dict, update: bool, force: bool
) -> bool:
    """Download a single file. Updates lock with SHA if --update is passed."""
    if update or name not in lock:
        if update:
            print(f"  Resolving latest SHA for {name} ...")
        url = resolve_github_raw_url(default_url)
        lock[name] = {"url": url}
    else:
        url = lock[name]["url"]

    if dest.exists() and not force and not update:
        print(f"  [up-to-date] {name}")
        return False

    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved -> {dest}")
    return True


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records -> {path}")


# jsut-label  (PER + KER lvs)
#    Single file: text_kana/basic5000.yaml
#    Format (one entry spans multiple lines):
#      text_level2: <original text>
#      kana_level2: <katakana reading>
#      phone_level3: <p1>-<p2>-...

JSUT_YAML_URL = (
    "https://raw.githubusercontent.com/prj-beatrice/jsut-label"
    "/refs/heads/master/text_kana/basic5000.yaml"
)
JSUT_YAML = RAW / "basic5000.yaml"


def parse_jsut_yaml() -> Iterator[dict]:
    """
    Parse basic5000.yaml into (text, kana, phonemes) triples.
    """
    current: dict = {}
    idx = 0

    with open(JSUT_YAML, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("text_level2:"):
                current["text"] = line.removeprefix("text_level2:").strip()
            elif line.startswith("kana_level2:"):
                current["kana"] = hira2kata(line.removeprefix("kana_level2:").strip())
            elif line.startswith("phone_level3:"):
                raw = line.removeprefix("phone_level3:").strip()
                current["phonemes"] = raw.split("-")
                idx += 1
                yield {
                    "id": f"BASIC5000_{idx:04d}",
                    "text": current.get("text", ""),
                    "kana": current.get("kana", ""),
                    "phonemes": current["phonemes"],
                }
                current = {}


# JVS-nonpara-kana dataset (KER lvs)
#    CSV: base, text, kana

JVS_CSV_URL = (
    "https://raw.githubusercontent.com/CyberAgentAILab"
    "/jvs_nonpara_kana/main/jvs_nonpara_kana.csv"
)
JVS_CSV = RAW / "jvs_nonpara_kana.csv"


def parse_jvs() -> Iterator[dict]:
    with open(JVS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "id": row["base"],
                "text": row["text"],
                "kana": hira2kata(row["kana"]),
            }


# ROHAN4600 (KER no_lvs)
#    Format: ROHAN4600_NNNN:text(ruby注記),kana
#    - colon separates ID from the rest
#    - last comma separates text(ruby) from kana
#    - ruby notation (漢字(よみ)) is stripped from text, leaving plain kanji

ROHAN_TXT_URL = (
    "https://raw.githubusercontent.com/mmorise/rohan4600"
    "/main/Rohan4600_transcript_utf8.txt"
)
ROHAN_TXT = RAW / "Rohan4600_transcript_utf8.txt"


def strip_ruby(text: str) -> str:
    """Remove ruby annotations: 漢字(よみ) -> 漢字"""
    return re.sub(r"\([^)]+\)", "", text)


def parse_rohan() -> Iterator[dict]:
    with open(ROHAN_TXT, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            colon_idx = line.find(":")
            if colon_idx == -1:
                continue
            uid = line[:colon_idx]
            rest = line[colon_idx + 1 :]
            comma_idx = rest.rfind(",")
            if comma_idx == -1:
                continue
            text_ruby = rest[:comma_idx]
            kana = rest[comma_idx + 1 :]
            yield {
                "id": uid,
                "text": strip_ruby(text_ruby),
                "kana": hira2kata(kana),
            }


# Joyo Kanji Yomi Benchmark  (KER lvs)
#    HuggingFace: sbintuitions/joyo-kanji-yomi-benchmark
#    columns: key, normalized_text, normalized_pron
#    normalized_pron: full-sentence katakana, target reading in <>
#    Uses lvs (long vowel symbol) form


def fetch_and_parse_joyo(lock: dict, update: bool, force: bool) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' not found; install with: pip install datasets")
        sys.exit(1)

    name = "joyo-kanji-yomi-benchmark"
    repo_id = f"sbintuitions/{name}"

    if update or name not in lock:
        if update:
            print(f"  Resolving latest SHA for {repo_id} via git ls-remote ...")
        url = f"https://huggingface.co/datasets/{repo_id}"
        sha = "main"
        try:
            res = subprocess.run(
                ["git", "ls-remote", url, "main"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0 and res.stdout:
                sha = res.stdout.split()[0]
        except FileNotFoundError:
            print("  [WARN] git command not found. Falling back to main branch.")
        except subprocess.TimeoutExpired:
            print("  [WARN] git ls-remote timed out. Falling back to main branch.")
        except Exception as e:
            print(f"  [WARN] Failed to resolve HF SHA: {e}")

        lock[name] = {"hf_dataset": repo_id, "revision": sha}

    revision = lock[name].get("revision", "main")

    if name in lock and not force and not update:
        print(f"  [up-to-date] {name} (from HF cache)")
    else:
        print(f"  Loading {repo_id} @ {revision[:8]} via HuggingFace datasets ...")

    ds = load_dataset(repo_id, split="train", revision=revision)
    for row in ds:
        pron_full = row["normalized_pron"]
        target_match = re.search(r"<([^>]+)>", pron_full)
        yield {
            "id": row["key"],
            "text": row["normalized_text"],
            "kana": hira2kata(re.sub(r"[<>]", "", pron_full)),
            "target_kana": hira2kata(target_match.group(1)) if target_match else "",
        }


# AJIMEE-Bench / JWTD_v2  (KER IME-derived, lvs)
#    IME task inverted: input=katakana, expected_output=surface

AJIMEE_JSON_URL = (
    "https://raw.githubusercontent.com/azooKey/AJIMEE-Bench"
    "/main/JWTD_v2/v1/evaluation_items.json"
)
AJIMEE_JSON = RAW / "JWTD_v2_evaluation_items.json"


def parse_ajimee() -> Iterator[dict]:
    with open(AJIMEE_JSON, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if not item.get("expected_output"):
            continue
        kana = hira2kata(item.get("input", ""))
        for i, surface in enumerate(item["expected_output"]):
            yield {
                "id": f"ajimee_{item['index']}_{i}",
                "text": surface,
                "kana": kana,
                "context": item.get("context_text", ""),
                "original_text": item.get("original_text", ""),
            }


def build_benchmarks(update: bool, force: bool) -> None:
    lock = load_lock()

    print("\njsut-label:")
    sync_file_source("jsut-label", JSUT_YAML_URL, JSUT_YAML, lock, update, force)
    jsut = list(parse_jsut_yaml())
    print(f"  Parsed {len(jsut)} entries from basic5000.yaml")

    per_records = [
        {
            "id": r["id"],
            "source": "jsut-label",
            "text": r["text"],
            "phonemes": r["phonemes"],
            "kana_long_vowel": "lvs",
        }
        for r in jsut
    ]
    write_jsonl(BENCH / "jsut_phoneme.jsonl", per_records)

    jsut_kana_records = [
        {
            "id": r["id"],
            "source": "jsut-label",
            "text": r["text"],
            "kana": r["kana"],
            "kana_long_vowel": "lvs",
        }
        for r in jsut
    ]

    print("\nJVS-nonpara-kana dataset:")
    sync_file_source("jvs_nonpara_kana", JVS_CSV_URL, JVS_CSV, lock, update, force)
    jvs_records = [
        {
            "id": r["id"],
            "source": "jvs_nonpara_kana",
            "text": r["text"],
            "kana": r["kana"],
            "kana_long_vowel": "lvs",
        }
        for r in parse_jvs()
    ]

    print("\nJoyo Kanji Yomi Benchmark:")
    joyo_records = [
        {
            "id": r["id"],
            "source": "joyo-kanji-yomi-benchmark",
            "text": r["text"],
            "kana": r["kana"],
            "target_kana": r["target_kana"],
            "kana_long_vowel": "lvs",
        }
        for r in fetch_and_parse_joyo(lock, update, force)
    ]

    pron_records = jsut_kana_records + jvs_records + joyo_records
    write_jsonl(BENCH / "kana_lvs.jsonl", pron_records)

    print("\nROHAN:")
    sync_file_source("rohan4600", ROHAN_TXT_URL, ROHAN_TXT, lock, update, force)
    rohan_records = [
        {
            "id": r["id"],
            "source": "rohan4600",
            "text": r["text"],
            "kana": r["kana"],
            "kana_long_vowel": "no_lvs",
        }
        for r in parse_rohan()
    ]
    write_jsonl(BENCH / "kana_no_lvs.jsonl", rohan_records)

    print("\nAJIMEE-Bench:")
    sync_file_source("ajimee-bench", AJIMEE_JSON_URL, AJIMEE_JSON, lock, update, force)
    ajimee_records = [
        {
            "id": r["id"],
            "source": "ajimee-bench",
            "text": r["text"],
            "kana": r["kana"],
            "kana_long_vowel": "lvs",
            "context": r["context"],
        }
        for r in parse_ajimee()
    ]
    write_jsonl(BENCH / "kana_ctxt.jsonl", ajimee_records)

    save_lock(lock)

    print("\n=== Done ===")
    print(
        f"  jsut_phoneme.jsonl  : {len(per_records):>6} records  (PER, phoneme-level)"
    )
    print(f"  kana_lvs.jsonl : {len(pron_records):>6} records  (KER, ou/ei -> ー)")
    print(f"    jsut-label        : {len(jsut_kana_records):>6}")
    print(f"    jvs_nonpara_kana  : {len(jvs_records):>6}")
    print(f"    joyo-kanji-yomi   : {len(joyo_records):>6}")
    print(f"  kana_no_lvs.jsonl : {len(rohan_records):>6} records  (KER, ou/ei preserved)")
    print(
        f"  kana_ime.jsonl      : {len(ajimee_records):>6} records  (KER, IME-derived)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize japanese-g2p-benchmark data"
    )
    parser.add_argument(
        "--update", action="store_true", help="Fetch latest SHAs and update lock file"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download files using locked URLs"
    )
    args = parser.parse_args()

    build_benchmarks(update=args.update, force=args.force)
