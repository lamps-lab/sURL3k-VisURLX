#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def find_pdfs(input_dir: Path, recursive: bool = True) -> List[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(input_dir.glob(pattern))


def expected_markdown_path(workspace: Path, pdf_path: Path) -> Path:
    return workspace / "markdown" / f"{pdf_path.stem}.md"


def chunks(items: List[Path], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def write_log_row(log_csv: Path, row: dict):
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_csv.exists()

    with log_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdf_path",
                "expected_md",
                "status",
                "return_code",
                "seconds",
                "error",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_olmocr_chunk(args, pdf_chunk: List[Path]) -> int:
    cmd = [
        "olmocr",
        str(args.workspace),
        "--markdown",
    ]

    if args.server:
        cmd += ["--server", args.server]

    if args.model:
        cmd += ["--model", args.model]

    if args.api_key:
        cmd += ["--api_key", args.api_key]

    if args.workers is not None:
        cmd += ["--workers", str(args.workers)]

    if args.max_concurrent_requests is not None:
        cmd += ["--max_concurrent_requests", str(args.max_concurrent_requests)]

    if args.pages_per_group is not None:
        cmd += ["--pages_per_group", str(args.pages_per_group)]

    cmd += ["--pdfs"]
    cmd += [str(p) for p in pdf_chunk]

    print("\n[INFO] Running command:")
    print(" ".join(cmd[:8]) + f" ... ({len(pdf_chunk)} PDFs)")
    print(flush=True)

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--log_csv", default=None,
                        help="conversion log CSV (default: <workspace>/olmocr_conversion_log.csv)")

    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--non_recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # OLMOCR options documented for local/remote inference
    parser.add_argument("--server", default="", help="Optional OpenAI-compatible server URL.")
    parser.add_argument("--model", default="", help="Optional model name.")
    parser.add_argument("--api_key", default=os.environ.get("OLMOCR_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_concurrent_requests", type=int, default=None)
    parser.add_argument("--pages_per_group", type=int, default=None)

    args = parser.parse_args()

    args.input_dir = Path(args.input_dir)
    args.workspace = Path(args.workspace)
    args.log_csv = Path(args.log_csv) if args.log_csv else args.workspace / "olmocr_conversion_log.csv"

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    args.workspace.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(args.input_dir, recursive=not args.non_recursive)

    if not pdfs:
        print(f"[WARN] No PDFs found in {args.input_dir}")
        return

    print(f"[INFO] Found {len(pdfs)} PDFs")

    # Warn if duplicate stems exist, because OLMOCR markdown output is stem-based.
    stems = {}
    for pdf in pdfs:
        stems.setdefault(pdf.stem, []).append(pdf)

    duplicate_stems = {stem: paths for stem, paths in stems.items() if len(paths) > 1}
    if duplicate_stems:
        print("[WARN] Duplicate PDF stems found. OLMOCR may overwrite Markdown outputs with same stem.")
        for stem, paths in list(duplicate_stems.items())[:20]:
            print(f"  stem={stem}:")
            for p in paths:
                print(f"    {p}")
        print("[WARN] Rename duplicate PDFs before running if you need all outputs preserved.")

    if args.overwrite:
        pdfs_to_run = pdfs
    else:
        pdfs_to_run = [
            pdf for pdf in pdfs
            if not expected_markdown_path(args.workspace, pdf).exists()
        ]

    print(f"[INFO] PDFs to run: {len(pdfs_to_run)}")
    print(f"[INFO] Existing outputs skipped: {len(pdfs) - len(pdfs_to_run)}")

    if not pdfs_to_run:
        print("[INFO] Nothing to do.")
        return

    for idx, pdf_chunk in enumerate(chunks(pdfs_to_run, args.chunk_size), start=1):
        print(f"\n[INFO] Processing chunk {idx} with {len(pdf_chunk)} PDFs")
        start = time.time()
        return_code = run_olmocr_chunk(args, pdf_chunk)
        seconds = round(time.time() - start, 2)

        status = "ok" if return_code == 0 else "failed"
        error = "" if return_code == 0 else f"olmocr exited with code {return_code}"

        for pdf in pdf_chunk:
            write_log_row(
                args.log_csv,
                {
                    "pdf_path": str(pdf),
                    "expected_md": str(expected_markdown_path(args.workspace, pdf)),
                    "status": status,
                    "return_code": return_code,
                    "seconds": seconds,
                    "error": error,
                },
            )

        if return_code != 0:
            print(f"[ERROR] Chunk {idx} failed with return code {return_code}")
            sys.exit(return_code)

    print("\n[DONE] OLMOCR conversion completed.")
    print(f"[INFO] Markdown output directory: {args.workspace / 'markdown'}")
    print(f"[INFO] Log CSV: {args.log_csv}")


if __name__ == "__main__":
    main()
