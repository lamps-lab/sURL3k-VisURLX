#!/usr/bin/env python3
# One command for the olmOCR baseline:
#   PDF folder -> Markdown (olmOCR) -> URL extraction.
# Converts PDFs with olmOCR, then runs the body, footnote, and reference
# extractors over the Markdown. Each extractor writes <output>/<name>.csv and
# <output>/<name>.jsonl.

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERT = HERE / "olmocr_client.py"
EXTRACTORS = {
    "body": HERE / "extract_body.py",
    "footnote": HERE / "extract_footnote.py",
    "reference": HERE / "extract_reference.py",
}


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="olmOCR baseline: PDF -> Markdown -> URL extraction (body, footnote, reference).")
    ap.add_argument("-i", "--input", required=True,
                    help="folder of PDFs, or of .md files when --from-markdown is set")
    ap.add_argument("-o", "--output", required=True,
                    help="output folder (CSV/JSONL per module; markdown/ under the workspace)")
    ap.add_argument("--modules", nargs="+", choices=list(EXTRACTORS), default=list(EXTRACTORS),
                    help="which extractors to run (default: all three)")
    ap.add_argument("--workspace", default=None,
                    help="olmOCR workspace (default: <output>/olmocr_workspace)")
    ap.add_argument("--from-markdown", action="store_true",
                    help="input is already a directory of .md; skip the olmOCR conversion")
    ap.add_argument("--chunk-size", type=int, default=64, help="PDFs per olmOCR chunk")
    ap.add_argument("--workers", type=int, default=1, help="olmOCR workers")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_markdown:
        md_dir = Path(args.input)
        print(f"Skipping conversion; reading Markdown from {md_dir}\n")
    else:
        ws = Path(args.workspace) if args.workspace else out / "olmocr_workspace"
        print("Step 1: PDF -> Markdown (olmOCR)")
        run([sys.executable, str(CONVERT),
             "--input_dir", args.input,
             "--workspace", str(ws),
             "--log_csv", str(out / "olmocr_conversion_log.csv"),
             "--chunk_size", str(args.chunk_size),
             "--workers", str(args.workers)])
        md_dir = ws / "markdown"
        print()

    failures = []
    for name in args.modules:
        print(f"Step 2: extract ({name})")
        cmd = [sys.executable, str(EXTRACTORS[name]),
               "--input", str(md_dir),
               "--out_csv", str(out / f"{name}.csv"),
               "--out_jsonl", str(out / f"{name}.jsonl")]
        if name == "reference":
            cmd += ["--summary_json", str(out / "reference_summary.json")]
        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"  [!] {name} failed (exit {e.returncode})")
            failures.append(name)
        print()

    if failures:
        print(f"Done with failures: {', '.join(failures)}. Results under {out}")
        sys.exit(1)
    print(f"All done. Results under {out}")


if __name__ == "__main__":
    main()
