#!/usr/bin/env python3
# One command for the PyMuPDF-Layout baseline:
#   PDF folder -> per-page JSON + layout text -> URL extraction.
# body reads the layout text; footnote and reference read the per-page JSON.

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERT = HERE / "pymupdf_client.py"
EXTRACTORS = {
    "body": (HERE / "extract_body.py", "text"),
    "footnote": (HERE / "extract_footnote.py", "json"),
    "reference": (HERE / "extract_reference.py", "json"),
}


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="PyMuPDF-Layout baseline: PDF -> JSON+text -> URL extraction.")
    ap.add_argument("-i", "--input", required=True,
                    help="folder of PDFs, or a workspace with json/ and text/ when --from-converted")
    ap.add_argument("-o", "--output", required=True, help="output folder (per-module subfolders)")
    ap.add_argument("--modules", nargs="+", choices=list(EXTRACTORS), default=list(EXTRACTORS))
    ap.add_argument("--workspace", default=None,
                    help="conversion workspace (default: <output>/workspace)")
    ap.add_argument("--from-converted", action="store_true",
                    help="input is already a workspace with json/ and text/; skip conversion")
    ap.add_argument("--workers", type=int, default=1, help="PDFs converted in parallel")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ws = Path(args.input) if args.from_converted else (Path(args.workspace) if args.workspace else out / "workspace")
    json_dir = ws / "json"
    text_dir = ws / "text"

    if args.from_converted:
        print(f"Skipping conversion; reading json/ and text/ from {ws}\n")
    else:
        print("Step 1: PDF -> JSON + layout text")
        run([sys.executable, str(CONVERT), "--input", args.input,
             "--json-out", str(json_dir), "--text-out", str(text_dir),
             "--workers", str(args.workers)])
        print()

    failures = []
    for name in args.modules:
        script, kind = EXTRACTORS[name]
        src = text_dir if kind == "text" else json_dir
        print(f"Step 2: extract ({name}) from {kind}")
        try:
            run([sys.executable, str(script), "-i", str(src), "-o", str(out / name)])
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
