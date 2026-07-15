#!/usr/bin/env python3
# One command for the whole GROBID baseline:
#   PDF folder -> TEI (via a running GROBID server) -> URL extraction.
# Runs the body, footnote and reference extractors and writes each set of
# results into its own subfolder under the output directory.

import argparse
from pathlib import Path

import grobid_client
import extract_body
import extract_footnote
import extract_reference

MODULES = {
    "body": extract_body.run,
    "footnote": extract_footnote.run,
    "reference": extract_reference.run,
}


def main():
    ap = argparse.ArgumentParser(
        description="PDF -> TEI -> URL extraction (body, footnote, reference) with GROBID."
    )
    ap.add_argument("-i", "--input", required=True,
                    help="folder of PDFs, or of TEI XML when --from-tei is set")
    ap.add_argument("-o", "--output", required=True,
                    help="output folder (subfolders tei/, body/, footnote/, reference/ are created)")
    ap.add_argument("--modules", nargs="+", choices=list(MODULES), default=list(MODULES),
                    help="which extractors to run (default: all three)")
    ap.add_argument("--grobid-url", default=grobid_client.DEFAULT_GROBID_URL,
                    help="GROBID processFulltextDocument endpoint")
    ap.add_argument("--tei-dir", default=None,
                    help="where to write/read TEI (default: <output>/tei)")
    ap.add_argument("--from-tei", action="store_true",
                    help="input is already TEI XML; skip the GROBID conversion step")
    args = ap.parse_args()

    out_dir = Path(args.output)
    tei_dir = Path(args.tei_dir) if args.tei_dir else out_dir / "tei"

    if args.from_tei:
        tei_dir = Path(args.input)
        print(f"Skipping GROBID; reading TEI from {tei_dir}\n")
    else:
        print("Step 1: PDF -> TEI")
        grobid_client.convert(args.input, tei_dir, args.grobid_url)
        print()

    for name in args.modules:
        print(f"Step 2: extract ({name})")
        MODULES[name](tei_dir, out_dir / name)
        print()

    print(f"All done. Results under {out_dir}")


if __name__ == "__main__":
    main()
