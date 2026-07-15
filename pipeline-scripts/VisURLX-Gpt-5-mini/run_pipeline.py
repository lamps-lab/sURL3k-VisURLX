#!/usr/bin/env python3
"""
VisURLX master runner.

Runs the body, footnote, and reference modules over every PDF in an input
directory. Each PDF is passed to all three modules. A worker pool processes
several PDFs at once; within one PDF the three modules run in sequence, so the
concurrency comes from processing different PDFs in parallel.

Each module writes its own per-PDF JSON into its own output subfolder:
    <output>/body/        body module results
    <output>/footnote/    footnote module results
    <output>/reference/   reference module results

Usage:
    export OPENAI_API_KEY=sk-...
    python run_pipeline.py --input ./pdfs --output ./out

    # override the primary rendering DPI and use 4 workers
    python run_pipeline.py --input ./pdfs --output ./out --dpi 300 --workers 4

    # run only some modules
    python run_pipeline.py --input ./pdfs --output ./out --modules body,footnote
"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("ERROR: openai not installed. Run: pip install openai")

import body_module
import footnote_module
import reference_module


# Which modules exist and how to drive each one. For every module we record:
#   run          the module's process_pdf function
#   primary_dpi  the DPI globals that render the main page image. --dpi
#                overrides these. Strip/neighbour DPIs are left untouched so a
#                module's own low-resolution page strips stay as the authors set
#                them.
MODULES = {
    "body": {
        "module": body_module,
        "run": body_module.process_pdf,
        "primary_dpi": ["PAGE_DPI"],
    },
    "footnote": {
        "module": footnote_module,
        "run": footnote_module.process_pdf,
        "primary_dpi": ["DETECT_DPI", "RESTORE_DPI"],
    },
    "reference": {
        "module": reference_module,
        "run": reference_module.process_pdf,
        "primary_dpi": ["EXTRACT_DPI", "DETECT_DPI", "RESTORE_DPI", "PHASE01_DPI"],
    },
}


def apply_dpi_override(module_names, dpi):
    """Set each module's primary-image DPI globals to dpi. Neighbour/strip DPIs
    are not touched, so a module keeps its own value for the small page strips
    it renders only to finish a sentence that crosses a page boundary."""
    if dpi is None:
        return
    for name in module_names:
        spec = MODULES[name]
        for const in spec["primary_dpi"]:
            setattr(spec["module"], const, dpi)


def run_one_pdf(client, pdf_path, out_root, module_names):
    """Run the selected modules on a single PDF. Returns (pdf_name, per-module
    status dict). One module failing does not stop the others."""
    result = {}
    for name in module_names:
        spec = MODULES[name]
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            spec["run"](client, pdf_path, out_dir)
            result[name] = "ok"
        except Exception as e:
            result[name] = f"error: {e}"
            print(f"  [!] {name} failed on {pdf_path.name}: {e}")
            traceback.print_exc()
    return pdf_path.name, result


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the VisURLX body, footnote, and reference modules "
                    "over a directory of PDFs.")
    p.add_argument("--input", required=True, type=Path,
                   help="Directory of input PDFs (searched recursively).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output directory. Per-module subfolders are created "
                        "inside it.")
    p.add_argument("--dpi", type=int, default=None,
                   help="Override the primary page-rendering DPI for every "
                        "module. If omitted, each module uses its own default.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of PDFs processed in parallel (default 1). "
                        "Raise for throughput; higher values increase peak API "
                        "load and the risk of rate limiting.")
    p.add_argument("--modules", type=str, default="body,footnote,reference",
                   help="Comma-separated subset of modules to run "
                        "(default: body,footnote,reference).")
    p.add_argument("--api-key", type=str, default=None,
                   help="OpenAI API key. If omitted, the OPENAI_API_KEY "
                        "environment variable is used.")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: provide an API key with --api-key or set the "
                 "OPENAI_API_KEY environment variable.")

    module_names = [m.strip() for m in args.modules.split(",") if m.strip()]
    unknown = [m for m in module_names if m not in MODULES]
    if unknown:
        sys.exit(f"ERROR: unknown module(s): {', '.join(unknown)}. "
                 f"Choose from: {', '.join(MODULES)}.")

    in_dir = args.input.expanduser().resolve()
    if not in_dir.is_dir():
        sys.exit(f"ERROR: input directory not found: {in_dir}")

    pdf_files = sorted(in_dir.rglob("*.pdf"))
    if not pdf_files:
        sys.exit(f"ERROR: no PDFs found under {in_dir}")

    out_root = args.output.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    apply_dpi_override(module_names, args.dpi)

    client = OpenAI(api_key=api_key)

    print(f"Input     : {in_dir}")
    print(f"Output    : {out_root}")
    print(f"Modules   : {', '.join(module_names)}")
    print(f"DPI       : {'module defaults' if args.dpi is None else args.dpi}")
    print(f"Workers   : {args.workers}")
    print(f"PDFs found: {len(pdf_files)}\n")

    statuses = []
    if args.workers == 1:
        for pdf_path in pdf_files:
            print(f"Processing {pdf_path.name}")
            statuses.append(run_one_pdf(client, pdf_path, out_root, module_names))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one_pdf, client, pdf_path, out_root, module_names): pdf_path
                for pdf_path in pdf_files
            }
            for fut in as_completed(futures):
                pdf_path = futures[fut]
                try:
                    statuses.append(fut.result())
                except Exception as e:
                    print(f"  [!] worker crashed on {pdf_path.name}: {e}")
                    statuses.append((pdf_path.name, {"_worker": f"crash: {e}"}))

    # Summary
    print("\n" + "=" * 60)
    print("Done.")
    ok = sum(1 for _, r in statuses if all(v == "ok" for v in r.values()))
    print(f"  PDFs processed        : {len(statuses)}")
    print(f"  PDFs with all modules ok: {ok}")
    failed = [(name, r) for name, r in statuses
              if any(v != "ok" for v in r.values())]
    if failed:
        print(f"  PDFs with a failure   : {len(failed)}")
        for name, r in failed:
            bad = ", ".join(f"{k}={v}" for k, v in r.items() if v != "ok")
            print(f"    {name}: {bad}")
    print(f"  Output in             : {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
