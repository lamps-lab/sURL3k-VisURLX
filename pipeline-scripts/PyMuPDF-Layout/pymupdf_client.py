#!/usr/bin/env python3
# PDF to PyMuPDF-Layout output.
# For each PDF writes two things:
#   - per-page JSON:  <json_out>/<stem>/<stem>_NN.json   (footnote and reference input)
#   - layout text:    <text_out>/<stem>.layout.txt        (body input)
# The text file joins pages with "=== PAGE N ===" markers, which the body
# extractor uses to split pages.

import argparse
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import pymupdf.layout  # older versions need this before pymupdf4llm; newer bundle layout
except ImportError:
    pass
import pymupdf
import pymupdf4llm


def page_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def convert_one(task):
    pdf_path_str, json_root_str, text_root_str = task
    pdf_path = Path(pdf_path_str)
    json_root = Path(json_root_str)
    text_root = Path(text_root_str)
    stem = pdf_path.stem
    start = time.perf_counter()

    result = {"pdf": pdf_path.name, "status": "failed", "pages": 0,
              "pages_processed": 0, "seconds": 0.0, "error": ""}

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        result["error"] = f"cannot open PDF: {e}"
        return result

    try:
        total = doc.page_count
        result["pages"] = total
        pad = len(str(total)) if total else 1

        json_dir = json_root / stem
        json_paths = [json_dir / f"{stem}_{p:0{pad}d}.json" for p in range(1, total + 1)]
        text_path = text_root / f"{stem}.layout.txt"

        if text_path.exists() and text_path.stat().st_size > 0 and all(page_done(f) for f in json_paths):
            result["status"] = "skipped"
            doc.close()
            return result

        json_dir.mkdir(parents=True, exist_ok=True)
        text_root.mkdir(parents=True, exist_ok=True)

        processed = 0
        text_chunks = []
        for i in range(total):
            page_no = i + 1
            jp = json_paths[i]
            if not page_done(jp):
                page_json = pymupdf4llm.to_json(doc, pages=[i])
                tmp = jp.with_suffix(".json.tmp")
                tmp.write_text(page_json, encoding="utf-8")
                tmp.replace(jp)
                processed += 1
            page_text = pymupdf4llm.to_text(doc, pages=[i], header=False, footer=False)
            text_chunks.append(f"=== PAGE {page_no} ===\n{page_text}")

        text_path.write_text("\n".join(text_chunks), encoding="utf-8")
        doc.close()

        result["pages_processed"] = processed
        result["status"] = "completed"
        return result
    except Exception as e:
        try:
            doc.close()
        except Exception:
            pass
        result["error"] = f"error: {e}"
        return result
    finally:
        result["seconds"] = round(time.perf_counter() - start, 3)
        gc.collect()


def convert(pdf_dir, json_out, text_out, workers=1):
    in_dir = Path(pdf_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"PDF folder not found: {in_dir}")
    Path(json_out).mkdir(parents=True, exist_ok=True)
    Path(text_out).mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in in_dir.glob("*.pdf") if p.is_file())
    print(f"Found {len(pdfs)} PDFs in {in_dir}")
    print(f"JSON out: {json_out}")
    print(f"Text out: {text_out}\n")
    if not pdfs:
        return

    tasks = [(str(p), str(json_out), str(text_out)) for p in pdfs]
    rows = []
    if workers <= 1:
        for t in tasks:
            rows.append(convert_one(t))
            r = rows[-1]
            print(f"  {r['pdf']:40s} {r['status']:9s} pages={r['pages']} new={r['pages_processed']} {r['seconds']}s")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(convert_one, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r)
                print(f"  {r['pdf']:40s} {r['status']:9s} pages={r['pages']} new={r['pages_processed']} {r['seconds']}s")

    done = sum(1 for r in rows if r["status"] in ("completed", "skipped"))
    print(f"\nConverted {done}/{len(rows)} PDFs")


def main():
    ap = argparse.ArgumentParser(description="Convert PDFs to PyMuPDF-Layout per-page JSON and layout text.")
    ap.add_argument("-i", "--input", required=True, help="folder of PDFs")
    ap.add_argument("--json-out", required=True, help="output folder for per-page JSON (per-PDF subfolders)")
    ap.add_argument("--text-out", required=True, help="output folder for <stem>.layout.txt")
    ap.add_argument("--workers", type=int, default=1, help="PDFs converted in parallel")
    args = ap.parse_args()
    convert(args.input, args.json_out, args.text_out, args.workers)


if __name__ == "__main__":
    main()
