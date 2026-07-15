#!/usr/bin/env python3
# PDF to TEI using a running GROBID server.
# Walks a folder of PDFs, posts each to GROBID, and writes the TEI XML.
# Resumable: PDFs that already have a valid TEI are skipped. Transient
# failures (timeouts, 5xx, 429) are retried with exponential backoff;
# fatal ones (4xx client errors) fail immediately.

import argparse
import time
from collections import Counter
from pathlib import Path

import requests

DEFAULT_GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

GROBID_PARAMS = {
    "consolidateHeader": "0",
    "consolidateCitations": "0",
    "includeRawCitations": "1",
    "segmentSentences": "1",   # adds <s> elements, which the extractors rely on
}

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3
BACKOFF_BASE = 5
SLEEP_EVERY_N = 10
SLEEP_DURATION = 5

TRANSIENT_STATUSES = {206, 408, 429, 500, 502, 503, 504}
FATAL_STATUSES = {400, 401, 403, 404, 415}


def is_valid_xml_output(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return False
    head = path.read_text(encoding="utf-8", errors="ignore")[:500].lower()
    return "<tei" in head or "<?xml" in head


def process_one(pdf_path: Path, out_path: Path, grobid_url: str):
    if is_valid_xml_output(out_path):
        return True, "skip"

    last_label = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with pdf_path.open("rb") as fh:
                resp = requests.post(
                    grobid_url,
                    files={"input": fh},
                    data=GROBID_PARAMS,
                    timeout=REQUEST_TIMEOUT,
                )
            code = resp.status_code

            if code == 200:
                out_path.write_text(resp.text, encoding="utf-8")
                return True, "ok"
            if code == 204:
                return False, "no_content"
            if code in FATAL_STATUSES:
                return False, f"http_{code}"
            if code in TRANSIENT_STATUSES:
                last_label = f"http_{code}"
                if attempt < MAX_RETRIES:
                    sleep_s = BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"    retry {attempt}/{MAX_RETRIES} after {code} (sleep {sleep_s}s)")
                    time.sleep(sleep_s)
                    continue
                return False, last_label
            return False, f"http_{code}"

        except requests.exceptions.Timeout:
            last_label = "timeout"
        except requests.exceptions.ConnectionError:
            last_label = "conn_error"
        except Exception as e:
            last_label = f"exc_{type(e).__name__}"

        if attempt < MAX_RETRIES:
            sleep_s = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"    retry {attempt}/{MAX_RETRIES} after {last_label} (sleep {sleep_s}s)")
            time.sleep(sleep_s)

    return False, last_label


def convert(pdf_dir, tei_dir, grobid_url=DEFAULT_GROBID_URL):
    in_dir = Path(pdf_dir)
    out_dir = Path(tei_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.is_dir():
        raise SystemExit(f"PDF folder not found: {in_dir}")

    pdfs = sorted(p for p in in_dir.iterdir() if p.suffix.lower() == ".pdf")
    print(f"Found {len(pdfs)} PDFs in {in_dir}")
    print(f"Writing TEI XML to  {out_dir}")
    print(f"Endpoint            {grobid_url}\n")

    status = Counter()
    failed = []
    for i, pdf_path in enumerate(pdfs, 1):
        out_path = out_dir / f"{pdf_path.stem}.tei.xml"
        ok, label = process_one(pdf_path, out_path, grobid_url)
        status[label] += 1
        print(f"[{i:>3}/{len(pdfs)}]  {'ok ' if ok else 'x  '}{pdf_path.name:<40s}  {label}")
        if not ok:
            failed.append((pdf_path.name, label))
        if SLEEP_EVERY_N and i % SLEEP_EVERY_N == 0 and i < len(pdfs):
            time.sleep(SLEEP_DURATION)

    print("\nStatus summary:")
    for label, n in status.most_common():
        print(f"  {label:<14s}  {n}")
    if failed:
        print(f"\n{len(failed)} files failed. Re-run to retry only those:")
        for fname, label in failed[:20]:
            print(f"  - {fname}  ({label})")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Convert a folder of PDFs to GROBID TEI XML.")
    ap.add_argument("-i", "--input", required=True, help="folder of PDF files")
    ap.add_argument("-o", "--output", required=True, help="folder for the TEI XML output")
    ap.add_argument("--grobid-url", default=DEFAULT_GROBID_URL, help="GROBID processFulltextDocument endpoint")
    args = ap.parse_args()
    convert(args.input, args.output, args.grobid_url)


if __name__ == "__main__":
    main()
