#!/usr/bin/env python3
# Flatten the per-PDF JSON output into a single CSV.
# Walks the body/, footnote/, and reference/ subfolders of a run_pipeline
# output directory, tags each row with the module that produced it
# (location = body / footnote / reference), and writes:
#   paper_id, location, target, preceding, trailing, url

import argparse
import csv
import glob
import json
import os
from pathlib import Path

# location -> glob of the per-PDF item files to read in that subfolder.
# The *_sentences.json and *_references_with_urls.json dumps are skipped.
MODULES = [
    ("body",      "*_body_urls.json"),
    ("footnote",  "*_footnotes.json"),
    ("reference", "*_url_reference_citations.json"),
]

# filename suffixes stripped to recover the paper id
SUFFIXES = [
    "_body_urls", "_footnotes",
    "_url_reference_citations", "_url_reference_sentences",
    "_references_with_urls", "_sentences",
]

FIELDNAMES = ["paper_id", "location", "target", "preceding", "trailing", "url"]


def clean_paper_id(value, fallback_file=None):
    if value:
        name = os.path.basename(str(value).strip())
    elif fallback_file:
        name = os.path.basename(fallback_file)
    else:
        return ""
    name = os.path.splitext(name)[0]
    for suf in SUFFIXES:
        if name.endswith(suf):
            name = name[:-len(suf)]
            break
    return name


def clean_url(value):
    if value is None:
        return ""
    if isinstance(value, list):
        urls = []
        for u in value:
            if u is None:
                continue
            u = str(u).strip().strip('"').strip("'")
            if u:
                urls.append(u)
        return " | ".join(urls)
    value = str(value).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    return value.strip('"').strip("'")


def load_items(data):
    if isinstance(data, list):
        return data, ""
    if isinstance(data, dict):
        return data.get("items", []), data.get("pdf_file", "")
    return [], ""


def rows_from_file(json_file, location):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    items, file_pdf = load_items(data)
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pdf_file = item.get("pdf_file", file_pdf)
        rows.append({
            "paper_id": clean_paper_id(pdf_file, fallback_file=json_file),
            "location": location,
            "target": (item.get("restored_sentence")
                       or item.get("target_sentence")
                       or item.get("original_sentence")
                       or item.get("target")
                       or ""),
            "preceding": (item.get("preceding_sentence")
                          or item.get("preceding") or ""),
            "trailing": (item.get("trailing_sentence")
                         or item.get("trailing") or ""),
            "url": clean_url(item.get("url")
                             or item.get("url_printed")
                             or item.get("urls") or ""),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Flatten module JSON output into one CSV with a location column.")
    ap.add_argument("-i", "--input", required=True,
                    help="run_pipeline output directory (contains body/, footnote/, reference/)")
    ap.add_argument("-o", "--output", required=True, help="output CSV path")
    args = ap.parse_args()

    root = Path(args.input)
    rows = []
    counts = []
    for location, pattern in MODULES:
        subdir = root / location
        if not subdir.is_dir():
            continue
        files = sorted(glob.glob(str(subdir / "**" / pattern), recursive=True))
        before = len(rows)
        for jf in files:
            rows.extend(rows_from_file(jf, location))
        counts.append((location, len(files), len(rows) - before))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    for location, nf, nr in counts:
        print(f"{location:9s}: {nf} files, {nr} rows")
    print(f"total    : {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
