#!/usr/bin/env python3
# Reference URL extractor for GROBID TEI.
# For each in-text citation whose bibliography entry carries a URL, writes the
# URL, the reference text, the citing sentence, the restored sentence (with the
# reference spliced in at the marker), and the neighbouring sentences.

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def find_tei_files(tei_dir: Path) -> List[Path]:
    for pattern, recursive in [
        ("*.tei.xml", False),
        ("*.xml",     False),
        ("*.tei.xml", True),
        ("*.xml",     True),
    ]:
        files = sorted(tei_dir.rglob(pattern) if recursive
                       else tei_dir.glob(pattern))
        if files:
            if recursive:
                print(f"    Found {len(files)} '{pattern}' files via recursive "
                      f"search under {tei_dir.name}/")
            return files
    return []


#  URL HELPERS

def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = re.sub(r"\s+", "", url)
    url = re.sub(r"(%29)+$", "", url, flags=re.IGNORECASE)
    url = url.rstrip(".,;:)>]\"'")
    return url.lower()


def is_real_url(url: str) -> bool:
    s = url.strip()
    if not s:
        return False
    if re.match(r"^(https?|ftp)://", s, re.IGNORECASE):
        return True
    if re.match(r"^www\.", s, re.IGNORECASE):
        return True
    return False


#  MARKER HELPERS

def clean_marker(text: str) -> str:
    t = (text or "").strip()
    m = re.match(r"^\[([^\[\]]+)\]$", t)
    if m:
        t = m.group(1).strip()
    t = t.strip("[],; ")
    return t


#  SENTENCE SERIALISATION

def _elem_text(elem: etree._Element) -> str:
    return "".join(elem.itertext())


def serialize_sentence(s_elem: etree._Element,
                       expand_ref_id: Optional[str] = None,
                       expansion_text: Optional[str] = None) -> str:
    parts: List[str] = []

    if s_elem.text:
        parts.append(s_elem.text)

    for child in s_elem:
        local = etree.QName(child.tag).localname

        if local == "ref" and child.get("type") == "bibr":
            child_target = child.get("target", "").lstrip("#")
            child_text   = _elem_text(child)
            if expand_ref_id and child_target == expand_ref_id and expansion_text:
                marker = clean_marker(child_text)
                parts.append(f"[{marker} {expansion_text}]")
            else:
                parts.append(child_text)

        elif local == "ref" and child.get("type") == "url":
            parts.append(child.get("target", "") or _elem_text(child))

        else:
            parts.append(_elem_text(child))

        if child.tail:
            parts.append(child.tail)

    return re.sub(r"\s+", " ", "".join(parts)).strip()


#  CONTEXT EXTRACTION  ← NEW

def get_context(s_elem: etree._Element) -> Dict[str, Any]:
    parent = s_elem.getparent()
    if parent is None:
        return {
            "preceding_sentence": None,
            "trailing_sentence":  None,
            "at_paragraph_start": True,
            "at_paragraph_end":   True,
        }

    # Collect all <s> children of this parent (same paragraph)
    siblings: List[etree._Element] = [
        c for c in parent
        if etree.QName(c.tag).localname == "s"
    ]

    try:
        idx = siblings.index(s_elem)
    except ValueError:
        # s_elem is nested, not a direct child of parent, so skip sibling navigation
        return {
            "preceding_sentence": None,
            "trailing_sentence":  None,
            "at_paragraph_start": True,
            "at_paragraph_end":   True,
        }

    preceding = (
        serialize_sentence(siblings[idx - 1])
        if idx > 0 else None
    )
    trailing = (
        serialize_sentence(siblings[idx + 1])
        if idx < len(siblings) - 1 else None
    )

    return {
        "preceding_sentence": preceding,
        "trailing_sentence":  trailing,
        "at_paragraph_start": preceding is None,
        "at_paragraph_end":   trailing is None,
    }


#  REFERENCE TEXT RECONSTRUCTION (fallback)

def reconstruct_ref_text(bib: etree._Element) -> str:
    parts: List[str] = []

    authors = []
    for persName in bib.findall(f".//{{{TEI_NS}}}persName"):
        forenames = " ".join(
            (fn.text or "").strip()
            for fn in persName.findall(f"{{{TEI_NS}}}forename")
        )
        surname_el = persName.find(f"{{{TEI_NS}}}surname")
        surname = (surname_el.text or "").strip() if surname_el is not None else ""
        name = (forenames + " " + surname).strip()
        if name:
            authors.append(name)
    if authors:
        parts.append(", ".join(authors))

    for level in ("a", "m", "s", "j"):
        title_el = bib.find(f".//{{{TEI_NS}}}title[@level='{level}']")
        if title_el is not None and title_el.text:
            parts.append(title_el.text.strip())
            break

    for tag in ("monogr", "series"):
        for level in ("j", "m", "s"):
            venue_el = bib.find(
                f".//{{{TEI_NS}}}{tag}/{{{TEI_NS}}}title[@level='{level}']"
            )
            if venue_el is not None and venue_el.text:
                parts.append(venue_el.text.strip())
                break

    date_el = bib.find(f".//{{{TEI_NS}}}date[@type='published']")
    if date_el is not None:
        year_m = re.match(r"(\d{4})", date_el.get("when", ""))
        if year_m:
            parts.append(year_m.group(1))

    ptr = bib.find(f".//{{{TEI_NS}}}ptr")
    if ptr is not None:
        url = ptr.get("target", "")
        if url:
            parts.append(url)

    return ". ".join(p for p in parts if p)


#  REFERENCE INDEX

def build_ref_index(tei_root: etree._Element) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}

    for bib in tei_root.findall(
        f".//{{{TEI_NS}}}listBibl/{{{TEI_NS}}}biblStruct"
    ):
        ref_id = bib.get(f"{{{XML_NS}}}id", "")
        if not ref_id:
            continue

        url = ""
        ptr = bib.find(f".//{{{TEI_NS}}}ptr")
        if ptr is not None:
            candidate = (ptr.get("target") or "").strip()
            if is_real_url(candidate):
                url = candidate

        if not url:
            note = bib.find(f".//{{{TEI_NS}}}note[@type='raw_reference']")
            if note is not None and note.text:
                url_m = re.search(
                    r"https?://\S+|ftp://\S+|www\.\S+",
                    note.text, re.IGNORECASE
                )
                if url_m:
                    candidate = url_m.group(0).rstrip(".,;:)>]\"'")
                    if is_real_url(candidate):
                        url = candidate

        if not url:
            continue

        note_el = bib.find(f".//{{{TEI_NS}}}note[@type='raw_reference']")
        if note_el is not None and note_el.text and note_el.text.strip():
            raw_text = re.sub(r"\s+", " ", note_el.text).strip()
        else:
            raw_text = reconstruct_ref_text(bib)

        index[ref_id] = {"ref_id": ref_id, "url": url, "raw_text": raw_text}

    return index


#  PER-FILE PROCESSING

def process_tei(tei_path: Path, corpus: str) -> List[Dict[str, Any]]:
    try:
        tree = etree.parse(str(tei_path))
    except Exception as e:
        print(f"  [!] Failed to parse {tei_path.name}: {e}")
        return []

    root      = tree.getroot()
    paper_id  = tei_path.stem.replace(".tei", "")
    ref_index = build_ref_index(root)
    if not ref_index:
        return []

    body = root.find(f".//{{{TEI_NS}}}body")
    if body is None:
        return []

    items: List[Dict[str, Any]] = []

    for s_elem in body.iter(f"{{{TEI_NS}}}s"):
        bibr_refs = s_elem.findall(f".//{{{TEI_NS}}}ref[@type='bibr']")
        if not bibr_refs:
            continue

        url_refs = [
            (ref_el.get("target", "").lstrip("#"), ref_el)
            for ref_el in bibr_refs
            if ref_el.get("target", "").lstrip("#") in ref_index
        ]
        if not url_refs:
            continue

        original = serialize_sentence(s_elem)

        # Extract paragraph context once per sentence (shared across all
        # citations in the same sentence (neighbours are the same either way)
        ctx = get_context(s_elem)

        for ref_id, ref_el in url_refs:
            entry = ref_index[ref_id]
            items.append({
                "corpus":              corpus,
                "paper_id":            paper_id,
                "reference_id":        ref_id,
                "citation_marker":     clean_marker(_elem_text(ref_el)),
                "url":                 entry["url"],
                "reference_text":      entry["raw_text"],
                "original_sentence":   original,
                "restored_sentence":   serialize_sentence(
                    s_elem,
                    expand_ref_id  = ref_id,
                    expansion_text = entry["raw_text"],
                ),
                # paragraph context
                "preceding_sentence":  ctx["preceding_sentence"],
                "trailing_sentence":   ctx["trailing_sentence"],
                "at_paragraph_start":  ctx["at_paragraph_start"],
                "at_paragraph_end":    ctx["at_paragraph_end"],
            })

    return items


#  OUTPUT HELPERS

FIELDNAMES = [
    "corpus", "paper_id", "reference_id", "citation_marker",
    "url", "reference_text",
    "original_sentence", "restored_sentence",
    "preceding_sentence", "trailing_sentence",
    "at_paragraph_start", "at_paragraph_end",
]

def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


#  PER-CORPUS RUNNER

def process_corpus(tei_dir: Path, corpus: str,
                   out_dir: Path, all_handle) -> Dict[str, Any]:

    tei_files = find_tei_files(tei_dir)
    if not tei_files:
        print(f"  [!] No TEI files found in {tei_dir}")
        print(f"      Check that the path exists and contains *.tei.xml or *.xml files")
        return {"corpus": corpus, "files": 0, "items": 0, "papers_with_citations": 0}

    print(f"\n{'='*55}")
    print(f"Corpus : {corpus}  ({len(tei_files)} files)")
    print(f"{'='*55}")

    all_items: List[Dict] = []

    for tei_path in tei_files:
        items = process_tei(tei_path, corpus)
        all_items.extend(items)
        for it in items:
            all_handle.write(json.dumps(it, ensure_ascii=False) + "\n")
        all_handle.flush()
        if items:
            print(f"  {tei_path.name:45s}  ->  {len(items):4d} citations")

    if all_items:
        _write_jsonl(out_dir / f"{corpus}_restorations.jsonl", all_items)
        _write_csv(  out_dir / f"{corpus}_restorations.csv",   all_items)

    by_paper = defaultdict(int)
    by_url   = defaultdict(set)
    for it in all_items:
        by_paper[it["paper_id"]] += 1
        by_url[it["paper_id"]].add(it["url"])

    print(f"\n  Total items     : {len(all_items)}")
    print(f"  Papers with >=1  : {len(by_paper)}")
    print(f"  Unique URLs     : {sum(len(v) for v in by_url.values())}")

    return {
        "corpus":                corpus,
        "files":                 len(tei_files),
        "items":                 len(all_items),
        "papers_with_citations": len(by_paper),
    }


def run(input_dir, output_dir, corpus=None):
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"ERROR: input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = corpus or in_dir.name

    combined_jsonl = out_dir / "all_restorations.jsonl"
    with combined_jsonl.open("w", encoding="utf-8") as all_handle:
        summary = process_corpus(in_dir, corpus, out_dir, all_handle)

    summary_path = out_dir / "restoration_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print()
    print(f"Done. {corpus}: {summary['items']} items ({summary['files']} files)")
    print(f"Output: {out_dir}")
    print(f"  {corpus}_restorations.jsonl")
    print(f"  {corpus}_restorations.csv")
    print(f"  all_restorations.jsonl")
    print(f"  restoration_summary.csv")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Extract reference URLs and restored citing sentences from GROBID TEI.")
    ap.add_argument("-i", "--input", required=True, help="directory of GROBID TEI XML files")
    ap.add_argument("-o", "--output", required=True, help="directory for the CSV/JSONL output")
    ap.add_argument("--corpus", default=None, help="corpus label written into each row (default: input dir name)")
    args = ap.parse_args()
    run(args.input, args.output, args.corpus)


if __name__ == "__main__":
    main()
