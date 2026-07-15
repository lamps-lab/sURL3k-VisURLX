#!/usr/bin/env python3

import argparse
import json, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

#  CONFIG

EXCLUDE_URL_RE = re.compile(
    r"(creativecommons\.org|crossmark\.crossref\.org"
    r"|orcid\.org|crossref\.org/dialog"
    r"|biomedcentral\.com/\d{4}-\d+/\d+/\d+$"
    r"|springer\.com/article/.*$"
    r"|journals\.plos\.org/.*article/asset)"  # PLOS supplementary file links
    , re.I)

# URL regex for extracting from reference TEXT (no hyperlink needed)
_TEXT_URL_RE = re.compile(
    r'(?:https?://[^\s\]\[<>"\']+)'   # full http(s) URL
    r'|(?:doi\.org/[^\s\]\[<>"\']+)'  # doi.org without scheme
    r'|(?:doi:\s*10\.\d{4,}/[^\s\]\[<>"\']+)',  # doi:10.xxxx/...
    re.I
)


# ─── Text utilities ───────────────────────────────────────────────

def fulltext_block_text(block: Dict) -> str:
    if block.get("type") != 0:  # type 0 = text block
        return ""
    lines = []
    for line in block.get("lines", []):
        line_text = "".join(span["text"] for span in line.get("spans", []))
        lines.append(line_text)
    return _rejoin_lines(lines)


def box_full_text(box: Dict) -> str:
    lines = []
    for line in box.get("textlines", []):
        line_text = "".join(span["text"] for span in line.get("spans", []))
        lines.append(line_text)
    return _rejoin_lines(lines)


def _rejoin_lines(lines: List[str]) -> str:
    out = ""
    for i, line in enumerate(lines):
        if i == 0:
            out = line
        elif out.endswith("-"):
            out = out[:-1] + line.lstrip()
        else:
            out = out.rstrip() + " " + line.lstrip()
    return out


def block_bbox(block: Dict) -> Tuple[float, float, float, float]:
    bb = block.get("bbox", [0, 0, 0, 0])
    if isinstance(bb, list) and len(bb) >= 4:
        return tuple(bb[:4])
    return (0, 0, 0, 0)


def box_bbox(box: Dict) -> Tuple[float, float, float, float]:
    return (box.get("x0", 0), box.get("y0", 0), box.get("x1", 0), box.get("y1", 0))


def bbox_y_overlap(a: Tuple, b: Tuple, margin: float = 3.0) -> bool:
    return a[1] - margin <= b[3] and a[3] + margin >= b[1]


def is_good_url(url: str) -> bool:
    if not url:
        return False
    if EXCLUDE_URL_RE.search(url):
        return False
    return True


def clean_url(url: str) -> str:
    url = url.strip()
    url = re.sub(r'\.\s*Accessed\s*\d.*$', '', url, flags=re.I)
    url = url.rstrip('.,;:)]}>\'\" ')
    return url


# ─── Citation markers ────────────────────────────────────────────

_NUM_CITE = re.compile(r'\[(\d+(?:[,;\s\-–]+\d+)*)\]')


def expand_numeric(marker_inner: str) -> List[int]:
    nums = []
    for part in re.split(r'[,;\s]+', marker_inner.strip()):
        part = part.strip()
        m = re.match(r'^(\d+)[–\-](\d+)$', part)
        if m:
            nums.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


def find_citations(text: str) -> List[Dict]:
    hits = []
    for m in _NUM_CITE.finditer(text):
        hits.append({
            "marker": m.group(0), "refs": expand_numeric(m.group(1)),
            "start": m.start(), "end": m.end(), "type": "numeric",
        })
    return hits


# ─── Sentence splitting ──────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z\[\(])')


def sentence_containing(text: str, pos: int) -> str:
    boundaries = [0]
    for m in _SENT_SPLIT.finditer(text):
        boundaries.append(m.end())
    boundaries.append(len(text))
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if s <= pos < e:
            return text[s:e].strip()
    return text.strip()


def prev_and_next_sentence(text: str, pos: int) -> Tuple[Optional[str], Optional[str]]:
    boundaries = [0]
    for m in _SENT_SPLIT.finditer(text):
        boundaries.append(m.end())
    boundaries.append(len(text))
    sents = [text[boundaries[i]:boundaries[i + 1]].strip()
             for i in range(len(boundaries) - 1)
             if boundaries[i + 1] > boundaries[i]]
    for idx, (s, e) in enumerate(zip(boundaries, boundaries[1:])):
        if s <= pos < e:
            return (sents[idx - 1] if idx > 0 else None,
                    sents[idx + 1] if idx < len(sents) - 1 else None)
    return None, None


#  PHASE 1 ,  Reference extraction (IMPROVED)

def find_reference_start_page(toc: List, page_count: int) -> int:
    for _level, title, page in toc:
        if re.search(r'\breferences?\b', title, re.I):
            return int(page)
    return max(1, int(page_count * 0.85))


def _extract_urls_from_links(page: Dict, target_bbox: Tuple) -> List[str]:
    urls = []
    for lk in page.get("links", []):
        if lk.get("kind") != 2:
            continue
        uri = lk.get("uri", "")
        if not uri or not is_good_url(uri):
            continue
        link_bbox = tuple(lk.get("from", [0, 0, 0, 0]))
        if bbox_y_overlap(link_bbox, target_bbox):
            urls.append(clean_url(uri))
    return urls


def _extract_urls_from_text(text: str) -> List[str]:
    urls = []
    for m in _TEXT_URL_RE.finditer(text):
        url = clean_url(m.group(0))
        # Normalise doi: prefix
        if url.lower().startswith("doi:"):
            doi_part = url[4:].strip()
            url = "https://doi.org/" + doi_part
        elif url.lower().startswith("doi.org/"):
            url = "https://" + url
        if is_good_url(url) and len(url) > 10:
            urls.append(url)
    return urls


def extract_references_from_fulltext(pages: List[Dict], ref_start: int) -> Dict[int, Dict]:
    refs: Dict[int, Dict] = {}

    for page in sorted(pages, key=lambda p: p["page_number"]):
        pnum = page["page_number"]
        if pnum < ref_start:
            continue

        # ── Collect candidate reference entries from fulltext blocks ──────
        candidates = []  # (ref_num, text, bbox, source)

        for block in page.get("fulltext", []):
            if block.get("type") != 0:
                continue
            text = fulltext_block_text(block)
            if not text or len(text.strip()) < 15:
                continue
            bb = block_bbox(block)

            m = re.match(r'^\s*(\d+)[.\)]\s*', text)
            if m:
                num = int(m.group(1))
                ref_text = text[m.end():].strip()
                candidates.append((num, ref_text, bb, text, "fulltext"))

        # ── Supplement from boxes list-items (if not already captured) ────
        seen_nums = {c[0] for c in candidates}
        for box in page.get("boxes", []):
            if box["boxclass"] != "list-item":
                continue
            text = box_full_text(box)
            m = re.match(r'^\s*(\d+)[.\)]\s*', text)
            if not m:
                continue
            num = int(m.group(1))
            if num in seen_nums:
                continue
            ref_text = text[m.end():].strip()
            bb = box_bbox(box)
            candidates.append((num, ref_text, bb, text, "box"))

        # ── For each candidate, find URLs ─────────────────────────────────
        for num, ref_text, bb, raw, source in candidates:
            if num in refs:
                continue  # already found from a previous page

            # Strategy 1: kind=2 link annotations overlapping this block
            link_urls = _extract_urls_from_links(page, bb)

            # Strategy 2: regex extraction from the reference text itself
            text_urls = _extract_urls_from_text(ref_text)

            # Merge, preferring DOI URLs
            all_urls = []
            seen = set()
            for u in link_urls + text_urls:
                norm = u.lower().rstrip("/")
                if norm not in seen:
                    seen.add(norm)
                    all_urls.append(u)

            # Pick the best URL (prefer DOI, then NCBI, then other)
            url = None
            for u in all_urls:
                if "doi.org" in u.lower():
                    url = u
                    break
            if not url:
                for u in all_urls:
                    if "ncbi.nlm" in u.lower() or "pubmed" in u.lower():
                        url = u
                        break
            if not url and all_urls:
                url = all_urls[0]

            refs[num] = {
                "num": num,
                "text": ref_text,
                "url": url,
                "all_urls": all_urls,
                "raw": raw,
                "page": pnum,
                "source": source,
            }

    return refs


#  PHASE 2+3 ,  Citation detection and sentence restoration

BODY_CLASSES = {"text", "caption"}


def restore_citation(sentence: str, marker: str, ref: Dict) -> str:
    ref_inline = f"{marker}[{ref['text']}]"
    return sentence.replace(marker, ref_inline, 1)


def process_body_pages(
    pages: List[Dict],
    ref_start: int,
    refs: Dict[int, Dict],
) -> List[Dict]:
    citations: List[Dict] = []

    for page in sorted(pages, key=lambda p: p["page_number"]):
        pnum = page["page_number"]
        if pnum >= ref_start:
            continue

        for box in page.get("boxes", []):
            if box["boxclass"] not in BODY_CLASSES:
                continue

            full_text = box_full_text(box)
            if not full_text.strip():
                continue

            hits = find_citations(full_text)
            if not hits:
                continue

            for hit in hits:
                sentence = sentence_containing(full_text, hit["start"])
                prev_s, next_s = prev_and_next_sentence(full_text, hit["start"])

                for ref_num in hit["refs"]:
                    ref = refs.get(ref_num)
                    if ref is None or not ref.get("url"):
                        continue

                    restored = restore_citation(sentence, hit["marker"], ref)
                    citations.append({
                        "page": pnum,
                        "ref_num": ref_num,
                        "url": ref["url"],
                        "all_urls": ref.get("all_urls", []),
                        "marker": hit["marker"],
                        "original_sentence": sentence,
                        "restored_sentence": restored,
                        "ref_text": ref["text"],
                        "preceding_sentence": prev_s,
                        "trailing_sentence": next_s,
                        "at_paragraph_start": prev_s is None,
                        "at_paragraph_end": next_s is None,
                    })

    return citations


#  File I/O

def load_paper(paper_dir: Path) -> Tuple[List[Dict], List, int]:
    all_pages, toc, page_count = [], [], 0
    for jf in sorted(paper_dir.iterdir()):
        if jf.suffix != ".json":
            continue
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"    [WARN] {jf.name}: {e}")
            continue
        if not toc:
            toc = d.get("toc", [])
            page_count = d.get("page_count", 0)
        all_pages.extend(d.get("pages", []))
    return all_pages, toc, page_count


def process_paper(paper_dir: Path, output_dir: Path, all_jsonl_path: Path):
    stem = paper_dir.name
    pages, toc, page_count = load_paper(paper_dir)
    if not pages:
        print(f"  [SKIP] no pages: {stem}")
        return {}

    ref_start = find_reference_start_page(toc, page_count)
    refs = extract_references_from_fulltext(pages, ref_start)
    url_refs = {k: v for k, v in refs.items() if v.get("url")}
    citations = process_body_pages(pages, ref_start, refs)
    url_citations = [c for c in citations if c.get("url")]

    print(f"  {stem}: {len(refs)} refs, {len(url_refs)} with URLs, "
          f"ref pages {ref_start}–{page_count}, "
          f"{len(url_citations)} restored citations")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Citation JSON (evaluator-compatible format)
    cit_out = []
    for c in url_citations:
        cit_out.append({
            "reference_entry": c.get("ref_text", ""),
            "restored_sentence": c.get("restored_sentence", ""),
            "urls": [c["url"]],
            "citation_marker": c.get("marker", ""),
            "page": c.get("page", ""),
            "original_sentence": c.get("original_sentence", ""),
            "preceding_sentence": c.get("preceding_sentence"),
            "trailing_sentence": c.get("trailing_sentence"),
        })
    (output_dir / f"{stem}_url_reference_citations.json").write_text(
        json.dumps(cit_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reference JSON (evaluator-compatible format)
    ref_out = []
    for k, v in url_refs.items():
        ref_out.append({
            "full_text": v.get("raw", v.get("text", "")),
            "urls": v.get("all_urls", [v["url"]] if v.get("url") else []),
            "index": str(v.get("num", k)),
        })
    (output_dir / f"{stem}_references_with_urls.json").write_text(
        json.dumps(ref_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Append to JSONL
    with open(all_jsonl_path, "a", encoding="utf-8") as fh:
        for c in url_citations:
            row = {"paper": stem, **c}
            # Remove non-serialisable items
            for k in list(row.keys()):
                if row[k] is None:
                    row[k] = ""
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "paper": stem,
        "ref_page_start": ref_start,
        "total_refs": len(refs),
        "url_refs": len(url_refs),
        "total_citations": len(citations),
        "url_citations": len(url_citations),
    }


#  Main

def main():
    ap = argparse.ArgumentParser(description="Extract reference URLs from PyMuPDF-Layout per-page JSON.")
    ap.add_argument("-i", "--input", required=True, help="directory of per-PDF subfolders of per-page JSON")
    ap.add_argument("-o", "--output", required=True, help="output directory for the JSON results")
    args = ap.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    all_jsonl = output_root / "_ALL.citations.jsonl"
    summary_out = output_root / "_SUMMARY.json"
    all_jsonl.write_text("")

    paper_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
    print(f"Found {len(paper_dirs)} paper folder(s) in {input_root}\n")

    summary = []
    for paper_dir in tqdm(paper_dirs, desc="Papers"):
        try:
            stats = process_paper(paper_dir, output_root, all_jsonl)
            if stats:
                summary.append(stats)
                summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {paper_dir.name}: {e}")
            traceback.print_exc()

    total = sum(s.get("url_citations", 0) for s in summary)
    print(f"\n{'─' * 60}")
    print(f"Papers processed           : {len(summary)}")
    print(f"Total URL citations restored: {total}")
    print(f"Output directory           : {output_root}")


if __name__ == "__main__":
    main()
