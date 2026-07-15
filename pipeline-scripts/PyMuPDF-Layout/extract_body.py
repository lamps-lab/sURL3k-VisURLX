#!/usr/bin/env python3

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#  HARDCODED PATHS ,  edit these

RESUME     = True


#  URL REGEX (unchanged from v3)

_URL_BODY = r"[^\s<>\"'\)\]\}]"
_SCHEME_URL   = r"(?:https?|ftps?|sftp|s3|gs|az)\s*:?\s*//" + _URL_BODY + r"+"
_WWW_URL      = r"www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/" + _URL_BODY + r"*)?"
_DOI_PREFIXED = r"(?:doi:|(?:https?://)?(?:dx\.)?doi\.org/)10\.\d{3,}/" + _URL_BODY + r"+"
_DOI_BARE     = r"(?<![A-Za-z0-9])10\.\d{4,}/" + _URL_BODY + r"{3,}"

URL_REGEX = re.compile(
    r"(?P<url>" + _SCHEME_URL + r"|" + _DOI_PREFIXED + r"|" + _WWW_URL + r"|" + _DOI_BARE + r")",
    re.IGNORECASE,
)
_TRAILING_PUNCT = ".,;:!?)]}\u2019\u201d>"

def clean_url(url: str) -> str:
    while url and url[-1] in _TRAILING_PUNCT:
        url = url[:-1]
    return url


#  URL LINE-WRAP RECOVERY (unchanged from v3)

_URL_WRAP_RE = re.compile(
    r"(?P<head>(?:https?:\s*/+|ftps?://|sftp://|s3://|gs://|az://|www\.|doi:|doi\.org/|10\.\d+/)"
    r"[^\s\n]*[/\-~=?&%_.,])"
    r"\n[ \t]*"
    r"(?P<tail>[a-z0-9/~%._\-][^\s\n]*)",
    re.IGNORECASE,
)

def recover_wrapped_urls(text: str) -> str:
    for _ in range(5):
        new = _URL_WRAP_RE.sub(lambda m: m.group("head") + m.group("tail"), text)
        if new == text:
            break
        text = new
    return text


#  FOOTNOTE DETECTION ,  v4 expanded
#
# v3 caught: `> 9 http://...`, `> 9The package...`
# v3 MISSED:
#   - `8 http://mizar.org`         (no `>` prefix)
#   - `> 2http://...`              (no space between digit and URL)
#   - `> ∗ faeder@pitt.edu`        (symbol marker instead of digit)
#   - `> 3http://archive.stsci.edu` (also no space, multiple instances)

# Line-level: starts with footnote marker (used inside paragraphs)
_FOOTNOTE_LINE_RE = re.compile(
    r"^[ \t]*"
    r"(?:"
    # > prefix with digit OR footnote symbol
    r">\s*(?:\d{1,3}|[\u2217\u2020\u2021*†‡§¶∗])"
    r"|"
    # digit + space + URL (no > prefix) ,  common in some PyMuPDF outputs
    r"\d{1,3}\s+(?=https?://|www\.|doi:)"
    r")"
)

def is_footnote_line(line: str) -> bool:
    return bool(_FOOTNOTE_LINE_RE.match(line.lstrip()))


# Paragraph-level: whole paragraph is just a footnote (number + URL, no prose)
_FOOTNOTE_PARA_RE = re.compile(
    r"^"
    r"(?:>\s*)?"                                   # optional > prefix
    r"(?:\d{1,3}|[\u2217\u2020\u2021*†‡§¶∗])"      # number or symbol
    r"\s*"                                          # optional space
    r"(?:https?://|www\.|doi:|10\.\d)"             # then directly a URL
    r"\S+"
    r"\s*\.?\s*$",
)

def is_footnote_paragraph(para_normalized: str) -> bool:
    return bool(_FOOTNOTE_PARA_RE.match(para_normalized))


#  AUTHOR AFFILIATION DETECTION ,  new in v4

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_AFFIL_START_RE = re.compile(
    r"^[ \t]*(?:Department|Center|Centre|School|Faculty|Institute|"
    r"Laboratory|Lab|University|College|Division|Group|Office|"
    r"National|Federal|Royal)\b",
    re.IGNORECASE,
)

def is_affiliation_line(line: str) -> bool:
    if _AFFIL_START_RE.match(line):
        return True
    if _EMAIL_RE.search(line):
        stripped = line.rstrip()
        # No sentence-ending punctuation = likely a contact line, not prose
        if stripped and stripped[-1] not in ".!?":
            return True
        # Multiple emails = definitely a contact block
        if len(_EMAIL_RE.findall(line)) > 1:
            return True
    return False


#  REFERENCES / ACKS CUTOFFS (unchanged from v3)

_REF_HEADING_RE = re.compile(
    r"^[ \t]*"
    r"(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:References(?:\s+(?:and\s+Notes|Cited))?"
    r"|Bibliography"
    r"|Works\s+Cited"
    r"|Literature\s+Cited"
    r"|Cited\s+References"
    r"|Reference\s+List"
    r"|Sources)"
    r"[ \t]*:?[ \t]*\.?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

_ACK_HEADING_RE = re.compile(
    r"^[ \t]*"
    r"(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:Acknowledgments?"
    r"|Acknowledgements?"
    r"|Funding"
    r"|Funding\s+Information"
    r"|Author\s+Contributions)"
    r"[ \t]*:?[ \t]*\.?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

def find_body_end_offset(text: str) -> Optional[int]:
    refs_pos = None
    for m in _REF_HEADING_RE.finditer(text):
        refs_pos = m.start()
        break

    threshold = int(len(text) * 0.65)
    ack_pos = None
    for m in _ACK_HEADING_RE.finditer(text):
        if m.start() >= threshold:
            ack_pos = m.start()
            break

    cands = [p for p in (refs_pos, ack_pos) if p is not None]
    return min(cands) if cands else None


#  REFERENCE-ENTRY PATTERN (unchanged from v3)

_REF_ENTRY_START_RE = re.compile(r"^[ \t]*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.\s+[A-Z])")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

def is_reference_entry_paragraph(paragraph: str) -> bool:
    return bool(_REF_ENTRY_START_RE.match(paragraph)) and bool(_YEAR_RE.search(paragraph))


#  BOILERPLATE BLOCKLIST ,  v4 narrowed (don't drop W3 URLs with paths)
#
# v3 dropped `http://www.w3.org/2004/02/skos/core#` which appears in gold.
# v4 narrows the W3 patterns to only drop bare namespace URLs without paths.

_BOILERPLATE = [
    re.compile(r"^https?://creativecommons\.org/licenses/?\S*$", re.IGNORECASE),
    re.compile(r"^(?:https?://)?(?:www\.)?springerlink\.com/?$", re.IGNORECASE),
    re.compile(r"^https?://link\.springer\.com/?$", re.IGNORECASE),
    re.compile(r"^https?://(?:www\.)?ieee\.org/?$", re.IGNORECASE),
    re.compile(r"^https?://(?:www\.)?acm\.org/?$", re.IGNORECASE),
    re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)?10\.1145/0+\.0+$", re.IGNORECASE),
    re.compile(r"^https?://(?:www\.)?arxiv\.org/?$", re.IGNORECASE),
    re.compile(r"^https?://(?:www\.)?example\.(?:com|org|net)(?:/\S*)?$", re.IGNORECASE),
    re.compile(r"^https?://doi\.acm\.org/10\.1145/0+\.0+$", re.IGNORECASE),
    # v4: W3 only matches if ROOT (no path) ,  relaxed from v3
    re.compile(r"^https?://(?:www\.)?w3\.org/?$", re.IGNORECASE),
]

def is_boilerplate_url(url: str) -> bool:
    return any(pat.match(url) for pat in _BOILERPLATE)


#  SCHEME NORMALIZATION + TRUNCATION FILTER (unchanged)

_BROKEN_SCHEME_RE = re.compile(
    r"^(https?|ftps?|s?ftp|sftp|s3|gs|az)\s*:?\s*//(.+)",
    re.IGNORECASE | re.DOTALL,
)
def normalise_url_scheme(url: str) -> str:
    if not url:
        return url
    m = _BROKEN_SCHEME_RE.match(url.strip())
    return f"{m.group(1).lower()}://{m.group(2)}" if m else url

_TRUNCATED_URL_RE = re.compile(r"^(?:https?|ftps?|sftp|s3|gs|az)://(?:www\.?)?$", re.IGNORECASE)
def is_obvious_truncation(url: str) -> bool:
    if _TRUNCATED_URL_RE.match(url):
        return True
    if re.match(r"^https?://w{2,3}\.?$", url, re.IGNORECASE):
        return True
    m = re.match(r"^(?:https?|ftps?|sftp)://([^/?#]+)", url, re.IGNORECASE)
    if m and len(m.group(1).replace(".", "").replace("-", "")) < 3:
        return True
    return False


#  URL FINDER (unchanged interface)

def find_urls(text: str) -> List[Tuple[str, int, int]]:
    out = []
    for m in URL_REGEX.finditer(text):
        cleaned = clean_url(m.group("url"))
        if not cleaned:
            continue
        cleaned = normalise_url_scheme(cleaned)
        if is_boilerplate_url(cleaned):
            continue
        if is_obvious_truncation(cleaned):
            continue
        out.append((cleaned, m.start(), m.start() + len(cleaned)))
    return out


#  SENTENCE SEGMENTATION ,  v4 EXPANDED START CLASS
#
# v3 used: `(?=[A-Z0-9"'(\[])` for the sentence-start look-ahead.
# v3 BUG: didn't split `calculation. |D| was estimated...` because `|`
# wasn't in the start class. Many math-prose papers start sentences with
# math notation like `|x|`, Greek letters, bullets, special quotes.
#
# v4 expanded start class:
#   - A-Z 0-9 " ' ( [        (v3)
#   - |                       (math/code notation: |D|)
#   - “ ‘                     (typographic quotes)
#   - • – ,                    (bullets, en/em-dashes opening clause)
#   - Greek upper and lower   (Λ Φ μ θ etc. for math papers)
#   - ∗ † ‡                   (footnote markers when prose continues)

_ABBR = [
    "Fig", "Figs", "Eq", "Eqs", "Sec", "Tab", "Tabs", "Ref", "Refs",
    "Vol", "Vols", "No", "Nos", "pp", "p", "ed", "eds", "edn",
    "et al", "i.e", "e.g", "cf", "vs", "viz", "etc",
    "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Jr", "Sr",
    "U.S", "U.K", "E.U", "U.S.A",
]
_SENT = "\x01ABBR\x01"

# v4: expanded sentence-start lookahead
_SENT_START = (
    r"[A-Z0-9\"'(\[\|"          # ascii + pipe
    r"\u201c\u201d\u2018\u2019"  # typographic quotes
    r"\u2022\u2013\u2014"        # bullet, en/em-dash
    r"\u2217\u2020\u2021"        # asterisk operator, dagger, double-dagger
    r"\u0391-\u03A9\u03B1-\u03C9" # Greek upper + lower
    r"]"
)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=" + _SENT_START + r")")

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    p = text
    for a in _ABBR:
        p = re.sub(r"\b" + re.escape(a) + r"\.", a + _SENT, p)
    parts = _SENT_SPLIT_RE.split(p)
    return [s.replace(_SENT, ".").strip() for s in parts if s.replace(_SENT, ".").strip()]

def find_sentence_with_url(sentences, url):
    for i, s in enumerate(sentences):
        if url in s:
            return i
    return None


#  LAYOUT-TXT PAGE PARSING (unchanged)

_PAGE_MARKER_RE = re.compile(r"^=== PAGE (\d+) ===\s*$", re.MULTILINE)

def split_layout_text_to_pages(text: str) -> List[str]:
    matches = list(_PAGE_MARKER_RE.finditer(text))
    if matches:
        pages = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            pages.append(text[start:end])
        return pages
    if "\x0c" in text:
        return [p for p in text.split("\x0c")]
    return [text]


#  PARAGRAPH SPLITTING ,  v4 with URL-only para merging
#
# v3 BUG: when PyMuPDF Layout puts a URL on its own line surrounded by
# blank lines, our paragraph splitter creates a paragraph that IS just
# the URL. Then the "target_sentence" for that URL is the URL itself,
# not the prose sentence that introduced it. v4: merge URL-only
# paragraphs with the previous paragraph so the introducing prose is
# kept as context.

_PARA_RE = re.compile(r"\n[ \t]*\n+")

# A paragraph is "URL-only" if it's JUST a URL with optional trailing
# punctuation. CRITICAL: do NOT allow a footnote marker prefix here , 
# `> 9 http://...` is a footnote, not a URL-only body paragraph, and
# merging it with the previous paragraph would smuggle the footnote
# URL into body context. Footnote-marked URL paragraphs are caught and
# dropped by is_footnote_paragraph() / is_footnote_line() instead.
_URL_ONLY_PARA_RE = re.compile(
    r"^"
    r"(?:https?://|www\.|doi:|10\.\d)"               # URL prefix only (NO marker)
    r"\S+"
    r"\s*\.?\s*$",
)

def is_url_only_paragraph(para_normalized: str) -> bool:
    return bool(_URL_ONLY_PARA_RE.match(para_normalized))

def split_and_merge_paragraphs(page_text: str) -> List[str]:
    raw = [p.strip() for p in _PARA_RE.split(page_text) if p.strip()]
    merged: List[str] = []
    for p in raw:
        normalized = re.sub(r"\s+", " ", p).strip()
        if merged and is_url_only_paragraph(normalized):
            # Glue this URL-only paragraph onto the previous one with a
            # single space ,  preserves preceding prose as the URL's context.
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def normalize_paragraph(p: str) -> str:
    return re.sub(r"\s+", " ", p).strip()


#  PER-FILE PROCESSING

def process_layout_txt(txt_path: Path, out_dir: Path) -> Dict[str, Any]:
    pdf_stem = txt_path.name.replace(".layout.txt", "")
    if pdf_stem == txt_path.name:
        pdf_stem = txt_path.stem
    pdf_name = pdf_stem + ".pdf"

    text = txt_path.read_text(encoding="utf-8", errors="replace")
    text = recover_wrapped_urls(text)

    body_end = find_body_end_offset(text)
    body_text = text[:body_end] if body_end is not None else text

    pages = split_layout_text_to_pages(body_text)
    all_items: List[Dict[str, Any]] = []

    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        # v4: split + merge URL-only paragraphs in one pass
        for para_raw in split_and_merge_paragraphs(page_text):
            # Drop footnote lines INSIDE the paragraph
            kept = []
            for ln in para_raw.split("\n"):
                if is_footnote_line(ln):
                    continue
                if is_affiliation_line(ln):  # v4: drop affiliation lines
                    continue
                kept.append(ln)
            if not kept:
                continue
            para = normalize_paragraph("\n".join(kept))
            if not para:
                continue

            # v4: also catch paragraphs that ARE a footnote
            if is_footnote_paragraph(para):
                continue

            # Existing v3 check
            if is_reference_entry_paragraph(para):
                continue

            url_hits = find_urls(para)
            if not url_hits:
                continue

            sentences = split_sentences(para)
            for raw_url, _s, _e in url_hits:
                target_idx = find_sentence_with_url(sentences, raw_url)
                if target_idx is None:
                    target, preceding, trailing = para, None, None
                    at_start, at_end = True, True
                else:
                    target = sentences[target_idx]
                    preceding = sentences[target_idx - 1] if target_idx > 0 else None
                    trailing = (sentences[target_idx + 1]
                                if target_idx + 1 < len(sentences) else None)
                    at_start = target_idx == 0
                    at_end = target_idx == len(sentences) - 1

                all_items.append({
                    "pdf_file": pdf_name,
                    "page": page_idx,
                    "url_printed": raw_url,
                    "target_sentence": target,
                    "preceding_sentence": preceding,
                    "trailing_sentence": trailing,
                    "at_paragraph_start": at_start,
                    "at_paragraph_end": at_end,
                    "url_lines_joined": False,
                    "url_span_pages": False,
                })

    out_path = out_dir / f"{pdf_stem}_body_urls.json"
    out_path.write_text(
        json.dumps({"pdf_file": pdf_name, "items": all_items},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"pdf_file": pdf_name, "n_items": len(all_items),
            "body_end_at": body_end, "n_pages": len(pages)}


def rebuild_aggregate(out_dir: Path) -> int:
    agg = out_dir / "_ALL.url_footnotes.jsonl"
    per_pdf = sorted(out_dir.glob("*_body_urls.json"))
    n = 0
    with agg.open("w", encoding="utf-8") as f:
        for jp in per_pdf:
            try:
                payload = json.loads(jp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for it in payload.get("items", []):
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract body URLs and context from PyMuPDF-Layout .layout.txt files.")
    ap.add_argument("-i", "--input", required=True, help="directory of *.layout.txt files")
    ap.add_argument("-o", "--output", required=True, help="output directory for the JSON results")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        sys.exit(f"ERROR: input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        sys.exit(f"ERROR: no *.txt files in {input_dir}")

    if RESUME:
        done_stems = {p.stem.replace("_body_urls", "")
                      for p in output_dir.glob("*_body_urls.json")}
        before = len(txt_files)
        txt_files = [f for f in txt_files
                     if f.name.replace(".layout.txt", "").replace(".txt", "") not in done_stems]
        print(f"[resume] {before} files, {len(done_stems)} done, "
              f"{len(txt_files)} remaining")

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Processing {len(txt_files)} layout-txt file(s)\n")

    t_start = time.time()
    summary, failures = [], []
    for i, txt in enumerate(txt_files, start=1):
        try:
            res = process_layout_txt(txt, output_dir)
            summary.append(res)
            print(f"  [{i:4d}/{len(txt_files)}] {res['pdf_file']}: "
                  f"{res['n_items']} items, {res['n_pages']} pages")
        except Exception as e:
            import traceback
            print(f"[!] {txt.name}: {e}")
            traceback.print_exc()
            failures.append(txt.name)

    elapsed = time.time() - t_start
    total = sum(s["n_items"] for s in summary)
    print(f"\n[done] {len(summary)} files in {elapsed:.1f}s | {total} items")
    if failures:
        print(f"[done] Failures: {len(failures)} ,  {failures[:5]}")

    n_agg = rebuild_aggregate(output_dir)
    print(f"[done] Aggregate: {output_dir / '_ALL.url_footnotes.jsonl'} ({n_agg} items)")


if __name__ == "__main__":
    main()
