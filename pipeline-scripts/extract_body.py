#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# General text / Markdown helpers
# ---------------------------------------------------------------------------

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00ad": "", "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
}

REF_HEADINGS = {
    "references", "reference", "bibliography", "works cited", "literature cited",
    "reference list", "list of references", "sources", "citations"
}

BACKMATTER_HEADINGS = {
    "acknowledgements", "acknowledgments", "funding", "author contributions",
    "competing interests", "conflict of interest", "additional information",
    "supplementary information", "data availability", "code availability",
}

# A deliberately broad URL recognizer.  Filtering below decides body-vs-nonbody.
URL_RE = re.compile(r"""
(?<![\w@])
(
  (?:https?|ftp|ftps|sftp|s3|gs|az)\s*:?\s*/\s*/\s*[^\s<>{}\[\]"'`]+ |
  www\.[^\s<>{}\[\]"'`]+ |
  doi\s*:\s*10\.\d{4,9}/[^\s<>{}\[\]"'`]+ |
  10\.\d{4,9}/[^\s<>{}\[\]"'`]+ |
  (?:(?:github|gitlab|bitbucket)\.com|arxiv\.org|doi\.org)/[^\s<>{}\[\]"'`]+ |
  (?:[A-Za-z0-9][A-Za-z0-9-]{0,62}\.)+(?:com|org|net|edu|gov|info|io|ai|uk|de|fr|jp|cn|au|ca|nl|es|it|se|ch|eu|int|mil)\b(?:/[^\s<>{}\[\]"'`]*)?
)
""", re.I | re.X)

MD_LINK_RE = re.compile(r"!?\[([^\]\n]{0,300})\]\(([^)\s]+(?:\)[^\s.,;:]*)?)\)")
IMAGE_MD_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")

ABBREVIATIONS = {
    "e.g.", "i.e.", "cf.", "cfr.", "vs.", "etc.", "al.", "fig.", "figs.",
    "eq.", "eqs.", "ref.", "refs.", "sec.", "sect.", "ch.", "app.", "vol.",
    "no.", "pp.", "p.", "ed.", "eds.", "dr.", "mr.", "mrs.", "ms.", "prof.",
    "jr.", "sr.", "inc.", "ltd.", "co.", "u.s.", "u.k.", "u.s.a.", "ph.d.",
}

@dataclass
class MdFile:
    file: str
    paper_id: str
    text: str

@dataclass
class Paragraph:
    text: str
    start: int
    end: int

@dataclass
class Sentence:
    text: str
    start: int
    end: int

@dataclass
class UrlCandidate:
    url: str
    start: int
    end: int
    source: str  # raw_url or markdown_link


def normalize_chars(s: str) -> str:
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    return html.unescape(s).replace("\r\n", "\n").replace("\r", "\n")


def iter_markdown_files(path: str) -> Iterator[MdFile]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                if not name.lower().endswith(".md") or name.endswith("/"):
                    continue
                raw = z.read(name).decode("utf-8", "replace")
                yield MdFile(file=name, paper_id=Path(name).stem, text=normalize_chars(raw))
    elif p.is_dir():
        for f in sorted(p.rglob("*.md")):
            raw = f.read_text(encoding="utf-8", errors="replace")
            yield MdFile(file=str(f), paper_id=f.stem, text=normalize_chars(raw))
    elif p.is_file() and p.suffix.lower() == ".md":
        raw = p.read_text(encoding="utf-8", errors="replace")
        yield MdFile(file=str(p), paper_id=p.stem, text=normalize_chars(raw))
    else:
        raise SystemExit(f"Input is not a .zip, directory, or .md file: {path}")


def strip_code_blocks(text: str) -> str:
    # Preserve offsets by replacing fenced code with whitespace/newlines.
    def repl(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    return re.sub(r"(?ms)^```.*?^```\s*", repl, text)


def line_heading_key(line: str) -> str:
    x = re.sub(r"^\s{0,3}#{1,6}\s*", "", line.strip())
    x = re.sub(r"[:.\-–, \s]+$", "", x).strip().lower()
    return x


def find_reference_start(text: str) -> Optional[int]:
    matches: List[Tuple[int, str]] = []
    pos = 0
    for line in text.splitlines(True):
        key = line_heading_key(line)
        if key in REF_HEADINGS:
            matches.append((pos, key))
        pos += len(line)
    if not matches:
        return None
    cutoff = int(len(text) * 0.35)
    later = [m for m in matches if m[0] >= cutoff]
    return (later[0] if later else matches[-1])[0]


def looks_bibliographic_line(line: str) -> bool:
    if re.match(r"^\s*\d{1,4}[.)]\s+[A-Z][A-Za-z'’`-]+\s+[A-Z]", line):
        return True
    if re.search(r"\b(19|20)\d{2}[a-z]?\b", line) and re.search(r"\b(journal|proc\.|proceedings|press|university|vol\.|pages?|doi|lancet|nature|science|bioinformatics)\b", line, re.I):
        return True
    return False


def remove_standalone_url_footnote_lines(text: str) -> str:
    # Remove OCR-inserted footnote lines from the body extractor.  These belong
    # to the footnote restoration task, not direct body URLs.  Preserve line
    # count/rough offsets by replacing non-newline chars with spaces.
    chars = list(text)
    pos = 0
    marker_line = re.compile(r"^\s*(?:\[\^?\w+\]\s*:?|\^?\d{1,3}|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|[*†‡§¶])\s*[:.)\-]?\s+.*(?:https?://|ftp://|www\.|doi\s*:|10\.)", re.I)
    marker_url_tight = re.compile(r"^\s*(?:\[\^?\w+\]\s*:?|\^?\d{1,3}|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|[*†‡§¶])\s*[:.)\-]?\s*(?:https?://|ftp://|www\.|doi\s*:|10\.)", re.I)
    for line in text.splitlines(True):
        raw = line.rstrip("\n")
        if URL_RE.search(raw) and (marker_line.search(raw) or marker_url_tight.search(raw)) and not looks_bibliographic_line(raw):
            for i in range(pos, pos + len(line)):
                if chars[i] != "\n":
                    chars[i] = " "
        pos += len(line)
    return "".join(chars)

def body_region(text: str) -> str:
    text = strip_code_blocks(text)
    ref = find_reference_start(text)
    if ref is not None:
        text = text[:ref]
    text = remove_standalone_url_footnote_lines(text)
    return text


def paragraph_spans(text: str) -> List[Paragraph]:
    paras: List[Paragraph] = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        raw = m.group(0)
        # Drop image-only paragraphs and HTML/table-heavy blocks.
        if IMAGE_MD_RE.match(raw.strip()):
            continue
        joined = re.sub(r"(?<!-)\n(?!\s*[-*+]\s)", " ", raw).strip()
        joined = re.sub(r"\s+", " ", joined)
        if joined:
            paras.append(Paragraph(joined, m.start(), m.end()))
    return paras


def sentence_spans(paragraph: str) -> List[Sentence]:
    out: List[Sentence] = []
    start = 0
    n = len(paragraph)
    i = 0
    while i < n:
        ch = paragraph[i]
        if ch in ".?!":
            # URL/domain guard: do not split inside URLs/domains/DOIs.
            left_word = paragraph[max(0, i - 25): i + 1].split()[-1].lower() if paragraph[: i + 1].split() else ""
            if left_word in ABBREVIATIONS or re.search(r"\b(?:https?|ftp|www|doi)\.?$", left_word):
                i += 1; continue
            if i > 0 and i + 1 < n and paragraph[i - 1].isdigit() and paragraph[i + 1].isdigit():
                i += 1; continue
            # Consume closing quotes/brackets.
            j = i + 1
            while j < n and paragraph[j] in "\"'”’)]}":
                j += 1
            k = j
            while k < n and paragraph[k].isspace():
                k += 1
            if k >= n or paragraph[k].isupper() or paragraph[k].isdigit() or paragraph[k] in "([\"'“‘":
                sent = paragraph[start:j].strip()
                if sent:
                    # Span is in paragraph-local coordinates.
                    lead = len(paragraph[start:j]) - len(paragraph[start:j].lstrip())
                    trail = len(paragraph[start:j].rstrip())
                    out.append(Sentence(sent, start + lead, start + trail))
                start = k
                i = k
                continue
        i += 1
    if start < n:
        sent = paragraph[start:].strip()
        if sent:
            lead = len(paragraph[start:]) - len(paragraph[start:].lstrip())
            out.append(Sentence(sent, start + lead, n))
    return out


def clean_url(u: str) -> str:
    u = (u or "").strip()
    # Undo spaces accidentally introduced into schemes.
    u = re.sub(r"(?i)^(https?|ftp|ftps|sftp|s3|gs|az)\s*:?\s*/\s*/\s*", lambda m: m.group(1).lower() + "://", u)
    u = re.sub(r"(?i)^doi\s*:\s*", "doi:", u)
    u = u.strip("<>[]{}\"'`")
    while u and u[-1] in ".,;:!?'\"”’":
        u = u[:-1]
    # Remove unmatched closing parentheses, but keep balanced URL paths like wiki/Foo_(bar)
    while u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
    return u


def is_probable_email_context(text: str, start: int, end: int) -> bool:
    # If the URL-like domain is part of an email address such as
    # name@sub.example.edu, the regex may match only example.edu.  Look at
    # the full non-whitespace token around the candidate and reject if an @
    # occurs before the candidate within that token.
    token_s = start
    while token_s > 0 and not text[token_s - 1].isspace():
        token_s -= 1
    token_e = end
    while token_e < len(text) and not text[token_e].isspace():
        token_e += 1
    token = text[token_s:token_e]
    rel_start = start - token_s
    if "@" in token[:rel_start + 1]:
        return True
    lo = max(0, start - 80); hi = min(len(text), end + 80)
    ctx = text[lo:hi]
    return bool(re.search(r"[\w.+-]+@[\w.-]*" + re.escape(text[start:end].split("/")[0]), ctx, re.I))


def markdown_link_candidates(text: str) -> List[UrlCandidate]:
    cands: List[UrlCandidate] = []
    for m in MD_LINK_RE.finditer(text):
        if m.group(0).startswith("!"):
            continue
        label, url = m.group(1).strip(), clean_url(m.group(2))
        if not URL_RE.match(url):
            continue
        # Citation hyperlinks generated by Markdown are not printed body URLs.
        # Keep only when the visible label itself is URL-like or when the local
        # prose strongly indicates resource access.
        local = text[max(0, m.start() - 120): min(len(text), m.end() + 120)].lower()
        label_is_url = bool(URL_RE.search(label))
        resource_words = re.search(r"\b(available|download|repository|github|gitlab|source code|dataset|data set|database|access|deposited|supplement|figshare|zenodo|protocol|code)\b", local)
        citation_like = re.search(r"\b(et al\.?|\d{4}[a-z]?|fig\.|table|doi\.org/10\.|journal|lancet|nature|science)\b", label.lower() + " " + local)
        if label_is_url or (resource_words and not citation_like):
            cands.append(UrlCandidate(url=url, start=m.start(2), end=m.end(2), source="markdown_link"))
    return cands


def raw_url_candidates(text: str) -> List[UrlCandidate]:
    out: List[UrlCandidate] = []
    for m in URL_RE.finditer(text):
        u = clean_url(m.group(1))
        if not u:
            continue
        if is_probable_email_context(text, m.start(1), m.end(1)):
            continue
        out.append(UrlCandidate(url=u, start=m.start(1), end=m.end(1), source="raw_url"))
    return out


def line_at(text: str, pos: int) -> str:
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    if e < 0: e = len(text)
    return text[s:e]


def is_footnote_like_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:\[\^?\w+\]:?|\^?\d+|\d{1,3}|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|[*†‡§¶])\s*[:.)\-]?\s*(?:https?://|www\.|ftp://|doi\s*:|10\.)", line, re.I))


def is_caption_or_table(paragraph: str) -> bool:
    s = paragraph.strip()
    if s.startswith("<") and re.search(r"</?(table|tr|td|th|tbody|thead)\b", s, re.I):
        return True
    return bool(re.match(r"^(fig\.?|figure|table|algorithm|listing)\s*\d*\b", s, re.I))


def is_license_or_publisher_boilerplate(paragraph: str) -> bool:
    low = paragraph.lower()
    boiler = [
        "creative commons", "open access this article", "springer nature remains neutral",
        "reprints and permissions", "publisher's note", "publisher’s note", "copyright",
        "all rights reserved", "license", "correspondence and requests",
    ]
    return any(x in low for x in boiler)


def is_body_url_candidate(paragraph: str, cand: UrlCandidate, full_text: str) -> bool:
    line = line_at(full_text, cand.start)
    lowp = paragraph.lower()
    lowline = line.lower()
    url_low = cand.url.lower()
    if is_footnote_like_line(line):
        return False
    if IMAGE_MD_RE.match(line.strip()):
        return False
    if is_license_or_publisher_boilerplate(paragraph):
        return False
    if re.search(r"\b(e-?mail|email|corresponding author)\b", lowline):
        return False
    if re.search(r"\b(references|bibliography)\b", lowline) and len(line.strip()) < 40:
        return False
    if is_caption_or_table(paragraph):
        # Exception: scholarly-resource supplement/source-data panels.
        if not re.search(r"\b(source data|figure supplement|supplementary file|supplementary material|data availability|available at|deposited|figshare|zenodo|dryad|github|repository)\b", lowp):
            return False
    if re.search(r"</?(table|tr|td|th|tbody|thead)\b", paragraph, re.I):
        return False
    if re.search(r"\b(xmlns|namespace|schema|rdf|owl|sparql|prefix|select \*|curl |wget |ssh |localhost|127\.0\.0\.1)\b", lowp):
        return False
    if url_low.startswith(("mailto:", "tel:", "javascript:", "data:", "file:")):
        return False
    # Markdown DOI links around author-year citations are usually generated
    # citation links, not printed body URLs.
    if cand.source == "markdown_link" and "doi.org/10." in url_low and re.search(r"\b(et al\.?|\d{4}[a-z]?|fig\.|table)\b", lowp):
        if not re.search(r"\b(source data|figure supplement|supplementary file|data availability|deposited|dataset|figshare|zenodo)\b", lowp):
            return False
    return True


def locate_paragraph(paras: Sequence[Paragraph], pos: int) -> Optional[Tuple[int, Paragraph]]:
    for i, p in enumerate(paras):
        if p.start <= pos <= p.end:
            return i, p
    return None


def process_file(md: MdFile) -> List[Dict[str, object]]:
    text = body_region(md.text)
    paras = paragraph_spans(text)
    candidates = raw_url_candidates(text) + markdown_link_candidates(text)
    # De-duplicate exact same span/url from raw + markdown passes.
    seen_spans = set()
    rows: List[Dict[str, object]] = []
    for cand in sorted(candidates, key=lambda c: (c.start, c.end, c.url)):
        key = (cand.start, cand.end, cand.url.lower())
        if key in seen_spans:
            continue
        seen_spans.add(key)
        located = locate_paragraph(paras, cand.start)
        if not located:
            continue
        pidx, para = located
        if not is_body_url_candidate(para.text, cand, text):
            continue
        local_pos = max(0, min(len(para.text), cand.start - para.start))
        sents = sentence_spans(para.text)
        sent_idx = None
        for i, s in enumerate(sents):
            if s.start <= local_pos <= s.end:
                sent_idx = i; break
        if sent_idx is None:
            continue
        target = sents[sent_idx].text
        # Ensure the URL actually appears in target or the target contains the markdown link syntax around it.
        if cand.url not in target and clean_url(cand.url) not in target:
            # For URLs found in markdown target, the sentence may contain the whole [label](url).
            pass
        preceding = sents[sent_idx - 1].text if sent_idx > 0 else None
        trailing = sents[sent_idx + 1].text if sent_idx + 1 < len(sents) else None
        out_key = (md.paper_id, cand.url.lower(), target.lower())
        row = {
            "paper_id": md.paper_id,
            "file": md.file,
            "location": "body",
            "url_output": cand.url,
            "preceding_output": preceding,
            "target_output": target,
            "trailing_output": trailing,
            "at_paragraph_start": preceding is None,
            "at_paragraph_end": trailing is None,
            "source": cand.source,
            "start_char": cand.start,
            "end_char": cand.end,
        }
        rows.append(row)
    # final de-dupe by URL+target
    final: List[Dict[str, object]] = []
    seen = set()
    for r in rows:
        k = (r["paper_id"], str(r["url_output"]).lower(), str(r["target_output"]).lower())
        if k not in seen:
            seen.add(k); final.append(r)
    return final


def write_outputs(rows: List[Dict[str, object]], out_csv: str, out_jsonl: Optional[str]) -> None:
    fieldnames = [
        "paper_id", "file", "location", "url_output", "preceding_output", "target_output",
        "trailing_output", "at_paragraph_start", "at_paragraph_end", "source", "start_char", "end_char",
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True) if Path(out_csv).parent != Path('.') else None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract body URL sentence context from olmOCR Markdown.")
    ap.add_argument("--input", required=True, help="Input .zip, directory, or .md file")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_jsonl", default=None, help="Optional JSONL output path")
    args = ap.parse_args(argv)

    all_rows: List[Dict[str, object]] = []
    n_files = 0
    for md in iter_markdown_files(args.input):
        n_files += 1
        all_rows.extend(process_file(md))
    write_outputs(all_rows, args.out_csv, args.out_jsonl)
    print(f"Processed {n_files} Markdown file(s); wrote {len(all_rows)} body URL row(s) to {args.out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
