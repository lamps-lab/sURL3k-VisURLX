#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00ad": "", "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
}
SUP_TO_DIGIT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
DIGIT_TO_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

REF_HEADINGS = {
    "references", "reference", "bibliography", "works cited", "literature cited",
    "reference list", "list of references", "sources", "citations"
}
STOP_AFTER_REFS = {
    "acknowledgements", "acknowledgments", "author contributions", "competing interests",
    "conflict of interest", "additional information", "supplementary information",
    "reprints and permissions", "publisher's note", "publisher’s note", "open access",
    "correspondence", "ethics declarations", "data availability", "code availability",
}

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
class ReferenceEntry:
    ref_id: str
    index: Optional[str]
    full_text: str
    urls: List[str]
    year: Optional[str] = None
    first_author: Optional[str] = None
    second_author: Optional[str] = None
    keys: Set[str] = field(default_factory=set)

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
class CitationOccurrence:
    marker: str
    style: str
    start: int
    end: int
    surface: str
    visible_group: Optional[str]
    inside_parenthetical: bool
    ref: ReferenceEntry


def normalize_chars(s: str) -> str:
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    return html.unescape(s).replace("\r\n", "\n").replace("\r", "\n")


def norm_for_match(s: str) -> str:
    s = normalize_chars(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def iter_markdown_files(path: str) -> Iterator[MdFile]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                if not name.lower().endswith(".md") or name.endswith("/"):
                    continue
                yield MdFile(name, Path(name).stem, normalize_chars(z.read(name).decode("utf-8", "replace")))
    elif p.is_dir():
        for f in sorted(p.rglob("*.md")):
            yield MdFile(str(f), f.stem, normalize_chars(f.read_text(encoding="utf-8", errors="replace")))
    elif p.is_file() and p.suffix.lower() == ".md":
        yield MdFile(str(p), p.stem, normalize_chars(p.read_text(encoding="utf-8", errors="replace")))
    else:
        raise SystemExit(f"Input is not a .zip, directory, or .md file: {path}")


def strip_code_blocks(text: str) -> str:
    def repl(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    return re.sub(r"(?ms)^```.*?^```\s*", repl, text)


def line_heading_key(line: str) -> str:
    x = re.sub(r"^\s{0,3}#{1,6}\s*", "", line.strip())
    x = re.sub(r"[:.\-–, \s]+$", "", x).strip().lower()
    return x


def find_reference_heading(text: str) -> Optional[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(True):
        if line_heading_key(line) in REF_HEADINGS:
            matches.append((pos, pos + len(line)))
        pos += len(line)
    if not matches:
        return None
    cutoff = int(len(text) * 0.35)
    later = [m for m in matches if m[0] >= cutoff]
    return later[0] if later else matches[-1]


def split_body_and_refs(text: str) -> Tuple[str, str, Optional[int]]:
    text = strip_code_blocks(text)
    h = find_reference_heading(text)
    if not h:
        return text, "", None
    body = text[:h[0]]
    refs = text[h[1]:]
    # Stop after references once a non-reference back-matter heading appears.
    pos = 0
    for line in refs.splitlines(True):
        key = line_heading_key(line)
        if pos > 200 and (key in STOP_AFTER_REFS or re.match(r"^\s*E-?mail address\s*:", line, re.I)):
            refs = refs[:pos]
            break
        pos += len(line)
    return body, refs, h[0]


def clean_url(u: str) -> str:
    u = (u or "").strip()
    u = re.sub(r"(?i)^(https?|ftp|ftps|sftp|s3|gs|az)\s*:?\s*/\s*/\s*", lambda m: m.group(1).lower() + "://", u)
    u = re.sub(r"(?i)^doi\s*:\s*", "doi:", u)
    u = u.strip("<>[]{}\"'`")
    while u and u[-1] in ".,;:!?'\"”’":
        u = u[:-1]
    while u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
    return u


def urls_in(text: str) -> List[str]:
    candidates: List[str] = []
    for m in URL_RE.finditer(text):
        candidates.append(m.group(1))
    for m in MD_LINK_RE.finditer(text):
        if not m.group(0).startswith("!"):
            candidates.append(m.group(2))
    out: List[str] = []
    seen = set()
    for c in candidates:
        u = clean_url(c)
        if not u or u.lower().startswith(("mailto:", "tel:", "javascript:", "data:", "file:")):
            continue
        # Reject domains captured from email addresses, e.g. name@sub.example.edu.
        for mm in re.finditer(re.escape(c), text):
            ts = mm.start()
            while ts > 0 and not text[ts - 1].isspace():
                ts -= 1
            if "@" in text[ts:mm.start()+1]:
                u = ""
                break
        if not u:
            continue
        # In the reference task, keep doi.org URLs; skip bare doi: identifiers unless they are URL-like.
        if u.lower().startswith("doi:"):
            continue
        if u.lower() not in seen:
            seen.add(u.lower()); out.append(u)
    return out


def unwrap_markdown_links(text: str, keep_url_when_label_url: bool = False) -> str:
    def repl(m: re.Match) -> str:
        if m.group(0).startswith("!"):
            return ""
        label, url = m.group(1).strip(), clean_url(m.group(2))
        if keep_url_when_label_url and URL_RE.search(label):
            return label
        return label
    return MD_LINK_RE.sub(repl, text)


def normalize_reference_entry_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def parse_numbered_references(ref_text: str) -> List[ReferenceEntry]:
    lines = ref_text.splitlines()
    starts = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+", line):
            starts.append(i)
    entries: List[ReferenceEntry] = []
    if len(starts) < 2:
        return entries
    starts.append(len(lines))
    for a, b in zip(starts, starts[1:]):
        block = "\n".join(lines[a:b]).strip()
        if not block:
            continue
        m = re.match(r"^\s*(?:\[(?P<b>\d{1,4})\]|(?P<n>\d{1,4})[.)])\s+(?P<body>.*)", block, re.S)
        if not m:
            continue
        idx = m.group("b") or m.group("n")
        full = normalize_reference_entry_text(block)
        entries.append(ReferenceEntry(ref_id=idx, index=idx, full_text=full, urls=urls_in(full)))
    return entries


def paragraph_blocks(text: str) -> List[str]:
    blocks = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S):
        block = normalize_reference_entry_text(m.group(0))
        if block:
            blocks.append(block)
    return blocks


def parse_unnumbered_references(ref_text: str) -> List[ReferenceEntry]:
    blocks = paragraph_blocks(ref_text)
    # Some PDFs have one entry per line without blank lines.  Split those blocks
    # when a new line looks like an author-initial start and the previous text
    # already ended like a full reference.
    expanded: List[str] = []
    for block in blocks:
        raw_lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if len(raw_lines) > 1:
            current = ""
            for line in raw_lines:
                new_ref = bool(re.match(r"^(?:[A-Z](?:\.|[a-z]+)[A-Za-z'’`.-]*\s+){0,4}[A-Z][A-Za-z'’`-]+\b", line)) and bool(re.search(r"\b(19|20)\d{2}[a-z]?\b", current))
                if current and new_ref:
                    expanded.append(normalize_reference_entry_text(current)); current = line
                else:
                    current = (current + " " + line).strip()
            if current:
                expanded.append(normalize_reference_entry_text(current))
        else:
            expanded.append(block)
    entries: List[ReferenceEntry] = []
    for i, block in enumerate(expanded, start=1):
        us = urls_in(block)
        # Keep all entries for key building? For this restoration we only need URL-bearing.
        entries.append(ReferenceEntry(ref_id=f"U{i}", index=None, full_text=block, urls=us))
    return entries


def trim_front_matter_for_citations(body: str) -> str:
    # Avoid author affiliations and title-page numeric superscripts being read
    # as bibliography citations.  Keep Abstract if present; otherwise start at
    # Introduction/Background when found reasonably early.
    headings = ["abstract", "background", "introduction", "1 introduction", "1. introduction"]
    pos = 0
    candidates = []
    for line in body.splitlines(True):
        key = line_heading_key(line)
        if pos < max(3000, int(len(body) * 0.35)) and (key in headings or key.startswith("abstract") or key.startswith("1 introduction") or key.startswith("introduction")):
            candidates.append(pos)
        pos += len(line)
    if candidates:
        return body[candidates[0]:]
    return body

def parse_references(ref_text: str) -> List[ReferenceEntry]:
    numbered = parse_numbered_references(ref_text)
    entries = numbered if numbered else parse_unnumbered_references(ref_text)
    for e in entries:
        add_reference_keys(e, numeric_mode=bool(numbered))
    return entries


def extract_year(text: str) -> Optional[str]:
    years = re.findall(r"\b((?:19|20)\d{2}[a-z]?)\b", text)
    return years[-1] if years else None


def extract_author_surnames(entry: str) -> Tuple[Optional[str], Optional[str]]:
    # Use text before first year or before title-ish second sentence.
    y = re.search(r"\b(?:19|20)\d{2}[a-z]?\b", entry)
    head = entry[: y.start()] if y else entry[:180]
    head = re.sub(r"^\s*(?:\[?\d+\]?|\d+[.)])\s+", "", head).strip()
    # Common initials-first: "S.A. Berger and A. Stamatakis."
    surnames = re.findall(r"(?:^|[,;&]|\band\b|\s)(?:[A-Z]\.?\s*){1,4}([A-Z][A-Za-z'’`-]{2,})\b", head)
    if not surnames:
        # Surname-first: "Berger SA, Stamatakis A:"
        surnames = re.findall(r"(?:^|,\s*|;\s*|\band\s+|&\s*)([A-Z][A-Za-z'’`-]{2,})(?:\s+[A-Z]\b|,|:|\.)", head)
    # Remove obvious non-author words.
    bad = {"Department", "University", "Press", "Journal", "Proceedings", "Available", "Retrieved"}
    surnames = [s for s in surnames if s not in bad]
    first = surnames[0] if surnames else None
    second = surnames[1] if len(surnames) > 1 else None
    return first, second


def add_reference_keys(e: ReferenceEntry, numeric_mode: bool) -> None:
    e.urls = urls_in(e.full_text)
    if e.index:
        idx = re.sub(r"\D", "", e.index) or e.index.strip("[]")
        e.keys.add(idx); e.keys.add(f"[{idx}]")
    if numeric_mode:
        return
    e.year = extract_year(e.full_text)
    e.first_author, e.second_author = extract_author_surnames(e.full_text)
    if not e.year or not e.first_author:
        return
    fa, yr = e.first_author, e.year
    bases = {f"{fa}, {yr}", f"{fa} {yr}", f"{fa} ({yr})", f"{fa} et al., {yr}", f"{fa} et al. {yr}", f"{fa} et al. ({yr})"}
    if e.second_author:
        sa = e.second_author
        bases.update({f"{fa} and {sa}, {yr}", f"{fa} and {sa} {yr}", f"{fa} and {sa} ({yr})",
                      f"{fa} & {sa}, {yr}", f"{fa} & {sa} {yr}", f"{fa} & {sa} ({yr})"})
    for b in bases:
        e.keys.add(b)
        e.keys.add(f"({b})")
        e.keys.add(f"[{b}]")


def paragraph_spans(text: str) -> List[Paragraph]:
    paras: List[Paragraph] = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        raw = m.group(0)
        if IMAGE_MD_RE.match(raw.strip()):
            continue
        joined = re.sub(r"(?<!-)\n", " ", raw).strip()
        joined = re.sub(r"\s+", " ", joined)
        if joined:
            paras.append(Paragraph(joined, m.start(), m.end()))
    return paras


def sentence_spans(paragraph: str) -> List[Sentence]:
    out: List[Sentence] = []
    start = 0; n = len(paragraph); i = 0
    while i < n:
        if paragraph[i] in ".?!":
            left_word = paragraph[max(0, i - 25): i + 1].split()[-1].lower() if paragraph[: i + 1].split() else ""
            if left_word in ABBREVIATIONS:
                i += 1; continue
            if i > 0 and i + 1 < n and paragraph[i - 1].isdigit() and paragraph[i + 1].isdigit():
                i += 1; continue
            j = i + 1
            while j < n and paragraph[j] in "\"'”’)]}":
                j += 1
            k = j
            while k < n and paragraph[k].isspace():
                k += 1
            if k >= n or paragraph[k].isupper() or paragraph[k].isdigit() or paragraph[k] in "([\"'“‘":
                seg = paragraph[start:j]; sent = seg.strip()
                if sent:
                    lead = len(seg) - len(seg.lstrip()); trail = len(seg.rstrip())
                    out.append(Sentence(sent, start + lead, start + trail))
                start = k; i = k; continue
        i += 1
    if start < n:
        seg = paragraph[start:]; sent = seg.strip()
        if sent:
            lead = len(seg) - len(seg.lstrip())
            out.append(Sentence(sent, start + lead, n))
    return out


def locate_paragraph(paras: Sequence[Paragraph], pos: int) -> Optional[Tuple[int, Paragraph]]:
    for i, p in enumerate(paras):
        if p.start <= pos <= p.end:
            return i, p
    return None


def expand_numeric_group(content: str) -> List[str]:
    # Remove locator text after a clear leading citation number.
    content = content.replace("–", "-").replace(", ", "-").replace("−", "-")
    parts = re.split(r"\s*,\s*|\s*;\s*", content)
    nums: Set[int] = set()
    for part in parts:
        part = part.strip()
        mrange = re.match(r"^(\d{1,4})\s*-\s*(\d{1,4})\b", part)
        if mrange:
            a, b = int(mrange.group(1)), int(mrange.group(2))
            if 0 < a <= b <= a + 100:
                nums.update(range(a, b + 1))
            continue
        m = re.match(r"^(\d{1,4})\b", part)
        if m:
            nums.add(int(m.group(1)))
    return [str(x) for x in sorted(nums)]


def build_lookup(entries: Sequence[ReferenceEntry]) -> Dict[str, List[ReferenceEntry]]:
    lookup: Dict[str, List[ReferenceEntry]] = {}
    for e in entries:
        if not e.urls:
            continue
        for k in e.keys:
            nk = norm_for_match(k.strip("[]() "))
            if nk:
                lookup.setdefault(nk, []).append(e)
        if e.index:
            idx = re.sub(r"\D", "", e.index) or e.index
            lookup.setdefault(idx, []).append(e)
    return lookup


def choose_ref(lookup: Dict[str, List[ReferenceEntry]], marker: str) -> Optional[ReferenceEntry]:
    key = norm_for_match(marker.strip("[]() "))
    refs = lookup.get(key)
    if refs:
        return refs[0]
    return None


def is_inside_parenthetical(text: str, start: int, end: int) -> bool:
    left = text.rfind("(", 0, start)
    right = text.find(")", end)
    if left == -1 or right == -1:
        return False
    # no sentence boundary between open and citation and close is near
    return start - left < 180 and right - end < 180


def detect_bracket_numeric(body: str, lookup: Dict[str, List[ReferenceEntry]]) -> List[CitationOccurrence]:
    occs: List[CitationOccurrence] = []
    # [15], [15, Chapter 6], [21-23], [6,7]
    for m in re.finditer(r"\[((?:\d{1,4})(?:\s*(?:,|;|-|–|, |−)\s*\d{1,4})*(?:\s*,\s*(?:Chapter|Ch\.|§|Sec\.|Section|p\.|pp\.|Thm\.|Theorem|Eq\.|Fig\.|Table)[^\]]*)?)\]", body):
        content = m.group(1)
        members = expand_numeric_group(content)
        if not members:
            continue
        for mem in members:
            ref = choose_ref(lookup, mem)
            if ref:
                occs.append(CitationOccurrence(mem, "bracket_number", m.start(), m.end(), m.group(0), m.group(0), is_inside_parenthetical(body, m.start(), m.end()), ref))
    return occs


def detect_superscript_numeric(body: str, lookup: Dict[str, List[ReferenceEntry]]) -> List[CitationOccurrence]:
    occs: List[CitationOccurrence] = []
    for m in re.finditer(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:[,\-–, −][⁰¹²³⁴⁵⁶⁷⁸⁹]+)*", body):
        token = m.group(0).translate(SUP_TO_DIGIT).replace("–", "-").replace(", ", "-").replace("−", "-")
        members = expand_numeric_group(token)
        for mem in members:
            ref = choose_ref(lookup, mem)
            if ref:
                occs.append(CitationOccurrence(mem, "superscript_number", m.start(), m.end(), m.group(0), m.group(0), is_inside_parenthetical(body, m.start(), m.end()), ref))
    return occs


def detect_baseline_numeric(body: str, lookup: Dict[str, List[ReferenceEntry]]) -> List[CitationOccurrence]:
    occs: List[CitationOccurrence] = []
    numeric_markers = sorted([k for k in lookup.keys() if k.isdigit()], key=lambda x: (-len(x), int(x)))
    if not numeric_markers:
        return occs
    marker_alt = "|".join(re.escape(k) for k in numeric_markers)
    # token ending in an allowed marker followed by punctuation/space. Token must contain a letter.
    pat = re.compile(rf"(?P<token>\b[A-Za-z][A-Za-z0-9_./+\-]*?(?P<marker>{marker_alt}))(?=[,.;:)\]\s])")
    for m in pat.finditer(body):
        token = m.group("token"); marker = m.group("marker")
        # Reject common biomedical names where the digit is likely part of the entity, unless punctuation style strongly suggests citation.
        core = token[:-len(marker)] if marker else token
        if not re.search(r"[A-Za-z]", core):
            continue
        # Avoid short section/table/figure labels and file names.
        if re.match(r"^(Fig|Table|Eq|Sec|Ref|Chapter|KIC|PMC|PMID|COVID|H1N1|H5N1|SARS|MERS)-?", token, re.I):
            # But allow when the core is long enough and previous chars suggest a sentence/prose word, not a label.
            if len(core) < 10:
                continue
        ref = choose_ref(lookup, marker)
        if ref:
            s = m.start("marker"); e = m.end("marker")
            occs.append(CitationOccurrence(marker, "baseline_numeric", s, e, marker, marker, is_inside_parenthetical(body, s, e), ref))
    return occs


def detect_author_year(body: str, entries: Sequence[ReferenceEntry], lookup: Dict[str, List[ReferenceEntry]]) -> List[CitationOccurrence]:
    occs: List[CitationOccurrence] = []
    seen_keys: Set[str] = set()
    for e in entries:
        if not e.urls:
            continue
        for key in sorted(e.keys, key=len, reverse=True):
            bare = key.strip("[]() ")
            if not bare or bare.isdigit() or len(bare) < 6:
                continue
            nk = norm_for_match(bare)
            if nk in seen_keys:
                continue
            seen_keys.add(nk)
            # Match bare, parenthesized, or bracketed forms.  Allow comma optional before year.
            escaped = re.escape(bare)
            escaped = escaped.replace(re.escape(", "), r",?\s+").replace(re.escape(" "), r"\s+")
            pat = re.compile(rf"(?<![A-Za-z0-9])(?:\({escaped}\)|\[{escaped}\]|{escaped})(?![A-Za-z0-9])", re.I)
            for m in pat.finditer(body):
                surf = m.group(0)
                occs.append(CitationOccurrence(bare, "author_year", m.start(), m.end(), surf, None, surf.startswith("(") or is_inside_parenthetical(body, m.start(), m.end()), e))
    return occs


def detect_citations(body: str, entries: Sequence[ReferenceEntry]) -> List[CitationOccurrence]:
    lookup = build_lookup(entries)
    occs: List[CitationOccurrence] = []
    occs.extend(detect_bracket_numeric(body, lookup))
    occs.extend(detect_superscript_numeric(body, lookup))
    occs.extend(detect_baseline_numeric(body, lookup))
    occs.extend(detect_author_year(body, entries, lookup))
    # De-duplicate same marker/surface/span/ref.
    final: List[CitationOccurrence] = []
    seen = set()
    for o in sorted(occs, key=lambda x: (x.start, x.end, x.marker, x.ref.ref_id)):
        k = (o.start, o.end, o.ref.ref_id)
        if k not in seen:
            seen.add(k); final.append(o)
    return final


def find_sentence_for_occurrence(text: str, start: int) -> Optional[Tuple[Paragraph, int, List[Sentence], int]]:
    paras = paragraph_spans(text)
    located = locate_paragraph(paras, start)
    if not located:
        return None
    _, para = located
    raw_prefix = text[para.start:start]
    local_pos = len(re.sub(r"(?<!-)\n", " ", raw_prefix))
    sents = sentence_spans(para.text)
    for i, s in enumerate(sents):
        if s.start <= local_pos <= s.end:
            return para, i, sents, local_pos
    return None


def restore_citation_in_sentence(sentence: str, sent_local_start: int, occ_local_start: int, occ_len: int, marker: str, ref_text: str) -> str:
    rel_s = max(0, min(len(sentence), occ_local_start - sent_local_start))
    rel_e = max(rel_s, min(len(sentence), rel_s + occ_len))
    replacement = f"[{marker} {ref_text}]"
    return sentence[:rel_s] + replacement + sentence[rel_e:]


def process_file(md: MdFile) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    body_raw, refs_text, ref_start = split_body_and_refs(md.text)
    body_raw = trim_front_matter_for_citations(body_raw)
    entries = parse_references(refs_text) if refs_text else []
    url_entries = [e for e in entries if e.urls]
    # Unwrap markdown links in body so generated citation hyperlinks behave like printed citations.
    body = unwrap_markdown_links(body_raw)
    occs = detect_citations(body, entries)
    rows: List[Dict[str, object]] = []
    seen = set()
    for occ in occs:
        fs = find_sentence_for_occurrence(body, occ.start)
        if not fs:
            continue
        para, sent_idx, sents, occ_local = fs
        target = sents[sent_idx].text
        preceding = sents[sent_idx - 1].text if sent_idx > 0 else None
        trailing = sents[sent_idx + 1].text if sent_idx + 1 < len(sents) else None
        restored = restore_citation_in_sentence(target, sents[sent_idx].start, occ_local, occ.end - occ.start, occ.marker, occ.ref.full_text)
        for url in occ.ref.urls:
            k = (md.paper_id, occ.marker, url.lower(), target.lower(), occ.start)
            if k in seen:
                continue
            seen.add(k)
            rows.append({
                "paper_id": md.paper_id,
                "file": md.file,
                "location": "reference",
                "citation_marker": occ.marker,
                "citation_style": occ.style,
                "url_output": url,
                "reference_entry": occ.ref.full_text,
                "preceding_output": preceding,
                "target_output": restored,
                "original_sentence": target,
                "trailing_output": trailing,
                "at_paragraph_start": preceding is None,
                "at_paragraph_end": trailing is None,
                "inside_parenthetical": occ.inside_parenthetical,
                "visible_group": occ.visible_group,
                "citation_surface": occ.surface,
                "citation_start_char": occ.start,
                "reference_index": occ.ref.index,
            })
    stats = {"reference_entries": len(entries), "url_reference_entries": len(url_entries), "citation_occurrences": len(occs)}
    return rows, stats


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    fields = [
        "paper_id", "file", "location", "citation_marker", "citation_style", "url_output",
        "reference_entry", "preceding_output", "target_output", "original_sentence", "trailing_output",
        "at_paragraph_start", "at_paragraph_end", "inside_parenthetical", "visible_group",
        "citation_surface", "citation_start_char", "reference_index",
    ]
    if Path(path).parent != Path('.'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_jsonl(rows: List[Dict[str, object]], path: Optional[str]) -> None:
    if not path: return
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Restore URL-bearing references in olmOCR Markdown.")
    ap.add_argument("--input", required=True, help="Input .zip, directory, or .md file")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_jsonl", default=None, help="Optional JSONL output path")
    ap.add_argument("--summary_json", default=None, help="Optional per-corpus summary JSON path")
    args = ap.parse_args(argv)

    all_rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, int]] = {}
    n_files = 0
    for md in iter_markdown_files(args.input):
        n_files += 1
        rows, stats = process_file(md)
        all_rows.extend(rows); summary[md.paper_id] = stats | {"rows": len(rows)}
    write_csv(all_rows, args.out_csv)
    write_jsonl(all_rows, args.out_jsonl)
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump({"files": n_files, "rows": len(all_rows), "papers": summary}, f, ensure_ascii=False, indent=2)
    print(f"Processed {n_files} Markdown file(s); wrote {len(all_rows)} reference-restored row(s) to {args.out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
