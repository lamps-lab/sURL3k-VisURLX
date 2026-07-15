#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00ad": "", "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
}

SUP_TO_DIGIT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
DIGIT_TO_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

REF_HEADINGS = {
    "references", "reference", "bibliography", "works cited", "literature cited",
    "reference list", "list of references", "sources", "citations"
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
class Footnote:
    marker: str
    marker_norm: str
    footnote_text: str
    urls: List[str]
    start: int
    end: int
    kind: str  # standalone or inline

@dataclass
class MarkerOcc:
    start: int
    end: int
    surface: str
    score: int


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


def find_reference_start(text: str) -> Optional[int]:
    matches: List[int] = []
    pos = 0
    for line in text.splitlines(True):
        if line_heading_key(line) in REF_HEADINGS:
            matches.append(pos)
        pos += len(line)
    if not matches:
        return None
    cutoff = int(len(text) * 0.35)
    later = [m for m in matches if m >= cutoff]
    return later[0] if later else matches[-1]


def body_region(text: str) -> str:
    text = strip_code_blocks(text)
    ref = find_reference_start(text)
    if ref is not None:
        text = text[:ref]
    return text


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
    seen = set(); out: List[str] = []
    for m in URL_RE.finditer(text):
        u = clean_url(m.group(1))
        if not u or u.lower().startswith(("mailto:", "tel:", "javascript:", "data:", "file:")):
            continue
        if u.lower() not in seen:
            seen.add(u.lower()); out.append(u)
    return out


def marker_norm(marker: str) -> str:
    m = marker.strip()
    m = m.strip("[]()^:.")
    m = m.translate(SUP_TO_DIGIT)
    return m


def looks_bibliographic(line: str) -> bool:
    # Reject numbered reference entries like "37. Omar R, de Waal A: ... http://... 1995".
    if re.match(r"^\s*\d{1,4}[.)]\s+[A-Z][A-Za-z'’`-]+\s+[A-Z]", line):
        return True
    if re.search(r"\b(19|20)\d{2}[a-z]?\b", line) and re.search(r"\b(journal|proc\.|proceedings|press|university|vol\.|pages?|doi|lancet|nature|science|bioinformatics)\b", line, re.I):
        return True
    return False


# Standalone footnote definitions.  This intentionally requires the marker to
# be at the line start and the URL to appear immediately or after a short note,
# which prevents most numbered bibliography entries from being misread.
FOOTNOTE_LINE_RE = re.compile(r"""
^
\s*
(?:
  \[\^?(?P<bracket>[A-Za-z0-9]+)\]\s*:?
 | \^(?P<caret>[A-Za-z0-9]+)\s*:?
 | (?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹]+)
 | (?P<num>\d{1,3})
 | (?P<sym>[*†‡§¶])
)
\s*[:.)\-]?\s*
(?P<body>.*?(?:https?://|ftp://|www\.|doi\s*:\s*10\.|10\.\d{4,9}/|(?:github|gitlab|bitbucket)\.com/|(?:[A-Za-z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai)\b).*)
$""", re.I | re.X)

INLINE_FOOTNOTE_RE = re.compile(r"\\footnote\{([^{}]*(?:https?://|www\.|ftp://|doi\s*:|10\.)[^{}]*)\}", re.I)


def extract_footnotes_and_clean(text: str) -> Tuple[List[Footnote], str]:
    footnotes: List[Footnote] = []
    remove_ranges: List[Tuple[int, int]] = []

    # Inline LaTeX footnotes first.  They are restored in place.
    for k, m in enumerate(INLINE_FOOTNOTE_RE.finditer(text), start=1):
        content = re.sub(r"\s+", " ", m.group(1).strip())
        us = urls_in(content)
        if us:
            footnotes.append(Footnote(marker=f"inline{k}", marker_norm=f"inline{k}", footnote_text=content,
                                      urls=us, start=m.start(), end=m.end(), kind="inline"))

    # Standalone footnote lines.
    pos = 0
    for line in text.splitlines(True):
        raw_line = line.rstrip("\n")
        m = FOOTNOTE_LINE_RE.match(raw_line)
        if m and not looks_bibliographic(raw_line):
            marker = next(g for g in [m.group("bracket"), m.group("caret"), m.group("sup"), m.group("num"), m.group("sym")] if g)
            body = m.group("body").strip()
            us = urls_in(body)
            if us:
                footnotes.append(Footnote(marker=marker, marker_norm=marker_norm(marker), footnote_text=body,
                                          urls=us, start=pos, end=pos + len(line), kind="standalone"))
                remove_ranges.append((pos, pos + len(line)))
        pos += len(line)

    chars = list(text)
    for s, e in remove_ranges:
        for i in range(s, e):
            # Preserve newlines so offsets remain stable.
            if chars[i] != "\n":
                chars[i] = " "
    cleaned = "".join(chars)
    return footnotes, cleaned


def paragraph_spans(text: str) -> List[Paragraph]:
    paras: List[Paragraph] = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        raw = m.group(0)
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
            if left_word in ABBREVIATIONS or re.search(r"\b(?:https?|ftp|www|doi)\.?$", left_word):
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
                seg = paragraph[start:j]
                sent = seg.strip()
                if sent:
                    lead = len(seg) - len(seg.lstrip()); trail = len(seg.rstrip())
                    out.append(Sentence(sent, start + lead, start + trail))
                start = k; i = k; continue
        i += 1
    if start < n:
        seg = paragraph[start:]
        sent = seg.strip()
        if sent:
            lead = len(seg) - len(seg.lstrip())
            out.append(Sentence(sent, start + lead, n))
    return out


def locate_paragraph(paras: Sequence[Paragraph], pos: int) -> Optional[Tuple[int, Paragraph]]:
    for i, p in enumerate(paras):
        if p.start <= pos <= p.end:
            return i, p
    return None


def marker_variants(marker: str) -> List[str]:
    n = marker_norm(marker)
    vals = [marker, n]
    if n.isdigit():
        vals.append(n.translate(DIGIT_TO_SUP))
    # unique, longest first
    uniq: List[str] = []
    for v in vals:
        if v and v not in uniq:
            uniq.append(v)
    return sorted(uniq, key=len, reverse=True)


def find_marker_occurrences(text: str, fn: Footnote) -> List[MarkerOcc]:
    if fn.kind == "inline":
        return [MarkerOcc(fn.start, fn.end, text[fn.start:fn.end], 0)]
    occs: List[MarkerOcc] = []
    n = marker_norm(fn.marker)
    if not n:
        return occs
    variants = marker_variants(fn.marker)
    patterns: List[re.Pattern] = []
    for v in variants:
        ev = re.escape(v)
        if n.isdigit():
            # Superscript or baseline marker attached to a word, or bracket marker.
            patterns.extend([
                re.compile(rf"\[{ev}\]"),
                re.compile(rf"(?<=[A-Za-z_\)\]]){ev}(?=[,.;:!?\)\]\s])"),
                re.compile(rf"(?<=[A-Za-z_\)\]])\^?{ev}(?=[,.;:!?\)\]\s])"),
            ])
        else:
            patterns.append(re.compile(rf"(?<=[A-Za-z_\)\]]){ev}(?=[,.;:!?\)\]\s])"))
    seen = set()
    for pat in patterns:
        for m in pat.finditer(text):
            s, e = m.span()
            if (s, e) in seen:
                continue
            seen.add((s, e))
            # Do not pick occurrences inside URLs or obvious reference/citation brackets far away.
            local = text[max(0, s - 80): min(len(text), e + 80)]
            if URL_RE.search(local) and s > fn.start - 50 and s < fn.end + 50:
                continue
            distance = abs(s - fn.start)
            # Prefer markers before the footnote block, then nearby after (for split sentences).
            directional_penalty = 0 if s <= fn.start else 500
            # Penalize title/author-like area at very beginning, unless the footnote itself is there.
            front_penalty = 2000 if s < 1000 and fn.start > 2500 else 0
            occs.append(MarkerOcc(s, e, text[s:e], distance + directional_penalty + front_penalty))
    return sorted(occs, key=lambda o: o.score)


def restore_marker_in_sentence(sentence: str, local_s: int, local_e: int, footnote_text: str) -> str:
    # Insert exactly where marker surface was.  Ensure a readable space before the bracket.
    before = sentence[:local_s]
    after = sentence[local_e:]
    insert = f"[{footnote_text}]"
    if before and not before[-1].isspace() and not before.endswith("["):
        insert = " " + insert
    return before + insert + after


def find_sentence_for_occurrence(text: str, pos: int) -> Optional[Tuple[Paragraph, int, List[Sentence]]]:
    paras = paragraph_spans(text)
    located = locate_paragraph(paras, pos)
    if not located:
        return None
    _, para = located
    # local position after joining single newlines may differ.  Approximate by
    # counting non-newline characters from paragraph start to pos.
    raw_prefix = text[para.start:pos]
    local_pos = len(re.sub(r"(?<!-)\n", " ", raw_prefix))
    sents = sentence_spans(para.text)
    for i, s in enumerate(sents):
        if s.start <= local_pos <= s.end:
            return para, i, sents
    return None


def process_file(md: MdFile, emit_unmatched: bool = False) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    raw_body = body_region(md.text)
    footnotes, cleaned = extract_footnotes_and_clean(raw_body)
    rows: List[Dict[str, object]] = []
    unmatched: List[Dict[str, object]] = []
    seen = set()

    for fn in footnotes:
        occs = find_marker_occurrences(cleaned, fn)
        if not occs:
            rec = {"paper_id": md.paper_id, "file": md.file, "marker": fn.marker, "urls": fn.urls,
                   "footnote_text": fn.footnote_text, "reason": "marker_not_found"}
            unmatched.append(rec)
            continue
        occ = occs[0]
        fs = find_sentence_for_occurrence(cleaned, occ.start)
        if not fs:
            unmatched.append({"paper_id": md.paper_id, "file": md.file, "marker": fn.marker, "urls": fn.urls,
                              "footnote_text": fn.footnote_text, "reason": "sentence_not_found"})
            continue
        para, sent_idx, sents = fs
        target = sents[sent_idx].text
        # Map occurrence offset into joined paragraph coordinates.
        raw_para_prefix = cleaned[para.start:occ.start]
        occ_local = len(re.sub(r"(?<!-)\n", " ", raw_para_prefix))
        surf_len = len(occ.surface)
        # For inline \footnote, the local surface includes the full command; for
        # standalone markers the surface is just the marker/bracket.
        local_s = max(sents[sent_idx].start, min(occ_local, sents[sent_idx].end)) - sents[sent_idx].start
        local_e = min(len(target), local_s + surf_len)
        restored = restore_marker_in_sentence(target, local_s, local_e, fn.footnote_text)
        preceding = sents[sent_idx - 1].text if sent_idx > 0 else None
        trailing = sents[sent_idx + 1].text if sent_idx + 1 < len(sents) else None
        for url in fn.urls:
            key = (md.paper_id, fn.marker_norm, url.lower(), target.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "paper_id": md.paper_id,
                "file": md.file,
                "location": "footnote",
                "footnote_marker": fn.marker,
                "url_output": url,
                "footnote_text": fn.footnote_text,
                "preceding_output": preceding,
                "target_output": restored,
                "original_sentence": target,
                "trailing_output": trailing,
                "at_paragraph_start": preceding is None,
                "at_paragraph_end": trailing is None,
                "marker_surface": occ.surface,
                "marker_start_char": occ.start,
                "footnote_start_char": fn.start,
                "kind": fn.kind,
            })
    return rows, unmatched


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    fields = [
        "paper_id", "file", "location", "footnote_marker", "url_output", "footnote_text",
        "preceding_output", "target_output", "original_sentence", "trailing_output",
        "at_paragraph_start", "at_paragraph_end", "marker_surface", "marker_start_char",
        "footnote_start_char", "kind",
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
    ap = argparse.ArgumentParser(description="Restore URL-bearing footnotes in olmOCR Markdown.")
    ap.add_argument("--input", required=True, help="Input .zip, directory, or .md file")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_jsonl", default=None, help="Optional JSONL output path")
    ap.add_argument("--unmatched_jsonl", default=None, help="Optional JSONL of URL footnotes whose marker/sentence was not found")
    args = ap.parse_args(argv)

    all_rows: List[Dict[str, object]] = []
    all_unmatched: List[Dict[str, object]] = []
    n_files = 0
    for md in iter_markdown_files(args.input):
        n_files += 1
        rows, unmatched = process_file(md)
        all_rows.extend(rows); all_unmatched.extend(unmatched)
    write_csv(all_rows, args.out_csv)
    write_jsonl(all_rows, args.out_jsonl)
    write_jsonl(all_unmatched, args.unmatched_jsonl)
    print(f"Processed {n_files} Markdown file(s); wrote {len(all_rows)} restored footnote row(s) to {args.out_csv}; unmatched={len(all_unmatched)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
