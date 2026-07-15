#!/usr/bin/env python3
"""
Footnote URL extraction and citing-sentence restoration.

Two calls per page:
  1. DETECT   bottom strip of the page plus its text layer -> URL-bearing
              footnotes (marker, verbatim footnote text, URLs) as strict JSON.
  2. EXTRACT  previous-page bottom strip, full page, next-page top strip, plus
              the detected footnotes -> for each footnote, the citing sentence
              that carries its superscript marker, that sentence with the
              footnote content spliced in at the marker position, and the
              preceding and trailing sentences of the same paragraph.

Pages with no URL-bearing footnote never reach the second call. Pages after a
references heading are skipped without an API call.

    export OPENAI_API_KEY=...
    python footnote_extract.py --input <pdf_dir> --output <out_dir>

Requires: pymupdf, openai, tqdm
"""

import argparse
import base64
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("ERROR: PyMuPDF not installed. Run: pip install pymupdf")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("ERROR: openai not installed. Run: pip install openai")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# ------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------

DETECT_MODEL  = "gpt-5-mini"
RESTORE_MODEL = "gpt-5-mini"

DETECT_DPI        = 300    # bottom strip sent to DETECT
DETECT_START_FRAC = 0.50   # crop starts 50% down the page
RESTORE_DPI       = 300    # full page sent to EXTRACT
TOP_DPI           = 300    # previous / next page strips
TOP_FRAC          = 0.50    # fraction of the next page top to include
PREV_PAGE_FRAC    = 0.50   # fraction of the previous page bottom to include
MAX_NEXT_PAGES    = 1      # number of next-page strips to send

DELAY_SECONDS = 0.3        # pause between API calls


# ------------------------------------------------------------------
#  SCHEMAS
# ------------------------------------------------------------------

DETECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_1_based": {"type": "integer"},
        "has_url_footnotes": {"type": "boolean"},
        "footnotes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "marker":        {"type": "string"},
                    "footnote_text": {"type": "string"},
                    "urls":          {"type": "array", "items": {"type": "string"}},
                },
                "required": ["marker", "footnote_text", "urls"],
            },
        },
    },
    "required": ["page_1_based", "has_url_footnotes", "footnotes"],
}


EXTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "base_page_1_based": {"type": "integer"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                   
                    "original_sentence":    {"type": "string"},
                    "restored_sentence":    {"type": "string"},
                    "footnote_marker":      {"type": "string"},
                    "footnote_content":     {"type": "string"},
                    "url": {"type": "array", "items": {"type": "string"}},
                    "sentence_spans_pages":      {"type": "boolean"},
                    "sentence_starts_prev_page": {"type": "boolean"},
                    "needs_more_pages":          {"type": "boolean"},
        
                    "preceding_sentence": {"type": ["string", "null"]},
                    "trailing_sentence":  {"type": ["string", "null"]},
                    "at_paragraph_start": {"type": "boolean"},
                    "at_paragraph_end":   {"type": "boolean"},
                },
                "required": [
                    "original_sentence",
                    "restored_sentence",
                    "footnote_marker",
                    "footnote_content",
                    "url",
                    "sentence_spans_pages",
                    "sentence_starts_prev_page",
                    "needs_more_pages",
                    "preceding_sentence",
                    "trailing_sentence",
                    "at_paragraph_start",
                    "at_paragraph_end",
                ],
            },
        },
    },
    "required": ["base_page_1_based", "items"],
}


# ------------------------------------------------------------------
#  DETECT PROMPT
# ------------------------------------------------------------------

def make_detect_prompt(page_1_based: int, text_layer: str = "") -> str:
    text_section = ""
    if text_layer:
        text_section = (
            "\nTEXT LAYER (extracted from the PDF for this same bottom strip):\n"
            "---\n"
            + text_layer[:3000]
            + "\n---\n"
            "\n"
            "NO AUTO-CORRECTION (strict)\n"
            "- The page image is the authoritative source for every URL character. Your background knowledge about website names, domain conventions, or correct spellings must not influence the transcribed URL.\n"
            "\n"
            "WHEN TO USE THE TEXT LAYER FOR URL CHARACTERS\n"
            "Use the text layer to resolve a URL character only when you are genuinely uncertain what that character is from the image alone — that is, when the printed glyph is visually indistinguishable between two or more possibilities at that font size and rendering. Do not use the text layer to override a character you can already read with confidence from the image.\n"
            "\n"
            "Visually ambiguous character pairs that may cause genuine uncertainty in PDF fonts include, but are not limited to:\n"
            "- l (lowercase L) / I (uppercase I) / 1 (digit one)\n"
            "- O (uppercase O) / 0 (digit zero)\n"
            "- rn / m\n"
            "- B / 8\n"
            "- S / 5\n"
            "- Z / 2\n"
            "- cl / d\n"
            "- vv / w\n"
            "- D / 0 (in some heavy serif fonts)\n"
            "- G / 6 (in some fonts)\n"
            "\n"
            "When you face genuine visual uncertainty over one of these pairs and the text layer contains a character at that position, prefer the text layer character for that specific position. Do not apply this rule to characters you can already read clearly from the image — the text layer is a disambiguation aid for genuinely uncertain glyphs only, not a general override.\n"
            "\n"
            "Do not use the text layer to invent a URL that is not visibly present in the image.\n"
        )
    return f"""You are shown the BOTTOM PORTION of a PDF page — this is the footnote area of page {page_1_based}.{text_section}

YOUR TASK:
Extract EVERY footnote printed on this image that contains at least one web URL.

WHAT COUNTS AS A URL:
Any address that identifies a resource retrievable over the web, with or without an
explicit scheme. This covers scheme-prefixed addresses (http, https, ftp, and similar
transfer schemes), addresses beginning with "www.", and bare domain names, with or
without a path. Never skip an address because it carries no scheme, and never skip one
because of which site it points to.

NOT a URL:
  - a local file path with no domain, such as /home/user/file.txt
  - an email address, with or without a "mailto:" prefix

  • BIBLIOGRAPHIC REFERENCES — this is the most important exclusion:
    Any entry that contains BOTH of the following is a bibliographic reference — skip it entirely,
    even if it also contains a URL:
      1. One or more author names (surnames + initials, or full names)
      2. A publication venue + year (journal, conference, book, report, volume, pages, year)

    This applies regardless of how the entry is formatted — it may or may not be numbered,
    may use [N], N., or no prefix at all. The numbering style is irrelevant.
    The CONTENT is what matters: author names + venue/year = reference, always skip.

    A FOOTNOTE contains none of that. It is short text whose primary content is a URL
    or a brief functional description + URL, with no author names or publication venue.

      FOOTNOTE (include):   "4 http://example.org/p/gridsolver"
                            "* Available at http://tools.example.net/parser"
                            "¹⁷ http://data.example.org/archive/spectra"
 
      REFERENCE (skip):     "[4] A. Fenwick, R. Oduya, T. Marsh. A sparse grid method for... 2012."
                            "Halvorsen L, Iyer M. A revised algorithm for adaptive meshing. 1997."
                            "17. L. Halvorsen and M. Iyer. Journal of Numerical Methods, 6:229–269, 1997."

    Ask yourself: "Does this entry describe a published work by named authors?"
    If yes → REFERENCE, skip. If no → potential FOOTNOTE, check for URL.

FOR EACH QUALIFYING FOOTNOTE:
  marker        = the footnote marker exactly as printed (number, letter, *, †, ‡, §, ¶, etc.)
  footnote_text = the COMPLETE verbatim footnote text — every single word as printed.
                  Do NOT summarise, shorten, or paraphrase even one word.
                  If a URL is split across two lines by a hyphen or line-break, join it into
                  the single correct unbroken URL in footnote_text.
  urls          = list of every URL/web address in this footnote (all of them)

ACCURACY RULES — NON-NEGOTIABLE:
• NEVER invent, guess, or fabricate any URL, word, or marker.
• If there is no clearly visible footnote marker attached to the entry, do not extract it.
• NEVER truncate a URL with "..." — transcribe it fully as printed.
• NEVER skip a footnote because its URL lacks "http" or uses an unusual scheme.
• If a footnote number/letter is printed directly before the URL with no space (e.g. "1http://..."),
  set marker to just the number/letter and strip the leading marker digit from footnote_text and urls.
• If part of a URL is genuinely illegible and you cannot form a usable URL, skip that footnote
  entirely — a partial URL is worse than no URL.
• PRESERVE www. EXACTLY — if the printed URL starts with "www." keep it; if it does not,
  do not add it. Never normalise http://www.example.com to http://example.com or vice versa.
  Transcribe the URL character-for-character as it is physically printed on the page.
• If a URL is wrapped in parentheses like (http://example.com/path), transcribe the URL
  without the surrounding parentheses.

Return strict JSON matching the schema. If no qualifying footnotes exist, return has_url_footnotes=false and footnotes=[].
""".strip()


# ------------------------------------------------------------------
#  EXTRACT PROMPT  (restoration + paragraph context, one call)
# ------------------------------------------------------------------

EXTRACT_SYSTEM = r"""
You are a meticulous document analyst specialising in academic, legal, and technical PDFs.

════════════════════════════════════════
IMAGES YOU WILL RECEIVE
════════════════════════════════════════
  IMAGE 0  →  BOTTOM STRIP of the PREVIOUS page — provided SOLELY to recover the opening
               of a citing sentence that started on the previous page and continues here.
               Do NOT extract or use footnotes from IMAGE 0.
  IMAGE 1  →  FULL PRIMARY PAGE — the page you are analysing. All footnote markers live here.
  IMAGE 2+ →  TOP STRIPS of the FOLLOWING page(s) — provided SOLELY to complete sentences
               that overflow from IMAGE 1. Do NOT extract or use footnotes from these images.

════════════════════════════════════════
FUNDAMENTAL RULE
════════════════════════════════════════
The footnotes have already been detected and are provided to you as structured JSON.
You do NOT need to find or re-read footnotes from the images.
Your only jobs are:
  1. Find the citing sentence in IMAGE 1 body text for each provided footnote marker.
  2. If that sentence is incomplete (started on the previous page or continues onto the next),
     use IMAGE 0 and/or IMAGE 2+ to complete it.
  3. Restore the footnote content into the complete sentence.

════════════════════════════════════════
TASK PER FOOTNOTE
════════════════════════════════════════
For each footnote in the provided JSON:

  a. Find the sentence in the BODY TEXT of IMAGE 1 where the footnote marker appears
     as a SUPERSCRIPT citation — a small raised character printed above the baseline.
     The body text is the main paragraph text — NOT the footnote zone at the bottom.

     IMPORTANT - SUPERSCRIPT vs INLINE CITATION — critical distinction:
     An inline reference like [3], (3), or [Smith, 2012] sits ON the text baseline
     inside brackets.  These are NOT footnote markers — they point to the reference
     list, not to a footnote.  If the provided marker number appears ONLY as [N] on
     the baseline and NEVER as a raised superscript, produce NO output item for it.

     IMPORTANT - MARKER NOT FOUND — produce NO output:
     If you search the entire body text of IMAGE 1 and cannot find the footnote marker
     as a raised superscript ANYWHERE, produce NO output item for that footnote.
     Do NOT fall back to finding the sentence that contains the URL and restoring it there.
     A sentence that already contains the URL inline (without any superscript marker) is
     NOT a citing sentence — it is a sentence that happens to share the same URL.
     Restoring into it would produce a false positive. Skip it entirely.

  b. IMPORTANT - CAPTURE THE COMPLETE CITING SENTENCE — NEVER TRUNCATE:

       CASE A — Sentence fully contained on IMAGE 1:
         Has a visible opening capital letter AND closing punctuation (. ? !) on IMAGE 1.
         → Transcribe fully from opening capital to closing punctuation.
         → Set sentence_spans_pages = false, sentence_starts_prev_page = false.

       CASE B — Sentence ends on IMAGE 1 but STARTED on the previous page:
         The first visible text of this sentence on IMAGE 1 is a mid-sentence continuation
         (i.e. it does not start with a capital letter that opens a new sentence, OR the
         text at the very top of IMAGE 1 body is clearly a continuation of a previous sentence).
         → Look at IMAGE 0 (bottom strip of previous page) to find where this sentence began.
         → Reconstruct: [opening from IMAGE 0] + [continuation on IMAGE 1 to closing punctuation]
         → Set sentence_spans_pages = true, sentence_starts_prev_page = true.

       CASE C — Sentence starts on IMAGE 1 but is cut off at the bottom (no closing punctuation):
         → Look at IMAGE 2 (and IMAGE 3 if provided) to find the continuation.
         → Reconstruct: [text from IMAGE 1] + [continuation up to closing punctuation]
         → Set sentence_spans_pages = true, sentence_starts_prev_page = false.
         → If even the additional images are not enough, set needs_more_pages = true.

       CASE D — Sentence BOTH started on previous page AND ends on a following page:
         → Use IMAGE 0 to find the opening AND IMAGE 2+ to find the closing.
         → Reconstruct: [opening from IMAGE 0] + [IMAGE 1 text] + [closing from IMAGE 2+]
         → Set sentence_spans_pages = true, sentence_starts_prev_page = true.

  c. Build restored_sentence by replacing each footnote marker IN-PLACE:
       Find the exact position of the footnote marker (superscript number/symbol) within
       the sentence text, and replace it with the full footnote_text inline at that position.

       Example — marker "4" in sentence:
         BEFORE: "...using the back40computing library⁴ means that this remains..."
         AFTER:  "...using the back40computing library [http://code.google.com/p/back40computing] means that this remains..."

       Rules:
       • Replace the marker itself with " [" + footnote_text + "]"
       • The footnote_text goes exactly where the superscript marker was — not at the end
       • Do NOT append anything to the end of the sentence
       • Use the exact footnote_text from the provided JSON — do NOT re-read or alter it
       • If the marker appears as a superscript digit directly attached to a word (e.g. "library⁴"),
         place the inline citation immediately after that word with a space before "["

  d. ONE SENTENCE = ONE OUTPUT ITEM. A sentence ends at its closing punctuation (. ? !)
     followed by a space and a capital letter. That period is a hard wall — never cross it.

     SCENARIO 1 — Multiple URL-footnote markers in the SAME sentence:
       "polySolve [9] is available as meshCore¹⁷ and gridPack¹⁸ as GridPack-Solve."
       → Produce ONE SEPARATE ITEM per URL-containing footnote marker.
         Each item is a copy of the same sentence with ONLY that one marker restored inline;
         the other markers remain as printed (unchanged superscript digits).
         Example — if footnotes 17 and 18 both have URLs:
           Item 1: "polySolve [9] is available as meshCore [http://tools.example.net/meshcore] and gridPack¹⁸ as GridPack-Solve."
           Item 2: "polySolve [9] is available as meshCore¹⁷ and gridPack [http://example.org/gridpack/suite] as GridPack-Solve."
         The footnote_content field must contain the COMPLETE verbatim text of that footnote — never shortened.
 
     SCENARIO 2 — Each marker in its OWN sentence (the common case that must NOT be merged):
       "polySolve [9] is publicly available as part of the package meshCore¹⁷. gridPack [17] is
        publicly available as GridPack-Solve from example.org¹⁸."
       → TWO separate items:
           Item 1: "polySolve [9] is publicly available as part of the package meshCore [http://tools.example.net/meshcore]."
           Item 2: "gridPack [17] is publicly available as GridPack-Solve from example.org [http://example.org/gridpack/suite]."
 

     The fact that footnotes 17 and 18 both appear on this page does NOT mean their
     citing sentences should be joined. Process each sentence independently.
     A period followed by a capital letter always starts a new item.

════════════════════════════════════════
ACCURACY RULES — NON-NEGOTIABLE
════════════════════════════════════════
• Use the footnote_text EXACTLY as given in the JSON — do not re-read from the image.
• URLs must be copied CHARACTER-FOR-CHARACTER from the JSON — never re-read a URL
  from the image.  The JSON version has already been verified against the PDF text
  layer and is authoritative.  Even if the image looks different, use the JSON.
• NEVER truncate original_sentence — always find its full opening AND closing punctuation.
• NEVER hallucinate sentence text, markers, or footnote content.
• NEVER extract footnotes from IMAGE 0, IMAGE 2, or any image other than IMAGE 1.
• original_sentence must include every word from its opening capital letter to its closing
  punctuation, even if that requires reading IMAGE 0 and/or IMAGE 2+.
• restored_sentence MUST have the footnote content inline at the marker position — NEVER at the end.
• Do NOT use "[CITE: ...]" or any wrapper tag — just " [" + footnote_text + "]" at the marker spot.

════════════════════════════════════════
JOB B — PARAGRAPH CONTEXT
════════════════════════════════════════
For EACH sentence you produced above, also return the sentences immediately
before and after it WITHIN THE SAME PARAGRAPH.

  preceding_sentence  the sentence immediately before it, or null
  trailing_sentence   the sentence immediately after it, or null

THE TARGET IS YOUR OWN SENTENCE.
You are not given a list of target sentences to locate. The target for Job B is
the original_sentence you have just produced for that footnote. You already know
where it sits on IMAGE 1. Work outward from there.

────────────────────────────────────────
READING THE PAGE
────────────────────────────────────────
COLUMNS. Scholarly PDFs are frequently two-column or multi-column. Identify the
column the sentence sits in by the visible gutter and stay inside it. The bottom
of one column and the top of the next are not adjacent text.

SENTENCES. A neighbour is a WHOLE sentence, from its opening capital letter to
its closing punctuation, however many printed lines it wraps across. Never return
a single wrapped line or any other fragment. A period inside an abbreviation is
not a sentence end (Fig., Eq., Dr., et al., i.e., e.g.); it ends a sentence only
when a new sentence visibly begins after it.

FOOTNOTE ZONE. The smaller-font block at the foot of the page, often under a
rule, is never body text and never a neighbour.

LAYOUT HINTS. Page-level hints from the PDF text layer may be supplied (column
count, column split, median line height). If a hint conflicts with the image,
trust the image.

────────────────────────────────────────
PARAGRAPH BOUNDARIES — RETURN NULL IN THAT DIRECTION
────────────────────────────────────────
Return null in a direction if ANY of these lies between the citing sentence and
the candidate neighbour:

  • a blank line, or a vertical gap noticeably larger than the normal line
    spacing of that column
  • a first-line paragraph indent. Judge indentation against the column's normal
    left edge, and apply it ONLY when the candidate or the citing sentence starts
    at the BEGINNING of a printed line. A sentence that starts mid-line, with
    prose to its left on the same line, is never a paragraph start, however far
    from the left edge it sits.
  • a section heading, a figure or table caption, or a standalone title
  • a displayed equation, a centred block, or a theorem-like block
  • a bulleted or numbered list. A prose lead-in ending in a colon is still prose
    and may be a neighbour; the bullets that follow it are a boundary.
  • the footnote zone
  • a column break

NOT boundaries on their own: a change of font, an ordinary line wrap, or a
discourse cue such as "For example," / "However," / "Therefore,".

Running headers, footers, and page numbers are never neighbours.

────────────────────────────────────────
FINDING THE NEIGHBOUR
────────────────────────────────────────
SAME-LINE NEIGHBOURS. The neighbour is often on the SAME printed line. If the
citing sentence begins mid-line, look to its LEFT on that line: a sentence ending
there is the preceding sentence. If it ends mid-line and prose continues to its
RIGHT, the next sentence starts there.

ASSEMBLING A FULL NEIGHBOUR. Find the nearest sentence end above (or sentence
start below), then extend it into the complete sentence: scan upward, or downward,
within the same column and paragraph, taking in the wrapped lines that belong to
it, until you reach that sentence's own opening capital letter or closing
punctuation, or a boundary from the list above, or the edge of the page. Only then
may you look at IMAGE 0 or IMAGE 2+, and only if the paragraph clearly continues
across the page break.

A neighbour that begins with a comma, a semicolon, a colon, a closing bracket, or
a connector word such as "and", "or", "but", "which", "that", "where", "while",
"including" is probably the tail of a sentence that started higher up. Scan upward
and take in the earlier lines. Do not use capitalisation alone as a signal: a
sentence can open with a lowercase variable or identifier.

WHEN IN DOUBT, RETURN NULL. If you cannot tell whether the candidate is in the
same paragraph, null is correct. Precision matters more than a guess.

────────────────────────────────────────
TRANSCRIBING NEIGHBOURS
────────────────────────────────────────
Transcribe each neighbour exactly as printed: spelling, capitalisation,
punctuation, visible superscripts. Return it as a single-line string. The only
permitted correction is rejoining a word broken by an end-of-line hyphen. Do not
paraphrase, tidy, complete, or invent text. Do NOT restore footnote content into
a neighbour: neighbours are returned as printed.

════════════════════════════════════════
OUTPUT
════════════════════════════════════════
One item per citing sentence, carrying the Job A fields (original_sentence,
restored_sentence, footnote_marker, footnote_content, url, sentence_spans_pages,
sentence_starts_prev_page, needs_more_pages) and the Job B fields
(preceding_sentence, trailing_sentence) for that same sentence.

Return strict JSON matching the schema.
""".strip()


def make_extract_user_prompt(page_1_based: int,
                             footnotes: List[Dict[str, Any]],
                             layout_hints: Dict[str, Any],
                             has_prev: bool) -> str:
    fn_json = json.dumps(footnotes, ensure_ascii=False, indent=2)
    hints_json = json.dumps(layout_hints, ensure_ascii=False)
    img0 = ("IMAGE 0 = bottom of the previous page.\n" if has_prev
            else "IMAGE 0 = not provided (this is the first page).\n")
    return (
        f"PRIMARY PAGE: {page_1_based}\n\n"
        f"URL footnotes already detected on this page:\n{fn_json}\n\n"
        f"Page-level layout hints (trust the image over these):\n{hints_json}\n\n"
        + img0 +
        "IMAGE 1 = the full primary page. Every footnote marker is here.\n"
        "IMAGE 2+ = top of the following page(s).\n\n"
        "For each footnote above, find its citing sentence in the body text of "
        "IMAGE 1 by its superscript marker, complete the sentence across pages "
        "if it runs over, splice the footnote content in at the marker "
        "position, and return the preceding and trailing sentences of the same "
        "paragraph. Return strict JSON."
    )


# ------------------------------------------------------------------
#  RENDERING
# ------------------------------------------------------------------

def render_png(doc: fitz.Document, page0: int, dpi: int,
               clip: Optional[fitz.Rect] = None) -> bytes:
    page = doc.load_page(page0)
    pix  = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    return pix.tobytes("png")


def bottom_clip(doc: fitz.Document, page0: int, start_frac: float) -> fitz.Rect:
    r = doc.load_page(page0).rect
    return fitz.Rect(0, r.height * start_frac, r.width, r.height)


def top_clip(doc: fitz.Document, page0: int, height_frac: float) -> fitz.Rect:
    r = doc.load_page(page0).rect
    return fitz.Rect(0, 0, r.width, r.height * height_frac)


def b64_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def make_blank_png(width_pt: int = 612, height_pt: int = 200,
                   dpi: int = 300) -> bytes:
    """White PNG used as a stand-in next-page top when on the last page."""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    scale = dpi / 72.0
    w = int(width_pt * scale)
    h = int(height_pt * scale)
    compressed = zlib.compress(b"".join(b"\x00" + b"\xff" * (w * 3) for _ in range(h)))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


# ------------------------------------------------------------------
#  TEXT LAYER  (no API call)
# ------------------------------------------------------------------

_REF_HEADINGS = frozenset({
   "references", "bibliography", "works cited", "literature cited",
    "literature", "sources", "citations", "notes",
    # Numbered variants that appear as headings
    "reference list", "list of references",
})


def page_is_references_section(doc: fitz.Document, page0: int) -> bool:
    """
    Return True if the first non-empty text block on this page is a known
    references or bibliography heading. Uses the text layer only, no API call.

    The check fires only on the page where the heading appears. That page is
    still processed, since a footnote can sit at the bottom of the first
    references page; only later pages are skipped. Continuation pages of a
    multi-page reference list do not start with a heading and are not caught
    here, but the detect prompt's bibliographic-entry rule (author plus
    venue and year is skipped) filters those entries individually.
    """
    page   = doc.load_page(page0)
    blocks = page.get_text("blocks")
    for block in blocks:
        text = block[4].strip()
        if not text:
            continue
        first_line = text.splitlines()[0].strip().lower().rstrip(":").rstrip(".")
        if first_line in _REF_HEADINGS:
            return True
        break  # first substantive block is not a heading → body page
    return False


def extract_bottom_text(doc: fitz.Document, page0: int, start_frac: float) -> str:
    """
    Extract raw text from the bottom strip via PyMuPDF.
    Passed to the detect model as URL-verification context so it can
    cross-check its visual OCR reading against the character-perfect text.
    Returns empty string for scanned / image-only PDFs.
    """
    page = doc.load_page(page0)
    rect = page.rect
    clip = fitz.Rect(0, rect.height * start_frac, rect.width, rect.height)
    return page.get_text("text", clip=clip).strip()


# ------------------------------------------------------------------
#  URL CLEANING
# ------------------------------------------------------------------

_WRAPPERS = [("(", ")"), ("[", "]"), ("<", ">"), ('"', '"'), ("'", "'")]


def _looks_like_url(s: str) -> bool:
    """Host-agnostic: a URL has a scheme, or a dot, and never whitespace."""
    if not s or any(c.isspace() for c in s):
        return False
    return "://" in s or "." in s


def clean_url(url: str) -> str:
    """Strip wrapping punctuation absorbed into the URL, whatever the host.

      (http://example.com/path)  -> http://example.com/path
      (example.org/tools/v2)     -> example.org/tools/v2
      <www.example.com>          -> www.example.com
      http://example.com.        -> http://example.com

    Parentheses that belong to the path survive:
      http://en.wikipedia.org/wiki/A_(disambiguation)  -> unchanged
    """
    url = url.strip()
    if not url:
        return url

    for opener, closer in _WRAPPERS:
        if url.startswith(opener) and url.endswith(closer):
            inner = url[1:-1]
            if _looks_like_url(inner):
                url = inner
                break

    if url.startswith("(") and ")" not in url:
        url = url[1:]
    url = url.rstrip(".,;")
    if url.endswith(")") and "(" not in url:
        url = url.rstrip(")")
    return url.strip()


def filter_mailto(footnotes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean, deduplicate, and strip mailto-only footnotes."""
    out = []
    for fn in footnotes:
        seen_u: set = set()
        cleaned: List[str] = []
        for u in fn.get("urls", []):
            if not isinstance(u, str):
                continue
            u = clean_url(u.strip())
            if not u or u.lower().startswith("mailto:"):
                continue
            if u not in seen_u:
                seen_u.add(u)
                cleaned.append(u)
        if cleaned:
            out.append({
                "marker":        fn.get("marker", ""),
                "footnote_text": fn.get("footnote_text", "").strip(),
                "urls":          cleaned,
            })
    return out


def norm_key(s: str) -> str:
    """Normalised lowercase key for deduplication."""
    return " ".join(s.strip().split()).lower()


def sentence_complete(s: str) -> bool:
    t = " ".join(s.strip().split())
    if not t:
        return True
    if t[-1] in ".?!":
        return True
    # handle closing quote/bracket after punctuation
    for closer in ('"', "'", "\u201d", "\u2019", ")", "]", "}"):
        if t.endswith(closer):
            inner = t[:-1].rstrip()
            if inner and inner[-1] in ".?!":
                return True
    return False


# ------------------------------------------------------------------
#  LAYOUT HINTS
# ------------------------------------------------------------------


def extract_layout_hints(doc: fitz.Document, page0: int) -> Dict[str, Any]:
    """
    Heuristic layout hints from the PDF text layer to reduce wrong-column and
    false paragraph joins. Passed to the extract call as context only. If a hint
    conflicts with the image, the model is told to trust the image.
    """
    page = doc.load_page(page0)
    r = page.rect
    w = float(r.width)

    try:
        d = page.get_text("dict")
    except Exception:
        return {"page_width": w, "two_column": False}

    lines = []
    spans_sizes = []
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            # concatenate spans
            txt_parts = []
            x0s=[]; y0s=[]; x1s=[]; y1s=[]
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if t:
                    txt_parts.append(t)
                bb = sp.get("bbox", None)
                if bb:
                    x0s.append(bb[0]); y0s.append(bb[1]); x1s.append(bb[2]); y1s.append(bb[3])
                sz = sp.get("size", None)
                if isinstance(sz, (int, float)) and sz > 0:
                    spans_sizes.append(sz)
            if not txt_parts or not x0s:
                continue
            text = "".join(txt_parts).strip()
            if not text:
                continue
            x0=min(x0s); y0=min(y0s); x1=max(x1s); y1=max(y1s)
            lines.append({"text": text, "bbox": (x0,y0,x1,y1)})

    if not lines:
        return {"page_width": w, "two_column": False}

    # Estimate line height
    heights = [ln["bbox"][3] - ln["bbox"][1] for ln in lines]
    heights = [h for h in heights if 2.0 <= h <= 60.0]
    median_h = sorted(heights)[len(heights)//2] if heights else 12.0

    # Column detection via x0 clustering
    x0s = sorted([ln["bbox"][0] for ln in lines])
    # Find a big gap in x0 distribution as split candidate
    split_x = None
    if len(x0s) >= 20:
        gaps = [(x0s[i+1]-x0s[i], i) for i in range(len(x0s)-1)]
        gaps.sort(reverse=True)
        biggest_gap, idx = gaps[0]
        # require meaningful separation
        if biggest_gap > 0.10*w:
            left_peak = x0s[idx]
            right_peak = x0s[idx+1]
            split_x = (left_peak + right_peak) / 2.0

    two_col = split_x is not None

    return {
        "page_width": w,
        "median_line_height": float(median_h),
        "two_column": two_col,
        "column_split_x": float(split_x) if split_x is not None else None,
    }


# ------------------------------------------------------------------
#  API CALLS
# ------------------------------------------------------------------

def call_detect(client: OpenAI, model: str, bottom_png: bytes,
                page_1_based: int, text_layer: str = "",
                max_retries: int = 3) -> Dict[str, Any]:
    prompt = make_detect_prompt(page_1_based, text_layer)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": "low"},
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text",  "text": prompt},
                        {"type": "input_image", "image_url": b64_data_url(bottom_png)},
                    ],
                }],
                text={
                    "format": {
                        "type":   "json_schema",
                        "name":   "detect_url_footnotes",
                        "strict": True,
                        "schema": DETECT_SCHEMA,
                    }
                },
            )
            data = json.loads(resp.output_text)
            data["page_1_based"] = page_1_based
            data["footnotes"]    = filter_mailto(data.get("footnotes", []))
            data["has_url_footnotes"] = bool(data["footnotes"])
            return data

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"    [!] detect failed after {max_retries} attempts: {e}")
                return {"page_1_based": page_1_based, "has_url_footnotes": False, "footnotes": []}


def call_extract(client: OpenAI, model: str, page_png: bytes,
                 next_top_pngs: List[bytes], page_1_based: int,
                 footnotes: List[Dict[str, Any]],
                 layout_hints: Dict[str, Any],
                 prev_bottom_png: Optional[bytes] = None,
                 max_retries: int = 3) -> Dict[str, Any]:
    """Restoration and paragraph context in one call over the same three images."""
    has_prev  = prev_bottom_png is not None
    user_text = make_extract_user_prompt(page_1_based, footnotes,
                                         layout_hints, has_prev)

    # IMAGE 0 (prev bottom, optional), IMAGE 1 (full page), IMAGE 2+ (next tops)
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    if has_prev:
        content.append({"type": "input_image", "image_url": b64_data_url(prev_bottom_png)})
    content.append({"type": "input_image", "image_url": b64_data_url(page_png)})
    for png in next_top_pngs:
        content.append({"type": "input_image", "image_url": b64_data_url(png)})

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": "medium"},
                input=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user",   "content": content},
                ],
                text={
                    "format": {
                        "type":   "json_schema",
                        "name":   "extract_footnote_urls",
                        "strict": True,
                        "schema": EXTRACT_SCHEMA,
                    }
                },
            )
            data = json.loads(resp.output_text)
            data["base_page_1_based"] = page_1_based
            return data

        except Exception as e:
            if attempt < max_retries:
                time.sleep(3)
            else:
                print(f"    [!] extract failed after {max_retries} attempts: {e}")
                return {"base_page_1_based": page_1_based, "items": []}


# ------------------------------------------------------------------
#  PER-PDF PROCESSING
# ------------------------------------------------------------------

def process_pdf(client: OpenAI, pdf_path: Path, out_dir: Path,
                clean_output_urls: bool = True) -> Dict[str, Any]:

    doc      = fitz.open(pdf_path)
    pdf_name = pdf_path.name
    n_pages  = doc.page_count

    global_seen: set = set()
    all_items: List[Dict[str, Any]] = []
    blank_top = make_blank_png(dpi=TOP_DPI)
    in_references_section = False

    for page0 in tqdm(range(n_pages), desc=pdf_name, leave=False):
        page_1 = page0 + 1

        # References check: Python only, zero API calls. The first references
        # page is still processed (it can carry a footnote); pages after it
        # are skipped.
        if in_references_section:
            continue
        if page_is_references_section(doc, page0):
            in_references_section = True
            print(f"  p{page_1:>4}: references heading detected, "
                  f"subsequent pages will be skipped")

        # ---- DETECT: bottom strip + its text layer ----
        b_clip      = bottom_clip(doc, page0, DETECT_START_FRAC)
        bottom_png  = render_png(doc, page0, dpi=DETECT_DPI, clip=b_clip)
        bottom_text = extract_bottom_text(doc, page0, DETECT_START_FRAC)
        det         = call_detect(client, DETECT_MODEL, bottom_png, page_1,
                                  text_layer=bottom_text)
        footnotes   = det.get("footnotes", [])

        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

        if not footnotes:
            continue  # no URL footnote on this page: no second call

        # ---- EXTRACT: restoration + context, one call ----
        page_png = render_png(doc, page0, dpi=RESTORE_DPI)

        prev_bottom_png: Optional[bytes] = None
        if page0 > 0:
            if PREV_PAGE_FRAC >= 1.0:
                prev_bottom_png = render_png(doc, page0 - 1, dpi=TOP_DPI)
            else:
                pb_clip = bottom_clip(doc, page0 - 1, 1.0 - PREV_PAGE_FRAC)
                prev_bottom_png = render_png(doc, page0 - 1, dpi=TOP_DPI, clip=pb_clip)

        next_top_pngs: List[bytes] = []
        for k in range(1, MAX_NEXT_PAGES + 1):
            if page0 + k >= n_pages:
                next_top_pngs.append(blank_top)
                break
            t_clip = top_clip(doc, page0 + k, TOP_FRAC)
            next_top_pngs.append(render_png(doc, page0 + k, dpi=TOP_DPI, clip=t_clip))

        layout_hints = extract_layout_hints(doc, page0)

        data = call_extract(
            client          = client,
            model           = RESTORE_MODEL,
            page_png        = page_png,
            next_top_pngs   = next_top_pngs,
            page_1_based    = page_1,
            footnotes       = footnotes,
            layout_hints    = layout_hints,
            prev_bottom_png = prev_bottom_png,
        )

        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

        items = data.get("items", [])

        for it in items:
            if it.get("needs_more_pages") or not sentence_complete(it.get("original_sentence", "")):
                print(f"    [!] p{page_1}: sentence may be incomplete; "
                      f"consider raising MAX_NEXT_PAGES or TOP_FRAC")
                break

        # Normalise empty neighbours to null and derive the paragraph flags.
        # The model is not asked for these two flags; they are a function of
        # whether the neighbour is null.
        for it in items:
            prev_s = it.get("preceding_sentence")
            next_s = it.get("trailing_sentence")
            prev_s = prev_s if (isinstance(prev_s, str) and prev_s.strip()) else None
            next_s = next_s if (isinstance(next_s, str) and next_s.strip()) else None
            it["preceding_sentence"] = prev_s
            it["trailing_sentence"]  = next_s
            it["at_paragraph_start"] = prev_s is None
            it["at_paragraph_end"]   = next_s is None

        # Deterministic URL cleaning on the output urls: strip wrapping
        # punctuation, drop mailto links, and deduplicate within each item.
        if clean_output_urls:
            for it in items:
                urls = it.get("url", [])
                seen_u: set = set()
                cleaned: List[str] = []
                for u in urls:
                    if not isinstance(u, str):
                        continue
                    u = clean_url(u)
                    if not u or u.lower().startswith("mailto:"):
                        continue
                    if u not in seen_u:
                        seen_u.add(u)
                        cleaned.append(u)
                it["url"] = cleaned

        # Deduplicate across pages and tag with provenance
        for it in items:
            restored = it.get("restored_sentence", "").strip()
            if not restored:
                continue
            key = norm_key(restored)
            if key in global_seen:
                continue
            global_seen.add(key)
            it["pdf_file"] = pdf_name
            it["page"]     = page_1
            all_items.append(it)

        found        = len([it for it in items if it.get("restored_sentence", "").strip()])
        spanned      = sum(1 for it in items if it.get("sentence_spans_pages"))
        prev_spanned = sum(1 for it in items if it.get("sentence_starts_prev_page"))
        tag_parts = []
        if spanned:      tag_parts.append(f"{spanned} spans-next")
        if prev_spanned: tag_parts.append(f"{prev_spanned} spans-prev")
        tag = f", {', '.join(tag_parts)}" if tag_parts else ""
        print(f"  p{page_1:>4}: {len(footnotes)} footnote(s) detected → "
              f"{found} sentence(s) restored{tag}")

    doc.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    items_path     = out_dir / f"{pdf_path.stem}_footnotes.json"
    sentences_path = out_dir / f"{pdf_path.stem}_sentences.json"

    items_path.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")
    sentences_path.write_text(
        json.dumps([it["restored_sentence"] for it in all_items],
                   ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "pdf_file":       pdf_name,
        "count":          len(all_items),
        "items_json":     str(items_path),
        "sentences_json": str(sentences_path),
    }


# ------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Footnote URL extraction and citing-sentence restoration.")
    p.add_argument("--input",  required=True, type=Path,
                   help="directory of PDFs (searched recursively)")
    p.add_argument("--output", required=True, type=Path,
                   help="directory for per-PDF JSON, the combined JSONL, and the summary")
    p.add_argument("--no-clean-output-urls", action="store_true",
                   help="write the model's raw url field to disk without running clean_url on it")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: set the OPENAI_API_KEY environment variable.")

    in_dir  = args.input.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()

    if not in_dir.is_dir():
        sys.exit(f"ERROR: input folder not found: {in_dir}")

    pdf_files = sorted(in_dir.rglob("*.pdf"))
    if not pdf_files:
        sys.exit(f"ERROR: no PDFs found under: {in_dir}")

    clean_output_urls = not args.no_clean_output_urls

    client = OpenAI(api_key=api_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input       : {in_dir}")
    print(f"Output      : {out_dir}")
    print(f"Detect      : {DETECT_MODEL}")
    print(f"Extract     : {RESTORE_MODEL}")
    print(f"PDFs found  : {len(pdf_files)}")
    print(f"Crop        : bottom {100*(1-DETECT_START_FRAC):.0f}% for detect  |  "
          f"top {100*TOP_FRAC:.0f}% of next {MAX_NEXT_PAGES} page(s) for extract\n")

    summaries = []
    all_jsonl = out_dir / "_ALL.url_footnotes.jsonl"

    with all_jsonl.open("w", encoding="utf-8") as jl:
        for pdf_path in tqdm(pdf_files, desc="PDFs"):
            print(f"\n{pdf_path.name}")
            try:
                summary = process_pdf(client, pdf_path, out_dir, clean_output_urls)
            except Exception as e:
                print(f"  [!] Failed: {e}")
                summary = {"pdf_file": pdf_path.name, "count": 0, "error": str(e)}
            summaries.append(summary)

            items_path = summary.get("items_json")
            if items_path and Path(items_path).exists():
                for it in json.loads(Path(items_path).read_text(encoding="utf-8")):
                    jl.write(json.dumps(it, ensure_ascii=False) + "\n")

    (out_dir / "_SUMMARY.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(s.get("count", 0) for s in summaries)

    print("\n" + "=" * 55)
    print("Done.")
    print(f"   PDFs processed        : {len(pdf_files)}")
    print(f"   Total sentences saved : {total}")
    print(f"   Per-PDF JSONs in      : {out_dir}")
    print(f"   Combined JSONL        : {all_jsonl}")
    print(f"   Summary               : {out_dir / '_SUMMARY.json'}")
    print("=" * 55)


if __name__ == "__main__":
    main()