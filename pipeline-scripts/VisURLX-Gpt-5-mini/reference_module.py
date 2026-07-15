#!/usr/bin/env python3
"""
Reference URL-Only Restorer v2: bibliography citations only
===========================================================

Detects bibliography citations in body text, restores each citing sentence by
inserting the full reference entry inline, and extracts the paragraph-local
context around each sentence. Restore and context run together in one call
(Phase 3), so context binds to each restored sentence directly.

Architecture
------------
Phase 0  (no API calls, per PDF)
    Scan each page's text layer for a reference-section heading. The first
    match is ref_start_page; every page from there on is the reference block.
    If the text layer has no heading, fall back to the Phase 0/1 VLM classifier
    on the last 30% of the paper instead of assuming there is no reference
    section.

Phase 1  (1 to N API calls, once per PDF)
    Render each reference-section page and send the images in one call (batched
    if there are many pages). The model returns each reference entry as JSON:
    index, authors, title, venue, year, urls, doi, full_text. Saved to
    <stem>_references_with_urls.json. Year suffixes ("2002a", "2019b") are
    kept. Author-name ligatures (ﬁ ﬂ ﬀ ﬃ ﬄ) are normalised to ASCII before
    match keys are built. Particle surnames ("van der Berg") and
    collaboration/organisation authors stay as single units.

Phase 2  (1 API call per body page with citations)
    Render the full body page at high DPI and send it to the detect model. The
    prompt covers bracketed numbers with grouped/ranged expansion, author-year
    (narrative and parenthetical), "Author [N]" style, alpha keys ([Smi23],
    [BFGS]), citations inside parentheticals, bracket locators ([N, Chapter 6]),
    superscript citations gated by reference-index lookup, and negative examples
    for equation/step numbers and math intervals. Grouped citations are expanded
    first, then filtered against allowed_markers.

Phase 3  (1 call per body page with citations)
    Restore and context in one call. Restore finds the citing sentence and
    replaces the marker inline with the full reference entry; a citation inside
    a parenthetical resolves to the outer host sentence, and the parenthetical
    stays intact. Context fills preceding_sentence, trailing_sentence,
    at_paragraph_start, and at_paragraph_end for the same sentence, giving the
    gold-aligned sliding-window format (initial / target / last).

Output
    Per PDF:
      <stem>_references_with_urls.json       URL-bearing reference entries
      <stem>_url_reference_citations.json    full restored citation items
      <stem>_url_reference_sentences.json    restored_sentence strings only

    Corpus-wide:
      _ALL.citations.jsonl                   appended per page
      _SUMMARY.json                          written per PDF

Requirements
    pip install pymupdf openai tqdm

Set OPENAI_API_KEY, fill CONFIG, then run:
    python reference_merged.py
"""

import argparse, base64, json, os, re, struct, sys, time, zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fitz
except ImportError:
    sys.exit("ERROR: install pymupdf  →  pip install pymupdf")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("ERROR: install openai  →  pip install openai")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it   # type: ignore


# ══════════════════════════════════════════════════════════════════
#  CONFIG
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
EXTRACT_MODEL   = "gpt-5-mini"          # Phase 1: reference list extraction
DETECT_MODEL    = "gpt-5-mini"          # Phase 2: citation marker detection
RESTORE_MODEL   = "gpt-5-mini"          # Phase 3: sentence restoration

INPUT_DIR = "./pdfs"
OUTPUT_DIR = "./out/reference"

EXTRACT_DPI    = 300     # reference-list pages; need URL fidelity
DETECT_DPI     = 300     # body pages sent to detect
RESTORE_DPI    = 300     # body pages for restoration
TOP_DPI        = 300     # prev/next page strips

TOP_FRAC        = 0.50   # fraction of next page top to include
PREV_PAGE_FRAC  = 0.50  # fraction of prev page bottom to include
MAX_NEXT_PAGES  = 1     # next-page strips sent for sentence completion


MIXED_PAGE_DETECT_BUFFER = 0.02

# Reference pages batched per extraction call.
REF_PAGES_PER_CALL = 6

DELAY_SECONDS   = 0.3

# Phase 0/1 (page-role identification + reference extraction) reuse the
# extraction model, DPI, and batch size.
PHASE01_MODEL          = EXTRACT_MODEL
PHASE01_DPI            = EXTRACT_DPI
PHASE01_PAGES_PER_CALL = REF_PAGES_PER_CALL

# ══════════════════════════════════════════════════════════════════
#  REFERENCE-SECTION HEADING STRINGS  (text-layer scan)
# ══════════════════════════════════════════════════════════════════

_REF_HEADINGS = frozenset({
    # English
    "references", "bibliography", "works cited", "literature cited",
    "literature", "sources", "citations", "notes",
    # Numbered variants that appear as headings
    "reference list", "list of references",
})


# ══════════════════════════════════════════════════════════════════
#  JSON SCHEMAS
# ══════════════════════════════════════════════════════════════════

# Phase 1: bibliography reference-list entry
REFLIST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index":         {"type": ["string", "null"]},   # "[1]", "1.", "Smith03"
                    "authors":       {"type": ["string", "null"]},
                    "title":         {"type": ["string", "null"]},
                    "venue":         {"type": ["string", "null"]},   # journal, conference, or book
                    "year":          {"type": ["string", "null"]},
                    "urls":          {"type": "array", "items": {"type": "string"}},
                    "doi":           {"type": ["string", "null"]},
                    "full_text":     {"type": "string"},             # verbatim entry as printed
                    # citation_keys: surface forms a body-text citation to this
                    # reference could take, given the paper's citation convention.
                    # The model decides these; no rule-based surname parsing.
                    "citation_keys": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "authors", "title", "venue",
                             "year", "urls", "doi", "full_text", "citation_keys"],
            },
        },
    },
    "required": ["references"],
}

# Phase 2: detected citation markers per body page
DETECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_1_based": {"type": "integer"},
        "has_citations": {"type": "boolean"},
        "markers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "marker":       {"type": "string"},  # exactly as in reference list
                    "marker_style": {
                        "type": "string",
                        "enum": [
                            "bracket_number",            # [1] [23]
                            "paren_number",              # (1) (23)
                            "author_year",               # (Smith, 2023), Smith (2023), (Smith et al. 2007)
                            "bracket_author_year",       # [Smith, 2023]
                            "alpha_key",                 # [Smi23] [BFGS]
                            "superscript_number",        # accepted only when the value
                                                         # resolves to an allowed
                                                         # reference-list index
                        ],
                    },
                    "citing_fragment":     {"type": "string"},  
                    "inside_parenthetical":{"type": "boolean"}, 
                    "visible_group":       {"type": ["string", "null"]}, 
                },
                "required": ["marker", "marker_style", "citing_fragment",
                             "inside_parenthetical", "visible_group"],
            },
        },
    },
    "required": ["page_1_based", "has_citations", "markers"],
}

RESTORE_CONTEXT_SCHEMA: Dict[str, Any] = {
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
                    # ── restoration fields ──
                    "original_sentence":        {"type": "string"},
                    "restored_sentence":        {"type": "string"},
                    "citation_marker":          {"type": "string"},
                    "reference_entry":          {"type": "string"},
                    "urls":                     {"type": "array", "items": {"type": "string"}},
                    "sentence_spans_pages":     {"type": "boolean"},
                    "sentence_starts_prev_page":{"type": "boolean"},
                    "needs_more_pages":         {"type": "boolean"},
                    "in_caption":               {"type": "boolean"},
                    "inside_parenthetical":     {"type": "boolean"},
                    # ── context fields ──
                    "preceding_sentence": {"type": ["string", "null"]},
                    "trailing_sentence":  {"type": ["string", "null"]},
                    "at_paragraph_start": {"type": "boolean"},
                    "at_paragraph_end":   {"type": "boolean"},
                },
                "required": [
                    "original_sentence", "restored_sentence",
                    "citation_marker", "reference_entry", "urls",
                    "sentence_spans_pages", "sentence_starts_prev_page",
                    "needs_more_pages", "in_caption", "inside_parenthetical",
                    "preceding_sentence", "trailing_sentence",
                    "at_paragraph_start", "at_paragraph_end",
                ],
            },
        },
    },
    "required": ["base_page_1_based", "items"],
}


# Phase 0: page-role identification
PHASE01_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "page_1_based": {"type": "integer"},
                    "role": {
                        "type": "string",
                        "enum": ["body", "mixed", "references"]
                    },
                    "reference_start_y_ratio": {
                        "type": ["number", "null"],
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "index":         {"type": ["string", "null"]},
                                "authors":       {"type": ["string", "null"]},
                                "title":         {"type": ["string", "null"]},
                                "venue":         {"type": ["string", "null"]},
                                "year":          {"type": ["string", "null"]},
                                "urls":          {"type": "array", "items": {"type": "string"}},
                                "doi":           {"type": ["string", "null"]},
                                "full_text":     {"type": "string"},
                                "citation_keys": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["index", "authors", "title", "venue",
                                         "year", "urls", "doi", "full_text",
                                         "citation_keys"],
                        },
                    },
                },
                "required": ["page_1_based", "role", "reference_start_y_ratio", "references"]
            }
        }
    },
    "required": ["pages"]
}

# ══════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════


# ── Phase 0/1: page-role identification + reference extraction ────────────

PHASE01_SYSTEM = """
You are a precise academic PDF layout analyst and bibliography parser.

Your task: for each provided academic PDF page image, do BOTH of the following:

1) classify the page as one of:
   body         → normal paper body text only
   mixed        → body text appears above, and bibliography / references begin lower on the same page
   references   → the page is part of the bibliography / reference list only

2) extract EVERY bibliography / reference-list entry visible on that page from the reference region only:
   • if role = body: references = []
   • if role = references: extract all visible reference entries on the page
   • if role = mixed: extract only the reference entries that appear BELOW where the bibliography begins

For every page:
- Return page_1_based.
- Return role.
- If role = mixed, return reference_start_y_ratio.
- If role is body or references, set reference_start_y_ratio = null.
- Return references for that page only.

When extracting references, follow these rules exactly:

════════════════════════════════════════
WHAT TO EXTRACT
════════════════════════════════════════
For every bibliography / reference-list entry you see, extract:

  index       The label the entry is cited by in the body text.
              Capture it EXACTLY as printed:
                numbered:     "1", "2", "42"
                bracketed:    "[1]", "[42]"
                named:        "Smith03", "BFGS", "Knu68"
                author-year:  "Smith et al., 2023"
              If no marker is visible, set index = null.

  authors     All author names as printed.
              - Preserve particles / prefixes / suffixes as part of the surname:
                "van der Berg", "de la Cruz", "von Neumann", "O'Brien", "St. George",
                "Jr.", "Sr.", "III".
              - Preserve organisation / collaboration authors verbatim:
                "LIGO Scientific Collaboration", "Virgo Collaboration",
                "The 1000 Genomes Project Consortium".
              - Normalise common ligatures in names to their ASCII equivalents
                inside the authors field:  ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi, ﬄ→ffl.
                (full_text must still be verbatim — only authors/title are
                ligature-normalised.)
              If absent, set null.
  title       The title of the work. Normalise ligatures the same way as for authors.
              If absent, set null.
  venue       Journal, conference, book, report series, thesis institution, or site name.
              If absent, set null.
  year        4-digit year OPTIONALLY followed by a letter suffix.
              Preserve the suffix exactly: "2002", "2002a", "2019b".
              Never drop the suffix — "Smith (2002a)" and "Smith (2002b)" are
              two DIFFERENT references and must be distinguishable by year.
              If absent, set null.

  urls        List of EVERY real printed URL / domain in this entry.
              A URL is any address that identifies a resource accessible over a
              network. It does not need an explicit scheme prefix to qualify. The
              test is: could a person or client program use this string to reach a
              real resource over the internet or a network?
              INCLUDE any of the following when printed in the entry:
                - Addresses with a resource-access scheme: http://, https://,
                  ftp://, ftps://, sftp://, s3://, gs://, az://, or any analogous
                  scheme that identifies a network-accessible resource location.
                - Addresses beginning with "www." with or without an explicit scheme.
                - Bare domain-name addresses with no scheme prefix: a human-readable
                  hostname string (letters, digits, hyphens, dots) that identifies a
                  real network host, with or without a path.
                - DOIs written as a web address: an https://doi.org/ or
                  http://dx.doi.org/ URL.
              NOT A URL — reject these wherever they appear in the entry:
                - Bare scholarly identifiers. A DOI is bare unless printed as a
                  doi.org or dx.doi.org web address: reject 10.NNNN/suffix and
                  reject doi:10.NNNN/suffix. Reject an arXiv identifier such as
                  arXiv:NNNN.NNNNN and a legacy arXiv identifier such as
                  astro-ph/NNNNNNN. These are identifiers, not addresses. Do not
                  turn one into a URL by adding a prefix yourself.
                - Standalone hostnames, server names, bare host:port strings, or
                  bare IPv4/IPv6 addresses used as machine identifiers rather than
                  web addresses.
                - Non-resource scheme strings whose purpose is messaging or
                  execution rather than addressing a resource: mailto:, tel:, sms:,
                  javascript:, data:, file:, and similar.
                - Namespace IRIs, XML namespace URIs, schema identifiers, or
                  RDF/OWL/SPARQL prefix declarations that resemble URLs but serve
                  as formal identifiers.
                - Example-style or placeholder URLs, unless visibly printed in the entry.
              If the entry contains no real printed URL/domain, set urls = [].

  doi         The DOI string alone if one is present anywhere in the entry. Set null if absent.

  full_text   The COMPLETE verbatim text of the entry exactly as printed,
              including the index label. Never shorten or paraphrase.
              Merge ordinary line wraps.
              If a URL breaks across lines, join it back into one URL even when the break happens
              after the scheme, e.g. "http:" on one line and "//domain/..." on the next line.
              Keep original ligatures and accented characters as printed.

  citation_keys
              The list of surface forms a body-text citation to THIS reference
              could plausibly take, given this paper's citation convention.
              This is how the downstream detector will match body-text
              citations back to this reference entry.

              Infer the convention from the reference list itself:
              - Entries labelled "[1]", "[2]", …  -> numeric convention
              - Entries with no label that start with "Author (Year)" or
                "Author, Year" -> author-year convention
              - Entries using compressed keys like "[Smi23]" or "[BFGS]"
                -> alpha-key convention

              PRODUCE ALL PLAUSIBLE FORMS. Examples:

              Numeric convention, entry "[5] Smith, J. ...":
                ["5", "[5]"]

              Author-year, entry "Desper, R. and Gascuel, O. (2002a). ...":
                ["Desper and Gascuel, 2002a",
                 "Desper and Gascuel (2002a)",
                 "Desper & Gascuel, 2002a",
                 "Desper & Gascuel (2002a)",
                 "(Desper and Gascuel, 2002a)",
                 "(Desper & Gascuel, 2002a)"]

              Author-year, single author "van der Berg, J. (2019). ...":
                ["van der Berg, 2019",
                 "van der Berg (2019)",
                 "(van der Berg, 2019)"]

              Author-year, three-or-more authors
              "Harry, G. M., the LIGO Scientific Collaboration & Virgo
               Collaboration (2010). ...":
                ["Harry et al., 2010",
                 "Harry et al. (2010)",
                 "Harry et al. 2010",
                 "LIGO Scientific Collaboration, 2010",
                 "LIGO Scientific Collaboration (2010)",
                 "(Harry et al. 2010)",
                 "(LIGO Scientific Collaboration 2010)"]

              Organisation-only author "Virgo Collaboration (2009). ...":
                ["Virgo Collaboration, 2009",
                 "Virgo Collaboration (2009)",
                 "(Virgo Collaboration 2009)"]

              Alpha-key convention, entry "[BFGS] Broyden, C. et al. ...":
                ["BFGS", "[BFGS]"]

              RULES FOR citation_keys:
              - Preserve the year suffix in every key ("2002a", not "2002").
              - Use the same diacritics/spelling as printed in authors.
              - Include both bracketed/parenthesised and bare forms.
              - For mixed human+organisation author lists, include BOTH the
                first-human-surname "et al." form AND the organisation form.
              - For numeric-convention papers, produce ONLY the bare number
                and the bracketed form — do NOT produce author-year keys
                (body text will never use them).
              - If the entry has no identifiable author AND no numeric
                index, return an empty array.

════════════════════════════════════════
ACCURACY RULES
════════════════════════════════════════
• NEVER fabricate, guess, or hallucinate any page role or field.
• NEVER truncate a URL with "…" — transcribe it completely.
• Preserve all characters in URLs exactly as printed.
• If a URL wraps across lines, join the wrapped pieces into one URL.
• Do NOT include mailto: addresses in urls[].
• Extract ONLY entries from the actual reference region.
• Do NOT extract body-text URLs, footnotes, or inline body references.
• Preserve year suffixes like "2002a", "2019b" in the year field.
• Preserve particle surnames ("van der Berg", "de la Cruz") intact.

Return strict JSON matching the provided schema.
""".strip()


def make_phase01_user_prompt(page_numbers: List[int]) -> str:
    pages_str = ", ".join(str(p) for p in page_numbers)
    return (
        f"These images are pages {pages_str} of one academic PDF. "
        f"For each page, classify it as body, mixed, or references. "
        f"If mixed, estimate where references begin using reference_start_y_ratio. "
        f"Also extract every bibliography/reference entry visible in the reference region of that page. "
        f"Return strict JSON."
    )


# ── Phase 1: reference list extraction ───────────────────────────

REFLIST_EXTRACT_SYSTEM = """
You are a precise academic document parser.
Your task: extract reference entries from the provided images of a
reference section / bibliography of an academic paper.

════════════════════════════════════════
WHICH ENTRIES TO EXTRACT  (READ FIRST)
════════════════════════════════════════
Extract ONLY reference entries that contain at least one REAL printed URL or
domain (as defined under `urls` below). SKIP every reference entry that has no
real printed URL — do not emit it at all.

  • A reference that prints at least one real URL or domain (per the `urls`
    definition below) → EXTRACT it.
  • A reference with only authors/title/venue/year and NO printed URL → SKIP it.
  • A reference whose only web-like token is a bare arXiv ID (arXiv:NNNN.NNNNN),
    a bare astro-ph/… identifier, or a bare doi:10.… string with no printed URL
    → treat as NO real URL → SKIP it.

If a page has 40 references but only 5 print a real URL, return exactly those 5.
Read every entry to make this decision, but emit only the URL-bearing ones.

════════════════════════════════════════
WHAT TO EXTRACT  (for each URL-bearing entry)
════════════════════════════════════════
For every entry you keep — regardless of formatting style — extract:

  index       The label the entry is cited by in the body text.
              Capture it EXACTLY as printed:
                numbered:     "1", "2", "42"
                bracketed:    "[1]", "[42]"
                named:        "Smith03", "BFGS", "Knu68"
                author-year:  "Smith et al., 2023"
              If no marker is visible, set index = null.

  authors     All author names as printed.
              - Preserve particles / prefixes / suffixes as part of the surname:
                "van der Berg", "de la Cruz", "von Neumann", "O'Brien", "St. George",
                "Jr.", "Sr.", "III".
              - Preserve organisation / collaboration authors verbatim:
                "LIGO Scientific Collaboration", "Virgo Collaboration",
                "The 1000 Genomes Project Consortium".
              - Normalise common ligatures inside the authors field:
                ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi, ﬄ→ffl.
                (full_text must still be verbatim — only authors/title are
                ligature-normalised.)
              If absent, set null.
  title       The title of the work. Normalise ligatures the same way as for authors.
              If absent, set null.
  venue       Journal, conference, book, report series, thesis institution, or site name.
              If absent, set null.
  year        4-digit year OPTIONALLY followed by a letter suffix.
              Preserve the suffix exactly: "2002", "2002a", "2019b".
              Never drop the suffix — "Smith (2002a)" and "Smith (2002b)" are
              two DIFFERENT references and must be distinguishable by year.
              If absent, set null.

  urls        List of EVERY real printed URL / domain in this entry.
              A URL is any address that identifies a resource accessible over a
              network. It does not need an explicit scheme prefix to qualify. The
              test is: could a person or client program use this string to reach a
              real resource over the internet or a network?
              INCLUDE any of the following when printed in the entry:
                - Addresses with a resource-access scheme: http://, https://,
                  ftp://, ftps://, sftp://, s3://, gs://, az://, or any analogous
                  scheme that identifies a network-accessible resource location.
                - Addresses beginning with "www." with or without an explicit scheme.
                - Bare domain-name addresses with no scheme prefix: a human-readable
                  hostname string (letters, digits, hyphens, dots) that identifies a
                  real network host, with or without a path.
                - DOIs written as a web address: an https://doi.org/ or
                  http://dx.doi.org/ URL.
              NOT A URL — reject these wherever they appear in the entry:
                - Bare scholarly identifiers. A DOI is bare unless printed as a
                  doi.org or dx.doi.org web address: reject 10.NNNN/suffix and
                  reject doi:10.NNNN/suffix. Reject an arXiv identifier such as
                  arXiv:NNNN.NNNNN and a legacy arXiv identifier such as
                  astro-ph/NNNNNNN. These are identifiers, not addresses. Do not
                  turn one into a URL by adding a prefix yourself.
                - Standalone hostnames, server names, bare host:port strings, or
                  bare IPv4/IPv6 addresses used as machine identifiers rather than
                  web addresses.
                - Non-resource scheme strings whose purpose is messaging or
                  execution rather than addressing a resource: mailto:, tel:, sms:,
                  javascript:, data:, file:, and similar.
                - Namespace IRIs, XML namespace URIs, schema identifiers, or
                  RDF/OWL/SPARQL prefix declarations that resemble URLs but serve
                  as formal identifiers.
                - Example-style or placeholder URLs, unless visibly printed in the entry.

  doi         The DOI string alone if one is present anywhere in the entry. Set null if absent.

  full_text   The COMPLETE verbatim text of the entry exactly as printed,
              including the index label. Never shorten or paraphrase.
              Merge ordinary line wraps.
              If a URL breaks across lines, join the wrapped pieces into one URL even when the break happens
              after the scheme, e.g. "http:" on one line and "//domain/..." on the next line.
              Keep original ligatures and accented characters as printed.

  citation_keys
              The list of surface forms a body-text citation to THIS reference
              could plausibly take, given this paper's citation convention.
              This is how the downstream detector will match body-text
              citations back to this reference entry.

              Infer the convention from the reference list itself:
              - Entries labelled "[1]", "[2]", …  -> numeric convention
              - Entries with no label that start with "Author (Year)" or
                "Author, Year" -> author-year convention
              - Entries using compressed keys like "[Smi23]" or "[BFGS]"
                -> alpha-key convention

              PRODUCE ALL PLAUSIBLE FORMS. Examples:

              Numeric, "[5] Smith, J. ...":
                ["5", "[5]"]

              Author-year, "Desper, R. and Gascuel, O. (2002a). ...":
                ["Desper and Gascuel, 2002a",
                 "Desper and Gascuel (2002a)",
                 "Desper & Gascuel, 2002a",
                 "Desper & Gascuel (2002a)",
                 "(Desper and Gascuel, 2002a)",
                 "(Desper & Gascuel, 2002a)"]

              Single particle-surname author, "van der Berg, J. (2019). ...":
                ["van der Berg, 2019",
                 "van der Berg (2019)",
                 "(van der Berg, 2019)"]

              Mixed human + organisation authors, "Harry, G. M., the LIGO
              Scientific Collaboration & Virgo Collaboration (2010). ...":
                ["Harry et al., 2010",
                 "Harry et al. (2010)",
                 "Harry et al. 2010",
                 "LIGO Scientific Collaboration, 2010",
                 "LIGO Scientific Collaboration (2010)",
                 "(Harry et al. 2010)",
                 "(LIGO Scientific Collaboration 2010)"]

              Organisation-only, "Virgo Collaboration (2009). ...":
                ["Virgo Collaboration, 2009",
                 "Virgo Collaboration (2009)",
                 "(Virgo Collaboration 2009)"]

              Alpha-key, "[BFGS] Broyden, C. et al. ...":
                ["BFGS", "[BFGS]"]

              RULES FOR citation_keys:
              - Preserve the year suffix in every key ("2002a", not "2002").
              - Use the same diacritics/spelling as printed in authors.
              - Include both bracketed/parenthesised and bare forms.
              - For mixed human+organisation author lists, include BOTH the
                first-human-surname "et al." form AND the organisation form.
              - For numeric-convention papers, produce ONLY the bare number
                and the bracketed form — do NOT produce author-year keys.
              - If the entry has no identifiable author AND no numeric
                index, return an empty array.

              IMPORTANT: infer the convention from ALL entries you can see on
              the page (including ones you are skipping for having no URL), so
              the keys you produce for the URL-bearing entries use the paper's
              actual convention.

════════════════════════════════════════
ACCURACY RULES — NON-NEGOTIABLE
════════════════════════════════════════
• NEVER fabricate, guess, or hallucinate any field.
• NEVER truncate a URL with "…" — transcribe it completely.
• Preserve all characters in URLs exactly as printed.
• If a URL wraps across lines, join the wrapped pieces into one URL.
• Do NOT include mailto: addresses in urls[].
• Process every entry visible in the images when deciding which have URLs, even if the list is long.
• Do NOT convert bare arXiv IDs, astro-ph identifiers, or bare doi: identifiers into URLs.
• Preserve year suffixes like "2002a", "2019b" in the year field.
• Preserve particle surnames ("van der Berg", "de la Cruz") intact.

Return strict JSON matching the provided schema.
""".strip()


def make_reflist_user_prompt(page_numbers: List[int]) -> str:
    pages_str = ", ".join(str(p) for p in page_numbers)
    return (
        f"These images show the reference section of an academic paper "
        f"(pages {pages_str}). "
        f"Extract every reference entry as described. Return strict JSON."
    )


# ── Phase 2: citation marker detection ───────────────────────────

DETECT_SYSTEM = """
You are a scholarly document analyst specialising in bibliography citation detection.

Your task: identify EVERY visible citation occurrence in the BODY TEXT of the provided page image
that cites one of the paper's extracted reference-list entries.

SCOPE
- Detect ONLY body-text citation markers that refer to the bibliography / reference list.
- Do NOT detect body URLs, inline self-contained references, or bare URLs.
- Do NOT detect anything in the reference list itself, headers/footers, page numbers, or running titles.
- DO detect citations that appear inside figure captions, table captions, or algorithm captions —
  these ARE in scope. Mark them the same as body citations; downstream restoration will flag them.
- DO detect citations inside parentheticals such as "(see [N])", "(cfr. [N])", "(MML [N])" —
  set inside_parenthetical = true for these.
- Return EVERY occurrence separately, no matter how many times the same marker appears.
- If the same reference is cited in five different sentences on this page, return five occurrences.
- If the same marker appears ten times on this page, return ten occurrences.
- Never cap or truncate repeated occurrences — completeness matters more than brevity here.

════════════════════════════════════════
CITATION PATTERNS TO DETECT
════════════════════════════════════════
Detect ALL of the following wherever they appear in body text or captions:

1. BRACKETED NUMBERS — the dominant scholarly citation style
   Examples of single markers:
     [1]   [23]   [42]
   Examples of grouped markers:
     [6,7]         → members: 6, 7
     [6, 7]        → members: 6, 7
     [21–23]       → members: 21, 22, 23   (range with en-dash)
     [21-23]       → members: 21, 22, 23   (range with hyphen)
     [21—23]       → members: 21, 22, 23   (range with em-dash)
     [21−23]       → members: 21, 22, 23   (range with Unicode minus U+2212)
     [7, 8, 12–15] → members: 7, 8, 12, 13, 14, 15  (mixed list + range)
     [16, 9, 24]   → members: 9, 16, 24    (out-of-order list — still a list, not a range)
     [1, 15,4]     → members: 1, 4, 15     (irregular spacing — still a list)
     [100 ,100]    → members: 100          (OCR artifact: dedupe repeated members)
     [10 – 12]     → members: 10, 11, 12   (extra spaces around dash — still a range)
     [15, Chapter 6]   → members: 15       (Chapter 6 is a LOCATOR, not a second citation)
     [15, §3.2]        → members: 15       (section locator)
     [15, Thm. 5.2]    → members: 15       (theorem locator)
     [15, p. 42]       → members: 15       (page locator)
   Rules:
   - Expand comma-separated grouped citations into separate markers.
   - Expand ranges into separate markers. Inclusive on both ends.
   - Dedupe numerically identical members inside the same group.
   - Treat any of these as a range dash: - (hyphen), – (en-dash), — (em-dash), − (Unicode minus).
   - When a bracket contains a number followed by a locator phrase like
     "Chapter N", "Ch. N", "§N", "Sec. N", "Section N", "p. N", "pp. N-M",
     "Thm. N", "Theorem N", "Def. N", "Lem. N", "Prop. N", "Cor. N",
     "Eq. N", "Fig. N", "Table N", "Alg. N" — the citation marker is only
     the leading number. The locator stays in citing_fragment.
   - Adjacent bracket citations are separate citations: [26] [27] → "26" and "27".
     Also [26][27] (no space) → "26" and "27".
   - Set visible_group to the literal visible bracket group when the marker
     is part of a multi-member group (e.g., visible_group = "[21–23]" for
     each of 21, 22, 23). For single-member brackets set visible_group = null.
   Style: bracket_number

2. "AUTHOR [N]"  — EXTREMELY COMMON — do not miss this pattern
   Examples:
     Kepler Conjecture [8]
     Kreft and Navarro [12, 13]
     Smith et al. [5]
     the Veque family [14, 15]
     OCamlJit2 [15]
   Rules:
   - The marker is the bracketed number(s); the author name stays in citing_fragment.
   - Apply the same grouped / range / locator expansion rules as pattern 1.
   - This is a NUMERIC citation; the author is just prose. Do NOT classify
     this as author_year even though an author name is present.
   Style: bracket_number

3. PARENTHESISED NUMBERS  (RARE — BE VERY CONSERVATIVE)
   Examples:
     (1) (23) (6,7) (21–23)
   Rules:
   - Expand grouped lists and ranges the same way as bracketed numbers.
   - REPORT a parenthesised number ONLY if you have high confidence it is a
     bibliography citation. The following uses are NOT citations and MUST be
     ignored:
       • Equation numbers introducing or referring to a displayed formula:
         "(1) λ(t, D) := …", "as shown in (3)", "substituting (4) into (5)"
       • List enumerators: "(1) first, (2) second, (3) third"
       • Step numbers in algorithms or procedures
       • Case numbers: "In case (i)", "(case 2)"
       • Subsection markers
       • Footnote markers when they are parenthesised
     When in doubt, do NOT report a paren_number.
   Style: paren_number

4. AUTHOR-YEAR CITATIONS (narrative and parenthetical)
   Narrative (subject-of-sentence) forms:
     Smith (2023)
     Smith et al. (2019)
     Smith and Jones (2021)
     Smith & Jones (2021)
   Parenthetical forms:
     (Smith, 2023)
     (Smith et al. 2007)           ← no comma before year
     (Smith & Jones, 2021)
     (Smith and Jones, 2021)
     (Virgo Collaboration 2009)    ← organisation author
     (the LIGO Scientific Collaboration 2010)
     (van der Berg, 2019)          ← particle surname
     (Desper and Gascuel, 2002a)   ← year suffix MUST be preserved
     (Desper and Gascuel, 2002b)   ← a different reference from 2002a
   Grouped parenthetical forms — separator is either comma or semicolon:
     (Smith, 2023; Jones, 2022)
     (Skilling 2004, Skilling 2006, Feroz et al. 2009)
     (Zwickl and Hillis, 2002; Pollock et al., 2002; Hillis et al., 2003)
     (Harry et al. 2010, Virgo Collaboration 2009)
   Rules:
   - Return one marker per cited reference in a group.
   - The parenthesised author-year may wrap across printed lines inside the
     parentheses. Treat the whole ( ... ) group as one citation regardless of
     line breaks inside it:
        (Robinson and Foulds,
         1981)                     ← one citation, marker = "Robinson and Foulds, 1981"
        (Skilling 2004, Skilling
         2006)                     ← two citations, "Skilling 2004" and "Skilling 2006"
   - Preserve year suffixes (2002a, 2019b) in the marker exactly as printed.
   - Treat "et al.", "et al", "et. al.", "et al.,", "et al .", "ef al." (common
     OCR error for italic "et al.") as equivalent forms of the same marker.
   - For organisation / collaboration authors, the whole organisation name is
     the "surname" (e.g., "Virgo Collaboration 2009" — marker surname is
     "Virgo Collaboration", NOT "Collaboration").
   - Normalise ligatures ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi, ﬄ→ffl when producing the
     marker string so it can match the author-year keys derived from the
     extracted references. The citing_fragment must stay verbatim.
   Style: author_year

5. BRACKETED AUTHOR-YEAR
   Examples:
     [Smith, 2023]
     [Jones & Lee, 2021]
     [van der Berg, 2019]
   Apply all rules from pattern 4.
   Style: bracket_author_year

6. ALPHA-NUMERIC KEYS
   Examples:
     [Smi23]   [BFGS]   [HTF09]   [GBC16]   [Knu68]
   Rules:
   - Accept ONLY if the bracketed token matches one of the allowed reference-list
     indices. Do NOT invent or generalise alpha-key patterns from the visible
     page alone.
   Style: alpha_key

7. SUPERSCRIPT NUMBERS  (PMC / medical / some IEEE styles)
   Examples (visible as small-font raised digits):
     …observed in recent studies¹.
     …previous work²,³,⁵ has shown…
     …confirmed⁴⁻⁶ that…            (range: 4, 5, 6)
   Rules:
   - Accept a superscript digit as a citation ONLY when its value EXACTLY
     matches one of the allowed reference-list indices AND the allowed-list
     itself is numeric (i.e., the paper uses numeric bibliography indices).
     If the paper's bibliography is author-year or alpha-key based, EVERY
     superscript digit is a footnote, not a citation — do NOT report it.
   - Superscripts can be grouped: "²,³" "²⁻⁴" "²⁻⁴,⁷".
     Expand the same way as bracketed numbers.
   - If a footnote zone at the bottom of the page uses the same numbering
     scheme, DO NOT report superscripts — the ambiguity is too high.
   Style: superscript_number

════════════════════════════════════════
GROUPED-CITATION EXPANSION INTERACTS WITH THE ALLOWED-MARKER LIST
════════════════════════════════════════
When an allowed-marker list is provided in the user prompt, the filtering
order is:
  1) Detect the full visible group (e.g., "[21–23]").
  2) Expand the group into its member markers (21, 22, 23).
  3) Filter the expanded members by the allowed list; drop members that are
     not allowed.
  4) For each surviving member, emit ONE marker item whose:
       • marker        = the expanded member (e.g., "22")
       • visible_group = the literal visible group ("[21–23]")
       • citing_fragment = a short window that contains the visible group
  5) If NO member survives, do not emit anything for this group.
Never drop a whole group just because the literal group string is not in the
allowed list — always expand first, filter second.

════════════════════════════════════════
CITING_FRAGMENT
════════════════════════════════════════
  citing_fragment
    A short exact excerpt (roughly 8–20 printed words) around THIS occurrence,
    sufficient to locate the host sentence in later phases.
    - For grouped citations, include the full visible grouped surface text
      (e.g., the fragment should visibly contain [21–23], [26] [27], or [6,7]).
    - For bracket locators like [15, Chapter 6], include the locator text.
    - For parenthetical citations like "(see [N])", include the parenthesis
      and the cue word ("see", "cf.", "e.g.", "cfr.", "specifically",
      "called X in").
    - Keep common abbreviations intact as tokens: "e.g.", "i.e.", "cf.",
      "et al.", "Fig.", "Eq.", "Ref.", "Sec." — do not split them.

════════════════════════════════════════
WHAT NOT TO REPORT
════════════════════════════════════════
• Bare URLs or body-text URLs (even if one matches a URL in a reference entry).
• Inline arXiv / astro-ph / bare DOI identifiers in body text
  ("arXiv:1001.5241v4", "astro-ph/0301001", "doi:10.1103/...").
• The paper's own arXiv identifier stamped in the header/footer (watermark).
• Document-element references that are NOT bibliography citations:
    "Table 3", "Fig. 2", "Figure 4a", "Section 4.1", "Chapter 6",
    "Eq. (5)", "Equation 3", "Algorithm 2", "Listing 1".
• Math intervals and coordinates that use bracket syntax but are not
  citations:
    "α ∈ [0, 1]"  — interval, not a citation
    "[-1, 1]"     — interval
    "[0, 2] × [0, 0.5]"  — coordinate domain
    "[100, 100]" appearing inside source code or a vector literal
    "[i, j]"      — matrix index
    "[n]" inside code comments / snippets
  When the surrounding context is mathematical (variables, equation-like
  expressions, code), treat bracketed numbers as math, not citations.
• Reference-list entries themselves, headers, footers, page numbers, running
  titles, and the paper's own title.

If the page has NO bibliography citation markers in body text, set
has_citations = false and markers = [].

Return strict JSON matching the provided schema.
""".strip()


def make_detect_user_prompt(page_1_based: int, allowed_markers: Optional[List[str]] = None,
                             index_style: str = "mixed") -> str:
    """Build the DETECT user prompt.

    index_style indicates whether the paper's extracted references use
    numeric indices ('numeric'), author-year keys ('author_year'),
    alpha keys ('alpha'), or a mixture ('mixed'). This lets us tell the
    model whether superscripts could even possibly be citations.
    """
    style_note = ""
    if index_style == "numeric":
        style_note = (
            "The paper's bibliography uses NUMERIC indices. Superscript "
            "numbers MAY be citations; report them only when they match an "
            "allowed marker. "
        )
    elif index_style in ("author_year", "alpha"):
        style_note = (
            "The paper's bibliography does NOT use numeric indices — it uses "
            f"{index_style.replace('_', '-')} keys. Therefore every superscript "
            "digit on the page is a footnote marker, not a citation. Do NOT "
            "report superscripts. "
        )
    else:
        style_note = (
            "The paper's bibliography uses a mixture of indexing styles. "
            "Report superscripts only when they clearly match a numeric entry "
            "in the allowed marker list. "
        )

    if allowed_markers:
        allowed_json = json.dumps(sorted(set(allowed_markers)), ensure_ascii=False)
        return (
            f"PAGE {page_1_based}. Identify every BODY-TEXT bibliography citation occurrence "
            f"that cites one of these extracted reference-list markers: {allowed_json}. "
            + style_note +
            "Detect only bibliography citation markers. Return every repeated occurrence separately. "
            "Expand grouped citations (e.g. [21–23], [26] [27], [6,7], [16, 9, 24]) into separate "
            "markers FIRST, then filter each expanded member against the allowed list. "
            "Do NOT drop a group just because its literal surface string is not in the allowed list. "
            "Return strict JSON."
        )
    return (
        f"PAGE {page_1_based}. Identify every BODY-TEXT bibliography citation marker occurrence. "
        + style_note +
        "Return every repeated occurrence separately and expand grouped citations such as [21–23], "
        "[26] [27], and [6,7] into separate markers. Return strict JSON."
    )

# ── Phase 3a: restore ────────────────────────────────────────────

RESTORE_SYSTEM = """
You are a meticulous document analyst specialising in academic PDFs and bibliography citation restoration.

════════════════════════════════════════
IMAGES YOU WILL RECEIVE
════════════════════════════════════════
  IMAGE 0  →  BOTTOM STRIP of the PREVIOUS page — use ONLY to recover the
               opening of a citing sentence that started there.
  IMAGE 1  →  FULL PRIMARY PAGE — the page you are analysing.
  IMAGE 2+ →  TOP STRIPS of FOLLOWING page(s) — use ONLY to complete sentences
               that overflow from IMAGE 1.

════════════════════════════════════════
INPUT
════════════════════════════════════════
You receive a JSON list called CITATIONS. Each entry already corresponds to
an extracted reference-list entry that contains a real printed URL/domain.
Each entry has:
  marker              — the resolved citation label (e.g. "21", "27", "Smith et al., 2019")
  marker_style        — how the citation appears in text
  reference_entry     — the full bibliography entry text to insert inline
  urls                — list of real printed URLs/domains from the reference entry
  citing_fragment     — a short excerpt near the citation to help locate the sentence
  visible_group       — when marker came from a multi-member group, the literal
                        bracket group as printed (e.g. "[21–23]"); otherwise null
  inside_parenthetical— true when the citation sits inside a ( ... ) construction

Use citing_fragment and visible_group ONLY as locating hints. Always transcribe from the image.

════════════════════════════════════════
TASK PER CITATION
════════════════════════════════════════
A) LOCATE the host sentence in the BODY TEXT of IMAGE 1 (captions also count as body).

Recognise these citation styles:
• bracket_number       — [N], [N,M], [N–M], adjacent [N] [M], "Author [N]", [N, Chapter 6]
• paren_number         — (N), (N,M), (N–M) — only when clearly a citation
• author_year          — Smith (2023), (Smith et al. 2007), (Virgo Collaboration 2009),
                         (Desper and Gascuel, 2002a), (van der Berg, 2019)
• bracket_author_year  — [Smith, 2023]
• alpha_key            — [Smi23]
• superscript_number   — ¹ ² ³ etc.  (gated by the allowed-marker list)

IMPORTANT FOR GROUPED MARKERS:
- A single visible citation group may yield multiple CITATIONS entries.
- Example: if visible_group = "[21–23]" and refs 21, 22, 23 all have URLs,
  you must produce three output items: one for 21, one for 22, one for 23.
  All three items share the SAME original_sentence. Each item inserts the
  corresponding reference_entry at the position of the visible group.
- Example: if visible_group = "[26] [27]", produce two output items.
- Example: if visible_group = "[6,7]", produce one output item per URL-bearing member.

IMPORTANT FOR BRACKET LOCATORS:
- "[15, Chapter 6]", "[15, §3]", "[15, Thm. 5.2]", "[15, p. 42]" — the
  citation marker is "15"; the locator ("Chapter 6") is NOT a second citation.
  When building restored_sentence, replace the ENTIRE visible bracket
  ("[15, Chapter 6]") with the reference entry — the locator text is
  consumed by the replacement.

IMPORTANT FOR PARENTHETICAL CITATIONS:
When inside_parenthetical = true (e.g., "(see [N])", "(cfr. [N])",
"(For a demonstration, see [N].)", "(MML [N])"):
  - The HOST SENTENCE is the OUTER prose sentence that the parenthetical
    attaches to, NOT the parenthetical itself. The parenthetical stays intact
    INSIDE the host sentence.
  - Example:
       Printed:  "Our approach works well (see also [15]). We evaluated…"
       Host sentence:  "Our approach works well (see also [15])."
       original_sentence:  "Our approach works well (see also [15])."
       restored_sentence:  "Our approach works well (see also [15 <ref entry here>])."
  - When the parenthetical is itself a standalone-looking clause between two
    sentences ("(MML [14])"), the host sentence is the preceding prose
    sentence to which the parenthetical belongs (indicated by the lack of an
    uppercase word starting a new sentence, and the proximity on the page).
    If the parenthetical truly stands alone, treat the parenthetical itself
    as the host "sentence".

If you cannot unambiguously locate the host sentence in IMAGE 1 body text,
produce NO output item for that citation.

B) CAPTURE THE COMPLETE HOST SENTENCE.

Always find the full sentence: from its opening capital letter to its closing
punctuation (. ? !), reading across page images if needed.

• Sentence fully on IMAGE 1 → transcribe fully.
  sentence_spans_pages = false, sentence_starts_prev_page = false.

• Sentence ends on IMAGE 1 but started on previous page → use IMAGE 0.
  sentence_spans_pages = true, sentence_starts_prev_page = true.

• Sentence starts on IMAGE 1 but continues past page bottom → use IMAGE 2+.
  sentence_spans_pages = true, sentence_starts_prev_page = false.
  If IMAGE 2+ still is not enough: needs_more_pages = true.

Short host sentences are OK and expected:
  "See [25]."                   ← three-word host sentence; transcribe as-is
  "(See [10].)"                 ← four-word parenthetical-as-sentence
  "Sources: [26] [27]."         ← legitimate
Do NOT try to "complete" or "fix up" short sentences by pulling in text from
neighbouring sentences — one sentence is one sentence.

Multiple citations per sentence:
  "As shown in [23] and later extended by [27], the algorithm converges."
  If CITATIONS contains both 23 and 27 (both URL-bearing), produce TWO output
  items, each with the same original_sentence, each inserting its own
  reference_entry inline at its own position. Produce TWO restored_sentence
  values (one with 23 replaced, one with 27 replaced), because each output
  item carries ONE replacement.

MULTI-COLUMN: Never wrap text from one column into another.
ABBREVIATION GUARD: Do not treat periods inside common abbreviations as
sentence endings. Treat these as non-terminal:
  e.g., i.e., cf., cfr., et al., et al, et. al.,
  Fig., Figs., Eq., Eqs., Ref., Refs., Sec., Sect., Ch., App., Vol., No.,
  pp., p., ed., eds., etc., vs., approx., resp.,
  Thm., Def., Prop., Lem., Cor., Alg.,
  Mr., Mrs., Ms., Dr., Prof., Jr., Sr.,
  St., Mt., U.S., U.K., U.S.A., Ph.D., M.D., Inc., Ltd., Co.,
  initials in author names like "J. K. Rowling", "A. N. Other".

C) BUILD restored_sentence.

Replace the citation occurrence INLINE with:
  "[" + marker_display + " " + reference_entry + "]"

Rules:
• The restored reference goes EXACTLY where the citation was — never appended at the end.
• Use reference_entry EXACTLY as provided.
• Use only the provided urls field. Never invent a URL.
• Transcribe original_sentence VERBATIM as printed. Include any OTHER citation
  markers that share the same sentence but are not being restored in this output
  item — leave them untouched in both original_sentence and restored_sentence.
• For bracket-locator citations like "[15, Chapter 6]", the ENTIRE bracket group
  including the locator is replaced by "[marker_display reference_entry]".
• For grouped visible citations ([21–23] or [26] [27]), the restored_sentence
  for the item with marker=22 replaces the ENTIRE visible group with the ref
  entry for 22. The sibling items (21, 23) each independently replace the
  ENTIRE visible group with their respective ref entries.
• One sentence citing multiple non-grouped URL-bearing references → ONE output
  item per citation entry.
• Same marker appearing in multiple different sentences → ONE output item per
  CITATIONS entry. If CITATIONS contains marker "5" three times (each with a
  different citing_fragment pointing to a different sentence), produce three
  output items, each with its own original_sentence and restored_sentence.
  Use citing_fragment to locate each sentence independently on the page.
  Do NOT merge them, do NOT deduplicate them, do NOT skip later ones.
• Never deduplicate repeated occurrences across different sentences.

D) ONE SENTENCE = ONE REPLACEMENT PER ITEM.
Never merge two sentences. Never perform two replacements in one output item.

E) FLAGS.
  in_caption = true when the host sentence is the caption of a figure, table,
  algorithm, or listing (the sentence starts with "Figure N", "Fig. N",
  "Table N", "Algorithm N", "Listing N", or is clearly styled as a caption).
  Captions can be legitimate citation hosts — DO restore them.

  inside_parenthetical = true when the citation marker itself sits inside a
  ( ... ) within the host sentence (as described in the parenthetical rule).

════════════════════════════════════════
ACCURACY
════════════════════════════════════════
• NEVER hallucinate sentence text, markers, or reference content.
• NEVER truncate original_sentence.
• NEVER invent a URL.
• Transcribe original_sentence VERBATIM as printed, including ligatures,
  accented characters, other citation markers, and inline math.
• Never silently "correct" an OCR-like rendering — if the printed page says
  "ﬁnally", original_sentence contains "ﬁnally".

Return strict JSON matching the provided schema.
""".strip()


def make_restore_user_prompt(page_1_based: int,
                              citations: List[Dict[str, Any]],
                              has_prev: bool) -> str:
    cit_json = json.dumps(citations, ensure_ascii=False, indent=2)

    # Count how many times each marker appears in this call. If a marker
    # appears more than once, that reference is cited in multiple sentences on
    # this page and each entry must produce its own output item.
    from collections import Counter
    marker_counts = Counter(c.get("marker", "") for c in citations)
    repeated = {m: n for m, n in marker_counts.items() if n > 1}
    repeat_note = ""
    if repeated:
        pairs = ", ".join(f'"{m}" \u00d7 {n}' for m, n in sorted(repeated.items()))
        repeat_note = (
            f"\nNOTE: The following marker(s) appear more than once in CITATIONS "
            f"because the same reference is cited in multiple different sentences "
            f"on this page: {pairs}. "
            f"Each entry has a different citing_fragment pointing to a different sentence. "
            f"You MUST produce one separate output item per entry — do NOT merge or skip any.\n"
        )

    img0_note = (
        "IMAGE 0 = bottom strip of the PREVIOUS page — use ONLY to recover "
        "a sentence that started there and continues here.\n"
        if has_prev else
        "IMAGE 0 = not provided (first page).\n"
    )
    return (
        f"PRIMARY PAGE: {page_1_based}\n\n"
        f"CITATIONS to restore ({len(citations)} total):\n{cit_json}\n"
        + repeat_note + "\n"
        + img0_note +
        "IMAGE 1 = full primary page.\n"
        "IMAGE 2+ = top strip(s) of following page(s) for sentence completion.\n\n"
        "For each citation entry: use citing_fragment to locate its specific citing "
        "sentence on the page, reconstruct it fully across page boundaries if needed, "
        "then replace the marker INLINE with the reference entry. "
        "Produce exactly one output item per input entry. Return strict JSON."
    )



# ══════════════════════════════════════════════════════════════════
#  RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════

# PyMuPDF renders at 72 DPI; the pixmap matrix scales that. Converting DPI to a
# zoom factor lets callers work in DPI.
def _dpi_to_zoom(dpi: float) -> float:
    return float(dpi) / 72.0


def render_png(doc: fitz.Document, page0: int, dpi: float,
               clip: Optional[fitz.Rect] = None) -> bytes:
    """Render a page (or a clipped sub-rectangle) to PNG bytes at the given DPI."""
    page = doc.load_page(page0)
    z = _dpi_to_zoom(dpi)
    mat = fitz.Matrix(z, z)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return pix.tobytes("png")


def bottom_clip(doc: fitz.Document, page0: int, start_frac: float) -> fitz.Rect:
    r = doc.load_page(page0).rect
    return fitz.Rect(0, r.height * start_frac, r.width, r.height)


def top_clip(doc: fitz.Document, page0: int, height_frac: float) -> fitz.Rect:
    r = doc.load_page(page0).rect
    return fitz.Rect(0, 0, r.width, r.height * height_frac)


def b64_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def make_blank_png(width_pt: int = 612, height_pt: int = 200,
                   dpi: float = TOP_DPI) -> bytes:
    """Blank white PNG placeholder for when there is no next page. Dimensions
    are in PDF points (1/72 inch) scaled to the requested DPI."""
    def chunk(ct: bytes, data: bytes) -> bytes:
        c = ct + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    z = _dpi_to_zoom(dpi)
    w, h = int(width_pt * z), int(height_pt * z)
    compressed = zlib.compress(b"".join(b"\x00" + b"\xff" * (w * 3) for _ in range(h)))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


# ══════════════════════════════════════════════════════════════════
#  PHASE 0: text-layer reference section detection
# ══════════════════════════════════════════════════════════════════

def find_ref_start_page(doc: fitz.Document) -> Optional[int]:
    """
    Scan every page's text layer for a known reference-section heading.
    Returns the 0-based index of the first such page, or None if not found.

    Checks every non-empty line of every text block, not just each block's
    first line, because some PDFs put the heading inside a block whose first
    line is a running header, column heading, or figure caption. For example,
    arXiv:1111.3806 has the heading "REFERENCES" that a first-line-only match
    missed.

    Matching is case-insensitive with the trailing colon or period stripped. A
    line also matches if it starts with a heading keyword, covering "References
    and Notes" or "Bibliography (Selected)". A match is accepted only when the
    line is plausible as a heading: at most 60 chars, and either standalone or
    followed by content typical of a reference-section start (a digit, a
    bracketed index, or a capitalised surname).
    """
    for page0 in range(doc.page_count):
        page   = doc.load_page(page0)
        blocks = page.get_text("blocks")          # list of (x0,y0,x1,y1,text,…)
        for block in blocks:
            raw = block[4].strip()
            if not raw:
                continue
            for line in raw.splitlines():
                line_stripped = line.strip()
                if not line_stripped or len(line_stripped) > 60:
                    continue
                key = line_stripped.lower().rstrip(".:").strip()
                if key in _REF_HEADINGS:
                    return page0
                for heading in _REF_HEADINGS:
                    if key.startswith(heading):
                        # Avoid matching "References:" mid-sentence: the
                        # heading must be the whole line or followed only by
                        # short decoration.
                        tail = key[len(heading):].strip()
                        if not tail or re.fullmatch(r"[0-9\s().,:;\-–—]*", tail):
                            return page0
    return None



def clip_to_y_ratio_with_buffer(doc: fitz.Document, page0: int, end_ratio: float,
                                 buffer: float = 0.0) -> fitz.Rect:
    """Clip to y=[0, end_ratio - buffer] of the page. A positive buffer shrinks
    the rectangle from the bottom, used on mixed pages to keep reference-list
    text out of the body-text image sent to detect."""
    r = doc.load_page(page0).rect
    ratio = max(0.0, min(1.0, end_ratio - buffer))
    y = ratio * r.height
    return fitz.Rect(0, 0, r.width, y)


def normalize_url(value: str) -> Optional[str]:
    """Normalize real printed URLs and domains only. Does not turn bare arXiv
    IDs, astro-ph identifiers, or bare doi: strings into URLs."""
    s = (value or "").strip().strip(".,;:()[]{}<>")
    if not s:
        return None
    s = re.sub(r"\s+", "", s)
    s = s.replace("http:///", "http://").replace("https:///", "https://")
    s = s.replace("http:////", "http://").replace("https:////", "https://")
    if s.lower().startswith(("arxiv:", "doi:")) or re.match(r"^astro-ph/", s, re.I):
        return None
    return s


def is_real_url(value: str) -> bool:
    s = (value or "").strip().strip(".,;:()[]{}<>")
    if not s:
        return False
    if s.lower().startswith(("arxiv:", "doi:")) or re.match(r"^astro-ph/", s, re.I):
        return False
    if re.match(r"^(https?|ftp)://", s, re.I):
        return True
    if re.match(r"^www\.", s, re.I):
        return True
    if re.match(r"^(github|gitlab)\.com/", s, re.I):
        return True
    if re.match(r"^(arxiv\.org|doi\.org)/", s, re.I):
        return True
    if re.match(r"^(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?::\d+)?(?:/.*)?$", s) and "." in s:
        return True
    return False


def filter_real_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls or []:
        cleaned = normalize_url(u)
        if not cleaned or not is_real_url(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out




# ══════════════════════════════════════════════════════════════════
#  POST-PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════

# Ligatures common in academic PDFs. They break string matching when one side
# (body text) has the ligature glyph and the other (a key from extracted
# authors) has the ASCII expansion.
_LIGATURE_MAP = {
    "\uFB00": "ff",   # ﬀ
    "\uFB01": "fi",   # ﬁ
    "\uFB02": "fl",   # ﬂ
    "\uFB03": "ffi",  # ﬃ
    "\uFB04": "ffl",  # ﬄ
    "\uFB05": "ft",   # ﬅ
    "\uFB06": "st",   # ﬆ
    # Dash and space variants
    "\u2013": "-",    # en-dash, normalised for range expansion
    "\u2014": "-",    # em-dash
    "\u2212": "-",    # minus sign U+2212
    "\u00A0": " ",    # non-breaking space
    "\u2009": " ",    # thin space
    "\u202F": " ",    # narrow no-break space
}

def normalize_text_for_matching(s: str) -> str:
    """Return a version of s for cross-side string matching. Expands ligatures
    to ASCII, maps en/em/Unicode-minus dashes to hyphen, and normalises
    whitespace. Case is left unchanged. Do not use this for values stored in
    original_sentence or full_text, which must stay verbatim; use it for keys,
    anchors, and variants.
    """
    if not s:
        return s
    out_chars = []
    for ch in s:
        out_chars.append(_LIGATURE_MAP.get(ch, ch))
    out = "".join(out_chars)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def norm_key(s: str) -> str:
    return " ".join(normalize_text_for_matching(s).strip().split()).lower()

def sentence_complete(s: str) -> bool:
    """Heuristic check for whether s ends with terminal punctuation.

    Conservative. True for clearly terminated sentences, including quoted and
    parenthesised forms. False for obvious mid-sentence truncations that end
    with a comma, colon, or conjunction.
    """
    t = " ".join(s.strip().split())
    if not t:
        return True
    # Strip trailing closers and see if what's left ends with sentence-terminal punctuation
    stripped = t
    for _ in range(3):
        if stripped and stripped[-1] in ('"', "'", "\u201D", "\u2019", ")", "]", "}"):
            stripped = stripped[:-1].rstrip()
        else:
            break
    if not stripped:
        return False
    if stripped[-1] in ".?!":
        # A trailing abbreviation period is not a sentence end. Soft check;
        # the restore phase is authoritative.
        tail = stripped.split()[-1].lower() if stripped.split() else ""
        if tail in {"e.g.", "i.e.", "cf.", "cfr.", "al.", "fig.", "eq.", "ref.",
                    "sec.", "ch.", "vol.", "no.", "pp.", "ed.", "eds.", "etc.",
                    "vs.", "inc.", "ltd.", "co.", "jr.", "sr.", "dr.", "mr.",
                    "mrs.", "ms.", "prof."}:
            return False
        return True
    # Ends with a connector or continuation marker -> definitely not complete
    if stripped.endswith((",", ";", ":", "-", "–", "—")):
        return False
    return False

def _fuzzy_key(s: str) -> str:
    """Stable key for fuzzy sentence matching across phases. The cross-page
    dedup pass uses it to recognise that the same sentence reported by two
    adjacent pages, with differing whitespace, ligatures, or trailing
    punctuation, is a duplicate.
    """
    return re.sub(r"\s+", " ",
                  normalize_text_for_matching(s).lower()).strip()


# ══════════════════════════════════════════════════════════════════
#  REFERENCE LOOKUP HELPERS
# ══════════════════════════════════════════════════════════════════

def build_ref_index(references: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup dict from normalised marker key to reference entry.

    Keys come from two fields the reference-extraction model provides:
      1) index: the label printed in the reference list ("1", "[12]", "Smith03")
      2) citation_keys: surface forms the model expects the reference could take
         when cited in body text. The model handles author ordering, particle
         surnames, collaboration conventions, "and" vs "&", "et al." spellings,
         and year suffixes, so this function does no surname parsing.

    This function only does surface character transformations: ligature, dash,
    and whitespace normalisation for cross-glyph matching; bracket stripping
    ([1] to 1, (Smith, 2020) to Smith, 2020); and trailing-punctuation
    stripping (.,;).
    """
    index: Dict[str, Dict] = {}

    def register(key_string: str, ref: Dict) -> None:
        for variant in _marker_variants(key_string):
            # Ligature/dash/whitespace-normalised form
            key_norm = norm_key(variant)
            if key_norm:
                index.setdefault(key_norm, ref)
            # Plain whitespace+case form as a non-ligature fallback
            key_plain = " ".join(variant.strip().split()).lower()
            if key_plain:
                index.setdefault(key_plain, ref)

    for ref in references:
        raw_index = (ref.get("index") or "").strip()
        if raw_index:
            register(raw_index, ref)
        for k in (ref.get("citation_keys") or []):
            if isinstance(k, str) and k.strip():
                register(k, ref)
    return index


def _marker_variants(raw: str) -> List[str]:
    """Return surface-form variants of a marker string.

    Character-level transformations only, no semantic parsing. The caller
    (build_ref_index or lookup_marker) applies ligature and whitespace
    normalisation on top of these variants via norm_key.
    """
    s = (raw or "").strip()
    if not s:
        return []
    variants = {s}
    # Strip outer brackets or parens
    stripped = re.sub(r"^[\[\(]|[\]\)]$", "", s)
    variants.add(stripped)
    # Strip trailing punctuation often appended in body text
    variants.add(stripped.rstrip(".,;:"))
    # Add bracketed and parenthesised forms so a key stored as "Smith, 2020"
    # matches a detected marker "(Smith, 2020)" and vice versa.
    variants.add(f"[{stripped}]")
    variants.add(f"({stripped})")
    return [v for v in variants if v]


def lookup_marker(marker: str, ref_index: Dict[str, Dict]) -> Optional[Dict]:
    """Look up a detected marker in the pre-built index.

    Per variant, try the ligature-normalised key first, then the plain
    whitespace+case key (which handles model ligature mismatches).
    """
    if not marker:
        return None
    for variant in _marker_variants(marker):
        key_norm = norm_key(variant)
        if key_norm and key_norm in ref_index:
            return ref_index[key_norm]
        key_plain = " ".join(variant.strip().split()).lower()
        if key_plain and key_plain in ref_index:
            return ref_index[key_plain]
    return None


def allowed_marker_list(references: List[Dict[str, Any]]) -> List[str]:
    """Return all surface-form markers the detect prompt may accept: the
    model-supplied citation_keys plus the printed reference-list index label,
    kept in raw (un-normalised) form so the detect model sees them as an author
    would have typed them.
    """
    out: List[str] = []
    seen = set()

    def add(val: str) -> None:
        v = (val or "").strip()
        if not v:
            return
        k = v.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(v)

    for ref in references:
        raw_index = (ref.get("index") or "").strip()
        if raw_index:
            add(raw_index)
            # Also register the stripped form so a reference printed as "[5]"
            # is accepted when body text cites it as "5".
            stripped = re.sub(r"^[\[\(]|[\]\)]$", "", raw_index).strip()
            if stripped and stripped != raw_index:
                add(stripped)
        for k in (ref.get("citation_keys") or []):
            if isinstance(k, str):
                add(k)
    return out


def detect_index_style(references: List[Dict[str, Any]]) -> str:
    """Classify the dominant indexing style of the reference list as one of
    'numeric', 'author_year', 'alpha', or 'mixed'. Passed to the detect prompt
    so it can decide whether superscript numbers are candidate citations or
    footnotes.
    """
    numeric = 0
    alpha = 0
    author_year_only = 0
    total_with_index = 0
    for ref in references:
        raw = (ref.get("index") or "").strip()
        stripped = re.sub(r"^[\[\(]|[\]\)]$", "", raw).strip().rstrip(".")
        if not stripped:
            # No explicit index label means author-year convention.
            if ref.get("authors") and ref.get("year"):
                author_year_only += 1
            continue
        total_with_index += 1
        if re.fullmatch(r"\d+", stripped):
            numeric += 1
        elif re.fullmatch(r"[A-Za-z]{2,}\d{0,4}[a-z]?", stripped) or \
             re.fullmatch(r"[A-Z]{2,}\+?\d{0,4}", stripped):
            alpha += 1

    if total_with_index == 0 and author_year_only > 0:
        return "author_year"
    if total_with_index == 0:
        return "mixed"
    # 80% or more of one style counts as pure.
    if numeric / max(1, total_with_index) >= 0.80:
        return "numeric"
    if alpha / max(1, total_with_index) >= 0.80:
        return "alpha"
    return "mixed"


def strip_leading_reference_marker(full_text: str, marker: str) -> str:
    """Remove the leading citation label from a bibliography entry before inline insertion.

    Examples:
      marker='3'     + '[3] Title...'   -> 'Title...'
      marker='Smith' + 'Smith. Title'   -> unchanged unless it is a clear leading label
    """
    text = (full_text or '').strip()
    if not text:
        return text

    variants = []
    m = (marker or '').strip()
    if m:
        variants.extend([
            rf'^\s*\[{re.escape(m)}\]\s*',
            rf'^\s*\({re.escape(m)}\)\s*',
            rf'^\s*{re.escape(m)}\.\s*',
            rf'^\s*{re.escape(m)}\s+',
        ])
        # If marker is numeric, also strip plain numeric forms around it.
        num = re.sub(r'^[\[(]?|[\])]?$', '', m)
        if num.isdigit():
            variants.extend([
                rf'^\s*\[{re.escape(num)}\]\s*',
                rf'^\s*\({re.escape(num)}\)\s*',
                rf'^\s*{re.escape(num)}\.\s*',
                rf'^\s*{re.escape(num)}\s+',
            ])

    for pat in variants:
        new_text = re.sub(pat, '', text, count=1)
        if new_text != text:
            return new_text.strip()
    return text


def normalize_marker_for_bracket(marker: str) -> str:
    """Normalize a citation marker for inline bracketed restoration.

    Examples:
      '[3]' -> '3'
      '(3)' -> '3'
      '3'   -> '3'
      'Smith et al., 2019' -> 'Smith et al., 2019'
    """
    m = (marker or '').strip()
    if not m:
        return m
    if (m.startswith('[') and m.endswith(']')) or (m.startswith('(') and m.endswith(')')):
        m = m[1:-1].strip()
    return m


# ── Phase 3b: context extraction ─────────────────────────────────

CONTEXT_SYSTEM = r"""
You are a meticulous document analyst specialising in academic PDFs.
Your job is to extract the immediate *paragraph-local* context around given target sentences
that appear in the BODY TEXT of an academic paper and contain bibliography citation markers.

You are given:
- IMAGE 0: bottom strip of previous page (optional; use only if paragraph clearly continues)
- IMAGE 1: full primary page — this is the page whose body text you are analysing
- IMAGE 2+: top strip(s) of next page(s) (optional; use only if paragraph clearly continues)

You are also given a JSON list called TARGETS. Each target has:
- original_sentence (string) REQUIRED — you MUST echo this back verbatim, character-for-character.
- match_hints (object) OPTIONAL — short snippets/variants to help you find the sentence.
- layout_hint (object) OPTIONAL — rough column/y positioning and gap/indent signals.

If a hint conflicts with the image, TRUST THE IMAGE.

════════════════════════════════════════
WHAT THESE SENTENCES ARE
════════════════════════════════════════
Every target sentence is a BODY-TEXT sentence that cites one or more bibliography references.
It will contain at least one citation marker in one of these styles — use the marker as your
primary visual anchor to locate the sentence on the page:

  • Bracketed numbers (single, grouped, ranged):
      [1]   [23]   [6,7]   [21–23]   [21-23]   [26][27]
      Bracket locators:   [15, Chapter 6]   [N, §3]   [N, Theorem 2]
  • "Author et al. [N]" mixed style:
      Smith et al. [5]   Jones and Lee [12]
  • Paren numbers (use only when paper's bibliography is numeric):
      (1)   (23)   (6,7)   (21–23)
  • Author-year — narrative (subject of sentence):
      Smith (2023)   Smith et al. (2019)   Smith and Jones (2021)   Smith & Jones (2021)
      van der Berg (2019)   Virgo Collaboration (2009)
  • Author-year — parenthetical (grouped, separated by comma or semicolon):
      (Smith, 2023)   (Smith et al. 2007)   (Smith & Jones, 2021)
      (van der Berg, 2019)   (Desper and Gascuel, 2002a)
      (Virgo Collaboration 2009)   (the LIGO Scientific Collaboration 2010)
      (Smith, 2023; Jones, 2022)   (Skilling 2004, Skilling 2006, Feroz et al. 2009)
  • Bracketed author-year:
      [Smith, 2023]   [Jones & Lee, 2021]   [van der Berg, 2019]
  • Alpha-numeric keys:
      [Smi23]   [BFGS]   [HTF09]   [GBC16]   [Knu68]
  • Superscript numbers (only in papers with numeric bibliography indices):
      …observed in recent studies¹.   …previous work²,³,⁵ has shown…
  • Citations inside parentheticals (the host sentence is the target):
      …found elsewhere (see [N])…   …(cfr. Smith, 2023)…   …(specifically [4,5])…

  Year suffixes MUST be preserved exactly: 2002a, 2019b, 2021c.
  Ligature-normalised forms are equivalent: ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi, ﬄ→ffl.

  The citation marker is a DISTINCTIVE VISUAL ANCHOR — bracketed numbers and
  author-year strings stand out typographically from surrounding prose, making
  the target sentence easy to locate on the page.

  Do NOT confuse citation markers with:
  - Footnote superscripts ¹ ² ³ that link to the footnote zone at the bottom
  - Reference-list entries in the bibliography section
  - Equation numbers (1), (3), or figure/table labels

════════════════════════════════════════
CORE GOAL (PER TARGET)
════════════════════════════════════════
For each target.original_sentence:
1) Locate the sentence in the BODY TEXT of IMAGE 1 (same column).
2) Extract:
   - preceding_sentence: the immediately preceding sentence in the SAME PARAGRAPH (or null)
   - trailing_sentence:  the immediately following sentence in the SAME PARAGRAPH (or null)
3) Set:
   - at_paragraph_start = (preceding_sentence is null)
   - at_paragraph_end   = (trailing_sentence  is null)

IMPORTANT: "Sentence" means the FULL sentence, even if it spans multiple printed lines.
Never return only the last wrapped line (or a fragment) of a multi-line sentence.

The neighbours you return are plain body-text sentences. They may or may not contain their
own citation markers — transcribe them exactly as printed either way.

════════════════════════════════════════
LOCATING POLICY
════════════════════════════════════════
Do NOT give up just because the sentence does not match perfectly character-by-character in print.
Scholarly PDFs often change appearance via ligatures, hyphenation, and spacing.

Treat the target as "located" if you can find a visually matching sentence using ANY of:
- The citation marker itself (e.g., "[23]", "(Smith et al., 2007)") — highly distinctive
- A distinctive 6–12 word span from anchor_start or anchor_end in match_hints
- match_hints variants (underscore ↔ space, normalised ligatures)

Once located, transcribe neighbours EXACTLY as printed.

If you truly cannot locate the target unambiguously, return preceding_sentence=null,
trailing_sentence=null and set both flags true.

════════════════════════════════════════
MULTI-COLUMN PAGES
════════════════════════════════════════
Many scholarly PDFs use two-column or multi-column layouts. You MUST:
- Identify the column of the target by the visible gutter(s).
- Search and extract context ONLY within that same column.
- Never "wrap" from the bottom of one column to the top of the next column.

════════════════════════════════════════
INDENTATION STYLE CALIBRATION
════════════════════════════════════════
Inspect nearby paragraphs in the SAME COLUMN to determine whether the document uses
FIRST-LINE INDENTATION as a paragraph-start convention.

CRITICAL GUARD: Do NOT treat a sentence as "indented paragraph start" if it begins MID-LINE.
Mid-line starts (text to the left on the same printed line) are NOT paragraph starts.

════════════════════════════════════════
PARAGRAPH / BLOCK BOUNDARIES (HARD STOPS)
════════════════════════════════════════
Return null in a direction if ANY of these appears between the target and the candidate:

A) Blank line / large vertical gap
B) First-line indentation — apply only when the neighbour starts at the BEGINNING of a line
C) Section heading / figure or table caption / standalone title
D) Displayed equation / centered block / theorem-like block
E) Lists / bullets / numbered items
   (a prose lead-in ending with ":" can still be a neighbour; bullets after it are a boundary)
F) Footnote zone / bibliography section — anything below a horizontal rule in smaller font,
   or any reference-list entries. Never use footnote text or bibliography entries as neighbours.
G) Column break — never cross columns

NOT boundaries: font changes alone, a normal line wrap, discourse cues ("However," "Therefore,").

When the boundary is ambiguous, prefer returning null (precision over guessing).

════════════════════════════════════════
SAME-LINE NEIGHBOURS
════════════════════════════════════════
The immediate neighbour can be on the SAME PRINTED LINE as the target.

Preceding: if the target begins mid-line, check the text to the left on that same line.
Trailing:  if the target ends mid-line, check if a sentence starts immediately after.

════════════════════════════════════════
SENTENCE ASSEMBLY — PREVENT PARTIAL SENTENCES
════════════════════════════════════════
Sentences frequently span multiple wrapped lines. Return the FULL sentence text.

For preceding_sentence:
1) Identify the nearest preceding sentence end closest to the target.
2) Scan UPWARD, prepending wrapped lines, until you reach a sentence boundary or hard stop.
   Use IMAGE 0 only if the paragraph clearly continues from the previous page.

For trailing_sentence:
1) Identify the nearest trailing sentence start closest to the target.
2) Scan DOWNWARD, appending wrapped lines, until you reach the sentence's end punctuation or a hard stop.
   Use IMAGE 2+ only if the paragraph clearly continues onto the next page.

Do NOT treat periods in common abbreviations as sentence ends:
  e.g., i.e., cf., et al., Fig., Eq., Ref., Sec., Dr., vs., Inc.

════════════════════════════════════════
TRANSCRIPTION RULES
════════════════════════════════════════
- Return neighbours as single-line strings.
- Preserve spelling, capitalisation, punctuation, and citation markers exactly as printed.
- Only fix obvious wrap-hyphenation (end-of-line hyphen continuing the same word).
- Do NOT paraphrase, clean up, or invent text.
- Do NOT output match_hints or layout_hint fields.

Return strict JSON matching the provided schema.
""".strip()


RESTORE_CONTEXT_SYSTEM = (
    "You will perform TWO jobs in a single response, returning ONE JSON object\n"
    "that matches the provided schema. Job A (RESTORATION) produces the\n"
    "restored_sentence and its citation/url/flag fields; Job B (CONTEXT)\n"
    "produces the preceding_sentence / trailing_sentence / paragraph-flag\n"
    "fields. Apply each job's rules exactly as written below. Both jobs operate\n"
    "on the SAME images.\n\n"
    "════════════════════════════════════════════════════════════════════\n"
    "JOB A — RESTORATION  (authoritative for the restore fields)\n"
    "════════════════════════════════════════════════════════════════════\n"
    + RESTORE_SYSTEM
    + "\n\n"
    "════════════════════════════════════════════════════════════════════\n"
    "JOB B — CONTEXT  (authoritative for the context fields)\n"
    "════════════════════════════════════════════════════════════════════\n"
    + CONTEXT_SYSTEM
    + "\n\n"
    "════════════════════════════════════════════════════════════════════\n"
    "OUTPUT — ONE ITEM PER CITING SENTENCE\n"
    "════════════════════════════════════════════════════════════════════\n"
    "Produce one items entry per restored citing sentence. Each entry carries\n"
    "BOTH the Job A fields (original_sentence, restored_sentence,\n"
    "citation_marker, reference_entry, urls, sentence_spans_pages,\n"
    "sentence_starts_prev_page, needs_more_pages, in_caption,\n"
    "inside_parenthetical) AND the Job B fields (preceding_sentence,\n"
    "trailing_sentence, at_paragraph_start, at_paragraph_end) for that same\n"
    "sentence. For Job B, treat each original_sentence you restored as the\n"
    "context target and follow the CONTEXT rules to fill the four context\n"
    "fields. Return strict JSON.\n"
)


def make_restore_context_user_prompt(page_1_based: int,
                                     citations: List[Dict],
                                     layout_hints: Dict[str, Any],
                                     has_prev: bool) -> str:
    """Concatenate the restore user prompt with a context instruction.

    A standalone context stage would receive the restored sentences as a
    TARGETS list from a prior restore call. In the merged call those sentences
    do not exist yet (the model restores them in the same pass), so there is no
    TARGETS handoff. Page-level layout_hints (two-column, split, line height)
    are still passed for column reasoning; per-target match hints only exist to
    locate an already-known sentence and do not apply here.
    """
    restore_part = make_restore_user_prompt(page_1_based, citations, has_prev)
    hints_json = json.dumps(layout_hints, ensure_ascii=False, indent=2)
    img2_note = (
        "IMAGE 2+ = top strips of following page(s) — use ONLY when a sentence "
        "is near the BOTTOM of IMAGE 1 and continues onto the next page.\n"
    )
    return (
        "############### JOB A — RESTORATION (user input) ###############\n"
        + restore_part
        + "\n\n############### JOB B — CONTEXT (user input) ###############\n"
        "For EACH sentence you restore in Job A, also extract its paragraph-local "
        "context on the SAME page: the immediately preceding and trailing sentences "
        "within the same paragraph and column, per the CONTEXT system rules "
        "(gap/indent/heading/caption/equation/list/footnote-zone/column-break and "
        "same-line neighbour checks). Transcribe neighbours verbatim as printed body "
        "text; set preceding_sentence/trailing_sentence to null at a paragraph "
        "boundary and set at_paragraph_start/at_paragraph_end accordingly.\n"
        + img2_note +
        f"\nPAGE-LEVEL LAYOUT HINTS (advisory; trust the image if it disagrees):\n{hints_json}\n\n"
        "Return ONE JSON object whose items each contain BOTH the Job A restore "
        "fields and the Job B context fields for the same sentence."
    )


# ── Phase 3b context helpers ──────────────────────────────────────

def _extract_layout_hints(doc: fitz.Document, page0: int) -> Dict[str, Any]:
    """Layout metadata from the PDF text layer, no API call.

    Provides two-column detection and median line height so the context model
    can judge gap ratios and column membership. Returns minimal defaults when
    the text layer is absent (scanned PDF).
    """
    page = doc.load_page(page0)
    r    = page.rect
    w    = float(r.width)

    try:
        d = page.get_text("dict")
    except Exception:
        return {"page_width": w, "two_column": False}

    lines: List[Dict[str, Any]] = []
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            txt_parts: List[str] = []
            x0s: List[float] = []
            y0s: List[float] = []
            x1s: List[float] = []
            y1s: List[float] = []
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if t:
                    txt_parts.append(t)
                bb = sp.get("bbox")
                if bb:
                    x0s.append(bb[0]); y0s.append(bb[1])
                    x1s.append(bb[2]); y1s.append(bb[3])
            if not txt_parts or not x0s:
                continue
            text = "".join(txt_parts).strip()
            if not text:
                continue
            lines.append({"text": text,
                           "bbox": (min(x0s), min(y0s), max(x1s), max(y1s))})

    if not lines:
        return {"page_width": w, "two_column": False}

    # Median line height
    heights = sorted(
        ln["bbox"][3] - ln["bbox"][1] for ln in lines
        if 2.0 <= ln["bbox"][3] - ln["bbox"][1] <= 60.0
    )
    median_h = heights[len(heights) // 2] if heights else 12.0

    # Two-column detection via gap in x0 distribution
    x0s_sorted = sorted(ln["bbox"][0] for ln in lines)
    split_x: Optional[float] = None
    if len(x0s_sorted) >= 20:
        gaps = [(x0s_sorted[i + 1] - x0s_sorted[i], i)
                for i in range(len(x0s_sorted) - 1)]
        biggest_gap, idx = max(gaps)
        if biggest_gap > 0.10 * w:
            split_x = (x0s_sorted[idx] + x0s_sorted[idx + 1]) / 2.0

    return {
        "page_width":        w,
        "median_line_height": float(median_h),
        "two_column":         split_x is not None,
        "column_split_x":     float(split_x) if split_x is not None else None,
    }



# ══════════════════════════════════════════════════════════════════
#  API CALL WRAPPERS
# ══════════════════════════════════════════════════════════════════

def _call(client: OpenAI, model: str, system: Optional[str],
          content: List[Dict], schema: Dict, schema_name: str,
          max_retries: int = 3, reasoning_effort: str = "medium") -> Dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    for attempt in range(1, max_retries + 1):
        try:
            req = {
                "model": model,
                "input": messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
            resp = client.responses.create(**req)
            return json.loads(resp.output_text)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                print(f"    [!] {schema_name} failed after {max_retries} attempts: {e}")
                return {}
    return {}



# ── Phase 0/1 ────────────────────────────────────────────────────

def call_phase01(client: OpenAI, model: str,
                 page_images: List[bytes],
                 page_numbers: List[int]) -> Dict:
    content: List[Dict] = [
        {"type": "input_text", "text": make_phase01_user_prompt(page_numbers)}
    ]
    for img in page_images:
        content.append({"type": "input_image", "image_url": b64_data_url(img)})

    return _call(client, model, PHASE01_SYSTEM, content,
                 PHASE01_SCHEMA, "identify_reference_layout_and_extract",
                 reasoning_effort="low")


# ── Phase 1 ──────────────────────────────────────────────────────

def call_extract_reflist(client: OpenAI, model: str,
                          page_images: List[bytes],
                          page_numbers: List[int]) -> List[Dict]:
    content: List[Dict] = [
        {"type": "input_text", "text": make_reflist_user_prompt(page_numbers)}
    ]
    for img in page_images:
        content.append({"type": "input_image", "image_url": b64_data_url(img)})

    data = _call(client, model, REFLIST_EXTRACT_SYSTEM, content,
                 REFLIST_SCHEMA, "extract_reference_list",
                 reasoning_effort="low")
    return data.get("references", [])


# ── Phase 2 ──────────────────────────────────────────────────────

def call_detect(client: OpenAI, model: str,
                page_png: bytes, page_1_based: int,
                allowed_markers: Optional[List[str]] = None,
                index_style: str = "mixed") -> Dict:
    content = [
        {"type": "input_text",  "text": make_detect_user_prompt(
            page_1_based, allowed_markers, index_style=index_style)},
        {"type": "input_image", "image_url": b64_data_url(page_png)},
    ]
    data = _call(client, model, DETECT_SYSTEM, content,
                 DETECT_SCHEMA, "detect_citation_markers",
                 reasoning_effort="low")
    data["page_1_based"] = page_1_based
    return data


# ── Phase 3 (merged): restore + context in ONE call ──────────────

def call_restore_context(client: OpenAI, model: str,
                         page_png: bytes, next_top_pngs: List[bytes],
                         page_1_based: int, citations: List[Dict],
                         layout_hints: Dict[str, Any],
                         prev_bottom_png: Optional[bytes] = None) -> Dict:
    """Restore citing sentences AND extract their paragraph-local context in a
    single call over the SAME three images (prev-bottom, full page, next-tops).
    RESTORE_SYSTEM and CONTEXT_SYSTEM are sent verbatim (concatenated in
    RESTORE_CONTEXT_SYSTEM); output is the union of both schemas. Image order
    and reasoning effort match the original call_restore."""
    has_prev  = prev_bottom_png is not None
    user_text = make_restore_context_user_prompt(
        page_1_based, citations, layout_hints, has_prev)
    content: List[Dict] = [{"type": "input_text", "text": user_text}]
    if has_prev:
        content.append({"type": "input_image", "image_url": b64_data_url(prev_bottom_png)})
    content.append({"type": "input_image", "image_url": b64_data_url(page_png)})
    for png in next_top_pngs:
        content.append({"type": "input_image", "image_url": b64_data_url(png)})

    data = _call(client, model, RESTORE_CONTEXT_SYSTEM, content,
                 RESTORE_CONTEXT_SCHEMA, "restore_and_context",
                 reasoning_effort="medium")
    data["base_page_1_based"] = page_1_based
    return data


# ══════════════════════════════════════════════════════════════════
#  PER-PDF PROCESSING
# ══════════════════════════════════════════════════════════════════

def _dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate items by (pdf_file, fuzzy_sentence_key, marker, urls).

    Handles the cross-page case: a sentence starting on page N and ending on
    page N+1 is reported by restore calls from both pages (via IMAGE 0 and
    IMAGE 2+ context). Without dedup it appears twice for the same marker.

    Keeps the first occurrence of each (fuzzy_key, marker, urls) triple.
    Different sentences that share a marker are kept; dedup never crosses
    different sentences.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = (
            it.get("pdf_file", ""),
            _fuzzy_key(it.get("original_sentence", "")),
            (it.get("citation_marker") or "").strip().lower(),
            tuple(sorted(u.lower() for u in (it.get("urls") or []))),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def process_pdf(client: OpenAI, pdf_path: Path,
                out_dir: Path,
                jsonl_handle=None) -> Dict[str, Any]:
    """Process a single PDF end to end.

    If jsonl_handle is given, each restored item is appended to it right after
    its page, so a crash partway through a long document keeps completed pages.
    """

    doc      = fitz.open(pdf_path)
    pdf_name = pdf_path.name
    n_pages  = doc.page_count
    blank    = make_blank_png(dpi=TOP_DPI)

    print(f"\n{pdf_name}  ({n_pages} pages)")

    # ── PHASE 0/1: page-role identification + extraction ─────────
    page_layout: Dict[int, Dict[str, Any]] = {}
    references: List[Dict] = []

    try:
        for batch_start in range(0, n_pages, PHASE01_PAGES_PER_CALL):
            batch = list(range(batch_start, min(n_pages, batch_start + PHASE01_PAGES_PER_CALL)))
            images = [render_png(doc, p, dpi=PHASE01_DPI) for p in batch]
            page_numbers = [p + 1 for p in batch]
            phase01_data = call_phase01(client, PHASE01_MODEL, images, page_numbers)
            for item in phase01_data.get("pages", []):
                page0 = item["page_1_based"] - 1
                page_layout[page0] = {
                    "page_1_based": item["page_1_based"],
                    "role": item.get("role"),
                    "reference_start_y_ratio": item.get("reference_start_y_ratio"),
                }
                references.extend(item.get("references", []))
            if DELAY_SECONDS > 0:
                time.sleep(DELAY_SECONDS)
    except Exception as e:
        print(f"  [!] Combined VLM phase 0/1 failed: {e}")

    # Decide body_pages / ref_pages, in three tiers:
    #   1. Use the Phase 0/1 result (role per page) when available.
    #   2. If Phase 0/1 missed some pages, fall back to text-layer detection of
    #      the reference heading plus reference-page extraction.
    #   3. If both miss (no heading in the text layer and nothing from the
    #      model), retry the classifier on the last 30% of pages, where
    #      references almost always live, instead of treating the whole paper
    #      as body and dropping every citation.
    if len(page_layout) != n_pages:
        ref_start = find_ref_start_page(doc)
        references = []
        if ref_start is None:
            # Last resort: classify the tail of the paper.
            tail_start = max(0, int(n_pages * 0.7))
            tail_pages = list(range(tail_start, n_pages))
            print(f"  [!] No reference heading in text layer; "
                  f"retrying VLM Phase 0/1 on tail pages {tail_start + 1}-{n_pages}…")
            try:
                for batch_start in range(0, len(tail_pages), PHASE01_PAGES_PER_CALL):
                    batch = tail_pages[batch_start: batch_start + PHASE01_PAGES_PER_CALL]
                    images = [render_png(doc, p, dpi=PHASE01_DPI) for p in batch]
                    page_numbers = [p + 1 for p in batch]
                    phase01_data = call_phase01(client, PHASE01_MODEL, images, page_numbers)
                    for item in phase01_data.get("pages", []):
                        page0 = item["page_1_based"] - 1
                        page_layout[page0] = {
                            "page_1_based": item["page_1_based"],
                            "role": item.get("role"),
                            "reference_start_y_ratio": item.get("reference_start_y_ratio"),
                        }
                        references.extend(item.get("references", []))
                    if DELAY_SECONDS > 0:
                        time.sleep(DELAY_SECONDS)
            except Exception as e:
                print(f"  [!] Tail-page VLM Phase 0/1 also failed: {e}")
            # Mark any still-unclassified page as body
            for p in range(n_pages):
                page_layout.setdefault(p, {
                    "page_1_based": p + 1,
                    "role": "body",
                    "reference_start_y_ratio": None,
                })
            body_pages = [p for p, v in sorted(page_layout.items())
                          if v.get("role") in ("body", "mixed")]
            ref_pages = [p for p, v in sorted(page_layout.items())
                         if v.get("role") in ("references", "mixed")]
        else:
            print(f"  References section starts at page {ref_start + 1} "
                  f"({n_pages - ref_start} reference page(s)) [heuristic fallback]")
            body_pages = list(range(ref_start))
            ref_pages = list(range(ref_start, n_pages))
            # Fill in page_layout with default body/references roles
            for p in range(n_pages):
                if p not in page_layout:
                    page_layout[p] = {
                        "page_1_based": p + 1,
                        "role": "references" if p in ref_pages else "body",
                        "reference_start_y_ratio": None,
                    }
            if ref_pages:
                print(f"  Phase 1 fallback: extracting {len(ref_pages)} reference page(s) …")
                for batch_start in range(0, len(ref_pages), REF_PAGES_PER_CALL):
                    batch = ref_pages[batch_start: batch_start + REF_PAGES_PER_CALL]
                    images = [render_png(doc, p, dpi=EXTRACT_DPI) for p in batch]
                    page_numbers = [p + 1 for p in batch]
                    batch_refs = call_extract_reflist(client, EXTRACT_MODEL, images, page_numbers)
                    references.extend(batch_refs)
                    if DELAY_SECONDS > 0:
                        time.sleep(DELAY_SECONDS)
                print(f"  Phase 1 fallback complete: {len(references)} total reference entries extracted.")
            else:
                print("  Phase 1 fallback skipped (no reference pages found).")
    else:
        body_pages = []
        ref_pages = []
        mixed_pages = []
        for page0 in range(n_pages):
            item = page_layout.get(page0, {})
            role = item.get("role")
            if role == "body":
                body_pages.append(page0)
            elif role == "references":
                ref_pages.append(page0)
            elif role == "mixed":
                body_pages.append(page0)
                ref_pages.append(page0)
                mixed_pages.append(page0)

        first_ref = min(ref_pages) + 1 if ref_pages else None
        if first_ref is None:
            print(f"  Combined VLM phase 0/1: no reference pages found; extracted {len(references)} entries.")
        else:
            mixed_note = f", mixed page(s): {[p + 1 for p in mixed_pages]}" if mixed_pages else ""
            print(f"  Combined VLM phase 0/1: first reference page {first_ref}; "
                  f"{len(ref_pages)} reference page(s){mixed_note}; "
                  f"{len(references)} total reference entries extracted.")

    # Keep only bibliography references that contain at least one REAL URL
    url_references: List[Dict[str, Any]] = []
    for ref in references:
        cleaned_urls = filter_real_urls(ref.get("urls", []))
        if not cleaned_urls:
            continue
        ref = dict(ref)
        ref["urls"] = cleaned_urls
        url_references.append(ref)

    print(f"  URL-bearing reference entries kept: {len(url_references)}")

    # Save the URL-bearing reference list (kept even when empty, for debugging).
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / f"{pdf_path.stem}_references_with_urls.json"
    ref_path.write_text(json.dumps(url_references, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # Early exit when nothing to restore
    if not url_references:
        items_path     = out_dir / f"{pdf_path.stem}_url_reference_citations.json"
        sentences_path = out_dir / f"{pdf_path.stem}_url_reference_sentences.json"
        items_path.write_text("[]", encoding="utf-8")
        sentences_path.write_text("[]", encoding="utf-8")
        doc.close()
        return {
            "pdf_file":            pdf_name,
            "total_ref_count":     len(references),
            "url_ref_count":       0,
            "citation_count":      0,
            "refs_json":           str(ref_path),
            "items_json":          str(items_path),
            "sentences_json":      str(sentences_path),
            "index_style":         "n/a",
        }

    ref_index = build_ref_index(url_references)
    allowed_markers = allowed_marker_list(url_references)
    index_style = detect_index_style(url_references)
    print(f"  Reference-list indexing style: {index_style}")

    # ── PHASE 2 + 3: body pages ───────────────────────────────────
    all_items: List[Dict]   = []

    for page0 in tqdm(body_pages, desc=pdf_name, leave=False):
        page_1 = page0 + 1

        layout_item = page_layout.get(page0, {})
        detect_clip = None
        if layout_item.get("role") == "mixed" and layout_item.get("reference_start_y_ratio") is not None:
            # Buffer the body region so a slightly-off estimate of
            # reference_start_y_ratio does not leak reference-list text into
            # the body image sent to detect.
            detect_clip = clip_to_y_ratio_with_buffer(
                doc, page0,
                float(layout_item["reference_start_y_ratio"]),
                buffer=MIXED_PAGE_DETECT_BUFFER,
            )

        # ── Phase 2: detect citation markers ─────────────────────
        page_png = render_png(doc, page0, dpi=DETECT_DPI, clip=detect_clip)
        det      = call_detect(client, DETECT_MODEL, page_png, page_1,
                                allowed_markers, index_style=index_style)

        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

        markers = det.get("markers", [])
        if not markers:
            continue  # no citations on this page

        # Resolve markers against the reference index. Superscript markers are
        # also gated here, accepted only when index_style is 'numeric', which
        # repeats what the detect prompt is already told.
        citations_for_restore: List[Dict] = []
        for m in markers:
            style = m.get("marker_style", "")

            if style == "superscript_number" and index_style != "numeric":
                continue

            ref = lookup_marker(m["marker"], ref_index)
            if ref is None:
                print(f"    [~] p{page_1}: marker '{m['marker']}' not found in reference list")
                continue

            urls = filter_real_urls(ref.get("urls", []) or [])
            if not urls:
                continue

            reference_entry = strip_leading_reference_marker(
                ref.get("full_text", ""),
                m["marker"],
            )
            citations_for_restore.append({
                "marker":               m["marker"],
                "marker_display":       normalize_marker_for_bracket(m["marker"]),
                "marker_style":         m["marker_style"],
                "citing_fragment":      m.get("citing_fragment", ""),
                "visible_group":        m.get("visible_group"),
                "inside_parenthetical": bool(m.get("inside_parenthetical", False)),
                "reference_entry":      reference_entry,
                "urls":                 urls,
                "doi":                  ref.get("doi", ""),
            })

        if not citations_for_restore:
            continue

        # ── Phase 3 (merged): restore + context in ONE call ───────
        restore_png = render_png(doc, page0, dpi=RESTORE_DPI, clip=detect_clip)

        prev_bottom_png: Optional[bytes] = None
        if page0 > 0:
            if PREV_PAGE_FRAC >= 1.0:
                prev_bottom_png = render_png(doc, page0 - 1, dpi=TOP_DPI)
            else:
                pb_clip         = bottom_clip(doc, page0 - 1, 1.0 - PREV_PAGE_FRAC)
                prev_bottom_png = render_png(doc, page0 - 1, dpi=TOP_DPI, clip=pb_clip)

        next_top_pngs: List[bytes] = []
        for k in range(1, MAX_NEXT_PAGES + 1):
            if page0 + k >= n_pages:
                next_top_pngs.append(blank)
                break
            t_clip = top_clip(doc, page0 + k, TOP_FRAC)
            next_top_pngs.append(render_png(doc, page0 + k, dpi=TOP_DPI, clip=t_clip))

        # Page-level layout hints only (per-target hints presuppose known
        # sentences, which the merged call produces itself).
        layout_hints = _extract_layout_hints(doc, page0)

        restore_data = call_restore_context(
            client          = client,
            model           = RESTORE_MODEL,
            page_png        = restore_png,
            next_top_pngs   = next_top_pngs,
            page_1_based    = page_1,
            citations       = citations_for_restore,
            layout_hints    = layout_hints,
            prev_bottom_png = prev_bottom_png,
        )
        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

        items = restore_data.get("items", [])

        # Flag-only pass for incomplete sentences.
        incomplete_count = 0
        for it in items:
            if it.get("needs_more_pages") or not sentence_complete(
                    it.get("original_sentence", "")):
                it["_incomplete_sentence"] = True
                incomplete_count += 1
        if incomplete_count:
            print(f"    [!] p{page_1}: {incomplete_count} sentence(s) may be incomplete")

        # Context fields (preceding_sentence, trailing_sentence,
        # at_paragraph_*) come back on each item from the merged call, bound to
        # the restore fields by object identity rather than a string match.
        # Normalise empty neighbours to null and derive the paragraph flags.
        for it in items:
            prev_s = it.get("preceding_sentence", None)
            next_s = it.get("trailing_sentence", None)
            prev_s = prev_s if (isinstance(prev_s, str) and prev_s.strip()) else None
            next_s = next_s if (isinstance(next_s, str) and next_s.strip()) else None
            it["preceding_sentence"] = prev_s
            it["trailing_sentence"]  = next_s
            it["at_paragraph_start"] = bool(prev_s is None)
            it["at_paragraph_end"]   = bool(next_s is None)

        # Collect restored occurrences without deduping within a page, since
        # different sentences can share a marker. The per-PDF dedup pass below
        # handles the cross-page case where the same (sentence, marker) pair is
        # reported by two adjacent pages.
        page_items = []
        for it in items:
            restored = it.get("restored_sentence", "").strip()
            if not restored:
                continue
            it["pdf_file"] = pdf_name
            it["page"]     = page_1
            page_items.append(it)

        all_items.extend(page_items)

        # Flush this page's items to JSONL now so a later crash keeps
        # completed pages.
        if jsonl_handle is not None:
            for it in page_items:
                jsonl_handle.write(json.dumps(it, ensure_ascii=False) + "\n")
            jsonl_handle.flush()

        found = sum(1 for it in page_items)
        ctx_found = sum(
            1 for it in page_items
            if it.get("preceding_sentence") is not None or it.get("trailing_sentence") is not None
        )
        print(f"  p{page_1:>4}: {len(markers)} marker(s) detected → "
              f"{len(citations_for_restore)} URL-bearing reference(s) resolved → "
              f"{found} sentence(s) restored → {ctx_found} with context")

    doc.close()

    # ── Cross-page duplicate removal ──────────────────────────────
    before = len(all_items)
    all_items = _dedup_items(all_items)
    removed = before - len(all_items)
    if removed:
        print(f"  Deduplicated {removed} cross-page duplicate sentence occurrence(s).")

    # ── Save outputs ──────────────────────────────────────────────
    items_path     = out_dir / f"{pdf_path.stem}_url_reference_citations.json"
    sentences_path = out_dir / f"{pdf_path.stem}_url_reference_sentences.json"

    items_path.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")
    sentences_path.write_text(
        json.dumps([it["restored_sentence"] for it in all_items],
                   ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "pdf_file":            pdf_name,
        "total_ref_count":     len(references),
        "url_ref_count":       len(url_references),
        "citation_count":      len(all_items),
        "refs_json":           str(ref_path),
        "items_json":          str(items_path),
        "sentences_json":      str(sentences_path),
        "index_style":         index_style,
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract URL-bearing reference citations from scholarly PDFs.")
    parser.add_argument(
        "input", nargs="?", default=INPUT_DIR,
        help="Path to a PDF file or a folder of PDFs (searched recursively). "
             "Defaults to INPUT_DIR in CONFIG.")
    parser.add_argument(
        "output", nargs="?", default=OUTPUT_DIR,
        help="Folder to write JSON results into. Defaults to OUTPUT_DIR in CONFIG.")
    args = parser.parse_args()

    api_key = OPENAI_API_KEY
    if not api_key:
        sys.exit(
            "ERROR: OPENAI_API_KEY is empty.\n"
            "Set it as an environment variable before running:\n"
            "  export OPENAI_API_KEY='sk-...'    (bash/zsh)\n"
            "  setx OPENAI_API_KEY sk-...        (Windows PowerShell, then new shell)"
        )

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()

    if in_path.is_dir():
        pdf_files = sorted(in_path.rglob("*.pdf"))
    elif in_path.is_file() and in_path.suffix.lower() == ".pdf":
        pdf_files = [in_path]
    else:
        sys.exit(f"ERROR: Input is not a PDF file or a folder: {in_path}")

    if not pdf_files:
        sys.exit(f"No PDFs found under: {in_path}")

    client = OpenAI(api_key=api_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input          : {in_path}")
    print(f"Output         : {out_dir}")
    print(f"Phase0/1 model : {PHASE01_MODEL}")
    print(f"Extract model  : {EXTRACT_MODEL}")
    print(f"Detect model   : {DETECT_MODEL}")
    print(f"Restore model  : {RESTORE_MODEL}")
    print(f"PDFs found     : {len(pdf_files)}\n")

    summaries: List[Dict[str, Any]] = []
    all_jsonl  = out_dir / "_ALL.citations.jsonl"
    summary_path = out_dir / "_SUMMARY.json"

    # Open the JSONL once per run. process_pdf streams items into it per page,
    # so a crash or kill mid-run leaves every completed page on disk.
    with all_jsonl.open("w", encoding="utf-8") as jl:
        for pdf_path in tqdm(pdf_files, desc="PDFs"):
            try:
                summary = process_pdf(client, pdf_path, out_dir, jsonl_handle=jl)
            except KeyboardInterrupt:
                # Interrupt without losing what is already on disk.
                print("  [!] Interrupted by user; writing partial summary and exiting.")
                summary_path.write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                raise
            except Exception as e:
                print(f"  [!] Failed: {pdf_path.name}: {e}")
                summary = {"pdf_file": pdf_path.name, "error": str(e)}
            summaries.append(summary)

            # Rewrite the summary after each PDF so progress is visible mid-run.
            summary_path.write_text(
                json.dumps(summaries, ensure_ascii=False, indent=2),
                encoding="utf-8")

    total = sum(s.get("citation_count", 0) for s in summaries)
    total_refs = sum(s.get("url_ref_count", 0) for s in summaries)
    print("\n" + "=" * 60)
    print("All done!")
    print(f"   PDFs processed        : {len(pdf_files)}")
    print(f"   URL-bearing refs kept : {total_refs}")
    print(f"   Total sentences saved : {total}")
    print(f"   Per-PDF JSONs in      : {out_dir}")
    print(f"   Combined JSONL        : {all_jsonl}")
    print(f"   Summary               : {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()