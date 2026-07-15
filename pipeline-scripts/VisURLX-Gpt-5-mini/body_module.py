

import base64
import json
import os
import re
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


#
#  CONFIG
#

OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY", "")
SINGLE_MODEL  = "gpt-5-mini"   # one-call detect + target sentence

INPUT_DIR  = Path("./pdfs")
OUTPUT_DIR = Path("./out/body")

# Rendering
PAGE_DPI       = 300   
TOP_DPI        = 300   
TOP_FRAC       = 0.50  
                       
                      
PREV_PAGE_FRAC = 0.50 
MAX_NEXT_PAGES = 1    

DELAY_SECONDS  = 0.2   



BODY_URL_SINGLE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_1_based": {"type": "integer"},
        "has_body_urls": {"type": "boolean"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url_printed":       {"type": "string"},
                    "original_sentence": {"type": "string"},
                    "preceding_sentence": {"type": ["string", "null"]},
                    "trailing_sentence":  {"type": ["string", "null"]},
                    "at_paragraph_start": {"type": "boolean"},
                    "at_paragraph_end":   {"type": "boolean"},
                    "url_lines_joined":  {"type": "boolean"},
                    "url_span_pages":    {"type": "boolean"},
                },
                "required": [
                    "url_printed", "original_sentence",
                    "preceding_sentence", "trailing_sentence",
                    "at_paragraph_start", "at_paragraph_end",
                    "url_lines_joined", "url_span_pages",
                ],
            },
        },
    },
    "required": ["page_1_based", "has_body_urls", "items"],
}


#
#  PROMPTS
#

BODY_URL_SINGLE_SYSTEM = r"""
You are a document analyst working on scholarly PDFs. Your job is to find web-address URLs printed in the BODY TEXT of a page, and for each one return its sentence and that sentence's neighbours.

IMAGES YOU RECEIVE
- IMAGE 0 (optional) = BOTTOM CROP of the PREVIOUS page. Provided ONLY so you
  can recover the opening of a URL or sentence that began on the previous page
  and continues onto IMAGE 1. It is NOT a detection surface.
- IMAGE 1 = the FULL CURRENT page. This is the ONLY page you detect URLs on.
- IMAGE 2+ (optional) = TOP CROP(S) of the FOLLOWING page(s). Provided ONLY so
  you can finish a URL or sentence that starts on IMAGE 1 and spills past the
  bottom of IMAGE 1. They are NOT detection surfaces.

You are also given the OCR/text-layer text for IMAGE 1. Use the page image as the
primary source for layout, region, and reading-order decisions. Use the text layer
only to resolve individual URL characters, under the rules below.

YOUR TASK — THREE JOBS IN ONE PASS
(1) DETECTION — decide which URLs on IMAGE 1 are kept.
(2) TARGET-SENTENCE ASSEMBLY — for each kept URL, return the full sentence that
    contains it.
(3) PARAGRAPH-LOCAL CONTEXT — for each kept URL, return the sentence immediately
    before and the sentence immediately after that target sentence within the SAME
    paragraph, and flag whether the target is at the paragraph start or end.

Because all three happen together, the occurrence you are reporting is the one you
are reading right now; you never need a separate locator.

DETECTION SCOPE — STRICT
Detect and report a body URL ONLY if it is printed on IMAGE 1 (the current page).
NEVER emit an item for a URL whose printed occurrence lives on IMAGE 0 or IMAGE 2+.
A URL you can see only on a neighbour crop is there purely to let you complete a
wrapped URL or sentence whose body is on IMAGE 1 — it is not itself a current-page
detection. If a URL appears on a neighbour crop but does not continue an IMAGE 1
occurrence, ignore it completely.

Report every body URL printed on IMAGE 1, and for each, return the full
sentence that contains it. If the same URL appears in more than one sentence on
IMAGE 1, emit each occurrence as its own item with its own sentence; do not
merge them and do not drop one as a duplicate.

WHAT IS BODY TEXT
The body text is the author's own content on the page: the abstract, all paper sections
including the Acknowledgements and Appendices, section headings, figure and table
captions, and equations.

The KIND of text does not matter. A URL in a caption or an equation is body text just as
much as a URL in a running paragraph. Do not reject a URL merely because the text around
it is not flowing prose.

WHERE the URL is printed does matter, and you must check it before keeping the URL.

LOCATION CHECK (MANDATORY)
Look at the page image and decide where the URL is printed. Keep it ONLY if it is printed
in the body text. Reject it if it is printed in any of the following, even when the text
reads like ordinary prose:

- FOOTNOTE AREA. Sits at the bottom of the page, usually below a horizontal rule and set
  in a smaller font, with each entry tied to a superscript marker in the body above. A
  footnote often reads exactly like a body sentence. Reject it anyway. This includes a
  note stating where the final published version of this paper can be found.

- BIBLIOGRAPHY OR REFERENCE ENTRY. A list of cited works, each carrying author names and a
  publication venue with a year. Reject any URL inside such an entry, however it is
  formatted, whether or not it is numbered, and even if the entry begins with words like
  "Available at".

- HEADER, FOOTER, RUNNING TITLE, PAGE NUMBER, AUTHOR OR AFFILIATION BLOCK.

- PUBLISHER BOILERPLATE. Text placed on the page by the publisher rather than written by
  the authors: copyright and license notices such as Creative Commons statements, terms of
  use, open-access statements, ORCID links, a link to the published version of the article,
  and links to services that validate a citation, report updates to the article, or
  generate a citation string.

- DATA TABLE ROW. A row of a data table in which a domain-like string sits alongside
  numeric columns, counts, scores, citation keys, or record identifiers. The domain-like
  string there is a row label or a dataset name, not a web address the authors are
  directing the reader to. Reject it.

- CODE, COMMAND, OR CONFIGURATION BLOCK. Shell transcripts, terminal sessions, program
  output, logs, source code, and configuration or markup blocks. Reject a web-like string
  that appears only as part of such a block.

WHAT COUNTS AS A URL
A URL is any address that identifies a resource accessible over a network. It does not need an explicit scheme prefix to qualify. The test is: could a person or client program use this string to reach a real resource over the internet or a network?

INCLUDE any of the following:
- Addresses with a resource-access scheme: http://, https://, ftp://, ftps://, sftp://, s3://, gs://, az://, or any analogous scheme that identifies a network-accessible resource location.
- Addresses beginning with "www." with or without an explicit scheme.
- Bare domain-name addresses with no scheme prefix — a human-readable hostname string (using letters, digits, hyphens, and dots) that identifies a real network host, with or without a path. These are valid regardless of whether they carry a path component.
- DOIs written as a web address: an https://doi.org/ or http://dx.doi.org/ URL.

NOT A URL — reject these wherever they appear:
- Bare scholarly identifiers. A DOI is bare unless it is printed as a doi.org or dx.doi.org web address: reject 10.NNNN/suffix and reject doi:10.NNNN/suffix. Reject an arXiv identifier such as arXiv:NNNN.NNNNN and a legacy arXiv identifier such as astro-ph/NNNNNNN. These are identifiers, not addresses. Do not turn one into a URL by adding a prefix yourself.
- Standalone hostnames, server names, bare host:port strings, or bare IPv4 or IPv6 addresses used as command arguments, resolver targets, record values, socket addresses, or machine identifiers rather than as web addresses.
- Non-resource scheme strings whose purpose is messaging or execution rather than addressing a resource: mailto:, tel:, sms:, javascript:, data:, file:, and similar.
- Namespace IRIs, XML namespace URIs, schema identifiers, or RDF/OWL/SPARQL prefix declarations that resemble URLs but serve as formal identifiers.
- Example-style or placeholder URLs, unless they are visibly printed in the body text of the page.

Before outputting a URL, confirm that it names a real web resource a reader could
visit, rather than an identifier, a namespace, a machine address, or a placeholder.

BROKEN SCHEME HANDLING
A URL may be printed with a typographic space inside the scheme or with a missing colon due to PDF line-breaking or OCR artefacts. This can happen with any scheme — http://, https://, ftp://, sftp://, and others.
Do NOT skip such a URL. Treat it as a valid web address and transcribe it exactly as printed (preserving the space or missing colon). The post-processing pipeline will normalise the scheme. Do not add or remove a scheme.

URL CONTINUATION ACROSS LINES, COLUMNS, AND PAGES
A URL may wrap onto the next printed line, continue into the next column, or continue across a page break.
- If a URL is clearly continued on the next line, join it into one continuous URL string.
- Do not join across a line break if a new paragraph/block begins between the pieces, or if it crosses to another column unless it is clearly the same sentence continuing in normal within-page reading order.
- Preserve hyphens exactly as printed. Remove a hyphen only when it is visibly inserted by line wrapping and the URL continues as the same uninterrupted token on the next line.
- If a URL begins on IMAGE 1 and continues past the bottom of the page, use IMAGE 2+ to finish it. If the IMAGE 1 occurrence is the tail of a URL that began on the previous page, use IMAGE 0 to recover its opening.
- Only cross a page break when the continuation is visually obvious and clearly the same URL token with no intervening paragraph or block boundary. The detected occurrence must still be anchored on IMAGE 1.

TRANSCRIPTION — NO AUTO-CORRECTION, NO HALLUCINATION
- Transcribe every URL and every sentence exactly as printed in the page image. Do not correct, normalise, paraphrase, or improve anything based on what it appears it should say.
- Do not use your knowledge of what a URL typically looks like to fix apparent spelling errors, character swaps, or unusual strings. If the printed URL contains what looks like a misspelling or an unusual character sequence, copy it exactly as printed. Do not swap two adjacent characters because they look transposed.
- The page image is the authoritative source for every URL character. Your background knowledge about website names, domain conventions, or correct spellings must not influence the transcribed URL.
- Never invent a URL. Never output a partial or uncertain URL: if you cannot read the full URL reliably, skip it.
- Never output a URL that is printed in a reference entry, a footnote, a header, a footer, or a data table row, however much the surrounding text reads like body prose.
- If the detected URL cannot be verified inside original_sentence, output no item for it.

WHEN TO USE THE TEXT LAYER FOR URL CHARACTERS
Use the text layer to resolve a URL character only when you are genuinely uncertain what that character is from the image alone — that is, when the printed glyph is visually indistinguishable between two or more possibilities at that font size and rendering. Do not use the text layer to override a character you can already read with confidence from the image.

Visually ambiguous character pairs that may cause genuine uncertainty in PDF fonts include, but are not limited to:
- l (lowercase L) / I (uppercase I) / 1 (digit one)
- O (uppercase O) / 0 (digit zero)
- rn / m
- B / 8
- S / 5
- Z / 2
- cl / d
- vv / w
- D / 0 (in some heavy serif fonts)
- G / 6 (in some fonts)

When you face genuine visual uncertainty over one of these pairs and the text layer contains a character at that position, prefer the text layer character for that specific position. Do not apply this rule to characters you can already read clearly from the image — the text layer is a disambiguation aid for genuinely uncertain glyphs only, not a general override.

Do not use the text layer to invent a URL that is not visibly present in the page image.
Do not use the text layer to pull a URL from a different location if the visual location is not body text.


TARGET-SENTENCE ASSEMBLY (applies the same care as context extraction)
For each kept URL:
- Return original_sentence: the FULL sentence that contains the URL as printed
  in the body text, from its true opening word to its terminal punctuation,
  even if it spans multiple wrapped lines. Join wrapped lines into one
  single-line string.
- The URL must appear inside original_sentence exactly as printed.
- Walk BACKWARD from the URL to the sentence's true start (stop at the
  paragraph start or at terminal punctuation of a prior sentence; do not stop
  at abbreviation periods such as "et al.", "Fig.", "vs.", "cf.", a single
  capital letter, or a number). Apply the FRAGMENT CHECK: a candidate start
  that begins mid-clause with a lower-case continuation word is not a sentence
  start; keep walking backward.
- Walk FORWARD to the sentence's true end; do not stop at a closing
  parenthesis or bracket if the sentence continues. A parenthetical that opens
  before or closes after the URL is part of the SAME sentence and must be
  included.
- If the URL wraps across lines, reconstruct it exactly as printed as one
  continuous string. If a URL begins on IMAGE 1 and continues past the bottom of
  the page, use IMAGE 2+ to finish it; if the IMAGE 1 occurrence is the tail of
  a URL that began on the previous page, use IMAGE 0 to recover its opening.
  Only cross a page break when the continuation is visually obvious and clearly
  the same URL token with no intervening paragraph/block boundary. The detected
  occurrence must still be anchored on IMAGE 1.
- On multi-column pages, a single target sentence may begin near the bottom of
  one column and continue at the top of the next reading-order column on the
  SAME page. When the sentence has no terminal punctuation before the column
  ends and clearly resumes at the top of the next column with matching syntax
  and no paragraph break, assemble it across that within-page column break.
  Do not jump to the next column if the sentence already ends before the break.

PARAGRAPH-LOCAL CONTEXT
For each kept URL, after assembling original_sentence, extract its neighbours within
the SAME paragraph, in correct reading order:
- preceding_sentence: the sentence immediately before original_sentence in the same
  paragraph. If there is none (the target is the paragraph's first sentence), set
  preceding_sentence=null and at_paragraph_start=true.
- trailing_sentence: the sentence immediately after original_sentence in the same
  paragraph. If there is none (the target is the paragraph's last sentence), set
  trailing_sentence=null and at_paragraph_end=true.
- Stop at paragraph and block boundaries: a blank line or larger vertical gap, a
  new-paragraph indentation, a section heading or displayed equation, a list-item
  boundary, or the footnote region. A neighbour that would cross such a boundary
  does not exist; return null in that direction.
- Assemble each neighbour the same way as the target: join wrapped lines into one
  single-line string; do not return only the last wrapped line; do not stop at
  abbreviation periods. Within-page column continuation applies to neighbours too.
  If the target sentence sits at the very top of IMAGE 1 and its preceding sentence
  in the same paragraph is the tail end of the previous page, you may read that
  preceding sentence from IMAGE 0; if the target sits at the very bottom of IMAGE 1
  and its trailing sentence in the same paragraph spills onto the next page, you may
  read that trailing sentence from IMAGE 2+. Use a neighbour crop for context ONLY
  when the paragraph clearly continues across the page break with no intervening
  boundary.
- Transcribe neighbours exactly as printed; do not paraphrase or invent. If a
  neighbour cannot be read with confidence, return null in that direction rather
  than guessing.

OUTPUT
Set has_body_urls. For each kept body URL, return an item with:
- url_printed (the full URL as printed, joined if it wraps)
- original_sentence (the full target sentence containing the URL)
- preceding_sentence / trailing_sentence (string or null)
- at_paragraph_start / at_paragraph_end
- url_lines_joined (true if you joined the URL across line breaks)
- url_span_pages (true only if the URL clearly continues across a page break; mark true only when visually obvious)
Return strict JSON matching the schema.
""".strip()


def make_body_url_single_user_prompt(page_1_based: int, has_prev: bool,
                                     n_next: int, page_text: str) -> str:
    page_text = page_text.strip() if page_text else ""
    if has_prev:
        img0 = ("IMAGE 0 = bottom crop of the PREVIOUS page (page "
                f"{page_1_based - 1}). Use ONLY to complete a URL/sentence that "
                "began there and continues onto the current page. Do NOT detect "
                "URLs on it.\n")
    else:
        img0 = "IMAGE 0 = not provided (this is the first page).\n"
    if n_next > 0:
        img2 = ("IMAGE 2"
                + (f"..{1 + n_next}" if n_next > 1 else "")
                + " = top crop(s) of the FOLLOWING page(s). Use ONLY to finish a "
                "URL/sentence that starts on the current page and spills past its "
                "bottom. Do NOT detect URLs on them.\n")
    else:
        img2 = "IMAGE 2+ = not provided.\n"
    return (
        f"PAGE {page_1_based} — BODY URL DETECT + TARGET SENTENCE + CONTEXT\n"
        + img0
        + "IMAGE 1 = the FULL CURRENT page. Detect every body URL on THIS page "
          "only, and for each return its full target sentence plus paragraph-local "
          "context, per the system instructions.\n"
        + img2
        + "Neighbour crops are completion aids only — never a detection surface.\n"
        "Use the page text layer only to verify exact URL/DOI characters when visually needed.\n"
        "Return JSON that matches the provided schema.\n\n"
        "PRIMARY PAGE TEXT LAYER (verbatim — current page only)\n"
        "-----\n"
        f"{page_text}\n"
        "-----\n"
    )


def render_png(doc: fitz.Document, page0: int, dpi: int, clip: Optional[fitz.Rect] = None) -> bytes:
    page = doc.load_page(page0)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return pix.tobytes("png")


def bottom_clip(doc: fitz.Document, page0: int, start_frac: float) -> fitz.Rect:
    r = doc.load_page(page0).rect
    return fitz.Rect(0, r.height * start_frac, r.width, r.height)


def top_clip(doc: fitz.Document, page0: int, height_frac: float) -> fitz.Rect:
    r = doc.load_page(page0).rect
    return fitz.Rect(0, 0, r.width, r.height * height_frac)


def b64_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def get_page_text(doc: fitz.Document, page0: int) -> str:
    page = doc.load_page(page0)
    return page.get_text("text")


def text_layer_looks_usable(page_text: str,
                            min_chars: int = 20,
                            max_replacement_ratio: float = 0.02,
                            min_alnum_ratio: float = 0.50) -> bool:
    if not page_text:
        return False
    t = page_text.strip()
    if len(t) < min_chars:
        return True
    replacement = t.count("\ufffd")
    if replacement / len(t) > max_replacement_ratio:
        return False
    alnum = sum(1 for c in t if c.isalnum())
    if alnum / len(t) < min_alnum_ratio:
        return False
    return True


def make_blank_png(width_pt: int = 612, height_pt: int = 200, dpi: int = 200) -> bytes:
    """White PNG — used as a stand-in next-page top when on the last page."""
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


_REF_HEADINGS = frozenset({
    "references", "bibliography", "works cited", "literature cited",
    "literature", "sources", "références", "literatur", "bibliografía",
    "bibliographie", "referências", "referencias",
})


def _is_ref_heading_block(block_text: str) -> bool:
    text = block_text.strip()
    if not text:
        return False
    lines = text.splitlines()
    if len(lines) > 3 or len(text) > 60:
        return False  # too long to be a standalone heading
    first_line = lines[0].strip().lower().rstrip(":.").strip()
    return first_line in _REF_HEADINGS


def page_is_references_section(doc: fitz.Document, page0: int) -> bool:
    page = doc.load_page(page0)
    for block in page.get_text("blocks"):
        text = block[4].strip()
        if not text:
            continue
        # First non-empty block
        return _is_ref_heading_block(text)
    return False


def page_references_heading_y(doc: fitz.Document, page0: int) -> Optional[float]:
    """Scan ALL blocks on the page and return the y-coordinate (top of block)
    of the first references heading found, or None.

    Call this only after page_is_references_section() returned False — i.e.
    the page does not start with a references heading but may have one mid-page.
    """
    page = doc.load_page(page0)
    for block in page.get_text("blocks"):
        text = block[4].strip()
        if not text:
            continue
        if _is_ref_heading_block(text):
            return float(block[1])  # y0 of the heading block
    return None



_BROKEN_SCHEME_RE = re.compile(
    r'^(https?|ftps?|s?ftp|sftp|s3|gs|az)\s*:?\s*//'   # any resource-access scheme
    r'(.+)',
    re.IGNORECASE | re.DOTALL,
)


def _normalise_url_scheme(url: str) -> str:

    stripped = url.strip()
    m = _BROKEN_SCHEME_RE.match(stripped)
    if m:
        return f"{m.group(1).lower()}://{m.group(2)}"
    return url


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def url_in_page_text(url: str, page_text: str) -> bool:
    if not url or not page_text:
        return False
    u = _squash(url)
    if len(u) < 8:          
        return False
    return u in _squash(page_text)


def call_body_url_single(client: OpenAI, model: str, page_png: bytes, page_1_based: int,
                         page_text: str,
                         prev_bottom_png: Optional[bytes] = None,
                         next_top_pngs: Optional[List[bytes]] = None,
                         max_retries: int = 3) -> Dict[str, Any]:
    """One call: detect body URLs on the current page and return, for each, the
    full target sentence plus paragraph-local context.

    IMAGE POLICY (single call, multiple images):
      IMAGE 0   = bottom crop of the previous page (prev_bottom_png), optional
      IMAGE 1   = full current page (page_png)
      IMAGE 2.. = top crop(s) of following page(s) (next_top_pngs), optional

    Detection is restricted to IMAGE 1 by the prompt. The neighbour crops exist
    only so the model can complete a URL/sentence (or a paragraph-local
    neighbour) that wraps across a page break. A URL whose printed occurrence is
    on a neighbour crop is never reported as a current-page item. The
    cross-page de-duplication pass in process_pdf is the deterministic safety
    net against a neighbour-crop URL leaking in as a current-page detection."""
    has_prev = prev_bottom_png is not None
    next_top_pngs = next_top_pngs or []
    n_next = len(next_top_pngs)

    user_text = make_body_url_single_user_prompt(
        page_1_based, has_prev=has_prev, n_next=n_next, page_text=page_text)

    # Image order: IMAGE 0 (prev crop) → IMAGE 1 (current full) → IMAGE 2+ (next crops)
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
                    {"role": "system", "content": BODY_URL_SINGLE_SYSTEM},
                    {"role": "user", "content": content},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "body_url_single",
                        "strict": True,
                        "schema": BODY_URL_SINGLE_SCHEMA,
                    }
                },
            )
            data = json.loads(resp.output_text)
            data["page_1_based"] = page_1_based
            items = data.get("items", [])
            data["has_body_urls"] = bool(items)
            return data
        except Exception as e:
            if attempt < max_retries:
                time.sleep(3)
            else:
                print(f"    [!] body-url single failed after {max_retries} attempts: {e}")
                return {"page_1_based": page_1_based, "has_body_urls": False, "items": []}


#
#  PER-PDF PROCESSING
#

def process_pdf(client: OpenAI, pdf_path: Path, out_dir: Path) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    pdf_name = pdf_path.name
    n_pages = doc.page_count

    all_items: List[Dict[str, Any]] = []
    blank_top = make_blank_png(dpi=TOP_DPI)
    in_references_section = False


    for page0 in tqdm(range(n_pages), desc=pdf_name, leave=False):
        page_1 = page0 + 1

        # Reference-section gate
        if in_references_section:
            continue

        if page_is_references_section(doc, page0):
            # The very first content block is a references heading.
            # Skip this page entirely (it's all references) and all later pages.
            in_references_section = True
            print(f"  p{page_1:>4}: references heading at page top — page skipped, "
                  f"subsequent pages will also be skipped")
            continue

        ref_y = page_references_heading_y(doc, page0)
        if ref_y is not None:
            in_references_section = True
            print(f"  p{page_1:>4}: references heading detected mid-page (y≈{ref_y:.0f}px) — "
                  f"processing body content above it; subsequent pages will be skipped")
      

  
        page_png  = render_png(doc, page0, dpi=PAGE_DPI)
        page_text = get_page_text(doc, page0)

        model_text_layer = page_text if text_layer_looks_usable(page_text) else ""

        prev_bottom_png: Optional[bytes] = None
        if page0 > 0:
            pb_clip = bottom_clip(doc, page0 - 1, 1.0 - PREV_PAGE_FRAC)
            prev_bottom_png = render_png(doc, page0 - 1, dpi=TOP_DPI, clip=pb_clip)

        next_top_pngs: List[bytes] = []
        for k in range(1, MAX_NEXT_PAGES + 1):
            if page0 + k >= n_pages:
                next_top_pngs.append(blank_top)
                break
            nt_clip = top_clip(doc, page0 + k, TOP_FRAC)
            next_top_pngs.append(render_png(doc, page0 + k, dpi=TOP_DPI, clip=nt_clip))

        res = call_body_url_single(
            client, SINGLE_MODEL,
            page_png=page_png,
            page_1_based=page_1,
            page_text=model_text_layer,
            prev_bottom_png=prev_bottom_png,
            next_top_pngs=next_top_pngs,
        )

        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

        items = res.get("items", [])
        if not items:
            print(f"  p{page_1:>4}:  0 body URL(s)")
            continue

        for it in items:
            it["pdf_file"] = pdf_name
            it["page"]     = page_1
            # Normalise broken URL schemes
            it["url_printed"] = _normalise_url_scheme(it.get("url_printed", ""))
            #  Page ownership: is this URL actually in THIS page's text layer?
            # Used by the cross-page dedup pass below to drop phantoms the model
            # picked up from a neighbour-page strip.
            it["_in_own_page_text"] = url_in_page_text(it.get("url_printed", ""), page_text)

        print(f"  p{page_1:>4}:  {len(items)} body URL(s) detected + target sentence")
        all_items.extend(items)

        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

    doc.close()

    by_url: Dict[str, List[Dict[str, Any]]] = {}
    for it in all_items:
        by_url.setdefault(_squash(it.get("url_printed", "")), []).append(it)

    kept: List[Dict[str, Any]] = []
    for _key, group in by_url.items():
        pages = {it["page"] for it in group}
        if len(pages) <= 1:
            kept.extend(group)              
            continue
        owners = [it for it in group if it.get("_in_own_page_text")]
        if owners:
            dropped = len(group) - len(owners)
            if dropped:
                pg = sorted({it["page"] for it in group})
                op = sorted({it["page"] for it in owners})
                print(f"  [dedup] {group[0].get('url_printed','')[:60]} "
                      f"emitted on pages {pg} → kept on {op}")
            kept.extend(owners)
        else:
            kept.extend(group)              

    for it in kept:
        it.pop("_in_own_page_text", None)    

    out = {"pdf_file": pdf_name, "model": SINGLE_MODEL, "items": kept}
    out_path = out_dir / f"{Path(pdf_name).stem}_body_urls.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key or api_key == "API-token":
        sys.exit("ERROR: set OPENAI_API_KEY in the CONFIG block or as an environment variable.")

    in_dir  = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)

    pdfs = sorted([p for p in in_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        print(f"No PDFs found in {in_dir}")
        return

    print(f"Found {len(pdfs)} PDF(s). Output → {out_dir}\n")

    n_done = 0

    for pdf_path in pdfs:
        print(f"\n=== Processing {pdf_path.name} ===")
        try:
            process_pdf(client, pdf_path, out_dir)
            n_done += 1
        except Exception as e:
            print(f"[!] Failed on {pdf_path.name}: {e}")

    print(f"\n=== Done: {n_done} PDF(s) processed ===")


if __name__ == "__main__":
    main()