#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


#  CONFIGURATION  (edit these)



#  VENDORED HELPERS  (from the user's parser.py ,  unchanged unless noted)

_CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
_ZERO_WIDTH_RE    = re.compile(r'[\u200B\u200C\u200D\uFEFF]')
_MULTI_SPACE_RE   = re.compile(r'[ \t]{2,}')
_MULTI_NL_RE      = re.compile(r'\n{3,}')
_DEHYPHEN_RE      = re.compile(r'(\w)[\u00AD\-]\n\s*(\w)')


def strip_control_chars(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS_RE.sub("", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    return s


def squash_ws_preserve_newlines(s: str, max_consecutive_blank_lines: int = 1) -> str:
    if not s:
        return ""
    s = strip_control_chars(s)
    lines = s.split("\n")
    out_lines = []
    blank_run = 0
    for line in lines:
        line = line.replace("\t", " ")
        indent_len = len(line) - len(line.lstrip(" "))
        indent = " " * indent_len
        rest = line.lstrip(" ")
        rest = _MULTI_SPACE_RE.sub(" ", rest).rstrip()
        cleaned = indent + rest
        if cleaned.strip() == "":
            blank_run += 1
            if blank_run <= max_consecutive_blank_lines:
                out_lines.append("")
        else:
            blank_run = 0
            out_lines.append(cleaned)
    s2 = "\n".join(out_lines)
    s2 = _MULTI_NL_RE.sub("\n\n", s2)
    return s2.strip("\n")


def get_page_height(page_data: Dict[str, Any]) -> float:
    return page_data.get('height', 792.0)


def get_main_font_size(page_data: Dict[str, Any]) -> float:
    sizes = []
    for box in (page_data.get('boxes') or []):
        if box.get('boxclass') == 'text':
            for line in (box.get('textlines') or []):
                for span in (line.get('spans') or []):
                    sizes.append(round(span.get('size', 0), 2))
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def clean_marker(text: str) -> str:
    if not text:
        return ""
    return text.strip().strip('[]().')


_MARKER_KEY_RE = re.compile(
    r'^(?:\[)?(\d+|[*†‡§¶‖\u2020\u2021\u002A\u2217\u22C6\u2605\u2606]+)(?:\]|\.)?'
)


def extract_marker_key(text: str) -> Optional[str]:
    text = (text or "").strip()
    m = _MARKER_KEY_RE.match(text)
    if m:
        return m.group(1)
    return None


def is_footnote_box(box: Dict[str, Any], main_font_size: float, page_height: float) -> bool:
    box_class = box.get('boxclass')
    if box_class == 'footnote':
        return True
    if box_class in ('list-item', 'text'):
        y0 = box.get('y0', 0)
        if y0 < (page_height * 0.5):
            return False
        textlines = box.get('textlines') or []
        if not textlines:
            return False
        spans = textlines[0].get('spans') or []
        if not spans:
            return False
        first_span = spans[0]
        span_size = round(first_span.get('size', 0), 2)
        if span_size < main_font_size:
            text = strip_control_chars(first_span.get('text', '')).strip()
            all_text = strip_control_chars("".join(
                [s.get('text', '') for l in textlines for s in l.get('spans', [])]
            )).strip()
            if all_text.isdigit():
                return False
            if extract_marker_key(text):
                return True
    return False


def _span_sort_key(span: Dict[str, Any]):
    b = span.get("bbox") or [0, 0, 0, 0]
    return (b[0], b[2])


def _should_insert_space(prev_text: str, curr_text: str, gap: float) -> bool:
    if not prev_text or not curr_text:
        return False
    if curr_text[0] in ",.;:!?)]}%":
        return False
    if curr_text[0] in "'\u2019":
        return False
    if prev_text[-1] in "([{":
        return False
    if prev_text[-1] in "-\u00AD\u2010\u2011\u2012\u2013\u2212":
        return False
    return gap >= 0.6


#  TIGHTENED MARKER PARSER  (modified from parse_span_text)
#
# Differences from the original parser.py:
#   1. Digit markers are accepted only when small-font OR Unicode-superscript.
#      Brackets ALONE are no longer a sufficient signal: in many papers the
#      typesetter renders footnote citations as [N] in small font AND also
#      uses plain [N] at body font for bibliographic citations to the
#      references list ,  the only reliable discriminator is font size.
#      So:  [3] @ 6.97pt (small)  -> marker  (footnote superscript)
#           [3] @ 9.96pt (body)   -> text    (bib ref or plain number)
#            3  @ 6.97pt (small)  -> marker  (bare superscript)
#            3  @ 9.96pt (body)   -> text    (plain number)
#   2. Unicode superscript digits (¹²³⁴⁵⁶⁷⁸⁹⁰) are translated to ASCII first,
#      and the span is treated as effectively small-font for marker matching.
#      This catches typesetters who used literal superscript characters.
#   3. Symbol markers (*, †, ‡, §, ¶, ‖, ⋆, etc.) are font-independent: brackets
#      and small-font are both fine, body font is also fine. Symbols are
#      unambiguous because they aren't used for bibliographic citations.

_SYMBOL_MARKER_RE = re.compile(r'^[*†‡§¶‖\u2020\u2021\u002A\u2217\u22C6\u2605\u2606]+$')

_PARSE_SPLIT_RE = re.compile(
    r'(\[[^\]]+\]|\d+|[*†‡§¶‖\u2020\u2021\u002A\u2217\u22C6\u2605\u2606]+)'
)

_MULTI_INNER_RE = re.compile(r'[,\s]+')

# Map U+2070-2079, U+00B9, U+00B2, U+00B3 -> ASCII digits.
SUPERSCRIPT_DIGIT_TRANS = str.maketrans({
    '\u2070': '0', '\u00B9': '1', '\u00B2': '2', '\u00B3': '3',
    '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
    '\u2078': '8', '\u2079': '9',
})


def _is_symbol_marker(s: str) -> bool:
    return bool(_SYMBOL_MARKER_RE.match(s))


def parse_span_text_tightened(text: str, is_small_font: bool, registry: Dict[str, Any]):
    if not text:
        return []

    # 1. Translate Unicode superscript digits -> ASCII; remember whether we did.
    translated = text.translate(SUPERSCRIPT_DIGIT_TRANS)
    has_unicode_sup = (translated != text)
    text = translated
    effective_small = is_small_font or has_unicode_sup

    tokens = []
    parts = _PARSE_SPLIT_RE.split(text)

    for p in parts:
        if not p:
            continue
        clean_p = clean_marker(p)
        is_bracketed = p.startswith('[') and p.endswith(']')

        if clean_p in registry:
            symbol_marker = _is_symbol_marker(clean_p)
            if symbol_marker:
                # Symbols (*, †, ‡, §, ⋆, etc.) are unambiguous -> always a marker.
                tokens.append(('marker', clean_p))
            else:
                # Digit marker. Discrimination is by FONT SIZE, not by brackets:
                #   small-font digit (with or without brackets) -> footnote superscript
                #   body-font digit (with or without brackets) -> bib ref / plain number -> drop
                if effective_small:
                    tokens.append(('marker', clean_p))
                else:
                    tokens.append(('text', p))
        elif is_bracketed:
            # Multi-marker bracket like [3, 4] or [*†].
            # Same font-size rule: emit only if span is small-font OR all sub-markers
            # are symbols (which are font-independent).
            inner = p[1:-1]
            sub_parts = [sp.strip() for sp in _MULTI_INNER_RE.split(inner) if sp.strip()]
            valid_sub_markers = []
            all_valid = bool(sub_parts)
            all_symbols = True
            for sp in sub_parts:
                clean_sp = clean_marker(sp)
                if clean_sp in registry:
                    valid_sub_markers.append(clean_sp)
                    if not _is_symbol_marker(clean_sp):
                        all_symbols = False
                else:
                    all_valid = False
                    break
            if all_valid and valid_sub_markers and (all_symbols or effective_small):
                for vm in valid_sub_markers:
                    tokens.append(('marker', vm))
            else:
                tokens.append(('text', p))
        else:
            tokens.append(('text', p))

    return tokens


def build_line_with_cites(spans: List[Dict[str, Any]],
                          main_font_size: float,
                          registry: Dict[str, Any]) -> Tuple[str, float]:
    spans_sorted = sorted(spans or [], key=_span_sort_key)
    line_out = ""
    prev_span = None
    prev_emitted_tail = ""
    line_sizes = []

    for span in spans_sorted:
        raw_text = strip_control_chars(span.get("text", ""))
        if not raw_text:
            prev_span = span
            continue

        size = round(span.get("size", 0), 2)
        if size > 0:
            line_sizes.append(size)
        is_small_font = size < (main_font_size - 0.5)

        if prev_span is not None:
            b1 = prev_span.get("bbox") or [0, 0, 0, 0]
            b2 = span.get("bbox") or [0, 0, 0, 0]
            gap = (b2[0] - b1[2])
            if gap > 0 and _should_insert_space(prev_emitted_tail, raw_text, gap):
                if line_out and not line_out.endswith((" ", "\n")):
                    line_out += " "
                    prev_emitted_tail = " "

        for kind, content in parse_span_text_tightened(raw_text, is_small_font, registry):
            if kind == "marker":
                citation_tag = f"[CITE:{content}]"
                line_out += citation_tag
                prev_emitted_tail = citation_tag[-1]
            else:
                line_out += content
                if content:
                    prev_emitted_tail = content[-1]

        prev_span = span

    primary_size = (Counter(line_sizes).most_common(1)[0][0]
                    if line_sizes else main_font_size)
    return line_out.rstrip(), primary_size


def extract_footnote_definitions(page_data: Dict[str, Any],
                                 main_font_size: float):
    registry: Dict[str, bool] = {}
    footnotes_ordered: List[Dict[str, str]] = []
    box_indices: set = set()

    page_height = get_page_height(page_data)

    for i, box in enumerate(page_data.get('boxes') or []):
        if not is_footnote_box(box, main_font_size, page_height):
            continue
        box_indices.add(i)

        lines_content = []
        marker_clean = None

        for line_idx, line in enumerate(box.get('textlines') or []):
            spans = line.get('spans') or []
            if not spans:
                continue

            if line_idx == 0:
                first_text = strip_control_chars(spans[0].get('text', '')).strip()
                extracted_key = extract_marker_key(first_text)
                if extracted_key:
                    marker_clean = extracted_key
                    registry[marker_clean] = True

            line_text, _ = build_line_with_cites(spans, main_font_size, registry)
            if line_text:
                lines_content.append(line_text)

        full_content = squash_ws_preserve_newlines(
            " ".join(lines_content)).replace("\n", " ").strip()

        if marker_clean:
            prefix = f"[CITE:{marker_clean}]"
            tmp = full_content.lstrip()
            if tmp.startswith(prefix):
                full_content = tmp[len(prefix):].lstrip()

            footnotes_ordered.append({
                'marker':  marker_clean,
                'content': full_content,
            })

    return registry, footnotes_ordered, box_indices


#  REFERENCES-SECTION CUTOFF

_REF_HEADINGS = frozenset({
    "references", "bibliography", "works cited", "literature cited",
    "literature", "sources", "références", "literatur", "bibliografía",
    "bibliografia", "bibliographie", "referências", "referencias",
})

_HEADING_NUMBER_PREFIX_RE = re.compile(r'^\d+(?:\.\d+)*\s*\.?\s*')


def _normalize_heading(text: str) -> str:
    norm = (text or "").strip().lower().rstrip(':').rstrip('.').strip()
    norm = _HEADING_NUMBER_PREFIX_RE.sub('', norm).strip()
    return norm


def find_references_cutoff(page_paths: List[Path]) -> Tuple[Optional[int], Optional[float]]:
    for page_path in page_paths:
        try:
            data = json.loads(page_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        pages = data.get('pages') or []
        for page in pages:
            page_num = page.get('page_number')
            for box in (page.get('boxes') or []):
                if box.get('boxclass') != 'section-header':
                    continue
                text = ' '.join(
                    sp.get('text', '')
                    for ln in (box.get('textlines') or [])
                    for sp in (ln.get('spans') or [])
                ).strip()
                if _normalize_heading(text) in _REF_HEADINGS:
                    return page_num, float(box.get('y0', 0))
    return None, None


#  URL DETECTION
#
# Mirrors the URL definition in the VLM detect prompt:
#   - http://, https://, ftp://
#   - www. (scheme-less)
#   - doi:NN.NNNN/...
#   - arxiv:NNNN.NNNNN
#   - github.com/..., gitlab.com/..., bitbucket.org/...
#   - Excludes mailto:

_URL_TERM = r'[^\s<>\[\]\(\)\{\}"\'`,;]'

URL_PATTERNS = [
    rf'https?://{_URL_TERM}+',
    rf'ftp://{_URL_TERM}+',
    rf'(?<![A-Za-z0-9])www\.{_URL_TERM}+',
    rf'(?<![A-Za-z])doi:\s?10\.\d+/{_URL_TERM}+',
    r'(?<![A-Za-z])arxiv:\s?\d{4}\.\d{4,5}(?:v\d+)?',
    rf'(?<![A-Za-z/.])github\.com/{_URL_TERM}+',
    rf'(?<![A-Za-z/.])gitlab\.com/{_URL_TERM}+',
    rf'(?<![A-Za-z/.])bitbucket\.org/{_URL_TERM}+',
]
URL_RE = re.compile('|'.join(f'(?:{p})' for p in URL_PATTERNS), re.IGNORECASE)


def clean_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    for opener, closer in [('(', ')'), ('[', ']'), ('<', '>'), ('"', '"'), ("'", "'")]:
        if url.startswith(opener) and url.endswith(closer):
            url = url[1:-1]
            break
    if url.startswith('(') and ')' not in url:
        url = url[1:]
    url = url.rstrip('.,;:')
    if url.endswith(')') and '(' not in url:
        url = url.rstrip(')')
    return url.strip()


def extract_urls(content: str) -> List[str]:
    if not content:
        return []
    raw_matches = URL_RE.findall(content)
    out: List[str] = []
    seen: set = set()
    for u in raw_matches:
        u = clean_url(u)
        if not u or u.lower().startswith('mailto:'):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


#  BIBLIOGRAPHIC-REFERENCE EXCLUSION

# Author-pattern: e.g. "F. Lastname", "F.M. Lastname", "F.-M. Lastname",
# "Lastname, F.", "Lastname F.M.".
_AUTHOR_INITIAL_RE = re.compile(
    r'\b(?:[A-Z]\.\s?){1,3}[A-Z][a-zA-Z\-]+\b'
    r'|'
    r'\b[A-Z][a-zA-Z\-]+,\s?(?:[A-Z]\.\s?){1,3}'
)

_YEAR_RE = re.compile(r'\b(?:19|20)\d{2}\b')

_VENUE_RE = re.compile(
    r'\b(?:'
    r'Proc\.|Proceedings|Conference|Workshop|Symposium|Journal|Trans\.'
    r'|IEEE|ACM|Notices|Vol\.|Volume|pages?|pp\.|editors?|eds?\.'
    r')\b',
    re.IGNORECASE,
)


def looks_like_bib_ref(content: str) -> bool:
    if not content:
        return False
    has_year   = bool(_YEAR_RE.search(content))
    has_author = bool(_AUTHOR_INITIAL_RE.search(content))
    has_venue  = bool(_VENUE_RE.search(content))
    return has_year and (has_author or has_venue)


#  STAGE 1 ,  DETECT URL FOOTNOTES PER PAGE

def detect_url_footnotes(page_data: Dict[str, Any], main_font_size: float):
    registry, all_footnotes, fn_box_indices = extract_footnote_definitions(
        page_data, main_font_size,
    )
    url_footnotes = []
    for fn in all_footnotes:
        content = fn.get('content', '')
        if looks_like_bib_ref(content):
            continue
        urls = extract_urls(content)
        if not urls:
            continue
        url_footnotes.append({
            'marker':  fn['marker'],
            'content': content,
            'urls':    urls,
        })
    return registry, url_footnotes, fn_box_indices


#  STAGE 2 ,  STRUCTURED PAGE PROCESSING

def process_page_structured(page_data: Dict[str, Any],
                            cutoff_y0: Optional[float]):
    main_font_size = get_main_font_size(page_data)
    registry, url_footnotes, fn_box_indices = detect_url_footnotes(
        page_data, main_font_size
    )

    boxes = list(page_data.get('boxes') or [])
    body_lines = []

    for i, box in enumerate(boxes):
        if i in fn_box_indices:
            continue
        if box.get('boxclass') == 'page-header':
            continue
        if cutoff_y0 is not None and box.get('y0', 0) >= cutoff_y0:
            continue

        for line_idx, line in enumerate(box.get('textlines') or []):
            spans = line.get('spans') or []
            if not spans:
                continue
            line_text, primary_size = build_line_with_cites(spans, main_font_size, registry)
            if not line_text.strip():
                continue
            # bbox of the textline = union of span bboxes
            xs0, ys0, xs1, ys1 = [], [], [], []
            for sp in spans:
                bb = sp.get('bbox') or [0, 0, 0, 0]
                xs0.append(bb[0]); ys0.append(bb[1])
                xs1.append(bb[2]); ys1.append(bb[3])
            line_bbox = (min(xs0), min(ys0), max(xs1), max(ys1)) if xs0 else (0, 0, 0, 0)

            body_lines.append({
                'text':       line_text,
                'bbox':       line_bbox,
                'box_class':  box.get('boxclass'),
                'box_index':  i,
                'line_index': line_idx,
                'font_size':  primary_size,
            })

    return {
        'main_font_size': main_font_size,
        'url_footnotes':  url_footnotes,
        'body_lines':     body_lines,
        'all_registry':   registry,
    }


#  STAGE 3 ,  CITING SENTENCE + RESTORATION  (CASE A only)

# Sentence boundary inside a paragraph: a period/?/! followed by whitespace and
# either a capital letter, an opening quote+capital, or a digit (rare, but e.g.
# "...test set. 23 of these..."). We DELIBERATELY do NOT split on
# common abbreviations (Fig., Eq., e.g., i.e., et al., Dr., Mr., Mrs., Ms., Jr.,
# Sr., vs., etc., No., Nos., pp., vol.).

_ABBREV_TAIL_RE = re.compile(
    r'(?:'
    r'\b(?:Fig|Eq|Eqs|Sec|Ch|App|Ref|Refs|No|Nos|pp|p|vol|Vol|et\s?al|i\.e|e\.g|etc|cf'
    r'|Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|vs)\.'
    r'|'
    r'\b[A-Z]\.'                # single-letter initial like "J."
    r')\s*$'
)

_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])'
    r'(?=\s+["\u201C]?[A-Z])'
)


def _is_sentence_boundary(prefix: str) -> bool:
    if _ABBREV_TAIL_RE.search(prefix):
        return False
    return True


def _split_paragraph_into_sentences(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    # Find candidate split positions, then filter by abbreviation guard.
    sentences: List[str] = []
    cursor = 0
    for m in _SENTENCE_SPLIT_RE.finditer(paragraph):
        end = m.start()  # split between cursor..end, then continue from end+leading_space
        prefix = paragraph[cursor:end]
        if not _is_sentence_boundary(prefix):
            continue
        sentence = prefix.strip()
        if sentence:
            sentences.append(sentence)
        # advance cursor past the whitespace after punctuation
        # m.start() is just after the punctuation; whitespace begins at m.start().
        ws_match = re.match(r'\s+', paragraph[end:])
        cursor = end + (len(ws_match.group(0)) if ws_match else 0)
    # tail
    tail = paragraph[cursor:].strip()
    if tail:
        sentences.append(tail)

    return sentences


def _join_lines_into_paragraph(box_lines: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for k, ln in enumerate(box_lines):
        t = ln['text']
        if k > 0:
            prev = parts[-1] if parts else ''
            if prev.endswith('-') and t and t[0].islower() and len(prev) >= 2 and prev[-2].isalpha():
                # de-hyphenate: strip the trailing hyphen of prev, append t directly
                parts[-1] = prev[:-1] + t
                continue
        parts.append(t)
    return ' '.join(parts).strip()


def build_paragraphs(body_lines: List[Dict[str, Any]]
                     ) -> List[Tuple[int, str, str, List[str]]]:
    paragraphs: List[Tuple[int, str, str, List[str]]] = []
    if not body_lines:
        return paragraphs

    cur_box = body_lines[0]['box_index']
    cur_class = body_lines[0]['box_class']
    cur_lines = [body_lines[0]]

    for ln in body_lines[1:]:
        if ln['box_index'] != cur_box:
            paragraph = _join_lines_into_paragraph(cur_lines)
            sentences = _split_paragraph_into_sentences(paragraph)
            paragraphs.append((cur_box, cur_class, paragraph, sentences))
            cur_box = ln['box_index']
            cur_class = ln['box_class']
            cur_lines = [ln]
        else:
            cur_lines.append(ln)

    # last paragraph
    paragraph = _join_lines_into_paragraph(cur_lines)
    sentences = _split_paragraph_into_sentences(paragraph)
    paragraphs.append((cur_box, cur_class, paragraph, sentences))

    return paragraphs


#  STAGE 3+4 ,  EXTRACT ITEMS PER PAGE

_CITE_TAG_RE = re.compile(r'\[CITE:([^\]]+)\]')
_WS_RE = re.compile(r'\s+')


def _strip_cite_tags(text: str) -> str:
    out = _CITE_TAG_RE.sub('', text)
    return _WS_RE.sub(' ', out).strip()


def _replace_cite_with_content(sentence: str, marker: str, footnote_text: str) -> str:
    needle = f'[CITE:{marker}]'
    replacement = f' [{footnote_text}]'
    out = sentence.replace(needle, replacement)
    # Collapse leading-space-then-bracket if the marker was at the start.
    out = re.sub(r'\s{2,}', ' ', out).strip()
    return out


def extract_items_for_page(page_data: Dict[str, Any],
                           cutoff_y0: Optional[float],
                           pdf_file: str,
                           page_num: int) -> List[Dict[str, Any]]:
    proc = process_page_structured(page_data, cutoff_y0)
    url_footnotes = proc['url_footnotes']
    body_lines = proc['body_lines']

    if not url_footnotes:
        return []

    # Index footnotes by marker for quick lookup.
    fn_by_marker = {fn['marker']: fn for fn in url_footnotes}
    url_markers = set(fn_by_marker.keys())

    paragraphs = build_paragraphs(body_lines)
    items: List[Dict[str, Any]] = []

    for box_index, box_class, paragraph, sentences in paragraphs:
        if not sentences:
            continue
        for s_idx, sentence in enumerate(sentences):
            cite_markers_here = _CITE_TAG_RE.findall(sentence)
            if not cite_markers_here:
                continue
            url_markers_in_sentence = [m for m in cite_markers_here if m in url_markers]
            if not url_markers_in_sentence:
                continue

            # Determine preceding / trailing within same paragraph (box).
            prev_s = sentences[s_idx - 1] if s_idx > 0 else None
            next_s = sentences[s_idx + 1] if s_idx + 1 < len(sentences) else None
            preceding = _strip_cite_tags(prev_s) if prev_s else None
            trailing  = _strip_cite_tags(next_s) if next_s else None

            # SCENARIO 1: multiple URL markers in same sentence -> one item per marker.
            for marker in url_markers_in_sentence:
                fn = fn_by_marker[marker]
                footnote_text = fn['content']
                urls = list(fn['urls'])

                original = _strip_cite_tags(sentence)
                restored = _replace_cite_with_content(sentence, marker, footnote_text)
                # Strip any remaining [CITE:m'] tokens for OTHER markers in restored.
                # (We keep them as visible "superscripts" in the VLM convention,
                # but the [CITE:m'] form is our placeholder, not the printed form.
                # Drop them ,  the original_sentence already has them stripped.)
                restored = _CITE_TAG_RE.sub('', restored)
                restored = _WS_RE.sub(' ', restored).strip()

                items.append({
                    'original_sentence':  original,
                    'restored_sentence':  restored,
                    'footnote_marker':    marker,
                    'footnote_content':   footnote_text,
                    'url':                urls,
                    'preceding_sentence': preceding if (preceding and preceding.strip()) else None,
                    'trailing_sentence':  trailing  if (trailing  and trailing.strip())  else None,
                    'pdf_file':           pdf_file,
                    'page':               page_num,
                })

    return items


#  PDF-LEVEL ORCHESTRATION

_PAGE_NUM_RE = re.compile(r'^(.+?)_(\d+)$')


def gather_pdf_pages(input_dir: Path) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}

    subdirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if subdirs:
        for sub in sorted(subdirs):
            pages = sorted(sub.glob('*.json'))
            if not pages:
                continue
            # Sort by page number, falling back to filename order.
            def page_num_of(p: Path) -> int:
                m = _PAGE_NUM_RE.match(p.stem)
                return int(m.group(2)) if m else 0
            pages.sort(key=page_num_of)
            result[sub.name] = pages
        if result:
            return result

    # Flat layout fallback.
    flat_jsons = sorted(input_dir.glob('*.json'))
    grouping: Dict[str, List[Tuple[int, Path]]] = {}
    for j in flat_jsons:
        m = _PAGE_NUM_RE.match(j.stem)
        if m:
            pdf_id = m.group(1)
            grouping.setdefault(pdf_id, []).append((int(m.group(2)), j))
    for pid, lst in grouping.items():
        lst.sort(key=lambda t: t[0])
        result[pid] = [p for _, p in lst]

    return result


def process_pdf(pdf_id: str, page_paths: List[Path]) -> List[Dict[str, Any]]:
    cutoff_page, cutoff_y0 = find_references_cutoff(page_paths)
    items: List[Dict[str, Any]] = []

    for page_path in page_paths:
        try:
            data = json.loads(page_path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  [!] {page_path.name}: cannot parse JSON: {e}")
            continue

        for page in (data.get('pages') or []):
            page_num = page.get('page_number', 0)

            # Apply references cutoff:
            #   - on cutoff_page, only process boxes with y0 < cutoff_y0
            #   - on later pages, skip entirely.
            if cutoff_page is not None:
                if page_num > cutoff_page:
                    continue
                page_cutoff_y0 = cutoff_y0 if page_num == cutoff_page else None
            else:
                page_cutoff_y0 = None

            page_items = extract_items_for_page(
                page_data=page,
                cutoff_y0=page_cutoff_y0,
                pdf_file=pdf_id,
                page_num=page_num,
            )
            items.extend(page_items)

    # Global dedup by normalized restored_sentence.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for it in items:
        key = ' '.join(it['restored_sentence'].strip().lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    return deduped


def write_pdf_outputs(out_dir: Path, pdf_id: str, items: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / f"{pdf_id}_footnotes.json"
    sentences_path = out_dir / f"{pdf_id}_sentences.json"

    items_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    sentences_path.write_text(
        json.dumps([it['restored_sentence'] for it in items], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return items_path, sentences_path


def run(input_dir: str, output_dir: str) -> None:
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    if not in_dir.is_dir():
        sys.exit(f"ERROR: input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_to_pages = gather_pdf_pages(in_dir)
    if not pdf_to_pages:
        sys.exit(f"No JSONs found under: {in_dir}")

    print(f"Input          : {in_dir}")
    print(f"Output         : {out_dir}")
    print(f"PDFs found     : {len(pdf_to_pages)}")
    print()

    summaries: List[Dict[str, Any]] = []
    all_jsonl = out_dir / "_ALL.url_footnotes.jsonl"

    with all_jsonl.open('w', encoding='utf-8') as jl:
        for pdf_id in sorted(pdf_to_pages):
            pages = pdf_to_pages[pdf_id]
            print(f" {pdf_id}  ({len(pages)} pages)")
            try:
                items = process_pdf(pdf_id, pages)
            except Exception as e:
                import traceback; traceback.print_exc()
                summaries.append({'pdf_file': pdf_id, 'count': 0, 'error': str(e)})
                continue

            items_path, _ = write_pdf_outputs(out_dir, pdf_id, items)
            for it in items:
                jl.write(json.dumps(it, ensure_ascii=False) + '\n')

            print(f"  -> {len(items)} item(s) -> {items_path.name}")
            summaries.append({'pdf_file': pdf_id, 'count': len(items),
                              'items_json': str(items_path)})

    (out_dir / '_SUMMARY.json').write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    total = sum(s.get('count', 0) for s in summaries)
    print()
    print('=' * 55)
    print(f" Done.")
    print(f"   PDFs processed       : {len(pdf_to_pages)}")
    print(f"   Total items          : {total}")
    print(f"   Per-PDF JSONs in     : {out_dir}")
    print(f"   Combined JSONL       : {all_jsonl}")
    print(f"   Summary              : {out_dir / '_SUMMARY.json'}")
    print('=' * 55)


#  ENTRY POINT

def main():
    ap = argparse.ArgumentParser(description="Extract footnote URLs from PyMuPDF-Layout per-page JSON.")
    ap.add_argument("-i", "--input", required=True, help="directory of per-page JSON (per-PDF subfolders)")
    ap.add_argument("-o", "--output", required=True, help="output directory for the JSON results")
    args = ap.parse_args()
    run(args.input, args.output)


if __name__ == '__main__':
    main()
