#!/usr/bin/env python3
# Body URL context extractor for GROBID TEI.
# Reads TEI XML produced by GROBID and writes, for each URL, the URL together
# with its target sentence and the neighbouring sentences in the same
# paragraph. Running text, the abstract, and figure/table captions are all
# covered and reported under location "body". Footnotes, references, table
# bodies and front/back matter are skipped.

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lxml import etree

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}

BODY_URL_CONTEXT_FIELDS = [
    "url_printed",
    "target_sentence",
    "preceding_sentence",
    "trailing_sentence",
    "at_paragraph_start",
    "at_paragraph_end",
    "url_lines_joined",
    "url_span_pages",
    "pdf_file",
    "page",
    "location",
]


#  URL / DOI PATTERNS

EXPLICIT_URL_RE = re.compile(
    r"(?i)\b(?:https?://|ftp://|www\.)[^\s<>()\[\]{}\"'`]+"
)

DOI_RE = re.compile(
    r"(?i)(?:\bdoi\s*:\s*)?\b10\.\d{4,9}/[^\s<>()\[\]{}\"'`]+"
)

BROKEN_SCHEME_RE = re.compile(
    r"^(https?|ftps?|s?ftp|sftp|s3|gs|az)\s*:?\s*//(.+)",
    re.IGNORECASE | re.DOTALL,
)

URL_START_RE = re.compile(r"(?i)(https?://|ftp://|www\.)")
URL_CONTINUATION_RE = re.compile(
    r"(?i)^(com|org|net|edu|gov|io|co|uk|de|fr|jp|cn|au|ca|nl|info|biz|name|int|mil|us|eu|ch|se|no|fi|es|it|ru|br|pl|in|ac|ai|dev|app)\b"
)

COMMON_ABBREVIATIONS = {
    "e.g.", "i.e.", "cf.", "cfr.", "fig.", "figs.", "eq.", "eqs.",
    "ref.", "refs.", "sec.", "sect.", "ch.", "app.", "vol.", "no.",
    "pp.", "p.", "ed.", "eds.", "etc.", "vs.", "approx.", "resp.",
    "dr.", "prof.", "mr.", "mrs.", "ms.", "jr.", "sr.", "inc.", "ltd.",
    "co.", "u.s.", "u.k.", "ph.d.", "m.d.",
}


#  BASIC TEI / TEXT HELPERS

def local_name(elem: etree._Element) -> str:
    try:
        return etree.QName(elem.tag).localname
    except Exception:
        return ""


def elem_text(elem: etree._Element) -> str:
    return "".join(elem.itertext())


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_url_for_compare(url: str) -> str:
    u = collapse_ws(url)
    u = u.strip("<>[]{}()\"'")
    u = u.rstrip(".,;:)>]}\"'")
    u = re.sub(r"\s+", "", u)
    return u.lower()


def _normalise_url_scheme(url: str) -> str:
    stripped = (url or "").strip()
    m = BROKEN_SCHEME_RE.match(stripped)
    if m:
        return f"{m.group(1).lower()}://{m.group(2)}"
    return url


def clean_url_printed(url: str) -> str:
    u = collapse_ws(url)
    if not u:
        return ""
    u = u.strip("<>[]{}\"'")
    u = u.rstrip(".,;:")
    if u.endswith(")") and "(" not in u:
        u = u[:-1].rstrip()
    return _normalise_url_scheme(u)


def serialize_sentence(s_elem: etree._Element) -> str:
    parts: List[str] = []
    if s_elem.text:
        parts.append(s_elem.text)
    for child in s_elem:
        if local_name(child) == "ref" and child.get("type") == "url":
            parts.append(child.get("target") or elem_text(child))
        else:
            parts.append(elem_text(child))
        if child.tail:
            parts.append(child.tail)
    return collapse_ws("".join(parts))


def nearest_ancestor(elem: etree._Element, tag: str) -> Optional[etree._Element]:
    cur = elem.getparent()
    while cur is not None:
        if local_name(cur) == tag:
            return cur
        cur = cur.getparent()
    return None


def is_inside_footnote(elem: etree._Element) -> bool:
    cur = elem.getparent()
    while cur is not None:
        if local_name(cur) == "note" and (cur.get("place") or "").lower() == "foot":
            return True
        cur = cur.getparent()
    return False


#  BODY/PROSE FILTERS

SKIP_ANCESTOR_TAGS = {
    "figure", "figDesc", "table", "listBibl", "biblStruct", "back", "front"
}

PUBLISHER_FOOTER_PATTERNS = [
    r"\bPLOS\s+(ONE|Biology|Medicine|Genetics|Pathogens|Computational Biology)\b.*\bdoi\.org\b",
    r"\bScientific Data\b.*\bdoi\.org\b",
    r"\bFrontiers in\b.*\bwww\.frontiersin\.org\b",
    r"\bPeerJ\b.*\bdoi\s*:\s*10\.",
    r"\bBMC\b.*\bdoi\.org\b",
    r"\bCreative Commons\b",
    r"\bopen access\b.*\blicen[cs]e\b",
    r"\bCopyright\b",
    r"^\s*©",
    r"^[A-Za-z][A-Za-z .&-]{2,}\s*\|.*\b(?:doi\.org|10\.\d{4,9}/)",
]

MACHINE_TEXT_PATTERNS = [
    r"\bxmlns\b",
    r"\bnamespace\b",
    r"\bprefix\b.*[:=]",
    r"^\s*(curl|wget|git\s+clone|sudo|python\s+|Rscript\s+|java\s+)\b",
    r"[{}<>]{3,}",
    r"\bSELECT\b.+\bFROM\b",
]


def is_eligible_p(p_elem: etree._Element) -> bool:
    if nearest_ancestor(p_elem, "body") is None:
        return False
    if is_inside_footnote(p_elem):
        return False
    cur = p_elem.getparent()
    while cur is not None:
        if local_name(cur) in SKIP_ANCESTOR_TAGS:
            return False
        cur = cur.getparent()
    return True


def is_eligible_sentence(s_elem: etree._Element) -> bool:
    if nearest_ancestor(s_elem, "body") is None:
        return False
    if nearest_ancestor(s_elem, "p") is None:
        return False
    if is_inside_footnote(s_elem):
        return False
    cur = s_elem.getparent()
    while cur is not None:
        if local_name(cur) in SKIP_ANCESTOR_TAGS:
            return False
        cur = cur.getparent()
    return True


def looks_like_publisher_footer(text: str) -> bool:
    t = collapse_ws(text)
    if not t:
        return True
    for pat in PUBLISHER_FOOTER_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    if t.count("|") >= 2 and re.search(r"(?i)(doi\.org|10\.\d{4,9}/|www\.)", t):
        return True
    return False


def looks_like_machine_text(text: str) -> bool:
    t = collapse_ws(text)
    if not t:
        return True
    for pat in MACHINE_TEXT_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    alpha_words = re.findall(r"[A-Za-z]{3,}", t)
    if len(alpha_words) <= 2 and re.search(r"[=;{}<>]", t):
        return True
    return False


def should_keep_body_url_sentence(text: str) -> bool:
    return not looks_like_publisher_footer(text) and not looks_like_machine_text(text)


#  LOGICAL SENTENCE RECONSTRUCTION WITHIN <p>

def paren_balance(text: str) -> int:
    return text.count("(") - text.count(")")


def ends_with_abbreviation(text: str) -> bool:
    words = re.findall(r"[A-Za-z.]+", text.lower())
    if not words:
        return False
    tail = words[-1]
    if tail in COMMON_ABBREVIATIONS:
        return True
    # Initials: "J." or "A. B."
    if re.search(r"(?:\b[A-Z]\.){1,4}\s*$", text):
        return True
    return False


def terminal_sentence_end(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return False
    stripped = t
    while stripped and stripped[-1] in "\"'”’)]}":
        stripped = stripped[:-1].rstrip()
    if not stripped:
        return False
    if stripped[-1] not in ".?!":
        return False
    if ends_with_abbreviation(stripped):
        return False
    return True


def url_split_join_needed(current: str, nxt: str) -> bool:
    c = current.rstrip()
    n = nxt.lstrip()
    if not c or not n:
        return False
    last_start = None
    for m in URL_START_RE.finditer(c):
        last_start = m.start()
    if last_start is None:
        return False
    tail = c[last_start:]
    if len(tail) > 120:
        return False
    if c.endswith(".") and URL_CONTINUATION_RE.match(n):
        return True
    if c.lower().endswith(("http://www.", "https://www.", "ftp://www.", "www.")):
        return True
    return False


def should_merge_with_next(current: str, nxt: str) -> bool:
    c = current.rstrip()
    n = nxt.lstrip()
    if not c or not n:
        return False

    if url_split_join_needed(c, n):
        return True

    # If GROBID ended a sentence at comma/semicolon/colon, it is almost always
    # a false split for our target/context purposes.
    if c.endswith((",", ";", ":")):
        return True

    # Open parenthetical often means the sentence is not complete yet.
    if paren_balance(c) > 0 and not terminal_sentence_end(c):
        return True

    # No true terminal punctuation yet.
    if not terminal_sentence_end(c):
        return True

    # Sometimes a following fragment starts with a lowercase connector because
    # GROBID split at an abbreviation or URL punctuation.
    if ends_with_abbreviation(c) and re.match(r"^(and|or|but|which|that|where|while|including|with|for|to|of|in)\b", n, re.I):
        return True

    return False


def join_fragments(current: str, nxt: str) -> str:
    c = current.rstrip()
    n = nxt.lstrip()
    if url_split_join_needed(c, n):
        return c + n
    return collapse_ws(c + " " + n)


class LogicalSentence:
    def __init__(self, s_elems: List[etree._Element], text: str) -> None:
        self.s_elems = s_elems
        self.text = text


def any_sentence(s_elem: etree._Element) -> bool:
    return True


def logical_sentences_in_p(p_elem, sentence_filter=is_eligible_sentence, fallback=False):
    raw = [s for s in p_elem.iter(f"{{{TEI_NS}}}s") if sentence_filter(s)]
    if not raw:
        # abstract and caption containers may not carry <s> children; treat the
        # whole container text as one sentence. Body paragraphs keep the strict
        # behaviour (no fallback), so body results are unchanged.
        if fallback:
            text = collapse_ws(elem_text(p_elem))
            return [LogicalSentence([p_elem], text)] if text else []
        return []
    groups: List[LogicalSentence] = []
    i = 0
    while i < len(raw):
        group = [raw[i]]
        text = serialize_sentence(raw[i])
        i += 1
        while i < len(raw):
            nxt_text = serialize_sentence(raw[i])
            if should_merge_with_next(text, nxt_text):
                text = join_fragments(text, nxt_text)
                group.append(raw[i])
                i += 1
            else:
                break
        groups.append(LogicalSentence(group, collapse_ws(text)))
    return groups


#  URL CANDIDATE EXTRACTION

def ref_type_url_candidates(s_elems: List[etree._Element]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    idx = 0
    for s_elem in s_elems:
        for ref in s_elem.findall(f".//{{{TEI_NS}}}ref[@type='url']"):
            raw = ref.get("target") or elem_text(ref)
            url = clean_url_printed(raw)
            if not url:
                continue
            if url.lower().startswith(("mailto:", "tel:", "javascript:", "data:", "file:")):
                continue
            out.append({
                "url_printed": url,
                "url_lines_joined": bool(re.search(r"\s", raw or "")),
                "url_span_pages": False,
                "source": "grobid_ref_url",
                "source_index": idx,
            })
            idx += 1
    return out


def regex_url_candidates(sentence_text: str, tagged_urls: List[str]) -> List[Dict[str, Any]]:
    tagged_norm = [normalize_url_for_compare(u) for u in tagged_urls]

    def already_tagged(u: str) -> bool:
        nu = normalize_url_for_compare(u)
        for tu in tagged_norm:
            if tu and (nu == tu or nu in tu):
                return True
        return False

    out: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []
    seen = set()

    for m in EXPLICIT_URL_RE.finditer(sentence_text):
        raw = m.group(0)
        url = clean_url_printed(raw)
        if not url or already_tagged(url):
            continue
        if url.lower().startswith(("mailto:", "tel:", "javascript:", "data:", "file:")):
            continue
        key = (m.start(), m.end(), normalize_url_for_compare(url))
        if key in seen:
            continue
        seen.add(key)
        occupied.append((m.start(), m.end()))
        out.append({
            "url_printed": url,
            "url_lines_joined": bool(re.search(r"\s", raw or "")),
            "url_span_pages": False,
            "source": "regex_explicit_url",
            "source_index": m.start(),
        })

    for m in DOI_RE.finditer(sentence_text):
        raw = m.group(0)
        url = clean_url_printed(raw)
        if not url or already_tagged(url):
            continue
        if any(a <= m.start() < b for a, b in occupied):
            continue
        key = (m.start(), m.end(), normalize_url_for_compare(url))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "url_printed": url,
            "url_lines_joined": bool(re.search(r"\s", raw or "")),
            "url_span_pages": False,
            "source": "regex_doi",
            "source_index": m.start(),
        })
    return out


def prefer_longer_url_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norms = [normalize_url_for_compare(c.get("url_printed", "")) for c in candidates]
    keep: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        ni = norms[i]
        if not ni:
            continue
        incomplete = False
        for j, nj in enumerate(norms):
            if i == j or not nj:
                continue
            if ni != nj and ni in nj and len(nj) >= len(ni) + 3:
                incomplete = True
                break
        if not incomplete:
            keep.append(cand)
    return keep


def candidate_key(candidate: Dict[str, Any]) -> Tuple[str, str]:
    norm = normalize_url_for_compare(candidate.get("url_printed", ""))
    source = candidate.get("source", "")
    index = str(candidate.get("source_index", ""))
    if source.startswith("regex"):
        return norm, index
    return norm, f"tag:{index}"


#  PER-TEI PROCESSING

def paper_id_from_tei(tei_path: Path) -> str:
    name = tei_path.name
    for suffix in (".tei.xml", ".xml", ".tei"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return tei_path.stem


def process_tei(tei_path: Path) -> Dict[str, Any]:
    try:
        tree = etree.parse(str(tei_path))
    except Exception as e:
        print(f"  [!] Failed to parse {tei_path.name}: {e}")
        pdf_file = f"{paper_id_from_tei(tei_path)}.pdf"
        return {"pdf_file": pdf_file, "items": []}

    root = tree.getroot()
    body = root.find(".//tei:body", NS)
    paper_id = paper_id_from_tei(tei_path)
    pdf_file = f"{paper_id}.pdf"
    items: List[Dict[str, Any]] = []

    # body prose
    if body is not None:
        for p_elem in body.iter(f"{{{TEI_NS}}}p"):
            if not is_eligible_p(p_elem):
                continue
            items.extend(extract_from_container(p_elem, pdf_file, "body", is_eligible_sentence))

    # abstract (in teiHeader, outside <body>)
    for abstract in root.findall(".//tei:abstract", NS):
        ps = list(abstract.iter(f"{{{TEI_NS}}}p"))
        for container in (ps or [abstract]):
            items.extend(extract_from_container(container, pdf_file, "body", any_sentence, fallback=True))

    # figure and table captions (<figDesc>, which the body prose pass excludes)
    if body is not None:
        for figdesc in body.iter(f"{{{TEI_NS}}}figDesc"):
            items.extend(extract_from_container(figdesc, pdf_file, "body", any_sentence, fallback=True))

    return {"pdf_file": pdf_file, "items": items}


def extract_from_container(container, pdf_file, location, sentence_filter, fallback=False):
    out: List[Dict[str, Any]] = []
    logical = logical_sentences_in_p(container, sentence_filter, fallback)
    for idx, ls in enumerate(logical):
        target = ls.text
        if not target or not should_keep_body_url_sentence(target):
            continue

        tag_candidates = ref_type_url_candidates(ls.s_elems)
        tag_urls = [c["url_printed"] for c in tag_candidates]
        candidates = prefer_longer_url_candidates(tag_candidates + regex_url_candidates(target, tag_urls))
        if not candidates:
            continue

        preceding = logical[idx - 1].text if idx > 0 else None
        trailing = logical[idx + 1].text if idx + 1 < len(logical) else None
        preceding = preceding if preceding and preceding.strip() else None
        trailing = trailing if trailing and trailing.strip() else None

        seen = set()
        for cand in candidates:
            key = candidate_key(cand)
            if key in seen:
                continue
            seen.add(key)
            item = {
                "url_printed": cand["url_printed"],
                "target_sentence": target,
                "preceding_sentence": preceding,
                "trailing_sentence": trailing,
                "at_paragraph_start": preceding is None,
                "at_paragraph_end": trailing is None,
                "url_lines_joined": bool(cand.get("url_lines_joined", False)),
                "url_span_pages": bool(cand.get("url_span_pages", False)),
                "pdf_file": pdf_file,
                "page": None,
                "location": location,
            }
            out.append({k: item[k] for k in BODY_URL_CONTEXT_FIELDS})
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_corpus(tei_dir: Path, out_dir: Path, label: str) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tei_files = sorted(tei_dir.rglob("*.tei.xml")) or sorted(tei_dir.rglob("*.xml"))

    print(f"\n{'=' * 70}")
    print(f"Input label : {label}")
    print(f"Input dir   : {tei_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"TEI files   : {len(tei_files)}")
    print(f"{'=' * 70}")

    if not tei_files:
        raise SystemExit(f"ERROR: no .tei.xml or .xml files found under: {tei_dir}")

    summary_rows: List[Dict[str, Any]] = []
    all_items: List[Dict[str, Any]] = []

    for tei_path in tqdm(tei_files, desc=label):
        out = process_tei(tei_path)
        paper_id = paper_id_from_tei(tei_path)
        out_path = out_dir / f"{paper_id}_body_urls.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        count = len(out.get("items", []))
        all_items.extend(out.get("items", []))
        summary_rows.append({
            "pdf_file": out.get("pdf_file", f"{paper_id}.pdf"),
            "tei_file": str(tei_path),
            "count": count,
            "items_json": str(out_path),
        })
        if count:
            print(f"  {tei_path.name:42s} -> {count:4d} body URL item(s)")

    all_jsonl = out_dir / "_ALL.body_urls.jsonl"
    write_jsonl(all_jsonl, all_items)

    summary_path = out_dir / "_SUMMARY.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    corpus_summary = {
        "label": label,
        "input_dir": str(tei_dir),
        "tei_files": len(tei_files),
        "items": len(all_items),
        "out_dir": str(out_dir),
        "combined_jsonl": str(all_jsonl),
        "summary_json": str(summary_path),
    }
    (out_dir / "_RUN_SUMMARY.json").write_text(
        json.dumps(corpus_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone: {len(tei_files)} TEIs, {len(all_items)} body URL item(s)")
    print(f"Combined JSONL: {all_jsonl}")
    print(f"Summary       : {summary_path}")
    return corpus_summary


def run(tei_dir, out_dir, label=None):
    tei_dir = Path(tei_dir)
    out_dir = Path(out_dir)
    if not tei_dir.is_dir():
        raise SystemExit(f"ERROR: TEI directory not found: {tei_dir}")
    return process_corpus(tei_dir, out_dir, label or tei_dir.name)


def main():
    ap = argparse.ArgumentParser(description="Extract body URLs and their sentence context from GROBID TEI.")
    ap.add_argument("-i", "--input", required=True, help="directory of GROBID TEI XML files")
    ap.add_argument("-o", "--output", required=True, help="directory for the JSON output")
    ap.add_argument("--label", default=None, help="label shown in logs (default: input dir name)")
    args = ap.parse_args()
    run(args.input, args.output, args.label)


if __name__ == "__main__":
    main()