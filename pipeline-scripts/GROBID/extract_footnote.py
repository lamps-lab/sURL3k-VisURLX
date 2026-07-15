#!/usr/bin/env python3
# Footnote URL extractor for GROBID TEI.
# Primary pass: body <ref type="foot"> that link to a <note place="foot">
# carrying a URL. It then restores the citing sentence by splicing the
# footnote content back into the marker position.
# Two extra passes handle GROBID quirks (see README):
#   - bibr pass: footnote markers GROBID mislabels as <ref type="bibr">
#   - url-ref pass: URLs GROBID inlines as <ref type="url"> (off by default)

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from lxml import etree

# bibr pass adds true positives without hurting precision on our test papers.
# url-ref pass is off by default: its true positives are already caught by the
# bibr pass, and it adds false positives.
ENABLE_BIBR_RECOVERY = True
ENABLE_URL_REF_RECOVERY = False

TEI_NS = "http://www.tei-c.org/ns/1.0"
def Q(tag: str) -> str:
    return f"{{{TEI_NS}}}{tag}"
XML_ID_ATTR = "{http://www.w3.org/XML/1998/namespace}id"


# URL detection (same regex across the GROBID and PyMuPDF baselines)
_URL_TERM = r"[^\s<>\[\]\(\)\{\}\"'`,;]"
URL_PATTERNS = [
    rf"https?://{_URL_TERM}+",
    rf"ftp://{_URL_TERM}+",
    rf"(?<![A-Za-z0-9])www\.{_URL_TERM}+",
    rf"(?<![A-Za-z])doi:\s?10\.\d+/{_URL_TERM}+",
    r"(?<![A-Za-z])arxiv:\s?\d{4}\.\d{4,5}(?:v\d+)?",
    rf"(?<![A-Za-z/.])github\.com/{_URL_TERM}+",
    rf"(?<![A-Za-z/.])gitlab\.com/{_URL_TERM}+",
    rf"(?<![A-Za-z/.])bitbucket\.org/{_URL_TERM}+",
]
URL_RE = re.compile("|".join(f"(?:{p})" for p in URL_PATTERNS), re.IGNORECASE)


def clean_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    for opener, closer in [("(", ")"), ("[", "]"), ("<", ">"), ('"', '"'), ("'", "'")]:
        if url.startswith(opener) and url.endswith(closer):
            url = url[1:-1]
            break
    if url.startswith("(") and ")" not in url:
        url = url[1:]
    url = url.rstrip(".,;:")
    if url.endswith(")") and "(" not in url:
        url = url.rstrip(")")
    return url.strip()


def extract_urls(content: str) -> List[str]:
    if not content:
        return []
    out, seen = [], set()
    for u in URL_RE.findall(content):
        u = clean_url(u)
        if not u or u.lower().startswith("mailto:"):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


#  TREE NAVIGATION HELPERS

def _local(elem) -> str:
    return etree.QName(elem).localname


def find_references_div(root):
    return root.find(f".//{Q('div')}[@type='references']")


def is_inside(elem, ancestor) -> bool:
    if ancestor is None:
        return False
    cur = elem
    while cur is not None:
        if cur is ancestor:
            return True
        cur = cur.getparent()
    return False


def ancestor_of_type(elem, tag_local: str):
    cur = elem.getparent()
    while cur is not None:
        if _local(cur) == tag_local:
            return cur
        cur = cur.getparent()
    return None


def previous_sentence_in_paragraph(s_elem):
    parent = s_elem.getparent()
    if parent is None or _local(parent) != "p":
        return None
    sib = s_elem.getprevious()
    while sib is not None:
        if _local(sib) == "s":
            return sib
        sib = sib.getprevious()
    return None


def next_sentence_in_paragraph(s_elem):
    parent = s_elem.getparent()
    if parent is None or _local(parent) != "p":
        return None
    sib = s_elem.getnext()
    while sib is not None:
        if _local(sib) == "s":
            return sib
        sib = sib.getnext()
    return None


#  FOOTNOTE MAP

def collect_note_text(note_elem) -> str:
    return " ".join(" ".join(note_elem.itertext()).split())


def build_footnote_map(root, refs_div) -> Dict[str, Dict[str, Any]]:
    fmap: Dict[str, Dict[str, Any]] = {}
    for note in root.findall(f".//{Q('note')}[@place='foot']"):
        if is_inside(note, refs_div):
            continue
        xml_id = note.get(XML_ID_ATTR)
        if not xml_id:
            continue
        content = collect_note_text(note)
        urls = extract_urls(content)
        if not urls:
            continue
        marker = note.get("n") or xml_id
        fmap[xml_id] = {"marker": marker, "content": content, "urls": urls}
    return fmap


def index_footnotes_by_marker(fmap: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {fn["marker"]: xid for xid, fn in fmap.items()}


#  SENTENCE SERIALISATION
#
# Modes:
#   serialize_sentence(s, target_xml_id=None)
#       Drops every <ref type="foot"> placeholder text.
#       Used to build original_sentence.
#
#   serialize_sentence(s, target_xml_id='foot_3', target_marker='10', fmap=...)
#       Replaces:
#         - the <ref type="foot" target="#foot_3"> with ' [<content>]'
#         - the <ref type="bibr">10</ref> with ' [<content>]'  (RECOVERY 1)
#       Drops other <ref type="foot"> placeholders.
#       Used to build restored_sentence.

def serialize_sentence(s_elem,
                       target_xml_id: Optional[str] = None,
                       target_marker: Optional[str] = None,
                       fmap: Optional[Dict[str, Dict[str, Any]]] = None,
                       inline_url_target: Optional[str] = None) -> str:
    out_parts: List[str] = []
    if s_elem.text:
        out_parts.append(s_elem.text)

    def visit(elem):
        tag = _local(elem)
        is_foot_ref = (tag == "ref" and elem.get("type") == "foot")
        is_bibr_ref = (tag == "ref" and elem.get("type") == "bibr")

        if is_foot_ref:
            ref_target = (elem.get("target") or "").lstrip("#")
            if (target_xml_id and ref_target == target_xml_id
                    and fmap and ref_target in fmap):
                out_parts.append(f" [{fmap[ref_target]['content']}]")
            # otherwise drop the placeholder text
        elif is_bibr_ref and target_marker is not None:
            ref_text = (elem.text or "").strip()
            if ref_text == target_marker and fmap and target_xml_id in fmap:
                out_parts.append(f" [{fmap[target_xml_id]['content']}]")
            else:
                # keep the bibr's printed text in the sentence
                if elem.text:
                    out_parts.append(elem.text)
                for child in elem:
                    visit(child)
                    if child.tail:
                        out_parts.append(child.tail)
        else:
            if elem.text:
                out_parts.append(elem.text)
            for child in elem:
                visit(child)
                if child.tail:
                    out_parts.append(child.tail)

    for child in s_elem:
        visit(child)
        if child.tail:
            out_parts.append(child.tail)

    return " ".join("".join(out_parts).split())


#  ITEM BUILDER

def build_item_from_ref(ref_elem,
                        target_xml_id: str,
                        fmap: Dict[str, Dict[str, Any]],
                        pdf_id: str,
                        target_marker: Optional[str] = None) -> Optional[Dict[str, Any]]:
    s_elem = ancestor_of_type(ref_elem, "s")
    if s_elem is None:
        return None
    fn = fmap[target_xml_id]
    original = serialize_sentence(s_elem, target_xml_id=None)
    restored = serialize_sentence(
        s_elem,
        target_xml_id=target_xml_id,
        target_marker=target_marker,
        fmap=fmap,
    )
    prev_s = previous_sentence_in_paragraph(s_elem)
    next_s = next_sentence_in_paragraph(s_elem)
    preceding = serialize_sentence(prev_s) if prev_s is not None else None
    trailing  = serialize_sentence(next_s) if next_s is not None else None

    return {
        "original_sentence":  original,
        "restored_sentence":  restored,
        "footnote_marker":    fn["marker"],
        "footnote_content":   fn["content"],
        "url":                list(fn["urls"]),
        "preceding_sentence": preceding if (preceding and preceding.strip()) else None,
        "trailing_sentence":  trailing  if (trailing  and trailing.strip())  else None,
        "pdf_file":           pdf_id,
        "page":               0,
    }


#  url-ref pass: inline URL citations

def build_item_from_url_ref(ref_elem, pdf_id: str) -> Optional[Dict[str, Any]]:
    s_elem = ancestor_of_type(ref_elem, "s")
    if s_elem is None:
        return None
    raw_url = (ref_elem.get("target") or ref_elem.text or "").strip()
    url = clean_url(raw_url)
    if not url or url.lower().startswith("mailto:"):
        return None
    # Validate it's a real URL by running the extractor regex
    if not extract_urls(url):
        return None

    sentence = serialize_sentence(s_elem)
    prev_s = previous_sentence_in_paragraph(s_elem)
    next_s = next_sentence_in_paragraph(s_elem)
    preceding = serialize_sentence(prev_s) if prev_s is not None else None
    trailing  = serialize_sentence(next_s) if next_s is not None else None

    return {
        "original_sentence":  sentence,
        "restored_sentence":  sentence,   # URL is already inline
        "footnote_marker":    "",         # no separate marker
        "footnote_content":   url,
        "url":                [url],
        "preceding_sentence": preceding if (preceding and preceding.strip()) else None,
        "trailing_sentence":  trailing  if (trailing  and trailing.strip())  else None,
        "pdf_file":           pdf_id,
        "page":               0,
    }


#  PER-PAPER PROCESSING

def process_tei(tei_path: Path, pdf_id: str) -> List[Dict[str, Any]]:
    try:
        tree = etree.parse(str(tei_path))
    except etree.XMLSyntaxError as e:
        print(f"  [!] {tei_path.name}: invalid XML ({e})")
        return []
    root = tree.getroot()

    refs_div = find_references_div(root)
    fmap = build_footnote_map(root, refs_div)
    if not fmap:
        return []

    marker_to_xid = index_footnotes_by_marker(fmap)
    items: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    linked_xids: Set[str] = set()    # populated by primary pass; used by RECOVERY 1

    def maybe_emit(item):
        if item is None:
            return
        key = " ".join(item["restored_sentence"].lower().split())
        if not key or key in seen_keys:
            return
        seen_keys.add(key)
        items.append(item)

    # primary pass: <ref type="foot">
    for ref in root.findall(f".//{Q('ref')}[@type='foot']"):
        if is_inside(ref, refs_div):
            continue
        target_id = (ref.get("target") or "").lstrip("#")
        if target_id not in fmap:
            continue
        linked_xids.add(target_id)
        maybe_emit(build_item_from_ref(ref, target_id, fmap, pdf_id))

    # bibr pass: footnote markers mislabelled as <ref type="bibr">
    if ENABLE_BIBR_RECOVERY:
        for ref in root.findall(f".//{Q('ref')}[@type='bibr']"):
            if is_inside(ref, refs_div):
                continue
            ref_text = (ref.text or "").strip()
            if not ref_text or ref_text not in marker_to_xid:
                continue
            target_id = marker_to_xid[ref_text]
            if target_id in linked_xids:
                # already covered by the primary pass; skip to avoid double-counting
                continue
            maybe_emit(build_item_from_ref(
                ref, target_id, fmap, pdf_id,
                target_marker=ref_text,
            ))

    # url-ref pass: URLs inlined as <ref type="url">
    if ENABLE_URL_REF_RECOVERY:
        for ref in root.findall(f".//{Q('ref')}[@type='url']"):
            if is_inside(ref, refs_div):
                continue
            maybe_emit(build_item_from_url_ref(ref, pdf_id))

    return items


#  ORCHESTRATION

def write_pdf_outputs(out_dir: Path, pdf_id: str, items: List[Dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / f"{pdf_id}_footnotes.json"
    sentences_path = out_dir / f"{pdf_id}_sentences.json"
    items_path.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    sentences_path.write_text(
        json.dumps([it["restored_sentence"] for it in items],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return items_path


def run(input_dir, output_dir):
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"ERROR: input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    teis = sorted(in_dir.glob("*.tei.xml"))
    if not teis:
        raise SystemExit(f"No *.tei.xml files found under: {in_dir}")

    print(f"Input      : {in_dir}")
    print(f"Output     : {out_dir}")
    print(f"TEI files  : {len(teis)}")
    print(f"Extra passes: bibr={ENABLE_BIBR_RECOVERY}  url-ref={ENABLE_URL_REF_RECOVERY}\n")

    summaries: List[Dict[str, Any]] = []
    all_jsonl = out_dir / "_ALL.url_footnotes.jsonl"
    with all_jsonl.open("w", encoding="utf-8") as jl:
        for tei_path in teis:
            pdf_id = tei_path.name[:-len(".tei.xml")]
            items = process_tei(tei_path, pdf_id)
            write_pdf_outputs(out_dir, pdf_id, items)
            for it in items:
                jl.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"   {pdf_id:25s}  ->  {len(items):3d} item(s)")
            summaries.append({"pdf_file": pdf_id, "count": len(items)})

    (out_dir / "_SUMMARY.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(s["count"] for s in summaries)
    print()
    print(f"Done. papers={len(teis)}  items={total}")
    print(f"Per-PDF JSON in : {out_dir}")
    print(f"Combined JSONL  : {all_jsonl}")


def main():
    ap = argparse.ArgumentParser(description="Extract footnote URLs and restored citing sentences from GROBID TEI.")
    ap.add_argument("-i", "--input", required=True, help="directory of GROBID TEI XML files")
    ap.add_argument("-o", "--output", required=True, help="directory for the JSON output")
    args = ap.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
