#!/usr/bin/env python3
import os, csv, re, argparse, collections, unicodedata
from urllib.parse import urlsplit, urlunsplit

# ─────────────────────────────────────────────────────────────────────────────
# Normalization + boilerplate filter — inlined VERBATIM from exact_url_evaluator.py
# / scope_prefilter.py so this file has no project dependencies.
# ─────────────────────────────────────────────────────────────────────────────
_URL_IN_FIELD_RE = re.compile(
    r"""(?ix)
    \b(
        (?:https?|s?ftps?|sftp|s3|gs|az)://[^\s\]\[<>"']+
        |
        www\.[^\s\]\[<>"']+
        |
        doi:\s*[^\s\]\[<>"'(),;|]+
        |
        DOI:\s*[^\s\]\[<>"'(),;|]+
        |
        (?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,10}(?::\d+)?(?:/[^\s\]\[<>"'|]*)?
        |
        \d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?:/[^\s\]\[<>"'|]*)?
    )
    """
)

def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    replacements = {
        "\u00a0": " ",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "\u200b": "",
        "\ufeff": "",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return re.sub(r"\s+", " ", s).strip()

def _normalize_paper_id(pid: str) -> str:
    """Normalize common paper-id variants without being corpus-specific."""
    pid = _norm_text(pid)
    pid = re.sub(r"(?i)\.pdf$", "", pid).strip()
    # arXiv safety: 1205.208 -> 1205.2080
    if re.fullmatch(r"\d{4}\.\d{3}", pid):
        pid += "0"
    return pid

def _repair_url_spacing(s: str) -> str:
    """Conservative repair for PDF text extraction spaces inside URLs."""
    s = _norm_text(s)
    # LaTeX tilde artifacts that survive PDF extraction inside URLs, e.g.
    # http://host/$\sim$user , http://host/{\sim}user , http://host/\~user .
    # These are a single character (~), never a multi-URL separator.
    s = re.sub(r"\$\s*\\sim\s*\$", "~", s)
    s = re.sub(r"\{\s*\\sim\s*\}", "~", s)
    s = re.sub(r"\\sim(?![A-Za-z])", "~", s)
    s = s.replace(r"\~", "~")
    for c in "˜∼～∽\u02dc\u223c\u223b\u2053\uff5e\u0303\u033e\u0334":
        s = s.replace(c, "~")

    # Repair broken protocol and common URL separator spacing.
    s = re.sub(r"(?i)\bhttps?\s*:\s*/\s*/", lambda m: re.sub(r"\s+", "", m.group(0)), s)
    s = re.sub(r"(?i)\b(?:s?ftps?|sftp|s3|gs|az)\s*:\s*/\s*/", lambda m: re.sub(r"\s+", "", m.group(0)), s)
    s = re.sub(r"(?i)\bwww\.\s+", "www.", s)
    s = re.sub(r"(?i)\bdoi:\s+", "doi:", s)

    # Remove spaces adjacent to URL separators, not all spaces globally.
    s = re.sub(r"(?<=\w)\s+(?=[/?#&=:%._~\-])", "", s)
    s = re.sub(r"(?<=[/?#&=:%._~\-])\s+(?=\w)", "", s)
    s = re.sub(r"(?<=\.)\s+(?=\w)", "", s)
    s = re.sub(r"(?<=\w)\s+(?=\.)", "", s)
    return s

def _strip_balanced_wrappers(u: str) -> str:
    u = u.strip()
    # Repeatedly strip balanced/simple wrappers around the whole URL field.
    wrappers = [("[", "]"), ("(", ")"), ("{", "}"), ("<", ">"), ('"', '"'), ("'", "'"), ("`", "`")]
    changed = True
    while changed and len(u) >= 2:
        changed = False
        for left, right in wrappers:
            if u.startswith(left) and u.endswith(right):
                u = u[1:-1].strip()
                changed = True
    return u

def _normalise_url(u: str) -> str:
    """Normalize one URL string for exact comparison."""
    if not isinstance(u, str):
        return ""

    u = _repair_url_spacing(u)
    u = _strip_balanced_wrappers(u)

    # Remove common list/CSV quote residue.
    u = u.strip().strip(",; ")
    u = re.sub(r"^['\"]+|['\"]+$", "", u).strip()

    # Trim trailing punctuation that is usually sentence punctuation.
    while u and u[-1] in ".,;:)]}>\"'`":
        u = u[:-1].strip()

    # DOI cleanup.
    u = re.sub(r"(?i)^doi:\s*https?://(?:dx\.)?doi\.org/", "doi:", u)
    u = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "doi:", u)

    # PDF line-break artifacts inside URL after extraction.
    u = re.sub(r"\s+", "", u)
    if not u:
        return ""

    # DOI values should compare by DOI content, independent of doi.org form.
    if re.match(r"(?i)^doi:", u):
        doi = re.sub(r"(?i)^doi:", "", u).strip().lower().rstrip("/")
        return f"doi:{doi}"
    if re.match(r"(?i)^10\.\d{4,9}/", u):
        return f"doi:{u.lower().rstrip('/')}"

    candidate = u if re.match(r"(?i)^[a-z][a-z0-9+.-]*://", u) else "http://" + u
    try:
        parts = urlsplit(candidate)
    except Exception:
        return u.lower().rstrip("/")

    netloc = (parts.netloc or "").lower()
    path = (parts.path or "")
    query = (parts.query or "")

    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = re.sub(r"/{2,}", "/", path)
    path = path.rstrip("/")

    # Preserve case in path? For evaluation, previous scripts lowercased path.
    # Keep that behavior for normalized exact matching consistency.
    path = path.lower()
    query = query.lower()

    return urlunsplit(("", netloc, path, query, "")).lstrip("//")

def _split_url_field(raw: str) -> list[str]:
    """
    Split a URL column into candidate chunks.

    Handles the main format requested by the user:
        url1 | url2 | url3

    Also tolerates JSON/list-like wrappers and comma-separated multiple URLs
    when a comma is followed by another URL-like start.
    """
    raw = _repair_url_spacing(str(raw or ""))
    if not raw:
        return []

    # Remove simple list wrappers but do not remove inner commas/pipes.
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()

    chunks: list[str] = []
    for pipe_part in raw.split("|"):
        pipe_part = pipe_part.strip()
        if not pipe_part:
            continue

        comma_parts = re.split(
            r""",\s*(?=(?:https?://|s?ftp://|sftp://|s3://|gs://|az://|www\.|doi:|DOI:|10\.\d{4,9}/|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,10}|\d{1,3}(?:\.\d{1,3}){3}))""",
            pipe_part,
        )
        chunks.extend([c.strip() for c in comma_parts if c.strip()])

    return chunks

def _extract_norm_urls_from_field(raw: str) -> set[str]:
    """Extract normalized URL set from a gold or prediction `url` column."""
    out: set[str] = set()

    for chunk in _split_url_field(raw):
        # First extract URL-like spans from the chunk.
        found = [_normalise_url(m.group(1)) for m in _URL_IN_FIELD_RE.finditer(chunk)]
        found = [u for u in found if u]

        # If regex found nothing, normalize the whole chunk as a fallback.
        if not found:
            maybe = _normalise_url(chunk)
            if maybe:
                found = [maybe]

        out.update(found)

    return out

