#!/usr/bin/env python3
import os, csv, re, argparse, collections, unicodedata
from urllib.parse import urlsplit, urlunsplit

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


def _norm_text(s):
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


def _normalize_paper_id(pid):
    pid = _norm_text(pid)
    pid = re.sub(r"(?i)\.pdf$", "", pid).strip()
    if re.fullmatch(r"\d{4}\.\d{3}", pid):
        pid += "0"
    return pid


def _repair_url_spacing(s):
    s = _norm_text(s)
    s = re.sub(r"\$\s*\\sim\s*\$", "~", s)
    s = re.sub(r"\{\s*\\sim\s*\}", "~", s)
    s = re.sub(r"\\sim(?![A-Za-z])", "~", s)
    s = s.replace(r"\~", "~")
    for c in "˜∼～∽\u02dc\u223c\u223b\u2053\uff5e\u0303\u033e\u0334":
        s = s.replace(c, "~")

    s = re.sub(r"(?i)\bhttps?\s*:\s*/\s*/", lambda m: re.sub(r"\s+", "", m.group(0)), s)
    s = re.sub(r"(?i)\b(?:s?ftps?|sftp|s3|gs|az)\s*:\s*/\s*/", lambda m: re.sub(r"\s+", "", m.group(0)), s)
    s = re.sub(r"(?i)\bwww\.\s+", "www.", s)
    s = re.sub(r"(?i)\bdoi:\s+", "doi:", s)

    s = re.sub(r"(?<=\w)\s+(?=[/?#&=:%._~\-])", "", s)
    s = re.sub(r"(?<=[/?#&=:%._~\-])\s+(?=\w)", "", s)
    s = re.sub(r"(?<=\.)\s+(?=\w)", "", s)
    s = re.sub(r"(?<=\w)\s+(?=\.)", "", s)
    return s


def _strip_balanced_wrappers(u):
    u = u.strip()
    wrappers = [("[", "]"), ("(", ")"), ("{", "}"), ("<", ">"), ('"', '"'), ("'", "'"), ("`", "`")]
    changed = True
    while changed and len(u) >= 2:
        changed = False
        for left, right in wrappers:
            if u.startswith(left) and u.endswith(right):
                u = u[1:-1].strip()
                changed = True
    return u


def _normalise_url(u):
    if not isinstance(u, str):
        return ""

    u = _repair_url_spacing(u)
    u = _strip_balanced_wrappers(u)

    u = u.strip().strip(",; ")
    u = re.sub(r"^['\"]+|['\"]+$", "", u).strip()

    while u and u[-1] in ".,;:)]}>\"'`":
        u = u[:-1].strip()

    u = re.sub(r"(?i)^doi:\s*https?://(?:dx\.)?doi\.org/", "doi:", u)
    u = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "doi:", u)

    u = re.sub(r"\s+", "", u)
    if not u:
        return ""

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
    path = path.lower()
    query = query.lower()

    return urlunsplit(("", netloc, path, query, "")).lstrip("//")


def _split_url_field(raw):
    raw = _repair_url_spacing(str(raw or ""))
    if not raw:
        return []

    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()

    chunks = []
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


def _extract_norm_urls_from_field(raw):
    out = set()
    for chunk in _split_url_field(raw):
        found = [_normalise_url(m.group(1)) for m in _URL_IN_FIELD_RE.finditer(chunk)]
        found = [u for u in found if u]
        if not found:
            maybe = _normalise_url(chunk)
            if maybe:
                found = [maybe]
        out.update(found)
    return out


BOILER_HOST = re.compile(
    r"(?i)(orcid\.org|creativecommons\.org|crossmark|/?mailto:|(?:^|[^a-z])tel:|javascript:)"
)


def norm_pid(p):
    return _normalize_paper_id(p)


def norm_urls(field):
    return _extract_norm_urls_from_field(field)


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    return P, R, (2 * P * R / (P + R) if P + R else 0)


def load(path, drop_boilerplate=False):
    d = collections.defaultdict(set)
    rd = csv.DictReader(open(path, encoding="utf-8"))
    col = next((c for c in ["url", "url_raw", "url_norm"] if c in rd.fieldnames), None)
    if col is None:
        raise SystemExit(f"{path}: no url column (need url/url_raw/url_norm)")
    for r in rd:
        pid = norm_pid(r["paper_id"])
        for u in norm_urls(r[col]):
            if drop_boilerplate and BOILER_HOST.search(u):
                continue
            d[pid].add(u)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    a = ap.parse_args()

    gold = load(a.gold, drop_boilerplate=True)

    print(f"{'pipeline':<26}{'P':>9}{'R':>9}{'F1':>9}")
    for name in a.sets:
        path = name if name.endswith(".csv") else name + ".csv"
        pred = load(path)
        tp = fp = fn = 0
        for pid in set(pred) | set(gold):
            p, g = pred.get(pid, set()), gold.get(pid, set())
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
        P, R, F = prf(tp, fp, fn)
        label = os.path.basename(path)[:-4]
        print(f"{label:<26}{P:>9.4f}{R:>9.4f}{F:>9.4f}")


if __name__ == "__main__":
    main()