import argparse, os, re, sys, unicodedata
from collections import defaultdict
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from rapidfuzz import fuzz

FUZZY_THRESHOLD = 80

BOILER_HOST = re.compile(
    r"(?i)(orcid\.org|creativecommons\.org|crossmark|/?mailto:|(?:^|[^a-z])tel:|javascript:)"
)

def _npid(pid):
    pid = re.sub(r"\.pdf$", "", str(pid).strip(), flags=re.I)
    pid = re.sub(r"v\d+$", "", pid).strip()
    m = re.match(r"^(\d{4}\.)(\d{1,3})$", pid)
    return m.group(1) + m.group(2).ljust(4, "0") if m else pid

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
    pid = _norm_text(pid)
    pid = re.sub(r"(?i)\.pdf$", "", pid).strip()
    if re.fullmatch(r"\d{4}\.\d{3}", pid):
        pid += "0"
    return pid

def _repair_url_spacing(s: str) -> str:
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

def _strip_balanced_wrappers(u: str) -> str:
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

def _normalise_url(u: str) -> str:
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

def _split_url_field(raw: str) -> list[str]:
    raw = _repair_url_spacing(str(raw or ""))
    if not raw:
        return []

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
    out: set[str] = set()

    for chunk in _split_url_field(raw):
        found = [_normalise_url(m.group(1)) for m in _URL_IN_FIELD_RE.finditer(chunk)]
        found = [u for u in found if u]

        if not found:
            maybe = _normalise_url(chunk)
            if maybe:
                found = [maybe]

        out.update(found)

    return out

def _format_url_set(urls: set[str]) -> str:
    return " | ".join(sorted(u for u in urls if u))

def _norm_target(t: str) -> str:
    return _norm_text(t)
def _first_existing_value(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col in row.index:
            val = str(row.get(col, "")).strip()
            if val:
                return val
    return ""

def _merge_pipe_values(existing: str, new_value: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for chunk in [existing, new_value]:
        for part in str(chunk or "").split("|"):
            value = part.strip()
            if value and value not in seen:
                seen.add(value)
                values.append(value)
    return " | ".join(values)

def _build_rows(df: pd.DataFrame, side: str, dataset_name: str = "") -> list[dict]:
    collapsed: dict[tuple[str, str], dict] = {}

    for row_idx, r in df.iterrows():
        pid = _normalize_paper_id(str(r.get("paper_id", "")))
        target_raw = str(r.get("target", "")).strip()
        target_norm = _norm_target(target_raw)
        key = (pid, target_norm)

        raw_url = str(r.get("url", "")).strip()
        urls = _extract_norm_urls_from_field(raw_url)

        if key not in collapsed:
            collapsed[key] = {
                "side": side,
                "paper_id": pid,
                "target_raw": target_raw,
                "target_norm": target_norm,
                "preceding": str(r.get("preceding", "")).strip(),
                "trailing": str(r.get("trailing", "")).strip(),
                "urls": set(urls),
                "split": str(r.get("split", "")).strip(),
                "label": _first_existing_value(r, ["label", "Label"]),
                "location": str(r.get("location", "")).strip(),
                "url_raw": raw_url,
                "dataset_name": dataset_name,
                "source_row_indices": [int(row_idx) if isinstance(row_idx, int) or str(row_idx).isdigit() else row_idx],
                "source_rows": [r.to_dict()],
            }
        else:
            collapsed[key]["urls"] |= urls
            collapsed[key]["url_raw"] = _merge_pipe_values(collapsed[key].get("url_raw", ""), raw_url)
            collapsed[key]["source_row_indices"].append(int(row_idx) if isinstance(row_idx, int) or str(row_idx).isdigit() else row_idx)
            collapsed[key]["source_rows"].append(r.to_dict())

            if not collapsed[key].get("split"):
                collapsed[key]["split"] = str(r.get("split", "")).strip()
            if not collapsed[key].get("label"):
                collapsed[key]["label"] = _first_existing_value(r, ["label", "Label"])
            if not collapsed[key].get("location"):
                collapsed[key]["location"] = str(r.get("location", "")).strip()
            if not collapsed[key].get("preceding"):
                collapsed[key]["preceding"] = str(r.get("preceding", "")).strip()
            if not collapsed[key].get("trailing"):
                collapsed[key]["trailing"] = str(r.get("trailing", "")).strip()

    return list(collapsed.values())

def _shared_exact_urls(g: dict, p: dict) -> set[str]:
    return set(g.get("urls", set())) & set(p.get("urls", set()))

def _can_match(g: dict, p: dict, threshold: float) -> bool:
    if not _shared_exact_urls(g, p):
        return False
    sim = fuzz.ratio(g["target_norm"], p["target_norm"])
    return sim >= threshold

def _bipartite_match(g_list: list[dict], p_list: list[dict], threshold: float) -> list[int]:
    n_g, n_p = len(g_list), len(p_list)
    adj: list[list[int]] = [[] for _ in range(n_g)]
    scores: dict[tuple[int, int], tuple[int, float]] = {}

    for gi, g in enumerate(g_list):
        for pi, p in enumerate(p_list):
            shared = _shared_exact_urls(g, p)
            if not shared:
                continue
            sim = float(fuzz.ratio(g["target_norm"], p["target_norm"]))
            if sim < threshold:
                continue
            adj[gi].append(pi)
            scores[(gi, pi)] = (len(shared), sim)

    for gi in range(n_g):
        adj[gi].sort(key=lambda pi: (-scores.get((gi, pi), (0, 0.0))[0], -scores.get((gi, pi), (0, 0.0))[1]))

    match_p = [-1] * n_p
    match_g = [-1] * n_g

    def dfs(gi: int, visited: set[int]) -> bool:
        for pi in adj[gi]:
            if pi in visited:
                continue
            visited.add(pi)
            if match_p[pi] == -1 or dfs(match_p[pi], visited):
                match_p[pi] = gi
                match_g[gi] = pi
                return True
        return False

    order = sorted(
        range(n_g),
        key=lambda gi: (
            len(adj[gi]),
            -max((scores[(gi, pi)][0] for pi in adj[gi]), default=0),
            -max((scores[(gi, pi)][1] for pi in adj[gi]), default=0.0),
        ),
    )
    for gi in order:
        dfs(gi, set())

    return match_g

LOCATIONS = ["body", "footnote", "reference"]

def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def _boiler_gold(url):
    return any(BOILER_HOST.search(u) for u in _extract_norm_urls_from_field(url))

def _load_located(path, role):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("paper_id", "target", "url", "location"):
        if col not in df.columns:
            sys.exit(f"ERROR: {role} file '{path}' is missing required column '{col}'. "
                     f"Found columns: {list(df.columns)}")
    df = df.copy()
    df["__loc"] = df["location"].astype(str).str.strip().str.lower()
    return df

def _sub(df, loc):
    return df[df["__loc"] == loc].reset_index(drop=True)

def eval_body_footnote(gold_df, pred_df, threshold=FUZZY_THRESHOLD):
    g_rows = _build_rows(gold_df, "gold", "gold") if len(gold_df) else []
    p_rows = _build_rows(pred_df, "pred", "pred") if len(pred_df) else []
    by_g, by_p = {}, {}
    for g in g_rows: by_g.setdefault(g["paper_id"], []).append(g)
    for p in p_rows: by_p.setdefault(p["paper_id"], []).append(p)
    TP = FP = FN = 0
    for pid in set(by_g) | set(by_p):
        gl, pl = by_g.get(pid, []), by_p.get(pid, [])
        gold_open = set(range(len(gl))); pred_used = set()
        m = _bipartite_match(gl, pl, threshold)
        for gi, pi in enumerate(m):
            if pi != -1: gold_open.discard(gi); pred_used.add(pi)
        TP += len(gl) - len(gold_open)
        for pi in range(len(pl)):
            if pi not in pred_used: FP += 1
        for gi in sorted(gold_open):
            if _boiler_gold(gl[gi].get("url_raw", "")): continue
            FN += 1
    return TP, FP, FN

def eval_reference(gold_df, pred_df, thr=FUZZY_THRESHOLD):
    gold = {}; gold_pids = set(); gold_raw = {}
    for r in gold_df.to_dict("records"):
        pid = _npid(r.get("paper_id", "")); t = (r.get("target", "") or "").strip()
        raw = r.get("url", ""); urls = _extract_norm_urls_from_field(raw)
        if not (pid and urls and t): continue
        gold_pids.add(pid)
        for u in urls:
            gold.setdefault((pid, u), {"sents": []})["sents"].append(t)
            gold_raw.setdefault((pid, u), raw)
    pred = defaultdict(list); pred_rows = []
    for r in pred_df.to_dict("records"):
        pid = _npid(r.get("paper_id", ""))
        if not pid: continue
        raw = r.get("url", ""); tgt = r.get("target", "") or ""
        us = _extract_norm_urls_from_field(raw)
        for u in us: pred[(pid, u)].append(tgt)
        pred_rows.append((pid, us))
    gold_keys = set(gold.keys())
    TP = FN = 0
    for key, g in gold.items():
        matched = key in pred and any(
            fuzz.token_sort_ratio(gs, ps) >= thr for gs in g["sents"] for ps in pred[key])
        if matched: TP += 1
        elif not _boiler_gold(gold_raw.get(key, "")): FN += 1
    fp_keys = set()
    for pid, us in pred_rows:
        if pid not in gold_pids: continue
        for u in us:
            if (pid, u) not in gold_keys: fp_keys.add((pid, u))
    return TP, len(fp_keys), FN

def run(pred_name, gold_path, pred_path):
    gold_df = _load_located(gold_path, "gold")
    pred_df = _load_located(pred_path, "baseline")
    res = {}
    res["body"]      = eval_body_footnote(_sub(gold_df, "body"),      _sub(pred_df, "body"))
    res["footnote"]  = eval_body_footnote(_sub(gold_df, "footnote"),  _sub(pred_df, "footnote"))
    res["reference"] = eval_reference(_sub(gold_df, "reference"),     _sub(pred_df, "reference"))
    tp = sum(v[0] for v in res.values()); fp = sum(v[1] for v in res.values()); fn = sum(v[2] for v in res.values())
    res["OVERALL"] = (tp, fp, fn)

    print(f"  {'location':10} {'P':>8} {'R':>8} {'F1':>8}")
    for loc in ["body", "footnote", "reference", "OVERALL"]:
        P, R, F = prf(*res[loc])
        print(f"  {loc:10} {P:8.4f} {R:8.4f} {F:8.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default="sURL-3K.csv")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.pred))[0]
    run(name, a.gold, a.pred)