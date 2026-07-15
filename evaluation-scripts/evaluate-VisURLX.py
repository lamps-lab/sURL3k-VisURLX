#!/usr/bin/env python3
import argparse
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Sequence, Set
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from rapidfuzz import fuzz

EXPL_URL_RE = re.compile(
    r"""(?ix)
    \b(
      https?://[^\s\]\[<>\"']+
      |
      www\.[^\s\]\[<>\"']+
    )
    """
)
URL_IN_FIELD_RE = re.compile(
    r"""(?ix)
    \b(
      https?://[^\s\]\[<>\"']+
      |
      www\.[^\s\]\[<>\"']+
      |
      (?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?::\d+)?(?:/[^\s\]\[<>\"']*)?
      |
      \d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?:/[^\s\]\[<>\"']*)?
    )
    """
)
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{3,5})(?:v\d+)?")

def norm_text(s: object) -> str:
    s = unicodedata.normalize("NFKC", "" if s is None else str(s))
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
        "∼": "~",
        "˜": "~",
        "～": "~",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_paper_id(pid: object) -> str:
    pid = norm_text(pid)
    pid = pid.replace(".pdf", "")
    m = ARXIV_ID_RE.search(pid)
    if m:
        pid = m.group(1)
    m = re.fullmatch(r"(\d{4}\.\d{3})", pid)
    if m:
        return pid + "0"
    return pid


def repair_url_spacing(s: object) -> str:
    s = norm_text(s)
    s = re.sub(r"(?i)\bhttps?\s*:\s*/\s*/", lambda m: re.sub(r"\s+", "", m.group(0)), s)
    s = re.sub(r"(?i)\bwww\.\s+", "www.", s)
    s = re.sub(r"(?<=\w)\s+(?=[/?#&=:%._~\-])", "", s)
    s = re.sub(r"(?<=[/?#&=:%._~\-])\s+(?=\w)", "", s)
    s = re.sub(r"(?<=\.)\s+(?=\w)", "", s)
    s = re.sub(r"(?<=\w)\s+(?=\.)", "", s)
    return s


def normalize_url(u: object) -> str:
    u = repair_url_spacing(u)
    u = re.sub(r"\[last visit:[^\]]*$", "", u, flags=re.I).strip()
    u = u.strip("[](){}<>\"'`")

    while u and u[-1] in ".,;:)]}>":
        u = u[:-1]

    u = re.sub(r"\s+", "", u)
    if not u:
        return ""

    candidate = u if re.match(r"(?i)^https?://", u) else "http://" + u
    try:
        parts = urlsplit(candidate)
    except Exception:
        return u.lower().rstrip("/")

    netloc = (parts.netloc or "").lower()
    path = (parts.path or "")
    query = (parts.query or "")

    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = re.sub(r"/{2,}", "/", path).rstrip("/").lower()
    query = query.lower()

    normalized = urlunsplit(("", netloc, path, query, "")).lstrip("//")
    return normalized


def split_url_field(raw: object, side: str) -> List[str]:
    raw = repair_url_spacing(raw)
    if not raw:
        return []
    if side == "pred":
        parts = [p.strip() for p in raw.split("|")]
    else:
        parts = re.split(
            r""",\s*(?=(?:https?://|www\.|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}|\d{1,3}(?:\.\d{1,3}){3}))""",
            raw,
        )
    return [p for p in parts if p]


def explicit_urls_from_target(target: object) -> List[str]:
    text = repair_url_spacing(target)
    out, seen = [], set()
    for m in EXPL_URL_RE.finditer(text):
        u = normalize_url(m.group(1))
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_url_list(url_field: object, side: str, target: object = "") -> List[str]:
    out, seen = [], set()
    for part in split_url_field(url_field, side):
        found = [normalize_url(m.group(1)) for m in URL_IN_FIELD_RE.finditer(part)]
        found = [u for u in found if u]
        if not found:
            maybe = normalize_url(part)
            found = [maybe] if maybe else []
        for u in found:
            if u and u not in seen:
                seen.add(u)
                out.append(u)

    for u in explicit_urls_from_target(target):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def canonicalize_target(target: object) -> str:
    s = norm_text(target)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def target_skeleton(target: object) -> str:
    s = canonicalize_target(target)
    s = EXPL_URL_RE.sub(" URL ", s)
    s = s.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ")
    s = re.sub(r"(?<!\w)\d+\s+(?=URL\b)", "", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def pick_col(row: pd.Series, names: Sequence[str], default: str = "") -> str:
    for name in names:
        if name in row.index:
            return str(row.get(name, default))
    return default


def build_rows(df: pd.DataFrame, side: str, dataset_name: str = "", urls_from_target: bool = True) -> List[Dict]:
    rows: List[Dict] = []
    for idx, row in df.iterrows():
        paper_id = normalize_paper_id(pick_col(row, ["paper_id", "pdf_file", "file"]))
        target = pick_col(row, ["target", "restored_sentence", "target_sentence"])
        url_raw = pick_col(row, ["url", "Url", "urls", "URL"])
        url_list = extract_url_list(url_raw, side=side, target=target if urls_from_target else "")
        label = pick_col(row, ["actual_label", "Label", "label"])
        rows.append(
            {
                "side": side,
                "dataset_name": dataset_name,
                "row_index": int(idx),
                "paper_id": paper_id,
                "split": pick_col(row, ["split"]),
                "actual_label": label,
                "file": pick_col(row, ["file", "pdf_file"], default=(paper_id + ".pdf" if paper_id else "")),
                "location": pick_col(row, ["location"]),
                "preceding": pick_col(row, ["preceding", "preceding_sentence"]),
                "target_raw": target,
                "target_norm": canonicalize_target(target),
                "target_skeleton": target_skeleton(target),
                "trailing": pick_col(row, ["trailing", "trailing_sentence"]),
                "url_raw": url_raw,
                "url_list_norm": url_list,
                "url_set_norm": frozenset(url_list),
            }
        )
    return rows


def merge_duplicate_rows(rows: List[Dict], skeleton_threshold: float = 97.0) -> List[Dict]:
    by_paper: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["paper_id"]].append(r)

    merged_rows: List[Dict] = []
    for paper_id, lst in by_paper.items():
        used = [False] * len(lst)
        for i, base in enumerate(lst):
            if used[i]:
                continue
            used[i] = True
            group = [base]
            current_urls = set(base["url_set_norm"])
            for j, cand in enumerate(lst):
                if used[j]:
                    continue
                sim = float(fuzz.ratio(base["target_skeleton"], cand["target_skeleton"]))
                cand_urls = set(cand["url_set_norm"])
                overlap = bool(current_urls & cand_urls)
                subset = current_urls.issubset(cand_urls) or cand_urls.issubset(current_urls)
                if sim >= skeleton_threshold and (overlap or subset or not current_urls or not cand_urls):
                    used[j] = True
                    group.append(cand)
                    current_urls |= cand_urls

            rep = max(group, key=lambda x: (len(x["url_set_norm"]), len(x["target_raw"])))
            rep = rep.copy()
            rep["merged_from_rows"] = [g["row_index"] for g in group]
            rep["url_set_norm"] = frozenset(current_urls)
            rep["url_list_norm"] = sorted(current_urls)
            merged_rows.append(rep)
    return merged_rows

def collapse_rows(rows: List[Dict]) -> List[Dict]:
    merged: Dict[tuple, Dict] = {}
    for r in rows:
        key = (r["paper_id"], r["target_norm"])
        if key not in merged:
            r = dict(r)
            r["merged_from_rows"] = [r["row_index"]]
            merged[key] = r
            continue
        base = merged[key]
        urls = set(base["url_set_norm"]) | set(r["url_set_norm"])
        base["url_set_norm"] = frozenset(urls)
        base["url_list_norm"] = sorted(urls)
        base["merged_from_rows"].append(r["row_index"])
    return list(merged.values())


def match_pairs(gold: List[Dict], pred: List[Dict], adj: Dict[int, List[int]],
                g2p: List[int], p2g: List[int]) -> None:
    def augment(i: int, seen: Set[int]) -> bool:
        for j in adj.get(i, []):
            if j in seen:
                continue
            seen.add(j)
            if p2g[j] == -1 or augment(p2g[j], seen):
                p2g[j] = i
                g2p[i] = j
                return True
        return False

    order = sorted(range(len(gold)), key=lambda i: (len(adj.get(i, [])), gold[i]["row_index"]))
    for i in order:
        if g2p[i] == -1:
            augment(i, set())


def shared_urls(g: Dict, p: Dict) -> Set[str]:
    return set(g["url_set_norm"]) & set(p["url_set_norm"])


def accepted_score(pred: Dict, urls: Set[str], accepted: Dict, paper_id: str) -> float:
    best = 0.0
    for u in urls:
        for t in accepted.get((paper_id, u), []):
            s = float(fuzz.ratio(t, pred["target_norm"]))
            if s > best:
                best = s
    return best


def evaluate_simple(gold: List[Dict], pred: List[Dict], threshold: float) -> Dict[str, int]:
    gold = collapse_rows(gold)
    pred = collapse_rows(pred)
    bg, bp = defaultdict(list), defaultdict(list)
    for r in gold:
        bg[r["paper_id"]].append(r)
    for r in pred:
        bp[r["paper_id"]].append(r)

    tp = fp = fn = 0
    for paper_id in set(bg) | set(bp):
        gl, pl = bg.get(paper_id, []), bp.get(paper_id, [])
        adj: Dict[int, List[int]] = {}
        score: Dict[tuple, float] = {}
        for i, g in enumerate(gl):
            for j, p in enumerate(pl):
                sh = shared_urls(g, p)
                if not sh:
                    continue
                s = float(fuzz.ratio(g["target_norm"], p["target_norm"]))
                if s >= threshold:
                    adj.setdefault(i, []).append(j)
                    score[(i, j)] = s
        for i in adj:
            adj[i].sort(key=lambda j: (-score[(i, j)], pl[j]["row_index"]))
        g2p, p2g = [-1] * len(gl), [-1] * len(pl)
        match_pairs(gl, pl, adj, g2p, p2g)
        tp += sum(1 for j in g2p if j != -1)
        fn += sum(1 for j in g2p if j == -1)
        fp += sum(1 for i in p2g if i == -1)
    return {"tp": tp, "fp": fp, "fn": fn, "extra": 0, "stage1": tp, "stage2": 0}


def evaluate_reference(gold: List[Dict], pred: List[Dict], accepted: Dict,
                       threshold: float, dumps: Dict[str, list]) -> Dict[str, int]:
    gold = merge_duplicate_rows(gold)
    bg, bp = defaultdict(list), defaultdict(list)
    for r in gold:
        bg[r["paper_id"]].append(r)
    for r in pred:
        bp[r["paper_id"]].append(r)

    stage1 = stage2 = extra = fp = fn = 0
    for paper_id in set(bg) | set(bp):
        gl, pl = bg.get(paper_id, []), bp.get(paper_id, [])
        sh = [[shared_urls(g, p) for p in pl] for g in gl]
        g2p, p2g = [-1] * len(gl), [-1] * len(pl)

        adj: Dict[int, List[int]] = {}
        s1: Dict[tuple, float] = {}
        for i, g in enumerate(gl):
            for j, p in enumerate(pl):
                if not sh[i][j]:
                    continue
                s = float(fuzz.ratio(g["target_norm"], p["target_norm"]))
                if s >= threshold:
                    adj.setdefault(i, []).append(j)
                    s1[(i, j)] = s
        for i in adj:
            adj[i].sort(key=lambda j: (-len(sh[i][j]), -s1[(i, j)], pl[j]["row_index"]))
        match_pairs(gl, pl, adj, g2p, p2g)
        stage1 += sum(1 for j in g2p if j != -1)

        adj2: Dict[int, List[int]] = {}
        s2: Dict[tuple, float] = {}
        for i in range(len(gl)):
            if g2p[i] != -1:
                continue
            for j in range(len(pl)):
                if p2g[j] != -1 or not sh[i][j]:
                    continue
                s = accepted_score(pl[j], sh[i][j], accepted, paper_id)
                if s >= threshold:
                    adj2.setdefault(i, []).append(j)
                    s2[(i, j)] = s
        for i in adj2:
            adj2[i].sort(key=lambda j: (-len(sh[i][j]), -s2[(i, j)], pl[j]["row_index"]))
        before = sum(1 for j in g2p if j != -1)
        match_pairs(gl, pl, adj2, g2p, p2g)
        stage2 += sum(1 for j in g2p if j != -1) - before
        for i, j in enumerate(g2p):
            if j != -1 and (i, j) in s2:
                dumps["stage2"].append({"paper_id": paper_id, "url": gl[i]["url_raw"],
                                        "accepted_similarity": round(s2[(i, j)], 2),
                                        "gold_target": gl[i]["target_raw"],
                                        "output_target": pl[j]["target_raw"]})

        claimed: Set[str] = set()
        for i, j in enumerate(g2p):
            if j != -1:
                claimed |= set(gl[i]["url_set_norm"])
        for j, p in enumerate(pl):
            if p2g[j] != -1:
                continue
            urls = set(p["url_set_norm"]) & claimed
            s = accepted_score(p, urls, accepted, paper_id) if urls else 0.0
            if urls and s >= threshold:
                extra += 1
                dumps["extra"].append({"paper_id": paper_id, "url": p["url_raw"],
                                       "accepted_similarity": round(s, 2),
                                       "output_target": p["target_raw"]})
            else:
                fp += 1
                dumps["fp"].append({"paper_id": paper_id, "url": p["url_raw"],
                                    "url_claimed_by_gold": "yes" if urls else "no",
                                    "best_accepted_similarity": round(s, 2),
                                    "output_target": p["target_raw"]})
        fn += sum(1 for j in g2p if j == -1)
    return {"tp": stage1 + stage2, "fp": fp, "fn": fn, "extra": extra,
            "stage1": stage1, "stage2": stage2}


def load_accepted(path: str) -> Dict:
    accepted: Dict = defaultdict(list)
    for r in build_rows(pd.read_csv(path, dtype=str, keep_default_na=False), "gold",
                        dataset_name="accepted", urls_from_target=False):
        for u in r["url_set_norm"]:
            accepted[(r["paper_id"], u)].append(r["target_norm"])
    return accepted


def scores(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--accepted_set", required=True)
    ap.add_argument("--threshold", type=float, default=80.0)
    ap.add_argument("--dump_dir", default="")
    args = ap.parse_args()

    out = pd.read_csv(args.output, dtype=str, keep_default_na=False)
    gold_all = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    accepted = load_accepted(args.accepted_set)
    dumps = {"stage2": [], "extra": [], "fp": []}

    total = {"tp": 0, "fp": 0, "fn": 0, "extra": 0}
    print(f"{'location':<12}{'P':>9}{'R':>9}{'F1':>9}")
    for location in ["body", "footnote", "reference"]:
        gold = build_rows(gold_all[gold_all["location"] == location], "gold",
                          dataset_name=args.gold,
                          urls_from_target=(location == "reference"))
        pred = build_rows(out[out["location"] == location], "pred",
                          dataset_name=args.output, urls_from_target=(location == "reference"))
        if location == "reference":
            res = evaluate_reference(gold, pred, accepted, args.threshold, dumps)
        else:
            res = evaluate_simple(gold, pred, args.threshold)
        for k in total:
            total[k] += res[k]
        p, r, f = scores(res["tp"], res["fp"], res["fn"])
        print(f"{location:<12}{p:>9.4f}{r:>9.4f}{f:>9.4f}")

    p, r, f = scores(total["tp"], total["fp"], total["fn"])
    print(f"{'overall':<12}{p:>9.4f}{r:>9.4f}{f:>9.4f}")

    if args.dump_dir:
        for name, rows in dumps.items():
            pd.DataFrame(rows).to_csv(f"{args.dump_dir}/reference_{name}.csv", index=False)


if __name__ == "__main__":
    main()