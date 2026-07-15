# Classification evaluation

This folder scores the end-to-end OADS classification results. Each pipeline
extracts URLs with their context, and those records are passed to the EnSU
classifier. The script here matches the classified predictions against the gold
labels and reports precision, recall, and F1 per class for every pipeline.

## Contents

- `evaluate_classification.py` — the scoring script.
- `exact_url_evaluator.py` — URL and paper-id normalization used by the scorer.
- `*_all_predictions.csv` — one prediction file per pipeline (GROBID, Gpt-5-mini,
  PyMuPDF, Qwen, olmOCR). Each file holds the classified extraction records.
- `*-gold.csv` — the gold labels, split by corpus (arXiv, pmc) and location
  (body, footnote, reference). The scorer reads the `test` split from these.

## Requirements

- Python 3.9 or newer
- pandas

```
pip install pandas
```

## Running

Run the script from inside this folder. It reads the gold and prediction files
from the current directory, so no paths are needed.

```
python3 evaluate_classification.py
```

Options:

- `--variant {1,2,3,4}` (default 1) — how stage2 matches and the counting key
  are treated. Variant 1 counts stage2 matches as false positives and keys on
  distinct (paper_id, url) pairs.
- `--granularity {fine,coarse,binary}` (default fine):
  - `fine` — six classes (general-url, third-party-dataset,
    author-provided-dataset, third-party-software, author-provided-software,
    project).
  - `coarse` — four groups (general-url; dataset = the two dataset classes;
    software = the two software classes; project).
  - `binary` — two groups (OADS = classes 1 through 5; not-OADS = general-url).

Examples:

```
python3 evaluate_classification.py --granularity binary
python3 evaluate_classification.py --variant 3 --granularity coarse
```

## Output

For each pipeline the script prints a per-class table with precision, recall,
and F1, followed by the macro average (unweighted mean of the per-class F1) and
the micro average (computed from the pooled counts).


