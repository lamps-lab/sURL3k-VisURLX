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

```
  Gpt-5-mini
    class                           P      R     F1
    general-url                 0.911  0.799  0.852
    third-party-dataset         0.781  0.751  0.766
    author-provided-dataset     0.639  0.676  0.657
    third-party-software        0.883  0.759  0.816
    author-provided-software    0.664  0.833  0.739
    project                     0.500  0.625  0.556
    MACRO                                     0.731
    MICRO                       0.877  0.789  0.831
```

## Results (variant 1, fine)

Micro-averaged F1 per pipeline:

| pipeline    | P     | R     | F1    |
|-------------|-------|-------|-------|
| Gpt-5-mini  | 0.877 | 0.789 | 0.831 |
| GROBID      | 0.879 | 0.685 | 0.770 |
| olmOCR      | 0.826 | 0.593 | 0.690 |
| Qwen        | 0.793 | 0.651 | 0.715 |
| PyMuPDF     | 0.770 | 0.518 | 0.620 |

The Gpt-5-mini micro F1 of 0.831 is the 83.1% weighted F1 reported in the paper.
Per-class numbers are printed when the script runs.
