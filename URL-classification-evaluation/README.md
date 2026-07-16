# Classification evaluation

This folder scores the end-to-end OADS classification results. Each pipeline
extracts URLs with their context, and those records are passed to the EnSU
classifier. The script matches the classified predictions against the gold
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

For each pipeline it prints a per-class table of precision, recall, and F1,
followed by the macro average (unweighted mean of the per-class F1) and the
micro average (from the pooled counts).

## Granularity

The `--granularity` option controls how the six labels are grouped before
scoring. All three commands report P, R, and F1 per class for every pipeline.

### fine (default): six classes

```
python3 evaluate_classification.py
python3 evaluate_classification.py --granularity fine
```

Classes: general-url, third-party-dataset, author-provided-dataset,
third-party-software, author-provided-software, project.

### coarse: four groups

```
python3 evaluate_classification.py --granularity coarse
```

Groups the two dataset classes into `dataset` and the two software classes into
`software`. Classes: general-url, dataset, software, project.

### binary: two groups

```
python3 evaluate_classification.py --granularity binary
```

Groups all five OADS classes (the dataset, software, and project classes) into
`OADS`, against `not-OADS` (general-url). Classes: OADS, not-OADS.

## Results

Micro-averaged F1 per pipeline.

### fine

| pipeline    | P     | R     | F1    |
|-------------|-------|-------|-------|
| Gpt-5-mini  | 0.877 | 0.789 | 0.831 |
| GROBID      | 0.879 | 0.685 | 0.770 |
| olmOCR      | 0.825 | 0.592 | 0.689 |
| Qwen        | 0.793 | 0.650 | 0.714 |
| PyMuPDF     | 0.770 | 0.518 | 0.620 |

The Gpt-5-mini micro F1 of 0.831 is the 83.1% weighted F1 reported in the paper.

### coarse (Gpt-5-mini)

| class       | P     | R     | F1    |
|-------------|-------|-------|-------|
| general-url | 0.911 | 0.799 | 0.852 |
| dataset     | 0.797 | 0.778 | 0.787 |
| software    | 0.886 | 0.833 | 0.859 |
| project     | 0.500 | 0.625 | 0.556 |
| MICRO       | 0.894 | 0.804 | 0.846 |

### binary (Gpt-5-mini)

| class    | P     | R     | F1    |
|----------|-------|-------|-------|
| OADS     | 0.875 | 0.835 | 0.855 |
| not-OADS | 0.911 | 0.799 | 0.852 |
| MICRO    | 0.900 | 0.810 | 0.853 |
