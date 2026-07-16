# Evaluation

Reproduces the URL extraction, target sentence extraction, and OADS-URL
classification results reported in the paper. All pipelines are evaluated against
sURL-3K (`dataset/sURL-3K.csv`).

## Layout

```
dataset/
├── sURL-3K.csv                 gold benchmark
└── accepted_set.csv            accepted target sentences for multiply-cited references
evaluation-scripts/
├── evaluate_urls.py            URL extraction evaluation
├── evaluate_heuristic.py       heuristic target sentence extraction evaluation
└── evaluate-VisURLX.py         VisURLX target sentence extraction evaluation
URL-evaluation/                 URL extraction output from each pipeline
├── GROBID.csv
├── GPT-5-mini.csv
├── PyMuPDF-Layout.csv
├── Qwen.csv
└── olmOCR.csv
heuristic-evaluation/           heuristic baseline extraction output
├── GROBID.csv
├── PyMuPDF-Layout.csv
└── olmOCR.csv
VisURLX-evaluation/             VisURLX extraction output
├── Gpt-5-mini.csv
└── Qwen-32b.csv
URL-classification-evaluation/
└── classification-evaluation/
    ├── evaluate_classification.py
    ├── exact_url_evaluator.py
    ├── {GROBID,Gpt-5-mini,PyMuPDF,Qwen,olmOCR}_all_predictions.csv
    └── {arXiv,pmc}-{body,footnote,reference}-gold.csv
```

## Correctness criteria

An extracted target sentence counts as correct when both hold:

1. The URL matches the gold URL exactly after normalization: Unicode NFKC and
   ligature folding, lowercasing, internal-whitespace removal, scheme and
   leading `www.` stripping, and trailing `/` and `.` trimming, applied
   identically to both URLs.
2. The target sentence reaches a fuzzy similarity of at least 80 against the gold
   target, scored by the character-level rapidfuzz `fuzz.ratio` on a 0 to 100
   scale.

For records that satisfy both criteria, the preceding and trailing sentences are
scored separately, each against its gold counterpart at the same threshold.

For a reference cited more than once, sURL-3K keeps one occurrence as the
representative target sentence and retains the restored target sentences of the
remaining occurrences as an accepted set (`dataset/accepted_set.csv`). A predicted
record is matched first against the representative target sentence; if it does not
match, it is matched against the accepted set. A match in either case counts as a
true positive, and the reference is credited once regardless of how many
occurrences are extracted.

## Requirements

- Python 3.9 or newer
- `pip install pandas rapidfuzz` (`evaluate_urls.py` uses the standard library
  only)

## URL extraction

`evaluate_urls.py` scores each predicted URL against the gold URLs of the same
paper under Criterion 1, aggregated over all document locations. The prediction
files hold `paper_id,url`.

```bash
python evaluation-scripts/evaluate_urls.py \
  --gold dataset/sURL-3K.csv \
  --sets URL-evaluation/GROBID URL-evaluation/GPT-5-mini \
         URL-evaluation/PyMuPDF-Layout URL-evaluation/Qwen URL-evaluation/olmOCR
```

## Target sentence extraction

Scores each URL together with its target sentence under both criteria, reported
by document location (body, footnote, reference) and aggregated end to end.

Heuristic pipelines (GROBID, PyMuPDF-Layout, olmOCR), one run per pipeline:

```bash
python evaluation-scripts/evaluate_heuristic.py \
  --pred heuristic-evaluation/GROBID.csv \
  --gold dataset/sURL-3K.csv
```

VisURLX pipelines (GPT-5-mini, Qwen3-VL-32B). The reference location is matched
against the accepted set as described above:

```bash
python evaluation-scripts/evaluate-VisURLX.py \
  --output VisURLX-evaluation/Gpt-5-mini.csv \
  --gold dataset/sURL-3K.csv \
  --accepted_set dataset/accepted_set.csv
```
```bash
python evaluation-scripts/evaluate-VisURLX.py \
  --output VisURLX-evaluation/Qwen-32b.csv \
  --gold dataset/sURL-3K.csv \
  --accepted_set dataset/accepted_set.csv
```

`--threshold` sets the fuzzy cutoff (default 80). `--dump_dir DIR` writes the
per-URL stage-two, extra, and false-positive breakdowns for the reference
location.

## OADS-URL classification

`evaluate_classification.py` scores the predicted OADS class of each extracted
record against the gold `test` split, using `exact_url_evaluator.py` for URL and
paper-id normalization. It reads its prediction and gold CSVs from the current
directory.

```bash
cd URL-classification-evaluation/classification-evaluation
python evaluate_classification.py                       # variant 1, fine
python evaluate_classification.py --granularity binary  # OADS vs not-OADS
python evaluate_classification.py --variant 3 --granularity coarse
```

- `--variant {1,2,3,4}` controls how stage-two matches and the counting key are
  handled.
- `--granularity {fine,coarse,binary}` scores the six OADS classes, four groups,
  or OADS against not-OADS.
