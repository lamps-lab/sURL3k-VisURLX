# End-to-End Contextualized URL Extraction from Scholarly Papers

This repository accompanies the paper *End-to-End Contextualized URL Extraction
from Scholarly Papers*. It contains the extraction pipelines, the sURL-3K
dataset, the annotation guideline, and the evaluation scripts.

## Repository layout

```
├── README.md
├── dataset/
│   ├── sURL-3K.csv                 gold benchmark
│   └── accepted_set.csv            accepted target sentences for multiply-cited references
├── annotation-guideline/
│   └── surl3k_annotation_guidelines.pdf
├── pipeline-scripts/               extraction pipelines, one folder each with its own README
│   ├── VisURLX-Gpt-5-mini/
│   ├── GROBID/
│   ├── PyMuPDF-Layout/
│   └── olmOCR/
├── evaluation-scripts/             evaluators (see EVALUATION.md)
│   ├── evaluate_urls.py
│   ├── evaluate_heuristic.py
│   └── evaluate-VisURLX.py
├── URL-evaluation/                 URL extraction output from each pipeline
├── heuristic-evaluation/           heuristic baseline extraction output
├── VisURLX-evaluation/             VisURLX extraction output
├── URL-classification-evaluation/  OADS-URL classification evaluation
└── EVALUATION.md                   evaluation procedure and commands
```

## Pipelines

Each pipeline has its own README under `pipeline-scripts/`. VisURLX reads each
page as an image and calls a multimodal model. The three heuristic baselines
convert the PDF first (GROBID to TEI, PyMuPDF-Layout to text and JSON, olmOCR to
Markdown) and extract URLs from the converted output.

## Dataset

- `dataset/sURL-3K.csv` is the manually annotated gold benchmark.
- `dataset/accepted_set.csv` holds the accepted target sentences for references
  cited more than once.
- `annotation-guideline/surl3k_annotation_guidelines.pdf` is the annotation
  guideline.

## Evaluation

See `EVALUATION.md` for the URL extraction, target sentence extraction, and
OADS-URL classification procedures and commands.
