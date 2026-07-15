# sURL3k-VisURLX

Scholarly URL extraction from rendered PDF page images with a multimodal large
language model. VisURLX reads each page as an image and extracts every URL
together with its context, for URLs that appear in the body, in footnotes, and
in the reference list. This repository contains the extraction pipeline, the
evaluation script, and the data needed to reproduce the reported results.

## Repository layout

```
sURL3k-VisURLX/
├── README.md                     this file
├── requirements.txt              Python dependencies
├── src/
│   ├── run_pipeline.py           master runner over a directory of PDFs
│   ├── body_module.py           body module (single prompt)
│   ├── footnote_module.py       footnote module (detection + restoration)
│   └── reference_module.py       reference module (parse + detection + restoration)
├── evaluation/
│   ├── evaluate.py               scores predictions against the gold set
│   └── EVALUATION.md             scoring procedure
└── data/
    ├── gold/
    │   └── sURL3K_gold_merged.csv        gold benchmark (3,347 rows)
    └── predictions/
        └── GPT-5-mini.csv                merged pipeline output
```

## The three modules

VisURLX processes each URL location with its own module.

- **Body.** A single prompt reads the page image and returns each body URL with
  its target sentence and the neighbouring sentences.
- **Footnote.** A prompt chain. The first prompt detects URL-bearing footnotes
  on the bottom strip of the page. The second restores each footnote into its
  citing sentence and returns the sentence with its context.
- **Reference.** A prompt chain. It parses the reference list for URL-bearing
  entries, detects their in-text citation markers, and restores each reference
  into its citing sentence.

Every module writes one JSON file per input PDF into its own output subfolder.

## Requirements

- Python 3.9 or newer
- An OpenAI API key with access to the model used in the paper (`gpt-5-mini`)

Install the dependencies:

```bash
pip install -r requirements.txt
```

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

## Running the full pipeline

The master runner passes every PDF in the input directory to all three modules.

```bash
cd src
python run_pipeline.py --input /path/to/pdfs --output /path/to/out
```

This creates:

```
out/
├── body/         one <stem>_body_urls.json per PDF
├── footnote/     footnote results per PDF
└── reference/    reference results per PDF
```

### Options

| Option        | Default                    | Meaning                                                                 |
|---------------|----------------------------|-------------------------------------------------------------------------|
| `--input`     | required                   | Directory of input PDFs, searched recursively.                          |
| `--output`    | required                   | Output directory. Per-module subfolders are created inside it.          |
| `--dpi`       | each module's own default  | Overrides the primary page-rendering resolution for every module.       |
| `--workers`   | 1                          | Number of PDFs processed in parallel.                                   |
| `--modules`   | `body,footnote,reference`  | Comma-separated subset of modules to run.                               |
| `--api-key`   | `OPENAI_API_KEY` env var   | OpenAI API key.                                                         |

### Notes on `--dpi`

`--dpi` sets the resolution of the main page image each module reads. It does
not change the resolution of the small previous or next page strips that a
module renders only to complete a sentence that crosses a page boundary. Those
keep each module's own value, so overriding `--dpi` does not disturb the strip
resolutions used in the reported runs. To reproduce the paper exactly, omit
`--dpi` and let each module use its default.

### Notes on `--workers`

Concurrency comes from processing several PDFs at once; the three modules for a
single PDF run one after another. The number of API calls and the tokens per
call do not depend on `--workers`, so wall-clock time changes but total cost
does not. Higher values raise peak API load and the chance of hitting a rate
limit, so the default is 1. Raise it once you have confirmed your account's rate
limits.

## Running a single module

Each module can also run on its own. The command differs by module.

```bash
# body: reads INPUT_DIR / OUTPUT_DIR set at the top of the file,
#       or edit those two constants
python body_module.py

# footnote: flag-based
python footnote_module.py --input /path/to/pdfs --output /path/to/out/footnote

# reference: positional
python reference_module.py /path/to/pdfs /path/to/out/reference
```

The master runner is the recommended entry point; the standalone commands are
for inspecting one module in isolation.

## Evaluation

The evaluation script scores a prediction CSV against the gold benchmark and
reports precision, recall, and F1 for each URL location. See
`evaluation/EVALUATION.md` for the procedure and the exact command.

```bash
cd evaluation
python evaluate.py \
    --gold ../data/gold/sURL3K_gold_merged.csv \
    --pred ../data/predictions/GPT-5-mini.csv
```

## Data

- `data/gold/sURL3K_gold_merged.csv` is the manually annotated gold benchmark
  (3,347 URL instances). It is the target for evaluation.
- `data/predictions/GPT-5-mini.csv` is the merged VisURLX output that produced
  the reported scores.

## Prompts

The full text of every prompt is contained in the module source files under
`src/`. No prompt text is abbreviated.

## Cost

The pipeline calls a paid API. Cost scales with the number of pages and the
rendering DPI. Running the full benchmark incurs real charges on your own key.
Start with a small subset to estimate cost before running the full corpus.
