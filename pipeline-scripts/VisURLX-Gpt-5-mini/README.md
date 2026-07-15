# VisURLX

MLLM pipeline for extracting scholarly URLs and their citing context from PDFs.
Three modules run over each PDF, each targeting a different document location:

- `body`: URLs in running text, with the target sentence and the preceding and
  following sentence.
- `footnote`: URLs in footnotes, restored into the citing sentence at the
  marker.
- `reference`: URLs from bibliography entries, restored into the citing sentence
  with the reference entry inline.

Each module renders page images and calls the OpenAI API (`gpt-5-mini`).
`run_pipeline.py` drives all three over a directory of PDFs.

## Layout

```
run_pipeline.py      entry point: runs the modules over a directory of PDFs
body_module.py       body URLs
footnote_module.py   footnote URLs
reference_module.py  reference URLs
flatten_to_csv.py    combine the per-PDF JSON output into one CSV
```

## Prerequisites

- Python 3.9+
- `pip install pymupdf openai tqdm`
- An OpenAI API key with access to `gpt-5-mini`

Set the key once:

```bash
export OPENAI_API_KEY=sk-...
```

## Usage

Run all three modules over a directory of PDFs (searched recursively):

```bash
python run_pipeline.py --input ./pdfs --output ./out
```

Results are written to per-module subfolders: `out/body/`, `out/footnote/`,
`out/reference/`.

Options:

```bash
python run_pipeline.py --input ./pdfs --output ./out --dpi 300 --workers 4
python run_pipeline.py --input ./pdfs --output ./out --modules body,footnote
python run_pipeline.py --input ./pdfs --output ./out --api-key sk-...
```

- `--dpi` overrides the primary page-rendering DPI for every module (default:
  each module's own value, already 300). Page-strip DPI is left untouched.
- `--workers` sets how many PDFs are processed in parallel (default 1). Higher
  values raise peak API load and rate-limit risk. Within one PDF the three
  modules run in sequence.
- `--modules` selects a subset (default `body,footnote,reference`).

Each module also exposes `process_pdf(client, pdf_path, out_dir)` for
programmatic use, and has its own `main()` for running a single module
standalone.

## Configuration

Models and the constants below are set at the top of each module. Only the
primary DPI is exposed on the command line (`--dpi`); the rest are edited in
the source.


**Rendering DPI.** Primary page images render at 300 across all modules
(`PAGE_DPI`; `DETECT_DPI`/`RESTORE_DPI`; `EXTRACT_DPI`/`DETECT_DPI`/`RESTORE_DPI`).
`--dpi` overrides these. Cross-page strips use `TOP_DPI`: 300 in body and
footnote, 150 in reference; `TOP_DPI` is not affected by `--dpi`.

**Cross-page crop.** When a URL or sentence spans a page break, a strip of the
neighbouring page is included:

| Constant         | body | footnote | reference | Meaning                                   |
|------------------|------|----------|-----------|-------------------------------------------|
| `PREV_PAGE_FRAC` | 0.5  | 0.5      | 0.5       | fraction of the previous page bottom       |
| `TOP_FRAC`       | 0.5  | 0.5      | 0.5       | fraction of the next page top              |
| `MAX_NEXT_PAGES` | 1    | 1        | 1         | number of next-page strips included        |

Footnote detection also crops the bottom of each page from halfway down
(`DETECT_START_FRAC = 0.5`).

**Other constants.** Reference batches reference-list pages
(`REF_PAGES_PER_CALL = 6`) and uses `MIXED_PAGE_DETECT_BUFFER = 0.02`. Body
paces API calls with `DELAY_SECONDS = 0.2`.

## Output

`run_pipeline.py` writes one JSON per PDF into each module's subfolder:

```
out/body/<stem>_body_urls.json
out/footnote/<stem>_footnotes.json
out/footnote/<stem>_sentences.json
out/reference/<stem>_references_with_urls.json
out/reference/<stem>_url_reference_citations.json
out/reference/<stem>_url_reference_sentences.json
```

Running a module standalone through its own `main()` additionally writes the
combined files (`_ALL.url_footnotes.jsonl`, `_ALL.citations.jsonl`,
`_SUMMARY.json`).

## Flatten to CSV

`flatten_to_csv.py` combines the per-PDF JSON into one CSV. It reads the
`body/`, `footnote/`, and `reference/` subfolders of a `run_pipeline` output
directory and writes one row per URL occurrence, tagged with the module that
produced it.

```bash
python flatten_to_csv.py --input ./out --output ./out/all_urls.csv
```

Columns: `paper_id`, `location` (`body` / `footnote` / `reference`), `target`,
`preceding`, `trailing`, `url`. `target` is the restored citing sentence for
footnote and reference and the citing sentence for body. Multiple URLs on one
item are joined with ` | `. Uses the standard library only (no pandas).