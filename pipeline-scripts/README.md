# olmOCR URL extraction baseline

olmOCR-based baseline for extracting scholarly URLs and their citing context.
Each PDF is converted to Markdown with olmOCR, then three extractors read the
Markdown and write URLs with paragraph-local context:

- `body`: URLs in running prose, excluding references, footnotes, and
  headers/footers/license boilerplate.
- `footnote`: URLs in footnotes, restored into the body sentence that carries
  the footnote marker.
- `reference`: URLs from bibliography entries, matched to each in-text citation
  and restored inline at the marker.

Conversion needs olmOCR and a GPU. The three extractors are pure Python standard
library and run anywhere.

## Layout

```
run.py                one command: PDF -> Markdown -> extraction
olmocr_client.py      PDF -> Markdown via the olmOCR CLI
extract_body.py       Markdown -> body URLs
extract_footnote.py   Markdown -> footnote URLs
extract_reference.py  Markdown -> reference URLs
run_olmocr.sh         SLURM job script for the conversion step
```

## Prerequisites

- Conversion: olmOCR installed (see its documentation), a CUDA GPU, and poppler
  (`pdftoppm`). olmOCR pulls in its own model and dependencies.
- Extraction: Python 3.9+, standard library only. No packages to install.

## Run the whole pipeline

`run.py` converts the PDFs and runs all three extractors:

```bash
python run.py --input ./pdfs --output ./out
```

Markdown goes to `out/olmocr_workspace/markdown/`, and each extractor writes
`out/body.{csv,jsonl}`, `out/footnote.{csv,jsonl}`, `out/reference.{csv,jsonl}`
(plus `out/reference_summary.json`).

Options:

```bash
python run.py --input ./pdfs --output ./out --modules body reference
python run.py --input ./md   --output ./out --from-markdown
python run.py --input ./pdfs --output ./out --workspace ./ws --chunk-size 32 --workers 2
```

`--from-markdown` skips conversion and treats `--input` as a directory of `.md`.
The runner shells out to the four scripts below, which can also be used on their
own.

## Convert PDFs to Markdown

```bash
python olmocr_client.py --input_dir ./pdfs --workspace ./olmocr_workspace
```

Markdown is written to `<workspace>/markdown/<stem>.md`. The step is resumable:
PDFs whose Markdown already exists are skipped unless `--overwrite` is set. A
per-chunk conversion log is written to `--log_csv`.

Options include `--chunk_size`, `--workers`, `--pages_per_group`, and, for a
remote OpenAI-compatible backend, `--server` / `--model` / `--api_key`.

On a cluster, `run_olmocr.sh` is an sbatch wrapper that activates the olmOCR
conda environment, sets the Hugging Face cache, and runs the conversion on one
GPU.

## Extract URLs

Each extractor takes `--input` (a directory of `.md`, a `.zip` of `.md`, or a
single `.md`), a required `--out_csv`, and an optional `--out_jsonl`:

```bash
python extract_body.py      --input ./olmocr_workspace/markdown --out_csv body.csv       --out_jsonl body.jsonl
python extract_footnote.py  --input ./olmocr_workspace/markdown --out_csv footnote.csv   --out_jsonl footnote.jsonl
python extract_reference.py --input ./olmocr_workspace/markdown --out_csv reference.csv  --out_jsonl reference.jsonl
```

`extract_footnote.py` also accepts `--unmatched_jsonl` (footnotes with
a URL but no matched body marker). `extract_reference.py` also accepts
`--summary_json` (per-corpus counts).

## Output

Every row carries `paper_id`, `file`, a `location` column
(`body` / `footnote` / `reference`), `url_output`, and paragraph-local context
(`preceding_output`, `target_output`, `trailing_output`, `at_paragraph_start`,
`at_paragraph_end`). The three CSVs share these columns, so they can be
concatenated directly.

Full columns per extractor:

```
body:      paper_id, file, location, url_output, preceding_output, target_output,
           trailing_output, at_paragraph_start, at_paragraph_end, source,
           start_char, end_char

footnote:  paper_id, file, location, footnote_marker, url_output, footnote_text,
           preceding_output, target_output, original_sentence, trailing_output,
           at_paragraph_start, at_paragraph_end, marker_surface, marker_start_char,
           footnote_start_char, kind

reference: paper_id, file, location, citation_marker, citation_style, url_output,
           reference_entry, preceding_output, target_output, original_sentence,
           trailing_output, at_paragraph_start, at_paragraph_end,
           inside_parenthetical, visible_group, citation_surface,
           citation_start_char, reference_index
```

## Notes

- The extractors are standalone; run the three separately. Only conversion needs
  a GPU.
- `target_output` is the citing sentence: the body sentence for `body`, and the
  restored sentence (footnote or reference text inserted at the marker) for
  `footnote` and `reference`.
