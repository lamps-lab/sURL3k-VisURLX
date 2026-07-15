# PyMuPDF-Layout URL extraction baseline

PyMuPDF-Layout baseline for extracting scholarly URLs and their citing context.
Each PDF is converted with PyMuPDF-Layout to per-page JSON and a layout text
file, then three extractors produce URLs with paragraph-local context:

- `body`: URLs in running prose. Reads the layout text.
- `footnote`: URLs in footnotes, restored into the citing sentence at the
  marker. Reads the per-page JSON.
- `reference`: URLs from bibliography entries, matched to each in-text citation
  and restored inline at the marker. Reads the per-page JSON.

The body extractor consumes text; the footnote and reference extractors consume
JSON. The converter produces both so all three run from the same conversion.

## Layout

```
run.py                one command: PDF -> JSON + text -> extraction
pymupdf_client.py     PDF -> per-page JSON and layout text
extract_body.py       layout text -> body URLs
extract_footnote.py   per-page JSON -> footnote URLs
extract_reference.py  per-page JSON -> reference URLs
```

## Prerequisites

- Python 3.9+
- `pip install -r requirements.txt` (PyMuPDF4LLM; recent versions bundle the
  layout feature, so no separate package is needed)

The extractors themselves need only the standard library.

## Run the whole pipeline

```bash
python run.py --input ./pdfs --output ./out
```

Conversion output goes to `out/workspace/json/<stem>/<stem>_NN.json` and
`out/workspace/text/<stem>.layout.txt`. Extractor results go to `out/body/`,
`out/footnote/`, and `out/reference/`.

Options:

```bash
python run.py --input ./pdfs --output ./out --modules body reference
python run.py --input ./ws   --output ./out --from-converted
python run.py --input ./pdfs --output ./out --workspace ./ws --workers 8
```

`--from-converted` treats `--input` as a workspace that already has `json/` and
`text/` subfolders and skips conversion.

## Run individual stages

Convert PDFs (produces both formats):

```bash
python pymupdf_client.py --input ./pdfs --json-out ./ws/json --text-out ./ws/text --workers 8
```

Run each extractor on its own input format:

```bash
python extract_body.py      -i ./ws/text -o ./out/body
python extract_footnote.py  -i ./ws/json -o ./out/footnote
python extract_reference.py -i ./ws/json -o ./out/reference
```

Conversion and body extraction are resumable: a PDF with complete outputs, or a
text file already extracted, is skipped on a re-run.

## Input formats

The converter joins pages in the text file with `=== PAGE N ===` markers, which
the body extractor uses to split pages. The per-page JSON is the PyMuPDF4LLM
layout structure (`pages` with `boxes`, `boxclass`, `textlines`/`spans`, and
`fulltext`), which the footnote and reference extractors read. The layout text
is rendered with headers and footers dropped (`header=False, footer=False`); the
JSON keeps all content, since PyMuPDF4LLM does not drop headers/footers for JSON.

## Output

`body/`: `<stem>_body_urls.json` per paper and combined `_ALL.url_footnotes.jsonl`.

`footnote/`: `<stem>_footnotes.json`, `<stem>_sentences.json`, combined
`_ALL.url_footnotes.jsonl`, and `_SUMMARY.json`.

`reference/`: `<stem>_url_reference_citations.json`,
`<stem>_references_with_urls.json`, combined `_ALL.citations.jsonl`, and
`_SUMMARY.json`.
