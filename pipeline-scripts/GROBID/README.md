# GROBID URL extraction baseline

GROBID-based baseline for extracting URLs and their citing context from
scholarly PDFs. Each PDF is converted to TEI XML with GROBID, then three
extractors run over the TEI:

- `body`: URLs in running text, the abstract, and figure/table captions. Each
  item carries the target sentence, the preceding and following sentence in the
  same paragraph.
- `footnote`: URLs in footnotes, resolved to the body marker and spliced into
  the citing sentence.
- `reference`: URLs from bibliography entries, resolved to the in-text citation
  and spliced into the citing sentence.

The full pipeline (PDF to TEI to extraction) runs from one command. Each
extractor also runs standalone against existing TEI.

## Layout

```
run.py                 pipeline entry point: PDF -> TEI -> extraction
grobid_client.py       PDF -> TEI via the GROBID service
extract_body.py        TEI -> body URLs
extract_footnote.py    TEI -> footnote URLs
extract_reference.py   TEI -> reference URLs
setup_grobid.sh        build and start a local GROBID server
requirements.txt
```

## Prerequisites

- Python 3.9+
- A GROBID server reachable over HTTP (developed against 0.8.1)
- `pip install -r requirements.txt`

## GROBID server

The pipeline needs a GROBID server on port 8070. Via Docker (the 0.8.1 CRF
image is CPU-only and runs on x86_64 and arm64):

```bash
docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.1
```

To build from source instead (as used in Colab):

```bash
bash setup_grobid.sh
```

Verify:

```bash
curl http://localhost:8070/api/isalive   # -> true
```

The client requests sentence segmentation (`segmentSentences=1`); the extractors
rely on the resulting `<s>` elements for sentence context. TEI produced by other
means must be sentence-segmented.

## Usage

Full pipeline over a directory of PDFs:

```bash
python run.py --input path/to/pdfs --output path/to/output
```

TEI is written to `output/tei/` and extractor results to `output/body/`,
`output/footnote/`, and `output/reference/`.

Options:

```bash
python run.py --input pdfs --output out --modules body reference
python run.py --input pdfs --output out --grobid-url http://host:8070/api/processFulltextDocument
python run.py --from-tei --input path/to/tei --output out   # skip conversion
```

Individual stages:

```bash
python grobid_client.py     -i path/to/pdfs -o path/to/tei
python extract_body.py      -i path/to/tei  -o out/body
python extract_footnote.py  -i path/to/tei  -o out/footnote
python extract_reference.py -i path/to/tei  -o out/reference
```

Conversion is resumable: PDFs that already have a valid TEI are skipped, so a
re-run only processes what failed.

## Output

`body/`: per-paper `<paper>_body_urls.json` and combined `_ALL.body_urls.jsonl`.

```
url_printed, target_sentence, preceding_sentence, trailing_sentence,
at_paragraph_start, at_paragraph_end, url_lines_joined, url_span_pages,
pdf_file, page, location
```

`location` is `body`, `abstract`, or `caption`. Filter on it to score the three
separately or together.

`footnote/`: per-paper `<paper>_footnotes.json` and combined
`_ALL.url_footnotes.jsonl`.

```
original_sentence, restored_sentence, footnote_marker, footnote_content,
url, preceding_sentence, trailing_sentence, pdf_file, page
```

`restored_sentence` is the citing sentence with the footnote text reinserted at
the marker.

`reference/`: per-corpus `<corpus>_restorations.{csv,jsonl}` and combined
`all_restorations.jsonl`.

```
corpus, paper_id, reference_id, citation_marker, url, reference_text,
original_sentence, restored_sentence, preceding_sentence, trailing_sentence,
at_paragraph_start, at_paragraph_end
```

## Notes

- All three extractors consume the same TEI; conversion runs once.
- Context sentences are paragraph-local and never cross a `<p>` boundary.
- arXiv identifiers are normalised to preserve a trailing zero
  (`1102.093` -> `1102.0930`) during file matching.