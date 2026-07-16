# sURL-3K

sURL-3K is a manually annotated collection of 3,376 scholarly URLs in the text
layer of 352 scholarly papers, along with their contexts. Each instance consists
of a URL, the document location of the URL, its context (the target sentence
where the URL is mentioned together with the preceding and trailing sentences),
and one of six OADS classes. For URLs that appear in footnotes or references
whose context is not in the proximity of the URLs, the footnote or reference
that carries the URL is restored into the corresponding sentence in the main
text.

## Corpus

The 352 papers are drawn from two scholarly PDF corpora that differ
systematically in layout, citation style, and URL placement conventions. Of the
352 papers, 192 are from arXiv and 160 are from CORD-19. The corpus comprises 238
journal articles, 11 conference proceedings, and 103 preprints, spanning 11
disciplines published from 2005 to 2020. Papers were selected so that the corpus
exhibits diverse layouts and reference styles, single-column and two-column
pages, numeric and author-year citation styles, and a significant fraction of
URLs in footnotes.

## Source PDFs

The 352 source PDFs are available at
<https://figshare.com/s/bc67b15c4c37395fecbd>. Each file is named by its paper
identifier and matches the `paper_id` field in `sURL-3K.csv`.

## Files

- `sURL-3K.csv` is the manually annotated gold benchmark. Each row is one URL
  instance with its context, document location, and OADS class.
- `accepted_set.csv` holds the accepted target sentences for references cited
  more than once. It follows the same schema as `sURL-3K.csv`.
- `surl3k_annotation_guidelines.pdf` (under `annotation-guideline/`) is the
  annotation guideline.

## Schema

`sURL-3K.csv` has the following columns.

| Column | Description |
| --- | --- |
| `paper_id` | Identifier of the source paper, an arXiv identifier for arXiv papers or a PMC identifier for CORD-19 papers. |
| `preceding` | The sentence preceding the target sentence in the same paragraph. Empty when the target is the first sentence of its paragraph. |
| `target` | The target sentence that carries the URL. For a footnote or reference URL, the footnote or reference entry is restored into the citing sentence by replacing its marker. |
| `trailing` | The sentence following the target sentence in the same paragraph. Empty when the target is the last sentence of its paragraph. |
| `url` | The URL, transcribed verbatim, with any URL wrapped across lines, columns, or pages rejoined into a single string. |
| `Label` | The OADS class of the URL, one of the six classes below. |
| `location` | The document location of the URL, one of `body`, `footnote`, or `reference`. |

## URL classes

Each URL is labeled with one of six classes from an established OADS labeling
scheme. Five are OADS classes: a `third-party-dataset` or
`author-provided-dataset` for a dataset provided by others or by the paper's
authors, a `third-party-software` or `author-provided-software` for software
provided by others or by the authors, and a `project` for a site hosting both
data and software. The sixth, `general-url`, is the non-OADS class.

## Document locations

A paper is partitioned into author content and boilerplate. The author content
is divided into three document locations. The body covers the abstract, all
paper sections including the Acknowledgements and Appendices, display text such
as titles and section headings, figure captions, table captions, table content,
and equations. The footnote location is at the bottom of a page. The reference
location comprises the entries of the bibliography at the end of the paper. URLs
in the boilerplate location, such as publisher metadata and citation-service
links, are not labeled.

## Distribution

The 3,376 instances are distributed across the two axes below.

| URL class | arXiv | CORD-19 | Total |
| --- | --- | --- | --- |
| general-url | 587 | 1,685 | 2,272 |
| third-party-dataset | 104 | 164 | 268 |
| author-provided-dataset | 18 | 27 | 45 |
| third-party-software | 392 | 150 | 542 |
| author-provided-software | 220 | 13 | 233 |
| project | 12 | 4 | 16 |

| Document location | arXiv | CORD-19 | Total |
| --- | --- | --- | --- |
| body | 286 | 578 | 864 |
| footnote | 653 | 15 | 668 |
| reference | 394 | 1,450 | 1,844 |

## Annotation

For each paper an annotator visually inspected each PDF page and located every
URL in the author content. URLs appearing as visible text were annotated, and
URLs present only as clickable hyperlinks in the annotation layer were excluded.
A URL was annotated in any surface form, whether it carried a scheme or was a
scheme-less domain or domain-path string. Bare scholarly identifiers such as DOI
strings and arXiv identifiers were excluded unless written as a complete URL.

Footnote and reference URLs are linked to their context through a marker in the
body. The full footnote or reference entry that carries the URL is restored into
the citing sentence by replacing the marker with it. Restoration follows an
instance-level rule. When a sentence carries several URL-bearing footnotes or
cites several URL-bearing references, each marker yields a separate instance, so
the same sentence may recur with a different restored URL. For a reference cited
more than once, sURL-3K keeps one occurrence as the representative target
sentence and retains the restored target sentences of the remaining occurrences
in `accepted_set.csv`.

To assess annotation reliability, the same annotator relabeled the URL class of
every instance three months after the first pass, without access to the earlier
labels. The two passes agree for 97.4% of instances (κ = 0.95). The annotation
was additionally cross-checked against an independent MLLM annotator on 15.8% of
sURL-3K.
