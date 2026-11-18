# Corpus extractor

Turns the company reports in `corpus/natural/` (PDF, plus one HTML page) into structured,
searchable text for the RAG study. Python 3 with PyMuPDF (`fitz`, tested with 1.27.1);
standard library otherwise.

## Usage

```
python tools/extract/extract.py --manifest corpus/natural/manifest.csv \
       --out corpus/extracted --files "snapp\snapp_annual-report_1402_fa.pdf" [...] [--force] [--pages 10-20]
```

- `--files` takes `local_path` values from the manifest. Rows whose `notes` flag OCR / image-only /
  garbled text are out of scope and refused.
- A file is skipped when its output folder already has `extract_status.json` with the same source
  sha256, extractor version and page range. `--force` re-extracts.
- Each file is written to a hidden temporary folder and renamed into place only when complete.
- `--pages` replaces the whole output folder with that range. It cannot build one file in parts,
  so run files whole (the longest file takes under a minute).
- One JSON line per file goes to stdout: status, verdict, page count, seconds, first reasons.

## Outputs, per source file: `corpus/extracted/<company>/<file stem>/`

| file | content |
|---|---|
| `document.json` | manifest row, source sha256, metadata, outline, and per page: blocks (`heading` with level, `paragraph`, `list_item`, `caption`, `header_footer`, `table_ref`, `image_ref`) in reading order with `text_raw`, `text_norm`, bbox, font size, bold, direction |
| `document.md` | reading-order Markdown: `<!-- page N -->` markers, headings, lists, tables as pipe tables, image links; running headers/footers left out |
| `tables/pNNN_tK.csv` / `.json` | one per detected table; cells hold `text_norm` (JSON also `text_raw`, bbox, colspan); RTL tables are written with the rightmost column first |
| `images/pNNN_iK.png|jpg` | embedded rasters of at least 150 px, de-duplicated by content |
| `quality.json` | per-page and document checks, the verdict (`ok` / `suspect` / `fail`) with reasons, and notes on every repair applied |
| `extract_status.json` | completion marker: sha256, extractor version, page range, verdict, seconds |

## How Persian text is rebuilt

PDFs store Persian either in logical order or in visual order (often as presentation forms),
and some producers mix both. The extractor never trusts stream order for direction:

1. Glyphs come from `page.get_texttrace()` in content-stream order, one entry per glyph (a
   lam-alef ligature stays one entry). MuPDF's `rawdict` at the same glyph position corrects
   what texttrace gets wrong: a space glyph that a font maps to ZWNJ, InDesign ZWNJ written as
   an ActualText space, glyphs texttrace cannot map, and InDesign small caps.
2. Glyphs are grouped into lines by baseline; a glyph painted twice (fake bold, fill+stroke,
   offset shadow) is kept once.
3. Tokens (Arabic word, Latin word, number, space, neutral) are cut from the stream. Arabic
   letters inside a token are ordered by glyph geometry. A token whose glyphs run one way is
   kept or reversed; a word written as several runs in mixed order is sorted right to left.
   Combining marks stay with the letter they are drawn on.
4. Tokens are sorted left to right by position, and an inverse bidi pass on tokens gives
   logical order (RTL lines reversed, embedded LTR runs kept left to right).
5. Brackets and guillemets are mirrored where needed. Whether a font codes a bracket glyph by
   its drawn shape or by its logical meaning is measured per glyph over the whole document,
   from which side of its content the glyph sits on.
6. Lines become blocks and blocks are put in reading order (recursive XY cut, RTL-aware).
   Running headers and footers are found by repetition across pages.

Repairs that depend on the document are gated by evidence from the same document, and each
one is recorded in `quality.json` notes:

- glyphs without a Unicode mapping: recovered from sibling font subsets that map the same glyph
  id, or from the document's own vocabulary plus `fa_lexicon.txt` (the letter that turns the
  glyph's words into known words, with frequency weighting and repeated passes);
- lam-alef ligatures mapped to a bare alef: restored per font when that yields more known
  words, then undone per word where the word without the lam is the known one;
- spurious narrow letter glyphs: dropped per font when that matches more MuPDF plain-text words;
- glyphs mapped to a letter but used as justification kashida: read as tatweel;
- harakat that only come from a broken letter+mark ToUnicode entry: dropped;
- fonts whose codes are symbol garbage: dropped. In Canva-style fonts only the ActualText
  phrases are kept, with overlapping phrase tails trimmed;
- words set as separate text objects without space glyphs: spaced at visible gaps;
- table text that no table cell claimed stays in the page text.

## Normalization (`text_norm`, `fa_text.normalize`)

- Unicode NFKC: presentation forms become base letters, compatibility digits and ligatures are
  unfolded.
- Arabic yeh (ي) and alef maksura (ى) become Persian yeh (ی); Arabic kaf (ك) becomes keheh (ک);
  heh doachashmee (ھ) and heh goal (ہ) become heh (ه); yeh barree (ے) becomes ی.
- Tatweel is removed.
- ZWNJ-like joiners between two Arabic letters (ZWNJ, ZWSP, RLM, legacy NOT SIGN) become one
  ZWNJ. Other joiners, soft hyphens and bidi controls are removed, along with control and
  private-use characters.
- Whitespace runs collapse to one space.
- Persian and Arabic-Indic digits become ASCII digits. `٫` becomes `.`, `٬` becomes `,`, `٪`
  becomes `%`. Between digits, U+060D becomes `.` and U+201A/U+201E become `,`.
- A minus stored after a percentage (`2.3-%`, `%13-`) becomes `-2.3%` / `-13%`.
- Kept as in the source: the Arabic comma `،` used as a thousands separator, the Persian slash
  decimal (`33/86`), harakat, and a percent sign placed before the number.

`text_raw` keeps the extracted characters before normalization.

## Files

| file | role |
|---|---|
| `extract.py` | CLI, manifest handling, idempotence, atomic output, HTML path |
| `pdf_extract.py` | glyph collection, per-document repairs, lines, tables, images |
| `fa_text.py` | character classes, tokens, geometric ordering, inverse bidi, bracket mirroring, normalization, order tests |
| `assemble.py` | headers/footers, block types, headings, reading order, Markdown, quality checks |
| `html_extract.py` | the HTML report (DOM order is logical order) |
| `fa_lexicon.txt`, `build_lexicon.py` | Persian word list used only to confirm glyph and lam-alef repairs and for the known-word share in quality.json |

## Quality checks (`quality.json`)

Per page: character counts against MuPDF plain text, function words against their reversed
forms (و در به از که این ... against زا هب هک ...), swapped common bigrams, unmapped and garbage
glyphs, ASCII symbol garbage, ZWNJ and harakat counts against plain text, and empty or blank
pages. Per document it adds: the manifest title, the company name and month names found
verbatim; «گزارش سالانه» in normal against swapped order; function words compared with MuPDF
plain text; function words glued by ZWNJ; the share of known Persian words; and unmapped glyphs
spread over pages. Pages that render blank are not counted as missing text.

Persian word order is judged only by these text checks, never by looking at a rendered page.
Digits and Latin text can be checked against a render.

## Known limits

- No OCR: image-only pages and outlined text yield nothing (verdict `fail` when most pages are
  like that).
- Vector charts are not digitised. Their labels come out as loose lines and can run together
  when separate labels share a baseline.
- Tables need ruling lines (`find_tables`, lines strategy). Borderless statements come out as
  one paragraph per column group, and grids drawn over charts or infographics can be taken for
  tables.
- Reading order is geometric, so side-by-side boxes whose lines share baselines can interleave.
- Text hidden under later fills, text set glyph by glyph on curves, and fonts with no usable
  mapping and no glyph names (e.g. Melkradar's IRANYekanMobileFN) cannot be recovered.
- Glyphs that neither a sibling font nor the vocabulary can decide stay dropped, so those words
  miss a letter. `quality.json` counts them.
- Source-side typography is kept: missing or misplaced ZWNJ, spaces one letter off, a ZWNJ glyph
  used as the word space, and a wrong ToUnicode that maps letters to other letters.
