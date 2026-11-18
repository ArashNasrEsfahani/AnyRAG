# Shared retrieval and evidence foundation (2026-09-24)

For the LEX3 and AGENTIC5 builds and the HYBRID control. Astra's items 2 (shared evidence construction) and 4
(shared precomputed hybrid index), `work/analysis/UPGRADE_PLAN.md`. Existing methods (B2, B3, LEX2, AGENTIC3...) are
untouched; everything here is new modules and a new approach name.

## 1. `ragbench.retrieval.hybrid`

```python
from ragbench.retrieval.hybrid import HybridIndex, Hit, rrf_fuse, RRF_K

hx = HybridIndex.load(corpus_set, cfg, client=None)     # raises FileNotFoundError/RuntimeError if absent or stale
hx = HybridIndex.open(corpus_set, cfg, client=None, embed=False)   # load, else build (embed=False: cache only)
hx = HybridIndex.build(corpus_set, cfg, client=None, embed=True)   # index time only

hits = hx.search(query_text, doc_ids=None, k=40, channels=("bm25", "dense"), query_vector=None,
                 depth=None, client=None)                # -> HitList (a list of Hit) with hits.diagnostics
```

* **Same chunks as B2/B3**: `baselines.build_chunks` with `config.yaml chunking` and `baselines.DEFAULTS["chunker"]`;
  index/embedding text `"<doc title> > <section path>\n<chunk>"`; B2's tokenizer and bm25s parameters.
* **Same vectors as B3**: B3's shared cache `<index_root>/_embed_cache/embeddings.sqlite`, B3's keys
  (`embed_text_key(cfg, text)` == `VectorRetrieval._text_key`), B3's normalisation (bit-identical). Existing B3 vectors
  are cache hits.
* **Persisted** under `<index_root>/_hybrid/<set>-<fp8>/` (`chunks.jsonl`, `bm25/`, `embeddings.npy`, `has_vec.npy`,
  `hybrid_meta.json`, `vector_meta.json`, `prebuild_log.jsonl`). Fingerprint = corpus fingerprint + chunking + chunker +
  tokenizer + bm25 + embed model/digest/truncate/num_ctx (`hybrid.signature(cfg)`).
* **`Hit`**: `doc_id, section_id, unit_ids, text` (verbatim chunk), `span` ([start, end) in `Document.text`),
  `ranks={"bm25": r|None, "dense": r|None}` (1-based, within the scope), `scores={"bm25": s|None, "dense": cos|None}`,
  `rrf`, plus `chunk` (the candidate's identity), `section_ids, partial_unit_ids, mode`. `hit.to_item(rank)` gives a
  `RetrievedItem(kind="chunk", score=rrf, meta={ranks, scores, rrf, chunk, ...})`.
* **Fusion**: equal-weight RRF, k=60; each channel ranks the chunks in scope once, so a chunk makes ONE contribution
  per channel. Ties: best single-channel rank, then chunk number (independent of channel order). `depth` = per-channel
  list length fed to RRF (default `k`); `k` = hits returned. `rrf_fuse({"chan": [ids best first], ...}, k=60)` is the
  same fusion for any other candidate identities (e.g. LEX regions) - give each region ONE id per channel.
* **Scope**: `doc_ids` (list, or None / "corpus") masks the SAME full index: BM25 idf/avgdl are the whole set's.
  Unknown ids are listed in `diagnostics["scope_docs_unknown"]`.
* **Vectors are never computed at query time.** Chunks without a vector are left out of the dense channel and
  reported: `diagnostics["dense"] = {missing_vectors_in_scope, missing_docs, complete, query_vector: given|embedded,
  skipped?}` and a line in `diagnostics["warnings"]`. Only the question is embedded (`purpose="query-embed"`), through
  the `client` passed to `search` (or the index's), and only when `query_vector` is not given. No client and no
  vector: dense is skipped and said so; BM25 still answers.
* `hx.bm25_scores(text)` (full-index scores), `hx.scope_mask(doc_ids)`, `hx.hit_for_chunk(i)`, `hx.status()`,
  `hx.dense_complete`, `hx.chunks`, `hx.E` (n x d, zero rows where missing), `hx.has_vec`.

## 2. Prebuild CLI (run before any test; Ollama needed)

```
.venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set C
.venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set D --set A_fa --batch-size 16 --commit-every 4
.venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set C --status      # no build, no model call
.venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set C --no-embed    # chunks + BM25 + cached vectors only
```

Resumable (vectors committed to the shared cache in one transaction every `--commit-every` batches, and on any
failure); progress lines on stderr; below-normal priority (`--priority idle` for lowest); numpy threads 2.
`--max-batches N` embeds a slice and stops. From Python: `prebuild.prebuild(corpus_set, cfg, client, ...)`.

## 3. `ragbench.evidence.build_blocks`

```python
from ragbench.evidence import build_blocks, block_overlap, row_overlap, span_overlap
items, report = build_blocks(candidates, corpus, budget=None, requirements=None, question=None, cfg=None, opts=None)
ev = self.final_pack(items, index, query)      # every item fits whole: no cut, no drop
```

* `candidates`: best first; `Hit`s, `RetrievedItem`s or dicts with `doc_id` and `span` (or `unit_ids`), optional
  `ranks` (or `meta["ranks"]`) and `rrf`/`score`. Only the order and the spans matter; ranking is the caller's.
* `requirements`: `[{"id": "a", "terms": [...] | "metric": ..., "entity": str|[...], "period": str|[...]}]` - built from
  the QUESTION, never from gold. A block covers a requirement when its doc title + section path + text contain (after
  `fold`) at least one value of every field given.
* Blocks: prose 700-1,000 chars, verbatim, sentence/line bounded, inside one section's run of non-table units (never
  across headings, never into a table); tables = caption (in-unit caption line, or the caption unit directly before the
  table in the same section) + the repeated column-heading line + relevant rows (support rows matching question /
  requirement tokens; all support rows when none match), <= 1,200 chars, overflow rows go to a further block with
  disjoint rows. Overlapping / directly adjacent support in one section is merged BEFORE selection; table units of one
  table merge by row.
* Selection: one block per requirement first (full coverage preferred, then best partial), then relevance; cap 6
  (`max_blocks`), up to 8 (`max_blocks_req`) when reserved blocks need it; cost = exact rendering
  (`item_header` + "\n" + text, "\n\n" between items); a non-fitting block is trimmed (sentence prefix / fewer rows)
  when >= 300 chars fit, else dropped. The result is re-checked with `pack_evidence`.
* Order: relevance; comparisons (requirements with > 1 entity) grouped by entity in requirement order; series (> 1
  requirement period, or > 1 year in the question) by period. `opts={"order": ...}` overrides.
* Items: `RetrievedItem(kind="block", span=[a, b] or None for a non-contiguous table block, meta={block: prose|table,
  spans, rows: [[unit_id, row]], ranks, rrf, candidates, requirements, reserved_for, trimmed, caption_unit, table})`.
* `report`: `mode, budget, rendered_chars, n_candidates, n_supports, merged_supports, n_blocks, cap, blocks[...]`
  (doc, section, spans, rows, ranks, rrf, candidates, requirements, reserved_for, trimmed, rendered_chars),
  `dropped[...]` with `reason` budget | cap | redundant, `n_dropped`, `requirements{id: status covered|partial|missing,
  block}`, `unmapped`, `warnings`.
* Overlap helpers replace LEX's table-id overlap: `row_overlap(rows_a, rows_b)`, `span_overlap(a, b)`,
  `block_overlap(item_a, item_b)` (rows for tables, spans for prose, 0 across documents).
* Tunables: `evidence.DEFAULTS` (dev defaults, not optima). Observed on real A_en/C with BM25 only: 6 blocks used
  3.5k-6k of the 6k budget, because tiny sections give short blocks; a candidate for tuning on dev.

## 4. HYBRID control

`--approach hybrid` (variant `HYBRID`): hybrid search (k=40, depth 40) -> B2/B3's `_offer` (top_k, fill_budget,
max_candidates 40) -> shared packer and answer call. `--ablation blocks` (variant `HYBRID+blocks`): the same hits
through `build_blocks`. `doc_card` / `section_expand` as for B2/B3. Trace: `hybrid` (search diagnostics), `blocks`
(the report), `top` ([chunk, rrf, bm25 rank, dense rank]), `dense_complete`. At index time missing vectors are
embedded (like B3; `approaches.hybrid.HYBRID_DEFAULTS["embed_at_build"]`); prebuild first.

## Tests

`python -m ragbench.tests.test_hybrid` (offline, temp dirs, fake embedders): fusion math, one contribution per
channel, equality with B2 (bm25 only) and B3 (dense only), scope masks with fixed BM25 statistics, missing-vector
diagnostics, prebuild resumability, evidence blocks (lengths, sentence bounds, sections, merging, tables and rows,
requirement reservation, ordering, exact budget), and HYBRID / HYBRID+blocks end to end.
