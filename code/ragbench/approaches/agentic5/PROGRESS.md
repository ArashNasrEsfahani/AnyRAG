# AGENTIC5 build - progress

Worktree: `D:\KareOne\RAG Paper\.claude\worktrees\agent-a2c8e1a4c3e6ae935`, branch `worktree-agent-a2c8e1a4c3e6ae935`.
Package: `ragbench/approaches/agentic5/`. AGENTIC/2/3 untouched. This file is mirrored to
`D:\KareOne\RAG Paper\work\upgrades\agentic5\PROGRESS.md`.

## Status (2026-09-24): built, all offline suites green
- Branch merged with `upgrades` (the shared foundation + `ragbench/corpus`), commit c866ce2.
- `python -m ragbench.approaches.agentic5.test_agentic5` -> 96 checks, 0 failures.
- `python -m ragbench.approaches.agentic3.test_agentic3` -> 86 / 0.
- `python -m ragbench.approaches.agentic3.test_reconcile_sql` -> 709 historical cases, 0 failures. NOTE: this
  suite finds its historical runs relative to the CWD, so run it with cwd = the main tree and the worktree first on
  sys.path: `python -c "import sys; sys.path.insert(0, r'<worktree>'); import runpy;
  runpy.run_module('ragbench.approaches.agentic3.test_reconcile_sql', run_name='__main__')"`.
- `python -m ragbench.tests.test_core_integration` -> 68 / 0.
- Registered: `approaches/__init__.py` (APPROACHES["AGENTIC5"], AGENTIC5_FLAGS), `run.py` (flags, candidates= /
  shortlist= values, AGENTIC5_OPTIONS in TUNING_ENV, embed warm-up).
- No model/API call was made; tests use scripted clients and temp KBs under work/upgrades/agentic5/test_tmp.

## Modules
- defaults.py - DEFAULTS (AGENTIC3's + v5 keys), FLAGS; write_back=False, warm=False, no definition guard escape
- requirements.py - requirement matrix (entity x metric x period, operation), plan schema/prompt (4 fields appended
  last), period parser (years, ranges, quarters, halves, "سال 98"), field-by-field fallback for junk plans
- channels.py - foundation adapter: HybridIndex.load at query time (open(embed=False) at index time), search
  diagnostics copied; fallback = the KB path (facts/units + prebuilt passage vectors only); never embeds documents
- routing.py - card + capped best-2 chunk contribution (RRF k=60), shortlist <= 8 with retained entity/period matches,
  escape candidates, requirement-triggered expansion, no recency fallback
- allocate.py - requirements first (round robin), then channels; one escape slot; batch 1 <= 12
- claims.py - typed verify prompt/schema; binding (an unkeyed single entity is a label; several unkeyed entities need the doc title or snippet) (model req checked against source entity/period/metric; rebinding;
  unbound kept); annual vs quarter/half; unit verified against the snippet; value in base units
- reconcile5.py - identity (entity, metric, period key, def class, unit family); restatement lineage; disagreements
  kept; coverage per requirement; open series
- arith.py - difference / ratio / sum / order / series change from verified compatible operands, operand claim ids;
  unknown (never zero) otherwise
- evidence5.py - table (company | period | metric/definition | value | status | sources) + quote-centred verbatim
  snippets, fitted jointly, citations numbered once after packing, table units only from packed sources; leftover
  budget -> build_blocks for missing requirements
- approach.py - RequirementAgent / _Session5: plan; route; retrieve+allocate; verify 1; reconcile; verify 2 (missing
  or conflicting requirements; a plan repair consumes it, recorded); arithmetic; evidence; answer; read-only KB;
  deadline enforced at each model request (read timeout cut; verify skipped without time for it + the answer)

## Depends on the foundation merge
- `ragbench.retrieval.hybrid.HybridIndex` (load/open/search with client, HitList.diagnostics) and
  `ragbench.evidence.build_blocks` - both merged from `upgrades`. Without them AGENTIC5 still runs on the KB path
  (recorded as `channels.hybrid = absent`).

## Not done / next
- No run on real data (no model calls allowed here). First dev run: prebuild the hybrid index for the sets
  (`python -m ragbench.retrieval.prebuild --set ...`), then `ragbench.run --approach agentic5`.
- Verified source snippets use AGENTIC5's own quote-centred excerpt (so the cited text always contains the verified
  quote); build_blocks is used for unverified support only.
