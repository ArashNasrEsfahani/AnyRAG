# Post-hoc experiments, November 2026 — rules shared by every stream

These experiments come after the one-shot test (2026-10-01..05). They are **post-hoc and exploratory**.
The test split has been unsealed, so nothing here may be tuned on test questions.

## Paths
- Repo root: `D:\KareOne\RAG Paper`. Python: `D:\KareOne\RAG Paper\.venv\Scripts\python.exe`.
- Frozen code: `D:\KareOne\RAG-Paper-upgrades` @ 77924d4 (package `ragbench`). **Read-only.** You may put it on `sys.path`
  and import it, but never edit, commit or check out anything there.
- Frozen test data: `work\datasets_v2\test_freeze\` (protocol, pins, judge_pin.json, reader_pins.json, questions) and
  `work\datasets_v2\test_run\` (runs/<pool>/<arm>_<budget>/runs_*.jsonl with full traces, packs/<arm>/*.json with exact
  reader messages, replay/<reader>/..., judge/<reader>/verdicts_*.jsonl and needs_adjudication_*.jsonl,
  TEST_RESULTS.md, TEST_RESULTS_HUMAN_SYM.md, human_check/...). **Read-only.**
- Dev questions: `work\datasets_v2\questions\*_dev.jsonl` (test: `*_test.jsonl`).
- Paper analysis helpers: `paper\analysis\revision_stats.py` (machine/human verdict loading, connected components of
  source families, sign-p ranges over unresolved cells, cluster bootstrap). Reuse; do not edit.
- Your outputs go ONLY under `work\posthoc_2026-11\<your stream>\`. Code, logs, ledgers, results.

## No GPU
Do not load torch on CUDA, do not start or call Ollama, do not run bge-m3 or the reranker. Set
`CUDA_VISIBLE_DEVICES=""` in every process. Use what is saved: fused/reranked candidate lists, scores, chunk ids, selector
prompts and exact pack messages live in the run traces and packs. BM25 and regex on CPU are fine. If an experiment needs
a new dense or rerank computation, stop that experiment and report it as "needs GPU" instead.

## Money (shared OpenRouter key in `D:\KareOne\RAG Paper\.env`, OPENROUTER_API_KEY; never print it)
- Hard global floor: before every batch of at most 50 calls, GET `https://openrouter.ai/api/v1/key` and read
  `data.limit_remaining`. If it is below **$2.00**, stop all calls and report. Other streams spend in parallel.
- Your stream cap is in your brief. Request `usage: {"include": true}` and log every call (model, provider, tokens,
  `usage.cost`) to `<stream>\ledger.jsonl`. Stop at your cap.
- Estimate the cost of each experiment on 3 questions first, then extrapolate before running the rest.
- Cheap reference prices from the test ledgers: GLM-5.3-flash ≈ $0.27 per 1,000 reader calls; gpt-oss-120b judge
  ≈ $0.10 per 1,000 calls.

## Same reader, same judge
- Reader: `z-ai/glm-5.3-flash` with the provider pin and params from `reader_pins.json`, and the same answer prompt
  and answer-format routing as the test (see how packs/<arm>/*.json messages were built in ragbench; reuse that code).
- Judge: exactly the pinned judge in `judge_pin.json` (gpt-oss-120b, deepinfra/bf16, temperature 0, seed 42, no
  fallbacks) with the same rubric/prompt code as the test (`ragbench/eval/judge.py` and `test_freeze/run_test.py judge`).
  A reply that does not parse after 3 retries is **unresolved**. Report totals as bounds a..b over unresolved cells.
- Score on the 90 answerable test questions; report the 6 unanswerable separately (abstention).

## Comparisons
- Compare each new arm with the frozen arms on the same questions using the saved machine verdicts: paired wins/losses,
  exact two-sided sign test (range over unresolved assignments), paired bootstrap over questions and over the 20
  connected components (use the revision_stats helpers). Report by tier (T1/T2/T3) and language (fa/en).
- Report query-time LLM tokens per question and the number of calls. Latency is descriptive only.

## Working rules
- Smoke-test every new arm on at most 5 dev questions to debug. Then run it once on test. Do not iterate on test.
- Write append-only JSONL, one line per finished cell, keyed so that a rerun skips finished cells.
- Deliverable: `<stream>\RESULTS.md` with (1) what was run, exactly; (2) result tables with bounds; (3) cost spent;
  (4) anything that failed or was skipped and why; (5) a 3-6 sentence "facts for the paper" block, each fact with its
  source file. Numbers must come from files you wrote, never from memory.

## GPU (updated 2026-11-10 evening: the author lifted the one-at-a-time rule; run GPU work in parallel)
- RTX 4060 Laptop, 8 GB VRAM, shared by streams S11-S13 running at the same time. Ollama is running (bge-m3).
- No lock: do NOT wait for gpu.lock (delete it if you created one). Run GPU phases whenever ready, in parallel.
- The only guard is VRAM: check `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` before loading a model,
  keep your process under about 2.5 GB (fp16, moderate batch size), and on a CUDA out-of-memory error lower the batch
  size and retry rather than failing. Unload models when done.
- Cache vectors and scores to disk in your folder. Record GPU wall time per phase in RESULTS.md.
