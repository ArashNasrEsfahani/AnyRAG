"""Run one approach over a question set: build/load its index, answer every
(question, setting) sequentially, append one RunRecord per line, resume on restart.

    .venv/Scripts/python.exe -m ragbench.run --approach b2 --questions ragbench/data/questions/pilot.jsonl \
        --corpus A --settings fa-fa,en-en,fa-en,en-fa --limit 5 --out runs/pilot
    .venv/Scripts/python.exe -m ragbench.run --approach ours --ablation no_facts --ablation granularity=always_window ...
    .venv/Scripts/python.exe -m ragbench.run --approach ours --index-level L1 ...      # = --ablation index_level=L1
    .venv/Scripts/python.exe -m ragbench.run --approach agentic --ablation no_gap_fill ...

Output directory (relative 'runs/<name>' resolves to work/runs/<name>; other relative paths to the
project root; one directory can hold every approach of an experiment):
  runs_<variant>.jsonl          RunRecords (resume key: question_id, approach variant, setting;
                                errored records are retried on the next run unless --no-retry-errors)
  ledger_<variant>.jsonl        one line per model call (phase index | query | smoke), shared client ledger
  index_<variant>_<set>.json    copy of the index build_stats.json (calls/tokens/time over all build attempts, size;
                                cache-inclusive cached_cost; index_cost = index.json:index_cost of OURS L0-L2, else None)
  run_meta_<variant>.json       every invocation: arguments, models + digests, Ollama version, config hash,
                                question-file hash, approach source sha1, planned/done/error counts, ledger
                                totals by phase; plus per-question signatures (text, corpus, doc scope)
Resume refuses (exit 2; --allow-mixed overrides) when earlier records of the variant were made under another
config hash or index fingerprint, when ANY record or invocation in the output directory (any variant) was
answered under another Ollama version, or when an answered question's text/scope changed. A changed approach
source only warns.

Splits: every RunRecord carries `split` (dev for pilot*/dev* question files, test for test*, other otherwise) and
`meta` (ollama_version, questions_file, final, docs_restriction; answer_format, answer_prompt, answer_num_predict
from Approach.answer). A test file runs only with --final, and --final refuses question files, variants, per-corpus
settings, answer prompt versions and overrides not declared in config.yaml `final` (see build_parser / final_problems).
--dry-run prints the plan plus, per corpus set, the index directory the run would use and whether it is complete
(no index build, no model call).
--docs-from-questions restricts every approach's corpus to the documents the question file references
(doc_scope + gold units; both languages for A and B), identically for all approaches; stamped in run_meta and
on every record.
Multi-document experiments (2026-09-24, P0 list): a question file whose questions carry schema multidoc-q-v1 (or
meta.experiment "multidoc") refuses --docs-from-questions and --docs and requires --pool <frozen manifest>
(ragbench/pools.py): the pool masks each question's "corpus" scope, indexes stay those of the full frozen sets, and the
run refuses when the pool's frozen sets or documents changed. Such a run is an IDENTITY run: every record carries
meta.identity {experiment_id, question_id, pool_id, method, options (flags + variants_version), reader (chat model +
digest), replicate, setting, rid, code_sig, question_sig, pool_sha1}; the resume key is rid, and a record under
another code signature (approach sources + shared dependencies, dependency_sha1), question signature (question_sig_v2:
+ answer format, task family, pool ids) or pool is never resumed into (exit 2, use a new --out). Every invocation also
stores question_sigs_v2 and dependency_sha1; older directories are compared on them only once they have them.
Then: .venv/Scripts/python.exe -m ragbench.eval.aggregate --runs runs/pilot --questions ... --judge none

One GPU: never run two model-heavy runs at once (the client also serialises calls across processes).
"""
import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .approaches import (AGENTIC2_FLAGS, AGENTIC3_FLAGS, AGENTIC4_FLAGS, AGENTIC5_FLAGS, AGENTIC_FLAGS, APPROACHES,
                         HYBRID_FLAGS, LEX_FLAGS, LEX3_FLAGS, LEX5_FLAGS, LEX6_FLAGS, LEX7_FLAGS, LEX8_FLAGS, OURS_FLAGS,
                         approach_class)
from .config import load_config
from .contracts import (ANSWER_PROMPT_VERSIONS, RECORD_HASH_SECTIONS, RunLog, RunRecord, Setting, answer_format,
                        answer_prompt_version, load_questions)
from .corpus.registry import load_set, set_name_for
from .eval.aggregate import cached_index_cost, resolve_path
from .llm import Ledger, OllamaClient
from .util import atomic_write_json, now_iso, read_json, read_jsonl, safe_name, sha1_json, sha1_text

ALIASES = {k.lower(): k for k in APPROACHES}
GRANULARITY = ("always_section", "always_window")
OURS_INDEX_LEVELS = ("L0", "L1", "L2", "L3")   # OURS index levels (approaches/ours/levels.py); L3 = the full enrichment


B5_FLAGS = ("agent_answer",)
from .approaches import BASELINE_FLAGS  # noqa: E402  (B2 / B3: ragbench/variants.py options)      # B5 ablation: the agent's own answer instead of the shared answer call


def parse_flags(values, approach):
    flags = {}
    known = (B5_FLAGS if approach == "B5" else BASELINE_FLAGS if approach in ("B2", "B3") else
             HYBRID_FLAGS if approach == "HYBRID" else AGENTIC_FLAGS if approach == "AGENTIC" else
             AGENTIC2_FLAGS if approach == "AGENTIC2" else AGENTIC3_FLAGS if approach == "AGENTIC3" else
             AGENTIC4_FLAGS if approach == "AGENTIC4" else AGENTIC5_FLAGS if approach == "AGENTIC5" else
             LEX_FLAGS if approach in ("LEX", "LEX2") else LEX3_FLAGS if approach == "LEX3" else
             LEX5_FLAGS if approach == "LEX5" else LEX6_FLAGS if approach == "LEX6" else
             LEX7_FLAGS if approach == "LEX7" else LEX8_FLAGS if approach == "LEX8" else OURS_FLAGS)
    for v in values or []:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            k, val = (part.split("=", 1) + [None])[:2] if "=" in part else (part, None)
            k = k.strip()
            if k not in known:
                raise SystemExit("unknown ablation %r for %s; choose from %s" % (k, approach, known))
            if k == "granularity":
                if val not in GRANULARITY:
                    raise SystemExit("granularity must be one of %s" % (GRANULARITY,))
                flags[k] = val
            elif k == "index_level":
                lv = str(val or "").strip().upper()
                if lv not in OURS_INDEX_LEVELS:
                    raise SystemExit("index_level must be one of %s" % (OURS_INDEX_LEVELS,))
                if lv != "L3":                      # L3 is today's OURS: no flag, same variant name and index
                    flags[k] = lv
            elif approach == "LEX5" and k in ("selector", "view", "top"):
                from .approaches.lex5.approach import LEX5_DEFAULTS, SELECTORS, VIEWS
                ok = {"selector": tuple(SELECTORS), "view": tuple(VIEWS)}.get(k)
                if k == "top":
                    try:
                        val = int(val)
                    except (TypeError, ValueError):
                        raise SystemExit("top takes an integer, e.g. top=40")
                    if val < 1:
                        raise SystemExit("top must be >= 1")
                elif val not in ok:
                    raise SystemExit("%s must be one of %s" % (k, ok))
                if val != LEX5_DEFAULTS[k]:          # the default names no variant
                    flags[k] = val
            elif approach == "LEX6" and k == "rank":
                v = str(val or "").strip().lower()
                if v not in ("s2", "s5", "s2s5"):
                    raise SystemExit("rank must be one of s2, s5, s2s5")
                if v != "s2s5":                      # the default names no variant
                    flags[k] = v
            elif approach in ("LEX", "LEX2") and k == "window":
                try:
                    n = int(val)
                except (TypeError, ValueError):
                    raise SystemExit("window takes an integer number of characters, e.g. window=300")
                if n < 50:
                    raise SystemExit("window must be >= 50 characters")
                flags[k] = n
            elif approach == "HYBRID" and k == "doc":           # DOC idea: top-k documents (approaches/kwdoc.py)
                try:
                    n = int(val)
                except (TypeError, ValueError):
                    raise SystemExit("doc takes an integer number of documents, e.g. doc=2")
                if n < 1:
                    raise SystemExit("doc must be >= 1")
                flags[k] = n
            elif (approach in ("AGENTIC2", "AGENTIC3", "AGENTIC4") and k == "candidates") or \
                    (approach == "AGENTIC5" and k in ("candidates", "shortlist")):
                try:
                    n = int(val)
                except (TypeError, ValueError):
                    raise SystemExit("%s takes an integer, e.g. %s=32" % (k, k))
                if n < 1:
                    raise SystemExit("%s must be >= 1" % k)
                flags[k] = n
            else:
                if val not in (None, "true", "True", "1"):
                    raise SystemExit("ablation %s takes no value" % k)
                flags[k] = True
    if flags and approach not in ("OURS", "B5", "AGENTIC", "AGENTIC2", "AGENTIC3", "AGENTIC4", "AGENTIC5", "LEX", "LEX2",
                                  "B2", "B3", "HYBRID", "LEX3", "LEX5", "LEX6", "LEX7", "LEX8"):
        raise SystemExit("--ablation applies to --approach ours, agentic, agentic2, agentic3, agentic4, agentic5, lex, lex2, lex3, b2, b3, hybrid (and b5 "
                         "agent_answer) only")
    return flags


def error_record(approach, question, setting, err, cfg, run_id):
    """RunRecord for a failure outside Approach.answer() (e.g. index load); counts as an error."""
    s = Setting.parse(setting)
    return RunRecord(
        question_id=question.id, approach=approach.variant, approach_base=approach.name, flags=approach.flags,
        setting=s.name, q_lang=s.q_lang, d_lang=s.d_lang, corpus=question.corpus,
        corpus_set=set_name_for(question.corpus, s.d_lang), answer_text="", abstained=False, retrieved_units=[],
        evidence_units=[], evidence_items=[], evidence_chars=0, llm_calls=0, embed_calls=0, prompt_tokens=0,
        completion_tokens=0, embed_tokens=0, llm_wall_ms=0.0, load_ms=0.0, latency_ms=0.0, failed_calls=0,
        error=err, trace={"traceback": traceback.format_exc()[-2000:]},
        models={"chat": cfg.models["chat"], "embed": cfg.models["embed"]},
        config_hash=cfg.hash(*RECORD_HASH_SECTIONS), index_fingerprint="",
        ts=now_iso(), run_id=run_id)


def doc_key(doc_id):
    """'A.rpt-b.fa' -> 'rpt-b' (the key shared by both languages)."""
    return doc_id.split(".", 1)[1].rsplit(".", 1)[0] if doc_id.count(".") >= 2 else doc_id


def parse_docs(value):
    """--docs 'rpt-b,rpt-f' (keys, both languages) or full doc ids -> set of keys, or None."""
    if not value:
        return None
    return {doc_key(x.strip()) if x.strip().count(".") >= 2 else x.strip() for x in value.split(",") if x.strip()}


def restrict_set(cs, keys):
    """The CorpusSet reduced to the documents whose key is in `keys` (same set name, so Query.corpus_set
    still resolves to it; the fingerprint - and therefore every index directory - differs from the full set)."""
    if not keys:
        return cs
    return cs.subset([d for d in cs.doc_ids() if doc_key(d) in keys])


def question_in_docs(question, keys):
    if not keys:
        return True
    if question.doc_scope in (None, "corpus"):
        return False            # a corpus-wide question is not the same question over a subset
    return all(doc_key(d) in keys for d in question.doc_scope)


def question_split(path):
    """Split of a question file from its name: pilot*/dev* -> dev (all tuning), test* -> test, else other."""
    name = Path(str(path)).name.lower()
    if name.startswith("pilot") or name.startswith("dev"):
        return "dev"
    if name.startswith("test"):
        return "test"
    return "other"


def referenced_doc_keys(questions, corpus):
    """Document keys a question set references for one corpus: every doc_scope document and the document of
    every gold unit (either language). Keys are language-free, so A and B keep both language versions."""
    keys = set()
    for q in questions:
        if q.corpus != corpus:
            continue
        if isinstance(q.doc_scope, (list, tuple)):
            keys.update(doc_key(d) for d in q.doc_scope)
        for units in (q.gold_units or {}).values():
            keys.update(doc_key(u.split("::", 1)[0]) for u in units or [])
    return keys


TUNING_ENV = ("AGENTIC2_OPTIONS", "AGENTIC3_OPTIONS", "AGENTIC4_OPTIONS", "AGENTIC5_OPTIONS", "LEX_OPTIONS", "LEX3_OPTIONS")   # OPTIONS_ENV of approaches/agentic2 and agentic3: a JSON
                                                        # object of DEFAULTS overrides, for dev tuning only


def final_problems(args, cfg, variant, settings):
    """Why this invocation is not an allowed --final run (empty list = allowed).

    Declarations in config.yaml `final` (each optional key keeps the older, global rule when absent):
      variants          the only variants accepted
      question_files    the only question file NAMES accepted
      exclude_variants  {question file name: [variants refused on that file]}
      settings          a list (every corpus) or {corpus: [settings]}; an invocation may run any subset of its
                        corpus's list, a corpus without an entry runs nothing; absent: the global `settings`
      answer_prompts    {answer format: prompt version}; must equal contracts.ANSWER_PROMPT_VERSIONS (every format), so
                        a changed answer prompt needs a deliberate config change before --final runs again
      docs_from_questions  whether --docs-from-questions (the run's own question file) is required"""
    fin = cfg.get("final") or {}
    probs = []
    if variant not in (fin.get("variants") or []):
        probs.append("variant %s is not declared in config final.variants" % variant)
    qname = Path(str(args.questions)).name
    if fin.get("question_files") is not None and qname not in (fin.get("question_files") or []):
        probs.append("question file %s is not declared in config final.question_files %s" % (
            qname, fin.get("question_files")))
    excluded = (fin.get("exclude_variants") or {}).get(qname) or []
    if variant in excluded:
        probs.append("variant %s is excluded for %s in config final.exclude_variants" % (variant, qname))
    for opt, val in (("--ids", args.ids), ("--limit", args.limit), ("--docs", args.docs),
                     ("--approach-class", args.approach_class)):
        if val not in (None, "", False):
            probs.append("%s is a subset/override not allowed with --final" % opt)
    for opt, val in (("--allow-mixed", args.allow_mixed), ("--no-health", args.no_health)):
        if val:
            probs.append("%s is not allowed with --final" % opt)
    for var in TUNING_ENV:
        if os.environ.get(var):
            probs.append("%s is set: it overrides the approach's DEFAULTS and belongs to dev tuning only" % var)
    declared = set(cfg.get("settings") or [])
    bad = [s.name for s in settings if s.name not in declared]
    if bad:
        probs.append("settings %s are not declared in config settings" % bad)
    fin_settings = fin.get("settings")
    if fin_settings is not None:
        if isinstance(fin_settings, dict):
            per_corpus, where = fin_settings.get(args.corpus) or [], "final.settings.%s" % args.corpus
        else:
            per_corpus, where = fin_settings, "final.settings"
        bad = [s.name for s in settings if s.name not in set(per_corpus)]
        if bad:
            probs.append("settings %s are not declared in config %s %s" % (bad, where, list(per_corpus)))
    prompts = fin.get("answer_prompts")
    if prompts is not None:
        for fmt in sorted(set(ANSWER_PROMPT_VERSIONS) | set(prompts)):
            if prompts.get(fmt) != ANSWER_PROMPT_VERSIONS.get(fmt):
                probs.append("answer prompt for %s questions is %s in contracts but config final.answer_prompts "
                             "declares %s" % (fmt, ANSWER_PROMPT_VERSIONS.get(fmt), prompts.get(fmt)))
    want_dfq = bool(fin.get("docs_from_questions"))
    if want_dfq != (args.docs_from_questions is not None):
        probs.append("config final.docs_from_questions is %s: %s --docs-from-questions" % (
            want_dfq, "pass" if want_dfq else "do not pass"))
    if args.docs_from_questions not in (None, "__questions__"):
        probs.append("--docs-from-questions must use the run's own question file with --final")
    try:
        from .config import DEFAULT_CONFIG
        if Path(cfg.source).resolve() != DEFAULT_CONFIG.resolve():
            probs.append("config %s is not ragbench/config.yaml (RAGBENCH_CONFIG or overrides)" % cfg.source)
    except OSError:
        probs.append("config source %s cannot be resolved" % cfg.source)
    return probs


def test_protocol_problems(args, manifest_path, qpath, variant, environ=None, head=None, dirty=None):
    """Why this invocation is not an allowed run under a frozen v2 test protocol manifest (empty list = allowed).

    The datasets-v2 one-shot test (work/datasets_v2/test_freeze/PROTOCOL.md) uses an OpenRouter reader, per-question
    --ids interleaving and identity runs, which the older --final rule (config.yaml `final`, local reader) refuses; it
    is gated by its own manifest instead (test_freeze/MANIFEST.json). Every one of these must hold:
      the question file is a manifest question file with the frozen sha256; the variant is a manifest arm; --reader,
      --pool (manifest sha256), the env OPENROUTER_PROVIDER_PINS, READER_MAX_ATTEMPTS, LEX5B_SELECTOR_PIN and the arm's
      declared env (LEX8_IDX_DIR / LEX8_SYNTH_DIR) equal the manifest; no dev-tuning env; no --allow-mixed / --limit /
      --docs / --docs-from-questions / --final; the code worktree HEAD is the manifest's code_commit and ragbench/ has
      no uncommitted change."""
    import hashlib
    import subprocess
    env = os.environ if environ is None else environ
    probs = []
    try:
        man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return ["test protocol manifest %s unreadable: %s" % (manifest_path, e)]
    if args.final:
        probs.append("--final and --test-protocol are exclusive")
    qf = (man.get("question_files") or {}).get(Path(qpath).name)
    if qf is None:
        probs.append("question file %s is not in the manifest" % Path(qpath).name)
    elif hashlib.sha256(Path(qpath).read_bytes()).hexdigest() != qf.get("sha256"):
        probs.append("question file %s sha256 differs from the manifest" % Path(qpath).name)
    arms = {a["variant"]: a for a in man.get("arms") or []}
    arm = arms.get(variant)
    if arm is None:
        probs.append("variant %s is not a manifest arm %s" % (variant, sorted(arms)))
    if (args.reader or None) != man.get("reader"):
        probs.append("--reader %s != manifest reader %s" % (args.reader, man.get("reader")))
    pools = man.get("pools") or {}
    pid = Path(str(args.pool or "")).stem
    if pid not in pools:
        probs.append("--pool %s is not a manifest pool" % args.pool)
    else:
        try:
            from .pools import resolve_pool_path
            pp = resolve_pool_path(str(args.pool), load_config())
            if hashlib.sha256(Path(pp).read_bytes()).hexdigest() != pools[pid].get("sha256"):
                probs.append("pool manifest %s sha256 differs from the test manifest" % pid)
        except Exception as e:  # noqa: BLE001
            probs.append("pool %s cannot be resolved: %s" % (args.pool, e))
    try:
        pins = json.loads(env.get("OPENROUTER_PROVIDER_PINS") or "null")
    except ValueError:
        pins = "unparsable"
    if pins != man.get("reader_pins"):
        probs.append("env OPENROUTER_PROVIDER_PINS differs from manifest reader_pins")
    if str(env.get("READER_MAX_ATTEMPTS") or "") != str(man.get("reader_max_attempts")):
        probs.append("env READER_MAX_ATTEMPTS=%s != manifest %s" % (env.get("READER_MAX_ATTEMPTS"),
                                                                    man.get("reader_max_attempts")))
    if (env.get("LEX5B_SELECTOR_PIN") or None) != man.get("selector_pin_env"):
        probs.append("env LEX5B_SELECTOR_PIN differs from the manifest (%s)" % man.get("selector_pin_env"))
    for k, v in ((arm or {}).get("env") or {}).items():
        if env.get(k) != v:
            probs.append("env %s != the arm's declared %s" % (k, v))
    for var in TUNING_ENV:
        if env.get(var):
            probs.append("%s is set (dev tuning only)" % var)
    for opt, val in (("--allow-mixed", args.allow_mixed), ("--limit", args.limit), ("--docs", args.docs),
                     ("--docs-from-questions", args.docs_from_questions)):
        if val not in (None, "", False):
            probs.append("%s is not allowed under the test protocol" % opt)
    root = Path(__file__).resolve().parents[1]
    if head is None:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if dirty is None:
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "ragbench"], capture_output=True,
                               text=True).stdout.strip()
    if head != man.get("code_commit"):
        probs.append("code HEAD %s != manifest code_commit %s" % (head[:12], str(man.get("code_commit"))[:12]))
    if dirty:
        probs.append("ragbench/ has uncommitted changes")
    return probs


def directory_ollama_versions(out):
    """{version: [where]} over every run_meta_*.json invocation that answered something and every RunRecord's
    meta.ollama_version in an output directory (all variants)."""
    found = {}
    for f in sorted(Path(out).glob("run_meta_*.json")):
        for inv in (read_json(f) or {}).get("invocations") or []:
            v = (inv.get("ollama") or {}).get("version")
            if v and inv.get("n_ok"):
                found.setdefault(v, set()).add(f.name)
    for f in sorted(Path(out).glob("runs_*.jsonl")):
        for r in read_jsonl(f):
            v = ((r or {}).get("meta") or {}).get("ollama_version")
            if v and not r.get("error"):
                found.setdefault(v, set()).add(f.name)
    return {v: sorted(w) for v, w in found.items()}


def question_sig(question, needs_gold=False):
    """Hash of what an approach's answer depends on: both question texts, corpus and doc scope
    (plus the gold units for B1). Gold answers and answer types are excluded: aggregation re-scores
    every record against the current question file, so fixing them needs no rerun."""
    d = {"corpus": question.corpus, "fa": question.question_fa, "en": question.question_en,
         "scope": question.doc_scope}
    if needs_gold:
        d["gold"] = question.gold_units
    return sha1_json(d)[:16]


def question_sig_v2(question, needs_gold=False):
    """question_sig plus everything else the approach's request depends on (2026-09-24): the answer format and its
    source fields (type, answer_format, task_family) and the pool ids. Stored beside question_sig (run_meta
    question_sigs_v2), so older output directories keep resuming exactly as before."""
    d = {"base": question_sig(question, needs_gold), "answer_format": answer_format(question),
         "type": question.type, "answer_format_field": getattr(question, "answer_format", None),
         "task_family": getattr(question, "task_family", None), "pool_ids": question.pool_ids(),
         "schema": getattr(question, "schema", None)}
    return sha1_json(d)[:16]


# shared dependencies of every approach's answers (evidence packing, prompts, variants, corpus model, KB, retrieval);
# the files of whatever other ragbench module is already loaded when the run starts are added (dependency_files)
DEPENDENCY_DIRS = ("", "corpus", "kb", "approaches", "retrieval")   # non-recursive; approaches/ = its top-level modules
DEPENDENCY_EXCLUDE = ("run.py",)


def dependency_files():
    pkg = Path(__file__).resolve().parent
    files = set()
    for d in DEPENDENCY_DIRS:
        for p in (pkg / d).glob("*.py"):
            if "test" not in p.stem and p.name not in DEPENDENCY_EXCLUDE:
                files.add(p.resolve())
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(pkg)
        except ValueError:
            continue
        if rel.parts[0] in ("tests", "tools", "eval") or "test" in p.stem or p.name in DEPENDENCY_EXCLUDE or p.suffix != ".py":
            continue
        files.add(p)
    return sorted(files, key=lambda p: p.relative_to(pkg).as_posix())


def dependency_sha1():
    """sha1 over the shared dependency sources (dependency_files), recorded per invocation and, for identity runs,
    in every record's code signature: a revised contracts/variants/retrieval module never resumes into stale records."""
    pkg = Path(__file__).resolve().parent
    return sha1_text("".join(p.relative_to(pkg).as_posix() + "\0" + p.read_text(encoding="utf-8", errors="replace")
                             for p in dependency_files()))


PACK_SCHEMA = "answer-pack-v1"


def answer_messages_sha256(messages):
    import hashlib
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def save_answer_pack(approach, rec, pack_dir):
    """Multi-reader test (2026-10-01, test_freeze/PROTOCOL.md amendment 1): the exact input of the shared answer call
    (system + user messages with the rendered evidence, and its num_predict) is the PACK. Retrieval, reranking and any
    selector call run once per question x arm; tools/reader_replay.py sends the identical messages to the other
    readers. Every record carries meta.answer_messages_sha256; with pack_dir set the pack is also written to
    <pack_dir>/<variant>/<question>__<setting>.json. Called after the timed scope; changes no answer."""
    ac = getattr(approach, "_answer_call", None)
    if not ac or rec.error:
        return None
    h = answer_messages_sha256(ac["messages"])
    rec.meta["answer_messages_sha256"] = h
    if not pack_dir:
        return None
    d = Path(pack_dir) / safe_name(rec.approach)
    d.mkdir(parents=True, exist_ok=True)
    p = d / ("%s__%s.json" % (rec.question_id, rec.setting))
    atomic_write_json(p, {"schema": PACK_SCHEMA, "question_id": rec.question_id, "variant": rec.approach,
                          "setting": rec.setting, "q_lang": rec.q_lang,
                          "rid": ((rec.meta or {}).get("identity") or {}).get("rid"),
                          "messages": ac["messages"], "num_predict": ac["num_predict"], "purpose": ac["purpose"],
                          "messages_sha256": h, "evidence_chars": rec.evidence_chars, "at": now_iso()}, indent=1)
    return p


def record_identity(experiment_id, question_id, pool_id, variant, flags, variants_version, cfg, replicate, setting):
    """The identity of one RunRecord in an identity run (multi-document experiments): experiment, question, pool,
    method, options, model configuration (reader), replicate and setting. rid = its short hash, the resume key."""
    ident = {"experiment_id": experiment_id, "question_id": question_id, "pool_id": pool_id, "method": variant,
             "options": {"flags": dict(sorted((flags or {}).items())), "variants_version": variants_version},
             "reader": {"chat": cfg.models.get("chat"), "chat_digest": cfg.models.get("chat_digest")},
             "replicate": int(replicate), "setting": setting}
    ident["rid"] = sha1_json(ident)[:20]
    return ident


def approach_code_sha1(approach):
    """sha1 over the approach's source: its module file, or every non-test .py file of its package
    (e.g. approaches/ours/). Recorded per invocation so a code change mid-experiment is visible."""
    mod = sys.modules.get(type(approach).__module__)
    f = Path(getattr(mod, "__file__", "") or "")
    if not f.is_file():
        return None
    files = [f] if f.parent.name == "approaches" else sorted(p for p in f.parent.glob("*.py") if "test" not in p.stem)
    return sha1_text("".join(p.name + "\0" + p.read_text(encoding="utf-8", errors="replace") for p in files))


def _index_cost(index_dir):
    """index.json:index_cost (per-stage index cost written by the OURS index levels L0-L2), else None."""
    try:
        j = read_json(Path(index_dir) / "index.json")
    except (OSError, ValueError):
        return None
    return j.get("index_cost") if isinstance(j, dict) else None


def _short(text, n=70):
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def build_parser():
    ap = argparse.ArgumentParser(description="Run one ragbench approach over a question set (resumable).")
    ap.add_argument("--approach", help="b0|b1|b2|b3|b4|b5|ours|agentic|agentic2|agentic3|agentic4|agentic5|lex|lex2|lex3|lex5|lex6|lex5b|lex7|lex8|lex10|hybrid|setr")
    ap.add_argument("--ablation", action="append", default=[],
                    help="OURS flags, repeatable or comma list: no_enrichment, no_facts, no_normalization, "
                         "monolingual_patterns, granularity=always_section|always_window, facts_bare, "
                         "index_level=L0|L1|L2|L3; AGENTIC: no_guard, no_gap_fill, line_windows, no_doc_selection, "
                         "synth_answer; AGENTIC2: no_vectors, no_factdb, no_tree, no_verify, candidates=<n>; "
                         "AGENTIC3: no_typed, no_trigram, no_sql_reconcile, no_fastpath, verify_gate, "
                         "vectors_always, no_vectors, no_facets, no_tree_prior, warm, candidates=<n>; "
                         "AGENTIC4: no_graph, no_family, no_graph_quals, no_cards, no_vectors, no_facets, "
                         "no_concepts, candidates=<n>; "
                         "AGENTIC5: no_hybrid, no_dense, no_escape, no_gap_fill, no_arith, write_back, warm, "
                         "candidates=<n>, shortlist=<n>, doc_card, section_expand; "
                         "B2 / B3: doc_card, section_expand; HYBRID: blocks, doc_card, section_expand; "
                         "LEX / LEX2: llm_regex, free_keywords, fixed_window, title_routing, open_corpus, "
                         "window=<chars>, bm25, report_vocab, year_pass, no_labels; "
                         "LEX3: open_corpus, no_pick, no_lexical, no_morph, no_followup, no_escape, doc_card, section_expand; "
                         "B5: agent_answer")
    ap.add_argument("--index-level", default=None, choices=list(OURS_INDEX_LEVELS),
                    help="OURS index level, the same as --ablation index_level=<L>: L0 regex fact lines only, L1 + one "
                         "bilingual glossary per document, L2 + batched section keywords, L3 the full enrichment (default)")
    ap.add_argument("--questions", required=True, help="question JSONL")
    ap.add_argument("--corpus", required=True, choices=["A", "B", "C", "D", "K", "L"])
    ap.add_argument("--settings", default="fa-fa,en-en,fa-en,en-fa")
    ap.add_argument("--limit", type=int, default=None, help="first N questions of the corpus (after --ids)")
    ap.add_argument("--ids", default=None, help="comma list of question ids")
    ap.add_argument("--docs", default=None,
                    help="restrict the corpus to these documents (comma list of keys such as rpt-b,rpt-f, or "
                         "doc ids; both languages). Indexes are built over the subset only (their fingerprint differs "
                         "from the full set); questions with doc_scope 'corpus' or outside the subset are skipped. "
                         "For smoke tests, dev checks and cost estimates: B2 IDF and B4/B5 routing then see only the subset.")
    ap.add_argument("--docs-from-questions", nargs="?", const="__questions__", default=None, metavar="FILES",
                    help="restrict EVERY approach's corpus to the documents referenced by the question file's doc_scope "
                         "and gold units (both language versions for A and B). Without a value the --questions file "
                         "(all its questions, not only --ids/--limit) decides; a comma list of question files may be "
                         "given instead (not with --final). Questions with doc_scope 'corpus' still run, over the "
                         "restricted corpus. Combined with --docs: the intersection. Stamped in run_meta and records.")
    ap.add_argument("--final", action="store_true",
                    help="the test-split protocol: required for test*.jsonl question files; refuses variants and "
                         "overrides not declared in config.yaml `final` (subsets, --allow-mixed, --no-health, ...)")
    ap.add_argument("--test-protocol", default=None, metavar="MANIFEST",
                    help="the datasets-v2 one-shot test gate: a frozen manifest (work/datasets_v2/test_freeze/"
                         "MANIFEST.json) that a test*.jsonl run must match exactly (see test_protocol_problems); "
                         "exclusive with --final")
    ap.add_argument("--out", required=True, help="output dir; 'runs/<name>' = work/runs/<name>")
    ap.add_argument("--no-retry-errors", action="store_true", help="also skip keys whose last record errored")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="resume even if existing records were made under another config hash or index fingerprint")
    ap.add_argument("--max-consecutive-errors", type=int, default=5)
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up call before the first question")
    ap.add_argument("--no-health", action="store_true", help="do not query Ollama version/digests")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and, per corpus set, the index directory the run would use and whether it is "
                         "complete; then exit (no index build, no model call)")
    ap.add_argument("--approach-class", default=None, help=argparse.SUPPRESS)   # tests: module:Class
    ap.add_argument("--pool", default=None,
                    help="a frozen pool manifest (ragbench/pools.py; a path or a pool id under work/multidoc/pools/): its "
                         "documents mask every question's 'corpus' scope; indexes stay those of the full frozen set. "
                         "Required for multi-document question files (schema multidoc-q-v1 or meta.experiment "
                         "'multidoc'), which refuse --docs-from-questions and --docs")
    ap.add_argument("--experiment", default=None,
                    help="experiment id of an identity run (default: the questions' meta.experiment_id)")
    ap.add_argument("--replicate", type=int, default=0, help="replicate number of an identity run (default 0)")
    ap.add_argument("--variants-version", default=None, choices=["variants-v1", "variants-v2"],
                    help="semantics of the doc_card/section_expand options (default: v2 for multi-document question "
                         "files, v1 - the finished study's - otherwise)")
    ap.add_argument("--reader", default=None,
                    help="openrouter:<model id> - every chat call goes to that OpenRouter model instead of the local "
                         "one (embeddings stay local); dev only, budget-capped (llm_openrouter.py)")
    return ap


def main(argv=None, cfg=None, log=None):
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    log = log or (lambda s: print(s, flush=True))
    cfg = cfg or load_config()

    # approach -----------------------------------------------------------------
    if args.approach_class:
        mod, cls_name = args.approach_class.split(":")
        cls = getattr(importlib.import_module(mod), cls_name)
        name = cls.name
    else:
        if not args.approach or args.approach.lower() not in ALIASES:
            raise SystemExit("--approach must be one of %s" % ", ".join(k.lower() for k in APPROACHES))
        name = ALIASES[args.approach.lower()]
        cls = None
    if args.index_level:
        args.ablation = list(args.ablation) + ["index_level=%s" % args.index_level]
    flags = parse_flags(args.ablation, name)

    # plan ---------------------------------------------------------------------
    qpath = resolve_path(args.questions, cfg)
    split = question_split(qpath)
    if args.test_protocol and split != "test":
        log("--test-protocol is only for test*.jsonl question files")
        return 2
    if split == "test" and not args.final and not args.test_protocol:
        log("refusing to run the TEST question file %s without --final or --test-protocol (all tuning happens on the dev set, "
            "ragbench/data/questions/pilot.jsonl)" % qpath.name)
        return 2
    all_q = load_questions(qpath)
    qs = [q for q in all_q if q.corpus == args.corpus]
    if args.ids:
        want = [x.strip() for x in args.ids.split(",") if x.strip()]
        qs = [q for q in qs if q.id in set(want)]
    invalid = [(q.id, q.validate()) for q in qs if q.validate()]
    if invalid:
        log("WARNING: skipping %d invalid questions: %s" % (len(invalid), invalid[:5]))
        bad = {i for i, _ in invalid}
        qs = [q for q in qs if q.id not in bad]
    # multi-document experiment: frozen pools only, identity-keyed records -------------------------------------
    multidoc = any(q.is_multidoc() for q in all_q)
    pool = None
    if multidoc:
        if args.docs_from_questions is not None:
            log("refusing --docs-from-questions for a multi-document question file: it derives the search pool from "
                "the gold units; pass a frozen, independently chosen pool with --pool (ragbench/pools.py)")
            return 2
        if args.docs:
            log("refusing --docs for a multi-document question file: it rebuilds indexes over a subset; the pool "
                "(--pool) is the only scope and a mask over the full frozen set")
            return 2
        if not args.pool:
            log("a multi-document question file needs --pool <frozen pool manifest> (ragbench/pools.py)")
            return 2
    if args.pool:
        from .pools import PoolError, load_pool, resolve_pool_path
        try:
            pool = load_pool(resolve_pool_path(args.pool, cfg))
        except (PoolError, OSError, ValueError) as e:
            log("pool: %s" % e)
            return 2
        if pool.corpus != args.corpus:
            log("pool %s belongs to corpus %s, not %s" % (pool.pool_id, pool.corpus, args.corpus))
            return 2
        if multidoc:
            other = [q.id for q in qs if pool.pool_id not in q.pool_ids()]
            if other:
                log("--pool %s: skipping %d question(s) not declared for this pool: %s" % (pool.pool_id, len(other),
                                                                                          other[:10]))
            qs = [q for q in qs if pool.pool_id in q.pool_ids()]
    identity_mode = multidoc
    experiment_id = args.experiment
    if identity_mode and not experiment_id:
        ids_ = sorted({str((q.meta or {}).get("experiment_id")) for q in all_q if (q.meta or {}).get("experiment_id")})
        if len(ids_) != 1:
            log("an identity run needs one experiment id: pass --experiment (questions declare %s)" % (ids_ or "none"))
            return 2
        experiment_id = ids_[0]
    variants_version = args.variants_version or ("variants-v2" if multidoc else "variants-v1")
    subset_keys = parse_docs(args.docs)
    if subset_keys:
        outside = [q.id for q in qs if not question_in_docs(q, subset_keys)]
        if outside:
            log("--docs %s: skipping %d question(s) whose doc scope is 'corpus' or outside the subset: %s" % (
                ",".join(sorted(subset_keys)), len(outside), outside[:10]))
        qs = [q for q in qs if question_in_docs(q, subset_keys)]
    dfq_files, dfq_keys = [], None
    if args.docs_from_questions is not None:
        if args.docs_from_questions == "__questions__":
            dfq_files, src_q = [qpath], all_q
        else:
            dfq_files = [resolve_path(x.strip(), cfg) for x in args.docs_from_questions.split(",") if x.strip()]
            src_q = [q for f in dfq_files for q in load_questions(f)]
        dfq_keys = referenced_doc_keys(src_q, args.corpus)
        log("--docs-from-questions: corpus %s restricted to %d document key(s) referenced by %s: %s" % (
            args.corpus, len(dfq_keys), [f.name for f in dfq_files], sorted(dfq_keys)[:20]))
    if subset_keys is not None and dfq_keys is not None:
        doc_keys = subset_keys & dfq_keys
    else:
        doc_keys = subset_keys if subset_keys is not None else dfq_keys
    docs_restriction = None if doc_keys is None else {
        "mode": "+".join(m for m, on in (("docs", subset_keys is not None), ("docs_from_questions", dfq_keys is not None)) if on),
        "keys": sorted(doc_keys), "docs_arg": sorted(subset_keys) if subset_keys is not None else None,
        "question_files": [f.name for f in dfq_files] or None}
    _sets = {}

    def get_set(set_name):
        if set_name not in _sets:
            full = load_set(set_name, cfg)
            # an empty restriction is an empty corpus, never the full one
            _sets[set_name] = full if doc_keys is None else full.subset([d for d in full.doc_ids() if doc_key(d) in doc_keys])
            if doc_keys:
                missing = sorted(doc_keys - {doc_key(d) for d in _sets[set_name].doc_ids()})
                if missing:
                    log("WARNING: restricted document keys not found in %s: %s" % (set_name, missing))
            if docs_restriction is not None:
                docs_restriction.setdefault("doc_ids", {})[set_name] = _sets[set_name].doc_ids()
        return _sets[set_name]
    if args.limit is not None:
        qs = qs[: args.limit]
    settings = [Setting.parse(s) for s in args.settings.split(",") if s.strip()]
    plan = [(q, s) for q in qs for s in settings if q.supports(s)]
    n_unsupported = len(qs) * len(settings) - len(plan)

    out = resolve_path(args.out, cfg)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(path=None)
    if args.reader:
        if args.final:
            log("--reader is a dev-only model sweep; refusing it with --final")
            return 2
        if not args.reader.startswith("openrouter:"):
            log("--reader must be openrouter:<model id>")
            return 2
        from .llm_openrouter import OpenRouterReader, model_info
        rid = args.reader.split(":", 1)[1]
        info = model_info(rid, cfg.project_root / "work" / "openrouter" / "catalog.json")
        cfg.models["chat"] = "openrouter/" + rid           # enters the record config hash and every call-cache key
        cfg.models["chat_digest"] = "openrouter"
        # the answer reader's HTTP attempt policy: env READER_MAX_ATTEMPTS (frozen for the test; every ledger record
        # carries max_attempts); unset = the old default (6 retries = 7 attempts)
        mx = os.environ.get("READER_MAX_ATTEMPTS")
        client = OpenRouterReader(cfg, ledger, rid, supported=info.get("params"), attempts=int(mx) if mx else None)
        log("reader %s: max HTTP attempts %d (READER_MAX_ATTEMPTS=%s)" % (rid, client.or_retries + 1, mx or "unset"))
    else:
        client = OllamaClient(cfg, ledger)
    approach = (cls or approach_class(name))(cfg=cfg, client=client, flags=flags)
    approach.pool = pool
    approach.variants_version = variants_version
    variant = approach.variant
    tag = safe_name(variant)
    ledger.path = str(out / ("ledger_%s.jsonl" % tag))
    runlog = RunLog(out / ("runs_%s.jsonl" % tag))
    code_sha1 = approach_code_sha1(approach)
    dep_sha1 = dependency_sha1()
    idents = {}
    if identity_mode:
        # resume key = the record identity (experiment, question, pool, method, options, reader, replicate, setting);
        # a record counts as done only under the same code signature and question signature (else: refused below)
        code_sig = sha1_json({"approach": code_sha1, "deps": dep_sha1})[:16]
        for q, s in plan:
            idents[(q.id, s.name)] = dict(record_identity(experiment_id, q.id, pool.pool_id, variant, flags,
                                                          variants_version, cfg, args.replicate, s.name),
                                          code_sig=code_sig, question_sig=question_sig_v2(q, approach.needs_gold),
                                          pool_sha1=pool.sha1)
        rid_done = {((r.get("meta") or {}).get("identity") or {}).get("rid") for r in runlog.records()
                    if args.no_retry_errors or not r.get("error")} - {None}
        todo = [(q, s) for q, s in plan if idents[(q.id, s.name)]["rid"] not in rid_done]
    else:
        done = runlog.completed(include_errors=args.no_retry_errors)
        todo = [(q, s) for q, s in plan if (q.id, variant, s.name) not in done]
    sets = sorted({set_name_for(q.corpus, s.d_lang) for q, s in plan})
    log("%s on corpus %s: %d questions x %d settings -> %d planned (%d unsupported), %d already done, %d to run; sets %s; out %s" % (
        variant, args.corpus, len(qs), len(settings), len(plan), n_unsupported, len(plan) - len(todo), len(todo), sets, out))
    log("split %s (%s)%s" % (split, qpath.name, "; FINAL protocol" if args.final else ""))
    if args.final:
        probs = final_problems(args, cfg, variant, settings)
        if probs:
            for p_ in probs:
                log("FINAL: " + p_)
            log("refusing the --final run (declare deliberate changes in config.yaml `final` first)")
            return 2
    if args.test_protocol:
        probs = test_protocol_problems(args, args.test_protocol, qpath, variant)
        if probs:
            for p_ in probs:
                log("TEST PROTOCOL: " + p_)
            log("refusing the test run (it does not match the frozen manifest %s)" % args.test_protocol)
            return 2
        log("TEST PROTOCOL: matches %s" % args.test_protocol)
    if pool is not None:
        # the pool froze the FULL sets it masks (index statistics) and each document's content: both must still hold
        probs = pool.verify({n: get_set(n) for n in sorted(set(pool.manifest.get("frozen_sets") or {}) | set(sets))})
        not_frozen = [n for n in sets if n not in (pool.manifest.get("frozen_sets") or {})]
        if not_frozen:
            probs.append("sets %s are not frozen by pool %s" % (not_frozen, pool.pool_id))
        if probs:
            for p_ in probs:
                log("POOL: " + p_)
            log("refusing to run with pool %s (its frozen documents or sets changed)" % pool.pool_id)
            return 2
        log("pool %s (sha1 %s): %s; a scope mask over the full sets" % (
            pool.pool_id, pool.sha1[:12], {n: len(pool.doc_ids(n)) for n in sets}))
    if args.dry_run:
        for q, s in todo[:20]:
            log("  %s %s" % (q.id, s.name))
        dry = {"variant": variant, "corpus": args.corpus, "questions_file": qpath.name, "split": split,
               "final": bool(args.final), "index_only": bool(args.index_only), "out": str(out),
               "settings": [s.name for s in settings], "n_questions": len(qs), "planned": len(plan),
               "done": len(plan) - len(todo), "todo": len(todo), "by_setting": {}, "by_format": {}, "sets": []}
        if identity_mode or pool is not None or variants_version != "variants-v1":   # older dry runs keep their shape
            dry.update(identity_mode=identity_mode, experiment_id=experiment_id, replicate=args.replicate,
                       variants_version=variants_version, pool=pool.identity() if pool is not None else None,
                       rids=sorted(i["rid"] for i in idents.values()))
        for q, s in plan:
            dry["by_setting"].setdefault(s.name, {"planned": 0, "todo": 0})["planned"] += 1
            dry["by_format"].setdefault(answer_format(q), {"planned": 0, "todo": 0})["planned"] += 1
        for q, s in todo:
            dry["by_setting"][s.name]["todo"] += 1
            dry["by_format"][answer_format(q)]["todo"] += 1
        for set_name in sets:          # the index directories this run would build or load (no build here)
            cs = get_set(set_name)
            d = approach.index_dir(cs)
            st = read_json(d / "build_stats.json") or {}
            info = dict(cs.stats(), set=set_name, index_dir=str(d), exists=d.is_dir(), complete=bool(st.get("complete")),
                        doc_ids=cs.doc_ids())
            dry["sets"].append(info)
            log("  index %s %s: %s complete=%s (%d docs, %d sections, %d chars)" % (
                variant, set_name, d, info["complete"], info["docs"], info["sections"], info["chars"]))
        log("DRYRUN-JSON " + json.dumps(dry, ensure_ascii=False, sort_keys=True))
        return 0

    # resume guard: never mix records produced under another config, corpus/index fingerprint or question text
    cur_hash = cfg.hash(*RECORD_HASH_SECTIONS)
    old = [r for r in runlog.records() if r.get("approach") == variant]
    old_hashes = {r.get("config_hash") for r in old if not r.get("error")} - {cur_hash}
    mixed = []
    if old_hashes:
        mixed.append("existing records use config hash %s, current is %s" % (sorted(old_hashes), cur_hash))
    meta_path = out / ("run_meta_%s.json" % tag)
    prev_meta = read_json(meta_path) or {}
    sig_now = {q.id: question_sig(q, approach.needs_gold) for q in qs}
    old_sigs = prev_meta.get("question_sigs") or {}
    answered = {r.get("question_id") for r in old if not r.get("error")}
    changed = sorted(i for i in answered if i in sig_now and i in old_sigs and old_sigs[i] != sig_now[i])
    if changed:
        mixed.append("%d question(s) changed (text, corpus or doc scope%s) since their records were answered: %s" % (
            len(changed), ", gold units" if approach.needs_gold else "", changed[:10]))
    # the v2 signature adds the answer format, its source fields and the pool ids (compared only where an earlier
    # invocation stored it, so directories written before 2026-09-24 resume exactly as before)
    sig2_now = {q.id: question_sig_v2(q, approach.needs_gold) for q in qs}
    old_sigs2 = prev_meta.get("question_sigs_v2") or {}
    changed2 = sorted(i for i in answered if i in sig2_now and i in old_sigs2 and old_sigs2[i] != sig2_now[i]
                      and i not in changed)
    if changed2:
        mixed.append("%d question(s) changed their answer format, task family or pool since their records were "
                     "answered: %s" % (len(changed2), changed2[:10]))
    prev_vv = {i.get("variants_version") or "variants-v1" for i in prev_meta.get("invocations") or [] if i.get("n_ok")}
    if old and prev_vv - {variants_version}:
        mixed.append("earlier records of %s were answered under %s, this run uses %s (the doc_card / section_expand "
                     "semantics differ)" % (variant, sorted(prev_vv), variants_version))
    if identity_mode:
        stale, by_rid = [], {v["rid"]: v for v in idents.values()}
        for r in old:
            ri = (r.get("meta") or {}).get("identity") or {}
            cur = by_rid.get(ri.get("rid"))
            if cur is None or r.get("error"):
                continue
            diff = [k for k in ("code_sig", "question_sig", "pool_sha1") if ri.get(k) != cur[k]]
            if diff:
                stale.append("%s %s (%s)" % (r.get("question_id"), r.get("setting"), ",".join(diff)))
        if stale:
            log("REFUSED: %d record(s) in %s were produced under another code, question or pool signature (a "
                "revised method, contract, variant or question): %s. Identity runs never resume into stale records; "
                "use a new --out." % (len(stale), out, stale[:10]))
            return 2
    prev_code = {i.get("approach_code_sha1") for i in prev_meta.get("invocations") or [] if i.get("n_ok")} - {None, code_sha1}
    if prev_code and old and todo:
        log("WARNING: the %s source code changed since earlier records in %s were produced; new records will "
            "mix code versions (run_meta records the sha1 per invocation; consider a new --out)" % (variant, out))
    health = None if args.no_health else client.health()
    ollama_version = (health or {}).get("version") if (health or {}).get("reachable") else None
    if todo:
        # Ollama's auto-updater can swap the runtime mid-experiment; results may change with it. The guard covers
        # the whole output directory (every variant), so approaches compared in one directory share one runtime.
        dir_versions = directory_ollama_versions(out)
        if ollama_version:
            other = {v: w for v, w in dir_versions.items() if v != ollama_version}
            if other:
                mixed.append("Ollama version is now %s but records in %s were answered under %s" % (
                    ollama_version, out, {v: w[:4] for v, w in sorted(other.items())}))
        elif dir_versions and health is not None:
            log("WARNING: Ollama is unreachable, so its version cannot be compared with %s" % sorted(dir_versions))
        elif dir_versions:
            log("WARNING: --no-health: the Ollama version is not checked against %s in %s" % (sorted(dir_versions), out))
    if old and todo:
        for set_name in sets:
            fp = approach.index_fingerprint(get_set(set_name))
            stale = {r.get("index_fingerprint") for r in old if r.get("corpus_set") == set_name and not r.get("error")} - {fp, ""}
            if stale:
                mixed.append("existing %s records were answered over index fingerprint(s) %s, current is %s "
                             "(corpus or index settings changed)" % (set_name, sorted(x[:8] for x in stale), fp[:8]))
    if mixed:
        for m in mixed:
            log("WARNING: " + m)
        if not args.allow_mixed:
            log("refusing to resume into %s; use a new --out, or --allow-mixed to continue anyway" % out)
            return 2

    meta = prev_meta or {"approach": variant, "approach_base": name, "flags": flags, "invocations": []}
    meta.setdefault("question_sigs", {}).update(sig_now)
    meta.setdefault("question_sigs_v2", {}).update(sig2_now)
    inv = {"run_id": ledger.run_id, "started": now_iso(), "argv": list(argv if argv is not None else sys.argv[1:]),
           "questions_file": str(qpath), "questions_sha1": sha1_text(qpath.read_text(encoding="utf-8")),
           "corpus": args.corpus, "settings": [s.name for s in settings], "n_planned": len(plan),
           "n_todo": len(todo), "models": cfg.models, "approach_code_sha1": code_sha1,
           "docs": sorted(doc_keys) if doc_keys is not None else None,
           "docs_restriction": docs_restriction, "split": split, "final": bool(args.final),
           "ollama_version": ollama_version,
           "config_hash": cur_hash, "config_hash_sections": list(RECORD_HASH_SECTIONS),
           "answer_prompts": dict(ANSWER_PROMPT_VERSIONS),
           "answer_formats": {f: sum(1 for q, s in plan if answer_format(q) == f)
                              for f in sorted({answer_format(q) for q, s in plan})},
           "config_source": cfg.source, "evidence_budget": cfg.evidence["max_chars"],
           "dependency_sha1": dep_sha1, "variants_version": variants_version,
           "answer_prompts_all": {f: answer_prompt_version(f) for f in sorted({answer_format(q) for q, s in plan})},
           "pool": pool.identity() if pool is not None else None,
           "identity": {"experiment_id": experiment_id, "replicate": args.replicate} if identity_mode else None}
    if args.test_protocol:
        inv["test_protocol"] = {"manifest": str(args.test_protocol),
                                "sha1": sha1_text(Path(args.test_protocol).read_text(encoding="utf-8"))}
    rec_meta = {"ollama_version": ollama_version, "questions_file": qpath.name, "final": bool(args.final),
                "docs_restriction": None if docs_restriction is None else
                {k: docs_restriction[k] for k in ("mode", "keys", "question_files")}}
    if pool is not None:
        rec_meta["pool"] = pool.identity()
    if args.test_protocol:
        rec_meta["test_protocol_sha1"] = inv["test_protocol"]["sha1"]
    trace_stats = {"n": 0, "sums": {}}
    if health is not None:
        h = health
        inv["ollama"] = {k: h.get(k) for k in ("reachable", "version", "chat_present", "embed_present",
                                               "chat_digest_match", "embed_digest_match", "error")}
        if h.get("reachable") and not (h.get("chat_digest_match") and h.get("embed_digest_match")):
            log("WARNING: installed model digests differ from config: %s" % inv["ollama"])
    meta["invocations"].append(inv)
    atomic_write_json(meta_path, meta, indent=1)

    status = "ok"
    n_ok = n_err = 0
    try:
        # index ------------------------------------------------------------------
        for set_name in (sets if (todo or args.index_only) else []):
            t0 = time.time()
            cs = get_set(set_name)
            log("index %s over %s ..." % (variant, cs))
            try:
                h = approach.build_index(cs, ledger)
            finally:
                d = approach.index_dir(cs)
                st = read_json(d / "build_stats.json") or {}
                st.update(index_dir=str(d), corpus_stats=cs.stats(), copied_at=now_iso(),
                          cached_cost=cached_index_cost(d), docs=cs.doc_ids() if doc_keys is not None else "all",
                          docs_restriction=None if docs_restriction is None else docs_restriction["mode"],
                          index_cost=_index_cost(d))
                atomic_write_json(out / ("index_%s_%s.json" % (tag, set_name)), st, indent=1)
            tot = h.stats.get("totals") or {}
            log("index %s ready in %.1fs (build totals: %s; size %d bytes; complete=%s)" % (
                set_name, time.time() - t0, tot, h.stats.get("size_bytes") or 0, h.stats.get("complete")))
        if args.index_only:
            todo = []

        # warm-up (untimed, outside every question scope) ----------------------
        if todo and not args.no_warmup:
            try:
                client.chat([{"role": "user", "content": "Reply with OK."}], purpose="warmup", phase="smoke",
                            approach=variant, num_predict=2)
                if name in ("B3", "AGENTIC2", "AGENTIC3", "AGENTIC4", "AGENTIC5", "HYBRID", "LEX3", "LEX5"):   # bge-m3 at query time
                    client.embed(["warmup"], purpose="warmup", phase="smoke", approach=variant)
            except Exception as e:  # noqa: BLE001
                log("WARNING: warm-up failed: %s" % str(e)[:200])

        # questions ---------------------------------------------------------------
        consecutive = 0
        for i, (q, s) in enumerate(todo, 1):
            approach._answer_call = None
            try:
                rec = approach.answer(q, s)
            except Exception as e:  # noqa: BLE001  (failures outside _answer, e.g. index loading)
                rec = error_record(approach, q, s, "%s: %s" % (type(e).__name__, str(e)[:500]), cfg, ledger.run_id)
            rec.split = split
            rec.meta = dict(rec.meta or {}, **rec_meta)
            if identity_mode:
                rec.meta["identity"] = dict(idents[(q.id, s.name)])
            save_answer_pack(approach, rec, os.environ.get("RAGBENCH_PACK_DIR"))   # after the timed scope
            ts = (rec.trace or {}).get("stats") if isinstance(rec.trace, dict) else None
            if isinstance(ts, dict) and not rec.error:
                trace_stats["n"] += 1
                for k, v in ts.items():
                    if isinstance(v, (int, float)):
                        trace_stats["sums"][k] = trace_stats["sums"].get(k, 0) + float(v)
            runlog.append(rec)
            if rec.error:
                n_err += 1
                consecutive += 1
            else:
                n_ok += 1
                consecutive = 0
            log("[%d/%d] %s %s calls=%d ptok=%d ctok=%d %.1fs ev=%d units/%d chars%s | %s" % (
                i, len(todo), q.id, s.name, rec.llm_calls, rec.prompt_tokens, rec.completion_tokens,
                rec.latency_ms / 1000.0, len(rec.evidence_units), rec.evidence_chars,
                (" ERROR " + rec.error[:120]) if rec.error else "", _short(rec.answer_text)))
            if consecutive >= args.max_consecutive_errors:
                status = "stopped: %d consecutive errors" % consecutive
                log("STOP: " + status)
                break
    except KeyboardInterrupt:
        status = "interrupted"
        log("interrupted; completed records are saved, rerun the same command to resume")
    except Exception as e:  # noqa: BLE001
        status = "failed: %s: %s" % (type(e).__name__, str(e)[:300])
        log("FAILED: " + status)
        traceback.print_exc()
    finally:
        inv.update(finished=now_iso(), status=status, n_ok=n_ok, n_errors=n_err,
                   ledger_totals={ph: ledger.totals(phase=ph) for ph in ("index", "query", "smoke")},
                   ledger_file=ledger.path,
                   # approach-reported per-question counters (trace.stats), e.g. OURS plan_repair / widen, B4 narrowed_items
                   trace_stats=dict(trace_stats, rates={k: round(v / trace_stats["n"], 4)
                                                        for k, v in trace_stats["sums"].items()} if trace_stats["n"] else {}))
        atomic_write_json(meta_path, meta, indent=1)
    if identity_mode:
        ok_rids = {((r.get("meta") or {}).get("identity") or {}).get("rid") for r in runlog.records() if not r.get("error")}
        remaining = len([1 for q, s in plan if idents[(q.id, s.name)]["rid"] not in ok_rids])
    else:
        remaining = len([1 for q, s in plan if (q.id, variant, s.name) not in runlog.completed()])
    log("done: %d ok, %d errors this run; %d of %d planned still missing; status=%s" % (n_ok, n_err, remaining, len(plan), status))
    log("ledger totals: index=%s query=%s" % (inv["ledger_totals"]["index"], inv["ledger_totals"]["query"]))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
