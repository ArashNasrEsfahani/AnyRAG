"""Experiment 2: long context. Whole documents in the reader prompt instead of retrieved passages.

Arms (one prompt-token limit L_TOK for both, set by cost; see RESULTS.md):
  LC-GOLD-DOCS  ORACLE upper reference: the documents that contain the question's gold evidence units. When their
                full text exceeds the limit, each document is cut to a contiguous window (or, if its gold units lie
                too far apart, one window per unit) that contains its gold units, so the gold evidence is always
                shown. Answerable questions only (unanswerable ones have no gold document).
  LC-BM25-DOCS  The pool's documents ranked by a document-level BM25 score of the question (ragbench B2 tokenizer
                and bm25s parameters, one BM25 document = title + full text, IDF over the pool), added whole while
                they fit; the first document that does not fit is cut to the remaining space (head), then stop.
Prompt: the test's answer contract unchanged (system prompt by answer format, "Evidence:\\n...\\n\\nQuestion: ...",
num_predict by format); evidence rendered "[k] <document title>\\n<text>". Reader GLM-5.3-flash, pinned as the test.

    python lc.py build --split dev|test                       # packs + sizes + cost estimate, no calls
    python lc.py run --split dev|test --arm LC-GOLD-DOCS [--ids a,b] [--limit n]
    python lc.py judge --split dev|test --arm LC-GOLD-DOCS
"""
import argparse
import json
import sys
import time
from pathlib import Path

import common as C

L_TOK = 32000                  # prompt-token limit per call (cost-bound: ~$0.0048 per call at $0.15/M)
OVERHEAD_TOK = 700             # system prompt + question + markers, upper estimate
PRICE_IN, PRICE_OUT = 0.15e-6, 0.50e-6     # together endpoint, live listing (glm_endpoints_live.json)
ARMS = ("LC-GOLD-DOCS", "LC-BM25-DOCS")
POOL_ARM = "LC-POOL"          # whole pool, only for pools that fit the 1,048,575-token context (LAW-fa, LAW-en)
STATS = json.loads((C.S4 / "pool_stats.json").read_text(encoding="utf-8"))
TPC = STATS["tokens_per_char_glm_answer_median"]
EXPERIMENT = "s4-posthoc-lc"
READER_CHAT = "openrouter/z-ai/glm-5.3-flash"


def budget_chars(lang):
    return int((L_TOK - OVERHEAD_TOK) / TPC[lang])


# ------------------------------------------------------------------ questions
def questions(split):
    from ragbench.contracts import v2_runner_format
    out = []
    if split == "test":
        for qid, d in C.test_questions().items():
            v2 = d["meta"]["v2"]
            lang = v2["language"]
            out.append({"id": qid, "pool": d["pool_id"], "lang": lang, "setting": d["meta"]["setting"],
                        "text": d["question_fa"] if lang == "fa" else d["question_en"], "fmt": d["answer_format"],
                        "units": [u for v in d["gold_units"].values() for u in v],
                        "gold_docs": sorted(v2["doc_scope"]["oracle_docs"]), "answerable": v2["answerable"],
                        "tier": v2.get("tier") or v2.get("intended_tier")})
    else:
        for f in ("rpt", "book", "law"):
            for d in C.jl(C.QDIR / ("%s_dev.jsonl" % f)):
                lang = d["language"]
                out.append({"id": d["id"], "pool": d["pool_id"], "lang": lang, "setting": "%s-%s" % (lang, lang),
                            "text": d["question"], "fmt": v2_runner_format(d["answer_format"], d.get("intended_tier")),
                            "units": [e["unit_id"] for e in d.get("evidence") or []],
                            "gold_docs": sorted(d["doc_scope"].get("oracle_docs") or []),
                            "answerable": d["answerable"], "tier": d.get("tier") or d.get("intended_tier")})
    return out


def system_prompt(q, cfg):
    from ragbench import contracts as K
    abstain = cfg.evidence["abstain_token"]
    sysm = K.ANSWER_SYSTEMS[q["fmt"]].format(source_rule=K.SOURCE_RULE_EVIDENCE,
                                             abstain_rule=K.ABSTAIN_RULE_EVIDENCE.format(abstain=abstain),
                                             answer_lang=K.LANG_NAMES[q["lang"]], abstain=abstain)
    return sysm, K.answer_num_predict(q["fmt"], cfg)


# ------------------------------------------------------------------ evidence
def _snap(text, i, direction, slack=400):
    """Move a cut point to a nearby line break (readability); direction -1 = search backwards."""
    if i <= 0 or i >= len(text):
        return max(0, min(i, len(text)))
    if direction < 0:
        j = text.rfind("\n", max(0, i - slack), i)
    else:
        j = text.find("\n", i, min(len(text), i + slack))
    return j + 1 if (j >= 0 and direction < 0) else (j if j >= 0 else i)


def gold_ranges(doc, units, allot):
    """[(start, end)] of doc.text to show (allot chars at most) that contain every gold unit of this doc."""
    n = len(doc.text)
    if n <= allot:
        return [(0, n)]
    spans = sorted(doc.unit(u).span for u in units)
    lo, hi = spans[0][0], spans[-1][1]
    if hi - lo <= allot:
        pad = (allot - (hi - lo)) // 2
        s = max(0, lo - pad)
        e = min(n, s + allot)
        s = max(0, e - allot)
        s, e = _snap(doc.text, s, +1), _snap(doc.text, e, -1)
        if s > lo:
            s = lo
        if e < hi:
            e = hi
        return [(s, e)]
    per = max(1, allot // len(spans))
    out = []
    for a, b in spans:
        pad = max(0, (per - (b - a)) // 2)
        s, e = max(0, a - pad), min(n, b + pad)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def render(parts):
    """parts: [(doc, [(s, e)])] -> evidence text. Gaps and cut ends are marked [...]."""
    blocks = []
    for k, (doc, ranges) in enumerate(parts, 1):
        segs = []
        for s, e in ranges:
            seg = doc.text[s:e].strip("\n")
            segs.append(("[...]\n" if s > 0 else "") + seg + ("\n[...]" if e < len(doc.text) else ""))
        body = "\n".join(segs).replace("[...]\n[...]", "[...]")
        blocks.append("[%d] %s\n%s" % (k, doc.title, body))
    return "\n\n".join(blocks)


def retained(q, parts):
    """Gold units whose span lies wholly inside a shown range."""
    shown = {d.id: r for d, r in parts}
    keep = 0
    for u in q["units"]:
        did = u.split("::")[0]
        if did in shown:
            a, b = None, None
            for d, r in parts:
                if d.id == did:
                    a, b = d.unit(u).span
            keep += any(s <= a and b <= e for s, e in shown[did])
    return keep


_BM25 = {}


def bm25_rank(pool, docs, text):
    import bm25s
    from ragbench.approaches import baselines as BL
    from ragbench.retrieval.hybrid import query_tokens
    if pool not in _BM25:
        toks = [BL.tokenize(d.title + "\n" + d.text, BL.DEFAULTS["tokenizer"]) for d in docs]
        p = BL.DEFAULTS["bm25"]
        r = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
        r.index(toks, show_progress=False)
        _BM25[pool] = r
    r = _BM25[pool]
    qt = [t for t in query_tokens(text) if t in r.vocab_dict]
    sc = r.get_scores(qt) if qt else [0.0] * len(docs)
    order = sorted(range(len(docs)), key=lambda i: (-float(sc[i]), docs[i].id))
    return [(docs[i], float(sc[i])) for i in order]


def build_parts(arm, q, docs):
    by = {d.id: d for d in docs}
    B = budget_chars(q["lang"])
    info = {"budget_chars": B}
    if arm == "LC-GOLD-DOCS":
        gd = [by[d] for d in q["gold_docs"]]
        # water-filling: equal shares, a short document returns its unused share to the others
        allot, left, rest = {}, B, sorted(gd, key=lambda d: len(d.text))
        for i, d in enumerate(rest):
            share = left // (len(rest) - i)
            allot[d.id] = min(len(d.text), share)
            left -= allot[d.id]
        parts = []
        for d in gd:
            us = [u for u in q["units"] if u.split("::")[0] == d.id]
            parts.append((d, gold_ranges(d, us, allot[d.id])))
        info["docs"] = [{"doc": d.id, "doc_chars": len(d.text), "shown": sum(e - s for s, e in r),
                         "ranges": len(r), "whole": r == [(0, len(d.text))]} for d, r in parts]
    elif arm == POOL_ARM:
        parts = [(d, [(0, len(d.text))]) for d in docs]          # pool manifest order, question-independent
        info["budget_chars"] = None
        info["docs"] = [{"doc": d.id, "doc_chars": len(d.text), "shown": len(d.text), "whole": True,
                         "gold_doc": d.id in q["gold_docs"]} for d, _ in parts]
    else:
        ranked = bm25_rank(q["pool"], docs, q["text"])
        parts, used = [], 0
        for d, s in ranked:
            room = B - used - len(d.title) - 12
            if len(d.text) <= room:
                parts.append((d, [(0, len(d.text))]))
                used += len(d.text) + len(d.title) + 12
                continue
            if room >= 2000:
                e = _snap(d.text, room, -1)
                parts.append((d, [(0, e)]))
            break
        info["ranking"] = [{"doc": d.id, "score": round(s, 3)} for d, s in ranked[:8]]
        info["docs"] = [{"doc": d.id, "doc_chars": len(d.text), "shown": sum(e - s for s, e in r),
                         "whole": r == [(0, len(d.text))], "gold_doc": d.id in q["gold_docs"]} for d, r in parts]
    info["gold_units"] = len(q["units"])
    info["gold_units_retained"] = retained(q, parts)
    info["gold_docs_shown"] = sum(1 for d, _ in parts if d.id in q["gold_docs"])
    return parts, info


def build(split, arms=ARMS, ids=None):
    from ragbench.config import load_config
    from ragbench.run import answer_messages_sha256
    cfg = load_config()
    qs = questions(split)
    out = C.S4 / "lc_packs" / split
    rows = []
    for pool in C.POOLS:
        docs = C.pool_docs(pool)
        for q in [x for x in qs if x["pool"] == pool and (ids is None or x["id"] in ids)]:
            sysm, npred = system_prompt(q, cfg)
            if split == "test":      # the frozen pack's system prompt / num_predict / question must be identical
                pk = json.loads((C.TR / "packs" / "HYBRID15500" / ("%s__%s.json" % (q["id"], q["setting"])))
                                .read_text(encoding="utf-8"))
                assert pk["messages"][0]["content"] == sysm, q["id"]
                assert pk["num_predict"] == npred, q["id"]
                assert pk["messages"][1]["content"].endswith("\n\nQuestion: " + q["text"].strip()), q["id"]
            for arm in arms:
                if arm == "LC-GOLD-DOCS" and not q["gold_docs"]:
                    continue
                t0 = time.perf_counter()
                parts, info = build_parts(arm, q, docs)
                ev = render(parts)
                msgs = [{"role": "system", "content": sysm},
                        {"role": "user", "content": "Evidence:\n%s\n\nQuestion: %s" % (ev, q["text"].strip())}]
                pchars = sum(len(m["content"]) for m in msgs)
                est = round(pchars * TPC[q["lang"]])
                rec = {"schema": "s4-lc-pack-v1", "question_id": q["id"], "arm": arm, "split": split,
                       "pool": q["pool"], "lang": q["lang"], "setting": q["setting"], "tier": q["tier"],
                       "answerable": q["answerable"], "fmt": q["fmt"], "messages": msgs, "num_predict": npred,
                       "messages_sha256": answer_messages_sha256(msgs), "evidence_chars": len(ev),
                       "prompt_chars": pchars, "est_prompt_tokens": est, "build_ms": round((time.perf_counter() - t0) * 1000, 1),
                       "info": info}
                d = out / arm
                d.mkdir(parents=True, exist_ok=True)
                (d / ("%s.json" % q["id"])).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                rows.append({k: rec[k] for k in ("question_id", "arm", "pool", "lang", "tier", "answerable",
                                                 "evidence_chars", "prompt_chars", "est_prompt_tokens")}
                            | {"gold_units": info["gold_units"], "retained": info["gold_units_retained"],
                               "docs": len(info["docs"]), "whole_docs": sum(x["whole"] for x in info["docs"]),
                               "gold_docs_shown": info["gold_docs_shown"]})
    if arms != ARMS:
        for r in rows:
            C.append(out / "index_pool.jsonl", r)
            print(r)
        return
    (out / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        tok = sum(r["est_prompt_tokens"] for r in rs)
        print("%s %s: %d packs, est prompt tokens %d (%.0f/q), est input USD %.3f, gold units retained %d/%d, "
              "all-whole-doc packs %d" % (split, arm, len(rs), tok, tok / max(1, len(rs)), tok * PRICE_IN,
                                          sum(r["retained"] for r in rs), sum(r["gold_units"] for r in rs),
                                          sum(r["whole_docs"] == r["docs"] for r in rs)))


# ------------------------------------------------------------------ run
def identity(q, arm):
    from ragbench.util import sha1_json
    ident = {"experiment_id": EXPERIMENT, "question_id": q["question_id"], "pool_id": q["pool"], "method": arm,
             "options": {"flags": {}, "variants_version": "variants-v2", "l_tok": L_TOK},
             "reader": {"chat": READER_CHAT, "chat_digest": "openrouter"}, "replicate": 0, "setting": q["setting"]}
    ident["rid"] = sha1_json(ident)[:20]
    return ident


def run(split, arm, ids=None, limit=None, reserve_usd=0.03):
    import client as CL
    from ragbench.llm import BudgetExceeded
    pdir = C.S4 / "lc_packs" / split / arm
    odir = C.S4 / "lc_runs" / split / arm
    odir.mkdir(parents=True, exist_ok=True)
    runs = odir / ("runs_%s.jsonl" % arm)
    done = {r["question_id"] for r in C.jl(runs) if not r.get("error")}
    errs = {}
    for r in C.jl(runs):
        if r.get("error"):
            errs[r["question_id"]] = errs.get(r["question_id"], 0) + 1
    order = [q for q in questions(split) if (pdir / ("%s.json" % q["id"])).exists()]
    if ids:
        order = [q for q in order if q["id"] in ids]
    cfg, ledger = CL.cfg_and_ledger()
    cl = CL.reader(cfg, ledger)
    if arm == POOL_ARM:
        cl.or_retries = 0
    n = 0
    for qq in order:
        if qq["id"] in done or errs.get(qq["id"], 0) >= 2:
            continue
        if limit is not None and n >= limit:
            break
        pk = json.loads((pdir / ("%s.json" % qq["id"])).read_text(encoding="utf-8"))
        est = pk["est_prompt_tokens"] * PRICE_IN + (pk["num_predict"] + 1500) * PRICE_OUT
        if C.spent() + est > CL.CAP_USD - reserve_usd:
            print("STOP: next call (est $%.4f) would pass the cap (spent $%.4f)" % (est, C.spent()))
            break
        err, res = None, None
        with ledger.scope(phase="query", approach=arm, question_id=pk["question_id"], setting=pk["setting"]) as sc:
            t0 = time.perf_counter()
            try:
                res = cl.chat(pk["messages"], purpose="answer", num_predict=pk["num_predict"])
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                err = "%s: %s" % (type(e).__name__, str(e)[:500])
            lat = (time.perf_counter() - t0) * 1000.0
        summ = sc.summary()
        ans = (res.text.strip() if res is not None else "") or ""
        ident = identity(pk, arm)
        rec = {"question_id": pk["question_id"], "approach": arm, "approach_base": arm, "flags": {},
               "setting": pk["setting"], "q_lang": pk["lang"], "d_lang": pk["lang"], "answer_text": ans,
               "abstained": ans == cfg.evidence["abstain_token"], "evidence_chars": pk["evidence_chars"],
               "llm_calls": summ["llm_calls"], "prompt_tokens": summ["prompt_tokens"],
               "completion_tokens": summ["completion_tokens"], "llm_wall_ms": summ["llm_wall_ms"],
               "latency_ms": round(lat + pk.get("build_ms", 0), 1), "reader_ms": round(lat, 1),
               "failed_calls": summ["failed_calls"], "error": err, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "models": {"chat": READER_CHAT}, "split": split,
               "trace": {"lc": {"arm": arm, "l_tok": L_TOK, "pack_sha256": pk["messages_sha256"], "info": pk["info"],
                                "provider": getattr(res, "provider", None) if res else None,
                                "done_reason": res.done_reason if res else None,
                                "reasoning_tokens": getattr(res, "reasoning_tokens", None) if res else None,
                                "cost_usd": res.cost_usd if res else None}},
               "meta": {"identity": ident, "answer_format": pk["fmt"], "answer_num_predict": pk["num_predict"],
                        "answer_messages_sha256": pk["messages_sha256"]}}
        C.append(runs, rec)
        n += 1
        print("%s %s %s ptok=%s ctok=%s %.1fs $%s %s" % (arm, pk["question_id"], "ERROR " + err[:60] if err else "ok",
                                                      rec["prompt_tokens"], rec["completion_tokens"], lat / 1000,
                                                      res.cost_usd if res else None, ans[:50].replace("\n", " ")))
        sys.stdout.flush()
    print("spent so far $%.4f" % C.spent())


def judge(split, arm):
    import client as CL
    from ragbench.llm import BudgetExceeded
    from ragbench.tools import judge_multi as JM
    from ragbench.util import KVCache
    cfg, ledger = CL.cfg_and_ledger()
    cl, j = CL.judge_client(cfg, ledger)
    qfile = (C.FZ / "questions" / "test_v2judge_all.jsonl") if split == "test" else \
        (C.WORK / "datasets_v2" / "devrun" / "questions" / "dev_v2judge_all.jsonl")
    recs, skipped = JM.latest_identity_records([C.S4 / "lc_runs" / split / arm])
    qs = JM.load_question_dicts(str(qfile))
    out = C.S4 / "lc_judge" / split
    out.mkdir(parents=True, exist_ok=True)
    try:
        JM.judge_records(recs, qs, cl, j["model"], out, retry_failures=False, methods=None,
                         cache=KVCache(out / "judge_multi_cache.sqlite"), ledger=ledger, max_model_calls=None,
                         provider_pin=j["provider_pin"], max_attempts=j["max_judge_attempts"],
                         on_malformed=j["on_malformed"])
    except BudgetExceeded as e:
        print("BUDGET STOP: %s" % e)
    print("spent so far $%.4f" % C.spent())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "run", "judge"))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--arm", choices=ARMS + (POOL_ARM,))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        if a.arm == POOL_ARM:
            build(a.split, (POOL_ARM,), set(a.ids.split(",")))
        else:
            build(a.split)
    elif a.cmd == "run":
        run(a.split, a.arm, set(a.ids.split(",")) if a.ids else None, a.limit)
    else:
        judge(a.split, a.arm)
